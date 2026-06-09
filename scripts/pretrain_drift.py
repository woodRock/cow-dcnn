"""
Cross-sample pretraining for RT-drift-robust GC-MS alignment.

Positive pairs are matched peaks from *different* GC-MS runs of the same dataset.
Within fish_oil and mtbls288 separately, each sample is aligned to several reference
chromatograms using raw cosine similarity + Hungarian matching.  A matched pair
(ref_spectrum, query_spectrum) captures genuine inter-run spectral variation: the
same compound measured on the same instrument type on different days, giving real
detector noise, minor matrix effects, and true RT drift between the two scans.

Why the previous approach failed
---------------------------------
The original DriftDataset applied a piecewise-linear RT warp to a chromatogram and
extracted the spectrum at the warped peak position.  Because backward resampling is
used (warped[t] = chroma[inv_fwd(t)]), extracting at position fwd(pk) gives back
exactly chroma[pk] — the two "views" in each positive pair were the same spectrum.
The encoder received zero gradient from the RT transformation.

Training pipeline
-----------------
  1. Load all .npz chromatograms from fish_oil + mtbls288.
  2. For each dataset, align every sample to N_REFS randomly chosen references.
  3. Keep matches with cosine similarity ≥ SIM_THRESHOLD within MAX_DRIFT_MIN.
  4. Apply light spectral augmentation to each view independently.
  5. Fine-tune with NT-Xent, initialised from checkpoints/mona_simclr.pt.
  6. Save to checkpoints/drift_simclr.pt.
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import SpectrumEncoder, nt_xent_loss

DATA_DIR = Path(__file__).parent.parent / 'data'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

N_BINS      = 200
RUN_MIN     = 45.0
BIN_MIN     = RUN_MIN / N_BINS        # 0.225 min per bin

# Alignment hyperparameters for pair mining
SIM_THRESHOLD = 0.65    # minimum cosine similarity to accept a match
MAX_DRIFT_MIN = 4.0     # maximum RT gap (minutes) between matched peaks
N_REFS        = 5       # number of reference chromatograms per dataset


# ── Utilities ─────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _detect_peaks(chroma: np.ndarray, height_pct: int = 80,
                  min_dist: int = 5, n_peaks: int = 15) -> np.ndarray:
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, height_pct)
    pks, props = sp_find_peaks(tic, height=thr, distance=min_dist)
    if len(pks) > n_peaks:
        top = np.argsort(props['peak_heights'])[-n_peaks:]
        pks = pks[top]
    return np.sort(pks)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light spectral augmentation — calibrated to real inter-run variability."""
    x = x.copy()

    # Gaussian noise (σ = 0.02; lighter than SimCLR stage-1 σ = 0.04 because
    # cross-sample pairs already contain genuine noise)
    if rng.random() > 0.2:
        x = np.clip(x + rng.normal(0, 0.02, size=x.shape).astype(np.float32), 0, None)

    # Random ion dropout — simulates ion suppression in complex matrices
    if rng.random() > 0.3:
        mask = rng.random(size=x.shape) < 0.05
        x[mask] = 0.0

    # Per-ion intensity jitter — simulates detector response variation
    if rng.random() > 0.3:
        x = x * rng.uniform(0.85, 1.15, size=x.shape).astype(np.float32)

    norm = np.linalg.norm(x)
    return (x / norm).astype(np.float32) if norm > 1e-8 else x.astype(np.float32)


# ── Pair mining ───────────────────────────────────────────────────────────────

def _align_to_reference(ref_chroma: np.ndarray, q_chroma: np.ndarray,
                         sim_threshold: float, max_drift_min: float,
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return matched (ref_spec, query_spec) L2-normed pairs for one sample pair.
    Uses raw cosine similarity + windowed Hungarian matching.
    """
    ref_pks = _detect_peaks(ref_chroma)
    q_pks   = _detect_peaks(q_chroma)
    if len(ref_pks) == 0 or len(q_pks) == 0:
        return []

    ref_fps = np.array([_l2(ref_chroma[pk]) for pk in ref_pks])   # (Nr, 1000)
    q_fps   = np.array([_l2(q_chroma[pk])   for pk in q_pks])     # (Nq, 1000)

    # Cosine similarity — spectra are L2-normed, so dot product = cosine
    sim = q_fps @ ref_fps.T                                         # (Nq, Nr)

    # Time window mask
    ref_t = ref_pks * BIN_MIN
    q_t   = q_pks   * BIN_MIN
    drift_mask = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sim_c = sim.copy()
    sim_c[drift_mask] = -1.0

    row_ind, col_ind = linear_sum_assignment(-sim_c)
    ms   = sim[row_ind, col_ind]
    keep = (ms >= sim_threshold) & ~drift_mask[row_ind, col_ind]

    pairs = []
    for qi, ri in zip(row_ind[keep], col_ind[keep]):
        spec_q = _l2(q_chroma[q_pks[qi]])
        spec_r = _l2(ref_chroma[ref_pks[ri]])
        pairs.append((spec_r, spec_q))
    return pairs


def build_cross_sample_pairs(chroma_dir: Path, n_refs: int = N_REFS,
                              sim_threshold: float = SIM_THRESHOLD,
                              max_drift_min: float = MAX_DRIFT_MIN,
                              seed: int = 42,
                              ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Mine positive pairs within one dataset directory.

    n_refs reference chromatograms are chosen randomly; every other sample is
    aligned to each reference.  Total pairs ≈ n_refs × N_samples × avg_matches.
    """
    files = sorted(chroma_dir.glob('*.npz'))
    if len(files) < 2:
        print(f"  {chroma_dir}: fewer than 2 samples, skipping.")
        return []

    chromas = [np.load(p)['chroma'].astype(np.float32) for p in files]
    rng     = np.random.default_rng(seed)
    ref_idx = rng.choice(len(chromas), size=min(n_refs, len(chromas)), replace=False)

    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for ri in ref_idx:
        ref = chromas[ri]
        for qi, q in enumerate(chromas):
            if qi == ri:
                continue
            pairs = _align_to_reference(ref, q, sim_threshold, max_drift_min)
            all_pairs.extend(pairs)

    # Deduplicate: if the same (spec_r, spec_q) pair appears from two different
    # reference alignments, keep only one copy.
    seen: set[bytes] = set()
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in all_pairs:
        key = a.tobytes() + b.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((a, b))

    return unique


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealPairDataset(Dataset):
    """
    Wraps a list of (spec_a, spec_b) matched-peak pairs.
    Applies independent spectral augmentation to each view at sample time
    so the model sees different noise realisations each epoch.
    """
    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], seed: int = 0):
        self.pairs = pairs
        self.rng   = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sa, sb = self.pairs[idx]
        x1 = torch.from_numpy(augment(sa, self.rng))
        x2 = torch.from_numpy(augment(sb, self.rng))
        return x1, x2


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, epochs: int = 300, batch_size: int = 64,
          lr: float = 1e-4, seed: int = 42) -> None:

    print("Mining cross-sample positive pairs …")
    fish_pairs = build_cross_sample_pairs(DATA_DIR / 'fish_oil'  / 'chroma', seed=seed)
    m288_pairs = build_cross_sample_pairs(DATA_DIR / 'mtbls288'  / 'chroma', seed=seed+1)
    all_pairs  = fish_pairs + m288_pairs

    if not all_pairs:
        print("No pairs found — check that data/fish_oil/chroma and data/mtbls288/chroma exist.")
        return

    print(f"  fish_oil:  {len(fish_pairs)} pairs")
    print(f"  mtbls288:  {len(m288_pairs)} pairs")
    print(f"  total:     {len(all_pairs)} cross-sample positive pairs")

    ds = RealPairDataset(all_pairs, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)

    model = SpectrumEncoder().to(device)
    ckpt  = CKPT_DIR / 'mona_simclr.pt'
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Initialised from {ckpt.name}")
    else:
        print(f"  Warning: {ckpt} not found — training from scratch")

    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x1, x2 in dl:
            x1, x2 = x1.to(device), x2.to(device)
            loss = nt_xent_loss(model(x1), model(x2))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={total/len(dl):.4f}")

    out = CKPT_DIR / 'drift_simclr.pt'
    torch.save(model.state_dict(), out)
    print(f"  saved → {out}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    train(device)
    print("Done.")
