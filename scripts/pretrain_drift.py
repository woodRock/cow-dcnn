"""
Drift-robust pretraining for DilatedSpectrumEncoder using synthetic GC-MS data.

Avoids data leakage by training exclusively on MoNA reference spectra
(data/pretraining/spectra.h5, train split) — zero overlap with fish_oil or
mtbls288 which are used for all downstream evaluation.

Why synthetic chromatograms?
-----------------------------
Simply augmenting clean MoNA spectra trains the encoder to be noise-robust
but not background-robust: in a real GC-MS run, the spectrum extracted at
peak bin t contains a small contribution from co-eluting compounds whose
Gaussian tails reach t.  Synthetic chromatograms reproduce this exactly.

Training pipeline
------------------
  1. Load 8,598 MoNA EI spectra (train split, 1000-dim, L2-normed).
  2. Generate N_SYNTHETIC synthetic GC-MS sample pairs:
       a. Sample N_COMP_PER_SAMPLE compounds at random from MoNA.
       b. Place them at random RT positions with Gaussian peak shapes and
          exponential baseline noise → reference chromatogram.
       c. Apply per-compound RT drift (±MAX_DRIFT_BINS) → query chromatogram.
          Heights are scaled independently to simulate injection variability.
       d. Extract ±1-bin context-window spectra at peak positions in both runs.
          Each of the N_COMP_PER_SAMPLE peak positions yields one positive pair.
  3. Wrap pairs in SyntheticPairDataset, apply independent spectral augmentation
     to each view at sample time.
  4. Train DilatedSpectrumEncoder with NT-Xent loss (temperature=0.1).
  5. Save to checkpoints/drift_simclr.pt.
"""

from __future__ import annotations

import sys
import h5py
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import DilatedSpectrumEncoder, nt_xent_loss

DATA_DIR = Path(__file__).parent.parent / 'data'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'
MONA_H5  = Path('/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5')
CKPT_DIR.mkdir(exist_ok=True)

N_BINS = 200
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS

# Synthetic dataset parameters
N_ITERATIONS   = 10_000  # total gradient steps
LOG_EVERY      = 200     # print interval
N_COMP         = 30      # compounds placed per synthetic chromatogram
PEAK_SIGMA     = 2.5     # bins — ~0.5 min GC-MS peak width
MAX_DRIFT_BINS = 18      # ±18 bins = ±4.05 min maximum drift
NOISE_SCALE    = 0.005   # exponential baseline noise scale

# Pre-computed Gaussian peak shape weights (constant across all batches)
_OFFSETS = np.arange(-8, 9, dtype=np.int32)
_GAUSS_W = np.exp(-0.5 * (_OFFSETS / PEAK_SIGMA) ** 2).astype(np.float32)


# ── MoNA data loading ─────────────────────────────────────────────────────────

def load_mona(path: Path = MONA_H5, split: str = 'train') -> np.ndarray:
    """Return (N, 1000) float32 L2-normalised spectra from the requested split."""
    with h5py.File(path, 'r') as f:
        mask  = np.array(f['split']).astype(str) == split
        specs = f['spectra'][mask]
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (specs / norms).astype(np.float32)


# ── Synthetic chromatogram building ──────────────────────────────────────────

def _ctx_spec(chroma: np.ndarray, pk: int) -> np.ndarray:
    """±1-bin mean spectrum, L2-normalised."""
    lo = max(0, pk - 1)
    hi = min(len(chroma), pk + 2)
    v  = chroma[lo:hi].mean(axis=0)
    n  = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _random_positions(n: int, n_bins: int, min_gap: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Sample up to n RT bin positions with a minimum spacing of min_gap."""
    pool = np.arange(10, n_bins - 10)
    rng.shuffle(pool)
    positions: list[int] = []
    for c in pool.tolist():
        if not positions or min(abs(c - p) for p in positions) >= min_gap:
            positions.append(c)
        if len(positions) == n:
            break
    return np.array(sorted(positions))


def _build_chromatogram(compounds: np.ndarray, positions: np.ndarray,
                         heights: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Composite Gaussian peaks — fully vectorised via a single matmul.

    Builds W[bin, compound] = Σ_offset height * gaussian(offset) then
    computes chroma = W @ compounds.  No Python loops over compounds or
    offsets; replaces 510 iterator steps with one (N_BINS×n) @ (n×1000) call.
    """
    n = len(compounds)
    bins_all    = positions[:, None].astype(np.int32) + _OFFSETS[None, :]  # (n, 17)
    weights_all = heights[:, None] * _GAUSS_W[None, :]                      # (n, 17)

    bins_flat  = bins_all.ravel()
    w_flat     = weights_all.ravel()
    comp_idx   = np.repeat(np.arange(n, dtype=np.int32), 17)

    valid = (bins_flat >= 0) & (bins_flat < N_BINS)
    W = np.zeros((N_BINS, n), dtype=np.float32)
    W[bins_flat[valid], comp_idx[valid]] = w_flat[valid]    # unique (bin,comp) pairs

    chroma = W @ compounds   # (N_BINS, 1000)
    chroma += rng.exponential(NOISE_SCALE, size=chroma.shape).astype(np.float32)
    return chroma


# ── On-the-fly batch generation ───────────────────────────────────────────────

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light spectral augmentation applied independently to each SimCLR view."""
    x = x.copy()
    if rng.random() > 0.2:
        x = np.clip(x + rng.normal(0, 0.02, size=x.shape).astype(np.float32), 0, None)
    if rng.random() > 0.3:
        x[rng.random(size=x.shape) < 0.05] = 0.0
    if rng.random() > 0.3:
        x = x * rng.uniform(0.85, 1.15, size=x.shape).astype(np.float32)
    n = np.linalg.norm(x)
    return (x / n).astype(np.float32) if n > 1e-8 else x.astype(np.float32)


def _make_batch(mona_specs: np.ndarray, batch_size: int,
                rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    """
    Generate a fresh batch of batch_size positive pairs on-the-fly.

    Returns (x1, x2, rt1, rt2) where rt* are normalised RT positions in [0,1].
    Each synthetic sample contributes up to N_COMP pairs, so typically
    only 2-3 samples are needed per batch.
    """
    batch_a: list[np.ndarray] = []
    batch_b: list[np.ndarray] = []
    rts_a:   list[float] = []
    rts_b:   list[float] = []

    while len(batch_a) < batch_size:
        idx       = rng.choice(len(mona_specs), size=N_COMP, replace=False)
        compounds = mona_specs[idx]

        pks_ref = _random_positions(N_COMP, N_BINS, min_gap=6, rng=rng)
        if len(pks_ref) < 2:
            continue
        n_placed  = len(pks_ref)
        compounds = compounds[:n_placed]

        drift     = rng.integers(-MAX_DRIFT_BINS, MAX_DRIFT_BINS + 1, size=n_placed)
        pks_query = np.clip(pks_ref + drift, 2, N_BINS - 3)

        h_ref   = rng.uniform(0.3, 2.5, size=n_placed).astype(np.float32)
        h_query = (h_ref * rng.uniform(0.7, 1.3, size=n_placed)).astype(np.float32)

        chroma_ref   = _build_chromatogram(compounds, pks_ref,   h_ref,   rng)
        chroma_query = _build_chromatogram(compounds, pks_query, h_query, rng)

        for pk_r, pk_q in zip(pks_ref, pks_query):
            if len(batch_a) >= batch_size:
                break
            batch_a.append(augment(_ctx_spec(chroma_ref,   pk_r), rng))
            batch_b.append(augment(_ctx_spec(chroma_query, pk_q), rng))
            rts_a.append(float(pk_r) / N_BINS)
            rts_b.append(float(pk_q) / N_BINS)

    x1 = torch.from_numpy(np.stack(batch_a))
    x2 = torch.from_numpy(np.stack(batch_b))
    r1 = torch.tensor(rts_a, dtype=torch.float32)
    r2 = torch.tensor(rts_b, dtype=torch.float32)
    return x1, x2, r1, r2


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, n_iterations: int = N_ITERATIONS, batch_size: int = 64,
          lr: float = 1e-4, seed: int = 42) -> None:

    print("Loading MoNA train-split spectra …")
    mona = load_mona(MONA_H5, split='train')
    print(f"  {len(mona)} spectra  (1000-dim, L2-normed, no fish_oil/mtbls288)")

    model = DilatedSpectrumEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: DilatedSpectrumEncoder  ({n_params:,} params)")
    print(f"  Iterations:   {n_iterations}  |  batch: {batch_size}  |  log every: {LOG_EVERY}\n")

    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    rng   = np.random.default_rng(seed)

    model.train()
    running_loss = 0.0

    for it in range(1, n_iterations + 1):
        x1, x2, r1, r2 = _make_batch(mona, batch_size, rng)
        x1, x2 = x1.to(device), x2.to(device)
        r1, r2 = r1.to(device), r2.to(device)

        loss = nt_xent_loss(model(x1, r1), model(x2, r2))
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        running_loss += loss.item()
        if it % LOG_EVERY == 0 or it == n_iterations:
            print(f"  iter {it:>6d}/{n_iterations}  loss={running_loss / LOG_EVERY:.4f}")
            running_loss = 0.0

    out = CKPT_DIR / 'drift_simclr.pt'
    torch.save(model.state_dict(), out)
    print(f"  saved → {out}")


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")
    train(device)
    print("Done.")
