"""
Train ChromaWarpTransformer on synthetic GC-MS mini-datasets with known warps.

Each training iteration builds a mini-dataset of N_SAMPLES chromatograms:
  1. Sample N_COMP compounds from MoNA train split.
  2. Place them at random RT positions → consensus (reference) peak positions.
  3. For each sample: apply per-compound drift, enforcing monotone peak order
     (compounds don't swap elution order under realistic GC drift).
  4. Build the drifted chromatogram and compute the true dense inverse warp:
       true_warp[t] = where in the drifted chromatogram to sample for output bin t
     Derived by linear interpolation over the (pks_ref, pks_drifted) control points.
  5. Forward pass through ChromaWarpTransformer → predicted warps.
  6. Loss = warp MSE (normalised to [0,1]) + λ · smoothness regulariser.

N_SAMPLES is randomised each iteration (4–16) so the model generalises to
variable dataset sizes — it is tested on N=38 at inference time.

No real fish-oil data is used during training: zero leakage.
Parameters calibrated to the observed fish-oil batch drift characteristics:
  MAX_DRIFT_BINS = 13  (±2.9 min, matching observed mean pre-warp RT deviation ~2.5 min)
"""

from __future__ import annotations

import sys
import h5py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import ChromaWarpTransformer, warp_smoothness_loss

CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'
MONA_H5  = Path('/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5')
CKPT_DIR.mkdir(exist_ok=True)

N_BINS         = 200
MZ_BINS        = 1000
N_COMP         = 30           # compounds per synthetic chromatogram
PEAK_SIGMA     = 2.5          # bins — ~0.5 min GC-MS peak width
WITHIN_DRIFT   = 4            # ±4 bins = ±0.9 min — within-batch instrument noise
MAX_DRIFT_BINS = 25           # up to ±25 bins = ±5.6 min — covers observed p90 drift
MIN_DRIFT      = 5            # minimum cross-study offset: study 1 always meaningfully shifted
NOISE_SCALE    = 0.005        # exponential baseline noise
N_ITERATIONS   = 10_000
LOG_EVERY      = 200
LAMBDA_SMOOTH  = 0.01         # smoothness regulariser weight

_OFFSETS = np.arange(-8, 9, dtype=np.int32)
_GAUSS_W = np.exp(-0.5 * (_OFFSETS / PEAK_SIGMA) ** 2).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mona(path: Path = MONA_H5) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        mask  = np.array(f['split']).astype(str) == 'train'
        specs = f['spectra'][mask]
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (specs / norms).astype(np.float32)


# ── Synthetic chromatogram building ──────────────────────────────────────────

def _random_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.arange(10, N_BINS - 10)
    rng.shuffle(pool)
    positions: list[int] = []
    for c in pool.tolist():
        if not positions or min(abs(c - p) for p in positions) >= 6:
            positions.append(c)
        if len(positions) == n:
            break
    return np.array(sorted(positions), dtype=np.int32)


def _build_chromatogram(compounds: np.ndarray, positions: np.ndarray,
                         heights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n           = len(compounds)
    bins_all    = positions[:, None].astype(np.int32) + _OFFSETS[None, :]
    weights_all = heights[:, None] * _GAUSS_W[None, :]
    bins_flat   = bins_all.ravel()
    w_flat      = weights_all.ravel()
    comp_idx    = np.repeat(np.arange(n, dtype=np.int32), 17)
    valid       = (bins_flat >= 0) & (bins_flat < N_BINS)
    W = np.zeros((N_BINS, n), dtype=np.float32)
    W[bins_flat[valid], comp_idx[valid]] = w_flat[valid]
    chroma = W @ compounds
    chroma += rng.exponential(NOISE_SCALE, size=chroma.shape).astype(np.float32)
    # sqrt compression + per-scan L2 norm — matches fish_oil preprocessing
    chroma = np.sqrt(np.clip(chroma, 0, None))
    norms  = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return (chroma / norms).astype(np.float32)


def _drifted_positions(pks_ref: np.ndarray, global_offset: int,
                        within: int, rng: np.random.Generator) -> np.ndarray:
    """Apply global offset + per-compound noise; enforce monotone order."""
    drift    = rng.integers(-within, within + 1, size=len(pks_ref))
    pks_samp = pks_ref.copy().astype(np.int32) + global_offset + drift
    pks_samp[0] = int(np.clip(pks_samp[0], 2, N_BINS - 3))
    for i in range(1, len(pks_samp)):
        pks_samp[i] = max(int(pks_samp[i]), int(pks_samp[i - 1]) + 1)
    return np.clip(pks_samp, 2, N_BINS - 3).astype(np.int32)


def _generate_mini_dataset(mona: np.ndarray, n_samples: int,
                            rng: np.random.Generator):
    """
    Build a two-study mini-dataset: n_a study-0 samples + n_b study-1 samples.

    Study 0 (reference batch): small within-batch drift (±WITHIN_DRIFT bins).
    Study 1 (query batch): same within-batch noise PLUS a random global RT offset
      that simulates systematic cross-study retention-time drift.

    This mirrors real fish oil batches: sep2015 ≈ study 0, jul2016 ≈ study 1.
    The study_ids label lets the global attention distinguish RT drift from
    biological variation between samples.

    Returns (chromas [N, T, M], true_warps [N, T], study_ids [N]) or None.
    """
    idx       = rng.choice(len(mona), N_COMP, replace=False)
    compounds = mona[idx]

    pks_ref = _random_positions(N_COMP, rng)
    if len(pks_ref) < 5:
        return None
    n_placed  = len(pks_ref)
    compounds = compounds[:n_placed]

    # Cross-study offset: always a meaningful shift (≥ MIN_DRIFT), random direction.
    # Avoids training on near-zero offsets so the model learns to apply larger warps.
    magnitude     = int(rng.integers(MIN_DRIFT, MAX_DRIFT_BINS + 1))
    global_offset = magnitude * (1 if rng.random() > 0.5 else -1)

    n_a = max(1, n_samples // 2)
    n_b = n_samples - n_a

    chromas:   list[np.ndarray] = []
    warps:     list[np.ndarray] = []
    study_ids: list[int]        = []

    for study, n_study, offset in [(0, n_a, 0), (1, n_b, global_offset)]:
        for _ in range(n_study):
            pks_samp = _drifted_positions(pks_ref, offset, WITHIN_DRIFT, rng)
            heights  = rng.uniform(0.3, 2.5, n_placed).astype(np.float32)
            chroma   = _build_chromatogram(compounds, pks_samp, heights, rng)

            # Inverse warp: output bin t (consensus frame) → source bin in this sample
            true_warp = np.interp(
                np.arange(N_BINS, dtype=np.float32),
                pks_ref.astype(np.float32),
                pks_samp.astype(np.float32),
            ).clip(0, N_BINS - 1).astype(np.float32)

            chromas.append(chroma)
            warps.append(true_warp)
            study_ids.append(study)

    return (
        np.stack(chromas),
        np.stack(warps),
        np.array(study_ids, dtype=np.int64),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, n_iterations: int = N_ITERATIONS, lr: float = 3e-4, seed: int = 42):
    print("Loading MoNA train-split spectra …")
    mona = load_mona()
    print(f"  {len(mona)} spectra  (1000-dim, L2-normed)")

    model = ChromaWarpTransformer().to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  ChromaWarpTransformer  {n_p:,} parameters")
    print(f"  MAX_DRIFT=±{MAX_DRIFT_BINS} bins (±{MAX_DRIFT_BINS * 45.0 / N_BINS:.2f} min)")
    print(f"  N_SAMPLES per iter: 4–16 (randomised, split ~50/50 study 0 / study 1)")
    print(f"  N_COMP={N_COMP}   WITHIN_DRIFT=±{WITHIN_DRIFT}   cross-study offset=[{MIN_DRIFT},{MAX_DRIFT_BINS}] bins (directional)")
    print(f"  Iterations={n_iterations}   λ_smooth={LAMBDA_SMOOTH}\n")

    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    rng     = np.random.default_rng(seed)
    running = 0.0

    for it in range(1, n_iterations + 1):
        n_samples = int(rng.integers(4, 17))   # variable N so model generalises
        result    = _generate_mini_dataset(mona, n_samples, rng)
        if result is None:
            continue

        chromas_np, true_warps_np, sids_np = result
        chromas    = torch.from_numpy(chromas_np).to(device)      # [N, T, M]
        true_warps = torch.from_numpy(true_warps_np).to(device)   # [N, T]
        study_ids  = torch.from_numpy(sids_np).to(device)         # [N]

        _, pred_warps = model(chromas, study_ids)

        # MSE in normalised [0,1] space + smoothness on predicted warp
        T    = N_BINS
        loss = (
            F.mse_loss(pred_warps / (T - 1), true_warps / (T - 1))
            + LAMBDA_SMOOTH * warp_smoothness_loss(pred_warps)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()
        if it % LOG_EVERY == 0 or it == n_iterations:
            print(f"  iter {it:>6d}/{n_iterations}  loss={running / LOG_EVERY:.5f}")
            running = 0.0

    out = CKPT_DIR / 'warp_transformer.pt'
    torch.save(model.state_dict(), out)
    print(f"\n  saved → {out}")


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
