"""
Cross-study contrastive pretraining for DilatedSpectrumEncoder.

Extends pretrain_drift.py with cross-study simulation:
  - Partial compound overlap: N_SHARED metabolites appear in both "studies";
    N_UNIQUE compounds are unique to each study (co-elution background only).
  - Larger RT drift (±MAX_DRIFT_BINS = 25 bins ≈ ±5.6 min).
  - Study-level batch effects: random global intensity scale per study.
  - Per-compound spectral distortion between studies (lognormal σ=0.15)
    to simulate instrument-specific EI fragment-response differences.

Positive pairs come ONLY from shared metabolite peaks.  Study-unique peaks
contribute realistic co-elution background but do not enter the NT-Xent loss.

Why partial overlap?
--------------------
Cross-study alignment fails not just because of RT drift but because each
study contains a different subset of detectable metabolites.  A within-study
pretrained encoder sees ALL compounds in both views and learns pure drift
invariance.  Here, only N_SHARED of N_SHARED+N_UNIQUE compounds appear in
both simulated studies, forcing the encoder to identify compound identity
from spectral shape alone — the key skill needed for cross-study anchoring.

Saves: checkpoints/cross_study_simclr.pt
"""

from __future__ import annotations

import sys
import h5py
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import DilatedSpectrumEncoder, nt_xent_loss

DATA_DIR = Path(__file__).parent.parent / 'data'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'
MONA_H5  = Path('/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5')
CKPT_DIR.mkdir(exist_ok=True)

N_BINS  = 200
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS

N_ITERATIONS = 10_000
LOG_EVERY    = 200
PEAK_SIGMA   = 2.5    # bins — ~0.5 min GC-MS peak width
NOISE_SCALE  = 0.005  # exponential baseline noise

# Cross-study-specific parameters
N_SHARED            = 15            # shared metabolites → positive pairs
N_UNIQUE            = 15            # study-specific compounds → background only
MAX_DRIFT_BINS      = 25            # ±25 bins = ±5.625 min  (vs ±18 within-study)
STUDY_SCALE_RANGE   = (0.5, 2.0)   # per-study global intensity batch effect
SPECTRAL_DISTORTION = 0.15         # lognormal σ for per-fragment variation

_OFFSETS = np.arange(-8, 9, dtype=np.int32)
_GAUSS_W = np.exp(-0.5 * (_OFFSETS / PEAK_SIGMA) ** 2).astype(np.float32)


# ── MoNA data loading ─────────────────────────────────────────────────────────

def load_mona(path: Path = MONA_H5, split: str = 'train') -> np.ndarray:
    with h5py.File(path, 'r') as f:
        mask  = np.array(f['split']).astype(str) == split
        specs = f['spectra'][mask]
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (specs / norms).astype(np.float32)


# ── Synthetic chromatogram building ──────────────────────────────────────────

def _ctx_spec(chroma: np.ndarray, pk: int) -> np.ndarray:
    lo = max(0, pk - 1)
    hi = min(len(chroma), pk + 2)
    v  = chroma[lo:hi].mean(axis=0)
    n  = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _random_positions(n: int, n_bins: int, min_gap: int,
                       rng: np.random.Generator) -> np.ndarray:
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
    return chroma


def _distort_spectrum(spec: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Per-fragment lognormal distortion — simulates instrument EI response variation."""
    d = spec * rng.lognormal(0.0, sigma, size=spec.shape).astype(np.float32)
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 1e-8 else d.astype(np.float32)


# ── On-the-fly batch generation ───────────────────────────────────────────────

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    if rng.random() > 0.2:
        x = np.clip(x + rng.normal(0, 0.02, size=x.shape).astype(np.float32), 0, None)
    if rng.random() > 0.3:
        x[rng.random(size=x.shape) < 0.08] = 0.0   # 8% masking (heavier than within-study)
    if rng.random() > 0.3:
        x = x * rng.uniform(0.80, 1.20, size=x.shape).astype(np.float32)
    n = np.linalg.norm(x)
    return (x / n).astype(np.float32) if n > 1e-8 else x.astype(np.float32)


def _make_batch(mona_specs: np.ndarray, batch_size: int,
                rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    """
    Generate a batch of cross-study positive pairs on-the-fly.

    Each synthetic iteration produces two "studies" with N_SHARED shared
    metabolites and N_UNIQUE unique background compounds each.  Only shared
    metabolite peaks enter the NT-Xent loss as positive pairs.
    """
    batch_a: list[np.ndarray] = []
    batch_b: list[np.ndarray] = []
    rts_a:   list[float] = []
    rts_b:   list[float] = []

    n_pool = N_SHARED + 2 * N_UNIQUE
    replace = n_pool > len(mona_specs)

    while len(batch_a) < batch_size:
        pool_idx = rng.choice(len(mona_specs), size=n_pool, replace=replace)

        shared_idx   = pool_idx[:N_SHARED]
        unique_a_idx = pool_idx[N_SHARED : N_SHARED + N_UNIQUE]
        unique_b_idx = pool_idx[N_SHARED + N_UNIQUE:]

        # Study A: shared compounds (undistorted) + study-A-unique background
        shared_a = mona_specs[shared_idx]
        unique_a = mona_specs[unique_a_idx]

        # Study B: same compounds but instrument-distorted + study-B-unique background
        shared_b = np.array([_distort_spectrum(mona_specs[i], SPECTRAL_DISTORTION, rng)
                              for i in shared_idx])
        unique_b = mona_specs[unique_b_idx]

        # RT positions for shared compounds in study A
        pks_shared_a = _random_positions(N_SHARED, N_BINS, min_gap=6, rng=rng)
        n_s = len(pks_shared_a)
        if n_s < 2:
            continue

        # Study B: same compounds shifted by cross-study drift
        drift        = rng.integers(-MAX_DRIFT_BINS, MAX_DRIFT_BINS + 1, size=n_s)
        pks_shared_b = np.clip(pks_shared_a + drift, 2, N_BINS - 3)

        # Background compounds placed independently in each study
        pks_uniq_a = _random_positions(N_UNIQUE, N_BINS, min_gap=6, rng=rng)
        pks_uniq_b = _random_positions(N_UNIQUE, N_BINS, min_gap=6, rng=rng)
        n_ua, n_ub = len(pks_uniq_a), len(pks_uniq_b)

        # Assemble compound arrays and positions for each study
        compounds_a = np.concatenate([shared_a[:n_s], unique_a[:n_ua]])
        compounds_b = np.concatenate([shared_b[:n_s], unique_b[:n_ub]])
        pks_a       = np.concatenate([pks_shared_a,   pks_uniq_a])
        pks_b       = np.concatenate([pks_shared_b,   pks_uniq_b])

        # Study-level batch effects: independent global intensity scales
        scale_a = float(rng.uniform(*STUDY_SCALE_RANGE))
        scale_b = float(rng.uniform(*STUDY_SCALE_RANGE))
        h_a = (rng.uniform(0.3, 2.5, size=len(pks_a)) * scale_a).astype(np.float32)
        h_b = (rng.uniform(0.3, 2.5, size=len(pks_b)) * scale_b).astype(np.float32)

        chroma_a = _build_chromatogram(compounds_a, pks_a, h_a, rng)
        chroma_b = _build_chromatogram(compounds_b, pks_b, h_b, rng)

        # Only shared peaks → positive pairs in NT-Xent
        for pk_a_i, pk_b_i in zip(pks_shared_a[:n_s], pks_shared_b):
            if len(batch_a) >= batch_size:
                break
            batch_a.append(augment(_ctx_spec(chroma_a, pk_a_i), rng))
            batch_b.append(augment(_ctx_spec(chroma_b, pk_b_i), rng))
            rts_a.append(float(pk_a_i) / N_BINS)
            rts_b.append(float(pk_b_i) / N_BINS)

    x1 = torch.from_numpy(np.stack(batch_a[:batch_size]))
    x2 = torch.from_numpy(np.stack(batch_b[:batch_size]))
    r1 = torch.tensor(rts_a[:batch_size], dtype=torch.float32)
    r2 = torch.tensor(rts_b[:batch_size], dtype=torch.float32)
    return x1, x2, r1, r2


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, n_iterations: int = N_ITERATIONS, batch_size: int = 64,
          lr: float = 1e-4, seed: int = 42) -> None:

    print("Loading MoNA train-split spectra …")
    mona = load_mona(MONA_H5, split='train')
    print(f"  {len(mona)} spectra  (1000-dim, L2-normed, no downstream data)")

    model = DilatedSpectrumEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : DilatedSpectrumEncoder  ({n_params:,} params)")
    print(f"  N_SHARED={N_SHARED}  N_UNIQUE={N_UNIQUE}  "
          f"MAX_DRIFT=±{MAX_DRIFT_BINS} bins (±{MAX_DRIFT_BINS*BIN_MIN:.2f} min)")
    print(f"  Study scale  : {STUDY_SCALE_RANGE}   "
          f"Spectral distortion σ={SPECTRAL_DISTORTION}")
    print(f"  Iterations   : {n_iterations}  |  batch: {batch_size}  "
          f"|  log every: {LOG_EVERY}\n")

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

    out = CKPT_DIR / 'cross_study_simclr.pt'
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
