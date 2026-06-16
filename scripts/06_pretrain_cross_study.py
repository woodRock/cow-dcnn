"""
06_pretrain_cross_study.py — Cross-study SimCLR pretraining with PeakSequenceTransformer.

Trains PeakSequenceTransformer on synthetic cross-study chromatogram pairs.

Architecture:
  Per-peak: dilated-CNN over the m/z axis → d_model-dim token
  Global:   sinusoidal RT positional encoding + Transformer self-attention over
            all N detected peaks in a chromatogram

Why a Transformer over peak sequences?
  Compound elution order is conserved across studies even when absolute RTs drift
  (compound A always elutes before compound B).  A local-window encoder (e.g.
  ChromaSpectrumEncoder) sees at most ±7 bins (3.4 min) of RT context and cannot
  capture this global monotonic constraint.  The Transformer sees the full sequence
  of N peaks at once, so each peak's embedding encodes both its m/z identity and
  its position in the elution order — a signal that is consistent across studies.

Training:
  Each iteration generates N_PAIRS_PER_ITER synthetic "study pairs".  Each pair has
  N_SHARED shared compounds (same order, different absolute RT) and N_UNIQUE study-
  specific background compounds.  All N_TOTAL = 25 peaks are encoded together through
  the Transformer.  Only the N_SHARED shared-peak embeddings enter the NT-Xent loss.

Output: checkpoints/cross_study_simclr.pt
"""
from __future__ import annotations

import math
import sys
import h5py
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import PeakSequenceTransformer, nt_xent_loss

DATA_DIR = Path(__file__).parent.parent / 'data'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'
MONA_H5  = Path('/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5')
CKPT_DIR.mkdir(exist_ok=True)

N_BINS  = 200
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS

N_ITERATIONS = 10_000
LOG_EVERY    = 200
BATCH_SIZE   = 64
PEAK_SIGMA   = 2.5    # bins — ~0.5 min GC-MS peak width
NOISE_SCALE  = 0.005  # exponential baseline noise

# Cross-study-specific parameters — tuned to fish oil batch drift characteristics:
#   Same species/instrument across dates → high peak recurrence, stable spectra, moderate drift.
#   Observed: mean pre-warp RT dev 2.5 min, precision=1.0 (negligible spectral distortion).
N_SHARED            = 20            # shared metabolites → positive pairs (same-species ~80% overlap)
N_UNIQUE            = 5             # study-specific background (few batch-exclusive peaks)
N_TOTAL             = N_SHARED + N_UNIQUE   # fixed sequence length (25 peaks)
MAX_DRIFT_BINS      = 13            # ±13 bins = ±2.925 min — calibrated to fish oil observed drift
STUDY_SCALE_RANGE   = (0.8, 1.4)   # per-study global intensity batch effect (same instrument)
SPECTRAL_DISTORTION = 0.04         # lognormal σ — lowered: fish oil spectra near-identical across batches

# Each batch encodes N_PAIRS_PER_ITER full chromatogram pairs.
# N_SHARED embeddings per pair → total > BATCH_SIZE, trimmed to BATCH_SIZE.
N_PAIRS_PER_ITER = math.ceil(BATCH_SIZE / N_SHARED)   # = 4

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
    """3-bin mean fingerprint at peak pk, L2-normalised. Shape: (1000,)."""
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
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
    d = spec * rng.lognormal(0.0, sigma, size=spec.shape).astype(np.float32)
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 1e-8 else d.astype(np.float32)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SimCLR augmentations for a single (1000,) L2-normed m/z fingerprint."""
    x = x.copy()
    if rng.random() > 0.2:
        x = np.clip(x + rng.normal(0, 0.02, size=x.shape).astype(np.float32), 0, None)
    if rng.random() > 0.3:
        x[rng.random(size=x.shape) < 0.08] = 0.0
    if rng.random() > 0.3:
        x *= float(rng.uniform(0.80, 1.20))
    n = np.linalg.norm(x)
    return (x / n).astype(np.float32) if n > 1e-8 else x.astype(np.float32)


# ── Batch generation ──────────────────────────────────────────────────────────

def _generate_pair(mona_specs: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """
    Generate one pair of synthetic chromatograms sharing N_SHARED compounds.

    Both chromatograms have exactly N_TOTAL peaks sorted by ascending RT.
    The N_SHARED shared peaks appear in the same relative elution order in both
    studies but with RT drift of up to ±MAX_DRIFT_BINS bins.

    Returns:
        fps_a  : (N_TOTAL, 1000)  augmented per-peak fingerprints for study A
        rts_a  : (N_TOTAL,)      normalised RT positions for study A [0, 1]
        fps_b  : (N_TOTAL, 1000)  augmented per-peak fingerprints for study B
        rts_b  : (N_TOTAL,)      normalised RT positions for study B [0, 1]
        mask_a : (N_TOTAL,) bool  True at positions of shared peaks in study A
        mask_b : (N_TOTAL,) bool  True at positions of shared peaks in study B
    """
    n_pool  = N_SHARED + 2 * N_UNIQUE
    replace = n_pool > len(mona_specs)

    while True:
        pool_idx     = rng.choice(len(mona_specs), size=n_pool, replace=replace)
        shared_idx   = pool_idx[:N_SHARED]
        unique_a_idx = pool_idx[N_SHARED:N_SHARED + N_UNIQUE]
        unique_b_idx = pool_idx[N_SHARED + N_UNIQUE:]

        shared_a = mona_specs[shared_idx]
        unique_a = mona_specs[unique_a_idx]
        shared_b = np.array([_distort_spectrum(mona_specs[i], SPECTRAL_DISTORTION, rng)
                              for i in shared_idx])
        unique_b = mona_specs[unique_b_idx]

        pks_shared_a = _random_positions(N_SHARED, N_BINS, min_gap=6, rng=rng)
        n_s = len(pks_shared_a)
        if n_s < N_SHARED:
            continue

        drift        = rng.integers(-MAX_DRIFT_BINS, MAX_DRIFT_BINS + 1, size=n_s)
        pks_shared_b = np.clip(pks_shared_a + drift, 2, N_BINS - 3)

        pks_uniq_a = _random_positions(N_UNIQUE, N_BINS, min_gap=6, rng=rng)
        pks_uniq_b = _random_positions(N_UNIQUE, N_BINS, min_gap=6, rng=rng)
        n_ua = len(pks_uniq_a)
        n_ub = len(pks_uniq_b)

        compounds_a = np.concatenate([shared_a[:n_s], unique_a[:n_ua]])
        compounds_b = np.concatenate([shared_b[:n_s], unique_b[:n_ub]])
        pks_a_all   = np.concatenate([pks_shared_a, pks_uniq_a])
        pks_b_all   = np.concatenate([pks_shared_b, pks_uniq_b])

        scale_a = float(rng.uniform(*STUDY_SCALE_RANGE))
        scale_b = float(rng.uniform(*STUDY_SCALE_RANGE))
        h_a = (rng.uniform(0.3, 2.5, size=len(pks_a_all)) * scale_a).astype(np.float32)
        h_b = (rng.uniform(0.3, 2.5, size=len(pks_b_all)) * scale_b).astype(np.float32)

        chroma_a = _build_chromatogram(compounds_a, pks_a_all, h_a, rng)
        chroma_b = _build_chromatogram(compounds_b, pks_b_all, h_b, rng)

        # is_shared: first n_s entries (shared) are True, remaining (unique) are False
        is_shared_a = np.array([True] * n_s + [False] * n_ua, dtype=bool)
        is_shared_b = np.array([True] * n_s + [False] * n_ub, dtype=bool)

        # Sort by RT — Transformer input must be in ascending RT order
        sort_a = np.argsort(pks_a_all)
        sort_b = np.argsort(pks_b_all)
        pks_a_sorted  = pks_a_all[sort_a]
        pks_b_sorted  = pks_b_all[sort_b]
        mask_a_sorted = is_shared_a[sort_a]
        mask_b_sorted = is_shared_b[sort_b]

        # Extract fingerprints with per-peak augmentation
        fps_a_list = [augment(_ctx_spec(chroma_a, int(pk)), rng) for pk in pks_a_sorted]
        fps_b_list = [augment(_ctx_spec(chroma_b, int(pk)), rng) for pk in pks_b_sorted]

        # Pack into fixed-length N_TOTAL arrays (pad with zeros if somehow short)
        fps_a  = np.zeros((N_TOTAL, 1000), dtype=np.float32)
        fps_b  = np.zeros((N_TOTAL, 1000), dtype=np.float32)
        rts_a  = np.zeros(N_TOTAL, dtype=np.float32)
        rts_b  = np.zeros(N_TOTAL, dtype=np.float32)
        mask_a = np.zeros(N_TOTAL, dtype=bool)
        mask_b = np.zeros(N_TOTAL, dtype=bool)

        na = min(len(fps_a_list), N_TOTAL)
        nb = min(len(fps_b_list), N_TOTAL)
        fps_a[:na]  = np.array(fps_a_list[:na])
        fps_b[:nb]  = np.array(fps_b_list[:nb])
        rts_a[:na]  = pks_a_sorted[:na].astype(np.float32) / N_BINS
        rts_b[:nb]  = pks_b_sorted[:nb].astype(np.float32) / N_BINS
        mask_a[:na] = mask_a_sorted[:na]
        mask_b[:nb] = mask_b_sorted[:nb]

        return fps_a, rts_a, fps_b, rts_b, mask_a, mask_b


def _make_batch(mona_specs: np.ndarray, n_pairs: int,
                rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """
    Stack n_pairs chromatogram pairs along a batch dimension.

    Returns:
        fps_a   : (P, N_TOTAL, 1000)
        rts_a   : (P, N_TOTAL)
        fps_b   : (P, N_TOTAL, 1000)
        rts_b   : (P, N_TOTAL)
        masks_a : (P, N_TOTAL) bool
        masks_b : (P, N_TOTAL) bool
    """
    pairs = [_generate_pair(mona_specs, rng) for _ in range(n_pairs)]
    return (
        np.stack([p[0] for p in pairs]),
        np.stack([p[1] for p in pairs]),
        np.stack([p[2] for p in pairs]),
        np.stack([p[3] for p in pairs]),
        np.stack([p[4] for p in pairs]),
        np.stack([p[5] for p in pairs]),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, n_iterations: int = N_ITERATIONS, batch_size: int = BATCH_SIZE,
          lr: float = 1e-4, seed: int = 42) -> None:

    print("Loading MoNA train-split spectra …")
    mona = load_mona(MONA_H5, split='train')
    print(f"  {len(mona)} spectra  (1000-dim, L2-normed)")

    model = PeakSequenceTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : PeakSequenceTransformer  ({n_params:,} params)")
    print(f"  Sequence len : {N_TOTAL} peaks  ({N_SHARED} shared + {N_UNIQUE} unique per study)")
    print(f"  Pairs/iter   : {N_PAIRS_PER_ITER}  →  {N_PAIRS_PER_ITER * N_SHARED} embeddings "
          f"trimmed to batch_size={batch_size}")
    print(f"  MAX_DRIFT=±{MAX_DRIFT_BINS} bins (±{MAX_DRIFT_BINS * BIN_MIN:.2f} min)")
    print(f"  Study scale: {STUDY_SCALE_RANGE}   Spectral distortion σ={SPECTRAL_DISTORTION}")
    print(f"  Iterations: {n_iterations}  |  lr: {lr}  |  log every: {LOG_EVERY}\n")

    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    rng   = np.random.default_rng(seed)

    model.train()
    running_loss = 0.0

    for it in range(1, n_iterations + 1):
        fps_a, rts_a, fps_b, rts_b, masks_a, masks_b = _make_batch(mona, N_PAIRS_PER_ITER, rng)

        fps_a_t = torch.from_numpy(fps_a).to(device)   # (P, N_TOTAL, 1000)
        rts_a_t = torch.from_numpy(rts_a).to(device)   # (P, N_TOTAL)
        fps_b_t = torch.from_numpy(fps_b).to(device)
        rts_b_t = torch.from_numpy(rts_b).to(device)

        # Encode full peak sequences — Transformer attends across all N_TOTAL peaks
        emb_a = model.encode(fps_a_t, rts_a_t)   # (P, N_TOTAL, 128)
        emb_b = model.encode(fps_b_t, rts_b_t)

        # Vectorised extraction of shared-peak embeddings (one MPS op, not P small ones)
        ma = torch.from_numpy(masks_a).to(device)   # (P, N_TOTAL) bool
        mb = torch.from_numpy(masks_b).to(device)
        z1 = emb_a[ma][:batch_size]   # (B, 128) — shared peaks across all pairs
        z2 = emb_b[mb][:batch_size]

        loss = nt_xent_loss(model.project(z1), model.project(z2))
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
