"""
Supervised pretraining for ChromaRegNet on synthetic GC-MS chromatogram pairs.

Uses the same MoNA-based synthetic data generator as pretrain_drift.py — same
compounds, same Gaussian peak shapes, same random RT drift model.  The key
difference: instead of SimCLR on per-peak spectra, ChromaRegNet is trained to
predict the ground-truth displacement field from (query TIC, reference TIC).

Ground-truth construction
--------------------------
Each synthetic pair has known per-compound drifts (pks_query - pks_ref).
These sparse anchor displacements are linearly interpolated to a dense
N_BINS-dim vector d[t], where d[t] = how far to shift the query at time t
to land on the reference.  Boundary conditions: d[0] = d[N_BINS-1] = 0.

Loss
----
  MSE(predicted_field, gt_field)  +  λ · smoothness(predicted_field)

This directly supervises the displacement magnitude and direction at every
RT bin, giving a much stronger signal than unsupervised NCC.

Checkpoint: checkpoints/regnet.pt
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from regnet import ChromaRegNet, smoothness_loss
from pretrain_drift import (
    MONA_H5, CKPT_DIR, N_BINS, N_COMP, NOISE_SCALE,
    load_mona, _build_chromatogram, _random_positions,
)

N_ITERATIONS  = 10_000
LOG_EVERY     = 200
BATCH_SIZE    = 32
LR            = 1e-3
LAMBDA_SMOOTH = 0.1
N_POOL        = 4_000

# Drift model calibrated to fish_oil empirical distribution:
#   74.6% of drifts within ±3 bins, p90=12 bins, max observed=28 bins.
#   Smooth correlated drift replaces previous independent per-compound uniform.
DRIFT_SIGMA   = 5.0   # std of first-harmonic amplitude (bins)
MAX_DISP      = 20    # hard clamp on predicted and GT displacement (bins)
N_HARMONICS   = 3     # number of sinusoidal components


# ── Smooth drift function ─────────────────────────────────────────────────────

def _smooth_drift_fn(rng: np.random.Generator,
                     n_bins: int      = N_BINS,
                     sigma: float     = DRIFT_SIGMA,
                     max_disp: float  = MAX_DISP,
                     n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """
    Smooth RT drift function as a sum of low-frequency sinusoids.

    Uses the basis sin(k·π·t/(n_bins-1)) for k=1..n_harmonics.  Each basis
    function is exactly 0 at t=0 and t=n_bins-1, so boundary conditions are
    satisfied automatically — no padding or pinning required.

    Higher harmonics receive decreasing amplitude (sigma/k), ensuring
    low-frequency drift dominates.  This matches real GC-MS behaviour where
    column-temperature drift accumulates slowly and smoothly over the run.

    Returns float32 array of shape (n_bins,).
    """
    t = np.linspace(0.0, np.pi, n_bins, dtype=np.float32)
    d = np.zeros(n_bins, dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        amp = float(rng.normal(0.0, sigma / k))
        d  += amp * np.sin(k * t)
    return np.clip(d, -max_disp, max_disp).astype(np.float32)


# ── Pool pre-generation ───────────────────────────────────────────────────────

def _pregenerate_pool(mona_specs: np.ndarray, n_pool: int,
                      rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    """
    Pre-generate n_pool synthetic (query_TIC, ref_TIC, gt_displacement) triples.

    Drift model: smooth sinusoidal function calibrated to fish_oil empirical
    distribution (σ=5 bins, max=20 bins).  All compounds in a chromatogram
    share the same smooth drift function — correlated displacement, not
    independent per-compound noise.  The GT field is the smooth function
    itself, so no re-interpolation is needed.
    """
    tics_q: list[np.ndarray] = []
    tics_r: list[np.ndarray] = []
    disps:  list[np.ndarray] = []

    while len(tics_q) < n_pool:
        idx       = rng.choice(len(mona_specs), size=N_COMP, replace=False)
        compounds = mona_specs[idx]

        pks_ref = _random_positions(N_COMP, N_BINS, min_gap=6, rng=rng)
        if len(pks_ref) < 2:
            continue
        n         = len(pks_ref)
        compounds = compounds[:n]

        # Smooth correlated drift: one function, applied to all compounds
        drift_fn     = _smooth_drift_fn(rng)
        drifts       = np.round(drift_fn[pks_ref]).astype(int)
        pks_query    = np.clip(pks_ref + drifts, 2, N_BINS - 3)

        h_ref   = rng.uniform(0.3, 2.5, size=n).astype(np.float32)
        h_query = (h_ref * rng.uniform(0.7, 1.3, size=n)).astype(np.float32)

        chroma_ref   = _build_chromatogram(compounds, pks_ref,   h_ref,   rng)
        chroma_query = _build_chromatogram(compounds, pks_query, h_query, rng)

        tic_r = chroma_ref.sum(  axis=1).astype(np.float32)
        tic_q = chroma_query.sum(axis=1).astype(np.float32)
        nr = np.linalg.norm(tic_r); tic_r = tic_r / nr if nr > 1e-8 else tic_r
        nq = np.linalg.norm(tic_q); tic_q = tic_q / nq if nq > 1e-8 else tic_q

        tics_r.append(tic_r)
        tics_q.append(tic_q)
        disps.append(drift_fn)   # GT IS the smooth function — no re-interpolation

        if len(tics_q) % 500 == 0:
            print(f"  generated {len(tics_q)}/{n_pool} …")

    return (
        torch.from_numpy(np.stack(tics_q)),
        torch.from_numpy(np.stack(tics_r)),
        torch.from_numpy(np.stack(disps)),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(device, n_iterations: int = N_ITERATIONS, batch_size: int = BATCH_SIZE,
          lr: float = LR, lam: float = LAMBDA_SMOOTH, seed: int = 42) -> None:

    print("Loading MoNA train-split spectra …")
    mona = load_mona(MONA_H5, split='train')
    print(f"  {len(mona)} spectra")

    print(f"Pre-generating pool of {N_POOL} synthetic pairs …")
    rng = np.random.default_rng(seed)
    pool_q, pool_r, pool_d = _pregenerate_pool(mona, N_POOL, rng)
    pool_q = pool_q.to(device)
    pool_r = pool_r.to(device)
    pool_d = pool_d.to(device)
    print(f"  done — pool shape: {pool_q.shape}\n")

    model   = ChromaRegNet(n_bins=N_BINS).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"ChromaRegNet: {n_param:,} parameters")
    print(f"Iterations: {n_iterations}  |  batch: {batch_size}  "
          f"|  λ_smooth: {lam}  |  log every: {LOG_EVERY}\n")

    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    t_rng = torch.Generator(device='cpu')
    t_rng.manual_seed(seed)

    model.train()
    run_mse = run_sm = 0.0

    for it in range(1, n_iterations + 1):
        idx     = torch.randint(0, N_POOL, (batch_size,), generator=t_rng)
        q_tic   = pool_q[idx]
        r_tic   = pool_r[idx]
        gt_disp = pool_d[idx]

        x    = torch.stack([q_tic, r_tic], dim=1)   # (B, 2, L)
        pred = model(x).clamp(-MAX_DISP, MAX_DISP)  # (B, L)

        l_mse = F.mse_loss(pred, gt_disp)
        l_sm  = smoothness_loss(pred)
        loss  = l_mse + lam * l_sm

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        run_mse += l_mse.item()
        run_sm  += l_sm.item()

        if it % LOG_EVERY == 0 or it == n_iterations:
            print(f"  iter {it:>6d}/{n_iterations}  "
                  f"mse={run_mse/LOG_EVERY:.4f}  "
                  f"smooth={run_sm/LOG_EVERY:.4f}")
            run_mse = run_sm = 0.0

    out = CKPT_DIR / 'regnet.pt'
    torch.save(model.state_dict(), out)
    print(f"\nSaved → {out}")


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
