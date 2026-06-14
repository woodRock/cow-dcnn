"""
Pretraining for SparseSpectrumTransformer using synthetic GC-MS data.

Data generation is identical to pretrain_drift.py (MoNA train split, synthetic
chromatograms, on-the-fly batch generation) — imported from there to avoid
duplication.  Only the model and checkpoint path differ.

Architecture: SparseSpectrumTransformer
  d_model=64, nhead=4, n_layers=3, dim_ff=128 → ~198 K params
  Tokenises each spectrum into its top-128 m/z peaks; self-attention over
  intensity-weighted m/z embeddings.

Checkpoint: checkpoints/transformer_simclr.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ and scripts/ to path before any local imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.optim as optim

# Reuse all data-generation utilities from pretrain_drift — no duplication
from pretrain_drift import (
    MONA_H5, CKPT_DIR, N_BINS, N_ITERATIONS, LOG_EVERY,
    load_mona, _make_batch,
)
from encoder import SparseSpectrumTransformer, nt_xent_loss


def train(device, n_iterations: int = N_ITERATIONS, batch_size: int = 64,
          lr: float = 1e-4, seed: int = 42) -> None:

    print("Loading MoNA train-split spectra …")
    mona = load_mona(MONA_H5, split='train')
    print(f"  {len(mona)} spectra  (1000-dim, L2-normed)")

    model = SparseSpectrumTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: SparseSpectrumTransformer  ({n_params:,} params)")
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

    out = CKPT_DIR / 'transformer_simclr.pt'
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
