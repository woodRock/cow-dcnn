"""
15_finetune_cross_batch.py — Cross-batch fine-tuning of ChromaWarpTransformer.

Adapts the synthetically pretrained model to real fish oil GC-MS data using
a held-out batch pair, so the evaluation pair (sep2015 vs jul2016) is never
seen during training — no leakage.

Split
-----
  Fine-tune : batch_jan2016 (study 0, 26 samples)
              batch_apr2016 (study 1, 10 samples)
              3-month acquisition gap — genuine cross-batch RT drift present
  Evaluate  : batch_sep2015 vs batch_jul2016  (script 14, untouched)

Why this helps
--------------
Synthetic MoNA pretraining (script 13) teaches the model the correct warp
direction and dataset-level reasoning, but the model has never seen fish oil
FAME spectra.  Fine-tuning on the jan→apr pair exposes it to:
  • Real fish oil fatty-acid m/z fingerprints (FAMEs dominate the chromatogram)
  • Realistic warp magnitudes for this instrument and protocol
  • The smooth monotone drift profile characteristic of GC column ageing

That domain knowledge then generalises to the sep2015→jul2016 pair because
the same instrument, method, and compound classes are involved — only the
time gap differs (3 months → 10 months).

Loss
----
  -mean_pairwise_TIC_r(aligned_A, aligned_B)   unsupervised, no class labels
  + λ_smooth * warp_velocity_variance           prevents jittery warps

Saves to: checkpoints/warp_transformer_crossbatch.pt
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import ChromaWarpTransformer, warp_smoothness_loss
from alignment import load_chromas

DATA_DIR = Path(__file__).parent.parent / 'data'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'

N_BINS        = 200
BIN_MIN       = 45.0 / N_BINS
N_ITERATIONS  = 1_000
LOG_EVERY     = 100
LR            = 5e-5    # conservative — preserve synthetic pretraining structure
LAMBDA_SMOOTH = 0.05
LAMBDA_REF    = 0.10   # anchor reference batch (study 0) near identity warp
AUG_EXTRA_MIN = 5      # augment batch-B with extra shift: uniform in [AUG_EXTRA_MIN, AUG_EXTRA_MAX]
AUG_EXTRA_MAX = 25     # bins — simulates larger time gaps without extra real data


# ── Augmentation ─────────────────────────────────────────────────────────────

def _shift_batch(chromas: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Cyclically shift the RT axis by `shift` bins (positive = shift right).
    Used to augment the query batch with an extra RT offset so the model
    learns to apply larger warps than the raw 3-month fine-tune gap provides.
    """
    return torch.roll(chromas, shifts=shift, dims=1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def neg_mean_tic_r(aligned: torch.Tensor, study_ids: torch.Tensor) -> torch.Tensor:
    """
    Negative mean pairwise Pearson TIC r between study A and study B.

    Gradient flows: loss → TIC of aligned chromas → interpolation alpha → warps
                         → warp_head → encoder weights.
    """
    tic   = aligned.sum(dim=-1)               # [N, T]
    tic_a = tic[study_ids == 0]               # [n_a, T]
    tic_b = tic[study_ids == 1]               # [n_b, T]
    tic_a = F.normalize(tic_a - tic_a.mean(dim=-1, keepdim=True), dim=-1)
    tic_b = F.normalize(tic_b - tic_b.mean(dim=-1, keepdim=True), dim=-1)
    return -(tic_a @ tic_b.T).mean()


# ── Main ──────────────────────────────────────────────────────────────────────

def finetune(train_a_dir: Path, train_b_dir: Path, label_a: str, label_b: str,
             n_iterations: int = N_ITERATIONS, lr: float = LR, seed: int = 42):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")

    print("Fine-tune pair (held-out from evaluation) …")
    chromas_a = load_chromas(train_a_dir / 'chroma')
    chromas_b = load_chromas(train_b_dir / 'chroma')
    n_a, n_b  = len(chromas_a), len(chromas_b)
    print(f"  {label_a}: {n_a} samples  (study 0 — reference)")
    print(f"  {label_b}: {n_b} samples  (study 1 — query)")
    print(f"  Evaluation pair remains: batch_sep2015 vs batch_jul2016\n")

    chromas_np = np.stack(chromas_a + chromas_b, axis=0)
    chromas_t  = torch.from_numpy(chromas_np).to(device)
    study_ids  = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long, device=device)

    # Load synthetically pretrained model
    ckpt_in = CKPT_DIR / 'warp_transformer.pt'
    model   = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt_in, map_location=device))
    model.to(device).train()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ckpt_in.name}  ({n_p:,} parameters)")

    # Baseline before fine-tuning
    with torch.no_grad():
        al0, w0 = model(chromas_t, study_ids)
    base_tic_r  = float(-neg_mean_tic_r(al0, study_ids))
    base_warp_a = float((w0[:n_a] - torch.arange(N_BINS, device=device).float()).abs().mean()) * BIN_MIN
    base_warp_b = float((w0[n_a:] - torch.arange(N_BINS, device=device).float()).abs().mean()) * BIN_MIN
    print(f"Baseline TIC r={base_tic_r:.4f}  warp_mag: {label_a}={base_warp_a:.3f}m  {label_b}={base_warp_b:.3f}m")
    print(f"\nFine-tuning: lr={lr}  iterations={n_iterations}  λ_smooth={LAMBDA_SMOOTH}\n")

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    rng   = np.random.default_rng(seed)
    torch.manual_seed(seed)

    best_loss  = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for it in range(1, n_iterations + 1):
        # Warp-scale augmentation: shift query batch (study 1) by a random extra
        # offset each iteration.  The TIC loss then rewards proportionally larger
        # warp predictions, teaching the model to generalise to bigger time gaps.
        extra_shift = int(rng.integers(AUG_EXTRA_MIN, AUG_EXTRA_MAX + 1))
        chromas_aug = chromas_t.clone()
        chromas_aug[study_ids == 1] = _shift_batch(
            chromas_t[study_ids == 1], extra_shift
        )

        aligned, warps = model(chromas_aug, study_ids)
        loss = (neg_mean_tic_r(aligned, study_ids)
                + LAMBDA_SMOOTH * warp_smoothness_loss(warps))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if float(loss) < best_loss:
            best_loss  = float(loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if it % LOG_EVERY == 0 or it == n_iterations:
            with torch.no_grad():
                # Evaluate on real (un-augmented) data for interpretable logging
                al_real, w_real = model(chromas_t, study_ids)
                tic_r  = float(-neg_mean_tic_r(al_real, study_ids))
                id_t   = torch.arange(N_BINS, dtype=torch.float32, device=device)
                warp_a = float((w_real[:n_a] - id_t).abs().mean()) * BIN_MIN
                warp_b = float((w_real[n_a:] - id_t).abs().mean()) * BIN_MIN
            print(f"  iter {it:>5d}/{n_iterations}  loss(aug)={float(loss):.4f}"
                  f"  TIC_r(real)={tic_r:.4f}"
                  f"  warp {label_a}={warp_a:.3f}m  {label_b}={warp_b:.3f}m")

    ckpt_out = CKPT_DIR / 'warp_transformer_crossbatch.pt'
    torch.save(best_state, ckpt_out)
    print(f"\n  saved → {ckpt_out}")
    print(f"  Best fine-tune TIC r ≈ {-best_loss + LAMBDA_SMOOTH * 0:.4f}")
    print(f"  (baseline was {base_tic_r:.4f})")
    print(f"\nNow run:  python3 scripts/14_align_dataset.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-a', default='fish_oil/batch_jan2016',
                        help='Fine-tune study A (default: batch_jan2016)')
    parser.add_argument('--train-b', default='fish_oil/batch_apr2016',
                        help='Fine-tune study B (default: batch_apr2016)')
    args = parser.parse_args()

    label_a = Path(args.train_a).name
    label_b = Path(args.train_b).name
    finetune(DATA_DIR / args.train_a, DATA_DIR / args.train_b, label_a, label_b)
