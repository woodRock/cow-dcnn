"""
ChromaWarpTransformer — end-to-end dataset-level GC-MS alignment.

Why this architecture beats pairwise anchor-based alignment
-----------------------------------------------------------
Existing pipeline: detect peaks → find 4-5 RANSAC anchors → fit PCHIP warp.
Bottleneck: the PCHIP warp is over-parameterised for 4-5 control points across
200 bins.  Even perfect anchor embeddings can't escape this.

This model bypasses the bottleneck entirely:
  1. Projects each RT bin's m/z spectrum into a d_model token  [N, T, d]
  2. Local transformer: attention over T (within-sample RT context)
  3. Global attention:  attention over N (cross-sample at each RT bin)
  4. Dense warp head: predicts a monotone warp field [N, T] directly —
     no anchor detection, no RANSAC, no PCHIP

The global attention at step 3 is the core innovation: at each RT bin t,
every sample attends to all other samples at that bin, discovering where the
consensus peak position is and how far it must shift its own warp.

Input : chromas [N, T, M]   N samples, T=200 RT bins, M=1000 m/z channels
Output: aligned [N, T, M],  warps [N, T]  (monotone, range [0, T-1])
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Warp primitives ───────────────────────────────────────────────────────────

def apply_warp(chromas: Tensor, warps: Tensor) -> Tensor:
    """
    Differentiable warp via linear interpolation.

    For each output bin t, samples chromas at fractional source position warps[t].
    Gradient flows through the interpolation weights (alpha = frac part of warp),
    not through the discrete index selection — same trick as spatial transformer nets.

    chromas : [N, T, M]
    warps   : [N, T]   source positions in [0, T-1]  (need not be integers)
    returns : [N, T, M]
    """
    N, T, M = chromas.shape
    idx_lo  = warps.floor().long().clamp(0, T - 2)          # [N, T]
    idx_hi  = (idx_lo + 1).clamp(0, T - 1)
    alpha   = (warps - warps.floor()).unsqueeze(-1)          # [N, T, 1]  gradient flows here
    n_idx   = torch.arange(N, device=chromas.device).unsqueeze(1)
    lo      = chromas[n_idx, idx_lo, :]                     # [N, T, M]
    hi      = chromas[n_idx, idx_hi, :]
    return lo + alpha * (hi - lo)


def make_monotone_warp(raw: Tensor) -> Tensor:
    """
    Unconstrained [N, T] logits → monotone warp in [0, T-1].

    softplus  : ensures positive increments (never decreasing)
    cumsum    : integrates increments into a monotone sequence
    normalise : maps last value to T-1 so the warp spans the full RT range
    """
    T          = raw.shape[-1]
    deltas     = F.softplus(raw)
    cumulative = deltas.cumsum(dim=-1)
    return cumulative / cumulative[:, -1:].clamp(min=1e-6) * (T - 1)


def warp_smoothness_loss(warps: Tensor) -> Tensor:
    """
    Penalise non-smooth warp velocity.

    Identity warp has constant velocity=1.0, variance=0.  Penalising variance of
    inter-bin differences pushes the model toward smooth, physically realistic warps.
    """
    velocity = warps[:, 1:] - warps[:, :-1]   # [N, T-1]  ideal ≈ 1.0 everywhere
    return velocity.var(dim=-1).mean()


# ── Transformer building blocks ───────────────────────────────────────────────

class _LocalLayer(nn.Module):
    """Pre-LN transformer layer attending over the T (RT) axis within each sample."""

    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, T, d]
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class _GlobalLayer(nn.Module):
    """
    Cross-sample attention layer — the dataset-level reasoning step.

    At each RT bin t, each sample attends to ALL N samples at that bin.
    This lets each sample see where other samples have signal at the same
    nominal RT position and adjust its warp estimate accordingly.

    The key reshape: [N, T, d] → [T, N, d] → attention over N → [N, T, d].
    No positional encoding over N (sample order is arbitrary and must not matter).
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, T, d]
        N, T, d = x.shape
        x_t      = x.permute(1, 0, 2)                             # [T, N, d]
        n1       = self.norm1(x_t)
        h, _     = self.attn(n1, n1, n1)                          # attention over N
        x_t      = x_t + h
        x_t      = x_t + self.ff(self.norm2(x_t))
        return x_t.permute(1, 0, 2)                               # [N, T, d]


# ── Main model ────────────────────────────────────────────────────────────────

class ChromaWarpTransformer(nn.Module):
    """
    End-to-end dataset-level GC-MS chromatogram alignment.

    Architecture
    ------------
    proj        : Linear(M, d) + LayerNorm — project each RT bin's m/z spectrum to token
    RT PE       : sinusoidal positional encoding over T (shared across all N samples)
    study_embed : Embedding(2, d) — injected per-sample before global layers so the
                  cross-sample attention knows which batch each token belongs to.
                  Without this the attention treats biological variation as RT drift.
    n_layers ×  :
      _LocalLayer  — self-attention over T within each sample  [N, T, d] → [N, T, d]
      _GlobalLayer — self-attention over N at each RT bin      [N, T, d] → [N, T, d]
    warp_head   : LayerNorm → Linear(d, d//2) → GELU → Linear(d//2, 1) → [N, T]
    monotone    : softplus + cumsum + normalise → warp in [0, T-1]
    apply_warp  : differentiable linear interpolation

    study_ids ([N] int64, values 0/1) must be supplied at both train and inference time.
    N can be any value — trained with N∈{4…16}, works at N=38+.
    """

    def __init__(
        self,
        mz_bins:   int = 1000,
        n_bins:    int = 200,
        d_model:   int = 128,
        n_heads:   int = 4,
        n_layers:  int = 3,
        n_studies: int = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.n_bins  = n_bins
        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(mz_bins, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.register_buffer('pe', self._sinusoidal_pe(n_bins, d_model))  # [T, d]

        # Learnable study embedding — tells global attention which batch each sample is from
        self.study_embed = nn.Embedding(n_studies, d_model)

        self.local_layers  = nn.ModuleList(
            [_LocalLayer(d_model, n_heads, dropout)  for _ in range(n_layers)]
        )
        self.global_layers = nn.ModuleList(
            [_GlobalLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.warp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Zero-init final layer → initial raw_warp ≈ 0 → softplus(0) ≈ 0.693
        # → cumsum evenly spaced → normalise → identity warp at init
        nn.init.zeros_(self.warp_head[-1].weight)
        nn.init.zeros_(self.warp_head[-1].bias)

    @staticmethod
    def _sinusoidal_pe(n: int, d: int) -> Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode(self, chromas: Tensor, study_ids: Tensor) -> Tensor:
        """
        [N, T, M], [N] int64 → [N, T, d]

        study_ids: 0 = reference batch, 1 = query batch.
        The per-sample study embedding is added before global attention so the
        cross-sample attention can distinguish RT drift from biological variation.
        """
        x = self.proj(chromas) + self.pe.unsqueeze(0)              # [N, T, d]
        x = x + self.study_embed(study_ids).unsqueeze(1)           # [N, 1, d] → [N, T, d]
        for local, glob in zip(self.local_layers, self.global_layers):
            x = local(x)    # attend within each sample over T
            x = glob(x)     # attend across all N samples at each T
        return x

    def forward(self, chromas: Tensor, study_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        chromas   : [N, T, M]
        study_ids : [N] int64  — 0 for reference batch, 1 for query batch
        returns   : (aligned [N, T, M],  warps [N, T])

        warps are monotone, in [0, T-1].  Identity = [0, 1, …, T-1].
        """
        enc     = self.encode(chromas, study_ids)
        raw     = self.warp_head(enc).squeeze(-1)   # [N, T]
        warps   = make_monotone_warp(raw)           # [N, T]  monotone
        aligned = apply_warp(chromas, warps)        # [N, T, M]
        return aligned, warps
