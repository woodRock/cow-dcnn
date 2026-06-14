"""
Unsupervised diffeomorphic registration for GC-MS retention time alignment.

ChromaRegNet: lightweight 1-D UNet that predicts a smooth displacement field
d(t) from a paired (query TIC, reference TIC).  The warp is applied
differentiably to the full 2-D chromatogram (all m/z channels simultaneously).

Training objective: NCC(warped_TIC, ref_TIC) + λ·||∇d||²
No peak detection, no anchor matching, no hand-tuned thresholds.
Trains unsupervised on any set of GC-MS runs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn



# ── Differentiable warp ───────────────────────────────────────────────────────

def warp_chroma(chroma: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """
    Apply a 1-D displacement field to all m/z channels of a chromatogram.

    chroma : (B, N_MZ, N_BINS)  — m/z channels first
    disp   : (B, N_BINS)        — displacement in bins (positive = pull right)
    Returns warped chromatogram (B, N_MZ, N_BINS).

    Manual linear interpolation via gather — avoids grid_sample whose backward
    pass is not implemented on MPS (Apple Silicon).
    """
    B, C, L = chroma.shape
    base = torch.arange(L, device=disp.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    src  = (base + disp).clamp(0, L - 1)          # (B, L) source positions in bins

    src0  = src.floor().long().clamp(0, L - 1)    # lower neighbour index
    src1  = (src0 + 1).clamp(0, L - 1)            # upper neighbour index
    alpha = (src - src0.float()).unsqueeze(1)       # (B, 1, L) interpolation weight

    idx0 = src0.unsqueeze(1).expand(B, C, L)
    idx1 = src1.unsqueeze(1).expand(B, C, L)
    v0   = chroma.gather(2, idx0)                  # (B, C, L)
    v1   = chroma.gather(2, idx1)
    return v0 + alpha * (v1 - v0)


# ── Losses ────────────────────────────────────────────────────────────────────

def ncc_loss(pred: torch.Tensor, target: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """1 − NCC, averaged over the batch. pred/target: (B, L)."""
    pm = pred   - pred.mean(  dim=-1, keepdim=True)
    tm = target - target.mean(dim=-1, keepdim=True)
    num   = (pm * tm).sum(dim=-1)
    denom = (pm.pow(2).sum(dim=-1) * tm.pow(2).sum(dim=-1)).sqrt() + eps
    return (1.0 - num / denom).mean()


def smoothness_loss(disp: torch.Tensor) -> torch.Tensor:
    """L2 penalty on first-order differences — encourages smooth warps."""
    return (disp[:, 1:] - disp[:, :-1]).pow(2).mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChromaRegNet(nn.Module):
    """
    Lightweight 1-D UNet for unsupervised RT displacement-field estimation.

    Input : (B, 2, N_BINS) — [query_tic, ref_tic], both L2-normalised
    Output: (B, N_BINS)    — smooth displacement field in RT bins

    Training loss: ncc_loss(warp(query_tic, disp), ref_tic)
                 + λ · smoothness_loss(disp)

    Inference: apply disp to the full 2-D chromatogram via warp_chroma.

    Architecture: 3-level UNet with skip connections (InstanceNorm for
    batch-size-1 inference stability).  Head zero-initialised → identity
    warp at the start of training.

    Default base_ch=16 → ~35 K parameters.
    N_BINS must be divisible by 8 (default 200 = 8 × 25 ✓).
    """

    def __init__(self, n_bins: int = 200, base_ch: int = 16):
        super().__init__()
        C = base_ch
        self.pool = nn.MaxPool1d(2)

        self.enc1 = _ConvBlock(2,   C)
        self.enc2 = _ConvBlock(C,   C * 2)
        self.enc3 = _ConvBlock(C*2, C * 4)
        self.bot  = _ConvBlock(C*4, C * 4)

        self.up3  = nn.ConvTranspose1d(C*4, C*4, 2, stride=2)
        self.dec3 = _ConvBlock(C*8, C*2)
        self.up2  = nn.ConvTranspose1d(C*2, C*2, 2, stride=2)
        self.dec2 = _ConvBlock(C*4, C)
        self.up1  = nn.ConvTranspose1d(C,   C,   2, stride=2)
        self.dec1 = _ConvBlock(C*2, C)

        self.head = nn.Conv1d(C, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, N_BINS) → disp: (B, N_BINS)"""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot( self.pool(e3))
        d  = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d  = self.dec2(torch.cat([self.up2(d), e2], dim=1))
        d  = self.dec1(torch.cat([self.up1(d), e1], dim=1))
        return self.head(d).squeeze(1)


# ── Numpy inference wrapper ───────────────────────────────────────────────────

def align_regnet(query: np.ndarray, ref: np.ndarray,
                 model: ChromaRegNet,
                 device: torch.device = torch.device('cpu'),
                 max_disp: int = 18) -> np.ndarray:
    """
    Align a query chromatogram onto the reference using a trained ChromaRegNet.

    query, ref : (N_BINS, N_MZ) float32 numpy arrays
    Returns warped query (N_BINS, N_MZ) float32.

    Computes TICs, normalises, runs the network to get d(t), then applies
    the displacement to all 1000 m/z channels simultaneously via warp_chroma.
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    q_tic = _norm(query.sum(axis=1).astype(np.float32))
    r_tic = _norm(ref.sum(  axis=1).astype(np.float32))

    x = torch.from_numpy(np.stack([q_tic, r_tic])).unsqueeze(0).to(device)
    with torch.no_grad():
        disp = model(x).clamp(-max_disp, max_disp)   # (1, N_BINS)

    q_t    = torch.from_numpy(query.T.astype(np.float32)).unsqueeze(0).to(device)
    warped = warp_chroma(q_t, disp).squeeze(0).cpu().numpy().T
    return warped.astype(np.float32)
