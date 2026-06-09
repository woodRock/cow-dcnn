"""
MLP encoder for GC-MS m/z spectra, trained with SimCLR.
The 128-dim embeddings replace raw cosine similarity in the alignment pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    def __init__(self, in_dim: int = 1000, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
        # Projection head used during SimCLR training only
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding. Shape: (B, emb_dim)."""
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected embedding for SimCLR loss. Shape: (B, 64)."""
        return F.normalize(self.proj(self.encode(x)), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent contrastive loss (SimCLR).
    z1, z2: L2-normalised projected embeddings, shape (B, D).
    Positives: (z1[i], z2[i]). Negatives: all other pairs in the batch.
    """
    B = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)               # (2B, D)
    sim = torch.mm(z, z.t()) / temperature         # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive indices: i pairs with i+B, i+B pairs with i
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)
