"""
Encoders for GC-MS m/z spectra, trained with SimCLR.

SpectrumEncoder          — 3-layer MLP (kept for gcms_simclr.pt / mona_simclr.pt compat)
DilatedSpectrumEncoder   — 1-D dilated-CNN over the m/z axis (drift_simclr.pt)
ChromaSpectrumEncoder    — 2-D encoder: dilated-CNN over m/z × dilated-CNN + dual-pool
                           over RT (local ±7-bin window, ChromaDCNN-inspired)
PeakSequenceTransformer  — Sequence encoder: dilated-CNN per peak + sinusoidal RT
                           positional encoding + Transformer self-attention over all N
                           detected peaks (cross_study_simclr.pt).  Captures the global
                           monotonic elution order constraint: compound A always elutes
                           before compound B even when absolute RTs drift across studies.

WINDOW_HALF : RT half-width for ChromaSpectrumEncoder peak windows (bins).
"""

WINDOW_HALF = 7   # ±7 bins → 15-bin window ≈ 3.4 min; covers GC-MS peak at 3σ

import math
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
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.encode(x)), dim=-1)


class _DilatedResBlock(nn.Module):
    """Two dilated conv layers with a residual skip and GELU activation."""
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # 'same' padding for odd kernels
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DilatedSpectrumEncoder(nn.Module):
    """
    1-D dilated CNN encoder for GC-MS m/z spectra.

    Treats the 1000-dim L2-normalised spectrum as a 1D signal over the m/z
    axis.  Dilated residual blocks capture structure at multiple scales:
      dilation 1-2   → isotope clusters (2-6 Da)
      dilation 4-8   → neutral losses (CO=28, H2O=18, common fragments)
      dilation 16-32 → compound-class signatures spanning 50-100+ Da

    The effective receptive field after 6 blocks (dilations 1→32, k=3) is
    2 × (1+2+4+8+16+32) × (3-1) = 252 m/z units per layer pair, or ~378 Da
    total — enough to see the full fragment fingerprint of most GC-MS compounds.

    RT conditioning: optional normalised retention time (0–1) is injected as
    an additive bias to the global-average-pooled feature map before the head,
    so the embedding encodes both spectral identity and chromatographic position.

    Parameter count: ~636 K
    """
    def __init__(self, in_dim: int = 1000, emb_dim: int = 128,
                 channels: int = 64, dropout: float = 0.1):
        super().__init__()
        # Stem: pointwise projection into channel space
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        # Six dilated residual blocks; dilation doubles each block
        self.blocks = nn.Sequential(
            _DilatedResBlock(channels, dilation=1),
            _DilatedResBlock(channels, dilation=2),
            _DilatedResBlock(channels, dilation=4),
            _DilatedResBlock(channels, dilation=8),
            _DilatedResBlock(channels, dilation=16),
            _DilatedResBlock(channels, dilation=32),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, emb_dim)
        # RT conditioning: scalar position → additive feature bias
        self.rt_proj = nn.Linear(1, channels)
        # SimCLR projection head (discarded at inference)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def encode(self, x: torch.Tensor,
               rt: torch.Tensor = None) -> torch.Tensor:
        """Return L2-normalised embedding. Shape: (B, emb_dim).

        rt: optional (B,) tensor of normalised retention time in [0, 1].
            When provided, the embedding encodes both spectral identity and
            chromatographic position — peaks at very different RT positions
            will be further apart in embedding space even with similar spectra.
        """
        h = x.unsqueeze(1)        # (B, 1, L)
        h = self.stem(h)          # (B, C, L)
        h = self.blocks(h)        # (B, C, L)
        h = h.mean(dim=-1)        # (B, C) — global average pool
        if rt is not None:
            h = h + self.rt_proj(rt.view(-1, 1).float())   # (B, C)
        h = self.drop(h)
        return F.normalize(self.head(h), dim=-1)

    def forward(self, x: torch.Tensor,
                rt: torch.Tensor = None) -> torch.Tensor:
        """Return projected embedding for SimCLR loss. Shape: (B, 64)."""
        return F.normalize(self.proj(self.encode(x, rt)), dim=-1)


class SparseSpectrumTransformer(nn.Module):
    """
    Sparse transformer encoder for GC-MS EI mass spectra.

    Rather than treating the 1000-dim spectrum as a dense 1D signal, extracts
    the top max_peaks m/z bins as tokens.  Each token is the learned m/z
    positional embedding scaled by the observed intensity, so the sequence
    length is ~50–200 tokens instead of 1000 — making full self-attention
    tractable and naturally sparse.

    Self-attention captures non-local fragment relationships that a dilated CNN
    can only approximate with a finite receptive field:
      • molecular ion ↔ neutral-loss fragments (M-15, M-18, M-28, M-CO)
      • compound-class signatures spanning the full m/z range
        (e.g. FAME series: 74, 87, M-31, M-59; steroids: 55, 69, 83 …)

    Architecture: Pre-LN transformer (more stable than Post-LN on small data)
    Defaults: d_model=64, nhead=4, n_layers=3, dim_ff=128  → ~198 K params.
    Same interface as DilatedSpectrumEncoder:
      encode(x, rt=None)  → (B, emb_dim)  L2-normed
      forward(x, rt=None) → (B, 64)       SimCLR projection head
    """

    def __init__(self, mz_vocab: int = 1000, d_model: int = 64,
                 nhead: int = 4, n_layers: int = 3, dim_ff: int = 128,
                 dropout: float = 0.1, max_peaks: int = 128,
                 emb_dim: int = 128):
        super().__init__()
        self.max_peaks = max_peaks

        # Learned m/z positional embeddings — one per m/z bin 0 … 999
        self.mz_embed = nn.Embedding(mz_vocab, d_model)

        # [CLS] token — pooled to produce the spectrum-level embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Pre-LN transformer encoder (norm_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )

        # Optional RT conditioning — additive bias on CLS output
        self.rt_proj = nn.Linear(1, d_model)

        # Output head: d_model → emb_dim
        self.head = nn.Linear(d_model, emb_dim)

        # SimCLR projection head (discarded at inference)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def _tokenise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dense (B, 1000) spectra to sparse (B, max_peaks, d_model) tokens.

        Top-max_peaks bins are selected by intensity; each token is the learned
        m/z embedding scaled by its intensity (low-intensity fragments contribute
        less; near-zero padding bins are zeroed and masked).

        Returns:
          tokens:   (B, max_peaks, d_model)
          key_mask: (B, max_peaks) bool — True = padding, excluded from attention
        """
        vals, idxs = x.topk(self.max_peaks, dim=-1, sorted=False)  # (B, P)
        key_mask   = vals < 1e-6                                     # near-zero → pad
        tokens     = self.mz_embed(idxs) * vals.unsqueeze(-1)       # (B, P, d_model)
        tokens     = tokens.masked_fill(key_mask.unsqueeze(-1), 0.0)
        return tokens, key_mask

    def encode(self, x: torch.Tensor,
               rt: torch.Tensor = None) -> torch.Tensor:
        """Return L2-normalised (B, emb_dim) embedding."""
        B = x.size(0)
        tokens, key_mask = self._tokenise(x)            # (B, P, d), (B, P)

        # Prepend [CLS]; its mask entry is always False (never padding)
        cls     = self.cls_token.expand(B, -1, -1)      # (B, 1, d)
        seq     = torch.cat([cls, tokens], dim=1)        # (B, 1+P, d)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        mask    = torch.cat([cls_pad, key_mask], dim=1)  # (B, 1+P)

        out = self.transformer(seq, src_key_padding_mask=mask)  # (B, 1+P, d)
        h   = out[:, 0]                                          # [CLS] → (B, d)

        if rt is not None:
            h = h + self.rt_proj(rt.view(-1, 1).float())

        return F.normalize(self.head(h), dim=-1)

    def forward(self, x: torch.Tensor,
                rt: torch.Tensor = None) -> torch.Tensor:
        """Return projected embedding for SimCLR loss. Shape: (B, 64)."""
        return F.normalize(self.proj(self.encode(x, rt)), dim=-1)


class _AttentionPool1d(nn.Module):
    """Soft attention pooling over the last dimension.  x: (B, C, N) → (B, C)"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)                    # (B, N, C)
        w   = F.softmax(self.score(x_t), dim=1)    # (B, N, 1)
        return (x_t * w).sum(dim=1)                # (B, C)


class ChromaSpectrumEncoder(nn.Module):
    """
    2-D peak encoder: combines the m/z and RT axes via dual pooling.

    Mirrors ChromatogramCNN's architecture (spec_proj → dilated ResBlocks over RT →
    max-pool ∥ attention-pool) but applied to a local RT×m/z window around each
    detected peak rather than the full chromatogram.

    Input : (B, W, 1000)  where W = 2·WINDOW_HALF + 1 RT bins, 1000 m/z bins.

    Step 1 — m/z encoding (shared DilatedCNN applied to each RT slice independently):
              (B, W, 1000) → (B, W, channels)
    Step 2 — RT encoding (dilated ResBlocks over the W-bin window):
              (B, channels, W) → (B, channels, W)
    Step 3 — Dual pooling over RT (chromaDCNN-style):
              max-pool       → (B, channels)   most-salient RT bin
              attention-pool → (B, channels)   soft evidence across all RT bins
    Step 4 — Concat → (B, 2·channels) → head → (B, emb_dim), L2-normalised
    """

    def __init__(self, in_dim: int = 1000, emb_dim: int = 128,
                 channels: int = 64, dropout: float = 0.1):
        super().__init__()

        # m/z encoder — same dilated-CNN as DilatedSpectrumEncoder, weights shared
        # across all RT slices in the window.
        self.mz_stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.mz_blocks = nn.Sequential(
            _DilatedResBlock(channels, dilation=1),
            _DilatedResBlock(channels, dilation=2),
            _DilatedResBlock(channels, dilation=4),
            _DilatedResBlock(channels, dilation=8),
            _DilatedResBlock(channels, dilation=16),
            _DilatedResBlock(channels, dilation=32),
        )

        # RT encoder — two dilated ResBlocks over the W-bin window axis.
        # W=15 is small so dilations 1 and 2 cover the full window.
        self.rt_blocks = nn.Sequential(
            _DilatedResBlock(channels, dilation=1),
            _DilatedResBlock(channels, dilation=2),
        )

        self.attn_pool = _AttentionPool1d(channels)
        self.drop      = nn.Dropout(dropout)
        self.head      = nn.Linear(2 * channels, emb_dim)
        self.proj      = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def _encode_mz(self, x: torch.Tensor) -> torch.Tensor:
        """Apply shared m/z CNN to every RT slice.  (B, W, 1000) → (B, W, channels)"""
        B, W, L = x.shape
        h = x.reshape(B * W, L).unsqueeze(1)   # (B·W, 1, 1000)
        h = self.mz_stem(h)                     # (B·W, C, 1000)
        h = self.mz_blocks(h)                   # (B·W, C, 1000)
        h = h.mean(dim=-1)                      # (B·W, C) — global avg-pool over m/z
        return h.reshape(B, W, -1)              # (B, W, C)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, W, 1000) → (B, emb_dim) L2-normalised embedding."""
        h = self._encode_mz(x)                  # (B, W, C)
        h = h.transpose(1, 2)                   # (B, C, W) — channels-first for Conv1d
        h = self.rt_blocks(h)                   # (B, C, W)
        x_max  = h.max(dim=-1).values           # (B, C)
        x_attn = self.attn_pool(h)              # (B, C)
        out    = torch.cat([x_max, x_attn], dim=-1)  # (B, 2C)
        out    = self.drop(out)
        return F.normalize(self.head(out), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, W, 1000) → (B, 64) SimCLR projected head."""
        return F.normalize(self.proj(self.encode(x)), dim=-1)


class PeakSequenceTransformer(nn.Module):
    """
    Sequence-level GC-MS peak encoder using self-attention across all chromatogram peaks.

    Unlike per-peak encoders, this model attends across the entire detected peak
    sequence so each embedding reflects both the compound's m/z identity and its
    position in the global elution order — a constraint preserved across studies even
    when absolute RTs drift by several minutes.

    Input (encode):
        fps : (B, N, 1000)  L2-normalised m/z fingerprints for N peaks (ascending RT)
        rts : (B, N)        normalised retention times in [0, 1]

    Step 1 — m/z encoding: shared dilated CNN applied per peak → (B, N, channels)
    Step 2 — RT positional encoding: sinusoidal at continuous RT value, added to tokens
    Step 3 — Transformer encoder: self-attention over all N peaks
    Step 4 — Head: Linear → (B, N, emb_dim), L2-normalised

    project(z) applies the SimCLR projection head to a subset of per-peak embeddings
    (shared peaks only) during training.
    """

    def __init__(self, in_dim: int = 1000, emb_dim: int = 128,
                 channels: int = 64, d_model: int = 64,
                 nhead: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # m/z encoder: same dilated CNN as DilatedSpectrumEncoder, weights shared per peak
        self.mz_stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.mz_blocks = nn.Sequential(
            _DilatedResBlock(channels, dilation=1),
            _DilatedResBlock(channels, dilation=2),
            _DilatedResBlock(channels, dilation=4),
            _DilatedResBlock(channels, dilation=8),
            _DilatedResBlock(channels, dilation=16),
            _DilatedResBlock(channels, dilation=32),
        )

        # Sinusoidal RT positional encoding (continuous, not discrete bin index)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer('_div_term', div)   # (d_model//2,)

        self.mz_to_d = nn.Identity() if channels == d_model else nn.Linear(channels, d_model)

        # Pre-LN Transformer over the N-peak sequence (batch_first for clean (B,N,d) layout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, emb_dim)

        # SimCLR projection head — only applied to shared-peak embeddings during training
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 64),
        )

    def _rt_pe(self, rts: torch.Tensor) -> torch.Tensor:
        """Sinusoidal RT positional encoding. rts: (B, N) → (B, N, d_model)."""
        rts = rts.unsqueeze(-1)                            # (B, N, 1)
        return torch.cat([
            torch.sin(rts * self._div_term),               # (B, N, d//2)
            torch.cos(rts * self._div_term),               # (B, N, d//2)
        ], dim=-1)

    def _encode_mz(self, fps: torch.Tensor) -> torch.Tensor:
        """Shared dilated CNN applied per peak. fps: (B, N, 1000) → (B, N, channels)."""
        B, N, L = fps.shape
        h = fps.reshape(B * N, L).unsqueeze(1)   # (B·N, 1, 1000)
        h = self.mz_stem(h)                       # (B·N, C, 1000)
        h = self.mz_blocks(h)                     # (B·N, C, 1000)
        h = h.mean(dim=-1)                        # (B·N, C) — global avg pool over m/z
        return h.reshape(B, N, -1)                # (B, N, C)

    def encode(self, fps: torch.Tensor, rts: torch.Tensor) -> torch.Tensor:
        """
        fps : (B, N, 1000) — L2-normalised m/z fingerprints (sorted by ascending RT)
        rts : (B, N)       — normalised RT positions in [0, 1]
        → (B, N, emb_dim) L2-normalised per-peak embeddings, globally context-aware
        """
        h = self._encode_mz(fps)                # (B, N, C)
        h = self.mz_to_d(h)                     # (B, N, d_model)
        h = h + self._rt_pe(rts)               # add RT positional encoding
        h = self.transformer(h)                 # (B, N, d_model) — global self-attention
        h = self.drop(h)
        return F.normalize(self.head(h), dim=-1)  # (B, N, emb_dim)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """SimCLR projection head for training.
        z: (B, emb_dim) shared-peak embeddings → (B, 64) L2-normalised."""
        return F.normalize(self.proj(z), dim=-1)

    def forward(self, fps: torch.Tensor, rts: torch.Tensor) -> torch.Tensor:
        """(B, N, 1000), (B, N) → (B, N, 64) SimCLR projections."""
        emb = self.encode(fps, rts)         # (B, N, emb_dim)
        B, N, D = emb.shape
        return self.project(emb.reshape(B * N, D)).reshape(B, N, -1)


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
