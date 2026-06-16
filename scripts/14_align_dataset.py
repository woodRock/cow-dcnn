"""
14_align_dataset.py — Dataset-level alignment via ChromaWarpTransformer.

Loads all study-A and study-B chromatograms, runs the entire set through
ChromaWarpTransformer in a single forward pass, and evaluates alignment quality.

Metrics (same as scripts 07 and 10 for direct comparison):
  TIC r      : mean Pearson correlation of TIC across all A×B pairs
  Study sil  : study-label silhouette in PCA-50 (lower = less batch effect)
  Warp mag   : mean |warp(t) − t| in minutes (how aggressively the model corrects)

Baselines compared:
  Unaligned       — raw chromatograms, no correction
  Raw cosine PCHIP — pairwise anchor-based alignment (same as script 07)
  WarpTransformer  — joint dataset-level alignment (this script)

The key difference from pairwise alignment: the transformer sees all N samples
simultaneously at inference, so each sample's warp is informed by the whole
dataset rather than a single reference sample.
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import ChromaWarpTransformer
from alignment import load_chromas, align_pair, warp_chroma_2d

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_BINS  = 200
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS


# ── Metric helpers ────────────────────────────────────────────────────────────

def _tic(c: np.ndarray) -> np.ndarray:
    return c.sum(axis=1)


def _max_proj(c: np.ndarray) -> np.ndarray:
    return c.max(axis=0)


def tic_correlations(chromas_a: list, chromas_b: list) -> float:
    """Mean Pearson TIC r across all |A|×|B| pairs."""
    corrs = [float(pearsonr(_tic(a), _tic(b))[0])
             for a in chromas_a for b in chromas_b]
    return float(np.mean(corrs))


def study_silhouette(chromas_a: list, chromas_b: list) -> float:
    """Study-label silhouette from PCA-50 of per-m/z max projections."""
    feats  = np.stack([_max_proj(c) for c in chromas_a + chromas_b])
    labels = np.array([0] * len(chromas_a) + [1] * len(chromas_b))
    n_comp = min(50, feats.shape[0] - 1)
    pca    = PCA(n_components=n_comp).fit_transform(feats)
    return float(silhouette_score(pca, labels))


# ── Alignment methods ─────────────────────────────────────────────────────────

def align_pairwise_cosine(chromas_a: list, chromas_b: list) -> list[np.ndarray]:
    """
    Pairwise PCHIP alignment: align each study-A chromatogram to study-B[0].
    (Same reference strategy as scripts 07–11.)
    """
    ref     = chromas_b[0]
    aligned = []
    for chroma in chromas_a:
        _, _, _, _, warp_fn = align_pair(chroma, ref, encode_fn=None, return_anchors=True)
        aligned.append(warp_chroma_2d(chroma, warp_fn) if warp_fn is not None else chroma.copy())
    return aligned


def align_with_transformer(all_chromas: list, model: ChromaWarpTransformer,
                            device: torch.device,
                            study_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint dataset alignment: process all N chromatograms in one forward pass.

    study_ids [N] int64 — 0 for reference batch, 1 for query batch.
    Conditioning on study membership lets the global attention distinguish
    RT drift from biological variation across samples.

    Returns:
        aligned_np : [N, T, M] — aligned chromatograms
        warps_np   : [N, T]    — monotone warp fields in [0, T-1]
    """
    stacked = np.stack(all_chromas, axis=0)          # [N, T, M]
    tensor  = torch.from_numpy(stacked).to(device)

    with torch.no_grad():
        aligned_t, warps_t = model(tensor, study_ids)

    return aligned_t.cpu().numpy(), warps_t.cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(study_a_dir: Path, study_b_dir: Path, label_a: str, label_b: str):
    print("Loading chromatograms …")
    chromas_a   = load_chromas(study_a_dir / 'chroma')
    chromas_b   = load_chromas(study_b_dir / 'chroma')
    n_a, n_b    = len(chromas_a), len(chromas_b)
    all_chromas = chromas_a + chromas_b
    print(f"  {label_a}: {n_a} samples")
    print(f"  {label_b}: {n_b} samples")
    print(f"  Total for joint alignment: {n_a + n_b}\n")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ckpt = CKPT_DIR / 'warp_transformer.pt'
    model = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)
    print(f"  Loaded ChromaWarpTransformer from {ckpt.name}\n")

    # ── Unaligned baseline ────────────────────────────────────────────────────
    tic_unaligned = tic_correlations(chromas_a, chromas_b)
    sil_unaligned = study_silhouette(chromas_a, chromas_b)
    print(f"Unaligned         TIC r={tic_unaligned:.3f}  study_sil={sil_unaligned:.3f}")

    # ── Raw cosine PCHIP (pairwise baseline) ──────────────────────────────────
    print("Aligning with raw cosine (pairwise) …")
    aligned_a_cos  = align_pairwise_cosine(chromas_a, chromas_b)
    tic_cos        = tic_correlations(aligned_a_cos, chromas_b)
    sil_cos        = study_silhouette(aligned_a_cos, chromas_b)
    print(f"Raw cosine PCHIP  TIC r={tic_cos:.3f}  study_sil={sil_cos:.3f}")

    # ── ChromaWarpTransformer (joint) ─────────────────────────────────────────
    # study 0 = sep2015 (reference), study 1 = jul2016 (query)
    study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long, device=device)
    print(f"Aligning {n_a + n_b} samples jointly with ChromaWarpTransformer …")
    aligned_np, warps_np = align_with_transformer(all_chromas, model, device, study_ids)
    aligned_a_wt = [aligned_np[i] for i in range(n_a)]
    aligned_b_wt = [aligned_np[i] for i in range(n_a, n_a + n_b)]

    tic_wt  = tic_correlations(aligned_a_wt, aligned_b_wt)
    sil_wt  = study_silhouette(aligned_a_wt, aligned_b_wt)
    # Warp magnitude: mean |warp(t) - t| in minutes
    identity    = np.arange(N_BINS, dtype=np.float32)[None, :]     # [1, T]
    warp_mag_bins = float(np.abs(warps_np - identity).mean())
    warp_mag_min  = warp_mag_bins * BIN_MIN
    # Per-study breakdown
    warp_a = float(np.abs(warps_np[:n_a] - identity).mean()) * BIN_MIN
    warp_b = float(np.abs(warps_np[n_a:] - identity).mean()) * BIN_MIN
    print(f"WarpTransformer   TIC r={tic_wt:.3f}  study_sil={sil_wt:.3f}  warp_mag={warp_mag_min:.3f}m")

    # ── Results table ─────────────────────────────────────────────────────────
    w = 72
    lines = [
        f"\n{'=' * w}",
        f"{'Method':<28} {'TIC r':>7}  {'Δ unaligned':>12}  {'Study sil':>10}",
        f"{'-' * w}",
        f"{'Unaligned':<28} {tic_unaligned:>7.3f}  {'—':>12}  {sil_unaligned:>10.3f}",
        f"{'Raw cosine PCHIP (pairwise)':<28} {tic_cos:>7.3f}  "
        f"{tic_cos - tic_unaligned:>+12.3f}  {sil_cos:>10.3f}",
        f"{'WarpTransformer (joint)':<28} {tic_wt:>7.3f}  "
        f"{tic_wt - tic_unaligned:>+12.3f}  {sil_wt:>10.3f}",
        f"{'=' * w}",
        f"TIC r     : mean Pearson r of TIC across all {label_a}×{label_b} pairs",
        f"Study sil : study-label silhouette in PCA-50  (lower = less batch effect)",
        f"",
        f"WarpTransformer warp magnitudes (mean |warp(t)−t| in minutes):",
        f"  All samples : {warp_mag_min:.3f} min",
        f"  {label_a:<20}: {warp_a:.3f} min",
        f"  {label_b:<20}: {warp_b:.3f} min",
        f"",
        f"Note: WarpTransformer aligns all {n_a + n_b} samples jointly to an implicit",
        f"consensus RT reference — no explicit reference sample required.",
    ]
    report = '\n'.join(lines)
    print(report)

    out = RESULTS_DIR / f'warp_transformer_{label_a}_{label_b}.txt'
    out.write_text(report + '\n')
    print(f"\n  results → {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dataset-level chromatogram alignment with ChromaWarpTransformer'
    )
    parser.add_argument('--study-a', default='fish_oil/batch_sep2015',
                        help='Path under data/ for study A')
    parser.add_argument('--study-b', default='fish_oil/batch_jul2016',
                        help='Path under data/ for study B')
    args = parser.parse_args()

    label_a = Path(args.study_a).name
    label_b = Path(args.study_b).name
    main(
        DATA_DIR / args.study_a,
        DATA_DIR / args.study_b,
        label_a, label_b,
    )
