"""
Batch effect assessment in feature space.

After applying each alignment method, this script measures whether wheat
(MTBLS21) and rice (MTBLS288) chromatograms are distinguishable in feature
space — i.e., how much of a dataset-level batch effect remains.

Protocol
--------
Reference: rice sample[0].  Every wheat sample is aligned to this reference.
Rice samples are aligned to rice[0] with the same method and parameters for a
fair comparison (within-study drift is smaller, so these warps are minor).

Features: per-m/z maximum projection — the standard metabolomics feature
vector that captures which compounds are present regardless of abundance.
Shape: (N_samples=119, 1000).  PCA → 50 components before k-NN.

Metrics:
  k-NN mixing  — for each sample, fraction of its k=10 nearest neighbours
                 from the OPPOSITE study.  Random expectation ≈ 0.34 (wheat
                 makes up 40/119 of the pool).  Higher = better mixing.

  Study accuracy — k-NN (k=5) leave-one-out accuracy for the binary study
                   label (wheat=0, rice=1).  Lower = less separable = less
                   batch effect.

  Silhouette score (study label, lower is better)
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from alignment import (align_pair, warp_chroma_2d, load_chromas,
                       load_encoder, max_projection)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

K_MIX   = 10
K_CLASS  = 5
N_PCA    = 50


class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


def align_all(chromas: list[np.ndarray], ref: np.ndarray,
              encode_fn=None) -> list[np.ndarray]:
    """Align every sample in chromas to ref; return list of aligned 2D chromatograms."""
    aligned = []
    for c in chromas:
        _, _, _, _, warp_fn = align_pair(c, ref, encode_fn=encode_fn,
                                         return_anchors=True)
        aligned.append(warp_chroma_2d(c, warp_fn))
    return aligned


def feature_matrix(chromas: list[np.ndarray]) -> np.ndarray:
    """Stack max-projection features into (N, 1000)."""
    return np.stack([max_projection(c) for c in chromas], axis=0)


def batch_metrics(X: np.ndarray, study_labels: np.ndarray) -> dict:
    """
    X: (N, D) feature matrix.  study_labels: 0=wheat, 1=rice.
    Returns k-NN mixing, study classification accuracy, and silhouette.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    n_components = min(N_PCA, X_sc.shape[0] - 1, X_sc.shape[1])
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    N = len(study_labels)

    # k-NN mixing: fraction of k nearest neighbours from the other study
    nbrs = NearestNeighbors(n_neighbors=K_MIX + 1).fit(X_pca)
    dists, idxs = nbrs.kneighbors(X_pca)

    mixing_ratios = []
    for i in range(N):
        neighbours = idxs[i, 1:]  # exclude self
        other_study = np.sum(study_labels[neighbours] != study_labels[i])
        mixing_ratios.append(other_study / K_MIX)
    mixing = float(np.mean(mixing_ratios))

    # k-NN leave-one-out study accuracy
    nbrs2 = NearestNeighbors(n_neighbors=K_CLASS + 1).fit(X_pca)
    _, idxs2 = nbrs2.kneighbors(X_pca)
    correct = 0
    for i in range(N):
        neighbours = idxs2[i, 1:]
        pred = int(np.bincount(study_labels[neighbours]).argmax())
        if pred == study_labels[i]:
            correct += 1
    study_acc = correct / N

    sil = float(silhouette_score(X_pca, study_labels))

    return {'mixing': mixing, 'study_acc': study_acc, 'silhouette': sil,
            'X_pca': X_pca, 'study_labels': study_labels}


def main(study_a_dir: Path, study_b_dir: Path, label_a: str, label_b: str) -> None:
    print("Loading chromatograms …")
    m21  = load_chromas(study_a_dir / 'chroma')
    m288 = load_chromas(study_b_dir / 'chroma')
    print(f"  {label_a}: {len(m21)} samples")
    print(f"  {label_b}: {len(m288)} samples")

    ref = m288[0]
    n_wheat = len(m21)
    n_rice  = len(m288)
    study_labels = np.array([0] * n_wheat + [1] * n_rice, dtype=int)

    random_mixing = n_rice / (n_wheat + n_rice)
    print(f"\nReference: {label_b} sample[0]")
    print(f"k-NN mixing  random expectation: {random_mixing:.3f}  "
          f"(higher = better cross-study mixing)")
    print(f"Study accuracy: lower is better (less batch effect)\n")

    methods: list[tuple[str, object]] = [('Unaligned', None), ('Raw cosine', None)]
    for label, fname in [('Drift encoder',       'drift_simclr.pt'),
                         ('Cross-study encoder', 'cross_study_simclr.pt')]:
        p = CKPT_DIR / fname
        if p.exists():
            methods.append((label, load_encoder(p)))
            print(f"Loaded  {label}  from {fname}")
        else:
            print(f"SKIP    {label}: {fname} not found")
    print()

    results = {}
    for i, (label, enc_fn) in enumerate(methods):
        print(f"Evaluating: {label} …", end=' ', flush=True)
        if label == 'Unaligned':
            aligned_wheat = m21
            aligned_rice  = m288
        else:
            aligned_wheat = align_all(m21,  ref, encode_fn=enc_fn)
            aligned_rice  = align_all(m288, ref, encode_fn=enc_fn)

        X = feature_matrix(aligned_wheat + aligned_rice)
        r = batch_metrics(X, study_labels)
        results[label] = r
        print(f"mixing={r['mixing']:.3f}  "
              f"study_acc={r['study_acc']:.3f}  "
              f"silhouette={r['silhouette']:.3f}")

    w = 26
    print(f"\n{'='*70}")
    print(f"{'Method':<{w}}  {'k-NN mixing':>12}  {'Study acc':>10}  "
          f"{'Silhouette':>11}")
    print(f"{'-'*70}")
    for lbl, r in results.items():
        print(f"{lbl:<{w}}  {r['mixing']:>12.3f}  "
              f"{r['study_acc']:>10.3f}  {r['silhouette']:>11.3f}")
    print(f"{'='*70}")
    print(f"k-NN mixing  : fraction of k={K_MIX} neighbours from opposite study  "
          f"(random ≈ {random_mixing:.2f})")
    print(f"Study acc    : k={K_CLASS} LOO accuracy for study label  (lower = less batch effect)")
    print(f"Silhouette   : study-label silhouette in PCA-{N_PCA} space  (lower = less batch effect)")
    print(f"Features     : per-m/z max projection → PCA-{N_PCA}")
    print(f"Studies      : {label_a} (study 0) vs {label_b} (study 1)")
    return results


def save_figure(results: dict, figs_dir: Path, label_a: str, label_b: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods     = list(results.keys())
    n_methods   = len(methods)
    study_colors = {0: '#E69F00', 1: '#56B4E9'}   # wheat=orange, rice=blue

    ncols = min(n_methods, 2)
    nrows = (n_methods + ncols - 1) // ncols
    fig_scatter, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, method in enumerate(methods):
        ax   = axes[i]
        Xp   = results[method]['X_pca']
        labs = results[method]['study_labels']
        for study_id, label, marker in [(0, 'Wheat (MTBLS21)', 'o'),
                                         (1, 'Rice (MTBLS288)', 's')]:
            mask = labs == study_id
            ax.scatter(Xp[mask, 0], Xp[mask, 1],
                       c=study_colors[study_id], label=label,
                       s=18, alpha=0.7, marker=marker, linewidths=0)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel('PC 1', fontsize=8)
        ax.set_ylabel('PC 2', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, markerscale=1.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig_scatter.suptitle(f'Batch Effect — PCA of max-projection features\n'
                         f'{label_a} × {label_b}', y=1.01)
    fig_scatter.tight_layout()
    out_pca = figs_dir / 'batch_effect_pca.pdf'
    fig_scatter.savefig(out_pca, bbox_inches='tight')
    plt.close(fig_scatter)
    print(f"Figure saved → {out_pca}")

    # Summary bar chart
    metrics = ['mixing', 'study_acc', 'silhouette']
    labels  = ['k-NN mixing ↑', 'Study acc ↓', 'Silhouette ↓']
    x = np.arange(len(methods))
    fig_bar, ax = plt.subplots(figsize=(7, 4))
    bar_w = 0.25
    for j, (metric, lbl) in enumerate(zip(metrics, labels)):
        vals = [results[m][metric] for m in methods]
        ax.bar(x + (j - 1) * bar_w, vals, bar_w, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Batch Effect Summary')
    ax.legend(fontsize=9)
    fig_bar.tight_layout()
    out_bar = figs_dir / 'batch_effect_summary.pdf'
    fig_bar.savefig(out_bar)
    plt.close(fig_bar)
    print(f"Figure saved → {out_bar}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015',
                    help='Study A directory relative to data/ (default: fish_oil/batch_sep2015)')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016',
                    help='Study B directory relative to data/ (default: fish_oil/batch_jul2016)')
    args = ap.parse_args()

    label_a     = Path(args.study_a).name
    label_b     = Path(args.study_b).name
    study_a_dir = DATA_DIR / args.study_a
    study_b_dir = DATA_DIR / args.study_b

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR = Path(__file__).parent.parent / 'figs'
    FIGS_DIR.mkdir(exist_ok=True)

    _results = {}
    txt_out = RESULTS_DIR / f'batch_effect_{label_a}_{label_b}.txt'
    with open(txt_out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            _results = main(study_a_dir, study_b_dir, label_a, label_b)
        finally:
            sys.stdout = orig
    print(f"Results saved → {txt_out}")

    if _results:
        save_figure(_results, FIGS_DIR, label_a, label_b)
