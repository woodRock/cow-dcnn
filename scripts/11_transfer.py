"""
Cross-study transfer classification.

Tests whether cross-study alignment preserves the biological signal within
each dataset while improving metabolome overlap across datasets.

Datasets:
  mtbls21  (wheat, 40 samples, 2 classes: CO2 treatment 0 vs 1)
  mtbls288 (rice,  79 samples, 4 classes: cultivar 0–3)

Protocol
--------
Reference: rice sample[0].  All samples are aligned to this reference using
each method.  Features: per-m/z max-projection (1000-dim), PCA → 50 dims.

Metrics (all use leave-one-out k-NN, k=5):
  Within-wheat accuracy   — k-NN on wheat features, CO2 treatment labels
  Within-rice  accuracy   — k-NN on rice features, cultivar labels
  Cross-study transfer    — for each wheat sample, retrieve its k=5 nearest
                            rice neighbours; majority-vote rice cultivar label
                            is NOT meaningful directly, but the cross-study
                            proximity in feature space IS: we report the
                            inter-study k-NN distance (lower = more overlap).

  Biological silhouette   — silhouette score using biological labels within
                            each dataset (higher = cleaner class separation).
  Study silhouette        — silhouette score for study label in the combined
                            119-sample space (lower = less batch effect).
"""

from __future__ import annotations

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

K_CLASS = 5
N_PCA   = 50


class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


def align_all(chromas: list[np.ndarray], ref: np.ndarray,
              encode_fn=None) -> list[np.ndarray]:
    aligned = []
    for c in chromas:
        _, _, _, _, warp_fn = align_pair(c, ref, encode_fn=encode_fn,
                                         return_anchors=True)
        aligned.append(warp_chroma_2d(c, warp_fn))
    return aligned


def feature_matrix(chromas: list[np.ndarray]) -> np.ndarray:
    return np.stack([max_projection(c) for c in chromas], axis=0)


def loo_knn_accuracy(X: np.ndarray, labels: np.ndarray, k: int = K_CLASS) -> float:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idxs = nbrs.kneighbors(X)
    correct = sum(
        int(np.bincount(labels[idxs[i, 1:]]).argmax()) == labels[i]
        for i in range(len(labels))
    )
    return correct / len(labels)


def cross_study_distance(X_wheat: np.ndarray, X_rice: np.ndarray,
                          k: int = K_CLASS) -> float:
    """Mean distance from each wheat sample to its k nearest rice neighbours."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_rice)
    dists, _ = nbrs.kneighbors(X_wheat)
    return float(dists.mean())


def evaluate(aligned_wheat: list[np.ndarray], aligned_rice: list[np.ndarray],
             y_wheat: np.ndarray, y_rice: np.ndarray) -> dict:
    X_w = feature_matrix(aligned_wheat)
    X_r = feature_matrix(aligned_rice)
    X_all = np.vstack([X_w, X_r])

    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)
    X_w_sc   = X_all_sc[:len(X_w)]
    X_r_sc   = X_all_sc[len(X_w):]

    n_comp = min(N_PCA, X_all_sc.shape[0] - 1, X_all_sc.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    X_all_pca = pca.fit_transform(X_all_sc)
    X_w_pca   = X_all_pca[:len(X_w)]
    X_r_pca   = X_all_pca[len(X_w):]

    within_wheat = loo_knn_accuracy(X_w_pca, y_wheat)
    within_rice  = loo_knn_accuracy(X_r_pca, y_rice)
    xstudy_dist  = cross_study_distance(X_w_pca, X_r_pca)

    study_labels = np.array([0] * len(X_w) + [1] * len(X_r), dtype=int)
    study_sil    = float(silhouette_score(X_all_pca, study_labels))

    bio_labels = np.concatenate([y_wheat, y_rice])
    bio_sil    = float(silhouette_score(X_all_pca, bio_labels))

    return {
        'within_wheat': within_wheat,
        'within_rice':  within_rice,
        'xstudy_dist':  xstudy_dist,
        'study_sil':    study_sil,
        'bio_sil':      bio_sil,
    }


def main() -> None:
    print("Loading chromatograms and labels …")
    m21  = load_chromas(DATA_DIR / 'mtbls21'  / 'chroma')
    m288 = load_chromas(DATA_DIR / 'mtbls288' / 'chroma')
    y21  = np.load(DATA_DIR / 'mtbls21'  / 'y.npy').astype(int)
    y288 = np.load(DATA_DIR / 'mtbls288' / 'y.npy').astype(int)
    print(f"  mtbls21  (wheat): {len(m21)} samples  "
          f"{len(np.unique(y21))}-class CO2 treatment")
    print(f"  mtbls288 (rice) : {len(m288)} samples  "
          f"{len(np.unique(y288))}-class cultivar")

    ref = m288[0]
    print(f"\nReference: rice sample[0]")
    print(f"Features : per-m/z max projection → PCA-{N_PCA}")
    print(f"k-NN k   : {K_CLASS}  (leave-one-out within each dataset)\n")

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
    for label, enc_fn in methods:
        print(f"Evaluating: {label} …", end=' ', flush=True)
        if label == 'Unaligned':
            aw, ar = m21, m288
        else:
            aw = align_all(m21,  ref, encode_fn=enc_fn)
            ar = align_all(m288, ref, encode_fn=enc_fn)
        r = evaluate(aw, ar, y21, y288)
        results[label] = r
        print(f"wheat acc={r['within_wheat']:.3f}  "
              f"rice acc={r['within_rice']:.3f}  "
              f"xdist={r['xstudy_dist']:.3f}  "
              f"study_sil={r['study_sil']:.3f}")

    w = 26
    print(f"\n{'='*85}")
    print(f"{'Method':<{w}}  {'Wheat acc':>10}  {'Rice acc':>9}  "
          f"{'X-dist':>7}  {'Study sil':>10}  {'Bio sil':>8}")
    print(f"{'-'*85}")
    for lbl, r in results.items():
        print(f"{lbl:<{w}}  {r['within_wheat']:>10.3f}  "
              f"{r['within_rice']:>9.3f}  "
              f"{r['xstudy_dist']:>7.3f}  "
              f"{r['study_sil']:>10.3f}  "
              f"{r['bio_sil']:>8.3f}")
    print(f"{'='*85}")
    print(f"Wheat/Rice acc: LOO k={K_CLASS} accuracy on CO2 treatment / cultivar labels")
    print(f"X-dist        : mean PCA distance from each wheat sample to k={K_CLASS} nearest rice")
    print(f"               (lower = more metabolome overlap after alignment)")
    print(f"Study sil     : silhouette for study label  (lower = less batch effect)")
    print(f"Bio sil       : silhouette for biological label across both datasets")
    print(f"               (higher = cleaner separation of shared biological variation)")
    return results


def save_figure(results: dict, figs_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    x       = np.arange(len(methods))
    bar_w   = 0.2
    colors  = ['#4878CF', '#6ACC65', '#D65F5F', '#956CB4']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: within-study classification accuracy
    ax = axes[0]
    wheat_acc = [results[m]['within_wheat'] for m in methods]
    rice_acc  = [results[m]['within_rice']  for m in methods]
    ax.bar(x - bar_w / 2, wheat_acc, bar_w, label='Wheat (CO2 treatment)', color='#E69F00')
    ax.bar(x + bar_w / 2, rice_acc,  bar_w, label='Rice (cultivar)',        color='#56B4E9')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('LOO k-NN accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Within-study biological signal\n(higher = better preserved)')
    ax.legend(fontsize=9)
    ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--', label='chance')

    # Right: batch effect — study silhouette and cross-study distance
    ax = axes[1]
    study_sil = [results[m]['study_sil']    for m in methods]
    xdist     = [results[m]['xstudy_dist']  for m in methods]
    ax2 = ax.twinx()
    ax.bar(x - bar_w / 2, study_sil, bar_w, label='Study silhouette ↓',   color='#D65F5F', alpha=0.85)
    ax2.bar(x + bar_w / 2, xdist,   bar_w, label='Cross-study distance ↓', color='#956CB4', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('Study silhouette')
    ax2.set_ylabel('Cross-study k-NN distance')
    ax.set_title('Cross-study batch effect\n(lower = better alignment)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    fig.suptitle('Transfer Evaluation — wheat (MTBLS21) × rice (MTBLS288)', y=1.02)
    fig.tight_layout()
    out = figs_dir / 'transfer.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out}")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR = Path(__file__).parent.parent / 'figs'
    FIGS_DIR.mkdir(exist_ok=True)

    _results = {}
    txt_out = RESULTS_DIR / 'transfer.txt'
    with open(txt_out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            _results = main()
        finally:
            sys.stdout = orig
    print(f"Results saved → {txt_out}")

    if _results:
        save_figure(_results, FIGS_DIR)
