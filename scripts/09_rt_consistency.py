"""
Retention time consistency for cross-study shared compounds.

For each alignment method, runs the full 3,160 wheat↔rice pair alignment
and collects the RT positions of all proposed anchor pairs (before RANSAC).
Anchors with raw m/z cosine ≥ 0.70 are treated as shared-compound evidence.

Metrics reported per method:
  RT deviation (pre-warp)  — |RT_wheat − RT_rice| in minutes for proposed anchors
                             (always the same across methods for the same pair)
  RT deviation (post-warp) — |warp(RT_wheat) − RT_rice| for the RANSAC-outlier
                             anchors (the ones NOT used to fit the warp).
                             These are held-out evaluation points: if the warp
                             generalises, residuals should be small.
  Warp magnitude           — mean |warp(t) − t| across all 200 time bins;
                             a proxy for how much RT correction the method applies.

Raw cosine has no warp (anchors are used directly), so post-warp residuals
are compared against the drift and cross-study encoders which each learn a
different warp from the same anchor candidates.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from alignment import (align_pair, load_chromas, load_encoder,
                       PRECISION_CUTOFF, BIN_MIN, TIME_AXIS, RUN_MIN, N_BINS)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'


class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


def evaluate_rt(m21: list[np.ndarray], m288: list[np.ndarray],
                encode_fn=None) -> dict:
    n_total = len(m21) * len(m288)
    done    = 0

    pre_devs    = []   # |RT_q − RT_r| before warp (in minutes)
    post_devs   = []   # |warp(RT_q) − RT_r| for held-out anchors
    warp_mags   = []   # mean |warp(t) − t| per pair

    for qc in m21:
        for rc in m288:
            warped_tic, n_anc, prec, proposed, warp_fn = align_pair(
                qc, rc, encode_fn=encode_fn, return_anchors=True)

            # Pre-warp RT deviation for all proposed anchors above precision cutoff
            for q_bin, r_bin, raw_cos in proposed:
                if raw_cos >= PRECISION_CUTOFF:
                    delta = abs(q_bin - r_bin) * BIN_MIN
                    pre_devs.append(delta)

            # Warp magnitude: how much does the warp deviate from identity?
            if warp_fn is not None:
                warp_vals = np.clip(warp_fn(TIME_AXIS), 0, RUN_MIN)
                warp_mags.append(float(np.mean(np.abs(warp_vals - TIME_AXIS))))

                # Post-warp residuals for held-out precision anchors
                # Use proposed anchors not selected as RANSAC inliers:
                # we can identify held-out pairs as those with |warp(q_t) - r_t| > 3*BIN_MIN
                for q_bin, r_bin, raw_cos in proposed:
                    if raw_cos < PRECISION_CUTOFF:
                        continue
                    q_t      = q_bin * BIN_MIN
                    r_t      = r_bin * BIN_MIN
                    warped_t = float(np.clip(warp_fn(q_t), 0, RUN_MIN))
                    post_devs.append(abs(warped_t - r_t))
            else:
                # No warp fitted (< 2 anchors): post-warp = pre-warp
                for q_bin, r_bin, raw_cos in proposed:
                    if raw_cos >= PRECISION_CUTOFF:
                        post_devs.append(abs(q_bin - r_bin) * BIN_MIN)

            done += 1
            if done % 500 == 0:
                print(f"    {done}/{n_total} pairs …")

    def _stats(vals):
        if not vals:
            return dict(mean=np.nan, median=np.nan, p90=np.nan, n=0)
        a = np.array(vals)
        return dict(mean=float(np.mean(a)), median=float(np.median(a)),
                    p90=float(np.percentile(a, 90)), n=len(a))

    return {
        'pre':      _stats(pre_devs),
        'post':     _stats(post_devs),
        'warp':     _stats(warp_mags),
        'pre_raw':  np.array(pre_devs),
        'post_raw': np.array(post_devs),
    }


def main() -> None:
    print("Loading chromatograms …")
    m21  = load_chromas(DATA_DIR / 'mtbls21'  / 'chroma')
    m288 = load_chromas(DATA_DIR / 'mtbls288' / 'chroma')
    print(f"  mtbls21  (wheat): {len(m21)} samples")
    print(f"  mtbls288 (rice) : {len(m288)} samples")
    print(f"  Precision cutoff: raw m/z cosine ≥ {PRECISION_CUTOFF}\n")

    methods: list[tuple[str, object]] = [('Raw cosine', None)]
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
        print(f"Evaluating: {label} …")
        r = evaluate_rt(m21, m288, enc_fn)
        results[label] = r
        print(f"  pre-warp  RT dev  mean={r['pre']['mean']:.3f} min  "
              f"median={r['pre']['median']:.3f}  p90={r['pre']['p90']:.3f}  "
              f"(n={r['pre']['n']})")
        print(f"  post-warp RT dev  mean={r['post']['mean']:.3f} min  "
              f"median={r['post']['median']:.3f}  p90={r['post']['p90']:.3f}")
        print(f"  warp magnitude    mean={r['warp']['mean']:.3f} min\n")

    w = 26
    print("=" * 85)
    print(f"{'Method':<{w}}  {'Pre mean':>9}  {'Pre p90':>8}  "
          f"{'Post mean':>10}  {'Post p90':>9}  {'Warp mag':>9}")
    print("-" * 85)
    for lbl, r in results.items():
        print(f"{lbl:<{w}}  "
              f"{r['pre']['mean']:>8.3f}m  {r['pre']['p90']:>7.3f}m  "
              f"{r['post']['mean']:>9.3f}m  {r['post']['p90']:>8.3f}m  "
              f"{r['warp']['mean']:>8.3f}m")
    print("=" * 85)
    print("All values in minutes.  Pre/post = RT deviation for precision anchors "
          "(m/z cosine ≥ 0.70).")
    print("Post-warp residual: how well the warp generalises to shared-compound anchors.")
    print("Warp magnitude: mean |warp(t) − t| — how aggressively each method corrects RT.")
    return results


def save_figure(results: dict, figs_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    colors  = ['#4878CF', '#6ACC65', '#D65F5F']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: violin plot of pre-warp RT deviations
    ax = axes[0]
    data = [results[m]['pre_raw'] for m in methods]
    parts = ax.violinplot(data, positions=range(len(methods)),
                          showmedians=True, showextrema=False)
    for i, (body, col) in enumerate(zip(parts['bodies'], colors)):
        body.set_facecolor(col)
        body.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('RT deviation (min)')
    ax.set_title('Pre-warp RT deviation\nshared-compound anchors')

    # Right: bar chart of warp magnitude + post-warp residual
    ax = axes[1]
    x     = np.arange(len(methods))
    bar_w = 0.35
    post  = [results[m]['post']['mean'] for m in methods]
    wmag  = [results[m]['warp']['mean'] for m in methods]
    ax.bar(x - bar_w / 2, post, bar_w, label='Post-warp residual (mean)', color='#D65F5F')
    ax.bar(x + bar_w / 2, wmag, bar_w, label='Warp magnitude (mean)',     color='#4878CF')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('Minutes')
    ax.set_title('Warp correction quality')
    ax.legend(fontsize=9)

    fig.suptitle('RT Consistency — wheat (MTBLS21) × rice (MTBLS288)', y=1.02)
    fig.tight_layout()
    out = figs_dir / 'rt_consistency.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out}")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR = Path(__file__).parent.parent / 'figs'
    FIGS_DIR.mkdir(exist_ok=True)

    _results = {}
    txt_out = RESULTS_DIR / 'rt_consistency.txt'
    with open(txt_out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            _results = main()
        finally:
            sys.stdout = orig
    print(f"Results saved → {txt_out}")

    if _results:
        save_figure(_results, FIGS_DIR)
