"""
Visualise cross-study alignment for a representative wheat×rice pair.

Produces results/alignment_figure.pdf with two panels:
  Top  : TIC overlays for each method (wheat reference vs aligned rice),
         with RANSAC-inlier anchors colour-coded by spectral precision.
  Bottom : warp functions for all aligned methods on the same pair.

Usage:
  python scripts/12_visualize_alignment.py
  python scripts/12_visualize_alignment.py --pair 7 23   # (wheat_idx, rice_idx)
  python scripts/12_visualize_alignment.py --n-sample 80
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from alignment import load_encoder

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS         = 200
RUN_MIN        = 45.0
BIN_MIN        = RUN_MIN / N_BINS
TIME_AXIS      = np.linspace(0, RUN_MIN, N_BINS)
MAX_DRIFT_MIN  = 20.0
SIM_THRESHOLD  = 0.30
PRECISION_CUTOFF = 0.70

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

METHOD_COLORS = {
    'Unaligned':           '#999999',
    'Raw cosine':          '#E69F00',
    'Drift encoder':       '#56B4E9',
    'Cross-study encoder': '#009E73',
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_chromas(chroma_dir: Path) -> list[np.ndarray]:
    return [np.load(p)['chroma'].astype(np.float32)
            for p in sorted(chroma_dir.glob('*.npz'))]


# ── Alignment primitives ──────────────────────────────────────────────────────

def detect_peaks(chroma, height_pct=80, min_dist=5, n_peaks=30):
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, height_pct)
    pks, props = sp_find_peaks(tic, height=thr, distance=min_dist)
    if len(pks) > n_peaks:
        top = np.argsort(props['peak_heights'])[-n_peaks:]
        pks = pks[top]
    return np.sort(pks)


def _l2(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _ctx_fp(chroma, pk):
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))


def fingerprints(chroma, pks):
    return np.array([_ctx_fp(chroma, pk) for pk in pks])


# ── Detailed single-pair alignment ────────────────────────────────────────────

def align_detail(query: np.ndarray, ref: np.ndarray, encode_fn=None) -> dict:
    """Run alignment and return all intermediates needed for plotting."""
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)

    _null = dict(
        warped_tic=q_tic, ref_tic=ref_tic, q_tic=q_tic,
        anchor_q_t=np.array([]), anchor_r_t=np.array([]),
        anchor_cosines=np.array([]),
        warp_ref=np.array([0., RUN_MIN]),
        warp_query=np.array([0., RUN_MIN]),
        tic_r=float(np.corrcoef(ref_tic, q_tic)[0, 1]),
        n_anchors=0,
    )

    ref_pks = detect_peaks(ref)
    q_pks   = detect_peaks(query)
    if len(ref_pks) < 2 or len(q_pks) < 2:
        return _null

    ref_fps = fingerprints(ref, ref_pks)
    q_fps   = fingerprints(query, q_pks)
    ref_rts = ref_pks.astype(np.float32) / N_BINS
    q_rts   = q_pks.astype(np.float32) / N_BINS

    ref_rep = encode_fn(ref_fps, ref_rts) if encode_fn is not None else ref_fps
    q_rep   = encode_fn(q_fps,   q_rts)   if encode_fn is not None else q_fps

    sim = cosine_similarity(q_rep, ref_rep)
    ref_t = TIME_AXIS[ref_pks]
    q_t   = TIME_AXIS[q_pks]
    drift_mask = np.abs(q_t[:, None] - ref_t[None, :]) > MAX_DRIFT_MIN
    sim_c = sim.copy(); sim_c[drift_mask] = -1.0
    row_ind, col_ind = linear_sum_assignment(-sim_c)

    avail = sim[~drift_mask].flatten()
    thr   = max(float(np.percentile(avail, 70)), SIM_THRESHOLD) if len(avail) >= 3 else SIM_THRESHOLD
    ms    = sim[row_ind, col_ind]
    keep  = (ms >= thr) & ~drift_mask[row_ind, col_ind]
    row_ind, col_ind = row_ind[keep], col_ind[keep]
    if len(row_ind) < 2:
        return _null

    raw_cosines = np.array([np.dot(q_fps[qi], ref_fps[ri])
                             for qi, ri in zip(row_ind, col_ind)])

    refined_q = []
    for idx in range(len(row_ind)):
        r_spec = ref_fps[col_ind[idx]]
        q_pk   = q_pks[row_ind[idx]]
        best_bin, best_s = q_pk, float(np.dot(r_spec, _ctx_fp(query, q_pk)))
        for delta in range(-3, 4):
            if delta == 0:
                continue
            cand = int(np.clip(q_pk + delta, 0, N_BINS - 1))
            s    = float(np.dot(r_spec, _ctx_fp(query, cand)))
            if s > best_s:
                best_s, best_bin = s, cand
        refined_q.append(best_bin)
    refined_q = np.array(refined_q)

    order    = np.argsort(col_ind)
    row_ind, col_ind = row_ind[order], col_ind[order]
    refined_q   = refined_q[order]
    raw_cosines = raw_cosines[order]

    ra = TIME_AXIS[ref_pks[col_ind]]
    qa = TIME_AXIS[refined_q]

    if len(ra) >= 4:
        try:
            ransac = RANSACRegressor(residual_threshold=3 * BIN_MIN,
                                     min_samples=2, random_state=42)
            ransac.fit(qa.reshape(-1, 1), ra)
            if ransac.inlier_mask_.sum() >= 2:
                ra = ra[ransac.inlier_mask_]
                qa = qa[ransac.inlier_mask_]
                raw_cosines = raw_cosines[ransac.inlier_mask_]
        except Exception:
            pass

    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]:
            mono.append(i)
    ra, qa = ra[mono], qa[mono]
    raw_cosines = raw_cosines[mono]
    if len(ra) < 2:
        return _null

    # warp: ref_time → query_time
    rf  = np.concatenate([[0.], ra, [RUN_MIN]])
    qf  = np.concatenate([[0.], qa, [RUN_MIN]])
    warp    = PchipInterpolator(rf, qf)
    warped  = interp1d(TIME_AXIS, q_tic, bounds_error=False,
                       fill_value=0.0)(warp(TIME_AXIS))
    tic_r   = float(np.corrcoef(ref_tic, warped)[0, 1])

    return dict(
        warped_tic=warped, ref_tic=ref_tic, q_tic=q_tic,
        anchor_q_t=qa, anchor_r_t=ra, anchor_cosines=raw_cosines,
        warp_ref=rf, warp_query=qf,
        tic_r=tic_r, n_anchors=len(ra),
    )


def quick_tic_r(query, ref, encode_fn=None):
    """Fast TIC r without saving intermediates (for pair selection)."""
    d = align_detail(query, ref, encode_fn)
    return d['tic_r']


# ── Representative pair selection ─────────────────────────────────────────────

def find_representative_pair(m21, m288, cross_enc, n_sample=60, rng_seed=42):
    """
    Sample n_sample pairs, run cross-study encoder, return (wi, ri) for the
    pair whose Δ TIC r (aligned − unaligned) is closest to the median.
    """
    rng  = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(m21) * len(m288), size=min(n_sample, len(m21) * len(m288)),
                      replace=False)
    pairs = [(int(i // len(m288)), int(i % len(m288))) for i in idxs]

    deltas, results = [], []
    for wi, ri in pairs:
        r_un  = float(np.corrcoef(m21[wi].sum(axis=1), m288[ri].sum(axis=1))[0, 1])
        r_cs  = quick_tic_r(m21[wi], m288[ri], cross_enc)
        deltas.append(r_cs - r_un)
        results.append((wi, ri, r_un, r_cs))

    deltas = np.array(deltas)
    median_delta = np.median(deltas)
    best = int(np.argmin(np.abs(deltas - median_delta)))
    wi, ri, r_un, r_cs = results[best]
    print(f"  Representative pair: A[{wi}] × B[{ri}]  "
          f"unaligned r={r_un:.3f}  cross-study r={r_cs:.3f}  "
          f"Δ={r_cs - r_un:+.3f}  (median Δ={median_delta:+.3f})")
    return wi, ri


# ── Plotting ──────────────────────────────────────────────────────────────────

def _norm_tic(tic):
    mx = tic.max()
    return tic / mx if mx > 0 else tic


def plot_alignment(results_by_method: dict, out_path: Path,
                   label_a: str = 'study_a', label_b: str = 'study_b') -> None:
    """
    results_by_method: OrderedDict of method_label → align_detail output dict
    First entry must be 'Unaligned'.
    """
    methods   = list(results_by_method.keys())
    n_aligned = len(methods) - 1   # excludes Unaligned

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(
        2, len(methods),
        height_ratios=[3, 1.6],
        hspace=0.45, wspace=0.12,
        left=0.06, right=0.98, top=0.93, bottom=0.08,
    )

    ref_tic = _norm_tic(list(results_by_method.values())[0]['ref_tic'])

    # ── Top row: TIC overlays ─────────────────────────────────────────────────
    ax_tic = []
    for col, (label, d) in enumerate(results_by_method.items()):
        ax = fig.add_subplot(gs[0, col])
        ax_tic.append(ax)

        color = METHOD_COLORS.get(label, '#333333')
        q_plot = _norm_tic(d['warped_tic'])

        ax.fill_between(TIME_AXIS, ref_tic, alpha=0.20, color='#444444')
        ax.plot(TIME_AXIS, ref_tic,  color='#333333', lw=1.0,
                label='Wheat (ref)')
        ax.plot(TIME_AXIS, q_plot,   color=color,     lw=1.0, alpha=0.85,
                label='Rice (aligned)')

        # Anchor ticks along x-axis
        if len(d['anchor_r_t']) > 0:
            prec = d['anchor_cosines'] >= PRECISION_CUTOFF
            for rt, ok in zip(d['anchor_r_t'], prec):
                ax.axvline(rt, ymin=0, ymax=0.06,
                           color='#2ca02c' if ok else '#d62728',
                           lw=1.5, alpha=0.8)

        r_val = d['tic_r']
        n_anc = d['n_anchors']
        ax.set_title(f'{label}\n$r={r_val:.3f}$, {n_anc} anchors',
                     fontsize=9, pad=4)
        ax.set_xlim(0, RUN_MIN)
        ax.set_ylim(-0.03, 1.15)
        ax.set_xlabel('Retention time (min)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Normalised TIC', fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend on first TIC panel
    legend_elements = [
        Line2D([0], [0], color='#333333', lw=1.5, label='Wheat TIC (reference)'),
        Line2D([0], [0], color='#888888', lw=1.5, label='Rice TIC (aligned)'),
        Line2D([0], [0], color='#2ca02c', lw=2,   label=f'Anchor cosine ≥ {PRECISION_CUTOFF}'),
        Line2D([0], [0], color='#d62728', lw=2,   label=f'Anchor cosine < {PRECISION_CUTOFF}'),
    ]
    ax_tic[0].legend(handles=legend_elements, fontsize=6.5,
                     loc='upper right', framealpha=0.7)

    # ── Bottom row: warp functions ────────────────────────────────────────────
    ax_warp = fig.add_subplot(gs[1, :])
    ax_warp.plot([0, RUN_MIN], [0, RUN_MIN], '--', color='#aaaaaa',
                 lw=1.2, label='Identity (no warp)', zorder=1)

    for label, d in results_by_method.items():
        if label == 'Unaligned' or len(d['warp_ref']) < 3:
            continue
        color = METHOD_COLORS.get(label, '#333333')
        t_ref = np.linspace(0, RUN_MIN, 400)
        t_q   = PchipInterpolator(d['warp_ref'], d['warp_query'])(t_ref)
        ax_warp.plot(t_ref, t_q, color=color, lw=1.8, label=label, zorder=2)
        ax_warp.scatter(d['anchor_r_t'], d['anchor_q_t'],
                        color=color, s=22, zorder=3, alpha=0.7)

    ax_warp.set_xlabel('Reference retention time (min)', fontsize=8)
    ax_warp.set_ylabel('Query retention time (min)', fontsize=8)
    ax_warp.set_title('Warp functions — query RT mapped to reference RT',
                      fontsize=9, pad=4)
    ax_warp.set_xlim(0, RUN_MIN)
    ax_warp.set_ylim(0, RUN_MIN)
    ax_warp.legend(fontsize=7.5, loc='upper left', framealpha=0.8)
    ax_warp.tick_params(labelsize=7)
    ax_warp.spines['top'].set_visible(False)
    ax_warp.spines['right'].set_visible(False)

    fig.suptitle(
        f'Cross-study alignment: {label_a} × {label_b}',
        fontsize=10, y=0.98,
    )

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015',
                    help='Study A directory relative to data/ (default: fish_oil/batch_sep2015)')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016',
                    help='Study B directory relative to data/ (default: fish_oil/batch_jul2016)')
    ap.add_argument('--pair',     nargs=2, type=int, metavar=('A', 'B'),
                    help='Fix study-A and study-B sample indices (0-based)')
    ap.add_argument('--n-sample', type=int, default=60,
                    help='Pairs to sample when auto-selecting (default 60)')
    ap.add_argument('--out',      type=Path, default=None)
    args = ap.parse_args()

    label_a     = Path(args.study_a).name
    label_b     = Path(args.study_b).name
    study_a_dir = DATA_DIR / args.study_a
    study_b_dir = DATA_DIR / args.study_b
    out_path    = args.out or RESULTS_DIR / f'alignment_figure_{label_a}_{label_b}.pdf'

    print("Loading chromatograms …")
    m21  = load_chromas(study_a_dir / 'chroma')
    m288 = load_chromas(study_b_dir / 'chroma')
    print(f"  {label_a}: {len(m21)} samples  |  {label_b}: {len(m288)} samples")

    print("\nLoading encoders …")
    encoders: dict[str, object | None] = {'Unaligned': None, 'Raw cosine': None}
    for label, fname in [('Drift encoder',       'drift_simclr.pt'),
                         ('Cross-study encoder', 'cross_study_simclr.pt')]:
        p = CKPT_DIR / fname
        if p.exists():
            encoders[label] = load_encoder(p, DEVICE)
            print(f"  Loaded {label} from {fname}")
        else:
            print(f"  SKIP {label}: {p} not found")

    if args.pair:
        wi, ri = args.pair
        print(f"\nUsing specified pair: {label_a}[{wi}] × {label_b}[{ri}]")
    else:
        print(f"\nSampling {args.n_sample} pairs to find representative …")
        cross_enc = encoders.get('Cross-study encoder')
        wi, ri = find_representative_pair(m21, m288, cross_enc, args.n_sample)

    print(f"\nRunning detailed alignment for {label_a}[{wi}] × {label_b}[{ri}] …")
    results = {}
    for label, enc_fn in encoders.items():
        d = align_detail(m21[wi], m288[ri], enc_fn)
        results[label] = d
        print(f"  {label:<24}  r={d['tic_r']:.3f}  anchors={d['n_anchors']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_alignment(results, out_path, label_a, label_b)


if __name__ == '__main__':
    main()
