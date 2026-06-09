"""
Figure: retention time drift between two replicate GC-MS runs.

Shows the TIC traces of two replicate samples from the same fish and body part
before and after COW alignment, with matched peak pairs annotated to make the
drift magnitude visible.

Usage (run from cow-dcnn root):
  python scripts/plot_retention_drift.py
  python scripts/plot_retention_drift.py --pair BCO_Liver_Female  # default
  python scripts/plot_retention_drift.py --pair SNA1_Gonad
  python scripts/plot_retention_drift.py --out figs/retention_drift.pdf
  python scripts/plot_retention_drift.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

DATA_DIR    = Path('data/fish_oil')
FIGS_DIR    = Path('figs')
RESULTS_DIR = Path('results')
FIGS_DIR.mkdir(exist_ok=True)

class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()

N_BINS  = 200
RUN_MIN = 45.0
TIME_AX = np.linspace(0, RUN_MIN, N_BINS)
BIN_MIN = RUN_MIN / N_BINS

# Sample pairs: (index_A, index_B, label_A, label_B, short_name)
PAIRS = {
    'BCO_Liver_Female':  (53, 72, 'BCO4 Liver Female A',   'BCO5 Liver Female A',  'Blue cod liver (female)'),
    'BCO_Liver_Male':    (55, 75, 'BCO4 Liver Male A',     'BCO5 Liver Male A',    'Blue cod liver (male)'),
    'SNA1_Gonad':        (3,  4,  'SNA1 Gonad A',          'SNA1 Gonad B',         'Snapper gonad'),
    'SNA1_Liver':        (7,  9,  'SNA1 Liver Female A',   'SNA1 Liver Male A',    'Snapper liver'),
    'TAR3_Liver_Female': (38, 39, 'TAR3 Liver Female A',   'TAR3 Liver Female B',  'Tarakihi liver'),
}

COLOURS = ['#2166ac', '#d6604d']   # blue, red


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_peaks(tic: np.ndarray, n_max: int = 12) -> np.ndarray:
    thr = np.percentile(tic, 82)
    pks, props = find_peaks(tic, height=thr, distance=4)
    if len(pks) > n_max:
        top = np.argsort(props['peak_heights'])[-n_max:]
        pks = np.sort(pks[top])
    return pks


def match_peaks(pks_a: np.ndarray, pks_b: np.ndarray,
                max_drift_bins: int = 20) -> list[tuple[int, int]]:
    """Hungarian matching within a time window."""
    ta = pks_a * BIN_MIN
    tb = pks_b * BIN_MIN
    dist = np.abs(ta[:, None] - tb[None, :])
    cost = dist.copy()
    cost[dist > max_drift_bins * BIN_MIN] = 1e6
    ri, ci = linear_sum_assignment(cost)
    return [(pks_a[r], pks_b[c])
            for r, c in zip(ri, ci)
            if dist[r, c] <= max_drift_bins * BIN_MIN]


def align_cow(query: np.ndarray, ref: np.ndarray,
              n_segments: int = 10, slack: int = 10) -> np.ndarray:
    """COW alignment (Nielsen et al. 1998) via DP on TIC segment correlations."""
    ref_tic = ref.sum(1); q_tic = query.sum(1)
    M, T = N_BINS, N_BINS // n_segments

    def seg_corr(r_seg, q_start, q_len):
        if q_start + q_len > M or q_len < 2: return -np.inf
        q_seg = q_tic[q_start:q_start + q_len]
        if len(q_seg) != len(r_seg):
            q_seg = np.interp(np.linspace(0, 1, len(r_seg)),
                              np.linspace(0, 1, len(q_seg)), q_seg)
        sr, sq = np.std(r_seg), np.std(q_seg)
        if sr < 1e-8 or sq < 1e-8: return 0.0
        return float(np.corrcoef(r_seg, q_seg)[0, 1])

    ref_segs = [ref_tic[i*T:(i+1)*T] for i in range(n_segments)]
    INF = -1e9
    dp_score = np.full(M+1, INF); dp_score[0] = 0.0
    all_prev = []
    for i in range(n_segments):
        new_score = np.full(M+1, INF); new_prev = np.full(M+1, -1, dtype=int)
        for bp in range(M+1):
            if dp_score[bp] == INF: continue
            for dq in range(max(1, T-slack), T+slack+1):
                bn = bp + dq
                if bn > M: break
                if i == n_segments-1 and bn != M: continue
                sc = dp_score[bp] + seg_corr(ref_segs[i], bp, dq)
                if sc > new_score[bn]: new_score[bn] = sc; new_prev[bn] = bp
        dp_score = new_score; all_prev.append(new_prev)

    boundaries = [M]; pos = M
    for layer in reversed(all_prev):
        prev = layer[pos]
        if prev < 0: return query.copy()
        boundaries.append(prev); pos = prev
    boundaries = list(reversed(boundaries))

    ref_pts = [i * T * BIN_MIN for i in range(n_segments + 1)]
    q_pts   = [b * BIN_MIN     for b in boundaries]
    for i in range(1, len(q_pts)):
        if q_pts[i] <= q_pts[i-1]: q_pts[i] = q_pts[i-1] + 0.01

    warp = interp1d(ref_pts, q_pts, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
    query_times = np.clip(warp(TIME_AX), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(query_times):
        idx = t / BIN_MIN
        t0 = int(np.clip(idx, 0, N_BINS-1)); t1 = min(t0+1, N_BINS-1)
        a  = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1-a)*query[t0] + a*query[t1]
    return warped.astype(np.float32)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(pair_key: str, out_path: Path) -> None:
    idx_a, idx_b, label_a, label_b, title = PAIRS[pair_key]

    npz_paths = sorted((DATA_DIR / 'chroma').glob('*.npz'))
    chroma_a  = np.load(npz_paths[idx_a])['chroma'].astype('float32')
    chroma_b  = np.load(npz_paths[idx_b])['chroma'].astype('float32')

    tic_a = chroma_a.sum(1)
    tic_b = chroma_b.sum(1)

    tic_a_n = tic_a / (np.linalg.norm(tic_a) + 1e-8)
    tic_b_n = tic_b / (np.linalg.norm(tic_b) + 1e-8)

    pks_a = detect_peaks(tic_a_n)
    pks_b = detect_peaks(tic_b_n)
    pairs = match_peaks(pks_a, pks_b)

    chroma_b_aligned = align_cow(chroma_b, chroma_a)
    tic_b_aligned    = chroma_b_aligned.sum(1)
    tic_b_aligned_n  = tic_b_aligned / (np.linalg.norm(tic_b_aligned) + 1e-8)

    pks_b_aligned = detect_peaks(tic_b_aligned_n)
    pairs_after   = match_peaks(pks_a, pks_b_aligned)

    r_before = pearsonr(tic_a_n, tic_b_n)[0]
    r_after  = pearsonr(tic_a_n, tic_b_aligned_n)[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    panel_data = [
        (axes[0], tic_a_n, tic_b_n,        r_before, 'Before alignment',     pairs,       True),
        (axes[1], tic_a_n, tic_b_aligned_n, r_after,  'After COW alignment',  pairs_after, False),
    ]
    for ax, tic_ref, tic_qry, r_val, subtitle, pk_pairs, show_drift in panel_data:
        ax.plot(TIME_AX, tic_ref, color=COLOURS[0], lw=1.4, label=label_a, zorder=3)
        ax.plot(TIME_AX, tic_qry, color=COLOURS[1], lw=1.4,
                alpha=0.85, label=label_b, zorder=2)
        ax.fill_between(TIME_AX, tic_ref, tic_qry,
                        alpha=0.12, color='grey', zorder=1)

        if show_drift:
            plotted = 0
            sorted_pairs = sorted(pk_pairs,
                                  key=lambda p: abs(TIME_AX[p[1]] - TIME_AX[p[0]]),
                                  reverse=True)
            for pa, pb in sorted_pairs:
                ta = TIME_AX[pa]; tb = TIME_AX[pb]
                drift_min = tb - ta
                if abs(drift_min) < 0.15:
                    continue
                ha = tic_a_n[pa]; hb = tic_b_n[pb]
                y_top = max(ha, hb) + 0.015

                ax.axvline(ta, color=COLOURS[0], lw=0.6, ls='--', alpha=0.45, zorder=1)
                ax.axvline(tb, color=COLOURS[1], lw=0.6, ls='--', alpha=0.45, zorder=1)

                ax.annotate('', xy=(tb, y_top), xytext=(ta, y_top),
                            arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8))
                tick_h = 0.004
                for tx in (ta, tb):
                    ax.plot([tx, tx], [y_top - tick_h, y_top + tick_h],
                            color='#555555', lw=0.8)

                ax.text((ta + tb) / 2, y_top + 0.006,
                        f'{drift_min:+.2f} min',
                        ha='center', va='bottom', fontsize=6.5, color='#333333')
                plotted += 1
                if plotted >= 6:
                    break

        ax.text(0.98, 0.93, f'r = {r_val:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))

        ax.set_title(subtitle, fontsize=10, loc='left', pad=4)
        ax.set_ylabel('Normalised TIC', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=-0.005)

    axes[1].set_xlabel('Retention time (min)', fontsize=9)

    handles = [
        mpatches.Patch(color=COLOURS[0], label=label_a),
        mpatches.Patch(color=COLOURS[1], label=label_b),
    ]
    axes[0].legend(handles=handles, fontsize=8.5, loc='upper left',
                   framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(f'Retention time drift — {title} (replicate A vs B)',
                 fontsize=11, y=0.98)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    print(f'  Before alignment: r = {r_before:.3f}')
    print(f'  After  alignment: r = {r_after:.3f}')
    print(f'  Matched peak pairs: {len(pairs)}')
    drifts = [(TIME_AX[pb] - TIME_AX[pa]) for pa, pb in pairs]
    if drifts:
        print(f'  Drift range: {min(drifts):+.2f} to {max(drifts):+.2f} min  '
              f'(mean |drift| = {np.mean(np.abs(drifts)):.2f} min)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='BCO_Liver_Female',
                    choices=list(PAIRS.keys()),
                    help='Which replicate pair to plot')
    ap.add_argument('--out',  default=None,
                    help='Output path (default: figs/retention_drift_<pair>.pdf)')
    ap.add_argument('--all',  action='store_true',
                    help='Generate figures for all pairs')
    args = ap.parse_args()

    if args.all:
        for key in PAIRS:
            out = FIGS_DIR / f'retention_drift_{key}.pdf'
            make_figure(key, out)
    else:
        out = Path(args.out) if args.out else FIGS_DIR / f'retention_drift_{args.pair}.pdf'
        make_figure(args.pair, out)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'retention_drift.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
