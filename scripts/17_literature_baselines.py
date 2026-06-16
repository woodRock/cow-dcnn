"""
17_literature_baselines.py — COW-TIC and icoshift as literature baselines.

Implements two classical TIC-only alignment algorithms and evaluates them
pairwise, matching the evaluation protocol of 16_align_testtime_calib.py.

Both methods operate purely on the Total Ion Chromatogram (TIC = sum over m/z)
and use no spectral identity information, in contrast to the spectral-anchor
PCHIP pipeline.

Methods
-------
COW-TIC   Nielsen et al. 1998, J. Chemometrics 12, 69–88.
             Correlation Optimized Warping: divides the reference TIC into
             n_segs equal segments and finds the query segment boundaries
             (within ±slack bins) that maximise the sum of pairwise Pearson
             correlations, via dynamic programming.

icoshift  Savorani et al. 2010, J. Magn. Reson. 202, 190–202.
             Interval Correlation Shifting: divides TIC into n_intervals equal
             segments and shifts each independently by the lag that maximises
             FFT cross-correlation with the corresponding reference segment.

Metrics match 16_align_testtime_calib.py:
  TIC r     — mean pairwise Pearson r across all A×B pairs
  Δ unaligned — improvement over unaligned baseline
  Study sil — study-label silhouette in PCA-50 space (lower = less batch effect)
               computed by aligning all study-A samples to study-B[0] as
               common reference, then max-projecting → PCA-50 → silhouette
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.signal import correlate
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DATA_DIR    = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS    = 200
RUN_MIN   = 45.0
BIN_MIN   = RUN_MIN / N_BINS           # 0.225 min per bin
TIME_AXIS = np.linspace(0, RUN_MIN, N_BINS)

# Max RT drift for cross-study (same as 07_cross_study.py)
MAX_DRIFT_BINS = int(round(20.0 / BIN_MIN))   # ~89 bins

# COW-TIC parameters
COW_N_SEGS = 4          # reference segments
COW_SLACK  = 22         # ±bins per interior node (~5 min)
                        # 4 nodes × 22 bins × 0.225 min = ~20 min total budget

# icoshift parameters
ICO_N_INTERVALS = 10    # equal-width intervals (200/10 = 20 bins = 4.5 min each)
ICO_MAX_SHIFT   = 20    # ±bins per interval (~4.5 min)


# ── I/O helpers ──────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


def load_chromas(chroma_dir: Path) -> list[np.ndarray]:
    return [np.load(p)['chroma'].astype(np.float32)
            for p in sorted(chroma_dir.glob('*.npz'))]


# ── COW-TIC ──────────────────────────────────────────────────────────────────

def _pearson_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two equal-length 1-D arrays (no error checking)."""
    a_z = a - a.mean()
    b_z = b - b.mean()
    na = np.linalg.norm(a_z)
    nb = np.linalg.norm(b_z)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a_z, b_z) / (na * nb))


def cow_tic_align(query_tic: np.ndarray, ref_tic: np.ndarray,
                  n_segs: int = COW_N_SEGS,
                  slack: int = COW_SLACK) -> tuple[np.ndarray, float]:
    """
    Correlation Optimized Warping on TIC profiles.

    Divides the reference into n_segs equal segments with fixed boundaries.
    Interior node positions in the query are free within ±slack bins.
    Dynamic programming selects the sequence that maximises the sum of
    segment-wise Pearson correlations.

    Returns
    -------
    warped_tic : (N_BINS,) float32
    warp_mag   : mean |warp(t) − t| in minutes
    """
    N = len(ref_tic)
    T = N // n_segs

    # Reference node positions (fixed, evenly spaced)
    r_nodes = [i * T for i in range(n_segs)] + [N - 1]

    # Candidate query positions for each node
    node_cands: list[list[int]] = []
    for i, rn in enumerate(r_nodes):
        if i == 0 or i == n_segs:
            node_cands.append([int(rn)])
        else:
            lo = max(1, rn - slack)
            hi = min(N - 2, rn + slack)
            node_cands.append(list(range(int(lo), int(hi) + 1)))

    # DP: dp[node] = {q_pos: cumulative_correlation}
    #     back[node] = {q_pos: prev_q_pos}
    dp   = [dict() for _ in range(n_segs + 1)]
    back = [dict() for _ in range(n_segs + 1)]
    dp[0][0] = 0.0

    for seg in range(n_segs):
        r_s = r_nodes[seg]
        r_e = r_nodes[seg + 1]
        r_seg = ref_tic[r_s:r_e + 1]
        r_len = len(r_seg)

        for q_s, cum in dp[seg].items():
            for q_e in node_cands[seg + 1]:
                if q_e <= q_s:
                    continue
                q_raw = query_tic[q_s:q_e + 1]
                if len(q_raw) < 2:
                    continue
                # Resample query segment to match reference segment length
                q_seg = np.interp(
                    np.linspace(0, 1, r_len),
                    np.linspace(0, 1, len(q_raw)),
                    q_raw,
                )
                cor = _pearson_fast(r_seg, q_seg)
                new_cum = cum + cor
                if new_cum > dp[seg + 1].get(q_e, -np.inf):
                    dp[seg + 1][q_e] = new_cum
                    back[seg + 1][q_e] = q_s

    # Backtrack from fixed last node
    q_last = r_nodes[-1]
    if q_last not in back[n_segs]:
        return query_tic.astype(np.float32), 0.0

    q_nodes_rev = [q_last]
    for seg in range(n_segs, 0, -1):
        prev = back[seg].get(q_nodes_rev[-1])
        if prev is None:
            return query_tic.astype(np.float32), 0.0
        q_nodes_rev.append(prev)
    q_nodes = list(reversed(q_nodes_rev))

    # Build warped TIC and compute warp magnitude
    warped   = np.empty(N, dtype=np.float32)
    abs_diff = []
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]
        q_s, q_e = q_nodes[seg], q_nodes[seg + 1]
        r_len = r_e - r_s + 1
        q_raw = query_tic[q_s:q_e + 1]

        warped[r_s:r_e + 1] = (
            np.interp(np.linspace(0, 1, r_len),
                      np.linspace(0, 1, len(q_raw)), q_raw)
            if len(q_raw) >= 2 else np.full(r_len, float(q_raw[0]) if len(q_raw) else 0.0)
        )
        # Warp magnitude: at each ref position t, the query position is linearly
        # interpolated between q_s and q_e — compare with the ref position t
        r_positions = np.arange(r_s, r_e + 1, dtype=float)
        q_positions = np.linspace(q_s, q_e, r_len)
        abs_diff.extend(np.abs(q_positions - r_positions).tolist())

    warp_mag = float(np.mean(abs_diff)) * BIN_MIN   # convert bins → minutes

    # Warp function: maps reference time (minutes) → query time (minutes)
    r_times = np.array(r_nodes, dtype=float) * BIN_MIN
    q_times = np.array(q_nodes, dtype=float) * BIN_MIN
    warp_fn = lambda t, _r=r_times, _q=q_times: np.interp(t, _r, _q)

    return warped, warp_mag, warp_fn


# ── icoshift ─────────────────────────────────────────────────────────────────

def icoshift_align(query_tic: np.ndarray, ref_tic: np.ndarray,
                   n_intervals: int = ICO_N_INTERVALS,
                   max_shift: int = ICO_MAX_SHIFT) -> tuple[np.ndarray, float]:
    """
    Interval Correlation Shifting (icoshift) on TIC profiles.
    Savorani et al. 2010, J. Magn. Reson. 202, 190–202.

    Divides the TIC into n_intervals equal-width segments. For each segment,
    uses FFT cross-correlation to find the integer shift (within ±max_shift)
    that maximises correlation with the corresponding reference segment. The
    shifted query value is taken by nearest-neighbour lookup with edge clamping.

    Returns
    -------
    warped_tic : (N_BINS,) float32
    warp_mag   : mean |shift| in minutes
    """
    N      = len(query_tic)
    warped = query_tic.copy().astype(np.float32)
    bounds = np.round(np.linspace(0, N, n_intervals + 1)).astype(int)

    best_lags        = []
    total_abs_shift  = 0.0
    n_bins_processed = 0

    for i in range(n_intervals):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        seg_len = hi - lo
        if seg_len < 4:
            continue

        r_seg = ref_tic[lo:hi].astype(np.float64)
        q_seg = query_tic[lo:hi].astype(np.float64)

        # Full cross-correlation via scipy (equivalent to FFT for short signals)
        # cc[lag + seg_len - 1] = sum_t ref[t] * query[t - lag]
        # Maximum at lag gives: aligning query shifted right by `lag` matches ref
        cc   = correlate(r_seg, q_seg, mode='full')
        lags = np.arange(-(seg_len - 1), seg_len)

        # Restrict to ±max_shift
        valid   = np.abs(lags) <= max_shift
        best_lag = int(lags[valid][np.argmax(cc[valid])])

        if best_lag != 0:
            # warped[k] = query_tic[k - best_lag] for k in [lo, hi)
            src = np.arange(lo, hi) - best_lag
            src = src.clip(0, N - 1)
            warped[lo:hi] = query_tic[src]

        best_lags.append(best_lag)
        total_abs_shift  += abs(best_lag) * seg_len
        n_bins_processed += seg_len

    warp_mag = (total_abs_shift / max(n_bins_processed, 1)) * BIN_MIN

    # Warp function: piecewise-constant per interval
    # warp_fn(t) = t - best_lag[interval(t)] * BIN_MIN
    _bounds    = bounds.astype(float) * BIN_MIN
    _lags_min  = np.array(best_lags, dtype=float) * BIN_MIN

    def warp_fn(t, _b=_bounds, _l=_lags_min):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(_b[1:], t)
        idx = np.clip(idx, 0, len(_l) - 1)
        return t - _l[idx]

    return warped, warp_mag, warp_fn


# ── Silhouette ────────────────────────────────────────────────────────────────

def _apply_warp_2d(chroma: np.ndarray, warp_fn) -> np.ndarray:
    """Apply a warp function (ref_time → query_time) to a 2-D chromatogram."""
    warped_times = np.clip(warp_fn(TIME_AXIS), 0.0, RUN_MIN)
    warped = np.zeros_like(chroma)
    for i, t in enumerate(warped_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0.0, 1.0))
        warped[i] = (1.0 - a) * chroma[t0] + a * chroma[t1]
    return warped.astype(np.float32)


def compute_silhouette(chromas_a: list[np.ndarray], chromas_b: list[np.ndarray],
                       align_fn) -> float:
    """
    Study-label silhouette in PCA-50 space (lower = less batch separation).

    All study-A samples are aligned to study-B[0] as the common reference
    using the supplied align_fn; study-B samples are kept raw. Per-m/z
    max-intensity projection → PCA-50 → silhouette score.
    """
    ref_chroma = chromas_b[0]
    ref_tic    = ref_chroma.sum(axis=1)

    aligned_a = []
    for qc in chromas_a:
        _, _, warp_fn = align_fn(qc.sum(axis=1), ref_tic)
        aligned_a.append(_apply_warp_2d(qc, warp_fn))

    all_chromas = aligned_a + list(chromas_b)
    feats  = np.stack([c.max(axis=0) for c in all_chromas])
    labels = np.array([0] * len(aligned_a) + [1] * len(chromas_b))
    n_comp = min(50, feats.shape[0] - 1)
    pca    = PCA(n_components=n_comp).fit_transform(feats)
    return float(silhouette_score(pca, labels))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_pairwise(
    chromas_a: list[np.ndarray],
    chromas_b: list[np.ndarray],
    align_fn,
) -> tuple[dict, list[float], list[float]]:
    """
    Align every sample in chromas_a onto every sample in chromas_b as reference.

    align_fn(query_tic, ref_tic) → (warped_tic, warp_mag_minutes, warp_fn)

    Returns
    -------
    stats      : dict with 'tic_r', 'warp_mag', 'study_sil'
    cors       : per-pair Pearson r list
    warp_mags  : per-pair warp magnitude list
    """
    b_tics  = [c.sum(axis=1) for c in chromas_b]
    cors, warp_mags = [], []
    n_total = len(chromas_a) * len(chromas_b)
    done = 0

    for qc in chromas_a:
        q_tic = qc.sum(axis=1)
        for rc, r_tic in zip(chromas_b, b_tics):
            warped, warp_mag, _ = align_fn(q_tic, r_tic)
            cor = float(np.corrcoef(r_tic, warped)[0, 1])
            cors.append(0.0 if np.isnan(cor) else cor)
            warp_mags.append(warp_mag)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{n_total} pairs …")

    study_sil = compute_silhouette(chromas_a, chromas_b, align_fn)

    return {
        'tic_r':     float(np.nanmean(cors)),
        'warp_mag':  float(np.nanmean(warp_mags)),
        'study_sil': study_sil,
    }, cors, warp_mags


# ── Main ─────────────────────────────────────────────────────────────────────

def main(study_a_dir: Path, study_b_dir: Path,
         label_a: str, label_b: str) -> None:

    print("Loading chromatograms …")
    chromas_a = load_chromas(study_a_dir / 'chroma')
    chromas_b = load_chromas(study_b_dir / 'chroma')
    n_pairs   = len(chromas_a) * len(chromas_b)
    print(f"  {label_a}: {len(chromas_a)} samples")
    print(f"  {label_b}: {len(chromas_b)} samples")
    print(f"  Total pairs  : {n_pairs}")
    print(f"  COW-TIC      : n_segs={COW_N_SEGS}  slack=±{COW_SLACK} bins "
          f"(±{COW_SLACK * BIN_MIN:.1f} min)")
    print(f"  icoshift     : n_intervals={ICO_N_INTERVALS}  "
          f"max_shift=±{ICO_MAX_SHIFT} bins (±{ICO_MAX_SHIFT * BIN_MIN:.1f} min)\n")

    # Unaligned baseline
    a_tics = [c.sum(axis=1) for c in chromas_a]
    b_tics = [c.sum(axis=1) for c in chromas_b]
    unaligned_cors = [
        float(np.corrcoef(r_tic, q_tic)[0, 1])
        for q_tic in a_tics
        for r_tic in b_tics
    ]
    unaligned_r = float(np.nanmean(unaligned_cors))
    print(f"Unaligned baseline  TIC r = {unaligned_r:.3f}\n")

    # Unaligned silhouette
    feats_unaligned = np.stack([c.max(axis=0) for c in chromas_a + chromas_b])
    labels_unaligned = np.array([0]*len(chromas_a) + [1]*len(chromas_b))
    n_comp = min(50, feats_unaligned.shape[0] - 1)
    pca_unaligned = PCA(n_components=n_comp).fit_transform(feats_unaligned)
    sil_unaligned = float(silhouette_score(pca_unaligned, labels_unaligned))

    methods: list[tuple[str, object]] = [
        ('COW-TIC',  lambda q, r: cow_tic_align(q, r)),
        ('icoshift', lambda q, r: icoshift_align(q, r)),
    ]

    results  = {}
    all_cors = {'Unaligned': unaligned_cors}

    for label, fn in methods:
        print(f"Evaluating: {label} …")
        stats, cors, _ = evaluate_pairwise(chromas_a, chromas_b, fn)
        results[label]  = stats
        all_cors[label] = cors
        print(f"  TIC r={stats['tic_r']:.3f}  warp_mag={stats['warp_mag']:.3f} min  "
              f"study_sil={stats['study_sil']:.3f}\n")

    # Summary table — matches warp_transformer_calib output format
    w = 34
    print("=" * 72)
    print(f"{'Method':<{w}} {'TIC r':>7}  {'Δ unaligned':>12}  {'Study sil':>10}")
    print("-" * 72)
    print(f"{'Unaligned':<{w}} {unaligned_r:>7.3f}  {'—':>12}  {sil_unaligned:>10.3f}")
    for lbl, r in results.items():
        delta = r['tic_r'] - unaligned_r
        print(f"{lbl:<{w}} {r['tic_r']:>7.3f}  {delta:>+12.3f}  "
              f"{r['study_sil']:>10.3f}")
    print("=" * 72)
    print(f"TIC r     : mean Pearson r across all {label_a}↔{label_b} pairs  (n={n_pairs})")
    print("Δ unaligned: improvement over unaligned baseline")
    print("Study sil : study-label silhouette in PCA-50  (lower = less batch effect)")

    # Wilcoxon signed-rank tests (vs Unaligned and vs each other)
    print(f"\n{'=' * 70}")
    print(f"Paired Wilcoxon signed-rank tests (n={n_pairs} pairs, two-sided)")
    print(f"{'-' * 70}")
    print(f"{'Comparison':<40}  {'stat':>10}  {'p-value':>12}  {'sig':>5}")
    print(f"{'-' * 70}")
    ordered = ['Unaligned'] + [lbl for lbl, _ in methods]
    comparisons = [(a, b) for i, a in enumerate(ordered) for b in ordered[i+1:]]
    for a, b in comparisons:
        ca = np.array(all_cors[a])
        cb = np.array(all_cors[b])
        stat, p = wilcoxon(ca, cb, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"  {b:<20} vs {a:<16}  {stat:>10.1f}  {p:>12.3e}  {sig:>5}")
    print(f"{'=' * 70}")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='mtbls21',
                    help='Study A dir relative to data/ (default: mtbls21 = wheat)')
    ap.add_argument('--study-b', default='mtbls288',
                    help='Study B dir relative to data/ (default: mtbls288 = rice)')
    args = ap.parse_args()

    label_a     = Path(args.study_a).name
    label_b     = Path(args.study_b).name
    study_a_dir = DATA_DIR / args.study_a
    study_b_dir = DATA_DIR / args.study_b

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f'literature_baselines_{label_a}_{label_b}.txt'
    with open(out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            main(study_a_dir, study_b_dir, label_a, label_b)
        finally:
            sys.stdout = orig
    print(f"\nResults saved → {out}")
