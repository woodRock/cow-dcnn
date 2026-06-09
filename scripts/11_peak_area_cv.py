"""
Peak Area Coefficient of Variation (CV) Analysis.

Evaluates whether alignment reduces within-class peak area variability —
the primary use case alignment was designed for in the metabolomics literature.
A standard metabolomics workflow integrates peak areas to build a compound ×
sample abundance table; RT drift corrupts this step before any classifier
is involved.

Protocol
--------
For each alignment method:
  1. Align all 103 fish_oil chromatograms to sample[0] as reference.
  2. Detect consensus peaks in the reference TIC (top-15 by height, ≥80th pct).
  3. At each consensus peak, define an integration window of ±WINDOW_BINS bins.
  4. For each sample, compute the integrated TIC area within that window.
  5. Within each class (fish species), compute the coefficient of variation
     CV = (σ / μ) × 100 % for each peak.
  6. Report: median CV, mean CV, fraction of peaks with CV < 15 % (standard
     metabolomics QC threshold for good reproducibility).

Datasets: fish_oil (103 samples, 4 classes: SNA, GUR, TAR, BCO)
Methods:  No alignment, Co-shift, icoshift, COW,
          m/z COW (cosine), m/z COW (drift encoder)
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import SpectrumEncoder

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
FISH_CHROMA = DATA_DIR / 'fish_oil' / 'chroma'

N_BINS   = 200
RUN_MIN  = 45.0
TIME_AX  = np.linspace(0, RUN_MIN, N_BINS)
BIN_MIN  = RUN_MIN / N_BINS          # 0.225 min per bin

WINDOW_BINS      = 4                 # integration half-window (±4 bins ≈ ±0.9 min)
CV_GOOD_PCT      = 15.0             # CV < 15 % = "good reproducibility" QC threshold
CLASS_NAMES      = {0: 'SNA', 1: 'GUR', 2: 'TAR', 3: 'BCO'}

RESULTS_DIR = Path(__file__).parent.parent / 'results'

class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


# ── Alignment primitives (from classify_fish_oil.py) ─────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _detect_peaks(chroma: np.ndarray) -> np.ndarray:
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, 80)
    pks, props = sp_find_peaks(tic, height=thr, distance=5)
    if len(pks) > 15:
        top = np.argsort(props['peak_heights'])[-15:]
        pks = pks[top]
    return np.sort(pks)


def _apply_int_shift(signal: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return signal.copy()
    out = np.roll(signal, lag)
    if lag > 0:
        out[:lag] = signal[0]
    else:
        out[lag:] = signal[-1]
    return out


def _warp_chroma(query: np.ndarray, warp_fn) -> np.ndarray:
    query_times = np.clip(warp_fn(TIME_AX), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(query_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1 - a) * query[t0] + a * query[t1]
    return warped.astype(np.float32)


def align_coshift(query: np.ndarray, ref: np.ndarray,
                  max_shift_bins: int = 30) -> np.ndarray:
    ref_tic = ref.sum(axis=1); q_tic = query.sum(axis=1)
    n = len(ref_tic)
    cc = np.correlate(ref_tic, q_tic, mode='full')
    lags = np.arange(-(n - 1), n)
    valid = np.abs(lags) <= max_shift_bins
    best_lag = int(lags[valid][np.argmax(cc[valid])])
    return np.stack([_apply_int_shift(query[:, mz], best_lag)
                     for mz in range(query.shape[1])], axis=1).astype(np.float32)


def align_icoshift(query: np.ndarray, ref: np.ndarray,
                   n_intervals: int = 10, max_shift_bins: int = 15) -> np.ndarray:
    ref_tic = ref.sum(axis=1); q_tic = query.sum(axis=1)
    seg_len = N_BINS // n_intervals
    warped  = np.empty_like(query)
    for s in range(n_intervals):
        lo = s * seg_len; hi = min(lo + seg_len, N_BINS)
        pad   = max_shift_bins
        ref_p = np.pad(ref_tic[lo:hi], pad); q_p = np.pad(q_tic[lo:hi], pad)
        cc = np.correlate(ref_p, q_p, mode='full')
        n  = len(ref_p); lags = np.arange(-(n - 1), n)
        valid = np.abs(lags) <= pad
        best_lag = int(lags[valid][np.argmax(cc[valid])]) if valid.any() else 0
        for mz in range(query.shape[1]):
            warped[lo:hi, mz] = _apply_int_shift(query[lo:hi, mz], -best_lag)
    return warped.astype(np.float32)


def align_cow(query: np.ndarray, ref: np.ndarray,
              n_segments: int = 10, slack: int = 10) -> np.ndarray:
    ref_tic = ref.sum(axis=1); q_tic = query.sum(axis=1)
    M, T, N = N_BINS, N_BINS // n_segments, n_segments

    def seg_corr(r_seg, q_start, q_len):
        if q_start + q_len > M or q_len < 2: return -np.inf
        q_seg = q_tic[q_start:q_start + q_len]
        if len(q_seg) != len(r_seg):
            q_seg = np.interp(np.linspace(0, 1, len(r_seg)),
                              np.linspace(0, 1, len(q_seg)), q_seg)
        sr, sq = np.std(r_seg), np.std(q_seg)
        if sr < 1e-8 or sq < 1e-8: return 0.0
        return float(np.corrcoef(r_seg, q_seg)[0, 1])

    ref_segs = [ref_tic[i * T:(i + 1) * T] for i in range(N)]
    INF = -1e9
    dp_score = np.full(M + 1, INF); dp_score[0] = 0.0
    all_prev = []
    for i in range(N):
        new_score = np.full(M + 1, INF); new_prev = np.full(M + 1, -1, dtype=int)
        for b_prev in range(M + 1):
            if dp_score[b_prev] == INF: continue
            for dq in range(max(1, T - slack), T + slack + 1):
                b_next = b_prev + dq
                if b_next > M: break
                if i == N - 1 and b_next != M: continue
                sc = dp_score[b_prev] + seg_corr(ref_segs[i], b_prev, dq)
                if sc > new_score[b_next]:
                    new_score[b_next] = sc; new_prev[b_next] = b_prev
        dp_score = new_score; all_prev.append(new_prev)

    boundaries = [M]; pos = M
    for layer in reversed(all_prev):
        prev = layer[pos]
        if prev < 0: return query.copy()
        boundaries.append(prev); pos = prev
    boundaries = list(reversed(boundaries))
    ref_pts = [i * T * BIN_MIN for i in range(N + 1)]
    q_pts   = [b * BIN_MIN     for b in boundaries]
    for i in range(1, len(q_pts)):
        if q_pts[i] <= q_pts[i - 1]:
            q_pts[i] = q_pts[i - 1] + 0.01
    warp = interp1d(ref_pts, q_pts, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp)


def align_mz_cow(query: np.ndarray, ref: np.ndarray,
                 encode_fn=None, sim_threshold: float = 0.5,
                 max_drift_min: float = 6.0) -> np.ndarray:
    ref_pks = _detect_peaks(ref);  q_pks = _detect_peaks(query)
    ref_fps = np.array([_l2(ref[pk])   for pk in ref_pks])
    q_fps   = np.array([_l2(query[pk]) for pk in q_pks])
    ref_rep = encode_fn(ref_fps) if encode_fn else ref_fps
    q_rep   = encode_fn(q_fps)   if encode_fn else q_fps
    ref_n = np.linalg.norm(ref_rep, axis=1, keepdims=True) + 1e-8
    q_n   = np.linalg.norm(q_rep,   axis=1, keepdims=True) + 1e-8
    sim   = (q_rep / q_n) @ (ref_rep / ref_n).T
    ref_t = ref_pks * BIN_MIN;  q_t = q_pks * BIN_MIN
    dm    = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sc    = sim.copy();  sc[dm] = -1.0
    ri, ci = linear_sum_assignment(-sc)
    keep   = (sim[ri, ci] >= sim_threshold) & ~dm[ri, ci]
    ri, ci = ri[keep], ci[keep]
    if len(ri) < 2: return query.copy()
    order = np.argsort(ci);  ri, ci = ri[order], ci[order]
    ra = TIME_AX[ref_pks[ci]];  qa = TIME_AX[q_pks[ri]]
    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]: mono.append(i)
    ra, qa = ra[mono], qa[mono]
    rf = np.concatenate([[0], ra, [RUN_MIN]])
    qf = np.concatenate([[0], qa, [RUN_MIN]])
    warp = interp1d(rf, qf, kind='linear', bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp)


# ── Peak area integration ─────────────────────────────────────────────────────

def integrated_areas(chromas: list[np.ndarray],
                     ref_peaks: np.ndarray,
                     window: int = WINDOW_BINS) -> np.ndarray:
    """
    Integrate TIC within ±window bins of each reference peak.
    Returns array of shape (n_samples, n_peaks).
    """
    areas = np.zeros((len(chromas), len(ref_peaks)), dtype=np.float64)
    for i, chroma in enumerate(chromas):
        tic = chroma.sum(axis=1).astype(np.float64)
        for j, pk in enumerate(ref_peaks):
            lo = max(0, pk - window)
            hi = min(N_BINS, pk + window + 1)
            areas[i, j] = np.trapezoid(tic[lo:hi])
    return areas


def within_class_cv(areas: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-peak CV within each class, pooled across classes.
    Returns a 1-D array of CV values (NaN for peaks with zero mean in any class).
    """
    all_cvs = []
    for cls in np.unique(y):
        mask     = y == cls
        cls_area = areas[mask]        # (n_in_class, n_peaks)
        means    = cls_area.mean(axis=0)
        stds     = cls_area.std(axis=0)
        valid    = means > 1e-8
        cv = np.where(valid, stds / means * 100.0, np.nan)
        all_cvs.append(cv)
    return np.concatenate(all_cvs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    y         = np.load(DATA_DIR / 'fish_oil' / 'y.npy').astype(np.int64)
    print(f"Loaded {len(npz_paths)} samples  "
          f"| classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    chromas = [np.load(p)['chroma'].astype(np.float32) for p in npz_paths]
    ref     = chromas[0]
    ref_peaks = _detect_peaks(ref)
    print(f"Reference peaks: {len(ref_peaks)}  "
          f"at {ref_peaks * BIN_MIN} min")
    print(f"Integration window: ±{WINDOW_BINS} bins (±{WINDOW_BINS * BIN_MIN:.2f} min)")

    # Drift encoder
    drift_enc_fn = None
    drift_ckpt   = CKPT_DIR / 'drift_simclr.pt'
    if drift_ckpt.exists():
        m = SpectrumEncoder().to('cpu')
        m.load_state_dict(torch.load(drift_ckpt, map_location='cpu'))
        m.eval()
        def drift_enc_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return m.encode(torch.from_numpy(x.astype('float32'))).numpy()
        print(f"Drift encoder loaded from {drift_ckpt.name}")

    align_methods = [
        ('No alignment',           None),
        ('Co-shift',               align_coshift),
        ('icoshift',               align_icoshift),
        ('COW',                    align_cow),
        ('m/z COW (cosine)',       lambda q, r: align_mz_cow(q, r)),
        ('m/z COW (drift enc.)',   lambda q, r: align_mz_cow(q, r, encode_fn=drift_enc_fn)),
    ]

    results = []
    for name, align_fn in align_methods:
        print(f"\nEvaluating: {name} …")
        if align_fn is None:
            aligned = chromas
        else:
            aligned = [ref] + [align_fn(c, ref) for c in chromas[1:]]

        areas = integrated_areas(aligned, ref_peaks)
        cvs   = within_class_cv(areas, y)
        valid = cvs[~np.isnan(cvs)]

        if len(valid) == 0:
            print("  WARNING: no valid CV values")
            results.append({'name': name, 'median': np.nan, 'mean': np.nan, 'good_pct': np.nan})
            continue

        median_cv  = float(np.median(valid))
        mean_cv    = float(np.mean(valid))
        good_pct   = float(np.mean(valid < CV_GOOD_PCT) * 100)

        print(f"  Median CV: {median_cv:.1f}%   "
              f"Mean CV: {mean_cv:.1f}%   "
              f"Peaks with CV < {CV_GOOD_PCT:.0f}%: {good_pct:.1f}%")
        results.append({'name': name, 'median': median_cv,
                        'mean': mean_cv, 'good_pct': good_pct})

    # Per-class breakdown for reference (no alignment vs best method)
    print(f"\n── Per-class median CV  [No alignment vs best method] ────────────────────")
    for name, align_fn in [align_methods[0], align_methods[-1]]:
        if align_fn is None:
            aligned = chromas
        else:
            aligned = [ref] + [align_fn(c, ref) for c in chromas[1:]]
        areas = integrated_areas(aligned, ref_peaks)
        row = []
        for cls in sorted(np.unique(y)):
            mask = y == cls
            cls_area = areas[mask]
            means = cls_area.mean(axis=0)
            stds  = cls_area.std(axis=0)
            valid = means > 1e-8
            cv = np.where(valid, stds / means * 100.0, np.nan)
            cv_valid = cv[~np.isnan(cv)]
            row.append(f"{CLASS_NAMES[cls]}={np.median(cv_valid):.1f}%"
                       if len(cv_valid) else f"{CLASS_NAMES[cls]}=n/a")
        print(f"  {name:<28}  {',  '.join(row)}")

    print('\n' + '=' * 72)
    print(f"{'Method':<28}  {'Median CV':>10}  {'Mean CV':>9}  "
          f"{'CV < {:.0f}%'.format(CV_GOOD_PCT):>10}")
    print('-' * 72)
    no_align_median = results[0]['median']
    for r in results:
        delta = (r['median'] - no_align_median) if r['median'] is not np.nan else np.nan
        delta_str = f" ({delta:+.1f})" if not np.isnan(delta) and r['name'] != 'No alignment' else ""
        print(f"{r['name']:<28}  {r['median']:>8.1f}%{delta_str:>8}  "
              f"{r['mean']:>8.1f}%  {r['good_pct']:>9.1f}%")
    print('=' * 72)
    print(f"Integration window: ±{WINDOW_BINS} bins (±{WINDOW_BINS * BIN_MIN:.2f} min)  |  "
          f"{len(ref_peaks)} reference peaks")
    print(f"CV pooled across {len(np.unique(y))} classes: "
          f"{',  '.join(f'{v}={k}' for k, v in CLASS_NAMES.items())}")
    print(f"CV < {CV_GOOD_PCT:.0f}% = metabolomics QC threshold for good reproducibility")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'peak_area_cv.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
