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
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import DilatedSpectrumEncoder, SparseSpectrumTransformer

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
    if len(pks) > 30:
        top = np.argsort(props['peak_heights'])[-30:]
        pks = pks[top]
    return np.sort(pks)


def _context_fp(chroma: np.ndarray, pk: int) -> np.ndarray:
    """Mean m/z spectrum over [pk-1, pk, pk+1], L2-normalised."""
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))


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
                 encode_fn=None,
                 max_drift_min: float = 6.0) -> np.ndarray:
    ref_pks = _detect_peaks(ref)
    q_pks   = _detect_peaks(query)
    # Context fingerprints: ±1-bin average reduces single-bin noise
    ref_fps = np.array([_context_fp(ref,   pk) for pk in ref_pks])
    q_fps   = np.array([_context_fp(query, pk) for pk in q_pks])
    # RT-aware encoding: pass normalised peak positions alongside spectra
    ref_rts = ref_pks.astype(np.float32) / N_BINS
    q_rts   = q_pks.astype(np.float32) / N_BINS
    if encode_fn:
        ref_rep = encode_fn(ref_fps, ref_rts)
        q_rep   = encode_fn(q_fps,   q_rts)
    else:
        ref_rep, q_rep = ref_fps, q_fps
    ref_n = np.linalg.norm(ref_rep, axis=1, keepdims=True) + 1e-8
    q_n   = np.linalg.norm(q_rep,   axis=1, keepdims=True) + 1e-8
    sim   = (q_rep / q_n) @ (ref_rep / ref_n).T
    ref_t = ref_pks * BIN_MIN
    q_t   = q_pks   * BIN_MIN
    dm    = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sc    = sim.copy()
    sc[dm] = -1.0
    ri, ci = linear_sum_assignment(-sc)
    # Adaptive threshold: top-30% of within-window similarities
    avail = sim[~dm].flatten()
    threshold = max(float(np.percentile(avail, 70)), 0.30) if len(avail) >= 3 else 0.35
    keep   = (sim[ri, ci] >= threshold) & ~dm[ri, ci]
    ri, ci = ri[keep], ci[keep]
    if len(ri) < 2:
        return query.copy()
    # Fine-tune each anchor: search ±3 bins for best spectral match
    refined_q = []
    for idx in range(len(ri)):
        r_spec = ref_fps[ci[idx]]
        q_pk   = q_pks[ri[idx]]
        best_bin, best_s = q_pk, np.dot(r_spec, _context_fp(query, q_pk))
        for delta in range(-3, 4):
            if delta == 0:
                continue
            cand = int(np.clip(q_pk + delta, 0, N_BINS - 1))
            s = np.dot(r_spec, _context_fp(query, cand))
            if s > best_s:
                best_s = s
                best_bin = cand
        refined_q.append(best_bin)
    refined_q = np.array(refined_q)
    order = np.argsort(ci)
    ri, ci, refined_q = ri[order], ci[order], refined_q[order]
    ra = TIME_AX[ref_pks[ci]]
    qa = TIME_AX[refined_q]
    # RANSAC: reject anchors that deviate from the main linear RT drift trend
    if len(ra) >= 4:
        from sklearn.linear_model import RANSACRegressor
        try:
            ransac = RANSACRegressor(residual_threshold=3 * BIN_MIN,
                                     min_samples=2, random_state=42)
            ransac.fit(qa.reshape(-1, 1), ra)
            if ransac.inlier_mask_.sum() >= 2:
                ra = ra[ransac.inlier_mask_]
                qa = qa[ransac.inlier_mask_]
        except Exception:
            pass
    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]:
            mono.append(i)
    ra, qa = ra[mono], qa[mono]
    rf = np.concatenate([[0], ra, [RUN_MIN]])
    qf = np.concatenate([[0], qa, [RUN_MIN]])
    # PCHIP: monotone cubic interpolation — smooth warp, no kinks at anchor points
    warp = PchipInterpolator(rf, qf)
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
        m = DilatedSpectrumEncoder().to('cpu')
        m.load_state_dict(torch.load(drift_ckpt, map_location='cpu'))
        m.eval()
        def drift_enc_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32'))
                r = torch.from_numpy(rt.astype('float32')) if rt is not None else None
                return m.encode(t, r).numpy()
        print(f"Drift encoder loaded from {drift_ckpt.name}")

    transformer_enc_fn = None
    transformer_ckpt   = CKPT_DIR / 'transformer_simclr.pt'
    if transformer_ckpt.exists():
        mt = SparseSpectrumTransformer().to('cpu')
        mt.load_state_dict(torch.load(transformer_ckpt, map_location='cpu'))
        mt.eval()
        def transformer_enc_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32'))
                r = torch.from_numpy(rt.astype('float32')) if rt is not None else None
                return mt.encode(t, r).numpy()
        print(f"Transformer encoder loaded from {transformer_ckpt.name}")

    align_methods = [
        ('No alignment',              None),
        ('Co-shift',                  align_coshift),
        ('icoshift',                  align_icoshift),
        ('COW',                       align_cow),
        ('m/z COW (cosine)',          lambda q, r: align_mz_cow(q, r)),
        ('m/z COW (drift enc.)',      lambda q, r: align_mz_cow(q, r, encode_fn=drift_enc_fn)),
        ('m/z COW (transformer)',     lambda q, r: align_mz_cow(q, r, encode_fn=transformer_enc_fn)),
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
    print(f"\n── Per-class median CV  [No alignment vs Co-shift] ────────────────────")
    for name, align_fn in [align_methods[0], align_methods[1]]:
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
