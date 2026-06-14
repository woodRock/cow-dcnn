"""
Compound Identification: RT + m/z Evidence Quality.

In a GC-MS metabolomics workflow, compound identification uses two orthogonal
evidence sources: the m/z fragmentation fingerprint and the retention time.
If RT is unreliable (due to drift), adding it as co-evidence *hurts* identifi-
cation: a shifted true compound scores below a false compound that happens to
sit at the expected RT. Alignment should make RT reliable, restoring its value.

Protocol
--------
Reference library: sample[0] (used as the reference run for all alignments).
Query set:         all 102 remaining fish_oil samples.

Ground truth: for each reference peak and each query sample, the query peak
with the highest m/z cosine similarity (≥ GT_THRESHOLD) within a ±GT_MAX_DRIFT
bin window is assumed to be the same compound — established from the raw,
unaligned chromatograms. The time window excludes spurious m/z matches across
compounds that happen to share fragmentation patterns but elute at very
different retention times (a real library search would impose the same constraint).
The GT pair set is fixed across all alignment conditions.

For each alignment method, align the query chromatogram and measure:

  Metric 1 — Mean RT deviation (bins / minutes) for GT compound pairs.
              Lower = the aligned peaks land closer to the reference peak,
              making RT a more reliable identification cue.

  Metric 2 — Combined-score rank-1 hit rate.
              For each reference peak, rank all query peaks by:
                combined = ALPHA * mz_cosine + (1-ALPHA) * rt_similarity
              where rt_similarity = max(0, 1 − |Δt| / MAX_RT_DEV_BINS).
              Hit = combined rank-1 agrees with the m/z-only GT.
              Before alignment (72.3 %): noisy RT causes combined to misrank
              true matches. After alignment: RT is reliable; hit rate rises.

Dataset:    fish_oil, 103 samples, 4 fish species
Reference:  sample[0]
Queries:    samples 1–102
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

N_BINS  = 200
RUN_MIN = 45.0
TIME_AX = np.linspace(0, RUN_MIN, N_BINS)
BIN_MIN = RUN_MIN / N_BINS   # 0.225 min per bin

GT_THRESHOLD     = 0.70   # m/z cosine threshold for same-compound ground truth
GT_MAX_DRIFT     = 30     # max |ΔRT| bins allowed for GT (excludes false m/z matches)
ALPHA            = 0.50   # weight on m/z vs RT in combined score
MAX_RT_DEV_BINS  = 20     # RT deviation at which rt_similarity → 0

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


def _fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    """Single-bin fingerprints — used for ground-truth matching (fixed across conditions)."""
    return np.array([_l2(chroma[pk]) for pk in pks])


def _context_fp(chroma: np.ndarray, pk: int) -> np.ndarray:
    """Mean m/z spectrum over [pk-1, pk, pk+1], L2-normalised — used for alignment anchors."""
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
    # Context fingerprints for alignment (distinct from single-bin GT fingerprints)
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
    warp = PchipInterpolator(rf, qf)
    return _warp_chroma(query, warp)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_sample(chroma_ref: np.ndarray,
                chroma_query: np.ndarray,
                ref_pks: np.ndarray,
                ref_fps: np.ndarray,
                gt_threshold: float = GT_THRESHOLD,
                gt_max_drift: int   = GT_MAX_DRIFT,
                alpha: float        = ALPHA,
                max_rt_dev: int     = MAX_RT_DEV_BINS,
                ) -> dict:
    """
    Evaluate one query chromatogram against the reference library.

    ref_pks / ref_fps are pre-computed from chroma_ref for efficiency.
    Ground truth: per-reference-peak, argmax m/z cosine within ±gt_max_drift
    bins, if that cosine ≥ gt_threshold. The time window excludes spurious m/z
    matches at implausibly different retention times.

    Returns: rt_devs (array), n_pairs, n_comb_hits
    """
    query_pks = _detect_peaks(chroma_query)
    if len(query_pks) == 0:
        return dict(rt_devs=np.array([]), n_pairs=0, n_comb_hits=0)

    query_fps = _fingerprints(chroma_query, query_pks)
    sim       = ref_fps @ query_fps.T   # (n_ref × n_query)

    # One-to-one GT matching: Hungarian assignment within time window
    dt_bins = np.abs(ref_pks[:, None].astype(float)
                     - query_pks[None, :].astype(float))
    sim_windowed = sim.copy()
    sim_windowed[dt_bins > gt_max_drift] = -1.0
    ri, ci = linear_sum_assignment(-sim_windowed)
    # Keep only confident one-to-one pairs
    keep = sim[ri, ci] >= gt_threshold
    ri_gt, ci_gt = ri[keep], ci[keep]

    if len(ri_gt) == 0:
        return dict(rt_devs=np.array([]), n_pairs=0, n_comb_hits=0)

    rt_devs = np.abs(
        ref_pks[ri_gt].astype(float) - query_pks[ci_gt].astype(float)
    )

    # Combined-score rank-1 hit rate
    # For each GT ref peak, does rank-1 by combined score agree with GT?
    rt_sim   = np.maximum(0.0, 1.0 - dt_bins / max_rt_dev)
    combined = alpha * sim + (1.0 - alpha) * rt_sim
    comb_ci  = np.argmax(combined, axis=1)

    n_comb_hits = int(np.sum(comb_ci[ri_gt] == ci_gt))

    # Override gt_ci for downstream consistency (only used for n_pairs/n_comb_hits)
    gt_ci = np.full(len(ref_pks), -1, dtype=int)
    gt_ci[ri_gt] = ci_gt
    confident = gt_ci >= 0

    return dict(rt_devs=rt_devs, n_pairs=int(confident.sum()),
                n_comb_hits=n_comb_hits)


def eval_all_samples(chromas: list[np.ndarray],
                     align_fn,
                     ref_pks: np.ndarray,
                     ref_fps: np.ndarray) -> dict:
    """
    Align and evaluate all query samples against the reference.
    align_fn(query, ref) → aligned_query; None = no alignment.
    Returns aggregated metrics.
    """
    ref = chromas[0]
    all_devs    = []
    total_pairs = 0
    total_hits  = 0

    for i in range(1, len(chromas)):
        query    = chromas[i]
        aligned  = query if align_fn is None else align_fn(query, ref)
        result   = eval_sample(ref, aligned, ref_pks, ref_fps)
        all_devs.extend(result['rt_devs'].tolist())
        total_pairs += result['n_pairs']
        total_hits  += result['n_comb_hits']

    all_devs = np.array(all_devs)
    mean_dev = float(np.mean(all_devs)) if len(all_devs) else np.nan
    hit_rate = total_hits / total_pairs if total_pairs else np.nan

    return dict(
        n_pairs      = total_pairs,
        mean_rt_bins = mean_dev,
        mean_rt_min  = mean_dev * BIN_MIN if not np.isnan(mean_dev) else np.nan,
        hit_rate_comb= hit_rate,
        rt_devs      = all_devs,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    chromas   = [np.load(p)['chroma'].astype(np.float32) for p in npz_paths]
    print(f"Loaded {len(chromas)} chromatograms  (reference: sample[0], "
          f"queries: samples 1–{len(chromas)-1})")
    print(f"GT threshold: m/z cosine ≥ {GT_THRESHOLD}  |  "
          f"Combined score: {ALPHA:.2f}×mz + {1-ALPHA:.2f}×rt_sim  |  "
          f"Max RT dev: {MAX_RT_DEV_BINS} bins ({MAX_RT_DEV_BINS * BIN_MIN:.2f} min)")

    # Pre-compute reference peaks / fingerprints once
    ref     = chromas[0]
    ref_pks = _detect_peaks(ref)
    ref_fps = _fingerprints(ref, ref_pks)
    print(f"Reference peaks: {len(ref_pks)}")

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
        ('No alignment',           None),
        ('Co-shift',               align_coshift),
        ('icoshift',               align_icoshift),
        ('COW',                    align_cow),
        ('m/z COW (cosine)',       lambda q, r: align_mz_cow(q, r)),
        ('m/z COW (drift enc.)',   lambda q, r: align_mz_cow(q, r, encode_fn=drift_enc_fn)),
        ('m/z COW (transformer)',  lambda q, r: align_mz_cow(q, r, encode_fn=transformer_enc_fn)),
    ]

    results = []
    print(f"\nEvaluating {len(align_methods)} conditions on {len(chromas)-1} query samples …")
    for name, align_fn in align_methods:
        print(f"  {name} …", end=' ', flush=True)
        res = eval_all_samples(chromas, align_fn, ref_pks, ref_fps)
        results.append({'name': name, **res})
        print(f"mean RT dev = {res['mean_rt_bins']:.2f}b  "
              f"comb hit = {res['hit_rate_comb']*100:.1f}%  "
              f"n = {res['n_pairs']}")

    # RT deviation distribution (≤5 / 6-15 / >15 bins) for each method
    print(f"\n── RT deviation distribution ─────────────────────────────────────────────")
    print(f"{'Method':<28}  {'0-5b':>5}  {'6-15b':>5}  {'>15b':>5}  "
          f"{'n total':>8}")
    print('-' * 60)
    for r in results:
        devs = r['rt_devs']
        n = len(devs)
        if n == 0:
            print(f"{r['name']:<28}  {'n/a':>5}")
            continue
        small  = int(np.sum(devs <= 5))
        medium = int(np.sum((devs > 5) & (devs <= 15)))
        large  = int(np.sum(devs > 15))
        print(f"{r['name']:<28}  {small:>5}  {medium:>5}  {large:>5}  {n:>8}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"{'Method':<28}  {'Mean RT dev (bins)':>18}  {'Mean RT dev (min)':>17}  "
          f"{'Comb hit%':>10}")
    print(f"{'-'*72}")
    no_align = results[0]
    for r in results:
        d_bins = r['mean_rt_bins']
        d_min  = r['mean_rt_min']
        hit    = r['hit_rate_comb'] * 100
        if r['name'] != 'No alignment':
            delta = d_bins - no_align['mean_rt_bins']
            delta_str = f" ({delta:+.2f})"
        else:
            delta_str = ""
        print(f"{r['name']:<28}  {d_bins:>15.2f}{delta_str:>5}  "
              f"{d_min:>16.3f}  {hit:>9.1f}%")
    print(f"{'='*72}")
    print(f"Ground truth: {no_align['n_pairs']} compound pairs  "
          f"(m/z cosine ≥ {GT_THRESHOLD}, best-match per ref peak,  "
          f"fixed across all conditions)")
    print(f"Combined hit%: fraction where rank-1 by combined score agrees with "
          f"m/z-only ground truth")
    print(f"  Before alignment: RT drift penalises true matches → hit rate = "
          f"{no_align['hit_rate_comb']*100:.1f}%")
    print(f"  Improvement = alignment makes RT evidence reliable for compound ID")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'library_matching.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
