"""
Classical ML baselines for GC-MS fish species classification.

Runs SVM, Random Forest, KNN, and PLS-DA across all alignment conditions.
No CNN classifiers. This is the evaluation that matters for the primary
metabolomics use case: a peak-feature table fed into standard statistical
or ML classifiers — the workflow alignment was designed for.

Feature
-------
Per-m/z maximum projection: for each of the 1000 m/z channels, take the
maximum intensity observed across all 200 RT bins. This 1000-dim vector is
the standard input for classical metabolomics ML (same as classify_fish_oil.py).

Classifiers
-----------
  PLS-DA   — Partial Least Squares Discriminant Analysis (inner CV for n_components)
  RF       — Random Forest (100 trees)
  SVM      — RBF kernel, grid-searched C and gamma
  KNN      — k=5, distance-weighted, StandardScaler

Alignment methods
-----------------
  No alignment, Co-shift, icoshift, COW,
  m/z COW (cosine), m/z COW (drift encoder)

Evaluation
----------
5-fold stratified CV × 3 seeds = 15 runs per condition.
Metrics: balanced accuracy (primary), macro-F1 (secondary).
"""

from __future__ import annotations

import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, '/Users/woodj/Desktop/chroma-dcnn/src')

from encoder import DilatedSpectrumEncoder, SparseSpectrumTransformer
from chroma_dcnn.evaluation.baselines import baseline_cv

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
FISH_CHROMA = DATA_DIR / 'fish_oil' / 'chroma'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS  = 200
RUN_MIN = 45.0
TIME_AX = np.linspace(0, RUN_MIN, N_BINS)
BIN_MIN = RUN_MIN / N_BINS

CV_SEEDS = [0, 1, 2]
CV_FOLDS = 5

class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


# ── Alignment functions (from classify_fish_oil.py) ──────────────────────────

def _l2(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)

def _detect_peaks(chroma):
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, 80)
    pks, props = sp_find_peaks(tic, height=thr, distance=5)
    if len(pks) > 30:
        top = np.argsort(props['peak_heights'])[-30:]
        pks = pks[top]
    return np.sort(pks)


def _context_fp(chroma, pk):
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))

def _apply_int_shift(signal, lag):
    if lag == 0:
        return signal.copy()
    out = np.roll(signal, lag)
    if lag > 0:
        out[:lag] = signal[0]
    else:
        out[lag:] = signal[-1]
    return out

def _warp_chroma(query, warp_fn):
    query_times = np.clip(warp_fn(TIME_AX), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(query_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1 - a) * query[t0] + a * query[t1]
    return warped.astype(np.float32)

def align_coshift(query, ref, max_shift_bins=30):
    ref_tic = ref.sum(axis=1); q_tic = query.sum(axis=1)
    n = len(ref_tic)
    cc = np.correlate(ref_tic, q_tic, mode='full')
    lags = np.arange(-(n - 1), n)
    valid = np.abs(lags) <= max_shift_bins
    best_lag = int(lags[valid][np.argmax(cc[valid])])
    return np.stack([_apply_int_shift(query[:, mz], best_lag)
                     for mz in range(query.shape[1])], axis=1).astype(np.float32)

def align_icoshift(query, ref, n_intervals=10, max_shift_bins=15):
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

def align_cow(query, ref, n_segments=10, slack=10):
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

def align_mz_cow(query, ref, encode_fn=None, max_drift_min=6.0):
    ref_pks = _detect_peaks(ref)
    q_pks   = _detect_peaks(query)
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
    rf2 = np.concatenate([[0], ra, [RUN_MIN]])
    qf  = np.concatenate([[0], qa, [RUN_MIN]])
    warp = PchipInterpolator(rf2, qf)
    return _warp_chroma(query, warp)


# ── Feature extraction ────────────────────────────────────────────────────────

def max_proj(paths):
    """Per-m/z maximum across RT bins → (n_samples, 1000)."""
    return np.stack([np.load(p)['chroma'].astype(np.float32).max(axis=0)
                     for p in paths])


def build_aligned(chromas, npz_paths, align_fn, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = chromas[0]
    out_paths = []
    for i, (chroma, src) in enumerate(zip(chromas, npz_paths)):
        out = out_dir / src.name
        if not out.exists():
            aligned = align_fn(chroma, ref) if i > 0 else chroma.copy()
            np.savez_compressed(out, chroma=aligned)
        out_paths.append(out)
    return out_paths


# ── KNN cross-validation (not in baseline_cv) ────────────────────────────────

def knn_cv(X, y, n_neighbors=5, seeds=CV_SEEDS, n_folds=CV_FOLDS):
    ba_list, f1_list = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('knn',    KNeighborsClassifier(n_neighbors=n_neighbors,
                                                weights='distance',
                                                metric='euclidean')),
            ])
            pipe.fit(X[tr], y[tr])
            preds = pipe.predict(X[te])
            ba_list.append(balanced_accuracy_score(y[te], preds))
            f1_list.append(f1_score(y[te], preds, average='macro'))
    return {'balanced_accuracy': np.array(ba_list),
            'macro_f1':          np.array(f1_list)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    y         = np.load(DATA_DIR / 'fish_oil' / 'y.npy').astype(np.int64)
    chromas   = [np.load(p)['chroma'].astype(np.float32) for p in npz_paths]
    print(f"Loaded {len(chromas)} samples  "
          f"| classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    drift_enc_fn = None
    drift_ckpt   = CKPT_DIR / 'drift_simclr.pt'
    if drift_ckpt.exists():
        m = DilatedSpectrumEncoder().to('cpu'); m.eval()
        m.load_state_dict(torch.load(drift_ckpt, map_location='cpu'))
        def drift_enc_fn(x, rt=None):
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32'))
                r = torch.from_numpy(rt.astype('float32')) if rt is not None else None
                return m.encode(t, r).numpy()
        print(f"Drift encoder loaded: {drift_ckpt.name}")

    transformer_enc_fn = None
    transformer_ckpt   = CKPT_DIR / 'transformer_simclr.pt'
    if transformer_ckpt.exists():
        mt = SparseSpectrumTransformer().to('cpu'); mt.eval()
        mt.load_state_dict(torch.load(transformer_ckpt, map_location='cpu'))
        def transformer_enc_fn(x, rt=None):
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32'))
                r = torch.from_numpy(rt.astype('float32')) if rt is not None else None
                return mt.encode(t, r).numpy()
        print(f"Transformer encoder loaded: {transformer_ckpt.name}")

    classifiers = [
        ('PLS-DA', lambda X, y: baseline_cv(X, y, 'plsda',
                                             seeds=CV_SEEDS, cv_strategy='kfold',
                                             cv_folds=CV_FOLDS)),
        ('RF',     lambda X, y: baseline_cv(X, y, 'rf',
                                             seeds=CV_SEEDS, cv_strategy='kfold',
                                             cv_folds=CV_FOLDS)),
        ('SVM',    lambda X, y: baseline_cv(X, y, 'svm',
                                             seeds=CV_SEEDS, cv_strategy='kfold',
                                             cv_folds=CV_FOLDS)),
        ('KNN',    lambda X, y: knn_cv(X, y)),
    ]

    tmp_root = Path(tempfile.mkdtemp(prefix='baselines_'))
    try:
        align_methods = [('No alignment', npz_paths, None)]

        print("\nPre-computing aligned chromatograms …")
        named_fns = [
            ('Co-shift [Savorani 2010]',
             lambda q, r: align_coshift(q, r)),
            ('icoshift [Savorani 2010]',
             lambda q, r: align_icoshift(q, r)),
            ('COW [Nielsen 1998]',
             lambda q, r: align_cow(q, r)),
            ('m/z COW (cosine)',
             lambda q, r: align_mz_cow(q, r)),
            ('m/z COW (drift enc.)',
             lambda q, r: align_mz_cow(q, r, encode_fn=drift_enc_fn)),
            ('m/z COW (transformer)',
             lambda q, r: align_mz_cow(q, r, encode_fn=transformer_enc_fn)),
        ]
        for name, fn in named_fns:
            tag     = (name.split('[')[0].strip()
                       .replace(' ', '_').replace('/', '').replace('(', '')
                       .replace(')', '').replace('.', ''))
            out_dir = tmp_root / tag
            paths   = build_aligned(chromas, npz_paths, fn, out_dir)
            align_methods.append((name, paths, out_dir))
            print(f"  {name}: done")

        all_results = []

        for align_name, paths, _ in align_methods:
            print(f"\n{'='*60}")
            print(f"Alignment: {align_name}")
            print('='*60)
            X = max_proj(list(paths))

            for clf_name, clf_fn in classifiers:
                print(f"  {clf_name} …", end=' ', flush=True)
                res = clf_fn(X, y)
                ba = res['balanced_accuracy']; f1 = res['macro_f1']
                print(f"bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                      f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")
                all_results.append({
                    'align': align_name,
                    'clf':   clf_name,
                    'ba':    ba,
                    'f1':    f1,
                })

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    clf_names = [c[0] for c in classifiers]
    col_w = 16

    print('\n' + '=' * (30 + col_w * len(clf_names)))
    header = f"{'Alignment':<30}" + ''.join(f"{c:>{col_w}}" for c in clf_names)
    print(header)
    print(f"{'Balanced Accuracy (mean ± std)':<30}")
    print('-' * (30 + col_w * len(clf_names)))

    by_align = {}
    for r in all_results:
        by_align.setdefault(r['align'], {})[r['clf']] = r

    align_order = [name for name, _, _ in align_methods]

    for metric, key in [('Balanced Accuracy', 'ba'), ('Macro-F1', 'f1')]:
        if key == 'f1':
            print(f'\n{metric}:')
            print('-' * (30 + col_w * len(clf_names)))
        for align_name in align_order:
            row = by_align.get(align_name, {})
            cells = []
            for clf_name in clf_names:
                if clf_name in row:
                    vals = row[clf_name][key]
                    cells.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
                else:
                    cells.append('—')
            print(f"{align_name:<30}" + ''.join(f"{c:>{col_w}}" for c in cells))
        print('=' * (30 + col_w * len(clf_names)))

    print('=' * (30 + col_w * len(clf_names)))
    print(f"5-fold × {len(CV_SEEDS)} seeds = {CV_FOLDS * len(CV_SEEDS)} runs per condition")
    print("Feature: per-m/z max-projection (1000-dim)")
    print("Classes: 0=SNA  1=GUR  2=TAR  3=BCO")
    print(f"KNN: k=5, distance-weighted  |  SVM: RBF, grid-searched C/gamma")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'baselines.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
