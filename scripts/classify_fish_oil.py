"""
Does explicit retention time alignment improve downstream fish species classification
when a pretrained chromatogram encoder is available?

Alignment methods
─────────────────
  Co-shift (global)    FFT whole-spectrum co-shift [Savorani 2010]
  icoshift             Interval-based FFT co-shift, 10 intervals [Savorani 2010]
  COW                  Correlation Optimised Warping, DP [Nielsen 1998]
  m/z COW (cosine)     COW anchored by m/z fingerprint cosine similarity
  m/z COW (drift enc.) COW anchored by drift encoder embeddings

Classifiers
───────────
  ChromatogramCNN  from_scratch    — dilated ResBlock 1D CNN, random init
  ChromatogramCNN  chroma_pretrain — same arch, next-frame-prediction pretraining
  PLS-DA                           — classical metabolomics baseline
  Random Forest                    — tree ensemble on per-m/z max-projection

All CNN conditions use ChromatogramCNN from chroma-dcnn so results are directly
comparable with the chroma-dcnn paper. Classical baselines use the per-m/z
max-projection of the aligned chromatogram.

Evaluation: 5-fold stratified CV × 3 seeds = 15 runs per condition.
Metric: balanced accuracy.

References
──────────
Savorani F, Tomasi G, Engelsen SB (2010) icoshift: A versatile tool for the
  rapid alignment of 1D NMR spectra. J Magn Reson 202:190-202.
Nielsen N-PV, Carstensen JM, Smedsgaard J (1998) Aligning of single and multiple
  wavelength chromatographic profiles for chemometric data analysis using
  correlation optimised warping. J Chromatogr A 805:17-35.
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, '/Users/woodj/Desktop/chroma-dcnn/src')

from encoder import DilatedSpectrumEncoder
from chroma_dcnn.models.chroma_cnn import ChromaCNNConfig
from chroma_dcnn.training.finetune_chroma import ChromaFinetuner
from chroma_dcnn.evaluation.baselines import baseline_cv

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
CHROMA_CKPT = Path('/Users/woodj/Desktop/chroma-dcnn/checkpoints/chroma_pretrain/best.pt')
FISH_CHROMA = DATA_DIR / 'fish_oil' / 'chroma'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

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

CV_CFG = {
    'model': {
        'mz_max': 1000,
        'cnn_channels': 128,
        'kernel_size': 7,
        'dropout': 0.3,
    },
    'task': {
        'num_classes': 4,
        'cv_folds': 5,
        'cv_seeds': [0, 1, 2],
        'cv_strategy': 'kfold',
    },
    'finetuning': {
        'epochs': 100,
        'batch_size': 16,
        'lr': 1e-3,
        'lr_scratch': 3e-3,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'early_stopping_patience': 20,
    },
    'pretrained_checkpoints': {
        'chroma_pretrain': str(CHROMA_CKPT) if CHROMA_CKPT.exists() else None,
    },
}


# ── Shared utilities ──────────────────────────────────────────────────────────

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
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))


def _warp_chroma(query: np.ndarray, warp_fn) -> np.ndarray:
    """Apply a time-warp function to a full 2D chromatogram (N_BINS × mz)."""
    query_times = np.clip(warp_fn(TIME_AX), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(query_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1 - a) * query[t0] + a * query[t1]
    return warped.astype(np.float32)


def _apply_int_shift(signal: np.ndarray, lag: int) -> np.ndarray:
    """Integer-sample shift; edges filled with nearest value."""
    if lag == 0:
        return signal.copy()
    out = np.roll(signal, lag)
    if lag > 0:
        out[:lag] = signal[0]
    else:
        out[lag:] = signal[-1]
    return out


# ── Published method 1: Co-shift (global, Savorani et al. 2010) ──────────────

def align_coshift(query: np.ndarray, ref: np.ndarray,
                  max_shift_bins: int = 30) -> np.ndarray:
    """
    Whole-spectrum co-shift via FFT cross-correlation.

    Special case of icoshift (Savorani et al., 2010) with inter='whole'.
    Finds the single integer lag that maximises TIC cross-correlation,
    then applies the same shift uniformly to all m/z channels.
    """
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)
    n       = len(ref_tic)

    cc   = np.correlate(ref_tic, q_tic, mode='full')
    lags = np.arange(-(n - 1), n)
    valid = np.abs(lags) <= max_shift_bins
    best_lag = int(lags[valid][np.argmax(cc[valid])])

    warped = np.stack([_apply_int_shift(query[:, mz], best_lag)
                       for mz in range(query.shape[1])], axis=1)
    return warped.astype(np.float32)


# ── Published method 2: icoshift (interval, Savorani et al. 2010) ────────────

def align_icoshift(query: np.ndarray, ref: np.ndarray,
                   n_intervals: int = 10,
                   max_shift_bins: int = 15) -> np.ndarray:
    """
    Interval-based co-shift (icoshift, Savorani et al., 2010).

    Divides the TIC into n_intervals equal segments. For each segment,
    finds the integer shift that maximises cross-correlation with the
    reference segment. Each interval of the 2D chromatogram is shifted
    independently by that lag, with edge values used to fill the gaps.
    """
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)
    seg_len = N_BINS // n_intervals
    warped  = np.empty_like(query)

    for s in range(n_intervals):
        lo = s * seg_len
        hi = min(lo + seg_len, N_BINS)

        ref_seg = ref_tic[lo:hi]
        q_seg   = q_tic[lo:hi]

        pad   = max_shift_bins
        ref_p = np.pad(ref_seg, pad)
        q_p   = np.pad(q_seg,   pad)

        cc    = np.correlate(ref_p, q_p, mode='full')
        n     = len(ref_p)
        lags  = np.arange(-(n - 1), n)
        valid = np.abs(lags) <= pad
        best_lag = int(lags[valid][np.argmax(cc[valid])]) if valid.any() else 0

        # Apply integer shift to every m/z channel in this interval.
        # lag < 0 → query is ahead → roll backward; lag > 0 → roll forward.
        for mz in range(query.shape[1]):
            seg = query[lo:hi, mz]
            warped[lo:hi, mz] = _apply_int_shift(seg, -best_lag)

    return warped.astype(np.float32)


# ── Published method 3: COW (Nielsen et al. 1998) ────────────────────────────

def align_cow(query: np.ndarray, ref: np.ndarray,
              n_segments: int = 10,
              slack: int = 10) -> np.ndarray:
    """
    Correlation Optimised Warping (Nielsen et al., 1998).

    Dynamic programming finds boundary points b[0..N] partitioning the query
    TIC into N segments such that the sum of Pearson correlations between
    reference segment i and query segment b[i]→b[i+1] is maximised, subject
    to each query segment length lying in [T-slack, T+slack] where T = M/N.
    """
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)
    M, T, N = N_BINS, N_BINS // n_segments, n_segments

    def seg_corr(r_seg: np.ndarray, q_start: int, q_len: int) -> float:
        if q_start + q_len > M or q_len < 2:
            return -np.inf
        q_seg = q_tic[q_start:q_start + q_len]
        if len(q_seg) != len(r_seg):
            q_seg = np.interp(np.linspace(0, 1, len(r_seg)),
                              np.linspace(0, 1, len(q_seg)), q_seg)
        sr, sq = np.std(r_seg), np.std(q_seg)
        if sr < 1e-8 or sq < 1e-8:
            return 0.0
        return float(np.corrcoef(r_seg, q_seg)[0, 1])

    ref_segs = [ref_tic[i * T:(i + 1) * T] for i in range(N)]

    INF = -1e9
    dp_score = np.full(M + 1, INF)
    dp_score[0] = 0.0
    # Store one prev-pointer array per segment so traceback spans all N layers
    all_prev = []

    for i in range(N):
        new_score = np.full(M + 1, INF)
        new_prev  = np.full(M + 1, -1, dtype=int)
        for b_prev in range(M + 1):
            if dp_score[b_prev] == INF:
                continue
            for dq in range(max(1, T - slack), T + slack + 1):
                b_next = b_prev + dq
                if b_next > M:
                    break
                if i == N - 1 and b_next != M:
                    continue
                sc = dp_score[b_prev] + seg_corr(ref_segs[i], b_prev, dq)
                if sc > new_score[b_next]:
                    new_score[b_next] = sc
                    new_prev[b_next]  = b_prev
        dp_score = new_score
        all_prev.append(new_prev)

    # Traceback through all N layers to recover N+1 boundaries [0, b1, ..., M]
    boundaries = [M]
    pos = M
    for layer in reversed(all_prev):   # walk back through layers N-1 → 0
        prev = layer[pos]
        if prev < 0:
            return query.copy()
        boundaries.append(prev)
        pos = prev
    boundaries = list(reversed(boundaries))   # [0, b1, ..., b_{N-1}, M]

    # ref boundary i corresponds to i * T bins; query boundary is boundaries[i]
    ref_pts = [i * T * BIN_MIN for i in range(N + 1)]   # N+1 points
    q_pts   = [b * BIN_MIN     for b in boundaries]      # N+1 points

    for i in range(1, len(q_pts)):
        if q_pts[i] <= q_pts[i - 1]:
            q_pts[i] = q_pts[i - 1] + 0.01

    warp = interp1d(ref_pts, q_pts, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp)


# ── m/z-anchored COW (our method) ────────────────────────────────────────────

def align_mz_cow(query: np.ndarray, ref: np.ndarray,
                 encode_fn=None,
                 sim_threshold: float = 0.40,
                 max_drift_min: float = 6.0) -> np.ndarray:
    ref_pks = _detect_peaks(ref)
    q_pks   = _detect_peaks(query)
    ref_fps = np.array([_context_fp(ref,   pk) for pk in ref_pks])
    q_fps   = np.array([_context_fp(query, pk) for pk in q_pks])
    ref_rep = encode_fn(ref_fps) if encode_fn else ref_fps
    q_rep   = encode_fn(q_fps)   if encode_fn else q_fps
    ref_norm = np.linalg.norm(ref_rep, axis=1, keepdims=True) + 1e-8
    q_norm   = np.linalg.norm(q_rep,   axis=1, keepdims=True) + 1e-8
    sim      = (q_rep / q_norm) @ (ref_rep / ref_norm).T
    ref_t = ref_pks * BIN_MIN
    q_t   = q_pks   * BIN_MIN
    dm    = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sc    = sim.copy()
    sc[dm] = -1.0
    ri, ci = linear_sum_assignment(-sc)
    keep   = (sim[ri, ci] >= sim_threshold) & ~dm[ri, ci]
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
    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]:
            mono.append(i)
    ra, qa = ra[mono], qa[mono]
    rf = np.concatenate([[0], ra, [RUN_MIN]])
    qf = np.concatenate([[0], qa, [RUN_MIN]])
    warp = PchipInterpolator(rf, qf)
    return _warp_chroma(query, warp)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_aligned_dataset(chromas: list[np.ndarray],
                           npz_paths: list[Path],
                           align_fn,
                           out_dir: Path) -> list[Path]:
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


def max_proj_features(paths: list[Path]) -> np.ndarray:
    return np.stack([np.load(p)['chroma'].astype(np.float32).max(axis=0)
                     for p in paths])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    y         = np.load(DATA_DIR / 'fish_oil' / 'y.npy').astype(np.int64)
    chromas   = [np.load(p)['chroma'].astype(np.float32) for p in npz_paths]
    print(f"Loaded {len(chromas)} samples  |  classes: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    drift_enc_fn = None
    drift_ckpt   = CKPT_DIR / 'drift_simclr.pt'
    if drift_ckpt.exists():
        m = DilatedSpectrumEncoder().to('cpu'); m.eval()
        m.load_state_dict(torch.load(drift_ckpt, map_location='cpu'))
        def drift_enc_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return m.encode(torch.from_numpy(x.astype('float32'))).numpy()
        print(f"Drift encoder loaded: {drift_ckpt.name}")

    tmp_root = Path(tempfile.mkdtemp(prefix='cow_dcnn_'))
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
        ]
        for name, fn in named_fns:
            tag     = name.split('[')[0].strip().replace(' ', '_').replace('/', '').replace('(', '').replace(')', '').replace('.', '')
            out_dir = tmp_root / tag
            paths   = build_aligned_dataset(chromas, npz_paths, fn, out_dir)
            align_methods.append((name, paths, out_dir))
            print(f"  {name}: done")

        cnn_conditions = ['from_scratch']
        if CHROMA_CKPT.exists():
            cnn_conditions.append('chroma_pretrain')

        all_results: list[dict] = []

        for align_name, paths, _ in align_methods:
            print(f"\n{'='*66}")
            print(f"Alignment: {align_name}")
            print('='*66)

            finetuner = ChromaFinetuner(CV_CFG, list(paths), y)
            for cond in cnn_conditions:
                print(f"\n  CNN [{cond}] …")
                res = finetuner.evaluate_condition(cond)
                ba, f1 = res['balanced_accuracy'], res['macro_f1']
                print(f"    bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                      f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")
                all_results.append({'label': f"{align_name}  |  CNN {cond}",
                                    'ba': ba, 'f1': f1})

            X = max_proj_features(list(paths))
            for clf in ['plsda', 'rf']:
                print(f"\n  {clf.upper()} …")
                res = baseline_cv(X, y, clf,
                                  seeds=CV_CFG['task']['cv_seeds'],
                                  cv_strategy='kfold',
                                  cv_folds=CV_CFG['task']['cv_folds'])
                ba, f1 = res['balanced_accuracy'], res['macro_f1']
                print(f"    bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                      f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")
                all_results.append({'label': f"{align_name}  |  {clf.upper()}",
                                    'ba': ba, 'f1': f1})

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print('\n' + '='*86)
    print(f"{'Condition':<52}  {'Bal. Acc.':>14}  {'Macro-F1':>14}")
    print('-'*86)
    prev_align = None
    for r in all_results:
        align = r['label'].split('  |  ')[0]
        if align != prev_align:
            if prev_align is not None:
                print()
            prev_align = align
        ba = r['ba'];  f1 = r['f1']
        print(f"{r['label']:<52}  "
              f"{np.mean(ba):.3f} ± {np.std(ba):.3f}  "
              f"{np.mean(f1):.3f} ± {np.std(f1):.3f}")
    print('='*86)
    print(f"5-fold × {len(CV_CFG['task']['cv_seeds'])} seeds = "
          f"{5 * len(CV_CFG['task']['cv_seeds'])} runs per condition")
    print("Classes: 0=SNA (snapper)  1=GUR (gurnard)  "
          "2=TAR (tarakihi)  3=BCO (blue cod)")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'classification.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
