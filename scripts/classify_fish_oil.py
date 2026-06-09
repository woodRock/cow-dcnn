"""
Does chromatogram alignment improve downstream fish species classification?

Alignment methods compared
──────────────────────────
  No alignment          — raw chromatograms, no preprocessing
  Global shift          — single lag from FFT cross-correlation of TICs
  Segment shift         — icoshift-style: per-segment FFT shift (10 segments)
  m/z COW (cosine)      — m/z fingerprint anchored warp, raw cosine similarity
  m/z COW (drift enc.)  — m/z fingerprint anchored warp, drift encoder

Classifiers
───────────
  ChromatogramCNN  from_scratch    — dilated ResBlock 1D CNN, random init
  ChromatogramCNN  chroma_pretrain — same arch, weights from next-frame prediction
  PLS-DA                           — classical metabolomics baseline
  Random Forest                    — tree ensemble on max-projection spectrum

All CNN conditions use ChromatogramCNN from chroma-dcnn (so results are directly
comparable with the chroma-dcnn paper). Classical baselines use the per-m/z
max-projection of the aligned chromatogram as their feature vector.

Evaluation: 5-fold stratified CV × 3 seeds = 15 runs per condition.
Metric: balanced accuracy and macro-F1.
"""

from __future__ import annotations

import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks, fftconvolve

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, '/Users/woodj/Desktop/chroma-dcnn/src')

from encoder import SpectrumEncoder
from chroma_dcnn.models.chroma_cnn import ChromaCNNConfig
from chroma_dcnn.training.finetune_chroma import ChromaFinetuner
from chroma_dcnn.evaluation.baselines import baseline_cv

DATA_DIR      = Path(__file__).parent.parent / 'data'
CKPT_DIR      = Path(__file__).parent.parent / 'checkpoints'
CHROMA_CKPT   = Path('/Users/woodj/Desktop/chroma-dcnn/checkpoints/chroma_pretrain/best.pt')
FISH_CHROMA   = DATA_DIR / 'fish_oil' / 'chroma'

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
        'cv_seeds': [0, 1, 2],      # 3 seeds × 5 folds = 15 runs
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


# ── Alignment helpers ─────────────────────────────────────────────────────────

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


def _warp_chroma(query: np.ndarray, warp_fn) -> np.ndarray:
    """Apply a time-warp function to the full 2D chromatogram."""
    query_times = np.clip(warp_fn(TIME_AX), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(query_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1 - a) * query[t0] + a * query[t1]
    return warped.astype(np.float32)


# ── Alignment method 1: global shift ─────────────────────────────────────────

def align_global_shift(query: np.ndarray, ref: np.ndarray,
                        max_shift_bins: int = 20) -> np.ndarray:
    """Find best global lag via FFT cross-correlation of TICs; shift full 2D chroma."""
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)
    corr    = fftconvolve(ref_tic, q_tic[::-1], mode='full')
    lags    = np.arange(-(N_BINS - 1), N_BINS)
    valid   = np.abs(lags) <= max_shift_bins
    best_lag = int(lags[valid][np.argmax(corr[valid])])  # positive = query is ahead

    ref_bins = np.arange(N_BINS, dtype=float) * BIN_MIN
    q_bins   = ref_bins + best_lag * BIN_MIN             # query_time = ref_time + lag
    warp_fn  = interp1d(ref_bins, q_bins, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp_fn)


# ── Alignment method 2: segment shift (icoshift-style) ───────────────────────

def align_segment_shift(query: np.ndarray, ref: np.ndarray,
                         n_segments: int = 10, max_shift_bins: int = 10) -> np.ndarray:
    """
    Divide TIC into n_segments; find best local shift for each via cross-correlation.
    Monotone piecewise-linear warp fitted through the segment-centre shifts.
    """
    ref_tic = ref.sum(axis=1)
    q_tic   = query.sum(axis=1)
    seg_len = N_BINS // n_segments

    ref_pts = [0.0]
    q_pts   = [0.0]

    for s in range(n_segments):
        lo  = s * seg_len
        hi  = min(lo + seg_len, N_BINS)
        mid = (lo + hi) / 2.0

        # Pad reference segment to allow cross-correlation with shift
        pad = max_shift_bins
        ref_seg = ref_tic[lo:hi]
        q_lo    = max(0, lo - pad)
        q_hi    = min(N_BINS, hi + pad)
        q_seg   = q_tic[q_lo:q_hi]

        corr = fftconvolve(ref_seg, q_seg[::-1], mode='full')
        lags = np.arange(-(len(q_seg) - 1), len(ref_seg))
        valid = np.abs(lags) <= pad
        if valid.any():
            best_lag = int(lags[valid][np.argmax(corr[valid])])
        else:
            best_lag = 0

        ref_pts.append(mid * BIN_MIN)
        q_pts.append(np.clip((mid + best_lag) * BIN_MIN, 0, RUN_MIN))

    ref_pts.append(RUN_MIN)
    q_pts.append(RUN_MIN)

    # Enforce monotonicity
    for i in range(1, len(q_pts)):
        if q_pts[i] <= q_pts[i - 1]:
            q_pts[i] = q_pts[i - 1] + 0.01

    warp_fn = interp1d(ref_pts, q_pts, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp_fn)


# ── Alignment method 3 & 4: m/z-anchored COW ─────────────────────────────────

def align_mz_cow(query: np.ndarray, ref: np.ndarray,
                  encode_fn=None,
                  sim_threshold: float = 0.5,
                  max_drift_min: float = 6.0) -> np.ndarray:
    ref_pks = _detect_peaks(ref);   q_pks = _detect_peaks(query)
    ref_fps = np.array([_l2(ref[pk]) for pk in ref_pks])
    q_fps   = np.array([_l2(query[pk]) for pk in q_pks])

    ref_rep = encode_fn(ref_fps) if encode_fn else ref_fps
    q_rep   = encode_fn(q_fps)   if encode_fn else q_fps

    sim = q_rep @ ref_rep.T
    sim /= (np.linalg.norm(q_rep, axis=1, keepdims=True) *
            np.linalg.norm(ref_rep, axis=1, keepdims=True).T + 1e-8)

    ref_t = ref_pks * BIN_MIN;  q_t = q_pks * BIN_MIN
    dm    = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sc    = sim.copy();  sc[dm] = -1.0
    ri, ci = linear_sum_assignment(-sc)
    ms = sim[ri, ci]
    keep = (ms >= sim_threshold) & ~dm[ri, ci]
    ri, ci = ri[keep], ci[keep]

    if len(ri) < 2:
        return query.copy()

    order = np.argsort(ci);  ri, ci = ri[order], ci[order]
    ra = TIME_AX[ref_pks[ci]];  qa = TIME_AX[q_pks[ri]]

    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]:
            mono.append(i)
    ra, qa = ra[mono], qa[mono]

    rf = np.concatenate([[0], ra, [RUN_MIN]])
    qf = np.concatenate([[0], qa, [RUN_MIN]])
    warp = interp1d(rf, qf, kind='linear', bounds_error=False, fill_value='extrapolate')
    return _warp_chroma(query, warp)


# ── Apply alignment to all samples, save to temp .npz dir ────────────────────

def build_aligned_dataset(chromas: list[np.ndarray],
                           npz_paths: list[Path],
                           align_fn,
                           out_dir: Path) -> list[Path]:
    """
    Apply align_fn(query, ref) to every sample and save aligned .npz files.
    Returns list of output paths in the same order as npz_paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = chromas[0]
    out_paths = []
    for i, (chroma, src_path) in enumerate(zip(chromas, npz_paths)):
        out_path = out_dir / src_path.name
        if not out_path.exists():
            aligned = align_fn(chroma, ref) if i > 0 else chroma.copy()
            np.savez_compressed(out_path, chroma=aligned)
        out_paths.append(out_path)
    return out_paths


# ── Max-projection features for classical baselines ──────────────────────────

def max_proj_features(npz_paths: list[Path]) -> np.ndarray:
    """Per-m/z maximum across all RT bins — (N, 1000) feature matrix."""
    feats = []
    for p in npz_paths:
        chroma = np.load(p)['chroma'].astype(np.float32)   # (200, 1000)
        feats.append(chroma.max(axis=0))
    return np.stack(feats)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    y         = np.load(DATA_DIR / 'fish_oil' / 'y.npy').astype(np.int64)
    chromas   = [np.load(p)['chroma'].astype(np.float32) for p in npz_paths]
    print(f"Loaded {len(chromas)} samples  |  classes: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Load drift encoder for m/z COW ───────────────────────────────────
    drift_enc_fn = None
    drift_ckpt   = CKPT_DIR / 'drift_simclr.pt'
    if drift_ckpt.exists():
        m = SpectrumEncoder().to('cpu'); m.eval()
        m.load_state_dict(torch.load(drift_ckpt, map_location='cpu'))
        def drift_enc_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return m.encode(torch.from_numpy(x.astype('float32'))).numpy()
        print(f"Drift encoder loaded: {drift_ckpt.name}")

    # ── Alignment conditions ──────────────────────────────────────────────
    tmp_root = Path(tempfile.mkdtemp(prefix='cow_dcnn_'))
    try:
        align_methods = [
            ('No alignment',         npz_paths,  None),
        ]

        print("\nPre-computing aligned chromatograms …")
        for name, fn in [
            ('Global shift',          lambda q, r: align_global_shift(q, r)),
            ('Segment shift',         lambda q, r: align_segment_shift(q, r)),
            ('m/z COW (cosine)',      lambda q, r: align_mz_cow(q, r)),
            ('m/z COW (drift enc.)',  lambda q, r: align_mz_cow(q, r, encode_fn=drift_enc_fn)),
        ]:
            tag     = name.replace(' ', '_').replace('/', '').replace('(', '').replace(')', '').replace('.', '')
            out_dir = tmp_root / tag
            paths   = build_aligned_dataset(chromas, npz_paths, fn, out_dir)
            align_methods.append((name, paths, out_dir))
            print(f"  {name}: done")

        # ── CNN model config ──────────────────────────────────────────────
        cnn_cfg     = ChromaCNNConfig(mz_max=1000, cnn_channels=128,
                                      kernel_size=7, num_classes=4, dropout=0.3)
        cnn_conditions = ['from_scratch']
        if CHROMA_CKPT.exists():
            cnn_conditions.append('chroma_pretrain')
            print(f"\nPretrained checkpoint: {CHROMA_CKPT.name}")
        else:
            print("\nchroma_pretrain checkpoint not found — CNN from_scratch only")

        # ── Run all conditions ────────────────────────────────────────────
        all_results: list[dict] = []

        for align_name, paths, _ in align_methods:
            print(f"\n{'='*60}")
            print(f"Alignment: {align_name}")
            print('='*60)

            # CNN conditions via ChromaFinetuner
            finetuner = ChromaFinetuner(CV_CFG, list(paths), y)
            for cond in cnn_conditions:
                print(f"\n  CNN [{cond}] …")
                res = finetuner.evaluate_condition(cond)
                ba, f1 = res['balanced_accuracy'], res['macro_f1']
                label = f"{align_name}  |  CNN {cond}"
                print(f"    bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                      f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")
                all_results.append({'label': label, 'ba': ba, 'f1': f1})

            # Classical baselines on max-projection features
            X_feat = max_proj_features(list(paths))
            for clf_name in ['plsda', 'rf']:
                print(f"\n  {clf_name.upper()} …")
                res = baseline_cv(X_feat, y, clf_name,
                                  seeds=CV_CFG['task']['cv_seeds'],
                                  cv_strategy='kfold',
                                  cv_folds=CV_CFG['task']['cv_folds'])
                ba, f1 = res['balanced_accuracy'], res['macro_f1']
                label = f"{align_name}  |  {clf_name.upper()}"
                print(f"    bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                      f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")
                all_results.append({'label': label, 'ba': ba, 'f1': f1})

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Summary table ─────────────────────────────────────────────────────
    print('\n' + '='*80)
    print(f"{'Condition':<45}  {'Bal. Acc.':>14}  {'Macro-F1':>14}")
    print('-'*80)

    prev_align = None
    for r in all_results:
        align = r['label'].split('  |  ')[0]
        if align != prev_align:
            if prev_align is not None:
                print()
            prev_align = align
        ba = r['ba'];  f1 = r['f1']
        print(f"{r['label']:<45}  "
              f"{np.mean(ba):.3f} ± {np.std(ba):.3f}  "
              f"{np.mean(f1):.3f} ± {np.std(f1):.3f}")

    print('='*80)
    print(f"5-fold × {len(CV_CFG['task']['cv_seeds'])} seeds = "
          f"{5 * len(CV_CFG['task']['cv_seeds'])} runs per condition")
    print("Classes: 0=SNA (snapper)  1=GUR (gurnard)  "
          "2=TAR (tarakihi)  3=BCO (blue cod)")


if __name__ == '__main__':
    main()
