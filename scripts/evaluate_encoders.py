"""
Compare alignment quality across all encoder methods.

Methods evaluated:
  Unaligned        — raw TIC correlation, no alignment
  Raw cosine       — m/z fingerprint cosine similarity + Hungarian + piecewise warp
  SimCLR encoder   — checkpoints/gcms_simclr.pt  (MoNA pretraining + GC-MS fine-tune)
  Drift encoder    — checkpoints/drift_simclr.pt (cross-sample peak pairs)

Datasets:
  Within-study  — fish_oil 103×103 pairwise TIC correlation
  Cross-study   — fish_oil × mtbls288 (103×79 pairs)

Metric: mean off-diagonal Pearson correlation of aligned TICs.
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import SpectrumEncoder, DilatedSpectrumEncoder, SparseSpectrumTransformer

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

N_BINS      = 200
RUN_MIN     = 45.0
TIME_AXIS   = np.linspace(0, RUN_MIN, N_BINS)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_chromas(chroma_dir: Path) -> list[np.ndarray]:
    return [np.load(p)['chroma'].astype(np.float32)
            for p in sorted(chroma_dir.glob('*.npz'))]


# ── Alignment primitives ──────────────────────────────────────────────────────

def detect_peaks(chroma: np.ndarray, height_pct: int = 80,
                 min_dist: int = 5, n_peaks: int = 30) -> np.ndarray:
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, height_pct)
    pks, props = sp_find_peaks(tic, height=thr, distance=min_dist)
    if len(pks) > n_peaks:
        top = np.argsort(props['peak_heights'])[-n_peaks:]
        pks = pks[top]
    return np.sort(pks)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _context_fp(chroma: np.ndarray, pk: int) -> np.ndarray:
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2_norm(chroma[lo:hi].mean(axis=0))


def fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    return np.array([_context_fp(chroma, pk) for pk in pks])


BIN_MIN = RUN_MIN / N_BINS


def align_with(query: np.ndarray, ref: np.ndarray,
               encode_fn=None,
               max_drift_min: float = 6.0) -> tuple[np.ndarray, int]:
    """
    Align query TIC onto ref time axis.
    encode_fn: callable (fps, rts) → (N,D); None = raw cosine on m/z fingerprints.
    Returns (warped_tic, n_anchors_used).
    """
    ref_pks = detect_peaks(ref)
    q_pks   = detect_peaks(query)
    ref_fps = fingerprints(ref, ref_pks)
    q_fps   = fingerprints(query, q_pks)

    # RT-aware encoding: pass normalised peak positions alongside spectra
    ref_rts = ref_pks.astype(np.float32) / N_BINS
    q_rts   = q_pks.astype(np.float32) / N_BINS
    if encode_fn:
        ref_rep = encode_fn(ref_fps, ref_rts)
        q_rep   = encode_fn(q_fps,   q_rts)
    else:
        ref_rep, q_rep = ref_fps, q_fps

    sim = cosine_similarity(q_rep, ref_rep)            # (n_q, n_ref)

    ref_t = TIME_AXIS[ref_pks]; q_t = TIME_AXIS[q_pks]
    drift_mask = np.abs(q_t[:, None] - ref_t[None, :]) > max_drift_min
    sim_c = sim.copy(); sim_c[drift_mask] = -1.0

    row_ind, col_ind = linear_sum_assignment(-sim_c)

    # Adaptive threshold: top-30% of within-window similarities
    avail = sim[~drift_mask].flatten()
    threshold = max(float(np.percentile(avail, 70)), 0.30) if len(avail) >= 3 else 0.35

    ms   = sim[row_ind, col_ind]
    keep = (ms >= threshold) & ~drift_mask[row_ind, col_ind]
    row_ind, col_ind = row_ind[keep], col_ind[keep]

    if len(row_ind) < 2:
        return query.sum(axis=1), 0

    # Fine-tune each anchor: search ±3 bins for best spectral match
    refined_q = []
    for idx in range(len(row_ind)):
        r_spec = ref_fps[col_ind[idx]]
        q_pk   = q_pks[row_ind[idx]]
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

    order = np.argsort(col_ind)
    row_ind, col_ind = row_ind[order], col_ind[order]
    refined_q = refined_q[order]
    ra = TIME_AXIS[ref_pks[col_ind]]
    qa = TIME_AXIS[refined_q]

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
    q_tic  = query.sum(axis=1)
    warped = interp1d(TIME_AXIS, q_tic, bounds_error=False, fill_value=0.0)(warp(TIME_AXIS))
    return warped, len(ra)


# ── Encoder loading ───────────────────────────────────────────────────────────

def _load_encoder(path: Path, arch: str = 'mlp'):
    if arch == 'transformer':
        m = SparseSpectrumTransformer().to(DEVICE)
    elif arch == 'cnn':
        m = DilatedSpectrumEncoder().to(DEVICE)
    else:
        m = SpectrumEncoder().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    if arch in ('cnn', 'transformer'):
        def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32')).to(DEVICE)
                r = torch.from_numpy(rt.astype('float32')).to(DEVICE) if rt is not None else None
                return m.encode(t, r).cpu().numpy()
    else:
        def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32')).to(DEVICE)
                return m.encode(t).cpu().numpy()
    return encode_fn


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_within(chromas: list[np.ndarray], encode_fn=None,
                    direct_align_fn=None,
                    max_drift_min: float = 6.0) -> float:
    """
    direct_align_fn: optional callable (query_chroma, ref_chroma) → warped_chroma.
    If provided, bypasses align_with entirely.
    """
    n = len(chromas)
    tics = [c.sum(axis=1) for c in chromas]
    total, count = 0.0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if direct_align_fn is not None:
                warped = direct_align_fn(chromas[j], chromas[i]).sum(axis=1)
            else:
                warped, _ = align_with(chromas[j], chromas[i],
                                       encode_fn=encode_fn, max_drift_min=max_drift_min)
            total += float(np.corrcoef(tics[i], warped)[0, 1])
            count += 1
    return total / count if count else 0.0


def evaluate_cross(fish: list[np.ndarray], m288: list[np.ndarray],
                   encode_fn=None, direct_align_fn=None,
                   max_drift_min: float = 15.0) -> float:
    fish_tics = [c.sum(axis=1) for c in fish]
    m288_tics = [c.sum(axis=1) for c in m288]
    total, count = 0.0, 0
    for fc, ft in zip(fish, fish_tics):
        for mc, mt in zip(m288, m288_tics):
            if direct_align_fn is not None:
                warped = direct_align_fn(mc, fc).sum(axis=1)
            else:
                warped, _ = align_with(mc, fc, encode_fn=encode_fn,
                                       max_drift_min=max_drift_min)
            total += float(np.corrcoef(ft, warped)[0, 1])
            count += 1
    return total / count if count else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    fish  = load_chromas(DATA_DIR / 'fish_oil' / 'chroma')
    m288  = load_chromas(DATA_DIR / 'mtbls288' / 'chroma')
    print(f"  fish_oil: {len(fish)} samples   mtbls288: {len(m288)} samples")

    fish_tics = [c.sum(axis=1) for c in fish]
    m288_tics = [c.sum(axis=1) for c in m288]

    # Unaligned baselines
    n = len(fish)
    unaligned_w = np.mean([np.corrcoef(fish_tics[i], fish_tics[j])[0, 1]
                            for i in range(n) for j in range(n) if i != j])
    unaligned_c = np.mean([np.corrcoef(ft, mt)[0, 1]
                            for ft in fish_tics for mt in m288_tics])

    # Build encoder map — gcms_simclr uses MLP; drift_simclr uses DilatedCNN
    encoders: dict[str, object] = {'Raw cosine': None}
    for name, fname, arch in [('SimCLR encoder',          'gcms_simclr.pt',      'mlp'),
                               ('Drift encoder (CNN)',     'drift_simclr.pt',     'cnn'),
                               ('Drift encoder (Transf.)', 'transformer_simclr.pt', 'transformer')]:
        p = CKPT_DIR / fname
        if p.exists():
            encoders[name] = _load_encoder(p, arch=arch)
            print(f"  Loaded {name} from {fname}")
        else:
            print(f"  SKIP {name}: {p} not found")

    # Run evaluations
    results: dict[str, dict] = {}
    for name, enc_fn in encoders.items():
        print(f"\nEvaluating: {name} …")
        w = evaluate_within(fish, encode_fn=enc_fn, max_drift_min=6.0)
        c = evaluate_cross(fish, m288, encode_fn=enc_fn, max_drift_min=15.0)
        results[name] = {'within': w, 'cross': c}
        print(f"  Within-study: {w:.3f}   Cross-study: {c:.3f}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Method':<22}  {'Within-study':>14}  {'Cross-study':>12}")
    print("-" * 65)
    print(f"{'Unaligned':<22}  {unaligned_w:>14.3f}  {unaligned_c:>12.3f}")
    for name in encoders:
        r = results[name]
        dw = r['within'] - unaligned_w
        dc = r['cross']  - unaligned_c
        print(f"{name:<22}  {r['within']:>14.3f} ({dw:+.3f})"
              f"  {r['cross']:>12.3f} ({dc:+.3f})")
    print("=" * 65)
    print("Values: mean Pearson TIC correlation (higher = better aligned)")
    print("Within drift window: 6 min   Cross drift window: 15 min")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'encoder_evaluation.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
