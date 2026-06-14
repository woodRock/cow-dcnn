"""
Cross-study alignment evaluation: mtbls21 (wheat, CO2 treatments) ↔
mtbls288 (rice, cultivar study).

Evaluates alignment quality on the hardest case: two studies from different
labs, species, and years with only partial metabolome overlap.

Methods compared:
  Unaligned             — raw TIC Pearson correlation, no alignment
  Raw cosine            — m/z fingerprint cosine + Hungarian + PCHIP warp
  Drift encoder         — checkpoints/drift_simclr.pt  (within-study pretrain)
  Cross-study encoder   — checkpoints/cross_study_simclr.pt  (this work)

Metrics:
  TIC Pearson r      — mean across all 40×79 = 3,160 wheat↔rice pairs
  Anchor precision   — fraction of matched peak pairs with raw m/z cosine ≥ 0.70
                       (proxy for truly matching the same compound across studies)
  Mean anchors used  — average number of RANSAC-inlier anchor pairs per alignment

The anchor precision metric provides an alignment-method-independent quality
check: regardless of how anchors were found (raw cosine or learned embeddings),
we ask whether the two matched peaks actually look like the same compound in
m/z fragmentation space.
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import DilatedSpectrumEncoder

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS    = 200
RUN_MIN   = 45.0
BIN_MIN   = RUN_MIN / N_BINS
TIME_AXIS = np.linspace(0, RUN_MIN, N_BINS)

MAX_DRIFT_MIN    = 20.0   # wider window for cross-study (different labs/years)
SIM_THRESHOLD    = 0.30   # lower floor for cross-study (more spectral variation)
PRECISION_CUTOFF = 0.70   # m/z cosine above which we call a match "same compound"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


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


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _ctx_fp(chroma: np.ndarray, pk: int) -> np.ndarray:
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))


def fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    return np.array([_ctx_fp(chroma, pk) for pk in pks])


# ── Cross-study alignment ─────────────────────────────────────────────────────

def align_with(query: np.ndarray, ref: np.ndarray,
               encode_fn=None) -> tuple[np.ndarray, int, float]:
    """
    Align query onto ref time axis, returning (warped_tic, n_anchors, precision).

    precision = fraction of proposed anchor pairs with raw m/z cosine >= PRECISION_CUTOFF.
    Computed before RANSAC so it reflects the initial match quality of each method.
    """
    ref_pks = detect_peaks(ref)
    q_pks   = detect_peaks(query)

    if len(ref_pks) < 2 or len(q_pks) < 2:
        return query.sum(axis=1), 0, 0.0

    ref_fps = fingerprints(ref, ref_pks)
    q_fps   = fingerprints(query, q_pks)

    ref_rts = ref_pks.astype(np.float32) / N_BINS
    q_rts   = q_pks.astype(np.float32) / N_BINS

    if encode_fn is not None:
        ref_rep = encode_fn(ref_fps, ref_rts)
        q_rep   = encode_fn(q_fps,   q_rts)
    else:
        ref_rep, q_rep = ref_fps, q_fps

    sim = cosine_similarity(q_rep, ref_rep)   # (n_q, n_ref)

    ref_t      = TIME_AXIS[ref_pks]
    q_t        = TIME_AXIS[q_pks]
    drift_mask = np.abs(q_t[:, None] - ref_t[None, :]) > MAX_DRIFT_MIN
    sim_c      = sim.copy()
    sim_c[drift_mask] = -1.0

    row_ind, col_ind = linear_sum_assignment(-sim_c)

    avail     = sim[~drift_mask].flatten()
    threshold = max(float(np.percentile(avail, 70)), SIM_THRESHOLD) if len(avail) >= 3 else SIM_THRESHOLD

    ms   = sim[row_ind, col_ind]
    keep = (ms >= threshold) & ~drift_mask[row_ind, col_ind]
    row_ind, col_ind = row_ind[keep], col_ind[keep]

    if len(row_ind) < 2:
        return query.sum(axis=1), 0, 0.0

    # Anchor precision: independent of how matches were found — use raw m/z cosine
    raw_cosines = np.array([np.dot(q_fps[qi], ref_fps[ri])
                             for qi, ri in zip(row_ind, col_ind)])
    precision = float(np.mean(raw_cosines >= PRECISION_CUTOFF))

    # Fine-tune each anchor with ±3-bin local spectral search
    refined_q = []
    for idx in range(len(row_ind)):
        r_spec = ref_fps[col_ind[idx]]
        q_pk   = q_pks[row_ind[idx]]
        best_bin, best_s = q_pk, float(np.dot(r_spec, _ctx_fp(query, q_pk)))
        for delta in range(-3, 4):
            if delta == 0:
                continue
            cand = int(np.clip(q_pk + delta, 0, N_BINS - 1))
            s = float(np.dot(r_spec, _ctx_fp(query, cand)))
            if s > best_s:
                best_s, best_bin = s, cand
        refined_q.append(best_bin)
    refined_q = np.array(refined_q)

    order    = np.argsort(col_ind)
    row_ind, col_ind = row_ind[order], col_ind[order]
    refined_q = refined_q[order]
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
        except Exception:
            pass

    mono = [0]
    for i in range(1, len(qa)):
        if qa[i] > qa[mono[-1]]:
            mono.append(i)
    ra, qa = ra[mono], qa[mono]

    if len(ra) < 2:
        return query.sum(axis=1), 0, precision

    rf     = np.concatenate([[0.0], ra, [RUN_MIN]])
    qf     = np.concatenate([[0.0], qa, [RUN_MIN]])
    warp   = PchipInterpolator(rf, qf)
    q_tic  = query.sum(axis=1)
    warped = interp1d(TIME_AXIS, q_tic, bounds_error=False,
                      fill_value=0.0)(warp(TIME_AXIS))
    return warped, len(ra), precision


# ── Encoder loading ───────────────────────────────────────────────────────────

def _load_cnn_encoder(path: Path):
    m = DilatedSpectrumEncoder().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x.astype('float32')).to(DEVICE)
            r = torch.from_numpy(rt.astype('float32')).to(DEVICE) if rt is not None else None
            return m.encode(t, r).cpu().numpy()
    return encode_fn


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(m21: list[np.ndarray], m288: list[np.ndarray],
             encode_fn=None) -> dict:
    """Evaluate alignment across all 40×79 wheat↔rice pairs."""
    m288_tics = [c.sum(axis=1) for c in m288]

    cors, precs, ancs = [], [], []
    n_total = len(m21) * len(m288)
    done    = 0

    for qc in m21:
        for rc, ref_tic in zip(m288, m288_tics):
            warped, n_anc, prec = align_with(qc, rc, encode_fn=encode_fn)
            cor = float(np.corrcoef(ref_tic, warped)[0, 1])
            cors.append(0.0 if np.isnan(cor) else cor)
            precs.append(prec)
            ancs.append(n_anc)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{n_total} pairs …")

    return {
        'tic_r':     float(np.nanmean(cors)),
        'precision': float(np.nanmean(precs)),
        'n_anchors': float(np.nanmean(ancs)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading chromatograms …")
    m21  = load_chromas(DATA_DIR / 'mtbls21'  / 'chroma')
    m288 = load_chromas(DATA_DIR / 'mtbls288' / 'chroma')
    print(f"  mtbls21  (wheat, CO2 treatments): {len(m21)} samples")
    print(f"  mtbls288 (rice, cultivar study) : {len(m288)} samples")
    print(f"  Total pairs : {len(m21) * len(m288)}  "
          f"|  drift window : {MAX_DRIFT_MIN} min  "
          f"|  precision cutoff : {PRECISION_CUTOFF}\n")

    # Unaligned baseline
    m21_tics  = [c.sum(axis=1) for c in m21]
    m288_tics = [c.sum(axis=1) for c in m288]
    unaligned_r = float(np.mean([
        np.corrcoef(ref_tic, q_tic)[0, 1]
        for q_tic in m21_tics
        for ref_tic in m288_tics
    ]))
    print(f"Unaligned baseline  TIC r = {unaligned_r:.3f}\n")

    # Methods: (label, encode_fn)
    methods: list[tuple[str, object]] = [('Raw cosine', None)]
    for label, fname in [('Drift encoder',       'drift_simclr.pt'),
                         ('Cross-study encoder', 'cross_study_simclr.pt')]:
        p = CKPT_DIR / fname
        if p.exists():
            methods.append((label, _load_cnn_encoder(p)))
            print(f"Loaded  {label}  from {fname}")
        else:
            print(f"SKIP    {label}: {p} not found")
    print()

    results = {}
    for label, enc_fn in methods:
        print(f"Evaluating: {label} …")
        r = evaluate(m21, m288, encode_fn=enc_fn)
        results[label] = r
        print(f"  TIC r={r['tic_r']:.3f}  precision={r['precision']:.3f}  "
              f"anchors={r['n_anchors']:.1f}\n")

    # Summary table
    w = 26
    print("=" * 75)
    print(f"{'Method':<{w}}  {'TIC r':>7}  {'Δ unaligned':>12}  "
          f"{'Precision':>10}  {'Anchors':>8}")
    print("-" * 75)
    print(f"{'Unaligned':<{w}}  {unaligned_r:>7.3f}  {'—':>12}  {'—':>10}  {'—':>8}")
    for lbl, r in results.items():
        delta = r['tic_r'] - unaligned_r
        print(f"{lbl:<{w}}  {r['tic_r']:>7.3f}  {delta:>+12.3f}  "
              f"{r['precision']:>10.3f}  {r['n_anchors']:>8.1f}")
    print("=" * 75)
    print("TIC r    : mean Pearson correlation across all wheat↔rice pairs")
    print("Precision: fraction of matched peak pairs with m/z cosine ≥ "
          f"{PRECISION_CUTOFF} (same-compound proxy)")
    print("Anchors  : mean RANSAC-inlier anchor pairs used per alignment")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / 'cross_study_alignment.txt'
    with open(out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            main()
        finally:
            sys.stdout = orig
    print(f"Results saved → {out}")
