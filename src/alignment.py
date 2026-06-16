"""Shared GC-MS cross-study alignment primitives."""
from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks as sp_find_peaks
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from encoder import DilatedSpectrumEncoder, ChromaSpectrumEncoder, PeakSequenceTransformer, WINDOW_HALF

N_BINS    = 200
RUN_MIN   = 45.0
BIN_MIN   = RUN_MIN / N_BINS
TIME_AXIS = np.linspace(0, RUN_MIN, N_BINS)

MAX_DRIFT_MIN    = 20.0
SIM_THRESHOLD    = 0.30
PRECISION_CUTOFF = 0.70


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _ctx_fp(chroma: np.ndarray, pk: int) -> np.ndarray:
    lo = max(0, pk - 1)
    hi = min(chroma.shape[0], pk + 2)
    return _l2(chroma[lo:hi].mean(axis=0))


def detect_peaks(chroma: np.ndarray, height_pct: int = 80,
                 min_dist: int = 5, n_peaks: int = 30) -> np.ndarray:
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, height_pct)
    pks, props = sp_find_peaks(tic, height=thr, distance=min_dist)
    if len(pks) > n_peaks:
        top = np.argsort(props['peak_heights'])[-n_peaks:]
        pks = pks[top]
    return np.sort(pks)


def fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    return np.array([_ctx_fp(chroma, pk) for pk in pks])


def peak_windows(chroma: np.ndarray, pks: np.ndarray,
                 half_width: int = WINDOW_HALF) -> np.ndarray:
    """Extract L2-normalised RT windows around each peak.
    Returns (N_peaks, 2*half_width+1, 1000)."""
    W   = 2 * half_width + 1
    out = np.zeros((len(pks), W, chroma.shape[1]), dtype=np.float32)
    for i, pk in enumerate(pks):
        for j, t in enumerate(range(int(pk) - half_width, int(pk) + half_width + 1)):
            if 0 <= t < chroma.shape[0]:
                v = chroma[t]
                n = np.linalg.norm(v)
                out[i, j] = (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
    return out


def align_pair(query: np.ndarray, ref: np.ndarray,
               encode_fn=None,
               return_anchors: bool = False):
    """
    Align query TIC onto ref time axis.

    Returns (warped_tic, n_anchors, precision) by default.
    With return_anchors=True, returns a 5-tuple:
      (warped_tic, n_anchors, precision, proposed_anchors, warp_fn)
    where proposed_anchors is a list of (q_bin, r_bin, raw_cosine) before RANSAC,
    and warp_fn is the fitted PchipInterpolator (query time → ref time), or None
    if alignment failed.
    """
    ref_pks = detect_peaks(ref)
    q_pks   = detect_peaks(query)

    _fail_tic = query.sum(axis=1)
    if len(ref_pks) < 2 or len(q_pks) < 2:
        return (_fail_tic, 0, 0.0, [], None, []) if return_anchors else (_fail_tic, 0, 0.0)

    ref_fps = fingerprints(ref, ref_pks)
    q_fps   = fingerprints(query, q_pks)

    ref_rts = ref_pks.astype(np.float32) / N_BINS
    q_rts   = q_pks.astype(np.float32) / N_BINS

    if encode_fn is not None:
        if getattr(encode_fn, 'uses_windows', False):
            ref_rep = encode_fn(peak_windows(ref,   ref_pks))
            q_rep   = encode_fn(peak_windows(query, q_pks))
        else:
            ref_rep = encode_fn(ref_fps, ref_rts)
            q_rep   = encode_fn(q_fps,   q_rts)
    else:
        ref_rep, q_rep = ref_fps, q_fps

    sim = cosine_similarity(q_rep, ref_rep)

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
        return (_fail_tic, 0, 0.0, [], None, []) if return_anchors else (_fail_tic, 0, 0.0)

    raw_cosines = np.array([np.dot(q_fps[qi], ref_fps[ri])
                            for qi, ri in zip(row_ind, col_ind)])
    precision   = float(np.mean(raw_cosines >= PRECISION_CUTOFF))

    proposed = [(int(q_pks[qi]), int(ref_pks[ri]), float(raw_cosines[k]))
                for k, (qi, ri) in enumerate(zip(row_ind, col_ind))]

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

    order     = np.argsort(col_ind)
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
        return (_fail_tic, 0, precision, proposed, None, []) if return_anchors else (_fail_tic, 0, precision)

    rf   = np.concatenate([[0.0], ra, [RUN_MIN]])
    qf   = np.concatenate([[0.0], qa, [RUN_MIN]])
    warp = PchipInterpolator(rf, qf)

    q_tic  = query.sum(axis=1)
    warped = interp1d(TIME_AXIS, q_tic, bounds_error=False,
                      fill_value=0.0)(warp(TIME_AXIS))

    # RANSAC-filtered anchor bin indices: (ref_bin, query_bin)
    ra_bins = np.round(ra / BIN_MIN).astype(int).clip(0, N_BINS - 1)
    qa_bins = np.round(qa / BIN_MIN).astype(int).clip(0, N_BINS - 1)
    ransac_anchors = list(zip(ra_bins.tolist(), qa_bins.tolist()))

    if return_anchors:
        return warped, len(ra), precision, proposed, warp, ransac_anchors
    return warped, len(ra), precision


def warp_chroma_2d(query: np.ndarray, warp_fn) -> np.ndarray:
    """Apply a time warp to a full 2-D chromatogram [N_BINS, 1000]."""
    if warp_fn is None:
        return query.copy()
    warped_times = np.clip(warp_fn(TIME_AXIS), 0, RUN_MIN)
    warped = np.zeros_like(query)
    for i, t in enumerate(warped_times):
        idx = t / BIN_MIN
        t0  = int(np.clip(idx, 0, N_BINS - 1))
        t1  = min(t0 + 1, N_BINS - 1)
        a   = float(np.clip(idx - t0, 0, 1))
        warped[i] = (1 - a) * query[t0] + a * query[t1]
    return warped.astype(np.float32)


def max_projection(chroma: np.ndarray) -> np.ndarray:
    """Per-m/z maximum intensity across all RT bins. Shape: (1000,)."""
    return chroma.max(axis=0)


def embed_chromatogram(chroma: np.ndarray, encode_fn) -> np.ndarray:
    """Mean-pool peak-level encoder embeddings into a chromatogram vector."""
    pks = detect_peaks(chroma)
    if len(pks) == 0:
        return np.zeros(128, dtype=np.float32)
    fps = fingerprints(chroma, pks)
    rts = pks.astype(np.float32) / N_BINS
    return encode_fn(fps, rts).mean(axis=0)


def load_encoder(path: Path, device: torch.device = None):
    """Load an encoder checkpoint and return an encode_fn.

    Auto-detects model type from state-dict keys:
      - PeakSequenceTransformer : has 'transformer.' keys  (full-sequence encoder)
      - ChromaSpectrumEncoder   : has 'mz_stem.' + 'rt_blocks.' keys  (local window)
      - DilatedSpectrumEncoder  : default (1-D per-peak CNN)

    The returned encode_fn signature is always (fps, rts) → embeddings:
      fps : (N, 1000) L2-normed m/z fingerprints for N detected peaks
      rts : (N,) normalised RT positions in [0, 1]
      → (N, emb_dim)

    ChromaSpectrumEncoder also has uses_windows=True; align_pair passes
    (N, W, 1000) RT windows instead of (N, 1000) fingerprints.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    state = torch.load(path, map_location=device)
    has_transformer = any(k.startswith('transformer.') for k in state.keys())
    has_mz_stem     = any(k.startswith('mz_stem')      for k in state.keys())

    if has_transformer:
        m = PeakSequenceTransformer().to(device)
        m.load_state_dict(state)
        m.eval()

        def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            # x: (N, 1000), rt: (N,) — all peaks passed together for global attention
            with torch.no_grad():
                fps_t = torch.from_numpy(x.astype('float32')).unsqueeze(0).to(device)
                rts_t = torch.from_numpy(rt.astype('float32')).unsqueeze(0).to(device)
                return m.encode(fps_t, rts_t).squeeze(0).cpu().numpy()

        encode_fn.uses_windows = False

    elif has_mz_stem:
        m = ChromaSpectrumEncoder().to(device)
        m.load_state_dict(state)
        m.eval()

        def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32')).to(device)
                return m.encode(t).cpu().numpy()

        encode_fn.uses_windows = True

    else:
        m = DilatedSpectrumEncoder().to(device)
        m.load_state_dict(state)
        m.eval()

        def encode_fn(x: np.ndarray, rt: np.ndarray = None) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x.astype('float32')).to(device)
                r = torch.from_numpy(rt.astype('float32')).to(device) if rt is not None else None
                return m.encode(t, r).cpu().numpy()

        encode_fn.uses_windows = False

    return encode_fn


def load_chromas(chroma_dir: Path) -> list[np.ndarray]:
    return [np.load(p)['chroma'].astype(np.float32)
            for p in sorted(Path(chroma_dir).glob('*.npz'))]
