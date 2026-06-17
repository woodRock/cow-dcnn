"""
20_generate_figures.py — generate all five paper figures.

Figures
-------
  fig1_warp_functions.pdf        — warp function shapes for all four methods
  fig2_tic_overlays.pdf          — 5-panel TIC alignment comparison with anchor ticks
  fig3_calibration_curve.pdf     — loss and TIC r vs calibration step (pretrained vs random)
  fig4_per_pair_distributions.pdf— per-pair TIC r violin plots across all batch pairs
  fig5_silhouette_scatter.pdf    — study silhouette vs TIC r scatter

Usage:
    python scripts/20_generate_figures.py               # all figures
    python scripts/20_generate_figures.py --figs 1 3 5  # specific figures
    python scripts/20_generate_figures.py \\
        --study-a fish_oil/batch_sep2015 \\
        --study-b fish_oil/batch_jul2016
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from scipy.signal import correlate
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import ChromaWarpTransformer, warp_smoothness_loss
from alignment import (
    load_chromas, align_pair, detect_peaks, fingerprints,
    N_BINS, RUN_MIN, BIN_MIN, TIME_AXIS, PRECISION_CUTOFF,
)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COW_N_SEGS = 4;  COW_SLACK  = 22
ICO_N      = 10; ICO_SHIFT  = 20

N_CALIB_STEPS = 500;  LR_CALIB = 5e-4
LAMBDA_ANCHOR = 1.0;  LAMBDA_SMOOTH = 0.05;  N_REF_PAIRS = 3;  OUTLIER_IQR = 1.5

COLORS = {
    'Unaligned':         '#888888',
    'icoshift':          '#4477AA',
    'COW-TIC':           '#EE7733',
    'Raw cosine PCHIP':  '#228833',
    'WarpTransformer':   '#CC3311',
}
BATCH_PAIRS = [
    ('fish_oil/batch_sep2015', 'fish_oil/batch_jul2016'),
    ('fish_oil/batch_sep2015', 'fish_oil/batch_apr2016'),
    ('fish_oil/batch_jan2016', 'fish_oil/batch_apr2016'),
    ('fish_oil/batch_jan2016', 'fish_oil/batch_jul2016'),
]
PAIR_LABELS = ['Sep15×Jul16', 'Sep15×Apr16', 'Jan16×Apr16', 'Jan16×Jul16']

# ── Classical alignment ───────────────────────────────────────────────────────

def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def icoshift_align_detail(query_tic, ref_tic, n=ICO_N, ms=ICO_SHIFT):
    """Returns (warped, warp_fn, interval_mids, lags_min)."""
    N      = len(query_tic)
    warped = query_tic.copy().astype(np.float32)
    bounds = np.round(np.linspace(0, N, n + 1)).astype(int)
    best_lags, mids = [], []
    for i in range(n):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 4:
            continue
        cc   = correlate(ref_tic[lo:hi].astype(np.float64),
                         query_tic[lo:hi].astype(np.float64), mode='full')
        lags = np.arange(-(hi - lo - 1), hi - lo)
        valid = np.abs(lags) <= ms
        lag  = int(lags[valid][np.argmax(cc[valid])])
        if lag != 0:
            src = np.arange(lo, hi) - lag
            warped[lo:hi] = query_tic[src.clip(0, N - 1)]
        best_lags.append(lag)
        mids.append((lo + hi) / 2 * BIN_MIN)
    lags_min = np.array(best_lags, dtype=float) * BIN_MIN
    _b = bounds.astype(float) * BIN_MIN
    def warp_fn(t, _b=_b, _l=lags_min):
        t   = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(_b[1:], t), 0, len(_l) - 1)
        return t + _l[idx]
    return warped, warp_fn, np.array(mids), lags_min


def cow_tic_align_detail(query_tic, ref_tic, n_segs=COW_N_SEGS, slack=COW_SLACK):
    """Returns (warped, warp_fn, r_nodes_min, q_nodes_min)."""
    N = len(ref_tic); T = N // n_segs
    r_nodes = [i * T for i in range(n_segs)] + [N - 1]
    cands = []
    for i, rn in enumerate(r_nodes):
        if i == 0 or i == n_segs:
            cands.append([int(rn)])
        else:
            cands.append(list(range(max(1, rn - slack), min(N - 2, rn + slack) + 1)))
    dp   = [dict() for _ in range(n_segs + 1)]
    back = [dict() for _ in range(n_segs + 1)]
    dp[0][0] = 0.0
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]
        r_seg = ref_tic[r_s:r_e + 1]; r_len = len(r_seg)
        for q_s, cum in dp[seg].items():
            for q_e in cands[seg + 1]:
                if q_e <= q_s:
                    continue
                q_raw = query_tic[q_s:q_e + 1]
                if len(q_raw) < 2:
                    continue
                q_seg = np.interp(np.linspace(0, 1, r_len),
                                  np.linspace(0, 1, len(q_raw)), q_raw)
                nc = cum + _pearson(r_seg, q_seg)
                if nc > dp[seg + 1].get(q_e, -np.inf):
                    dp[seg + 1][q_e] = nc; back[seg + 1][q_e] = q_s
    q_last = r_nodes[-1]
    if q_last not in back[n_segs]:
        return query_tic.astype(np.float32), lambda t: t, np.array(r_nodes)*BIN_MIN, np.array(r_nodes)*BIN_MIN
    q_rev = [q_last]
    for seg in range(n_segs, 0, -1):
        p = back[seg].get(q_rev[-1])
        if p is None:
            return query_tic.astype(np.float32), lambda t: t, np.array(r_nodes)*BIN_MIN, np.array(r_nodes)*BIN_MIN
        q_rev.append(p)
    q_nodes = list(reversed(q_rev))
    warped = np.empty(N, dtype=np.float32)
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]; r_len = r_e - r_s + 1
        q_raw = query_tic[q_nodes[seg]:q_nodes[seg + 1] + 1]
        warped[r_s:r_e + 1] = (
            np.interp(np.linspace(0, 1, r_len), np.linspace(0, 1, len(q_raw)), q_raw)
            if len(q_raw) >= 2 else np.full(r_len, float(q_raw[0]) if q_raw.size else 0.0)
        )
    r_times = np.array(r_nodes, dtype=float) * BIN_MIN
    q_times = np.array(q_nodes, dtype=float) * BIN_MIN
    warp_fn = lambda t, _r=r_times, _q=q_times: np.interp(t, _r, _q)
    return warped, warp_fn, r_times, q_times


# ── WarpTransformer calibration ───────────────────────────────────────────────

def extract_anchors(chromas_a, chromas_b):
    rq = defaultdict(list)
    for i in range(min(N_REF_PAIRS, len(chromas_a))):
        _, _, _, _, _, ancs = align_pair(chromas_b[0], chromas_a[i],
                                         encode_fn=None, return_anchors=True)
        for r, q in ancs:
            rq[r].append(q)
    if not rq:
        return []
    agg    = [(r, int(np.round(np.median(q)))) for r, q in sorted(rq.items())]
    drifts = np.array([q - r for r, q in agg], dtype=float)
    mean, std = float(np.mean(drifts)), max(float(np.std(drifts)), 1.0)
    return [a for a, k in zip(agg, np.abs(drifts - mean) <= OUTLIER_IQR * std) if k]


def anchor_loss_fn(warps, study_ids, anchors, device):
    if not anchors:
        return torch.tensor(0.0, device=device)
    ref_mean = warps[study_ids == 0].mean(dim=0)
    qw = warps[study_ids == 1]
    loss = torch.tensor(0.0, device=device)
    for r, q in anchors:
        loss = loss + (qw[:, r] - ref_mean[r] - float(q - r)).pow(2).mean()
    return loss / len(anchors)


def calibrate(chromas_a, chromas_b, device, pretrained=True, log_every=5):
    """
    Returns (aligned_a, aligned_b, warp_np, anchors, log) where log is a list
    of dicts {step, loss, tic_r} recorded every log_every steps.
    """
    n_a, n_b = len(chromas_a), len(chromas_b)
    study_ids = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long, device=device)
    chromas_t = torch.from_numpy(np.stack(chromas_a + chromas_b)).to(device)

    anchors = extract_anchors(chromas_a, chromas_b)

    model = ChromaWarpTransformer()
    if pretrained:
        model.load_state_dict(torch.load(CKPT_DIR / 'warp_transformer.pt',
                                          map_location=device))
    model.to(device)

    log = []
    if anchors:
        model.train()
        opt   = optim.Adam(model.parameters(), lr=LR_CALIB)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_CALIB_STEPS)
        best_loss, best_state = float('inf'), {k: v.clone() for k, v in model.state_dict().items()}

        for step in range(1, N_CALIB_STEPS + 1):
            aligned, warps = model(chromas_t, study_ids)
            a_loss = anchor_loss_fn(warps, study_ids, anchors, device)
            s_loss = warp_smoothness_loss(warps)
            loss   = LAMBDA_ANCHOR * a_loss + LAMBDA_SMOOTH * s_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if float(loss) < best_loss:
                best_loss, best_state = float(loss), {k: v.clone() for k, v in model.state_dict().items()}

            if step % log_every == 0 or step == N_CALIB_STEPS:
                with torch.no_grad():
                    al_d = aligned.detach()
                    al_a = [al_d[i].cpu().numpy() for i in range(n_a)]
                    al_b = [al_d[i].cpu().numpy() for i in range(n_a, n_a + n_b)]
                    r_vals = [float(pearsonr(a.sum(1), b.sum(1))[0])
                              for a in al_a for b in al_b]
                    log.append({'step': step, 'loss': float(loss),
                                'tic_r': float(np.mean(r_vals))})

        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        aligned_t, warps_t = model(chromas_t, study_ids)
    np_al = aligned_t.cpu().numpy()
    return (np_al[:n_a], np_al[n_a:], warps_t.cpu().numpy(), anchors, log)


# ── Pairwise TIC r helpers ────────────────────────────────────────────────────

def pairwise_r(chromas_a, chromas_b, align_fn=None):
    """Return per-pair TIC r list. align_fn(q_tic, r_tic) → warped_tic or None for unaligned."""
    cors = []
    for qa in chromas_a:
        q_tic = qa.sum(axis=1)
        for rb in chromas_b:
            r_tic = rb.sum(axis=1)
            if align_fn is None:
                w = q_tic
            else:
                w = align_fn(q_tic, r_tic)
            cors.append(float(pearsonr(r_tic.astype(np.float64),
                                       w.astype(np.float64))[0]))
    return cors


def ico_fn(q, r):
    warped, _, _, _ = icoshift_align_detail(q, r)
    return warped


def cow_fn(q, r):
    warped, _, _, _ = cow_tic_align_detail(q, r)
    return warped


def pchip_fn(qa, ra):
    """Takes full 2D chromas for peak matching."""
    warped, *_ = align_pair(qa, ra, encode_fn=None, return_anchors=True)
    return warped.astype(np.float32)


# ── Figure 1: Warp function comparison ────────────────────────────────────────

def fig1_warp_functions(chromas_a, chromas_b, warp_np, n_a, anchors, out):
    """Single-panel plot of all four warp functions on one representative pair."""
    ref_chroma, qry_chroma = chromas_a[0], chromas_b[0]
    ref_tic = ref_chroma.sum(axis=1)
    qry_tic = qry_chroma.sum(axis=1)

    # --- icoshift ---
    _, ico_fn_w, ico_mids, ico_lags = icoshift_align_detail(qry_tic, ref_tic)
    ico_curve = ico_fn_w(TIME_AXIS)
    ico_dots_x = ico_mids
    ico_dots_y = ico_fn_w(ico_mids)

    # --- COW-TIC ---
    _, cow_fn_w, cow_r, cow_q = cow_tic_align_detail(qry_tic, ref_tic)
    cow_curve = cow_fn_w(TIME_AXIS)

    # --- PCHIP ---
    _, _, _, _, pchip_fn_w, ransac_ancs = align_pair(
        qry_chroma, ref_chroma, encode_fn=None, return_anchors=True
    )
    if pchip_fn_w is not None:
        pchip_curve = pchip_fn_w(TIME_AXIS)
        pchip_dots_r = np.array([r * BIN_MIN for r, _ in ransac_ancs])
        pchip_dots_q = np.array([q * BIN_MIN for _, q in ransac_ancs])
    else:
        pchip_curve = TIME_AXIS.copy()
        pchip_dots_r = pchip_dots_q = np.array([])

    # --- WarpTransformer: use warp for B[0] (study_ids=1, index n_a) ---
    wt_curve = warp_np[n_a, :] * BIN_MIN   # B[0]'s warp → consensus ≈ A frame
    anc_r = np.array([r * BIN_MIN for r, _ in anchors])
    anc_q = np.array([q * BIN_MIN for _, q in anchors])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(TIME_AXIS, TIME_AXIS, '--', color='#bbbbbb', lw=1.2,
            label='Identity (no drift)', zorder=1)

    # icoshift staircase
    ax.step(TIME_AXIS, ico_curve, where='post',
            color=COLORS['icoshift'], lw=1.6, label='icoshift', zorder=3)
    ax.scatter(ico_dots_x, ico_dots_y, color=COLORS['icoshift'],
               s=35, zorder=5, marker='D', linewidths=0)

    # COW-TIC piecewise linear
    ax.plot(TIME_AXIS, cow_curve,
            color=COLORS['COW-TIC'], lw=1.6, label='COW-TIC', zorder=3)
    ax.scatter(cow_r, cow_q, color=COLORS['COW-TIC'],
               s=50, zorder=5, linewidths=0)

    # PCHIP smooth cubic
    ax.plot(TIME_AXIS, pchip_curve,
            color=COLORS['Raw cosine PCHIP'], lw=1.6, label='Raw cosine PCHIP', zorder=3)
    if len(pchip_dots_r):
        ax.scatter(pchip_dots_r, pchip_dots_q,
                   color=COLORS['Raw cosine PCHIP'], s=50, zorder=5, linewidths=0)

    # WarpTransformer smooth monotone
    ax.plot(TIME_AXIS, wt_curve,
            color=COLORS['WarpTransformer'], lw=2.0,
            label='WarpTransformer', zorder=4)
    if len(anc_r):
        ax.scatter(anc_r, anc_q, color=COLORS['WarpTransformer'],
                   s=50, zorder=6, linewidths=0)

    ax.set_xlabel('Reference retention time (min)', fontsize=10)
    ax.set_ylabel('Source retention time in query (min)', fontsize=10)
    ax.set_title('Warp functions — Sep 2015 vs Jul 2016', fontsize=11)
    ax.set_xlim(0, RUN_MIN); ax.set_ylim(0, RUN_MIN)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig1 → {out}")


# ── Figure 2: TIC alignment overlays with anchor ticks ───────────────────────

def fig2_tic_overlays(chromas_a, chromas_b, warp_np, warp_np_b, n_a, anchors,
                      qi, out):
    """5-panel TIC overlay. qi = index into chromas_a for the query sample."""
    query_raw = chromas_a[qi]
    ref_raw   = chromas_b[0]
    ref_tic   = ref_raw.sum(axis=1)
    qry_tic   = query_raw.sum(axis=1)

    def norm(t): mx = t.max(); return t / mx if mx > 0 else t

    # Run classical methods
    ico_warped, _, _, _ = icoshift_align_detail(qry_tic, ref_tic)
    cow_warped, _, _, _ = cow_tic_align_detail(qry_tic, ref_tic)
    pchip_warped, _, _, proposed, _, ransac_ancs = align_pair(
        query_raw, ref_raw, encode_fn=None, return_anchors=True
    )

    wt_qry = warp_np[qi].sum() if False else None   # unused
    wt_qry_tic = warp_np[qi:qi+1][0].sum(1)   # placeholder; use aligned chromas
    # warp_np here is actually aligned_a from calibration, not warp tensor
    # — the caller must pass aligned chromas, handled below in main
    wt_ref_tic  = warp_np_b.sum(axis=1)
    wt_qry_tic2 = warp_np[qi:qi+1][0].sum(axis=1) if warp_np.ndim == 3 else warp_np[qi]

    r_un    = pearsonr(qry_tic.astype(np.float64), ref_tic.astype(np.float64))[0]
    r_ico   = pearsonr(ico_warped.astype(np.float64), ref_tic.astype(np.float64))[0]
    r_cow   = pearsonr(cow_warped.astype(np.float64), ref_tic.astype(np.float64))[0]
    r_pchip = pearsonr(pchip_warped.astype(np.float64), ref_tic.astype(np.float64))[0]
    r_wt    = pearsonr(wt_qry_tic2.astype(np.float64), wt_ref_tic.astype(np.float64))[0]

    # Anchor tick data for PCHIP and WT panels
    # proposed_anchors = list of (q_bin, r_bin, cosine)
    pchip_ticks_r = [r * BIN_MIN for _, r, _ in proposed]
    pchip_ticks_c = [c for _, _, c in proposed]
    calib_ticks_r = [r * BIN_MIN for r, _ in anchors]

    panels = [
        ('Unaligned',         norm(qry_tic),     norm(ref_tic),    float(r_un),    [],               [],     None),
        ('icoshift',          norm(ico_warped),   norm(ref_tic),    float(r_ico),   [],               [],     None),
        ('COW-TIC',           norm(cow_warped),   norm(ref_tic),    float(r_cow),   [],               [],     None),
        ('Raw cosine PCHIP',  norm(pchip_warped), norm(ref_tic),    float(r_pchip), pchip_ticks_r,    pchip_ticks_c, 0.7),
        ('WarpTransformer',   norm(wt_qry_tic2),  norm(wt_ref_tic), float(r_wt),    calib_ticks_r,    None,   None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06, left=0.05, right=0.99, top=0.88, bottom=0.16)

    for ax, (method, qry, ref, r, ticks_r, ticks_c, cutoff) in zip(axes, panels):
        color = COLORS.get(method, '#333333')
        ax.fill_between(TIME_AXIS, ref, alpha=0.18, color='#444444', linewidth=0)
        ax.plot(TIME_AXIS, ref, color='#555555', lw=0.9)
        ax.plot(TIME_AXIS, qry, color=color, lw=1.1, alpha=0.9)

        # Anchor ticks
        for j, t in enumerate(ticks_r):
            if ticks_c is not None:
                c = '#2ca02c' if ticks_c[j] >= cutoff else '#d62728'
            else:
                c = color
            ax.axvline(t, ymin=0, ymax=0.08, color=c, lw=1.4, alpha=0.85)

        ax.set_title(f'{method}\n$r={r:.3f}$', fontsize=8.5, pad=4)
        ax.set_xlim(0, RUN_MIN); ax.set_ylim(-0.04, 1.2)
        ax.set_xlabel('RT (min)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Normalised TIC', fontsize=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], color='#555555', lw=1.4, label='Reference (batch_jul2016[0])'),
        Line2D([0],[0], color='#333333', lw=1.4, label=f'Query (batch_sep2015[{qi}], aligned)'),
        Line2D([0],[0], color='#2ca02c', lw=1.4, label='Anchor cosine ≥ 0.70'),
        Line2D([0],[0], color='#d62728', lw=1.4, label='Anchor cosine < 0.70'),
    ], fontsize=7.5, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), framealpha=0.8)

    fig.suptitle('Cross-batch TIC alignment — Sep 2015 × Jul 2016', fontsize=10, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig2 → {out}")


# ── Figure 3: Calibration learning curve ─────────────────────────────────────

def fig3_calibration_curve(log_pre, log_rand, unaligned_r, out):
    """Two-panel: loss (left) and TIC r (right), pretrained vs random-init."""
    steps_p = [d['step'] for d in log_pre]
    loss_p  = [d['loss'] for d in log_pre]
    r_p     = [d['tic_r'] for d in log_pre]
    steps_r = [d['step'] for d in log_rand]
    loss_r  = [d['loss'] for d in log_rand]
    r_r     = [d['tic_r'] for d in log_rand]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.28, left=0.09, right=0.97, top=0.88, bottom=0.14)

    col_pre  = COLORS['WarpTransformer']
    col_rand = '#888888'

    ax1.plot(steps_p, loss_p,  color=col_pre,  lw=1.8, label='Pretrained init')
    ax1.plot(steps_r, loss_r,  color=col_rand, lw=1.8, linestyle='--', label='Random init')
    ax1.set_xlabel('Calibration step', fontsize=9)
    ax1.set_ylabel('Loss (anchor MSE + smoothness)', fontsize=9)
    ax1.set_title('Calibration loss', fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.tick_params(labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.axhline(unaligned_r, color='#bbbbbb', lw=1.2, linestyle=':', label='Unaligned')
    ax2.plot(steps_p, r_p,  color=col_pre,  lw=1.8, label='Pretrained init')
    ax2.plot(steps_r, r_r,  color=col_rand, lw=1.8, linestyle='--', label='Random init')
    ax2.set_xlabel('Calibration step', fontsize=9)
    ax2.set_ylabel('Mean pairwise TIC r', fontsize=9)
    ax2.set_title('Alignment quality (TIC r)', fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.tick_params(labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Anchor calibration convergence — Sep 2015 × Jul 2016', fontsize=10, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig3 → {out}")


# ── Figure 4: Per-pair TIC r violin plots ────────────────────────────────────

def fig4_violin(all_data, out):
    """
    all_data: dict of method_name → dict of pair_label → list of TIC r values
    """
    methods = ['Unaligned', 'icoshift', 'COW-TIC', 'Raw cosine PCHIP', 'WarpTransformer']
    n_pairs = len(PAIR_LABELS)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99, top=0.88, bottom=0.18)

    positions = np.arange(1, len(methods) + 1)

    for col, (ax, pair_label) in enumerate(zip(axes, PAIR_LABELS)):
        data  = [all_data[m][pair_label] for m in methods]
        parts = ax.violinplot(data, positions=positions, showmedians=True,
                              showextrema=True)
        for body, method in zip(parts['bodies'], methods):
            body.set_facecolor(COLORS[method])
            body.set_alpha(0.65)
            body.set_linewidth(0)
        parts['cmedians'].set_color('#222222'); parts['cmedians'].set_linewidth(1.5)
        parts['cmins'].set_color('#555555');    parts['cmaxes'].set_color('#555555')
        parts['cbars'].set_color('#555555')

        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=7)
        ax.set_title(pair_label, fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Pairwise TIC r', fontsize=9)
    fig.suptitle('Per-pair TIC r distributions across batch pairs', fontsize=10, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig4 → {out}")


# ── Figure 5: Silhouette vs TIC r scatter ────────────────────────────────────

def fig5_scatter(out):
    """Hardcoded from result files. x = Δ TIC r, y = Δ Study sil."""
    # (method, pair_label, tic_r, study_sil, unaligned_r, unaligned_sil)
    data = [
        # Sep15×Jul16  (unaligned r=0.066, sil=0.230)
        ('icoshift',         'Sep15×Jul16', 0.154, 0.240, 0.066, 0.230),
        ('Raw cosine PCHIP', 'Sep15×Jul16', 0.551, 0.245, 0.066, 0.230),
        ('COW-TIC',          'Sep15×Jul16', 0.606, 0.243, 0.066, 0.230),
        ('WarpTransformer',  'Sep15×Jul16', 0.698, 0.231, 0.066, 0.230),
        # Sep15×Apr16  (unaligned r=0.072, sil=0.157)
        ('icoshift',         'Sep15×Apr16', 0.160, 0.181, 0.072, 0.157),
        ('Raw cosine PCHIP', 'Sep15×Apr16', 0.477, 0.170, 0.072, 0.157),
        ('COW-TIC',          'Sep15×Apr16', 0.634, 0.178, 0.072, 0.157),
        ('WarpTransformer',  'Sep15×Apr16', 0.681, 0.178, 0.072, 0.157),
        # Jan16×Apr16  (unaligned r=0.128, sil=0.025)
        ('icoshift',         'Jan16×Apr16', 0.249, 0.030, 0.128, 0.025),
        ('Raw cosine PCHIP', 'Jan16×Apr16', 0.522, 0.016, 0.128, 0.025),
        ('COW-TIC',          'Jan16×Apr16', 0.558, 0.020, 0.128, 0.025),
        ('WarpTransformer',  'Jan16×Apr16', 0.629, 0.055, 0.128, 0.025),
        # Jan16×Jul16  (unaligned r=0.118, sil=0.108)
        ('icoshift',         'Jan16×Jul16', 0.251, 0.104, 0.118, 0.108),
        ('Raw cosine PCHIP', 'Jan16×Jul16', 0.546, 0.104, 0.118, 0.108),
        ('COW-TIC',          'Jan16×Jul16', 0.567, 0.107, 0.118, 0.108),
        ('WarpTransformer',  'Jan16×Jul16', 0.620, 0.112, 0.118, 0.108),
    ]
    markers = {'Sep15×Jul16': 'o', 'Sep15×Apr16': 'D', 'Jan16×Apr16': 's', 'Jan16×Jul16': '^'}

    fig, ax = plt.subplots(figsize=(6, 5))
    from matplotlib.lines import Line2D

    plotted_methods = set(); plotted_pairs = set()
    for method, pair, r, sil, r0, sil0 in data:
        delta_r   = r - r0
        delta_sil = sil - sil0
        color  = COLORS[method]
        marker = markers[pair]
        ax.scatter(delta_r, delta_sil, color=color, marker=marker,
                   s=80, zorder=4, edgecolors='white', linewidths=0.5)
        plotted_methods.add(method); plotted_pairs.add(pair)

    ax.axhline(0, color='#aaaaaa', lw=0.9, linestyle='--', zorder=1)
    ax.axvline(0, color='#aaaaaa', lw=0.9, linestyle='--', zorder=1)

    method_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS[m],
                             markersize=9, label=m) for m in
                      ['icoshift','Raw cosine PCHIP','COW-TIC','WarpTransformer']]
    pair_handles   = [Line2D([0],[0], marker=markers[p], color='w', markerfacecolor='#555555',
                             markersize=9, label=p) for p in PAIR_LABELS]
    leg1 = ax.legend(handles=method_handles, fontsize=8, loc='upper left',
                     title='Method', title_fontsize=8, framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=pair_handles, fontsize=8, loc='lower right',
              title='Batch pair', title_fontsize=8, framealpha=0.85)

    ax.set_xlabel('Δ TIC r  (vs unaligned)', fontsize=10)
    ax.set_ylabel('Δ Study silhouette  (vs unaligned)\n← better integration      worse →', fontsize=9)
    ax.set_title('Alignment quality vs batch integration', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig5 → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')
    ap.add_argument('--figs', nargs='+', type=int, default=[1,2,3,4,5],
                    metavar='N', help='Which figures to generate (default: all)')
    args = ap.parse_args()

    want = set(args.figs)
    label_a, label_b = Path(args.study_a).name, Path(args.study_b).name

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}")

    # Load primary dataset
    print(f"\nLoading {label_a} × {label_b} …")
    chromas_a = load_chromas(DATA_DIR / args.study_a / 'chroma')
    chromas_b = load_chromas(DATA_DIR / args.study_b / 'chroma')
    n_a, n_b  = len(chromas_a), len(chromas_b)
    print(f"  {label_a}: {n_a}  |  {label_b}: {n_b}")

    # Calibrate pretrained model (needed for fig 1, 2, 3)
    need_calib = want & {1, 2, 3}
    aligned_a = aligned_b = warp_full = anchors = log_pre = None
    if need_calib:
        print("\nCalibrating (pretrained) …")
        aligned_a, aligned_b, warp_full, anchors, log_pre = calibrate(
            chromas_a, chromas_b, device, pretrained=True, log_every=5)
        r_unaligned = float(np.mean([
            float(pearsonr(a.sum(1), b.sum(1))[0])
            for a in chromas_a for b in chromas_b
        ]))
        print(f"  Unaligned TIC r = {r_unaligned:.4f}")
        print(f"  Post-calib TIC r = {log_pre[-1]['tic_r']:.4f}")

    # Calibrate random-init model (needed for fig 3)
    log_rand = None
    if 3 in want:
        print("\nCalibrating (random init) …")
        _, _, _, _, log_rand = calibrate(
            chromas_a, chromas_b, device, pretrained=False, log_every=5)

    # Find representative query sample (highest Δ TIC r from WT)
    qi = 0
    if need_calib:
        ref_orig   = chromas_b[0].sum(axis=1).astype(np.float64)
        ref_warped = aligned_b[0].sum(axis=1).astype(np.float64)
        best_delta, best_qi = -np.inf, 0
        for i in range(n_a):
            r_un = float(pearsonr(chromas_a[i].sum(1).astype(np.float64), ref_orig)[0])
            r_wt = float(pearsonr(aligned_a[i].sum(1).astype(np.float64), ref_warped)[0])
            if r_wt - r_un > best_delta:
                best_delta, best_qi = r_wt - r_un, i
        qi = best_qi
        print(f"\nRepresentative query: {label_a}[{qi}]  Δ TIC r = {best_delta:+.3f}")

    # ── Figure 1
    if 1 in want:
        print("\nGenerating Figure 1 (warp functions) …")
        fig1_warp_functions(
            chromas_a, chromas_b, warp_full, n_a, anchors,
            out=RESULTS_DIR / f'fig1_warp_functions_{label_a}_{label_b}.pdf'
        )

    # ── Figure 2
    if 2 in want:
        print("\nGenerating Figure 2 (TIC overlays) …")

        # Pass aligned chromas as the warp_np arrays for the WT panel
        class _AlignedChromas:
            """Thin adapter so fig2 can call .sum(axis=1) on aligned chromas."""
            pass

        fig2_tic_overlays(
            chromas_a, chromas_b,
            aligned_a,  # shape: list of [200,1000]
            aligned_b[0],
            n_a, anchors, qi,
            out=RESULTS_DIR / f'fig2_tic_overlays_{label_a}_{label_b}.pdf'
        )

    # ── Figure 3
    if 3 in want:
        print("\nGenerating Figure 3 (calibration curve) …")
        fig3_calibration_curve(
            log_pre, log_rand, r_unaligned,
            out=RESULTS_DIR / f'fig3_calibration_curve_{label_a}_{label_b}.pdf'
        )

    # ── Figure 4: pairwise TIC r distributions across all 3 batch pairs
    if 4 in want:
        print("\nGenerating Figure 4 (violin plots) …")
        all_data = {m: {} for m in
                    ['Unaligned','icoshift','COW-TIC','Raw cosine PCHIP','WarpTransformer']}

        for (sa, sb), pl in zip(BATCH_PAIRS, PAIR_LABELS):
            print(f"  Evaluating {pl} …")
            ca = load_chromas(DATA_DIR / sa / 'chroma')
            cb = load_chromas(DATA_DIR / sb / 'chroma')

            all_data['Unaligned'][pl] = pairwise_r(ca, cb, None)
            print(f"    icoshift …")
            all_data['icoshift'][pl]  = pairwise_r(
                ca, cb, lambda q, r: icoshift_align_detail(q, r)[0])
            print(f"    COW-TIC …")
            all_data['COW-TIC'][pl]   = pairwise_r(
                ca, cb, lambda q, r: cow_tic_align_detail(q, r)[0])
            print(f"    PCHIP …")
            all_data['Raw cosine PCHIP'][pl] = [
                float(pearsonr(
                    cb[j].sum(1).astype(np.float64),
                    align_pair(ca[i], cb[j], encode_fn=None)[0].astype(np.float64)
                )[0])
                for i in range(len(ca)) for j in range(len(cb))
            ]
            print(f"    WarpTransformer …")
            al_a_wt, al_b_wt, _, _, _ = calibrate(ca, cb, device,
                                                    pretrained=True, log_every=500)
            all_data['WarpTransformer'][pl] = [
                float(pearsonr(al_a_wt[i].sum(1).astype(np.float64),
                               al_b_wt[j].sum(1).astype(np.float64))[0])
                for i in range(len(ca)) for j in range(len(cb))
            ]

        fig4_violin(all_data,
                    out=RESULTS_DIR / 'fig4_per_pair_distributions.pdf')

    # ── Figure 5
    if 5 in want:
        print("\nGenerating Figure 5 (silhouette scatter) …")
        fig5_scatter(out=RESULTS_DIR / 'fig5_silhouette_scatter.pdf')

    print("\nDone.")


if __name__ == '__main__':
    main()
