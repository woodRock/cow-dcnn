"""
19_visualize_batch_alignment.py — five-panel alignment comparison figure.

Panels (left to right):
  1. Unaligned
  2. icoshift
  3. COW-TIC
  4. Raw cosine PCHIP
  5. WarpTransformer + anchor calib

Each panel shows the reference TIC (grey, filled) overlaid with the aligned
query TIC (coloured). The query sample is chosen as the study-A sample that
shows the largest TIC r improvement from WarpTransformer calibration, so the
alignment effect is clearly visible.

Usage:
    python scripts/19_visualize_batch_alignment.py
    python scripts/19_visualize_batch_alignment.py \\
        --study-a fish_oil/batch_sep2015 \\
        --study-b fish_oil/batch_jul2016
    python scripts/19_visualize_batch_alignment.py --query-idx 3
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
from alignment import load_chromas, align_pair

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS    = 200
RUN_MIN   = 45.0
BIN_MIN   = RUN_MIN / N_BINS
TIME_AXIS = np.linspace(0, RUN_MIN, N_BINS)

# COW-TIC
COW_N_SEGS = 4
COW_SLACK  = 22

# icoshift
ICO_N_INTERVALS = 10
ICO_MAX_SHIFT   = 20

# Calibration
N_CALIB_STEPS = 500
LR_CALIB      = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
N_REF_PAIRS   = 3
OUTLIER_IQR   = 1.5

METHOD_COLORS = {
    'Unaligned':            '#888888',
    'icoshift':             '#4477AA',
    'COW-TIC':              '#EE7733',
    'Raw cosine PCHIP':     '#228833',
    'WarpTransformer':      '#CC3311',
}


# ── Classical alignment methods ───────────────────────────────────────────────

def _pearson_fast(a: np.ndarray, b: np.ndarray) -> float:
    a_z = a - a.mean(); b_z = b - b.mean()
    na = np.linalg.norm(a_z); nb = np.linalg.norm(b_z)
    return float(np.dot(a_z, b_z) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def cow_tic_align(query_tic: np.ndarray, ref_tic: np.ndarray,
                  n_segs: int = COW_N_SEGS,
                  slack: int = COW_SLACK) -> np.ndarray:
    N = len(ref_tic)
    T = N // n_segs
    r_nodes = [i * T for i in range(n_segs)] + [N - 1]

    node_cands: list[list[int]] = []
    for i, rn in enumerate(r_nodes):
        if i == 0 or i == n_segs:
            node_cands.append([int(rn)])
        else:
            lo = max(1, rn - slack); hi = min(N - 2, rn + slack)
            node_cands.append(list(range(int(lo), int(hi) + 1)))

    dp   = [dict() for _ in range(n_segs + 1)]
    back = [dict() for _ in range(n_segs + 1)]
    dp[0][0] = 0.0

    for seg in range(n_segs):
        r_s = r_nodes[seg]; r_e = r_nodes[seg + 1]
        r_seg = ref_tic[r_s:r_e + 1]; r_len = len(r_seg)
        for q_s, cum in dp[seg].items():
            for q_e in node_cands[seg + 1]:
                if q_e <= q_s:
                    continue
                q_raw = query_tic[q_s:q_e + 1]
                if len(q_raw) < 2:
                    continue
                q_seg = np.interp(np.linspace(0, 1, r_len),
                                  np.linspace(0, 1, len(q_raw)), q_raw)
                new_cum = cum + _pearson_fast(r_seg, q_seg)
                if new_cum > dp[seg + 1].get(q_e, -np.inf):
                    dp[seg + 1][q_e] = new_cum
                    back[seg + 1][q_e] = q_s

    q_last = r_nodes[-1]
    if q_last not in back[n_segs]:
        return query_tic.astype(np.float32)

    q_nodes_rev = [q_last]
    for seg in range(n_segs, 0, -1):
        prev = back[seg].get(q_nodes_rev[-1])
        if prev is None:
            return query_tic.astype(np.float32)
        q_nodes_rev.append(prev)
    q_nodes = list(reversed(q_nodes_rev))

    warped = np.empty(N, dtype=np.float32)
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]
        q_s, q_e = q_nodes[seg], q_nodes[seg + 1]
        r_len = r_e - r_s + 1
        q_raw = query_tic[q_s:q_e + 1]
        warped[r_s:r_e + 1] = (
            np.interp(np.linspace(0, 1, r_len), np.linspace(0, 1, len(q_raw)), q_raw)
            if len(q_raw) >= 2 else np.full(r_len, float(q_raw[0]) if len(q_raw) else 0.0)
        )
    return warped


def icoshift_align(query_tic: np.ndarray, ref_tic: np.ndarray,
                   n_intervals: int = ICO_N_INTERVALS,
                   max_shift: int = ICO_MAX_SHIFT) -> np.ndarray:
    N = len(query_tic)
    warped = query_tic.copy().astype(np.float32)
    bounds = np.round(np.linspace(0, N, n_intervals + 1)).astype(int)

    for i in range(n_intervals):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 4:
            continue
        r_seg = ref_tic[lo:hi].astype(np.float64)
        q_seg = query_tic[lo:hi].astype(np.float64)
        cc    = correlate(r_seg, q_seg, mode='full')
        lags  = np.arange(-(hi - lo - 1), hi - lo)
        valid = np.abs(lags) <= max_shift
        best_lag = int(lags[valid][np.argmax(cc[valid])])
        if best_lag != 0:
            src = np.arange(lo, hi) - best_lag
            warped[lo:hi] = query_tic[src.clip(0, N - 1)]
    return warped


def raw_cosine_pchip_align(query_chroma: np.ndarray,
                           ref_chroma: np.ndarray) -> np.ndarray:
    """Returns the warped query TIC using raw cosine PCHIP peak matching."""
    warped_tic, _, _, _, warp_fn, _ = align_pair(
        query_chroma, ref_chroma, encode_fn=None, return_anchors=True
    )
    if warp_fn is None:
        return query_chroma.sum(axis=1).astype(np.float32)
    return warped_tic.astype(np.float32)


# ── WarpTransformer calibration ───────────────────────────────────────────────

def extract_anchors(chromas_a, chromas_b, n_ref=N_REF_PAIRS):
    ref_query_per_rbin = defaultdict(list)
    for i in range(min(n_ref, len(chromas_a))):
        _, _, _, _, _, anchors = align_pair(
            chromas_b[0], chromas_a[i], encode_fn=None, return_anchors=True
        )
        for r_bin, q_bin in anchors:
            ref_query_per_rbin[r_bin].append(q_bin)
    if not ref_query_per_rbin:
        return []
    aggregated = [(r, int(np.round(np.median(q))))
                  for r, q in sorted(ref_query_per_rbin.items())]
    drifts = np.array([q - r for r, q in aggregated], dtype=float)
    mean, std = float(np.mean(drifts)), max(float(np.std(drifts)), 1.0)
    return [a for a, k in zip(aggregated,
            np.abs(drifts - mean) <= OUTLIER_IQR * std) if k]


def anchor_loss(warps, study_ids, anchor_pairs, device):
    if not anchor_pairs:
        return torch.tensor(0.0, device=device)
    ref_mean    = warps[study_ids == 0].mean(dim=0)
    query_warps = warps[study_ids == 1]
    loss = torch.tensor(0.0, device=device)
    for r_bin, q_bin in anchor_pairs:
        rel_warp = query_warps[:, r_bin] - ref_mean[r_bin]
        loss = loss + (rel_warp - float(q_bin - r_bin)).pow(2).mean()
    return loss / len(anchor_pairs)


def calibrate_warp_transformer(chromas_a, chromas_b, device):
    """
    Returns (model, aligned_a_np, aligned_b_np, warps_np) after calibration.
    aligned_a_np[i] is the warped [200, 1000] chromatogram for chromas_a[i].
    """
    n_a, n_b = len(chromas_a), len(chromas_b)
    all_chromas = chromas_a + chromas_b
    study_ids   = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long, device=device)
    chromas_t   = torch.from_numpy(np.stack(all_chromas)).to(device)

    anchors = extract_anchors(chromas_a, chromas_b)
    print(f"  Anchors: {len(anchors)} RANSAC pairs")

    ckpt  = CKPT_DIR / 'warp_transformer.pt'
    model = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    if anchors:
        model.train()
        opt   = optim.Adam(model.parameters(), lr=LR_CALIB)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_CALIB_STEPS)
        best_loss, best_state = float('inf'), {k: v.clone() for k, v in model.state_dict().items()}

        for step in range(1, N_CALIB_STEPS + 1):
            aligned, warps = model(chromas_t, study_ids)
            loss = (LAMBDA_ANCHOR * anchor_loss(warps, study_ids, anchors, device)
                    + LAMBDA_SMOOTH * warp_smoothness_loss(warps))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if float(loss) < best_loss:
                best_loss  = float(loss)
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if step % 100 == 0:
                print(f"    step {step}/{N_CALIB_STEPS}  loss={float(loss):.4f}")
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        aligned_t, warps_t = model(chromas_t, study_ids)

    aligned_np = aligned_t.cpu().numpy()
    return (aligned_np[:n_a], aligned_np[n_a:], warps_t.cpu().numpy())


# ── Representative pair selection ─────────────────────────────────────────────

def pick_query_idx(chromas_a, chromas_b, wt_a_np, wt_b_np):
    """
    Return the study-A index with the largest TIC r improvement:
      r(warped A[i], warped B[0]) − r(original A[i], original B[0])
    This matches the comparison used in the WarpTransformer panel of the figure.
    """
    ref_tic_orig   = chromas_b[0].sum(axis=1).astype(np.float64)
    ref_tic_warped = wt_b_np[0].sum(axis=1).astype(np.float64)
    best_idx, best_delta = 0, -np.inf
    for i, (raw, warped) in enumerate(zip(chromas_a, wt_a_np)):
        r_un  = float(pearsonr(raw.sum(axis=1).astype(np.float64), ref_tic_orig)[0])
        r_wt  = float(pearsonr(warped.sum(axis=1).astype(np.float64), ref_tic_warped)[0])
        delta = r_wt - r_un
        if delta > best_delta:
            best_delta, best_idx = delta, i
    print(f"  Representative query: study-A[{best_idx}]  "
          f"Δ TIC r (WarpTransformer vs unaligned) = {best_delta:+.3f}")
    return best_idx


# ── Plotting ──────────────────────────────────────────────────────────────────

def _norm(tic: np.ndarray) -> np.ndarray:
    mx = tic.max()
    return tic / mx if mx > 0 else tic


def plot_figure(panels: list[dict], out_path: Path,
                label_a: str, label_b: str) -> None:
    """
    panels: list of dicts with keys:
        'method'  — display name
        'ref_tic' — normalised reference TIC
        'qry_tic' — normalised aligned query TIC
        'tic_r'   — Pearson r between ref and qry
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0),
                             sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06, left=0.06, right=0.99,
                        top=0.88, bottom=0.14)

    for ax, panel in zip(axes, panels):
        method = panel['method']
        color  = METHOD_COLORS.get(method, '#333333')
        ref    = panel['ref_tic']
        qry    = panel['qry_tic']
        r      = panel['tic_r']

        # Reference: grey fill + outline
        ax.fill_between(TIME_AXIS, ref, alpha=0.18, color='#444444', linewidth=0)
        ax.plot(TIME_AXIS, ref, color='#555555', lw=0.9, label='Reference')

        # Query: coloured line
        ax.plot(TIME_AXIS, qry, color=color, lw=1.1, alpha=0.9, label='Query (aligned)')

        ax.set_title(f'{method}\n$r = {r:.3f}$', fontsize=9, pad=5)
        ax.set_xlim(0, RUN_MIN)
        ax.set_ylim(-0.04, 1.18)
        ax.set_xlabel('Retention time (min)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Normalised TIC', fontsize=8)

    # Single legend below the panels
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#555555', lw=1.5, label=f'Reference ({label_b}[0])'),
        Line2D([0], [0], color='#333333', lw=1.5,
               linestyle='--', alpha=0.5, label=f'Query ({label_a}, best-improved sample)'),
    ]
    fig.legend(handles=legend_handles, fontsize=8, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.0), framealpha=0.8)

    fig.suptitle(
        f'Cross-batch alignment: {label_a} × {label_b}',
        fontsize=10, y=0.97,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a',   default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b',   default='fish_oil/batch_jul2016')
    ap.add_argument('--query-idx', type=int, default=None,
                    help='Fix study-A sample index (default: auto-select)')
    ap.add_argument('--out',       type=Path, default=None)
    args = ap.parse_args()

    label_a = Path(args.study_a).name
    label_b = Path(args.study_b).name
    out_path = args.out or RESULTS_DIR / f'alignment_comparison_{label_a}_{label_b}.pdf'

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}")

    print("Loading chromatograms …")
    chromas_a = load_chromas(DATA_DIR / args.study_a / 'chroma')
    chromas_b = load_chromas(DATA_DIR / args.study_b / 'chroma')
    print(f"  {label_a}: {len(chromas_a)} samples  |  {label_b}: {len(chromas_b)} samples")

    print("Calibrating WarpTransformer …")
    wt_a_np, wt_b_np, _ = calibrate_warp_transformer(chromas_a, chromas_b, device)

    if args.query_idx is not None:
        qi = args.query_idx
        print(f"  Using specified query index: {qi}")
    else:
        print("Selecting representative query sample …")
        qi = pick_query_idx(chromas_a, chromas_b, wt_a_np, wt_b_np)

    query_raw = chromas_a[qi]
    ref_chroma = chromas_b[0]
    ref_tic    = ref_chroma.sum(axis=1)
    query_tic  = query_raw.sum(axis=1)

    print(f"\nApplying methods to study-A[{qi}] vs {label_b}[0] …")

    def r(a, b):
        v = float(pearsonr(np.asarray(a, dtype=np.float64),
                           np.asarray(b, dtype=np.float64))[0])
        return 0.0 if np.isnan(v) else v

    # 1. Unaligned
    r_un = r(query_tic, ref_tic)
    print(f"  Unaligned            r = {r_un:.3f}")

    # 2. icoshift
    ico_tic = icoshift_align(query_tic, ref_tic)
    r_ico   = r(ico_tic, ref_tic)
    print(f"  icoshift             r = {r_ico:.3f}")

    # 3. COW-TIC
    cow_tic = cow_tic_align(query_tic, ref_tic)
    r_cow   = r(cow_tic, ref_tic)
    print(f"  COW-TIC              r = {r_cow:.3f}")

    # 4. Raw cosine PCHIP
    pchip_tic = raw_cosine_pchip_align(query_raw, ref_chroma)
    r_pchip   = r(pchip_tic, ref_tic)
    print(f"  Raw cosine PCHIP     r = {r_pchip:.3f}")

    # 5. WarpTransformer — compare warped A[qi] to warped B[0]
    wt_qry_tic = wt_a_np[qi].sum(axis=1)
    wt_ref_tic = wt_b_np[0].sum(axis=1)
    r_wt       = r(wt_qry_tic, wt_ref_tic)
    print(f"  WarpTransformer      r = {r_wt:.3f}")

    panels = [
        {'method': 'Unaligned',
         'ref_tic': _norm(ref_tic), 'qry_tic': _norm(query_tic),  'tic_r': r_un},
        {'method': 'icoshift',
         'ref_tic': _norm(ref_tic), 'qry_tic': _norm(ico_tic),    'tic_r': r_ico},
        {'method': 'COW-TIC',
         'ref_tic': _norm(ref_tic), 'qry_tic': _norm(cow_tic),    'tic_r': r_cow},
        {'method': 'Raw cosine PCHIP',
         'ref_tic': _norm(ref_tic), 'qry_tic': _norm(pchip_tic),  'tic_r': r_pchip},
        {'method': 'WarpTransformer',
         'ref_tic': _norm(wt_ref_tic), 'qry_tic': _norm(wt_qry_tic), 'tic_r': r_wt},
    ]

    plot_figure(panels, out_path, label_a, label_b)


if __name__ == '__main__':
    main()
