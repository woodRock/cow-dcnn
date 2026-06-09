"""
Drift-augmented pretraining for cross-lab GC-MS alignment.

Positive pairs: the m/z spectrum of a peak in the original chromatogram paired
with the same compound's spectrum after a random piecewise-linear time warp.
The encoder must learn to embed the same compound identically regardless of
where it elutes — directly simulating inter-lab retention time drift.

Pipeline:
  1. Load all 2D chromatograms (fish_oil + mtbls288).
  2. Initialise encoder from MoNA SimCLR checkpoint (general EI-MS features).
  3. Fine-tune with drift augmentation: for each peak, apply a random warp,
     extract the spectrum at the warped peak position, and train with NT-Xent.
"""

import sys
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks as sp_find_peaks
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import SpectrumEncoder, nt_xent_loss

DATA_DIR  = Path(__file__).parent.parent / 'data'
CKPT_DIR  = Path(__file__).parent.parent / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

N_BINS    = 200
RUN_MIN   = 45.0


# ── Warp utilities ─────────────────────────────────────────────────────────

def random_warp_fn(max_drift: int = 40, n_interior: int = 4,
                   rng: np.random.Generator = None):
    """
    Return a monotone piecewise-linear forward warp: original_bin → warped_bin.
    max_drift controls the maximum RT shift in time bins (40 bins ≈ 9 min).
    """
    if rng is None:
        rng = np.random.default_rng()
    orig_pts = np.linspace(0, N_BINS, n_interior + 2)
    warp_pts = orig_pts.copy()
    warp_pts[1:-1] += rng.uniform(-max_drift, max_drift, size=n_interior)
    # Enforce strict monotonicity from both ends inward
    warp_pts[0], warp_pts[-1] = 0.0, float(N_BINS)
    for i in range(1, len(warp_pts) - 1):
        lo = warp_pts[i - 1] + 1.0
        hi = warp_pts[i + 1] - 1.0
        warp_pts[i] = np.clip(warp_pts[i], lo, hi) if lo <= hi \
                      else (warp_pts[i - 1] + warp_pts[i + 1]) / 2.0
    return interp1d(orig_pts, warp_pts, kind='linear',
                    bounds_error=False, fill_value=(0.0, float(N_BINS)))


def apply_warp(chroma: np.ndarray, fwd_fn) -> np.ndarray:
    """
    Warp a chromatogram (N_BINS, 1000) along the time axis.
    fwd_fn: original_bin → warped_bin (forward warp).
    Uses backward resampling: warped[t] = chroma[inv_fwd(t)].
    """
    orig_bins = np.arange(N_BINS, dtype=float)
    warp_vals = np.clip(fwd_fn(orig_bins), 0, N_BINS - 1 - 1e-9)
    back_fn   = interp1d(warp_vals, orig_bins, kind='linear',
                         bounds_error=False, fill_value=(0.0, float(N_BINS - 1)))
    tw  = orig_bins
    src = np.clip(back_fn(tw), 0, N_BINS - 1 - 1e-9)
    t0  = src.astype(int)
    t1  = np.minimum(t0 + 1, N_BINS - 1)
    a   = (src - t0)[:, None]                           # (N_BINS, 1)
    return ((1 - a) * chroma[t0] + a * chroma[t1]).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _detect_peaks(chroma, height_pct=80, min_dist=5, n_peaks=12):
    tic = chroma.sum(axis=1)
    thr = np.percentile(tic, height_pct)
    pks, props = sp_find_peaks(tic, height=thr, distance=min_dist)
    if len(pks) > n_peaks:
        top = np.argsort(props['peak_heights'])[-n_peaks:]
        pks = pks[top]
    return np.sort(pks)


class DriftDataset(Dataset):
    """
    Each item is a (spec_original, spec_warped) positive pair.
    The two spectra come from the same peak position before and after a random
    time warp — teaching the encoder to be invariant to RT drift.
    """
    def __init__(self, chromas, pairs_per_chroma=10, max_drift=40, seed=42):
        self.chromas    = chromas
        self.ppc        = pairs_per_chroma
        self.max_drift  = max_drift
        self.rng        = np.random.default_rng(seed)
        self.peaks      = [_detect_peaks(c) for c in chromas]

    def __len__(self):
        return len(self.chromas) * self.ppc

    def __getitem__(self, idx):
        ci     = idx // self.ppc
        chroma = self.chromas[ci]
        pks    = self.peaks[ci]

        if len(pks) == 0:
            z = torch.zeros(1000, dtype=torch.float32)
            return z, z

        # Random forward warp + warped chromatogram
        fwd           = random_warp_fn(max_drift=self.max_drift, rng=self.rng)
        warped_chroma = apply_warp(chroma, fwd)

        # Pick a random peak; extract its spectrum before and after the warp
        pk      = int(self.rng.choice(pks))
        pk_w    = int(np.clip(round(float(fwd(pk))), 0, N_BINS - 1))

        spec_orig = torch.from_numpy(_l2(chroma[pk].copy()).astype(np.float32))
        spec_warp = torch.from_numpy(_l2(warped_chroma[pk_w].copy()).astype(np.float32))
        return spec_orig, spec_warp


# ── Training ───────────────────────────────────────────────────────────────

def load_chromas(dirs):
    chromas = []
    for d in dirs:
        for p in sorted(Path(d).glob('*.npz')):
            chromas.append(np.load(p)['chroma'].astype(np.float32))
    return chromas


def train(device, epochs=300, batch_size=128, lr=1e-4, max_drift=40):
    print("Loading chromatograms...")
    chromas = load_chromas([
        DATA_DIR / 'fish_oil' / 'chroma',
        DATA_DIR / 'mtbls288' / 'chroma',
    ])
    print(f"  {len(chromas)} chromatograms  ({len(chromas)*10} peak pairs/epoch approx)")

    ds = DriftDataset(chromas, pairs_per_chroma=10, max_drift=max_drift)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)

    model = SpectrumEncoder().to(device)
    ckpt  = CKPT_DIR / 'mona_simclr.pt'
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Initialised from {ckpt.name}")

    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x1, x2 in dl:
            x1, x2 = x1.to(device), x2.to(device)
            loss = nt_xent_loss(model(x1), model(x2))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={total/len(dl):.4f}")

    out = CKPT_DIR / 'drift_simclr.pt'
    torch.save(model.state_dict(), out)
    print(f"  saved → {out}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    train(device)
    print("Done.")
