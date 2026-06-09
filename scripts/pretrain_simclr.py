"""
SimCLR pretraining on GC-MS EI-MS spectra.

Stage 1: pretrain on 9,553 MoNA spectra (diverse EI-MS, general features).
Stage 2: fine-tune on chromatogram peak spectra from fish_oil + mtbls288
         (domain-specific, actual GC-MS peaks from the target instrument).

Augmentations simulate real measurement variability:
  - Gaussian noise on intensities
  - Random bin masking (missing ions)
  - Per-bin intensity jitter (detector response variation)

Usage:
    python scripts/pretrain_simclr.py
"""

import sys
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import SpectrumEncoder, nt_xent_loss

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent.parent / 'data'
CKPT_DIR  = Path(__file__).parent.parent / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

# ── Augmentation ───────────────────────────────────────────────────────────
def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to a single m/z spectrum (1000-dim, L2-normed)."""
    x = x.copy().astype(np.float32)

    # Gaussian noise
    if rng.random() > 0.2:
        noise = rng.normal(0, 0.04, size=x.shape).astype(np.float32)
        x = np.clip(x + noise, 0, None)

    # Random bin masking
    if rng.random() > 0.2:
        mask = rng.random(size=x.shape) < 0.12
        x[mask] = 0.0

    # Per-bin intensity jitter on non-zero bins
    if rng.random() > 0.3:
        scale = rng.uniform(0.7, 1.3, size=x.shape).astype(np.float32)
        x = x * scale

    # Re-normalise
    norm = np.linalg.norm(x)
    if norm > 0:
        x /= norm
    return x


class SimCLRDataset(Dataset):
    def __init__(self, spectra: np.ndarray, seed: int = 0):
        self.spectra = spectra.astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        x = self.spectra[idx]
        x1 = augment(x, self.rng)
        x2 = augment(x, self.rng)
        return torch.from_numpy(x1), torch.from_numpy(x2)


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def extract_peaks(chroma_dir: Path, n_peaks: int = 12) -> np.ndarray:
    """Extract L2-normalised m/z spectra at TIC peaks from all .npz files."""
    from scipy.signal import find_peaks as sp_find_peaks
    spectra = []
    for p in sorted(chroma_dir.glob('*.npz')):
        chroma = np.load(p)['chroma']          # (200, 1000)
        tic = chroma.sum(axis=1)
        threshold = np.percentile(tic, 80)
        peaks, props = sp_find_peaks(tic, height=threshold, distance=5)
        if len(peaks) > n_peaks:
            top = np.argsort(props['peak_heights'])[-n_peaks:]
            peaks = peaks[top]
        for pk in peaks:
            spectrum = chroma[pk].copy()
            norm = np.linalg.norm(spectrum)
            if norm > 0:
                spectra.append(spectrum / norm)
    return np.array(spectra, dtype=np.float32)


# ── Stage 1: pretrain on MoNA ──────────────────────────────────────────────
def stage1(device, epochs=150, batch_size=256, lr=3e-4):
    import h5py
    print("Stage 1: pretraining on MoNA spectra")
    with h5py.File(DATA_DIR / 'pretraining' / 'spectra.h5') as f:
        spectra = f['spectra'][:]

    ds = SimCLRDataset(spectra)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SpectrumEncoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, dl, opt, device)
        sched.step()
        if epoch % 25 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    torch.save(model.state_dict(), CKPT_DIR / 'mona_simclr.pt')
    print(f"  saved → {CKPT_DIR}/mona_simclr.pt\n")
    return model


# ── Stage 2: fine-tune on chromatogram peaks ───────────────────────────────
def stage2(model, device, epochs=200, batch_size=64, lr=5e-5):
    print("Stage 2: fine-tuning on fish_oil + mtbls288 chromatogram peaks")
    fish_peaks = extract_peaks(DATA_DIR / 'fish_oil'  / 'chroma')
    m288_peaks = extract_peaks(DATA_DIR / 'mtbls288'  / 'chroma')
    all_peaks  = np.concatenate([fish_peaks, m288_peaks], axis=0)
    print(f"  {len(fish_peaks)} fish_oil peaks + {len(m288_peaks)} mtbls288 peaks "
          f"= {len(all_peaks)} total")

    ds = SimCLRDataset(all_peaks)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, dl, opt, device)
        sched.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    torch.save(model.state_dict(), CKPT_DIR / 'gcms_simclr.pt')
    print(f"  saved → {CKPT_DIR}/gcms_simclr.pt\n")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = stage1(device)
    stage2(model, device)
    print("Done.")
