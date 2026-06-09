"""
Preprocess fish oil GC-MS CSVs into uniform-RT-binned 2D chromatograms.

The fish oil dataset contains GC-MS runs from four New Zealand fish species
(103 samples). Raw files are lab-internal CSVs; place them in data/fish_oil/raw/
before running this script, or copy from an existing chroma-dcnn checkout:

  cp -r /path/to/chroma-dcnn/data/fish_oil/raw/ data/fish_oil/raw/

  (Or use: python scripts/01_download_data.py --from-chroma-dcnn /path/to/chroma-dcnn)

Processing pipeline (matches MTBLS288 preprocessing):
  1. Read full 2D chromatogram (4802 scans × 344 m/z channels)
  2. Map m/z channels to 1000-bin grid
  3. Divide into N equal-width RT windows (default N=200, ~24 scans each)
  4. In each window select the highest-TIC scan (zero vector if all-zero)
  5. sqrt + L2 normalise each selected scan
  6. Save as [N, 1000] compressed .npz

Output (data/fish_oil/chroma/):
  {sample_id}.npz    float32 [200, 1000]  chromatogram per sample

Usage:
  python scripts/03_preprocess_fish_oil.py
  python scripts/03_preprocess_fish_oil.py --raw-dir data/fish_oil/raw --n-bins 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MZ_MAX = 1000
_FNAME_RE = re.compile(r"\d+ - (\w{3}\d+) .+\.csv", re.IGNORECASE)


def _read_mz_channels(filepath: Path) -> list[int]:
    with open(filepath, encoding="utf-8") as fh:
        fh.readline()
        row1 = fh.readline()
    return [
        int(float(p.strip()))
        for p in row1.split(",")[2:]
        if p.strip().replace(".", "").isdigit()
    ]


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def _bin_chromatogram(spectra: np.ndarray, tic: np.ndarray, n_bins: int) -> np.ndarray:
    n_scans = len(tic)
    bin_size = n_scans / n_bins
    result = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        start = int(i * bin_size)
        end = min(int((i + 1) * bin_size), n_scans)
        if start >= end:
            continue
        best = start + int(np.argmax(tic[start:end]))
        result[i] = spectra[best]
    return result


def process_file(filepath: Path, mz_channels: list[int], n_bins: int) -> np.ndarray:
    df = pd.read_csv(filepath, skiprows=2, header=None, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    tic = df.iloc[:, 1].values.astype(np.float32)
    raw_mz = df.iloc[:, 2:].values.astype(np.float32)

    spectra = np.zeros((len(tic), MZ_MAX), dtype=np.float32)
    for col_idx, mz in enumerate(mz_channels):
        if 0 <= mz < MZ_MAX and col_idx < raw_mz.shape[1]:
            spectra[:, mz] = raw_mz[:, col_idx]

    chroma = _bin_chromatogram(spectra, tic, n_bins)
    chroma = _sqrt_l2(chroma)
    return chroma


def main(raw_dir: Path, output_dir: Path, n_bins: int) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {raw_dir}")

    if not csv_files:
        print(
            "No CSV files found. The fish oil dataset is not publicly available.\n"
            "Copy raw CSVs to data/fish_oil/raw/ or use:\n"
            "  python scripts/01_download_data.py --from-chroma-dcnn /path/to/chroma-dcnn"
        )
        return

    first_csv = next((f for f in csv_files if _FNAME_RE.match(f.name)), None)
    if first_csv is None:
        raise RuntimeError(
            f"No CSV files matching expected filename pattern found in {raw_dir}.\n"
            f"Expected pattern: '<number> - <speciesID> <description>.csv'"
        )

    mz_channels = _read_mz_channels(first_csv)
    print(f"m/z channels: {len(mz_channels)}, range {mz_channels[0]}–{mz_channels[-1]}")
    print(f"RT bins: {n_bins}  (~{4802 // n_bins} scans per bin)")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for fpath in csv_files:
        if not _FNAME_RE.match(fpath.name):
            continue

        sample_id = fpath.stem.replace(" ", "_").replace("-", "_")
        out_path = output_dir / f"{sample_id}.npz"

        if out_path.exists():
            processed += 1
            continue

        chroma = process_file(fpath, mz_channels, n_bins)
        np.savez_compressed(out_path, chroma=chroma)
        processed += 1

        if processed % 20 == 0:
            print(f"  {processed} done …")

    print(f"\nDone. {processed} samples → {output_dir}")
    print(f"Each sample: chroma shape ({n_bins}, {MZ_MAX})")
    total_mb = sum(p.stat().st_size for p in output_dir.glob("*.npz")) / 1e6
    if total_mb > 0:
        print(f"Total size on disk: {total_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir",    default="data/fish_oil/raw",    type=Path)
    ap.add_argument("--output-dir", default="data/fish_oil/chroma", type=Path)
    ap.add_argument("--n-bins",     default=200, type=int,
                    help="Number of uniform RT bins (default 200)")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_bins)
