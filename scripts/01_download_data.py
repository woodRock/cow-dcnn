"""
Download raw GC-MS data for cow-dcnn.

Datasets:
  fish_oil  — New Zealand fish species GC-MS (4 species, 103 samples)
              Source: same lab data used in chroma-dcnn; copy from there if available.
  mtbls288  — MTBLS288 rice grain GC-MS (MetaboLights)
              Hu et al. (2016) Scientific Reports 6:20942
              80 samples: 4 cultivars × 5 developmental stages × 4 bio reps

Usage:
  python scripts/01_download_data.py                    # download all
  python scripts/01_download_data.py --dataset mtbls288 # one dataset only
  python scripts/01_download_data.py --from-chroma-dcnn # copy from local chroma-dcnn repo
"""

from __future__ import annotations

import argparse
import urllib.request
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'

# ── MTBLS288 FTP URLs ──────────────────────────────────────────────────────
# Raw NetCDF files from MetaboLights public FTP
MTBLS288_FTP_BASE = (
    "ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS288/FILES/"
)
MTBLS288_FILES = [f"13242ob_{i}.cdf" for i in range(1, 82) if i != 30]  # 80 files


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  downloaded {dest.name}")
        return True
    except Exception as e:
        print(f"  FAILED {desc}: {e}")
        return False


def download_mtbls288(raw_dir: Path) -> None:
    print(f"\nDownloading MTBLS288 ({len(MTBLS288_FILES)} files) → {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fname in MTBLS288_FILES:
        url = MTBLS288_FTP_BASE + fname
        if _download_file(url, raw_dir / fname, fname):
            ok += 1
    print(f"  {ok}/{len(MTBLS288_FILES)} files downloaded.")


def copy_from_chroma_dcnn(chroma_dcnn_dir: Path) -> None:
    """Copy processed .npz chroma files from an existing chroma-dcnn checkout."""
    print(f"\nCopying processed data from {chroma_dcnn_dir}")
    for dataset in ['fish_oil', 'mtbls288', 'mtbls71']:
        src_chroma = chroma_dcnn_dir / 'data' / dataset / 'chroma'
        dst_chroma = DATA_DIR / dataset / 'chroma'
        if src_chroma.exists():
            dst_chroma.mkdir(parents=True, exist_ok=True)
            n = 0
            for p in src_chroma.glob('*.npz'):
                shutil.copy2(p, dst_chroma / p.name)
                n += 1
            print(f"  {dataset}/chroma: copied {n} .npz files")
        for fname in ['X.npy', 'y.npy', 'groups.txt', 'sample_ids.txt']:
            src = chroma_dcnn_dir / 'data' / dataset / fname
            dst = DATA_DIR / dataset / fname
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  {dataset}/{fname}: copied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['fish_oil', 'mtbls288', 'all'],
                        default='all')
    parser.add_argument('--from-chroma-dcnn', metavar='PATH', default=None,
                        help='Copy processed .npz files from a local chroma-dcnn checkout '
                             '(skips raw download and preprocessing).')
    args = parser.parse_args()

    if args.from_chroma_dcnn:
        copy_from_chroma_dcnn(Path(args.from_chroma_dcnn))
        return

    if args.dataset in ('mtbls288', 'all'):
        download_mtbls288(DATA_DIR / 'mtbls288' / 'raw')

    if args.dataset in ('fish_oil', 'all'):
        print("\nfish_oil: no public download URL. "
              "Use --from-chroma-dcnn to copy from a local chroma-dcnn checkout, "
              "or place the raw CSV files in data/fish_oil/raw/.")

    print("\nNext step: python scripts/02_preprocess_data.py")


if __name__ == '__main__':
    main()
