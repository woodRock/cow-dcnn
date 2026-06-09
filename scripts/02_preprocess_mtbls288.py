"""
Download and preprocess MTBLS288 rice grain GC-MS data.

Source: MetaboLights MTBLS288
  Hu et al. (2016) "Identification of Conserved and Diverse Metabolic Shifts
  during Rice Grain Development" — Scientific Reports 6:20942
  80 GC-MS samples: 4 cultivars × 5 developmental stages × 4 biological reps.

Classification target: rice cultivar (4 classes)
  0 = Qingfengai  (japonica)
  1 = Nipponbare   (japonica)
  2 = 9311         (indica)
  3 = Nongken 58   (japonica)

Processing pipeline:
  1. Reconstruct dense [n_scans × 1000] spectra from sparse ANDI/netCDF
  2. Bin RT axis into N_BINS (200) equal windows; keep highest-TIC scan per bin
  3. sqrt + L2 normalise each bin
  4. Save per-sample [200, 1000] float32 chromatogram as compressed .npz
  5. Build sum-spectrum X.npy [N, 1000] for baseline methods

Output (data/mtbls288/):
  raw/           raw .cdf files (skipped if already present)
  chroma/        per-sample [200, 1000] chromatogram .npz files
  X.npy          [N, 1000] sqrt+L2-normalised sum spectra
  y.npy          [N] int64 cultivar labels
  groups.txt     biological replicate ID per sample (G-1 … G-4)
  sample_ids.txt CDF stem per sample

Usage:
  python scripts/02_preprocess_mtbls288.py
  python scripts/02_preprocess_mtbls288.py --workers 8
  python scripts/02_preprocess_mtbls288.py --no-download   # process already-downloaded files
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

MZ_MAX = 1000
N_BINS_DEFAULT = 200

FTP_BASE = (
    "ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS288/FILES/"
)
FILES_INDEX_URL = (
    "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS288/FILES/"
)
ASSAY_URL = (
    "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS288/"
    "a_MTBLS288_dynamic_metabolomics_of_rice_grain_metabolite_profiling_mass_spectrometry.txt"
)
SAMPLE_URL = (
    "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS288/"
    "s_MTBLS288.txt"
)

CULTIVAR_LABEL = {
    "Qingfengai": 0,
    "Nipponbare": 1,
    "9311": 2,
    "Nongken 58": 3,
}


def _get_filenames() -> list[str]:
    with urllib.request.urlopen(FILES_INDEX_URL, timeout=30) as r:
        html = r.read().decode()
    return re.findall(r'href="(13242ob_\d+\.cdf)"', html)


def _download(fname: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = FTP_BASE + fname
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _fetch_tsv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _build_cdf_to_meta(assay_rows: list[dict], sample_rows: list[dict]) -> dict[str, dict]:
    sample_meta = {}
    for row in sample_rows:
        cultivar = row.get("Factor Value[different cultivar]", "").strip()
        stage = row.get("Factor Value[rice seed developmental stage]", "").strip()
        sample_name = row.get("Sample Name", "").strip()
        if cultivar and cultivar != "blank" and sample_name:
            sample_meta[sample_name] = {"cultivar": cultivar, "stage": stage}

    cdf_meta = {}
    for row in assay_rows:
        sample_name = row.get("Sample Name", "").strip()
        raw_file = row.get("Raw Spectral Data File", "").strip()
        if sample_name not in sample_meta or not raw_file:
            continue
        cdf_stem = Path(raw_file).stem
        meta = sample_meta[sample_name]
        label = CULTIVAR_LABEL.get(meta["cultivar"])
        if label is None:
            continue
        m = re.search(r"(G-\d+)$", sample_name)
        group = m.group(1) if m else sample_name
        cdf_meta[cdf_stem] = {
            "cultivar": meta["cultivar"],
            "stage": meta["stage"],
            "label": label,
            "group": group,
            "sample_name": sample_name,
        }
    return cdf_meta


def _read_cdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netcdf_file(str(path), "r", mmap=False) as f:
        rt = f.variables["scan_acquisition_time"].data.copy()
        scan_idx = f.variables["scan_index"].data.copy()
        pt_count = f.variables["point_count"].data.copy()
        mass_vals = f.variables["mass_values"].data.copy()
        int_vals = f.variables["intensity_values"].data.copy()
        tic_raw = f.variables["total_intensity"].data.copy()

    n_scans = len(rt)
    spectra = np.zeros((n_scans, MZ_MAX), dtype=np.float32)
    for i in range(n_scans):
        s = int(scan_idx[i])
        e = s + int(pt_count[i])
        mz = np.round(mass_vals[s:e]).astype(np.int32)
        intensity = int_vals[s:e].astype(np.float32)
        valid = (mz >= 0) & (mz < MZ_MAX)
        np.add.at(spectra[i], mz[valid], intensity[valid])

    return spectra, tic_raw.astype(np.float32)


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


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def main(raw_dir: Path, output_dir: Path, n_bins: int, workers: int, no_download: bool) -> None:
    print("Fetching metadata from MetaboLights …")
    assay_rows = _fetch_tsv(ASSAY_URL)
    sample_rows = _fetch_tsv(SAMPLE_URL)
    cdf_meta = _build_cdf_to_meta(assay_rows, sample_rows)
    print(f"  Metadata for {len(cdf_meta)} samples loaded")

    if not no_download:
        print("\nFetching file list from EBI FTP …")
        fnames = sorted(_get_filenames())
        print(f"  Found {len(fnames)} CDF files")
        print(f"Downloading ({workers} parallel workers) …")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download, fn, raw_dir / fn): fn for fn in fnames}
            done = 0
            for fut in as_completed(futures):
                fn = futures[fut]
                done += 1
                if fut.exception():
                    print(f"  ERROR {fn}: {fut.exception()}")
                elif done % 20 == 0 or done == len(fnames):
                    print(f"  Downloaded {done}/{len(fnames)}")

    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    cdf_files = sorted(raw_dir.glob("*.cdf"))
    print(f"\n  Found {len(cdf_files)} CDF files in {raw_dir}")

    X_list, y_list, groups_list, ids_list = [], [], [], []
    skipped = 0

    for cdf_path in cdf_files:
        meta = cdf_meta.get(cdf_path.stem)
        if meta is None:
            print(f"  SKIP (no metadata): {cdf_path.name}")
            skipped += 1
            continue

        out_npz = chroma_dir / (cdf_path.stem + ".npz")
        if out_npz.exists():
            chroma = np.load(out_npz)["chroma"].astype(np.float32)
            sum_spec = chroma.sum(axis=0)
        else:
            spectra, tic = _read_cdf(cdf_path)
            chroma = _bin_chromatogram(spectra, tic, n_bins)
            chroma = _sqrt_l2(chroma)
            np.savez_compressed(out_npz, chroma=chroma)
            sum_spec = spectra.sum(axis=0)

        sum_spec_norm = _sqrt_l2(sum_spec[None])[0]
        X_list.append(sum_spec_norm)
        y_list.append(meta["label"])
        groups_list.append(meta["group"])
        ids_list.append(cdf_path.stem)

        if len(X_list) % 20 == 0:
            print(f"  {len(X_list)}/{len(cdf_files) - skipped} processed …")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    (output_dir / "groups.txt").write_text("\n".join(groups_list))
    (output_dir / "sample_ids.txt").write_text("\n".join(ids_list))

    inv = {v: k for k, v in CULTIVAR_LABEL.items()}
    classes, counts = np.unique(y, return_counts=True)
    print(f"\nSaved {len(X)} samples → {output_dir}")
    print(f"X shape: {X.shape}  |  dtype: {X.dtype}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({inv[c]}): {n} samples")
    print(f"Biological groups: {sorted(set(groups_list))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir",     default="data/mtbls288/raw",  type=Path)
    ap.add_argument("--output-dir",  default="data/mtbls288",      type=Path)
    ap.add_argument("--n-bins",      default=N_BINS_DEFAULT,       type=int)
    ap.add_argument("--workers",     default=8,                    type=int)
    ap.add_argument("--no-download", action="store_true",
                    help="Skip downloading; process already-present CDF files")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_bins, args.workers, args.no_download)
