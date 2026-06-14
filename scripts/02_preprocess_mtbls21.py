"""
Download and preprocess MTBLS21 wheat grain GC-MS data.

Source: MetaboLights MTBLS21
  Högy et al. (2009) "Effects of elevated atmospheric CO2 on grain quality of
  wheat: I. Grain yield and quality parameters" — Journal of Cereal Science.
  Spring wheat (Triticum aestivum cv. Triso) grown in a FACE (free-air CO2
  enrichment) experiment; two growing seasons (2005, 2006); ambient vs elevated
  CO2 conditions.

  40 unique biological samples + 40 technical _run02 replicates = 80 CDF files.
  Instrument: Thermo TRACE GC / PolarisQ MS; scan range 50–550 m/z.

This is the standard GC-MS alignment benchmark dataset used by BIPACE,
CeMAPP-DTW, and related alignment papers.  Technical replicates (_run02) are
ideal for within-sample alignment evaluation.

Classification target (y): CO2 treatment
  0 = ambient  (0.409 ml/L CO2, ring indices 1, 4, 7, 10, 13)
  1 = elevated  (0.537 ml/L CO2, ring indices 2, 5, 8, 11, 14)

Groups (groups.txt): cultivation year (2005 / 2006) — useful as batch label
  for cross-year alignment evaluation.

Processing pipeline (identical to 02_preprocess_mtbls288.py):
  1. Reconstruct dense [n_scans × 1000] spectra from sparse ANDI/netCDF
  2. Bin RT axis into N_BINS (200) equal windows; keep highest-TIC scan per bin
  3. sqrt + L2 normalise each bin
  4. Save per-sample [200, 1000] float32 chromatogram as compressed .npz
  5. Build sum-spectrum X.npy [N, 1000] for baseline methods

Output (data/mtbls21/):
  raw/           raw .cdf files
  chroma/        per-sample [200, 1000] chromatogram .npz files
  X.npy          [N, 1000] sqrt+L2-normalised sum spectra
  y.npy          [N] int64 CO2 treatment labels (0=ambient, 1=elevated)
  groups.txt     cultivation year per sample (2005 / 2006)
  sample_ids.txt CDF stem per sample

Usage:
  python scripts/05_preprocess_mtbls21.py
  python scripts/05_preprocess_mtbls21.py --workers 8
  python scripts/05_preprocess_mtbls21.py --no-download   # process already-downloaded files
  python scripts/05_preprocess_mtbls21.py --no-replicates # skip _run02 technical replicates
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

MZ_MAX       = 1000
N_BINS_DEFAULT = 200

FTP_BASE    = "ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS21/FILES/"
HTTP_BASE   = "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS21/"
SAMPLE_URL  = HTTP_BASE + "s_MTBLS21.txt"
ASSAY_URL   = HTTP_BASE + "a_MTBLS21.txt"

# CO2 concentration → binary label
CO2_LABEL = {
    "0.409": 0,  # ambient
    "0.537": 1,  # elevated
}

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Metadata ──────────────────────────────────────────────────────────────────

def _fetch_tsv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _build_meta(sample_rows: list[dict], assay_rows: list[dict]) -> dict[str, dict]:
    """
    Build a dict mapping CDF stem → {label, year, co2, is_replicate}.

    Sample metadata provides CO2 and year.  Assay metadata maps sample names
    to raw file names.  _run02 files share the same biological metadata as
    their parent.
    """
    # sample name → factors
    sample_factors: dict[str, dict] = {}
    for row in sample_rows:
        name = row.get("Sample Name", "").strip()
        year_raw = row.get("Factor Value[Cultivation Year]", "").strip()
        co2_raw  = row.get("Factor Value[CO2 Concentration]", "").strip()
        if not name or not year_raw or not co2_raw:
            continue
        # CO2 value may include trailing units — keep numeric part only
        co2_num = re.match(r"[\d.]+", co2_raw)
        if co2_num is None:
            continue
        co2_str = co2_num.group(0)
        label = CO2_LABEL.get(co2_str)
        if label is None:
            continue
        year = int(float(year_raw))
        sample_factors[name] = {"label": label, "year": year, "co2": co2_str}

    # assay: sample name → raw file
    meta: dict[str, dict] = {}
    for row in assay_rows:
        name     = row.get("Sample Name", "").strip()
        raw_file = row.get("Raw Spectral Data File", "").strip()
        if not raw_file:
            continue
        stem = Path(raw_file).stem
        # Match to biological parent (strip _run02 suffix)
        parent = re.sub(r"_run0\d+$", "", stem)
        factors = sample_factors.get(name) or sample_factors.get(parent)
        if factors is None:
            continue
        meta[stem] = {
            **factors,
            "is_replicate": "_run0" in stem,
        }
    return meta


# ── Download ──────────────────────────────────────────────────────────────────

def _list_cdf_files() -> list[str]:
    """Fetch the FILES/ directory listing and return all .cdf filenames."""
    with urllib.request.urlopen(HTTP_BASE + "FILES/", timeout=30) as r:
        html = r.read().decode()
    return sorted(re.findall(r'"(080314_0[56][^"]+\.cdf)"', html))


def _download(fname: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = FTP_BASE + fname
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            return
        except Exception as e:
            if attempt == retries:
                raise
            import time; time.sleep(2 ** attempt)  # exponential back-off


# ── Preprocessing (identical pipeline to 02_preprocess_mtbls288.py) ───────────

def _read_cdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netcdf_file(str(path), "r", mmap=False) as f:
        rt        = f.variables["scan_acquisition_time"].data.copy()
        scan_idx  = f.variables["scan_index"].data.copy()
        pt_count  = f.variables["point_count"].data.copy()
        mass_vals = f.variables["mass_values"].data.copy()
        int_vals  = f.variables["intensity_values"].data.copy()
        tic_raw   = f.variables["total_intensity"].data.copy()

    n_scans = len(rt)
    spectra = np.zeros((n_scans, MZ_MAX), dtype=np.float32)
    for i in range(n_scans):
        s = int(scan_idx[i])
        e = s + int(pt_count[i])
        mz        = np.round(mass_vals[s:e]).astype(np.int32)
        intensity = int_vals[s:e].astype(np.float32)
        valid     = (mz >= 0) & (mz < MZ_MAX)
        np.add.at(spectra[i], mz[valid], intensity[valid])

    return spectra, tic_raw.astype(np.float32)


def _bin_chromatogram(spectra: np.ndarray, tic: np.ndarray, n_bins: int) -> np.ndarray:
    n_scans  = len(tic)
    bin_size = n_scans / n_bins
    result   = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        start = int(i * bin_size)
        end   = min(int((i + 1) * bin_size), n_scans)
        if start >= end:
            continue
        best       = start + int(np.argmax(tic[start:end]))
        result[i]  = spectra[best]
    return result


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr   = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(raw_dir: Path, output_dir: Path, n_bins: int,
         workers: int, no_download: bool, no_replicates: bool) -> None:

    print("Fetching metadata from MetaboLights …")
    sample_rows = _fetch_tsv(SAMPLE_URL)
    assay_rows  = _fetch_tsv(ASSAY_URL)
    file_meta   = _build_meta(sample_rows, assay_rows)
    print(f"  Metadata for {len(file_meta)} CDF files loaded")

    if not no_download:
        print("\nFetching file list from EBI FTP …")
        fnames = _list_cdf_files()
        print(f"  Found {len(fnames)} CDF files")
        if no_replicates:
            fnames = [f for f in fnames if "_run0" not in f]
            print(f"  Skipping _run02 replicates → {len(fnames)} files")
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
                elif done % 10 == 0 or done == len(fnames):
                    print(f"  Downloaded {done}/{len(fnames)}")

    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    cdf_files = sorted(raw_dir.glob("*.cdf"))
    if no_replicates:
        cdf_files = [f for f in cdf_files if "_run0" not in f.name]
    print(f"\nProcessing {len(cdf_files)} CDF files …")

    X_list, y_list, groups_list, ids_list = [], [], [], []
    skipped = 0

    for cdf_path in cdf_files:
        meta = file_meta.get(cdf_path.stem)
        if meta is None:
            print(f"  SKIP (no metadata): {cdf_path.name}")
            skipped += 1
            continue

        out_npz = chroma_dir / (cdf_path.stem + ".npz")
        try:
            if out_npz.exists():
                chroma   = np.load(out_npz)["chroma"].astype(np.float32)
                sum_spec = chroma.sum(axis=0)
            else:
                spectra, tic = _read_cdf(cdf_path)
                chroma       = _bin_chromatogram(spectra, tic, n_bins)
                chroma       = _sqrt_l2(chroma)
                np.savez_compressed(out_npz, chroma=chroma)
                sum_spec = spectra.sum(axis=0)
        except Exception as e:
            print(f"  SKIP (read error): {cdf_path.name}: {e}")
            skipped += 1
            continue

        sum_spec_norm = _sqrt_l2(sum_spec[None])[0]
        X_list.append(sum_spec_norm)
        y_list.append(meta["label"])
        groups_list.append(str(meta["year"]))
        ids_list.append(cdf_path.stem)

        if len(X_list) % 10 == 0:
            print(f"  {len(X_list)}/{len(cdf_files) - skipped} processed …")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    (output_dir / "groups.txt").write_text("\n".join(groups_list))
    (output_dir / "sample_ids.txt").write_text("\n".join(ids_list))

    classes, counts = np.unique(y, return_counts=True)
    label_names = {0: "ambient CO2 (0.409 ml/L)", 1: "elevated CO2 (0.537 ml/L)"}
    year_counts  = {}
    for g in groups_list:
        year_counts[g] = year_counts.get(g, 0) + 1

    print(f"\nSaved {len(X)} samples → {output_dir}")
    print(f"X shape: {X.shape}  |  dtype: {X.dtype}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({label_names[c]}): {n} samples")
    print(f"Year breakdown: { {k: v for k, v in sorted(year_counts.items())} }")
    print(f"Skipped: {skipped}")
    print("\nNote: groups.txt encodes cultivation year (2005/2006) — use as")
    print("      batch label for cross-year vs within-year alignment evaluation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir",        default="data/mtbls21/raw",  type=Path)
    ap.add_argument("--output-dir",     default="data/mtbls21",      type=Path)
    ap.add_argument("--n-bins",         default=N_BINS_DEFAULT,      type=int)
    ap.add_argument("--workers",        default=8,                   type=int)
    ap.add_argument("--no-download",    action="store_true",
                    help="Skip downloading; process already-present CDF files")
    ap.add_argument("--no-replicates",  action="store_true",
                    help="Exclude _run02 technical replicates")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_bins,
         args.workers, args.no_download, args.no_replicates)
