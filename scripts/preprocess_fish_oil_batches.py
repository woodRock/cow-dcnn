"""
Preprocess raw fish oil GC-MS .txt files into batch-grouped chromatograms
for cross-study retention-shift analysis in cow-dcnn.

Reads the raw-subsetted-CVSs directory, which contains one .txt file per GC-MS
run. Acquisition date is encoded in the comment1 header field as YYMMDD.
Samples are grouped into four acquisition batches:

  batch_sep2015  —  2015-09-16/17  (GC-MS run 1)
  batch_jan2016  —  2016-01-12     (GC-MS run 2, ~4 months later)
  batch_apr2016  —  2016-04-29     (GC-MS run 3, ~7 months after run 1)
  batch_jul2016  —  2016-07-11     (GC-MS run 4, ~10 months after run 1)

Instrument blanks (comment1 contains 'BLANK') are skipped automatically.

Each batch directory written to data/fish_oil/ follows the cow-dcnn convention:
  chroma/        per-sample [N_BINS, MZ_MAX] float32 chromatogram .npz files
  X.npy          [N, MZ_MAX] sqrt+L2-normalised sum spectra (for baselines)
  y.npy          [N] int64 tissue-type labels
  groups.txt     fish individual per sample (BCO4, BCO5, SNA1, …)
  sample_ids.txt filename stem per sample

Tissue label encoding (written to data/fish_oil/tissue_labels.json):
  0 = Frame
  1 = Gonad        (Gonad / Gonad Female / Gonad Male)
  2 = Head
  3 = Liver        (Liver / Liver Female / Liver Male)
  4 = Rest of Guts (Rest of Guts / ROG)
  5 = Skin

Usage:
  python scripts/preprocess_fish_oil_batches.py
  python scripts/preprocess_fish_oil_batches.py --raw-dir /path/to/raw-subsetted-CVSs
  python scripts/preprocess_fish_oil_batches.py --n-bins 200 --mz-max 1000
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

MZ_MAX   = 1000   # m/z bins 0–999, 1 Da resolution
N_BINS   = 200    # RT bins (matches cow-dcnn convention)

# Map 6-digit date strings to batch names
DATE_TO_BATCH: dict[str, str] = {
    "150807": "batch_aug2015",   # only 2 head blanks — kept but tiny
    "150916": "batch_sep2015",
    "150917": "batch_sep2015",
    "160112": "batch_jan2016",
    "160429": "batch_apr2016",
    "160711": "batch_jul2016",
}

TISSUE_LABELS: dict[str, int] = {
    "Frame":        0,
    "Gonad":        1,
    "Head":         2,
    "Liver":        3,
    "Rest of Guts": 4,
    "Skin":         5,
}

# Ordered longest-first so multi-word patterns match before sub-strings
_TISSUE_PATTERNS: list[tuple[str, str]] = [
    ("Rest of Guts", "Rest of Guts"),
    ("Gonad Female",  "Gonad"),
    ("Gonad Male",    "Gonad"),
    ("Liver Female",  "Liver"),
    ("Liver Male",    "Liver"),
    ("Gonad",         "Gonad"),
    ("Liver",         "Liver"),
    ("Frame",         "Frame"),
    ("Head",          "Head"),
    ("Skin",          "Skin"),
    ("ROG",           "Rest of Guts"),
]

# ── Filename parsing ──────────────────────────────────────────────────────────

# e.g. "4 - BCO4 Frame A" or "5 - BCO5 Gonad Female C"
_FNAME_RE = re.compile(
    r"^\d+\s+-\s+"                      # leading "N - "
    r"(?P<species>[A-Z]+\d+)\s+"        # species code (BCO4, SNA1, …)
    r"(?P<rest>.+?)$",                  # everything after species
    re.IGNORECASE,
)


def _parse_filename(stem: str) -> tuple[str, str] | None:
    """Return (species, tissue) from a filename stem, or None if unparseable."""
    m = _FNAME_RE.match(stem.strip())
    if not m:
        return None
    species = m.group("species").upper()
    rest    = m.group("rest").strip()

    for pattern, tissue in _TISSUE_PATTERNS:
        if pattern.lower() in rest.lower():
            return species, tissue

    return None


# ── .txt file parser ──────────────────────────────────────────────────────────

_ACQ_DATE_RE   = re.compile(r"comment1\s*=\s*(\d{6})")
_BLANK_RE      = re.compile(r"comment1\s*=\s*\d{6}\s+BLANK", re.IGNORECASE)
_START_TIME_RE = re.compile(r"start_time\s*=\s*([\d.]+)")
_INTEG_RE      = re.compile(r"integ_intens\s*=\s*([\d.]+)")
_PACKET_RE     = re.compile(
    r"intensity\s*=\s*([\d.]+).*?mass/position\s*=\s*([\d.]+)"
)


def parse_txt(path: Path, mz_max: int = MZ_MAX) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Parse one raw GC-MS .txt file.

    Returns (spectra, tic, acq_date) where:
      spectra  : float32 [n_scans, mz_max]  — per-scan mass spectra
      tic      : float32 [n_scans]           — total ion current per scan
      acq_date : str, 6-digit YYMMDD

    Returns None for instrument blanks or files with no scan data.
    """
    acq_date: str | None = None
    is_blank  = False

    scans: list[tuple[float, float, dict[int, float]]] = []
    cur_rt    = 0.0
    cur_tic   = 0.0
    cur_peaks: dict[int, float] = {}
    in_scan   = False
    in_peaks  = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # Header metadata (before first ScanHeader)
            if not in_scan:
                if acq_date is None:
                    m = _ACQ_DATE_RE.match(line)
                    if m:
                        acq_date = m.group(1)
                if _BLANK_RE.match(line):
                    is_blank = True
                    break
                if line.startswith("ScanHeader #"):
                    in_scan = True
                    in_peaks = False
                continue

            # --- inside scan region ---
            if line.startswith("ScanHeader #"):
                scans.append((cur_rt, cur_tic, cur_peaks))
                cur_rt    = 0.0
                cur_tic   = 0.0
                cur_peaks = {}
                in_peaks  = False
                continue

            if line.startswith("start_time =") and "end_time" in line:
                m = _START_TIME_RE.search(line)
                if m:
                    cur_rt = float(m.group(1))
                continue

            if line.startswith("num_readings") and "integ_intens" in line:
                m = _INTEG_RE.search(line)
                if m:
                    cur_tic = float(m.group(1))
                continue

            if line == "DataPeaks":
                in_peaks = True
                continue

            if in_peaks and line.startswith("Packet #"):
                m = _PACKET_RE.search(line)
                if m:
                    intensity = float(m.group(1))
                    mz        = int(round(float(m.group(2))))
                    if 0 <= mz < mz_max and intensity > 0:
                        if cur_peaks.get(mz, 0.0) < intensity:
                            cur_peaks[mz] = intensity

    if is_blank or not scans or acq_date is None:
        return None

    # Append the last open scan
    scans.append((cur_rt, cur_tic, cur_peaks))

    n = len(scans)
    spectra = np.zeros((n, mz_max), dtype=np.float32)
    tic     = np.zeros(n, dtype=np.float32)

    for i, (rt, t, peaks) in enumerate(scans):
        tic[i] = t
        for mz, intensity in peaks.items():
            spectra[i, mz] = intensity

    return spectra, tic, acq_date


# ── Chromatogram processing ───────────────────────────────────────────────────

def bin_chromatogram(spectra: np.ndarray, tic: np.ndarray, n_bins: int) -> np.ndarray:
    """Divide RT axis into n_bins equal windows; keep the highest-TIC scan per window."""
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


def sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr   = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return (arr / (norms + eps)).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(raw_dir: Path, output_dir: Path, n_bins: int, mz_max: int) -> None:
    txt_files = sorted(raw_dir.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files under {raw_dir}")

    # --- Pass 1: parse every file and group by batch ---
    batches: dict[str, list[dict]] = {}

    for path in tqdm(txt_files, desc="Parsing"):
        if ".ipynb_checkpoints" in str(path):
            continue

        meta = _parse_filename(path.stem)
        if meta is None:
            print(f"  SKIP (unparseable filename): {path.name}")
            continue
        species, tissue = meta

        result = parse_txt(path, mz_max)
        if result is None:
            continue   # blank or empty
        spectra, tic, acq_date = result

        batch = DATE_TO_BATCH.get(acq_date)
        if batch is None:
            print(f"  SKIP (unknown date {acq_date}): {path.name}")
            continue

        batches.setdefault(batch, []).append({
            "path":    path,
            "species": species,
            "tissue":  tissue,
            "spectra": spectra,
            "tic":     tic,
            "stem":    path.stem,
        })

    print(f"\nBatch summary:")
    for b in sorted(batches):
        print(f"  {b}: {len(batches[b])} samples")

    # --- Pass 2: process and write each batch ---
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_name in sorted(batches):
        records    = batches[batch_name]
        batch_dir  = output_dir / batch_name
        chroma_dir = batch_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        X_list, y_list, groups_list, ids_list = [], [], [], []

        for rec in records:
            tissue_label = TISSUE_LABELS.get(rec["tissue"])
            if tissue_label is None:
                print(f"  SKIP (unknown tissue '{rec['tissue']}'): {rec['stem']}")
                continue

            chroma    = bin_chromatogram(rec["spectra"], rec["tic"], n_bins)
            chroma    = sqrt_l2(chroma)
            sum_spec  = sqrt_l2(rec["spectra"].sum(axis=0, keepdims=True))[0]

            safe_stem = re.sub(r"[^\w]", "_", rec["stem"])
            out_npz   = chroma_dir / f"{safe_stem}.npz"
            np.savez_compressed(out_npz, chroma=chroma)

            X_list.append(sum_spec)
            y_list.append(tissue_label)
            groups_list.append(rec["species"])
            ids_list.append(rec["stem"])

        if not X_list:
            print(f"  {batch_name}: no valid samples, skipping")
            continue

        np.save(batch_dir / "X.npy", np.stack(X_list).astype(np.float32))
        np.save(batch_dir / "y.npy", np.array(y_list, dtype=np.int64))
        (batch_dir / "groups.txt").write_text("\n".join(groups_list))
        (batch_dir / "sample_ids.txt").write_text("\n".join(ids_list))

        classes, counts = np.unique(y_list, return_counts=True)
        inv = {v: k for k, v in TISSUE_LABELS.items()}
        print(f"\n{batch_name}  ({len(X_list)} samples)")
        for c, n in zip(classes, counts):
            print(f"    {inv[c]:<16}: {n}")

    # Write shared metadata
    (output_dir / "tissue_labels.json").write_text(
        json.dumps(TISSUE_LABELS, indent=2)
    )

    manifest: dict[str, dict] = {}
    for batch_name in sorted(batches):
        batch_dir = output_dir / batch_name
        npz_files = sorted((batch_dir / "chroma").glob("*.npz"))
        if npz_files:
            y = np.load(batch_dir / "y.npy")
            groups = (batch_dir / "groups.txt").read_text().splitlines()
            inv = {v: k for k, v in TISSUE_LABELS.items()}
            manifest[batch_name] = {
                "n_samples":  len(npz_files),
                "tissues":    {inv[int(c)]: int(n)
                               for c, n in zip(*np.unique(y, return_counts=True))},
                "species":    sorted(set(groups)),
            }

    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"\nAll batches written to {output_dir}")
    print(f"Tissue label mapping → {output_dir / 'tissue_labels.json'}")
    print(f"Batch manifest       → {output_dir / 'batch_manifest.json'}")

    existing_flat = [
        output_dir / "chroma",
        output_dir / "X.npy",
        output_dir / "y.npy",
        output_dir / "groups.txt",
        output_dir / "sample_ids.txt",
    ]
    stale = [p for p in existing_flat if p.exists()]
    if stale:
        print(f"\nNote: old flat fish_oil dataset still present:")
        for p in stale:
            print(f"  {p}")
        print("Run with --clean to remove them.")


def clean_old_flat(output_dir: Path) -> None:
    targets = [
        output_dir / "chroma",
        output_dir / "X.npy",
        output_dir / "y.npy",
        output_dir / "groups.txt",
        output_dir / "sample_ids.txt",
    ]
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            print(f"  Removed directory: {t}")
        elif t.exists():
            t.unlink()
            print(f"  Removed file:      {t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[1].parent
                    / "fish-data-analysis" / "src" / "data" / "raw-subsetted-CVSs"),
        type=Path,
        help="Path to raw-subsetted-CVSs directory",
    )
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "fish_oil"),
        type=Path,
        help="Output directory (default: data/fish_oil/)",
    )
    ap.add_argument("--n-bins",  default=N_BINS,   type=int,
                    help="Number of RT bins (default 200)")
    ap.add_argument("--mz-max",  default=MZ_MAX,   type=int,
                    help="m/z bins (default 1000)")
    ap.add_argument("--clean",   action="store_true",
                    help="Remove old flat fish_oil dataset before writing")
    args = ap.parse_args()

    if args.clean:
        print(f"Removing old flat dataset from {args.output_dir} …")
        clean_old_flat(args.output_dir)

    main(args.raw_dir, args.output_dir, args.n_bins, args.mz_max)
