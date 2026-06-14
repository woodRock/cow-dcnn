"""
Download and preprocess EI-MS spectra for SimCLR pretraining.

Sources:
  MoNA (MassBank of North America) — bulk GC-MS export (~200 MB zip)
    https://mona.fiehnlab.ucdavis.edu/downloads
  MassBank EU — plain-text record files from GitHub releases
    https://github.com/MassBank/MassBank-data/releases

Pipeline:
  1. Download MoNA GC-MS JSON export (zip → extract)
  2. Download latest MassBank-data release from GitHub
  3. Parse + filter to EI-MS records with valid InChIKey
  4. Deduplicate by InChIKey (prefer MoNA; tie-break: most peaks)
  5. Bin to 1 Da grid (0–999 m/z), sqrt-compress, L2-normalise
  6. Write data/pretraining/spectra.h5

Output: data/pretraining/spectra.h5
  /spectra    float32 [N, 1000]  (sqrt + L2 normalised)
  /inchikeys  bytes   [N]
  /sources    bytes   [N]        ("mona" or "massbank")
  /split      bytes   [N]        ("train" or "val")

Dependencies: requests, h5py  (pip install requests h5py)

Usage:
  python scripts/04_download_pretrain_data.py
  python scripts/04_download_pretrain_data.py --force    # re-download even if cached
  python scripts/04_download_pretrain_data.py --mona-only
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import requests

DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR  = DATA_DIR / 'pretraining' / 'raw'
OUT_H5   = DATA_DIR / 'pretraining' / 'spectra.h5'

MZ_MAX = 1000

MONA_PUBLIC_EXPORT_URL = (
    "https://mona.fiehnlab.ucdavis.edu/downloads/retrieve/"
    "MoNA-export-GC-MS_Spectra.json.zip"
)
MASSBANK_RELEASE_API = (
    "https://api.github.com/repos/MassBank/MassBank-data/releases/latest"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
}
_EI_RE = re.compile(r"GC-EI|GC/EI|EI-B|GC-MS|gas chromatograph", re.IGNORECASE)
_PEAK_RE = re.compile(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)")


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_stream(url: str, dest: Path, desc: str = "") -> None:
    resp = requests.get(url, stream=True, timeout=300, headers=_HEADERS)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {desc}: {pct:.0f}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)", end="")
    print()


# ── MoNA ─────────────────────────────────────────────────────────────────────

def download_mona(raw_dir: Path, force: bool = False) -> Path | None:
    json_path = raw_dir / "mona_gcms.json"
    if json_path.exists() and not force:
        print(f"MoNA JSON already present: {json_path}")
        return json_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "MoNA-export-GC-MS_Spectra.json.zip"

    try:
        print("Downloading MoNA GC-MS export …")
        _download_stream(MONA_PUBLIC_EXPORT_URL, zip_path, "MoNA")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            if not names:
                raise RuntimeError("No JSON file inside zip")
            with zf.open(names[0]) as src, open(json_path, "wb") as dst:
                dst.write(src.read())
        zip_path.unlink()
        print(f"Extracted MoNA JSON → {json_path}")
        return json_path
    except Exception as e:
        print(f"MoNA download failed: {e}")
        print(
            "  Manual fallback: visit https://mona.fiehnlab.ucdavis.edu/downloads\n"
            f"  and extract the GC-MS JSON to:\n  {json_path}"
        )
        return None


def parse_mona(json_path: Path) -> Iterator[dict]:
    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)
    print(f"Loaded {len(records)} raw MoNA records")
    kept = 0
    for rec in records:
        meta = {m["name"].lower(): m["value"] for m in rec.get("metaData", [])}
        instrument = meta.get("instrument type", "") or meta.get("instrument", "")
        if not _EI_RE.search(instrument):
            continue

        compound = rec.get("compound", [{}])
        if isinstance(compound, list):
            compound = compound[0] if compound else {}
        inchikey = compound.get("inchiKey", "")
        if not inchikey or len(inchikey) < 14:
            continue

        smiles = ""
        for m in compound.get("metaData", []):
            if m.get("name", "").lower() == "smiles":
                smiles = m.get("value", "")
                break

        peaks = [(float(a), float(b))
                 for a, b in _PEAK_RE.findall(rec.get("spectrum", ""))
                 if float(a) <= 1000]
        if not peaks:
            continue

        kept += 1
        yield {"inchikey": inchikey, "smiles": smiles, "peaks": peaks, "source": "mona"}

    print(f"Retained {kept} EI-GC-MS records from MoNA")


# ── MassBank EU ───────────────────────────────────────────────────────────────

def download_massbank(raw_dir: Path, force: bool = False) -> Path:
    records_dir = raw_dir / "massbank_records"
    if records_dir.exists() and any(records_dir.rglob("*.txt")) and not force:
        n = sum(1 for _ in records_dir.rglob("*.txt"))
        print(f"MassBank records already present: {n} files")
        return records_dir

    print("Fetching MassBank-data latest release …")
    resp = requests.get(MASSBANK_RELEASE_API, timeout=30)
    resp.raise_for_status()
    release = resp.json()
    tag = release["tag_name"]
    zip_url = release["zipball_url"]
    print(f"  Latest release: {tag}")

    zip_path = raw_dir / f"massbank_{tag}.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _download_stream(zip_url, zip_path, "MassBank")

    print("Extracting MassBank records …")
    records_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
        for member in txt_members:
            data = zf.read(member)
            dest = records_dir / Path(member).name
            dest.write_bytes(data)

    zip_path.unlink()
    n = sum(1 for _ in records_dir.rglob("*.txt"))
    print(f"Extracted {n} MassBank record files → {records_dir}")
    return records_dir


def parse_massbank(records_dir: Path) -> Iterator[dict]:
    txt_files = list(records_dir.rglob("*.txt"))
    print(f"Parsing {len(txt_files)} MassBank record files …")
    kept = 0
    for i, txt_path in enumerate(txt_files):
        if i % 5000 == 0:
            print(f"\r  {i}/{len(txt_files)} …", end="")
        rec = _parse_mb_txt(txt_path)
        if rec is not None:
            kept += 1
            yield rec
    print(f"\nRetained {kept} EI records from MassBank")


def _parse_mb_txt(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if "EI" not in text.upper():
        return None

    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    for line in text.splitlines():
        line = line.strip()
        if line == "PK$PEAK: m/z int. rel.int.":
            in_peaks = True
            continue
        if in_peaks:
            if line.startswith("//"):
                in_peaks = False
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                    if mz <= 1000:
                        peaks.append((mz, intensity))
                except ValueError:
                    pass
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    inchikey = fields.get("CH$INCHI_KEY", "") or fields.get("CH$INCHIKEY", "")
    if not inchikey or len(inchikey) < 14 or not peaks:
        return None

    return {
        "inchikey": inchikey,
        "smiles": fields.get("CH$SMILES", ""),
        "peaks": peaks,
        "source": "massbank",
    }


# ── Deduplication + HDF5 ─────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    from collections import defaultdict
    by_key: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_key[rec["inchikey"]].append(rec)
    deduped = []
    for group in by_key.values():
        group.sort(key=lambda r: (r["source"] != "mona", -len(r["peaks"])))
        deduped.append(group[0])
    print(f"Deduplication: {len(records)} → {len(deduped)} unique InChIKeys")
    return deduped


def bin_and_normalise(peaks: list[tuple[float, float]]) -> np.ndarray:
    vec = np.zeros(MZ_MAX, dtype=np.float32)
    for mz, intensity in peaks:
        idx = int(round(mz))
        if 0 <= idx < MZ_MAX and intensity > 0:
            vec[idx] = max(vec[idx], float(intensity))
    vec = np.sqrt(np.maximum(vec, 0.0))
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def write_hdf5(records: list[dict], out_path: Path, val_fraction: float = 0.1,
               seed: int = 42) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(records)
    rng = np.random.default_rng(seed)

    spectra = np.zeros((n, MZ_MAX), dtype=np.float32)
    inchikeys, smiles_list, sources = [], [], []

    print(f"Preprocessing {n} spectra …")
    for i, rec in enumerate(records):
        spectra[i] = bin_and_normalise(rec["peaks"])
        inchikeys.append(rec["inchikey"].encode())
        smiles_list.append(rec.get("smiles", "").encode())
        sources.append(rec["source"].encode())
        if (i + 1) % 2000 == 0:
            print(f"\r  {i+1}/{n} …", end="")
    print()

    source_arr = np.array([s.decode() for s in sources])
    split = np.array(["train"] * n, dtype=object)
    for src in np.unique(source_arr):
        mask = np.where(source_arr == src)[0]
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_fraction))
        split[mask[:n_val]] = "val"

    n_train = int((split == "train").sum())
    n_val = int((split == "val").sum())
    print(f"Split: {n_train} train / {n_val} val")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("spectra",   data=spectra,                 compression="gzip", compression_opts=4)
        f.create_dataset("inchikeys", data=np.array(inchikeys))
        f.create_dataset("smiles",    data=np.array(smiles_list))
        f.create_dataset("sources",   data=np.array(sources))
        f.create_dataset("split",     data=np.array([s.encode() for s in split]))
        f.attrs["mz_max"]  = MZ_MAX
        f.attrs["n_total"] = n
        f.attrs["n_train"] = n_train
        f.attrs["n_val"]   = n_val

    print(f"Saved → {out_path}  ({n} spectra, {MZ_MAX} m/z bins)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(force: bool, mona_only: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    print("=== Step 1: MoNA ===")
    mona_json = download_mona(RAW_DIR, force=force)
    if mona_json:
        all_records.extend(parse_mona(mona_json))
    else:
        print("  Skipping MoNA (download failed).")

    if not mona_only:
        print("\n=== Step 2: MassBank EU ===")
        mb_dir = download_massbank(RAW_DIR, force=force)
        all_records.extend(parse_massbank(mb_dir))

    print(f"\n=== Step 3: Deduplication ({len(all_records)} total records) ===")
    unique = deduplicate(all_records)

    print("\n=== Step 4: Writing HDF5 ===")
    write_hdf5(unique, OUT_H5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force",     action="store_true",
                    help="Re-download even if cached files exist")
    ap.add_argument("--mona-only", action="store_true",
                    help="Skip MassBank EU download")
    args = ap.parse_args()
    main(args.force, args.mona_only)
