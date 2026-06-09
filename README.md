# cow-dcnn

GC-MS chromatogram alignment via m/z fingerprint matching and deep metric learning.

COW (Correlation Optimised Warping) guided by EI-MS compound identities rather than
raw TIC signal — each peak carries a 1000-dim m/z fingerprint; peaks are matched
across runs using cosine similarity (optionally lifted into a learned embedding space)
and a piecewise-linear warp is fit through the matched anchor pairs.

## Data setup

Raw data is excluded from version control (`data/**` in `.gitignore`).
Run the scripts below in order to reproduce the datasets from scratch.

### Option A — copy from an existing chroma-dcnn checkout (fastest)

If you already have a local `chroma-dcnn` repo with processed data:

```bash
python scripts/01_download_data.py --from-chroma-dcnn /path/to/chroma-dcnn
```

This copies all processed `.npz` chromatogram files directly, skipping raw
download and CDF parsing.

### Option B — download + preprocess from public sources

**MTBLS288** (rice grain GC-MS, MetaboLights, ~5 GB raw):

```bash
python scripts/02_preprocess_mtbls288.py          # download + preprocess
python scripts/02_preprocess_mtbls288.py --workers 8  # parallel download
```

**fish oil** (NZ fish species GC-MS; not publicly available):

Place the raw CSV files in `data/fish_oil/raw/`, then:

```bash
python scripts/03_preprocess_fish_oil.py
```

**Pretraining spectra** (MoNA + MassBank EU EI-MS, ~200 MB download):

```bash
python scripts/04_download_pretrain_data.py          # MoNA + MassBank
python scripts/04_download_pretrain_data.py --mona-only  # MoNA only
```

### Expected layout after preprocessing

```
data/
  fish_oil/
    chroma/       {sample_id}.npz  [200, 1000] float32
  mtbls288/
    chroma/       {stem}.npz       [200, 1000] float32
    X.npy         [80, 1000]  sum spectra
    y.npy         [80]        cultivar labels (0–3)
    groups.txt    biological replicate IDs
    sample_ids.txt
  pretraining/
    spectra.h5    [N, 1000]  sqrt+L2 normalised EI-MS spectra
```

## Pretraining the encoder

```bash
# Stage 1: MoNA spectra (general EI-MS features)
# Stage 2: fine-tune on chromatogram peaks (fish_oil + mtbls288)
python scripts/pretrain_simclr.py

# Drift-augmented variant (RT-invariant embeddings)
python scripts/pretrain_drift.py
```

Checkpoints are saved to `checkpoints/` (also excluded from git).

## Exploration

See `notebooks/01_explore_alignment.ipynb` for the full alignment pipeline:
peak detection, m/z fingerprint extraction, Hungarian matching, piecewise-linear
warping, FAME library identification, and encoder comparisons.

## Results

| Method | Within-study (fish_oil) | Cross-study (fish_oil × mtbls288) |
|---|---|---|
| Unaligned | 0.564 | 0.161 |
| Raw cosine | 0.662 | 0.162 |
| SimCLR encoder | 0.668 | 0.162 |
| Drift encoder | 0.599 | 0.162 |

Cross-study alignment remains unsolved: fish_oil and MTBLS288 have genuinely
different compound profiles (different matrices, different labs), not just shifted
retention times. Bridging this gap requires cross-study supervision — labelled
peak pairs or shared internal standards.
