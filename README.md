# cow-dcnn

![](meme.png)

GC-MS chromatogram alignment via m/z fingerprint matching and deep metric learning.

COW (Correlation Optimised Warping) guided by EI-MS compound identities rather than
raw TIC signal. Each chromatogram peak carries a 1000-dim m/z fingerprint; peaks are
matched across runs using cosine similarity (optionally lifted into a learned embedding
space) and a piecewise-linear warp is fit through the matched anchor pairs.

## Method

### Alignment pipeline

1. **Peak detection** — find TIC peaks in each sample (top-N by height, min separation 5 bins)
2. **M/z fingerprinting** — extract the L2-normalised m/z spectrum at each peak
3. **Peak matching** — cosine similarity + time-window constraint + Hungarian one-to-one matching
4. **Warp fitting** — piecewise-linear interpolation through matched anchor pairs, with monotonicity enforcement
5. **TIC resampling** — resample the query TIC onto the reference time axis

### Encoder pretraining

Two pretraining strategies are evaluated as a drop-in replacement for raw cosine similarity in step 3:

**SimCLR** (`pretrain_simclr.py`): contrastive pretraining on EI-MS spectra with spectral augmentation (Gaussian noise, random ion masking, intensity jitter). Stage 1 trains on 9,553 MoNA reference spectra for general EI-MS features; Stage 2 fine-tunes on chromatogram peak spectra from the target datasets.

**Cross-sample drift encoder** (`pretrain_drift.py`): positive pairs are matched peak spectra from *different* GC-MS runs of the same dataset. Within fish_oil and mtbls288 separately, each sample is aligned to several reference chromatograms and the matched (ref_spectrum, query_spectrum) pairs are used as genuine same-compound observations with real inter-run spectral variation. Initialised from the MoNA SimCLR checkpoint.

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

## Training

```bash
# SimCLR: MoNA pretraining + GC-MS fine-tune
python scripts/pretrain_simclr.py

# Cross-sample drift encoder
python scripts/pretrain_drift.py
```

Scripts auto-select CUDA → MPS → CPU. Checkpoints are saved to `checkpoints/` (excluded from git).

## Evaluation

```bash
python scripts/evaluate_encoders.py
```

Evaluates all available checkpoints on within-study (fish_oil 103×103) and cross-study
(fish_oil × mtbls288 103×79) TIC correlation. Metric: mean Pearson correlation of aligned TICs.

## Results

| Method | Within-study (fish\_oil) | Cross-study (fish\_oil × mtbls288) |
|---|---|---|
| Unaligned | 0.564 | 0.161 |
| Raw cosine | 0.662 | 0.162 |
| SimCLR encoder | 0.668 | 0.162 |
| Drift encoder | **0.675** | 0.162 |

The drift encoder (trained on 8,081 real cross-sample matched peak pairs) achieves
the best within-study alignment, outperforming both raw cosine similarity and the
augmentation-only SimCLR encoder.

Cross-study alignment is an open problem: fish_oil and MTBLS288 have genuinely
different compound profiles (different biological matrices, different labs) rather
than merely shifted retention times. No method improves meaningfully beyond the
unaligned baseline (0.161 → 0.162). Bridging this gap requires cross-study
supervision — labelled peak pairs, shared internal standards, or matched reference
compounds present in both datasets.

## Exploration

See `notebooks/01_explore_alignment.ipynb` for the full alignment pipeline:
peak detection, m/z fingerprint extraction, Hungarian matching, piecewise-linear
warping, FAME library identification against MoNA, and encoder comparisons.
