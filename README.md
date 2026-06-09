# cow-dcnn

![](meme.png)

Does explicit retention time alignment improve downstream GC-MS classification
when a pretrained chromatogram encoder is available?

COW (Correlation Optimised Warping) guided by EI-MS compound identities rather than
raw TIC signal. Each chromatogram peak carries a 1000-dim m/z fingerprint; peaks are
matched across runs using cosine similarity (optionally lifted into a learned embedding
space) and a piecewise-linear warp is fit through the matched anchor pairs.

**Short answer:** no. A CNN pretrained via next-frame prediction on raw chromatograms
achieves 0.964 balanced accuracy on fish species classification without any alignment,
and every alignment method we tested degrades it.

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

The downstream classifier uses **ChromatogramCNN** from the [chroma-dcnn](https://github.com/woodRock/chroma-dcnn) package, pretrained via next-frame prediction on raw chromatograms. This pretraining objective — predict the m/z spectrum at bin *t+1* given all preceding bins — teaches the model the temporal covariance structure of GC-MS elution without any alignment supervision.

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

### Alignment quality (TIC correlation)

| Method | Within-study (fish\_oil) | Cross-study (fish\_oil × mtbls288) |
|---|---|---|
| Unaligned | 0.564 | 0.161 |
| Raw cosine | 0.662 | 0.162 |
| SimCLR encoder | 0.668 | 0.162 |
| Drift encoder | **0.675** | 0.162 |

The drift encoder (trained on 8,081 real cross-sample matched peak pairs) achieves
the best within-study alignment. Cross-study alignment does not improve meaningfully
(0.161 → 0.162): fish_oil and MTBLS288 have different compound profiles, not merely
shifted retention times.

### Downstream classification (fish species, 4-class, 5-fold × 3 seeds)

Metric: balanced accuracy (mean ± std over 15 runs). Classifiers: ChromatogramCNN
with next-frame-prediction pretraining or random initialisation; PLS-DA and RF on the
per-m/z max-projection feature vector. Alignment methods cited from the literature.

| Alignment | CNN (pretrained) | CNN (from scratch) | PLS-DA | RF |
|---|---|---|---|---|
| No alignment | **0.964 ± 0.072** | 0.679 ± 0.196 | 0.325 ± 0.101 | 0.363 ± 0.076 |
| Co-shift [Savorani 2010] | 0.947 ± 0.130 | **0.764 ± 0.119** | 0.323 ± 0.097 | 0.362 ± 0.092 |
| icoshift [Savorani 2010] | 0.911 ± 0.148 | 0.659 ± 0.126 | 0.332 ± 0.109 | 0.353 ± 0.080 |
| COW [Nielsen 1998] | 0.888 ± 0.122 | 0.710 ± 0.165 | 0.322 ± 0.099 | 0.383 ± 0.094 |
| m/z COW (cosine) | 0.853 ± 0.129 | 0.681 ± 0.220 | 0.333 ± 0.083 | 0.391 ± 0.074 |
| m/z COW (drift enc.) | 0.856 ± 0.133 | 0.717 ± 0.139 | 0.329 ± 0.086 | **0.411 ± 0.078** |

**Key findings:**

- *Pretraining dominates alignment.* The pretrained CNN without any alignment (0.964)
  outperforms every aligned condition for the from-scratch model (best: 0.764).
  The pretraining gap is an order of magnitude larger than any alignment gain.

- *Every alignment method degrades the pretrained CNN.* The decline is monotone with
  alignment complexity: 0.964 → 0.947 (co-shift) → 0.911 (icoshift) → 0.888 (COW)
  → 0.853 (m/z COW). The pretrained model was trained on raw unaligned chromatograms;
  post-hoc warping takes the input out of its training distribution.

- *For models without pretraining, only simple alignment helps.* Co-shift improves
  from-scratch accuracy (+0.085) and reduces variance (σ: 0.196 → 0.119). Icoshift
  and m/z COW (cosine) do not improve over no alignment; only co-shift and COW yield
  a net gain.

- *Classical methods are alignment-insensitive.* PLS-DA (0.322–0.333) and RF
  (0.353–0.411) show no consistent improvement across alignment conditions.

**Limitations.** Results are from a single dataset (103 fish oil samples, 4 classes).
Classical baselines use a max-projection feature rather than a conventional aligned
peak table, which may understate their performance.

## Exploration

See `notebooks/01_explore_alignment.ipynb` for the full alignment pipeline:
peak detection, m/z fingerprint extraction, Hungarian matching, piecewise-linear
warping, FAME library identification against MoNA, and encoder comparisons.
