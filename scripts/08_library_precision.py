"""
Library-matched compound precision for cross-study alignment.

Replaces the m/z cosine ≥ 0.70 proxy used in 08_cross_study.py with
confirmed compound identity from the MoNA reference library.

For each alignment method:
  For every wheat↔rice pair (3,160 total), run alignment and collect the
  proposed anchor pairs (wheat peak fingerprint, rice peak fingerprint).
  Each peak is then matched against the MoNA library by cosine similarity.
  A pair is a TRUE compound match if:
    - both peaks have a library hit above LIB_THRESHOLD cosine
    - both hits share the same first-block InChIKey (14 chars, compound identity
      without stereochemistry)

Metrics reported per method:
  Library precision  — fraction of proposed anchor pairs that are true compound matches
  Coverage           — fraction of proposed anchors where BOTH peaks have any library hit
  Compound precision — same as library precision but conditioned on both having a hit
  (compare against the cosine proxy precision from 08_cross_study.py)
"""

from __future__ import annotations

import sys
import h5py
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from alignment import (align_pair, load_chromas, load_encoder,
                       PRECISION_CUTOFF, N_BINS, TIME_AXIS, RUN_MIN)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

LIB_THRESHOLD = 0.60   # min cosine to call a library match confident


class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()


def load_mona() -> tuple[np.ndarray, list[str]]:
    """Return (spectra [N,1000], inchikeys [N]) from pretraining HDF5."""
    h5_path = DATA_DIR / 'pretraining' / 'spectra.h5'
    with h5py.File(h5_path) as f:
        spectra   = f['spectra'][:]
        inchikeys = [k.decode() if isinstance(k, bytes) else k
                     for k in f['inchikeys'][:]]
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    spectra = (spectra / norms).astype(np.float32)
    return spectra, inchikeys


def top1_inchikey(fp: np.ndarray, lib_spectra: np.ndarray,
                  inchikeys: list[str]) -> tuple[str | None, float]:
    """Return (inchikey_prefix14, cosine) for the top library hit."""
    sims = lib_spectra @ fp.astype(np.float32)
    idx  = int(np.argmax(sims))
    cos  = float(sims[idx])
    if cos < LIB_THRESHOLD:
        return None, cos
    return inchikeys[idx][:14], cos


def evaluate_library_precision(m21: list[np.ndarray],
                                m288: list[np.ndarray],
                                lib_spectra: np.ndarray,
                                inchikeys: list[str],
                                encode_fn=None) -> dict:
    n_total   = len(m21) * len(m288)
    done      = 0

    true_matches   = 0   # both have hit, same InChIKey prefix
    covered_pairs  = 0   # both have a library hit
    total_proposed = 0   # all proposed anchor pairs

    for qc in m21:
        for rc in m288:
            _, _, _, proposed, _ = align_pair(qc, rc, encode_fn=encode_fn,
                                              return_anchors=True)
            for q_bin, r_bin, raw_cos in proposed:
                q_fp = (qc[max(0, q_bin-1):q_bin+2].mean(axis=0))
                r_fp = (rc[max(0, r_bin-1):r_bin+2].mean(axis=0))
                n = np.linalg.norm(q_fp)
                if n > 1e-8: q_fp /= n
                n = np.linalg.norm(r_fp)
                if n > 1e-8: r_fp /= n

                q_key, _ = top1_inchikey(q_fp, lib_spectra, inchikeys)
                r_key, _ = top1_inchikey(r_fp, lib_spectra, inchikeys)

                total_proposed += 1
                if q_key is not None and r_key is not None:
                    covered_pairs += 1
                    if q_key == r_key:
                        true_matches += 1

            done += 1
            if done % 500 == 0:
                print(f"    {done}/{n_total} pairs …")

    lib_precision  = true_matches  / total_proposed if total_proposed else 0.0
    coverage       = covered_pairs / total_proposed if total_proposed else 0.0
    cond_precision = true_matches  / covered_pairs  if covered_pairs  else 0.0

    return {
        'total_proposed': total_proposed,
        'lib_precision':  lib_precision,
        'coverage':       coverage,
        'cond_precision': cond_precision,
    }


def main() -> None:
    print("Loading chromatograms …")
    m21  = load_chromas(DATA_DIR / 'mtbls21'  / 'chroma')
    m288 = load_chromas(DATA_DIR / 'mtbls288' / 'chroma')
    print(f"  mtbls21  (wheat): {len(m21)} samples")
    print(f"  mtbls288 (rice) : {len(m288)} samples")

    print("Loading MoNA library …")
    lib_spectra, inchikeys = load_mona()
    print(f"  {len(inchikeys)} reference spectra  "
          f"(cosine threshold: {LIB_THRESHOLD})\n")

    methods: list[tuple[str, object]] = [('Raw cosine', None)]
    for label, fname in [('Drift encoder',       'drift_simclr.pt'),
                         ('Cross-study encoder', 'cross_study_simclr.pt')]:
        p = CKPT_DIR / fname
        if p.exists():
            methods.append((label, load_encoder(p)))
            print(f"Loaded  {label}  from {fname}")
        else:
            print(f"SKIP    {label}: {fname} not found")
    print()

    results = {}
    for label, enc_fn in methods:
        print(f"Evaluating: {label} …")
        r = evaluate_library_precision(m21, m288, lib_spectra, inchikeys, enc_fn)
        results[label] = r
        print(f"  library precision={r['lib_precision']:.3f}  "
              f"coverage={r['coverage']:.3f}  "
              f"compound precision={r['cond_precision']:.3f}  "
              f"(n proposed={r['total_proposed']})\n")

    w = 26
    print("=" * 75)
    print(f"{'Method':<{w}}  {'Lib precision':>14}  {'Coverage':>9}  "
          f"{'Cmpd precision':>15}")
    print("-" * 75)
    for lbl, r in results.items():
        print(f"{lbl:<{w}}  {r['lib_precision']:>14.3f}  "
              f"{r['coverage']:>9.3f}  {r['cond_precision']:>15.3f}")
    print("=" * 75)
    print("Lib precision : fraction of all proposed anchors mapping to same compound")
    print("Coverage      : fraction of proposed anchors where both peaks have a library hit")
    print("Cmpd precision: lib precision conditioned on both peaks having a hit")
    print(f"Library match: MoNA top-1 cosine ≥ {LIB_THRESHOLD}, InChIKey prefix (14 chars)")
    return results


def save_figure(results: dict, figs_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    x = np.arange(len(methods))
    bar_w = 0.25

    lib_p  = [results[m]['lib_precision']  for m in methods]
    cov    = [results[m]['coverage']       for m in methods]
    cond_p = [results[m]['cond_precision'] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - bar_w, lib_p,  bar_w, label='Library precision',          color='#4878CF')
    ax.bar(x,         cov,    bar_w, label='Coverage',                   color='#6ACC65')
    ax.bar(x + bar_w, cond_p, bar_w, label='Compound precision (cond.)', color='#D65F5F')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha='right')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1.05)
    ax.set_title('MoNA Library-Matched Compound Precision\n'
                 'wheat (MTBLS21) × rice (MTBLS288), 3,160 pairs')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    out = figs_dir / 'library_precision.pdf'
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved → {out}")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR = Path(__file__).parent.parent / 'figs'
    FIGS_DIR.mkdir(exist_ok=True)

    _results = {}

    txt_out = RESULTS_DIR / 'library_precision.txt'
    with open(txt_out, 'w') as fh:
        orig, sys.stdout = sys.stdout, _Tee(fh)
        try:
            _results = main()
        finally:
            sys.stdout = orig
    print(f"Results saved → {txt_out}")

    if _results:
        save_figure(_results, FIGS_DIR)
