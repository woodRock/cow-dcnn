"""
Ablation: TIC-only vs full 2D chromatogram for GC-MS classification.

Collapses the 2D chromatogram [N_bins, mz_max] to the Total Ion Chromatogram
(TIC) [N_bins, 1] by summing across the m/z axis. This removes all compound-
identity information from the m/z fingerprints and retains only the total ion
count at each retention time bin.

The ablation answers: does the m/z dimension add value over the raw TIC alone?

Classifiers
───────────
  ChromatogramCNN  from_scratch  — mz_max=1 (spec_proj is Linear(1→128))
  PLS-DA                         — on the 200-dim TIC vector
  Random Forest                  — on the 200-dim TIC vector

Note: chroma_pretrain is not applicable here. The pretrained spec_proj is
Linear(1000→128); collapsing to TIC requires Linear(1→128), which is a
different weight tensor and cannot be initialised from the checkpoint.

Compare output with classify_fish_oil.py (No alignment condition) to assess
the contribution of the m/z dimension.

Evaluation: 5-fold stratified CV × 3 seeds = 15 runs per condition.
Metric: balanced accuracy, macro-F1.
"""

from __future__ import annotations

import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, '/Users/woodj/Desktop/chroma-dcnn/src')

from chroma_dcnn.training.finetune_chroma import ChromaFinetuner
from chroma_dcnn.evaluation.baselines import baseline_cv

DATA_DIR    = Path(__file__).parent.parent / 'data'
FISH_CHROMA = DATA_DIR / 'fish_oil' / 'chroma'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

class _Tee:
    def __init__(self, fh):
        self._fh, self._orig = fh, sys.stdout
    def write(self, s):
        self._orig.write(s); self._fh.write(s)
    def flush(self):
        self._orig.flush(); self._fh.flush()

CV_CFG = {
    'model': {
        'mz_max': 1,          # TIC only: single value per RT bin
        'cnn_channels': 128,
        'kernel_size': 7,
        'dropout': 0.3,
    },
    'task': {
        'num_classes': 4,
        'cv_folds': 5,
        'cv_seeds': [0, 1, 2],
        'cv_strategy': 'kfold',
    },
    'finetuning': {
        'epochs': 100,
        'batch_size': 16,
        'lr': 3e-3,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'early_stopping_patience': 20,
    },
    # No pretrained_checkpoints: chroma_pretrain requires mz_max=1000
    'pretrained_checkpoints': {},
}


def collapse_to_tic(chroma: np.ndarray) -> np.ndarray:
    """[N_bins, mz_max] → [N_bins, 1]: sum across m/z axis."""
    return chroma.sum(axis=1, keepdims=True).astype(np.float32)


def main() -> None:
    npz_paths = sorted(FISH_CHROMA.glob('*.npz'))
    y         = np.load(DATA_DIR / 'fish_oil' / 'y.npy').astype(np.int64)
    print(f"Loaded {len(npz_paths)} samples  |  classes: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Input shape:  [200, 1000]  (2D chromatogram)")
    print(f"Ablated shape:[200, 1]     (TIC only — m/z summed)")

    tmp = Path(tempfile.mkdtemp(prefix='tic_ablation_'))
    try:
        # Write TIC-only .npz files
        tic_paths = []
        for p in npz_paths:
            chroma = np.load(p)['chroma'].astype(np.float32)
            tic    = collapse_to_tic(chroma)
            out    = tmp / p.name
            np.savez_compressed(out, chroma=tic)
            tic_paths.append(out)

        print(f"\nTIC-only files written to {tmp}")

        # CNN from_scratch on TIC
        print("\nCNN [from_scratch, TIC only] …")
        finetuner = ChromaFinetuner(CV_CFG, tic_paths, y)
        res       = finetuner.evaluate_condition('from_scratch')
        ba_cnn, f1_cnn = res['balanced_accuracy'], res['macro_f1']
        print(f"  bal_acc={np.mean(ba_cnn):.3f}±{np.std(ba_cnn):.3f}  "
              f"macro-F1={np.mean(f1_cnn):.3f}±{np.std(f1_cnn):.3f}")

        # Classical baselines on 200-dim TIC vector
        X_tic = np.stack([np.load(p)['chroma'].squeeze(1) for p in tic_paths])
        print(f"\nClassical baseline feature: {X_tic.shape}  (200-dim TIC vector)")

        results_classical = {}
        for clf in ['plsda', 'rf']:
            print(f"\n{clf.upper()} [TIC only] …")
            res = baseline_cv(X_tic, y, clf,
                              seeds=CV_CFG['task']['cv_seeds'],
                              cv_strategy='kfold',
                              cv_folds=CV_CFG['task']['cv_folds'])
            results_classical[clf] = res
            ba, f1 = res['balanced_accuracy'], res['macro_f1']
            print(f"  bal_acc={np.mean(ba):.3f}±{np.std(ba):.3f}  "
                  f"macro-F1={np.mean(f1):.3f}±{np.std(f1):.3f}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Summary — formatted to match classify_fish_oil.py output
    print('\n' + '='*86)
    print(f"{'Condition':<52}  {'Bal. Acc.':>14}  {'Macro-F1':>14}")
    print('-'*86)

    rows = [
        ('TIC only  |  CNN from_scratch',   ba_cnn,  f1_cnn),
        ('TIC only  |  PLSDA',              results_classical['plsda']['balanced_accuracy'],
                                            results_classical['plsda']['macro_f1']),
        ('TIC only  |  RF',                 results_classical['rf']['balanced_accuracy'],
                                            results_classical['rf']['macro_f1']),
    ]
    for label, ba, f1 in rows:
        print(f"{label:<52}  "
              f"{np.mean(ba):.3f} ± {np.std(ba):.3f}  "
              f"{np.mean(f1):.3f} ± {np.std(f1):.3f}")

    print('='*86)
    print(f"5-fold × {len(CV_CFG['task']['cv_seeds'])} seeds = "
          f"{5 * len(CV_CFG['task']['cv_seeds'])} runs per condition")
    print("Classes: 0=SNA (snapper)  1=GUR (gurnard)  "
          "2=TAR (tarakihi)  3=BCO (blue cod)")

    print("\n── Compare with full 2D (No alignment, from classify_fish_oil.py) ──────────")
    print("  CNN from_scratch   2D: 0.679 ± 0.196   TIC: (above)")
    print("  CNN chroma_pretrain 2D: 0.964 ± 0.072   TIC: n/a (incompatible weights)")
    print("  PLSDA              2D: 0.325 ± 0.101   TIC: (above)")
    print("  RF                 2D: 0.363 ± 0.076   TIC: (above)")


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    _out = RESULTS_DIR / 'tic_ablation.txt'
    with open(_out, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(_fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f'Results saved → {_out}')
