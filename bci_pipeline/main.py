"""
main.py — P300 BCI Pipeline Command-Line Interface.

Subcommands
-----------
  preprocess   Load and preprocess EEG data, print tensor stats.
  train        Run cross-validation + print mean AUC (does not persist model).
  evaluate     Train a model on full data and generate evaluation figures.
  simulate     Run real-time BCI stream simulation.

Examples
--------
    python main.py preprocess
    python main.py train --model lda --feature raw
    python main.py train --model svc --feature xdawn --no-smote
    python main.py evaluate --model lda --feature raw
    python main.py simulate --model lda --feature raw --events 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import MAT_FILE, REPORTS_DIR, CV_FOLDS, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess data and print tensor statistics."""
    import numpy as np
    from src.preprocessing import run_preprocessing_pipeline

    mat_path = Path(args.mat) if args.mat else MAT_FILE

    epochs, labels, chan_names, fs = run_preprocessing_pipeline(
        mat_path,
        apply_notch=args.notch,
        verbose=True,
    )

    print("\n── Tensor Statistics ──────────────────────────────────────")
    print(f"  Shape          : {epochs.shape}  (N × C × T)")
    print(f"  dtype          : {epochs.dtype}")
    print(f"  Max amplitude  : {float(np.max(np.abs(epochs))):.2f} µV")
    print(f"  Mean amplitude : {float(np.mean(epochs)):.4f} µV")
    print(f"  Channels       : {chan_names}")
    print("──────────────────────────────────────────────────────────\n")


def cmd_train(args: argparse.Namespace) -> None:
    """Run cross-validation and report metrics."""
    from src.preprocessing import run_preprocessing_pipeline
    from src.evaluate import cross_validate_pipeline

    mat_path = Path(args.mat) if args.mat else MAT_FILE

    print(f"[train] Loading and preprocessing data from: {mat_path}")
    epochs, labels, chan_names, fs = run_preprocessing_pipeline(mat_path, verbose=False)

    results = cross_validate_pipeline(
        pipeline_name=args.model,
        extractor_name=args.feature,
        epochs=epochs,
        labels=labels,
        chan_names=chan_names,
        n_splits=CV_FOLDS,
        use_smote=not args.no_smote,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    print(f"\n[train] ✅ Mean AUC-ROC = {results['mean_auc']:.4f} "
          f"± {results['std_auc']:.4f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Train on full data and generate evaluation figures."""
    from src.preprocessing import run_preprocessing_pipeline
    from src.evaluate import cross_validate_pipeline

    mat_path    = Path(args.mat) if args.mat else MAT_FILE
    reports_dir = Path(args.reports_dir) if args.reports_dir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate] Loading and preprocessing data from: {mat_path}")
    epochs, labels, chan_names, fs = run_preprocessing_pipeline(mat_path, verbose=False)

    results = cross_validate_pipeline(
        pipeline_name=args.model,
        extractor_name=args.feature,
        epochs=epochs,
        labels=labels,
        chan_names=chan_names,
        n_splits=CV_FOLDS,
        use_smote=not args.no_smote,
        random_state=RANDOM_STATE,
        reports_dir=reports_dir,
        verbose=True,
    )

    print(f"\n[evaluate] ✅ Figures saved to: {reports_dir}")
    print(f"[evaluate] Mean AUC-ROC = {results['mean_auc']:.4f} "
          f"± {results['std_auc']:.4f}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run real-time BCI stream simulation with a trained pipeline."""
    import warnings
    from src.preprocessing import run_preprocessing_pipeline
    from src.data_loader import load_recording
    from src.features import get_extractor
    from src.models import build_pipeline
    from src.simulator import EEGStreamSimulator

    mat_path = Path(args.mat) if args.mat else MAT_FILE

    print(f"[simulate] Loading raw data from: {mat_path}")
    rec = load_recording(mat_path, verbose=True)

    print("[simulate] Preprocessing for training the pipeline...")
    epochs, labels, chan_names, fs = run_preprocessing_pipeline(mat_path, verbose=False)

    # Train on the full dataset
    print(f"[simulate] Training {args.model.upper()}+{args.feature.upper()} pipeline...")
    if args.model == "eegnet":
        pipeline = build_pipeline("eegnet",
                                   n_channels=epochs.shape[1],
                                   n_samples=epochs.shape[2])
    else:
        extractor = get_extractor(args.feature, chan_names=chan_names)
        pipeline  = build_pipeline(args.model, extractor=extractor,
                                    use_smote=not args.no_smote)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_bin = (labels == 1).astype(int)
        pipeline.fit(epochs, y_bin)

    print("[simulate] Pipeline trained. Starting stream simulation...")

    sim = EEGStreamSimulator(
        data=rec.data,
        markers=rec.markers,
        pipeline=pipeline,
        fs=rec.fs,
        speed_factor=args.speed,
    )
    sim.run(max_events=args.events)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="P300 BCI Pipeline — End-to-End Neuroengineering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common options
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mat", default=None, metavar="PATH",
                        help="Path to s01.mat (default: auto-detect from config)")
    common.add_argument("--notch", action="store_true",
                        help="Apply 50/60 Hz notch filter during preprocessing")

    # Model options
    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--model", default="lda",
                             choices=["lda", "svc", "eegnet"],
                             help="Classifier to use (default: lda)")
    model_args.add_argument("--feature", default="raw",
                             choices=["raw", "xdawn", "riemann"],
                             help="Feature extraction strategy (default: raw)")
    model_args.add_argument("--no-smote", action="store_true",
                             help="Disable SMOTE oversampling")
    model_args.add_argument("--reports-dir", default=None, metavar="DIR",
                             help="Directory for output figures")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # preprocess
    sp = subparsers.add_parser("preprocess", parents=[common],
                                help="Load + preprocess data and print tensor stats")
    sp.set_defaults(func=cmd_preprocess)

    # train
    sp = subparsers.add_parser("train", parents=[common, model_args],
                                help="Run k-fold CV and report AUC-ROC")
    sp.set_defaults(func=cmd_train)

    # evaluate
    sp = subparsers.add_parser("evaluate", parents=[common, model_args],
                                help="Full evaluation with figures saved to reports/")
    sp.set_defaults(func=cmd_evaluate)

    # simulate
    sp = subparsers.add_parser("simulate", parents=[common, model_args],
                                help="Real-time BCI stream simulation")
    sp.add_argument("--events", type=int, default=30,
                    help="Number of events to process (default: 30)")
    sp.add_argument("--speed", type=float, default=50.0,
                    help="Playback speed multiplier (default: 50× real-time)")
    sp.set_defaults(func=cmd_simulate)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
