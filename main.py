"""
main.py — Phase 5: P300 BCI Speller System Entry Point.

This is the single entry point for the complete P300 BCI Speller system.
It orchestrates all five phases:

  Phase 1 (prep_engine)   — EEG preprocessing
  Phase 2 (model_engine)  — ML training & model persistence
  Phase 3 (decoder_engine)— Matrix decoding logic
  Phase 4 (app_ui)        — PyQt6 Light-Theme GUI
  Phase 5 (this file)     — System bootstrap & CLI

Usage
-----
    # Launch GUI (auto-trains model if missing)
    python main.py

    # Launch GUI with explicit paths
    python main.py --mat data_explore/s01.mat --model p300_lda_model.pkl

    # Train only (headless / server mode)
    python main.py --train-only

    # Run decoder smoke-test (no GUI)
    python main.py --decode-test --word HELLO

    # Quick pre-processing sanity check
    python main.py --preprocess-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — resolve project root and bci_pipeline package
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent          # "P300 Detection/"
_BCI  = _ROOT / "bci_pipeline"

for _p in (str(_ROOT), str(_BCI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAT   = _ROOT / "data_explore" / "s01.mat"
DEFAULT_MODEL = _ROOT / "p300_lda_model.pkl"

BANNER = r"""
 ╔═══════════════════════════════════════════════════════════════╗
 ║   P300 BCI Speller Detection & Simulation System  v1.0        ║
 ║   RSVP · LDA + xDAWN · 6×6 Matrix Paradigm                    ║
 ╚═══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_gui(args: argparse.Namespace) -> None:
    """Launch the PyQt6 speller GUI (default command)."""
    mat   = Path(args.mat)   if args.mat   else DEFAULT_MAT
    model = Path(args.model) if args.model else DEFAULT_MODEL

    # Validate paths
    if not mat.exists():
        logger.error("s01.mat not found: %s", mat)
        print(f"\n❌  Dataset not found: {mat}")
        print(f"    Please place s01.mat in:  {mat.parent}")
        sys.exit(1)

    # Auto-train model if missing
    if not model.exists():
        print(f"ℹ️   Model not found at: {model}")
        print("    Auto-training is available from the GUI ('⚙ Train Model' button).")
        print("    Or run:  python main.py --train-only\n")

    print(BANNER)
    print(f"  Dataset : {mat}")
    print(f"  Model   : {model}  {'[FOUND]' if model.exists() else '[NOT FOUND — train first]'}")
    print(f"\n  Launching GUI…\n")

    from app_ui import launch_app
    launch_app(mat_path=mat, model_path=model)


def cmd_train(args: argparse.Namespace) -> None:
    """Headless model training — trains and saves p300_lda_model.pkl."""
    mat   = Path(args.mat)   if args.mat   else DEFAULT_MAT
    model = Path(args.model) if args.model else DEFAULT_MODEL

    print(BANNER)
    print("  Mode: Headless training (no GUI)\n")

    if not mat.exists():
        logger.error("s01.mat not found: %s", mat)
        sys.exit(1)

    from model_engine import train_and_save

    pipeline, metrics = train_and_save(
        mat_path     = mat,
        model_path   = model,
        model_name   = args.model_type,
        feature_name = args.feature,
        use_smote    = not args.no_smote,
        n_folds      = args.folds,
        verbose      = True,
    )

    mean_auc = metrics["mean_auc"]
    std_auc  = metrics["std_auc"]
    status   = "✅  TARGET MET" if mean_auc >= 0.80 else "⚠️  Below 0.80"

    print("\n" + "═" * 65)
    print(f"  AUC-ROC  : {mean_auc:.4f} ± {std_auc:.4f}   {status}")
    print(f"  Model    : {model}")
    print("═" * 65 + "\n")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run preprocessing pipeline and print tensor statistics."""
    mat = Path(args.mat) if args.mat else DEFAULT_MAT

    print(BANNER)
    print("  Mode: Preprocessing only\n")

    if not mat.exists():
        logger.error("s01.mat not found: %s", mat)
        sys.exit(1)

    from prep_engine import load_and_preprocess

    result = load_and_preprocess(mat, verbose=True)

    print("\n── Tensor Statistics ─────────────────────────────────────────")
    print(f"  Selected channels  : {result.epochs.shape}  (N × ch × T)")
    print(f"  Full channels      : {result.epochs_full.shape}  (N × 32 × T)")
    print(f"  Max amplitude      : {float(result.epochs.max()):.2f} µV")
    print(f"  Mean amplitude     : {float(result.epochs.mean()):.4f} µV")
    print(f"  Channels           : {result.chan_names}")
    print("──────────────────────────────────────────────────────────────\n")


def cmd_decode_test(args: argparse.Namespace) -> None:
    """Run decoder smoke-test + full simulation (requires trained model)."""
    mat   = Path(args.mat)   if args.mat   else DEFAULT_MAT
    model = Path(args.model) if args.model else DEFAULT_MODEL
    word  = args.word.upper()

    print(BANNER)
    print(f"  Mode: Decoder simulation — word='{word}'\n")

    if not mat.exists():
        logger.error("s01.mat not found: %s", mat)
        sys.exit(1)

    if not model.exists():
        print(f"❌  Model not found: {model}")
        print("    Run 'python main.py --train-only' first.\n")
        sys.exit(1)

    from prep_engine    import load_and_preprocess
    from model_engine   import load_model
    from decoder_engine import batch_simulate, P300Decoder, GRID, CHAR_TO_POS

    print("[main] Loading model…")
    pipeline = load_model(model)

    print("[main] Preprocessing EEG data…")
    result = load_and_preprocess(mat, verbose=False)

    print(f"\n[main] Simulating '{word}'  ({args.reps} repetitions per character)…")
    results = batch_simulate(
        pipeline, result.epochs_full, result.labels,
        chars=word, n_reps=args.reps, verbose=True,
    )

    correct = sum(r["correct"] for r in results)
    print(f"\n  Character accuracy : {correct}/{len(results)}  "
          f"({100*correct/max(1,len(results)):.0f}%)")

    if results:
        mean_conf = sum(r["confidence"] for r in results) / len(results)
        print(f"  Mean confidence   : {mean_conf:.3f}")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="P300 BCI Speller Detection & Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch GUI (auto-detects model)
  python main.py --train-only             # Headless training only
  python main.py --preprocess-only        # Preprocess & print stats
  python main.py --decode-test --word HI  # Simulate spelling "HI"
  python main.py --mat data/s01.mat       # Custom dataset path
        """,
    )

    # Global path options (available to all modes)
    parser.add_argument(
        "--mat", default=None, metavar="PATH",
        help=f"Path to s01.mat (default: {DEFAULT_MAT})",
    )
    parser.add_argument(
        "--model", default=None, metavar="PATH",
        help=f"Path to p300_lda_model.pkl (default: {DEFAULT_MODEL})",
    )

    # Mode flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train-only", action="store_true",
        help="Train and save the model (headless — no GUI)",
    )
    mode.add_argument(
        "--preprocess-only", action="store_true",
        help="Run preprocessing only and print tensor statistics",
    )
    mode.add_argument(
        "--decode-test", action="store_true",
        help="Run decoder simulation test (requires trained model)",
    )

    # Training options
    train_grp = parser.add_argument_group("Training options (--train-only)")
    train_grp.add_argument(
        "--model-type", default="lda", choices=["lda", "svc"],
        metavar="TYPE",
        help="Classifier type: lda or svc (default: lda)",
    )
    train_grp.add_argument(
        "--feature", default="xdawn",
        choices=["raw", "xdawn", "riemann"],
        metavar="FEAT",
        help="Feature extractor: raw, xdawn, riemann (default: xdawn)",
    )
    train_grp.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling (not recommended — 1:14 imbalance)",
    )
    train_grp.add_argument(
        "--folds", type=int, default=5, metavar="N",
        help="Number of CV folds (default: 5)",
    )

    # Decode-test options
    dec_grp = parser.add_argument_group("Decode-test options (--decode-test)")
    dec_grp.add_argument(
        "--word", default="P300", metavar="WORD",
        help="Word to simulate spelling (default: P300)",
    )
    dec_grp.add_argument(
        "--reps", type=int, default=7, metavar="N",
        help="Flash repetitions per character (default: 7)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.train_only:
        cmd_train(args)
    elif args.preprocess_only:
        cmd_preprocess(args)
    elif args.decode_test:
        cmd_decode_test(args)
    else:
        cmd_gui(args)


if __name__ == "__main__":
    main()
