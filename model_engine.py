"""
model_engine.py — Phase 2: Feature Engineering & Classifier Pipeline.

Trains P300 binary classifiers (LDA with shrinkage + linear SVM) using
5-fold Stratified Cross-Validation, handles class imbalance via SMOTE and
class weighting, prints evaluation metrics (ROC-AUC, Precision, Recall,
Confusion Matrix), and saves the best model as ``p300_lda_model.pkl``.

Usage
-----
    # Train + evaluate + save model (all-in-one)
    python model_engine.py

    # Load a previously saved model for inference
    from model_engine import load_model, predict_proba
    pipeline = load_model()
    probs = predict_proba(pipeline, epochs_3d)   # shape (N,)

    # Programmatic training
    from model_engine import train_and_save
    pipeline, metrics = train_and_save(mat_path="data_explore/s01.mat")
    print(f"AUC-ROC: {metrics['mean_auc']:.4f}")
"""

from __future__ import annotations

import logging
import sys
import warnings
import argparse
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_BCI  = _ROOT / "bci_pipeline"
for _p in (str(_ROOT), str(_BCI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAT       = _ROOT / "data_explore" / "s01.mat"
DEFAULT_MODEL_PKL = _ROOT / "p300_lda_model.pkl"
RANDOM_STATE      = 42
CV_FOLDS          = 5
MARKER_TARGET     = 1
MARKER_NONTARGET  = 2


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def train_and_save(
    mat_path: str | Path = DEFAULT_MAT,
    model_path: str | Path = DEFAULT_MODEL_PKL,
    model_name: str = "lda",
    feature_name: str = "xdawn",
    use_smote: bool = True,
    n_folds: int = CV_FOLDS,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Train a P300 classifier and persist it to disk.

    Parameters
    ----------
    mat_path : str | Path
        Path to ``s01.mat``.
    model_path : str | Path
        Where to save the fitted ``joblib`` pickle (``p300_lda_model.pkl``).
    model_name : str
        Classifier: ``'lda'`` (default), ``'svc'``.
    feature_name : str
        Feature extractor: ``'xdawn'`` (default), ``'raw'``, ``'riemann'``.
    use_smote : bool
        Apply SMOTE inside training folds to counter 1:14 class imbalance.
    n_folds : int
        Number of stratified CV folds.
    verbose : bool
        Print detailed progress.

    Returns
    -------
    pipeline : sklearn-compatible fitted pipeline
    metrics : dict
        Keys: ``mean_auc``, ``std_auc``, ``mean_metrics``, ``confusion_matrix``,
        ``report_str``.
    """
    import joblib
    from prep_engine import load_and_preprocess
    from src.evaluate import cross_validate_pipeline
    from src.features import get_extractor
    from src.models import build_pipeline

    mat_path   = Path(mat_path).resolve()
    model_path = Path(model_path)

    # ── Load & preprocess (using full 32-ch tensor for xDAWN/Riemann) ────
    if verbose:
        print("\n" + "═" * 65)
        print("  P300 Model Engine — Phase 2: Feature Engineering & Training")
        print("═" * 65)
        print(f"\n[model_engine] Loading & preprocessing: {mat_path.name}")

    result = load_and_preprocess(mat_path, verbose=verbose)

    # Use full 32-channel epochs for xDAWN (spatial filter needs all channels)
    X = result.epochs_full       # (N, 32, T)
    y = result.labels            # (N,) — 1=target, 2=non-target
    chan_names = result.all_chan_names

    n_tgt = int(np.sum(y == MARKER_TARGET))
    n_nt  = int(np.sum(y == MARKER_NONTARGET))

    if verbose:
        print(f"\n[model_engine] Dataset summary:")
        print(f"  Epoch tensor   : {X.shape}  (N × C × T)")
        print(f"  Target epochs  : {n_tgt}   ({100*n_tgt/len(y):.1f}%)")
        print(f"  Non-target     : {n_nt}  ({100*n_nt/len(y):.1f}%)")
        print(f"  Imbalance      : 1 : {int(n_nt/max(1,n_tgt))}")
        print(f"\n[model_engine] Strategy: {model_name.upper()} + "
              f"{feature_name.upper()} | SMOTE={use_smote} | {n_folds}-fold CV")

    # ── Cross-validation ──────────────────────────────────────────────────
    reports_dir = _ROOT / "bci_pipeline" / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cv_results = cross_validate_pipeline(
        pipeline_name  = model_name,
        extractor_name = feature_name,
        epochs         = X,
        labels         = y,
        chan_names     = chan_names,
        n_splits       = n_folds,
        use_smote      = use_smote,
        random_state   = RANDOM_STATE,
        reports_dir    = reports_dir,
        verbose        = verbose,
    )

    mean_auc = cv_results["mean_auc"]
    std_auc  = cv_results["std_auc"]

    if verbose:
        auc_status = "✅  TARGET MET" if mean_auc >= 0.80 else "⚠️  Below 0.80 target"
        print(f"\n[model_engine] AUC-ROC = {mean_auc:.4f} ± {std_auc:.4f}  {auc_status}")

    # ── Train final model on ALL data ──────────────────────────────────────
    if verbose:
        print(f"\n[model_engine] Training final {model_name.upper()} pipeline "
              f"on full dataset ({len(y)} epochs)...")

    extractor = get_extractor(feature_name, chan_names=chan_names)
    pipeline  = build_pipeline(model_name, extractor=extractor, use_smote=use_smote)

    y_bin = (y == MARKER_TARGET).astype(np.int32)   # 1=target, 0=non-target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y_bin)

    # ── Save model ────────────────────────────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    if verbose:
        print(f"[model_engine] ✅  Model saved → {model_path}")
        print(f"[model_engine] Figures saved → {reports_dir}/\n")

    return pipeline, cv_results


def load_model(model_path: str | Path = DEFAULT_MODEL_PKL) -> Any:
    """Load a previously saved P300 classifier pipeline from disk.

    Parameters
    ----------
    model_path : str | Path
        Path to the ``p300_lda_model.pkl`` file produced by :func:`train_and_save`.

    Returns
    -------
    Any
        Fitted sklearn-compatible pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    import joblib
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run 'python model_engine.py' to train and save the model first."
        )
    pipeline = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return pipeline


def predict_proba(pipeline: Any, epochs: np.ndarray) -> np.ndarray:
    """Run inference on a batch of epochs and return P300 target probabilities.

    Parameters
    ----------
    pipeline : Any
        Fitted sklearn pipeline from :func:`load_model` or :func:`train_and_save`.
    epochs : np.ndarray
        Epoch tensor, shape ``(N, n_channels, T)``  (can be full 32-ch or
        channel-selected, depending on what the pipeline was trained with).

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — P(target) ∈ [0.0, 1.0] for each epoch.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba_2d = pipeline.predict_proba(epochs)   # (N, 2)
    # Column -1 is always P(target=1) — works for both sklearn and imblearn pipelines
    return proba_2d[:, -1].astype(np.float64)


def evaluate_model(
    pipeline: Any,
    epochs: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a fitted pipeline on labelled epoch data.

    Parameters
    ----------
    pipeline : Any
        Fitted sklearn pipeline.
    epochs : np.ndarray
        Epoch tensor, shape ``(N, C, T)``.
    labels : np.ndarray
        Labels, shape ``(N,)``. 1=target, 2=non-target.
    verbose : bool
        Print metrics to stdout.

    Returns
    -------
    dict[str, float]
        Keys: ``auc_roc``, ``balanced_accuracy``, ``f1_target``, ``mcc``.
    """
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score, f1_score,
        matthews_corrcoef, confusion_matrix, classification_report,
    )

    y_bin   = (labels == MARKER_TARGET).astype(np.int32)
    y_proba = predict_proba(pipeline, epochs)
    y_pred  = (y_proba >= 0.5).astype(np.int32)

    metrics = {
        "auc_roc":            float(roc_auc_score(y_bin, y_proba)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_bin, y_pred)),
        "f1_target":          float(f1_score(y_bin, y_pred, pos_label=1, zero_division=0)),
        "mcc":                float(matthews_corrcoef(y_bin, y_pred)),
    }

    if verbose:
        print("\n[model_engine] ── Held-out Evaluation Metrics ──────────────")
        for k, v in metrics.items():
            print(f"  {k:<25} {v:.4f}")
        cm = confusion_matrix(y_bin, y_pred, labels=[0, 1])
        print(f"\n  Confusion Matrix:\n{cm}")
        print(f"\n  Classification Report:\n"
              f"{classification_report(y_bin, y_pred, target_names=['Non-target','Target'], zero_division=0)}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P300 Phase-2: Train LDA/SVM classifier and save model",
    )
    parser.add_argument("--mat",     default=None, metavar="PATH",
                        help="Path to s01.mat")
    parser.add_argument("--out",     default=None, metavar="PATH",
                        help="Output path for model pickle (default: p300_lda_model.pkl)")
    parser.add_argument("--model",   default="lda", choices=["lda", "svc"],
                        help="Classifier (default: lda)")
    parser.add_argument("--feature", default="xdawn",
                        choices=["raw", "xdawn", "riemann"],
                        help="Feature extractor (default: xdawn)")
    parser.add_argument("--no-smote", action="store_true",
                        help="Disable SMOTE oversampling")
    parser.add_argument("--folds", type=int, default=CV_FOLDS,
                        help=f"CV folds (default: {CV_FOLDS})")
    args = parser.parse_args()

    mat   = Path(args.mat)   if args.mat else DEFAULT_MAT
    out   = Path(args.out)   if args.out else DEFAULT_MODEL_PKL

    pipeline, results = train_and_save(
        mat_path     = mat,
        model_path   = out,
        model_name   = args.model,
        feature_name = args.feature,
        use_smote    = not args.no_smote,
        n_folds      = args.folds,
        verbose      = True,
    )

    print("\n" + "═" * 65)
    print(f"  FINAL RESULT — AUC-ROC: {results['mean_auc']:.4f} "
          f"± {results['std_auc']:.4f}")
    if results['mean_auc'] >= 0.80:
        print("  ✅  AUC-ROC > 0.80 — Target achieved!")
    else:
        print("  ⚠️  AUC-ROC below 0.80 — consider xDAWN or more epochs")
    print("═" * 65)
