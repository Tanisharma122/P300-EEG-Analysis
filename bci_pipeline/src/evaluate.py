"""
src/evaluate.py — Cross-Validation & Metric Evaluation Engine.

Implements Stratified K-Fold cross-validation with comprehensive metrics
tailored for the severe class imbalance in P300 detection:

Primary metric : AUC-ROC  (Area Under the ROC Curve)
Secondary      : Balanced Accuracy, F1-Score (Target class), MCC

All figures are saved to ``reports/figures/`` as publication-quality PNGs.

Usage
-----
    from src.evaluate import cross_validate_pipeline
    from src.models import build_pipeline

    results = cross_validate_pipeline(
        pipeline_name='lda',
        extractor_name='raw',
        epochs=epochs,
        labels=labels,
        chan_names=chan_names,
    )
    print(results['mean_auc'])
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        CV_FOLDS, RANDOM_STATE, REPORTS_DIR, MARKER_TARGET, MARKER_NONTARGET,
    )
except ImportError:
    CV_FOLDS = 5
    RANDOM_STATE = 42
    REPORTS_DIR = Path("reports/figures")
    MARKER_TARGET = 1
    MARKER_NONTARGET = 2

logger = logging.getLogger(__name__)

# Labels for display (binary: 1=target, 0=non-target)
CLASS_NAMES = ["Non-target", "Target"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics for a single fold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (1=target, 0=non-target).
    y_pred : np.ndarray
        Predicted binary labels.
    y_proba : np.ndarray
        Probability of the target class (shape ``(n_samples,)``).

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    auc     = roc_auc_score(y_true, y_proba)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mcc     = matthews_corrcoef(y_true, y_pred)

    return {
        "auc_roc":          auc,
        "balanced_accuracy": bal_acc,
        "f1_target":         f1,
        "mcc":               mcc,
    }


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _plot_roc_curves(
    fold_fprs: list[np.ndarray],
    fold_tprs: list[np.ndarray],
    fold_aucs: list[float],
    title: str,
    save_path: Path,
) -> None:
    """Plot per-fold ROC curves + mean ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(fold_fprs)))

    for i, (fpr, tpr, auc_val) in enumerate(zip(fold_fprs, fold_tprs, fold_aucs)):
        ax.plot(fpr, tpr, color=colors[i], alpha=0.5, linewidth=1.2,
                label=f"Fold {i+1} (AUC={auc_val:.3f})")
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr  = np.std(interp_tprs, axis=0)
    mean_auc = np.mean(fold_aucs)

    ax.plot(mean_fpr, mean_tpr, color="#f9c74f", linewidth=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {np.std(fold_aucs):.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.15, color="#f9c74f")
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="Chance")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=11)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    legend = ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved: %s", save_path)


def _plot_pr_curves(
    fold_precs: list[np.ndarray],
    fold_recs: list[np.ndarray],
    fold_aps: list[float],
    title: str,
    save_path: Path,
) -> None:
    """Plot per-fold Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(fold_precs)))

    for i, (prec, rec, ap) in enumerate(zip(fold_precs, fold_recs, fold_aps)):
        ax.plot(rec, prec, color=colors[i], alpha=0.55, linewidth=1.2,
                label=f"Fold {i+1} (AP={ap:.3f})")

    mean_ap = np.mean(fold_aps)
    ax.set_xlabel("Recall", color="white", fontsize=11)
    ax.set_ylabel("Precision", color="white", fontsize=11)
    ax.set_title(f"{title}\n(Mean AP = {mean_ap:.3f})", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved: %s", save_path)


def _plot_confusion_matrix(
    cm_agg: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot aggregated confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_agg,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for text in ax.texts:
        text.set_color("white")
    ax.spines[:].set_color("#444")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)


def _plot_metrics_summary(
    all_metrics: list[dict[str, float]],
    title: str,
    save_path: Path,
) -> None:
    """Bar chart of mean ± std across folds for each metric."""
    metric_names = list(all_metrics[0].keys())
    means = [np.mean([m[k] for m in all_metrics]) for k in metric_names]
    stds  = [np.std([m[k] for m in all_metrics])  for k in metric_names]

    labels_map = {
        "auc_roc": "AUC-ROC",
        "balanced_accuracy": "Bal. Accuracy",
        "f1_target": "F1 (Target)",
        "mcc": "MCC",
    }
    display_names = [labels_map.get(k, k) for k in metric_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    colors = ["#f9c74f", "#90be6d", "#f3722c", "#577590"]
    bars = ax.bar(display_names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="#222", linewidth=0.7)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (mean ± std over 5 folds)", color="white", fontsize=10)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metrics summary saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main cross-validation runner
# ---------------------------------------------------------------------------

def cross_validate_pipeline(
    pipeline_name: str,
    extractor_name: str,
    epochs: np.ndarray,
    labels: np.ndarray,
    chan_names: list[str],
    n_splits: int = CV_FOLDS,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
    reports_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Stratified K-Fold cross-validation on a named BCI pipeline.

    Enforces the no-leakage rule: all fitting (scaler, SMOTE, spatial filter,
    covariance reference) happens strictly inside each training fold.

    Parameters
    ----------
    pipeline_name : str
        Model name: ``'lda'``, ``'svc'``, or ``'eegnet'``.
    extractor_name : str
        Feature strategy: ``'raw'``, ``'xdawn'``, or ``'riemann'``.
    epochs : np.ndarray
        Preprocessed epoch tensor, shape ``(N, 32, 512)``.
    labels : np.ndarray
        Labels array, shape ``(N,)``. Values: 1=target, 2=non-target.
    chan_names : list[str]
        Ordered channel names.
    n_splits : int
        Number of CV folds (default 5).
    use_smote : bool
        Apply SMOTE inside training folds.
    random_state : int
        RNG seed for reproducibility.
    reports_dir : Path | None
        Directory to save figure outputs. Defaults to config.REPORTS_DIR.
    verbose : bool
        Print fold-level results to stdout.

    Returns
    -------
    dict[str, Any]
        Keys: ``fold_metrics``, ``mean_auc``, ``std_auc``, ``mean_metrics``,
        ``confusion_matrix``, ``report_str``.
    """
    from src.features import get_extractor
    from src.models import build_pipeline

    if reports_dir is None:
        reports_dir = REPORTS_DIR
    reports_dir = Path(reports_dir)

    tag = f"{pipeline_name}_{extractor_name}"
    print(f"\n{'='*65}")
    print(f"  Cross-Validation: model={pipeline_name.upper()} | "
          f"features={extractor_name.upper()} | folds={n_splits}")
    print(f"{'='*65}")

    # Binarise labels: 1=target → 1, 2=non-target → 0
    y_bin = (labels == MARKER_TARGET).astype(np.int32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    fold_precs, fold_recs, fold_aps = [], [], []
    cm_agg = np.zeros((2, 2), dtype=int)
    all_y_true, all_y_proba = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(epochs, y_bin)):
        X_train, X_test = epochs[train_idx], epochs[test_idx]
        y_train, y_test = y_bin[train_idx],  y_bin[test_idx]

        # Build fresh pipeline each fold (prevents any state leakage)
        if pipeline_name == "eegnet":
            pipe = build_pipeline("eegnet", n_channels=epochs.shape[1],
                                   n_samples=epochs.shape[2])
        else:
            extractor = get_extractor(extractor_name, chan_names=chan_names)
            pipe = build_pipeline(
                pipeline_name,
                extractor=extractor,
                use_smote=use_smote,
                chan_names=chan_names,
            )

        # --- Fit ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        # --- Predict ---
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, -1]   # P(target)

        # Metrics
        metrics = _compute_fold_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(metrics)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fold_fprs.append(fpr); fold_tprs.append(tpr)
        fold_aucs.append(metrics["auc_roc"])

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        fold_precs.append(prec); fold_recs.append(rec); fold_aps.append(ap)

        # Confusion matrix
        cm_fold = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_agg += cm_fold

        # Aggregated predictions
        all_y_true.extend(y_test.tolist())
        all_y_proba.extend(y_proba.tolist())

        if verbose:
            print(
                f"  Fold {fold_idx+1}/{n_splits} | "
                f"AUC={metrics['auc_roc']:.4f} | "
                f"BalAcc={metrics['balanced_accuracy']:.4f} | "
                f"F1={metrics['f1_target']:.4f} | "
                f"MCC={metrics['mcc']:.4f}"
            )

    # --- Aggregate metrics ---
    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }
    std_metrics = {
        k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }

    # --- Final classification report ---
    all_y_true_arr  = np.array(all_y_true)
    all_y_proba_arr = np.array(all_y_proba)
    all_y_pred_arr  = (all_y_proba_arr >= 0.5).astype(int)
    report_str = classification_report(
        all_y_true_arr, all_y_pred_arr, target_names=CLASS_NAMES, zero_division=0
    )

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  MEAN RESULTS ({n_splits}-fold CV):")
        for k, v in mean_metrics.items():
            print(f"    {k:<25} {v:.4f} ± {std_metrics[k]:.4f}")
        print(f"\n  Classification Report (aggregated):\n{report_str}")

    # --- Save figures ---
    _plot_roc_curves(
        fold_fprs, fold_tprs, fold_aucs,
        title=f"ROC Curves — {pipeline_name.upper()} + {extractor_name.upper()}",
        save_path=reports_dir / f"{tag}_roc.png",
    )
    _plot_pr_curves(
        fold_precs, fold_recs, fold_aps,
        title=f"Precision-Recall — {pipeline_name.upper()} + {extractor_name.upper()}",
        save_path=reports_dir / f"{tag}_pr.png",
    )
    _plot_confusion_matrix(
        cm_agg,
        title=f"Confusion Matrix (Aggregated) — {pipeline_name.upper()} + {extractor_name.upper()}",
        save_path=reports_dir / f"{tag}_cm.png",
    )
    _plot_metrics_summary(
        fold_metrics,
        title=f"Metric Summary — {pipeline_name.upper()} + {extractor_name.upper()}",
        save_path=reports_dir / f"{tag}_metrics.png",
    )

    return {
        "pipeline_name":  pipeline_name,
        "extractor_name": extractor_name,
        "fold_metrics":   fold_metrics,
        "mean_metrics":   mean_metrics,
        "std_metrics":    std_metrics,
        "mean_auc":       mean_metrics["auc_roc"],
        "std_auc":        std_metrics["auc_roc"],
        "confusion_matrix": cm_agg,
        "report_str":     report_str,
    }
