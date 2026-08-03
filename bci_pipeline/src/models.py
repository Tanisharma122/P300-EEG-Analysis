"""
src/models.py — Imbalance-Aware Classifier Suite for P300 BCI.

Provides a unified BCIModelPipeline builder that wraps feature extractors +
classifiers into sklearn-compatible Pipeline objects, with optional SMOTE
oversampling applied strictly inside training folds.

Available model configurations
-------------------------------
- ``'lda'``     : Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
- ``'svc'``     : RBF-kernel SVM with balanced class weights.
- ``'eegnet'``  : PyTorch EEGNet (depthwise separable CNN for 32-ch EEG).

Usage
-----
    from src.models import build_pipeline
    from src.features import get_extractor

    extractor = get_extractor('xdawn', chan_names=chan_names)
    pipeline = build_pipeline('lda', extractor=extractor, use_smote=True)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        RANDOM_STATE, MARKER_TARGET, MARKER_NONTARGET,
        EEGNET_TEMPORAL_FILTERS, EEGNET_DEPTH_MULTIPLIER,
        EEGNET_DROPOUT_RATE, EEGNET_EPOCHS, EEGNET_BATCH_SIZE,
        EEGNET_LR, N_CHANNELS, N_SAMPLES_EPOCH, FS,
    )
except ImportError:
    RANDOM_STATE = 42
    MARKER_TARGET = 1
    MARKER_NONTARGET = 2
    EEGNET_TEMPORAL_FILTERS = 8
    EEGNET_DEPTH_MULTIPLIER = 2
    EEGNET_DROPOUT_RATE = 0.5
    EEGNET_EPOCHS = 50
    EEGNET_BATCH_SIZE = 32
    EEGNET_LR = 1e-3
    N_CHANNELS = 32
    N_SAMPLES_EPOCH = 512
    FS = 512.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EEGNet — PyTorch implementation
# ---------------------------------------------------------------------------

class EEGNetModule:
    """EEGNet architecture for P300 classification.

    A compact depthwise-separable CNN designed specifically for EEG BCIs,
    based on Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural
    Network for EEG-based Brain–Computer Interfaces".

    Architecture
    ------------
    Block 1: Temporal convolution (learns temporal filters)
            + Depthwise spatial convolution (learns channel-wise filters)
            + Batch norm + ELU + Average pool + Dropout
    Block 2: Depthwise separable convolution (learns temporal summary)
            + Batch norm + ELU + Average pool + Dropout
    Classifier: Flatten → Linear → output (2 classes)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_samples: int  = N_SAMPLES_EPOCH,
        n_classes: int  = 2,
        temporal_filters: int   = EEGNET_TEMPORAL_FILTERS,
        depth_multiplier: int   = EEGNET_DEPTH_MULTIPLIER,
        dropout_rate: float     = EEGNET_DROPOUT_RATE,
        fs: float = FS,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
        except ImportError:
            raise ImportError("PyTorch is required for EEGNet. pip install torch")

        self.n_channels      = n_channels
        self.n_samples       = n_samples
        self.n_classes       = n_classes
        self.temporal_filters = temporal_filters
        self.depth_multiplier = depth_multiplier
        self.dropout_rate    = dropout_rate
        self.fs              = fs

        self.model = self._build()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("EEGNet device: %s", self.device)

    def _build(self):
        import torch.nn as nn

        T = self.n_samples
        C = self.n_channels
        F1 = self.temporal_filters
        D  = self.depth_multiplier
        F2 = F1 * D
        kern_len = int(self.fs // 2)  # 256 for 512 Hz

        class EEGNetModel(nn.Module):
            def __init__(self, C, T, F1, D, F2, kern_len, dropout, n_classes):
                super().__init__()
                # Block 1: Temporal + Depthwise Spatial
                self.block1 = nn.Sequential(
                    # Temporal convolution: learns frequency-specific filters
                    nn.Conv2d(1, F1, kernel_size=(1, kern_len),
                              padding=(0, kern_len // 2), bias=False),
                    nn.BatchNorm2d(F1),
                    # Depthwise spatial convolution over all channels
                    nn.Conv2d(F1, F1 * D, kernel_size=(C, 1),
                              groups=F1, bias=False),
                    nn.BatchNorm2d(F1 * D),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout(dropout),
                )
                # Block 2: Separable (depthwise + pointwise) temporal
                self.block2 = nn.Sequential(
                    nn.Conv2d(F2, F2, kernel_size=(1, 16),
                              padding=(0, 8), groups=F2, bias=False),
                    nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(F2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 8)),
                    nn.Dropout(dropout),
                )
                # Compute flattened size
                with self._no_grad_context():
                    dummy = self._dummy_forward(C, T, F1, D, F2, kern_len, dropout)
                flat_size = dummy

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_size, n_classes),
                )

            @staticmethod
            def _no_grad_context():
                import torch
                return torch.no_grad()

            def _dummy_forward(self, C, T, F1, D, F2, kern_len, dropout):
                import torch
                x = torch.zeros(1, 1, C, T)
                x = self.block1(x)
                x = self.block2(x)
                return int(x.numel())

            def forward(self, x):
                # x: (batch, 1, channels, samples)
                x = self.block1(x)
                x = self.block2(x)
                return self.classifier(x)

        return EEGNetModel(C, T, F1, D, F2, kern_len, self.dropout_rate, self.n_classes)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for EEGNet (PyTorch).

    Inputs are expected as ``(n_epochs, n_channels, n_samples)`` 3-D arrays.
    Applies label binarization: target class → class 1, non-target → class 0.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 32).
    n_samples : int
        Epoch length in samples (default 512).
    epochs : int
        Training epochs (default 50).
    batch_size : int
        Mini-batch size (default 32).
    lr : float
        Adam learning rate (default 1e-3).
    dropout_rate : float
        Dropout probability (default 0.5).
    class_weight : str | None
        ``'balanced'`` applies inverse-frequency weights to the loss, or None.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_samples: int  = N_SAMPLES_EPOCH,
        epochs: int     = EEGNET_EPOCHS,
        batch_size: int = EEGNET_BATCH_SIZE,
        lr: float       = EEGNET_LR,
        dropout_rate: float = EEGNET_DROPOUT_RATE,
        class_weight: str | None = "balanced",
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_channels   = n_channels
        self.n_samples    = n_samples
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.dropout_rate = dropout_rate
        self.class_weight = class_weight
        self.random_state = random_state
        self._net = None
        self.classes_ = np.array([0, 1])

    def _to_tensor(self, X: np.ndarray):
        import torch
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        """Train EEGNet on epoch data.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_epochs, n_channels, n_samples)``.
        y : np.ndarray
            Labels. 1=target, 2=non-target (or 0/1 binary).
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)

        # Binarise labels: target(1) → 1, non-target(2) → 0
        y_bin = (y == MARKER_TARGET).astype(np.int64)

        # Build model
        eegnet_mod = EEGNetModule(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            dropout_rate=self.dropout_rate,
        )
        self._net    = eegnet_mod.model
        self._device = eegnet_mod.device

        # Add channel dimension: (N, C, T) → (N, 1, C, T)
        X_t = self._to_tensor(X[:, np.newaxis, :, :]).to(self._device)
        y_t = torch.tensor(y_bin, dtype=torch.long).to(self._device)

        # Class weights for imbalance
        if self.class_weight == "balanced":
            n_neg = int((y_bin == 0).sum())
            n_pos = int((y_bin == 1).sum())
            w = torch.tensor(
                [n_pos / (n_neg + n_pos + 1e-8), n_neg / (n_neg + n_pos + 1e-8)],
                dtype=torch.float32,
            ).to(self._device)
            criterion = nn.CrossEntropyLoss(weight=w)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        dataset   = TensorDataset(X_t, y_t)
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        for ep in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out  = self._net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            if (ep + 1) % 10 == 0:
                logger.debug("EEGNet epoch %d/%d — loss %.4f", ep + 1, self.epochs, epoch_loss / len(X))

        logger.info("EEGNet training complete (%d epochs).", self.epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n_epochs, 2)`` — [P(non-target), P(target)].
        """
        import torch
        import torch.nn.functional as F

        if self._net is None:
            raise RuntimeError("Call fit() before predict_proba().")

        self._net.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X[:, np.newaxis, :, :]).to(self._device)
            logits = self._net(X_t)
            proba  = F.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_lda_pipeline(
    extractor: BaseEstimator,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build an LDA pipeline with optional SMOTE oversampling.

    Parameters
    ----------
    extractor : BaseEstimator
        Fitted or unfitted feature extractor (sklearn transformer).
    use_smote : bool
        If True, use ``imblearn.pipeline.Pipeline`` with SMOTE in the
        training phase only.
    random_state : int
        Random seed for SMOTE.

    Returns
    -------
    Pipeline (sklearn or imblearn)
    """
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    scaler = StandardScaler()

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            smote = SMOTE(random_state=random_state, k_neighbors=3)
            return ImbPipeline([
                ("extract", extractor),
                ("scale",   scaler),
                ("smote",   smote),
                ("clf",     lda),
            ])
        except ImportError:
            logger.warning("imbalanced-learn not found — building LDA without SMOTE.")

    return Pipeline([
        ("extract", extractor),
        ("scale",   scaler),
        ("clf",     lda),
    ])


def build_svc_pipeline(
    extractor: BaseEstimator,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a balanced-weight RBF-SVM pipeline with optional SMOTE.

    Parameters
    ----------
    extractor : BaseEstimator
        Feature extractor (sklearn transformer).
    use_smote : bool
        If True, apply SMOTE inside training folds.
    random_state : int
        RNG seed.
    """
    svc = SVC(kernel="rbf", class_weight="balanced", probability=True,
              random_state=random_state)
    scaler = StandardScaler()

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            smote = SMOTE(random_state=random_state, k_neighbors=3)
            return ImbPipeline([
                ("extract", extractor),
                ("scale",   scaler),
                ("smote",   smote),
                ("clf",     svc),
            ])
        except ImportError:
            logger.warning("imbalanced-learn not found — building SVC without SMOTE.")

    return Pipeline([
        ("extract", extractor),
        ("scale",   scaler),
        ("clf",     svc),
    ])


def build_eegnet_pipeline(
    n_channels: int = N_CHANNELS,
    n_samples: int  = N_SAMPLES_EPOCH,
    epochs: int     = EEGNET_EPOCHS,
) -> EEGNetClassifier:
    """Return an EEGNetClassifier instance (no wrapping extractor needed).

    EEGNet operates directly on raw epoch tensors ``(N, C, T)``.
    """
    return EEGNetClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        epochs=epochs,
    )


def build_pipeline(
    model_name: str,
    extractor: BaseEstimator | None = None,
    use_smote: bool = True,
    chan_names: list[str] | None = None,
    **kwargs,
) -> Any:
    """Factory: create a complete model pipeline by name.

    Parameters
    ----------
    model_name : str
        One of ``'lda'``, ``'svc'``, ``'eegnet'``.
    extractor : BaseEstimator | None
        Feature extractor. Required for 'lda' and 'svc'. If None, defaults
        to ``RawWindowExtractor``.
    use_smote : bool
        Apply SMOTE inside pipeline (LDA/SVC only).
    chan_names : list[str] | None
        Passed to default extractor if extractor is None.
    **kwargs
        Extra args forwarded to the model constructor.

    Returns
    -------
    sklearn-compatible pipeline or EEGNetClassifier.
    """
    from src.features import RawWindowExtractor

    model_name = model_name.lower()

    if model_name == "eegnet":
        return build_eegnet_pipeline(**kwargs)

    if extractor is None:
        extractor = RawWindowExtractor(chan_names=chan_names)

    if model_name == "lda":
        return build_lda_pipeline(extractor, use_smote=use_smote)
    elif model_name == "svc":
        return build_svc_pipeline(extractor, use_smote=use_smote)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: 'lda', 'svc', 'eegnet'"
        )
