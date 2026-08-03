"""
src/features.py — P300 BCI Feature Extraction Engine.

Provides three sklearn-compatible feature extractors with fit/transform API:

1. RawWindowExtractor   — Downsample P300 window on key channels, flatten.
2. XDawnExtractor       — xDAWN spatial filter (maximizes SSNR for target class).
3. RiemannExtractor     — Covariance matrices projected to Tangent Space.

All extractors enforce the no-leakage rule: fitting happens only on training data
inside cross-validation folds.

Usage
-----
    from src.features import RawWindowExtractor, XDawnExtractor, RiemannExtractor

    extractor = XDawnExtractor(n_components=4)
    extractor.fit(X_train, y_train)
    X_feat = extractor.transform(X_test)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sps
from sklearn.base import BaseEstimator, TransformerMixin

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        FS, CHANNEL_NAMES, P300_CHANNELS,
        P300_WINDOW_START, P300_WINDOW_END,
        DOWNSAMPLE_FREQ, N_XDAWN_COMPONENTS,
        MARKER_TARGET, EPOCH_TMIN,
    )
except ImportError:
    FS = 512.0
    CHANNEL_NAMES = [
        "FP1", "AF3", "F7",  "F3",  "FC1", "FC5", "T7",  "C3",
        "CP1", "CP5", "P7",  "P3",  "Pz",  "PO3", "O1",  "Oz",
        "O2",  "PO4", "P4",  "P8",  "CP6", "CP2", "C4",  "T8",
        "FC6", "FC2", "F4",  "F8",  "AF4", "FP2", "FZ",  "Cz",
    ]
    P300_CHANNELS = ["Cz", "Pz", "CP1", "CP2", "FC1", "FC2", "C3", "C4"]
    P300_WINDOW_START = 0.25
    P300_WINDOW_END   = 0.50
    DOWNSAMPLE_FREQ   = 20.0
    N_XDAWN_COMPONENTS = 4
    MARKER_TARGET = 1
    EPOCH_TMIN = -0.2

logger = logging.getLogger(__name__)


def _get_channel_indices(chan_names: list[str], target_channels: list[str]) -> list[int]:
    """Return indices of target channels in the chan_names list."""
    name_to_idx = {n: i for i, n in enumerate(chan_names)}
    indices = []
    for ch in target_channels:
        if ch in name_to_idx:
            indices.append(name_to_idx[ch])
        else:
            logger.warning("Channel '%s' not found in %s — skipping.", ch, chan_names)
    return indices


# ---------------------------------------------------------------------------
# Strategy 1: Raw Window Downsampling
# ---------------------------------------------------------------------------

class RawWindowExtractor(BaseEstimator, TransformerMixin):
    """Extract and downsample the P300 time window across key channels.

    Selects the 250–500 ms window from P300-relevant channels (Cz, Pz, CP1,
    CP2, FC1, FC2, C3, C4), downsamples to ~20 Hz, and flattens to a 1-D
    feature vector per epoch.

    This extractor has no learnable parameters — ``fit`` is a no-op.

    Parameters
    ----------
    chan_names : list[str]
        Ordered list of all channel names in the epoch tensor.
    fs : float
        Sampling frequency in Hz.
    p300_channels : list[str] | None
        Channels to extract. Defaults to the 8 P300-relevant electrodes.
    window_start : float
        P300 window start in seconds (default 0.25).
    window_end : float
        P300 window end in seconds (default 0.50).
    downsample_freq : float
        Target frequency for downsampling (default 20 Hz).
    epoch_tmin : float
        Epoch start time relative to stimulus (default −0.2 s).
    """

    def __init__(
        self,
        chan_names: list[str] | None = None,
        fs: float = FS,
        p300_channels: list[str] | None = None,
        window_start: float = P300_WINDOW_START,
        window_end: float   = P300_WINDOW_END,
        downsample_freq: float = DOWNSAMPLE_FREQ,
        epoch_tmin: float = EPOCH_TMIN,
    ) -> None:
        self.chan_names     = chan_names or list(CHANNEL_NAMES)
        self.fs             = fs
        self.p300_channels  = p300_channels or list(P300_CHANNELS)
        self.window_start   = window_start
        self.window_end     = window_end
        self.downsample_freq = downsample_freq
        self.epoch_tmin     = epoch_tmin

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RawWindowExtractor":
        """No-op (stateless transformer)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract features from epoch tensor.

        Parameters
        ----------
        X : np.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            Feature matrix, shape ``(n_epochs, n_features)``.
        """
        n_epochs, n_channels, n_samples = X.shape
        t = np.linspace(self.epoch_tmin, self.epoch_tmin + n_samples / self.fs, n_samples)

        # Time mask for P300 window
        win_mask = (t >= self.window_start) & (t < self.window_end)

        # Channel indices
        ch_idx = _get_channel_indices(self.chan_names, self.p300_channels)
        if not ch_idx:
            raise ValueError("No valid P300 channels found — check chan_names.")

        window_data = X[:, ch_idx, :][:, :, win_mask]  # (N, n_p300_ch, n_win_samples)

        # Downsample each channel via scipy resample
        orig_len = window_data.shape[2]
        target_len = max(1, int(round(orig_len * self.downsample_freq / self.fs)))
        downsampled = sps.resample(window_data, target_len, axis=2)  # (N, n_ch, target_len)

        features = downsampled.reshape(n_epochs, -1)  # flatten
        logger.debug(
            "RawWindowExtractor: %d epochs → feature shape %s",
            n_epochs, features.shape,
        )
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# Strategy 2: xDAWN Spatial Filtering
# ---------------------------------------------------------------------------

class XDawnExtractor(BaseEstimator, TransformerMixin):
    """xDAWN spatial filter for P300 signal enhancement.

    Wraps ``pyriemann.spatialfilters.Xdawn`` which maximises the Signal-to-
    Signal-plus-Noise Ratio (SSNR) for the target ERP class.

    The spatial filter is fitted strictly on training data and applied to test
    data, preventing any cross-validation leakage.

    Parameters
    ----------
    n_components : int
        Number of xDAWN components to retain per class.
    chan_names : list[str]
        Ordered channel names (used for logging only).
    flatten : bool
        If True (default), flatten the filtered epochs to 1-D feature vectors.
    """

    def __init__(
        self,
        n_components: int = N_XDAWN_COMPONENTS,
        chan_names: list[str] | None = None,
        flatten: bool = True,
    ) -> None:
        self.n_components = n_components
        self.chan_names    = chan_names or list(CHANNEL_NAMES)
        self.flatten       = flatten
        self._xdawn = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XDawnExtractor":
        """Fit xDAWN spatial filters on training epochs.

        Parameters
        ----------
        X : np.ndarray
            Training epochs, shape ``(n_epochs, n_channels, n_samples)``.
        y : np.ndarray
            Labels (1=target, 2=non-target).
        """
        try:
            from pyriemann.spatialfilters import Xdawn
        except ImportError:
            raise ImportError(
                "pyriemann is required for XDawnExtractor. "
                "Install with: pip install pyriemann"
            )

        # Convert labels to binary: 1=target, 0=non-target (pyriemann convention)
        y_bin = (y == MARKER_TARGET).astype(int)

        self._xdawn = Xdawn(nfilter=self.n_components)
        self._xdawn.fit(X, y_bin)
        logger.info("XDawnExtractor fitted with %d components.", self.n_components)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted xDAWN filters.

        Parameters
        ----------
        X : np.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            Filtered + (optionally flattened) features.
        """
        if self._xdawn is None:
            raise RuntimeError("Call fit() before transform().")

        X_filtered = self._xdawn.transform(X)  # (N, 2*n_components, T)
        if self.flatten:
            return X_filtered.reshape(X_filtered.shape[0], -1).astype(np.float64)
        return X_filtered.astype(np.float64)


# ---------------------------------------------------------------------------
# Strategy 3: Riemannian Geometry
# ---------------------------------------------------------------------------

class RiemannExtractor(BaseEstimator, TransformerMixin):
    """Riemannian tangent-space feature extractor for P300 classification.

    Pipeline:
      1. Estimate regularised covariance matrix per epoch (``oas`` or ``lwf``).
      2. Project all covariance matrices from the SPD manifold to the Euclidean
         tangent space anchored at the geometric mean (fitted on training data).
      3. Return the upper-triangular tangent vectors as features.

    Parameters
    ----------
    estimator : str
        Covariance estimator type for ``pyriemann.estimation.Covariances``.
        Options: ``'cov'``, ``'lwf'``, ``'oas'``, ``'scm'``.
    metric : str
        Riemannian metric for ``TangentSpace``. Default ``'riemann'``.
    """

    def __init__(
        self,
        estimator: str = "oas",
        metric: str = "riemann",
    ) -> None:
        self.estimator = estimator
        self.metric    = metric
        self._cov      = None
        self._ts       = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RiemannExtractor":
        """Fit covariance estimator and TangentSpace reference on training data.

        Parameters
        ----------
        X : np.ndarray
            Training epochs, shape ``(n_epochs, n_channels, n_samples)``.
        y : np.ndarray | None
            Ignored (included for sklearn compatibility).
        """
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except ImportError:
            raise ImportError(
                "pyriemann is required for RiemannExtractor. "
                "Install with: pip install pyriemann"
            )

        self._cov = Covariances(estimator=self.estimator)
        cov_mats  = self._cov.fit_transform(X)       # (N, C, C)
        self._ts  = TangentSpace(metric=self.metric)
        self._ts.fit(cov_mats)
        logger.info(
            "RiemannExtractor fitted: estimator=%s, metric=%s, cov_shape=%s",
            self.estimator, self.metric, cov_mats.shape,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project epochs to Riemannian tangent space features.

        Parameters
        ----------
        X : np.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            Tangent vectors, shape ``(n_epochs, n_channels*(n_channels+1)/2)``.
        """
        if self._cov is None or self._ts is None:
            raise RuntimeError("Call fit() before transform().")

        cov_mats = self._cov.transform(X)
        features = self._ts.transform(cov_mats)
        logger.debug("RiemannExtractor: output feature shape %s", features.shape)
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

EXTRACTORS: dict[str, type] = {
    "raw":     RawWindowExtractor,
    "xdawn":   XDawnExtractor,
    "riemann": RiemannExtractor,
}


def get_extractor(
    name: str,
    chan_names: list[str] | None = None,
    **kwargs,
) -> BaseEstimator:
    """Factory: return a feature extractor by name.

    Parameters
    ----------
    name : str
        One of ``'raw'``, ``'xdawn'``, ``'riemann'``.
    chan_names : list[str] | None
        Passed to extractors that need it (RawWindow, xDAWN).
    **kwargs
        Additional keyword arguments forwarded to the extractor constructor.

    Returns
    -------
    BaseEstimator
        Unfitted sklearn-compatible transformer.
    """
    name = name.lower()
    if name not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor '{name}'. Choose from: {list(EXTRACTORS.keys())}"
        )
    cls = EXTRACTORS[name]
    if chan_names is not None and "chan_names" in cls.__init__.__code__.co_varnames:
        kwargs["chan_names"] = chan_names
    return cls(**kwargs)
