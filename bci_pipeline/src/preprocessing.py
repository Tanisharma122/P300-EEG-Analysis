"""
src/preprocessing.py — P300 EEG Signal Preprocessing Pipeline.

Implements a complete, leakage-free preprocessing pipeline:
  1. Zero-phase 4th-order Butterworth bandpass (0.1–30 Hz)
  2. Optional 50/60 Hz notch filter
  3. Epoch extraction (−200 ms → +800 ms per stimulus onset)
  4. Baseline correction (subtract mean of −200 ms → 0 ms window)
  5. Amplitude-threshold artifact rejection (>100 µV dropped)

All functions operate on NumPy arrays and are stateless (no sklearn fit/transform),
so they are safe to apply inside or outside cross-validation loops.

Usage
-----
    from src.data_loader import load_recording
    from src.preprocessing import run_preprocessing_pipeline

    epochs, labels, chan_names, fs = run_preprocessing_pipeline("data_explore/s01.mat")
    print(epochs.shape)   # (N_valid, 32, 512)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Config import with fallback
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        FS, BANDPASS_LO, BANDPASS_HI, FILTER_ORDER,
        EPOCH_TMIN, EPOCH_TMAX, BASELINE_TMAX,
        AMPLITUDE_THRESHOLD, MARKER_TARGET, MARKER_NONTARGET,
        N_SAMPLES_EPOCH,
    )
except ImportError:
    FS = 512.0
    BANDPASS_LO = 0.1
    BANDPASS_HI = 30.0
    FILTER_ORDER = 4
    EPOCH_TMIN = -0.2
    EPOCH_TMAX = 0.8
    BASELINE_TMAX = 0.0
    AMPLITUDE_THRESHOLD = 100.0
    MARKER_TARGET = 1
    MARKER_NONTARGET = 2
    N_SAMPLES_EPOCH = 512

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Filtering
# ---------------------------------------------------------------------------

def bandpass_filter(
    data: np.ndarray,
    fs: float = FS,
    lo: float = BANDPASS_LO,
    hi: float = BANDPASS_HI,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase 4th-order Butterworth bandpass filter.

    Uses second-order sections (SOS) + ``sosfiltfilt`` for numerical stability
    and exactly zero phase shift (no group-delay distortion).

    Parameters
    ----------
    data : np.ndarray
        Continuous EEG, shape ``(n_samples, n_channels)``, µV.
    fs : float
        Sampling frequency in Hz.
    lo : float
        Lower cutoff frequency in Hz.
    hi : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order (applied as ``order // 2`` in forward + backward pass,
        giving effective order ``order`` after ``sosfiltfilt``).

    Returns
    -------
    np.ndarray
        Filtered data, same shape as input.
    """
    nyq = fs / 2.0
    if lo <= 0 or hi >= nyq:
        raise ValueError(
            f"Bandpass [{lo}, {hi}] Hz is outside valid range (0, {nyq}) Hz."
        )

    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    filtered = sps.sosfiltfilt(sos, data, axis=0)
    logger.debug("Bandpass filter applied: %.2f–%.2f Hz, order=%d", lo, hi, order)
    return filtered.astype(np.float64)


def notch_filter(
    data: np.ndarray,
    fs: float = FS,
    freqs: list[float] | None = None,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase notch filter(s) to remove line-frequency noise.

    Parameters
    ----------
    data : np.ndarray
        EEG data, shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.
    freqs : list[float] | None
        Notch frequencies in Hz. Defaults to [50, 60].
    quality_factor : float
        Q factor of the IIR notch filter (higher = narrower notch).

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    if freqs is None:
        freqs = [50.0, 60.0]

    result = data.copy()
    for f0 in freqs:
        if f0 >= fs / 2:
            logger.warning("Notch frequency %.1f Hz >= Nyquist (%.1f Hz), skipping.", f0, fs / 2)
            continue
        b, a = sps.iirnotch(f0 / (fs / 2), Q=quality_factor)
        result = sps.filtfilt(b, a, result, axis=0)
        logger.debug("Notch filter applied at %.1f Hz", f0)

    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Epoch extraction
# ---------------------------------------------------------------------------

def extract_epochs(
    data: np.ndarray,
    markers: np.ndarray,
    fs: float = FS,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    target_code: int = MARKER_TARGET,
    nontarget_code: int = MARKER_NONTARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut continuous EEG into fixed-length epochs around stimulus onsets.

    Parameters
    ----------
    data : np.ndarray
        Filtered continuous EEG, shape ``(n_samples, n_channels)``.
    markers : np.ndarray
        Per-sample event codes, shape ``(n_samples,)``.
    fs : float
        Sampling frequency in Hz.
    tmin : float
        Start of epoch relative to stimulus onset (seconds, negative = pre-stim).
    tmax : float
        End of epoch relative to stimulus onset (seconds).
    target_code : int
        Marker value for target events.
    nontarget_code : int
        Marker value for non-target events.

    Returns
    -------
    epochs : np.ndarray
        Epoch array, shape ``(n_epochs, n_channels, n_epoch_samples)``.
    labels : np.ndarray
        Class labels, shape ``(n_epochs,)``. Values: 1=target, 2=non-target.
    """
    pre_samples  = int(round(-tmin * fs))   # samples before stimulus
    post_samples = int(round(tmax * fs))    # samples after stimulus
    epoch_len    = pre_samples + post_samples

    # Find all stimulus onsets
    event_mask   = (markers == target_code) | (markers == nontarget_code)
    event_idx    = np.nonzero(event_mask)[0]
    event_labels = markers[event_idx].astype(np.int32)

    n_total = data.shape[0]
    epochs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    dropped = 0

    for idx, lbl in zip(event_idx, event_labels):
        s0 = idx - pre_samples
        s1 = idx + post_samples
        if s0 < 0 or s1 > n_total:
            dropped += 1
            continue
        seg = data[s0:s1, :]       # (epoch_len, n_channels)
        epochs_list.append(seg.T)  # → (n_channels, epoch_len)
        labels_list.append(int(lbl))

    if dropped:
        logger.warning("Dropped %d epoch(s) at recording boundaries.", dropped)

    epochs = np.array(epochs_list, dtype=np.float64)   # (N, C, T)
    labels = np.array(labels_list, dtype=np.int32)      # (N,)

    logger.info(
        "Extracted %d epochs (%d target, %d non-target), dropped %d",
        len(epochs),
        int(np.sum(labels == target_code)),
        int(np.sum(labels == nontarget_code)),
        dropped,
    )
    return epochs, labels


# ---------------------------------------------------------------------------
# 3. Baseline correction
# ---------------------------------------------------------------------------

def baseline_correct(
    epochs: np.ndarray,
    fs: float = FS,
    tmin: float = EPOCH_TMIN,
    bl_tmax: float = BASELINE_TMAX,
) -> np.ndarray:
    """Subtract the pre-stimulus baseline mean from each epoch and channel.

    The baseline window is [tmin, bl_tmax] (relative to stimulus onset).
    The mean of each channel over this window is subtracted from the entire epoch.

    Parameters
    ----------
    epochs : np.ndarray
        Epochs array, shape ``(n_epochs, n_channels, n_samples)``.
    fs : float
        Sampling frequency in Hz.
    tmin : float
        Start of epoch (seconds, typically −0.2).
    bl_tmax : float
        End of baseline window (seconds, typically 0.0 = stimulus onset).

    Returns
    -------
    np.ndarray
        Baseline-corrected epochs, same shape.
    """
    pre_samples = int(round(-tmin * fs))     # total pre-stimulus samples
    bl_end_idx  = int(round((bl_tmax - tmin) * fs))  # baseline end sample

    baseline_mean = epochs[:, :, :bl_end_idx].mean(axis=2, keepdims=True)
    corrected = epochs - baseline_mean
    logger.debug(
        "Baseline correction: window [%d, %d] samples (%.0f–%.0f ms)",
        0, bl_end_idx, tmin * 1000, bl_tmax * 1000,
    )
    return corrected.astype(np.float64)


# ---------------------------------------------------------------------------
# 4. Artifact rejection
# ---------------------------------------------------------------------------

def amplitude_threshold_rejection(
    epochs: np.ndarray,
    labels: np.ndarray,
    threshold: float = AMPLITUDE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop epochs where any channel exceeds an amplitude threshold.

    Parameters
    ----------
    epochs : np.ndarray
        Shape ``(n_epochs, n_channels, n_samples)``.
    labels : np.ndarray
        Shape ``(n_epochs,)``.
    threshold : float
        Peak-to-peak amplitude threshold in µV.

    Returns
    -------
    epochs_clean : np.ndarray
        Retained epochs.
    labels_clean : np.ndarray
        Retained labels.
    kept_mask : np.ndarray
        Boolean mask of shape ``(n_epochs,)`` indicating kept epochs.
    """
    # Peak absolute amplitude per epoch across all channels and samples
    peak_amp = np.max(np.abs(epochs), axis=(1, 2))  # (n_epochs,)
    kept_mask = peak_amp <= threshold

    n_dropped = int(np.sum(~kept_mask))
    if n_dropped:
        logger.warning(
            "Amplitude rejection (threshold=%.1f µV): dropped %d / %d epochs.",
            threshold, n_dropped, len(epochs),
        )
    else:
        logger.info(
            "Amplitude rejection (threshold=%.1f µV): all %d epochs retained.",
            threshold, len(epochs),
        )

    return epochs[kept_mask], labels[kept_mask], kept_mask


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    path: str | Path,
    *,
    apply_notch: bool = False,
    notch_freqs: list[float] | None = None,
    amplitude_threshold: float = AMPLITUDE_THRESHOLD,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Execute the complete preprocessing pipeline on a .mat file.

    Steps:
      1. Load raw EEG via :func:`src.data_loader.load_recording`
      2. Bandpass filter (0.1–30 Hz, zero-phase Butterworth)
      3. Optional notch filter (50/60 Hz)
      4. Extract epochs (−200 ms → +800 ms)
      5. Baseline correct (−200 ms → 0 ms)
      6. Amplitude-threshold artifact rejection (>100 µV)

    Parameters
    ----------
    path : str | Path
        Path to the ``.mat`` file.
    apply_notch : bool
        Whether to apply the notch filter (default False — no line noise found).
    notch_freqs : list[float] | None
        Frequencies for notch filter. Defaults to [50, 60].
    amplitude_threshold : float
        Rejection threshold in µV.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    epochs : np.ndarray
        Clean epochs, shape ``(n_clean, 32, 512)``.
    labels : np.ndarray
        Labels, shape ``(n_clean,)``. 1=target, 2=non-target.
    chan_names : list[str]
        Ordered channel names.
    fs : float
        Sampling frequency (512.0).
    """
    # Lazy import to avoid circular dependency issues
    from src.data_loader import load_recording

    # Step 1 — Load
    if verbose:
        print("\n[preprocessing] Step 1/5 — Loading data...")
    rec = load_recording(path, verbose=verbose)

    # Step 2 — Bandpass
    if verbose:
        print("[preprocessing] Step 2/5 — Bandpass filter (0.1–30 Hz)...")
    data_filt = bandpass_filter(rec.data, fs=rec.fs)

    # Step 3 — Optional notch
    if apply_notch:
        if verbose:
            print(f"[preprocessing] Step 3/5 — Notch filter at {notch_freqs or [50, 60]} Hz...")
        data_filt = notch_filter(data_filt, fs=rec.fs, freqs=notch_freqs)
    else:
        if verbose:
            print("[preprocessing] Step 3/5 — Notch filter: SKIPPED (no line noise detected)")

    # Step 4 — Epoch extraction
    if verbose:
        print("[preprocessing] Step 4/5 — Extracting epochs (−200 ms → +800 ms)...")
    epochs, labels = extract_epochs(data_filt, rec.markers, fs=rec.fs)

    # Step 5 — Baseline correction
    if verbose:
        print("[preprocessing] Step 5/5 — Baseline correction (−200 ms → 0 ms)...")
    epochs = baseline_correct(epochs, fs=rec.fs)

    # Step 6 — Artifact rejection
    if verbose:
        print(f"[preprocessing] Artifact rejection (threshold={amplitude_threshold} µV)...")
    epochs, labels, _ = amplitude_threshold_rejection(epochs, labels, amplitude_threshold)

    if verbose:
        n_target     = int(np.sum(labels == MARKER_TARGET))
        n_nontarget  = int(np.sum(labels == MARKER_NONTARGET))
        print(f"\n[preprocessing] ✅ Pipeline complete:")
        print(f"   Epoch tensor shape : {epochs.shape}  (epochs × channels × samples)")
        print(f"   Target epochs      : {n_target}")
        print(f"   Non-target epochs  : {n_nontarget}")
        print(f"   Channels           : {rec.chan_names}")
        print(f"   Sampling rate      : {rec.fs} Hz\n")

    return epochs, labels, rec.chan_names, rec.fs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run P300 preprocessing pipeline")
    parser.add_argument("path", nargs="?", default=None, help="Path to s01.mat")
    parser.add_argument("--notch", action="store_true", help="Apply 50/60 Hz notch filter")
    args = parser.parse_args()

    if args.path is None:
        try:
            from config import MAT_FILE
            mat_path = MAT_FILE
        except ImportError:
            mat_path = Path("../data_explore/s01.mat")
    else:
        mat_path = Path(args.path)

    epochs, labels, chan_names, fs = run_preprocessing_pipeline(
        mat_path, apply_notch=args.notch
    )
    print(f"Final epochs shape: {epochs.shape}")
    print(f"Max amplitude (µV): {np.max(np.abs(epochs)):.2f}")
