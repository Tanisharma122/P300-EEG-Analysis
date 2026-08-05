"""
prep_engine.py — Phase 1: P300 EEG Data Parsing & Signal Preprocessing.

This module adapts the existing bci_pipeline preprocessing stack for the
P300 Speller simulation system.  It exposes a single public function
``load_and_preprocess`` that:

  1. Loads ``s01.mat`` via the HDF5 data loader.
  2. Applies a 4th-order Butterworth bandpass filter (0.5 – 12 Hz) to remove
     baseline drift and high-frequency noise.
  3. Extracts epochs in the window −100 ms → +800 ms relative to each flash
     trigger.
  4. Applies baseline correction (−100 ms → 0 ms pre-stimulus window).
  5. Selects key parieto-central channels (Cz, Pz, Oz, P3, P4 + flankers).
  6. Drops epochs with peak amplitude > 100 µV (artifact rejection).

Usage
-----
    from prep_engine import load_and_preprocess

    result = load_and_preprocess("data_explore/s01.mat")
    print(result.epochs.shape)        # (N, n_channels, T)
    print(result.labels[:5])          # [2 2 1 2 2] — 1=target, 2=non-target
    print(result.chan_names)          # ['Cz', 'Pz', 'Oz', ...]

Running standalone
------------------
    python prep_engine.py
    python prep_engine.py --mat path/to/s01.mat
"""

from __future__ import annotations

import sys
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Path bootstrap — make bci_pipeline importable regardless of cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_BCI  = _ROOT / "bci_pipeline"
for _p in (_ROOT, str(_BCI)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prep_engine")

# ---------------------------------------------------------------------------
# Default constants (aligned with Phase-1 spec)
# ---------------------------------------------------------------------------
DEFAULT_MAT        = _ROOT / "data_explore" / "s01.mat"

BANDPASS_LO: float = 0.5      # Hz   — per Phase-1 spec
BANDPASS_HI: float = 12.0     # Hz   — per Phase-1 spec
FILTER_ORDER: int  = 4        # 4th-order Butterworth

EPOCH_TMIN: float   = -0.100  # s  (−100 ms)
EPOCH_TMAX: float   =  0.800  # s  (+800 ms)
BASELINE_END: float =  0.000  # s  (0 ms = stimulus onset)

AMPLITUDE_THRESHOLD: float = 100.0  # µV

# P300-relevant channels to keep (others are discarded before feature extraction)
# These 8 electrodes account for the strongest P300 discriminability
# (Cz=16.2 µV, CP2=15.7, Pz=15.5 µV differential — from analysis report)
P300_CHANNELS: List[str] = [
    "Cz", "Pz", "Oz", "P3", "P4", "CP1", "CP2", "FC1", "FC2", "C3", "C4",
]

MARKER_TARGET    = 1
MARKER_NONTARGET = 2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """All outputs of the preprocessing pipeline.

    Attributes
    ----------
    epochs : np.ndarray
        Clean epoch tensor, shape ``(N, n_channels, T)`` in µV.
        Channels are the subset listed in ``chan_names``.
    labels : np.ndarray
        Labels, shape ``(N,)``.  1 = target, 2 = non-target.
    chan_names : list[str]
        Names of the retained channels (subset of full 32-ch cap).
    all_chan_names : list[str]
        Full 32-channel list from the recording.
    epochs_full : np.ndarray
        Epochs with **all** 32 channels retained (for models that need full
        spatial information, e.g. xDAWN / EEGNet).
        Shape ``(N, 32, T)``.
    fs : float
        Sampling frequency (Hz).
    event_samples : np.ndarray
        Sample indices of stimulus onsets in the original recording.
    event_labels : np.ndarray
        Labels of each event (1 or 2), aligned with ``event_samples``.
    n_target : int
        Number of retained target epochs.
    n_nontarget : int
        Number of retained non-target epochs.
    """

    epochs:         np.ndarray
    labels:         np.ndarray
    chan_names:     List[str]
    all_chan_names: List[str]
    epochs_full:    np.ndarray
    fs:             float
    event_samples:  np.ndarray
    event_labels:   np.ndarray
    n_target:       int = field(init=False)
    n_nontarget:    int = field(init=False)

    def __post_init__(self) -> None:
        self.n_target    = int(np.sum(self.labels == MARKER_TARGET))
        self.n_nontarget = int(np.sum(self.labels == MARKER_NONTARGET))

    def summary(self) -> str:
        T = self.epochs.shape[2]
        return (
            f"PreprocessResult | {len(self.labels)} epochs "
            f"({self.n_target} target, {self.n_nontarget} non-target) | "
            f"shape {self.epochs.shape} | fs={self.fs} Hz"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bandpass(data: np.ndarray, fs: float,
              lo: float = BANDPASS_LO, hi: float = BANDPASS_HI,
              order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Continuous EEG, shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency.
    lo, hi : float
        Bandpass cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray  — filtered data, same shape.
    """
    nyq = fs / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    if lo_n <= 0 or hi_n >= 1.0:
        raise ValueError(f"Bandpass [{lo}, {hi}] Hz outside valid range (0, {nyq}) Hz")
    sos = sps.butter(order, [lo_n, hi_n], btype="bandpass", output="sos")
    filtered = sps.sosfiltfilt(sos, data, axis=0)
    logger.info("Bandpass filter applied: %.1f–%.1f Hz (order=%d)", lo, hi, order)
    return filtered.astype(np.float64)


def _extract_epochs(
    data: np.ndarray, markers: np.ndarray, fs: float,
    tmin: float = EPOCH_TMIN, tmax: float = EPOCH_TMAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut EEG into fixed-length epochs around stimulus onsets.

    Parameters
    ----------
    data : np.ndarray  shape ``(n_samples, n_channels)``
    markers : np.ndarray  shape ``(n_samples,)``
    fs : float
    tmin, tmax : float  Epoch window in seconds.

    Returns
    -------
    epochs : np.ndarray  shape ``(N, n_channels, T)``
    labels : np.ndarray  shape ``(N,)``
    event_samples : np.ndarray  shape ``(N,)``  — onset indices
    event_labels_all : np.ndarray  shape ``(N,)``
    """
    pre  = int(round(-tmin * fs))
    post = int(round(tmax * fs))
    T    = pre + post

    event_mask   = (markers == MARKER_TARGET) | (markers == MARKER_NONTARGET)
    event_idx    = np.nonzero(event_mask)[0]
    event_labels = markers[event_idx].astype(np.int32)

    epochs_list, labels_list, sample_list = [], [], []
    n_total = data.shape[0]
    dropped = 0

    for idx, lbl in zip(event_idx, event_labels):
        s0, s1 = idx - pre, idx + post
        if s0 < 0 or s1 > n_total:
            dropped += 1
            continue
        epochs_list.append(data[s0:s1, :].T)   # → (C, T)
        labels_list.append(int(lbl))
        sample_list.append(int(idx))

    if dropped:
        logger.warning("Dropped %d epoch(s) at recording boundaries.", dropped)

    epochs = np.array(epochs_list, dtype=np.float64)     # (N, C, T)
    labels = np.array(labels_list, dtype=np.int32)        # (N,)
    ev_smp = np.array(sample_list, dtype=np.int64)        # (N,)

    logger.info(
        "Extracted %d epochs (%d target, %d non-target)",
        len(epochs), int(np.sum(labels == MARKER_TARGET)),
        int(np.sum(labels == MARKER_NONTARGET)),
    )
    return epochs, labels, ev_smp, event_labels


def _baseline_correct(
    epochs: np.ndarray, fs: float,
    tmin: float = EPOCH_TMIN, bl_end: float = BASELINE_END,
) -> np.ndarray:
    """Subtract pre-stimulus baseline mean from every epoch and channel.

    Parameters
    ----------
    epochs : np.ndarray  shape ``(N, C, T)``
    fs : float
    tmin : float  Start of epoch (s).
    bl_end : float  End of baseline window (s).

    Returns
    -------
    np.ndarray  Baseline-corrected epochs, same shape.
    """
    bl_samples = int(round((bl_end - tmin) * fs))  # number of baseline samples
    baseline_mean = epochs[:, :, :bl_samples].mean(axis=2, keepdims=True)
    corrected = epochs - baseline_mean
    logger.debug("Baseline correction: %d pre-stim samples (%.0f–%.0f ms)",
                 bl_samples, tmin * 1000, bl_end * 1000)
    return corrected.astype(np.float64)


def _select_channels(
    epochs: np.ndarray,
    all_chan_names: List[str],
    target_channels: List[str] = P300_CHANNELS,
) -> tuple[np.ndarray, List[str]]:
    """Keep only the specified P300-relevant channels.

    Parameters
    ----------
    epochs : np.ndarray  shape ``(N, C, T)``
    all_chan_names : list[str]
    target_channels : list[str]

    Returns
    -------
    epochs_sel : np.ndarray  shape ``(N, n_sel, T)``
    sel_names  : list[str]
    """
    name_idx = {n: i for i, n in enumerate(all_chan_names)}
    indices, sel_names = [], []
    for ch in target_channels:
        if ch in name_idx:
            indices.append(name_idx[ch])
            sel_names.append(ch)
        else:
            logger.warning("Channel '%s' not found — skipping.", ch)

    if not indices:
        logger.warning("No target channels found; returning all channels.")
        return epochs, list(all_chan_names)

    logger.info("Channel selection: %d → %d channels: %s", len(all_chan_names),
                len(indices), sel_names)
    return epochs[:, indices, :], sel_names


def _reject_artifacts(
    epochs: np.ndarray, labels: np.ndarray,
    ev_smp: np.ndarray, ev_lbl: np.ndarray,
    threshold: float = AMPLITUDE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop epochs where any channel exceeds the amplitude threshold.

    Parameters
    ----------
    epochs : np.ndarray  shape ``(N, C, T)``
    labels : np.ndarray  shape ``(N,)``
    ev_smp : np.ndarray  shape ``(N,)``  event sample indices
    ev_lbl : np.ndarray  shape ``(N,)``  raw event labels (pre-drop)
    threshold : float  µV

    Returns
    -------
    Filtered epochs, labels, ev_smp, ev_lbl.
    """
    peak = np.max(np.abs(epochs), axis=(1, 2))   # (N,)
    keep = peak <= threshold
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        logger.warning("Artifact rejection: dropped %d/%d epochs (>%.0f µV)",
                       n_dropped, len(epochs), threshold)
    else:
        logger.info("Artifact rejection: all %d epochs retained.", len(epochs))
    return epochs[keep], labels[keep], ev_smp[keep], ev_lbl[keep]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess(
    mat_path: str | Path = DEFAULT_MAT,
    *,
    bandpass_lo: float = BANDPASS_LO,
    bandpass_hi: float = BANDPASS_HI,
    epoch_tmin: float  = EPOCH_TMIN,
    epoch_tmax: float  = EPOCH_TMAX,
    p300_channels: List[str] | None = None,
    amplitude_threshold: float = AMPLITUDE_THRESHOLD,
    verbose: bool = True,
) -> PreprocessResult:
    """Execute the complete Phase-1 preprocessing pipeline.

    Steps
    -----
    1. Load raw EEG from ``mat_path`` (HDF5/MATLAB v7.3 format).
    2. Bandpass filter (default 0.5–12 Hz, 4th-order Butterworth, zero-phase).
    3. Extract epochs (default −100 ms → +800 ms per stimulus onset).
    4. Baseline correct (−100 ms → 0 ms window).
    5. Channel selection (keep P300 parieto-central electrodes).
    6. Amplitude-threshold artifact rejection (>100 µV dropped).

    Parameters
    ----------
    mat_path : str | Path
        Path to the ``.mat`` file.
    bandpass_lo, bandpass_hi : float
        Bandpass filter cutoffs in Hz.
    epoch_tmin, epoch_tmax : float
        Epoch window in seconds (relative to stimulus onset).
    p300_channels : list[str] | None
        Channels to retain. Defaults to the 11 P300-relevant electrodes.
    amplitude_threshold : float
        Peak amplitude rejection threshold (µV).
    verbose : bool
        Print step-by-step progress.

    Returns
    -------
    PreprocessResult
        Dataclass holding ``epochs``, ``labels``, ``chan_names``,
        ``epochs_full``, ``fs``, ``event_samples``, ``event_labels``.
    """
    from src.data_loader import load_recording   # uses bci_pipeline loader

    mat_path = Path(mat_path).resolve()
    p300_ch  = p300_channels if p300_channels is not None else P300_CHANNELS

    # ── Step 1: Load ──────────────────────────────────────────────────────
    if verbose:
        print("\n[prep_engine] ── Step 1/6  Loading EEG data ─────────────────")
    rec = load_recording(mat_path, verbose=verbose)
    if verbose:
        print(f"  Raw data shape : {rec.data.shape}  (samples × channels)")
        print(f"  Sampling rate  : {rec.fs} Hz")
        print(f"  Channels       : {len(rec.chan_names)}")

    # ── Step 2: Bandpass filter ────────────────────────────────────────────
    if verbose:
        print(f"\n[prep_engine] ── Step 2/6  Bandpass filter "
              f"({bandpass_lo}–{bandpass_hi} Hz) ───")
    data_filt = _bandpass(rec.data, rec.fs, lo=bandpass_lo, hi=bandpass_hi)

    # ── Step 3: Epoch extraction ───────────────────────────────────────────
    if verbose:
        print(f"\n[prep_engine] ── Step 3/6  Epoch extraction "
              f"({epoch_tmin*1000:.0f} ms → +{epoch_tmax*1000:.0f} ms) ───")
    epochs_full, labels, ev_smp, ev_lbl = _extract_epochs(
        data_filt, rec.markers, rec.fs,
        tmin=epoch_tmin, tmax=epoch_tmax,
    )
    if verbose:
        n_tgt = int(np.sum(labels == MARKER_TARGET))
        n_nt  = int(np.sum(labels == MARKER_NONTARGET))
        print(f"  Epochs         : {len(labels)}  ({n_tgt} target, {n_nt} non-target)")
        print(f"  Epoch shape    : {epochs_full.shape}  (N × C × T)")

    # ── Step 4: Baseline correction ───────────────────────────────────────
    if verbose:
        print("\n[prep_engine] ── Step 4/6  Baseline correction "
              f"({epoch_tmin*1000:.0f} ms → 0 ms) ───")
    epochs_full = _baseline_correct(epochs_full, rec.fs,
                                    tmin=epoch_tmin, bl_end=BASELINE_END)

    # ── Step 5: Channel selection ──────────────────────────────────────────
    if verbose:
        print(f"\n[prep_engine] ── Step 5/6  Channel selection "
              f"({len(p300_ch)} channels) ───")
    epochs_sel, sel_names = _select_channels(epochs_full, rec.chan_names, p300_ch)
    if verbose:
        print(f"  Selected       : {sel_names}")

    # ── Step 6: Artifact rejection ─────────────────────────────────────────
    if verbose:
        print(f"\n[prep_engine] ── Step 6/6  Artifact rejection "
              f"(threshold={amplitude_threshold} µV) ───")
    epochs_sel, labels, ev_smp, ev_lbl = _reject_artifacts(
        epochs_sel, labels, ev_smp, ev_lbl, threshold=amplitude_threshold,
    )
    # Align full-channel epochs to same kept indices (rebuild from ev_smp)
    # Rebuild full-channel epochs for kept indices
    pre  = int(round(-epoch_tmin * rec.fs))
    post = int(round(epoch_tmax  * rec.fs))
    kept_full = np.stack([
        data_filt[s - pre : s + post, :].T
        for s in ev_smp
    ])  # (N_kept, 32, T)
    kept_full = _baseline_correct(kept_full, rec.fs,
                                  tmin=epoch_tmin, bl_end=BASELINE_END)

    if verbose:
        n_tgt = int(np.sum(labels == MARKER_TARGET))
        n_nt  = int(np.sum(labels == MARKER_NONTARGET))
        T = epochs_sel.shape[2]
        dur_ms = T / rec.fs * 1000
        print(f"\n[prep_engine] ✅  Pipeline complete:")
        print(f"  Final epoch tensor : {epochs_sel.shape}  (N × channels × samples)")
        print(f"  Full-ch tensor     : {kept_full.shape}")
        print(f"  Epoch duration     : {dur_ms:.0f} ms  ({T} samples @ {rec.fs} Hz)")
        print(f"  Target epochs      : {n_tgt}")
        print(f"  Non-target epochs  : {n_nt}")
        print(f"  Imbalance ratio    : 1:{int(n_nt/max(1, n_tgt))}")
        print(f"  Max amplitude      : {float(np.max(np.abs(epochs_sel))):.2f} µV")
        print()

    return PreprocessResult(
        epochs         = epochs_sel,
        labels         = labels,
        chan_names     = sel_names,
        all_chan_names = rec.chan_names,
        epochs_full    = kept_full,
        fs             = rec.fs,
        event_samples  = ev_smp,
        event_labels   = ev_lbl,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P300 Phase-1: Load & preprocess EEG data from s01.mat",
    )
    parser.add_argument(
        "--mat", default=None, metavar="PATH",
        help=f"Path to s01.mat (default: {DEFAULT_MAT})",
    )
    parser.add_argument(
        "--lo", type=float, default=BANDPASS_LO,
        help=f"Bandpass low cutoff Hz (default {BANDPASS_LO})",
    )
    parser.add_argument(
        "--hi", type=float, default=BANDPASS_HI,
        help=f"Bandpass high cutoff Hz (default {BANDPASS_HI})",
    )
    args = parser.parse_args()

    mat = Path(args.mat) if args.mat else DEFAULT_MAT
    result = load_and_preprocess(mat, bandpass_lo=args.lo, bandpass_hi=args.hi, verbose=True)
    print(result.summary())
