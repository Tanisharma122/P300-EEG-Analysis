"""
src/data_loader.py — HDF5 data ingestion for P300 RSVP EEG datasets.

Provides safe, typed loading of MATLAB v7.3 (.mat) files stored as HDF5,
specifically targeting the /RSVP group structure produced by the dataset.

Usage
-----
    from src.data_loader import load_recording
    rec = load_recording("data_explore/s01.mat")
    print(rec.data.shape)       # (166770, 32)
    print(rec.fs)               # 512.0
    print(rec.chan_names[:4])   # ['FP1', 'AF3', 'F7', 'F3']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Import config with a fallback so the module works even when run standalone
try:
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        HDF5_DATA_KEY, HDF5_MARKERS_KEY, HDF5_SRATE_KEY, HDF5_LABELS_KEY,
        CHANNEL_NAMES, FS,
    )
except ImportError:
    HDF5_DATA_KEY    = "/RSVP/data"
    HDF5_MARKERS_KEY = "/RSVP/markers_target"
    HDF5_SRATE_KEY   = "/RSVP/srate"
    HDF5_LABELS_KEY  = "/RSVP/chanlocs/labels"
    CHANNEL_NAMES    = [
        "FP1", "AF3", "F7",  "F3",  "FC1", "FC5", "T7",  "C3",
        "CP1", "CP5", "P7",  "P3",  "Pz",  "PO3", "O1",  "Oz",
        "O2",  "PO4", "P4",  "P8",  "CP6", "CP2", "C4",  "T8",
        "FC6", "FC2", "F4",  "F8",  "AF4", "FP2", "FZ",  "Cz",
    ]
    FS = 512.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass
class EEGRecording:
    """Container for a raw continuous EEG recording.

    Attributes
    ----------
    data : np.ndarray
        Continuous EEG, shape ``(n_samples, n_channels)``, in µV.
    markers : np.ndarray
        Per-sample marker array, shape ``(n_samples,)``.
        Values: 0 = no event, 1 = target, 2 = non-target.
    fs : float
        Sampling frequency in Hz.
    chan_names : list[str]
        Ordered list of electrode names (length == n_channels).
    source_path : str
        Absolute path to the source .mat file.
    """

    data: np.ndarray
    markers: np.ndarray
    fs: float
    chan_names: list[str]
    source_path: str = ""

    # Derived / cached
    _chan_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._chan_index = {name: i for i, name in enumerate(self.chan_names)}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def channel_index(self, name: str) -> int:
        """Return the column index for a named electrode."""
        try:
            return self._chan_index[name]
        except KeyError:
            raise ValueError(f"Channel '{name}' not found. Available: {self.chan_names}")

    def summary(self) -> str:
        target_count = int(np.sum(self.markers == 1))
        nontarget_count = int(np.sum(self.markers == 2))
        return (
            f"EEGRecording | {self.n_samples} samples × {self.n_channels} channels "
            f"| {self.fs} Hz | {self.duration_s:.1f} s | "
            f"Targets: {target_count} | Non-targets: {nontarget_count}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _decode_channel_labels(f: h5py.File) -> list[str]:
    """Decode MATLAB cellstr channel labels stored as HDF5 object references.

    MATLAB v7.3 stores cell arrays of strings as groups of object references.
    Each reference points to a uint16 array of character codes.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    list[str]
        Decoded channel name strings.
    """
    try:
        label_refs = f[HDF5_LABELS_KEY]
        names: list[str] = []
        for i in range(label_refs.shape[0]):
            ref = label_refs[i, 0]
            char_codes = f[ref][()].flatten()
            names.append("".join(chr(int(c)) for c in char_codes))
        return names
    except Exception as exc:
        logger.warning(
            "Could not decode channel labels from HDF5 (%s). "
            "Falling back to config defaults.",
            exc,
        )
        return list(CHANNEL_NAMES)


def _read_scalar_dataset(f: h5py.File, key: str) -> float:
    """Read a scalar HDF5 dataset (stored as 1-element array) as a float."""
    val = f[key][()].flatten()
    return float(val[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_recording(
    path: str | Path,
    *,
    verbose: bool = True,
) -> EEGRecording:
    """Load a MATLAB v7.3 RSVP EEG file and return an :class:`EEGRecording`.

    Parameters
    ----------
    path : str | Path
        Path to the ``.mat`` file (HDF5 format).
    verbose : bool
        If True, log a one-line summary after loading.

    Returns
    -------
    EEGRecording
        Populated data container.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required HDF5 datasets are missing from the file.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading EEG data from: %s", path)

    with h5py.File(str(path), "r") as f:
        # --- Verify required keys ---
        for key in (HDF5_DATA_KEY, HDF5_MARKERS_KEY, HDF5_SRATE_KEY):
            if key not in f:
                raise KeyError(
                    f"Required HDF5 dataset '{key}' not found in {path}. "
                    f"Available root groups: {list(f.keys())}"
                )

        # --- Load data (samples × channels) ---
        data: np.ndarray = f[HDF5_DATA_KEY][()].astype(np.float64)
        logger.debug("Raw data shape: %s", data.shape)

        # Ensure (samples, channels) orientation
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            logger.debug("Transposing data to (samples, channels)")
            data = data.T

        # --- Load markers ---
        markers: np.ndarray = f[HDF5_MARKERS_KEY][()].flatten().astype(np.int32)

        # --- Sampling rate ---
        fs = _read_scalar_dataset(f, HDF5_SRATE_KEY)

        # --- Channel names ---
        chan_names = _decode_channel_labels(f)

    # Sanity checks
    if data.shape[0] != markers.shape[0]:
        raise ValueError(
            f"Data length ({data.shape[0]}) and marker length ({markers.shape[0]}) mismatch."
        )

    rec = EEGRecording(
        data=data,
        markers=markers,
        fs=fs,
        chan_names=chan_names,
        source_path=str(path),
    )

    if verbose:
        logger.info(rec.summary())
        print(f"[data_loader] {rec.summary()}")

    return rec


def get_event_samples(
    markers: np.ndarray,
    target_code: int = 1,
    nontarget_code: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sample indices and labels for all stimulus events.

    Parameters
    ----------
    markers : np.ndarray
        Per-sample marker array.
    target_code : int
        Marker value for target stimuli.
    nontarget_code : int
        Marker value for non-target stimuli.

    Returns
    -------
    event_samples : np.ndarray
        Sample indices where a stimulus occurred (shape ``(n_events,)``).
    event_labels : np.ndarray
        Corresponding labels: 1 = target, 2 = non-target (shape ``(n_events,)``).
    """
    event_mask = (markers == target_code) | (markers == nontarget_code)
    event_samples = np.nonzero(event_mask)[0]
    event_labels = markers[event_samples].astype(np.int32)
    return event_samples, event_labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Load and inspect an RSVP EEG .mat file")
    parser.add_argument("path", nargs="?", default=None, help="Path to s01.mat")
    args = parser.parse_args()

    # Default to config path if none given
    if args.path is None:
        try:
            from config import MAT_FILE
            mat_path = MAT_FILE
        except ImportError:
            mat_path = Path("../data_explore/s01.mat")
    else:
        mat_path = Path(args.path)

    rec = load_recording(mat_path)
    ev_samples, ev_labels = get_event_samples(rec.markers)

    print(f"\n{'='*60}")
    print(f"  File         : {rec.source_path}")
    print(f"  Data shape   : {rec.data.shape}  (samples × channels)")
    print(f"  Sampling rate: {rec.fs} Hz")
    print(f"  Duration     : {rec.duration_s:.2f} s")
    print(f"  Channels     : {rec.chan_names}")
    print(f"  Events total : {len(ev_samples)}")
    print(f"  Targets      : {int(np.sum(ev_labels == 1))}")
    print(f"  Non-targets  : {int(np.sum(ev_labels == 2))}")
    print(f"{'='*60}\n")
