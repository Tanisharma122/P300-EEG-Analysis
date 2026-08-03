"""
config.py — Central configuration constants for the P300 BCI pipeline.

All hyperparameters, paths, and domain constants live here so that every
module imports from a single source of truth.
"""

from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent          # bci_pipeline/
REPO_ROOT    = PROJECT_ROOT.parent                      # P300 Detection/
DATA_DIR     = REPO_ROOT / "data_explore"
MAT_FILE     = DATA_DIR / "s01.mat"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# EEG / Dataset constants
# ---------------------------------------------------------------------------
FS: float = 512.0                  # Sampling frequency (Hz)
N_CHANNELS: int = 32
N_SAMPLES_EPOCH: int = 512         # 1000 ms × 512 Hz

# HDF5 key paths inside s01.mat
HDF5_DATA_KEY    = "/RSVP/data"
HDF5_MARKERS_KEY = "/RSVP/markers_target"
HDF5_SRATE_KEY   = "/RSVP/srate"
HDF5_LABELS_KEY  = "/RSVP/chanlocs/labels"

# Marker values
MARKER_TARGET     = 1
MARKER_NONTARGET  = 2

# Channel names (32 electrodes, 10-20 system)
CHANNEL_NAMES: list[str] = [
    "FP1", "AF3", "F7",  "F3",  "FC1", "FC5", "T7",  "C3",
    "CP1", "CP5", "P7",  "P3",  "Pz",  "PO3", "O1",  "Oz",
    "O2",  "PO4", "P4",  "P8",  "CP6", "CP2", "C4",  "T8",
    "FC6", "FC2", "F4",  "F8",  "AF4", "FP2", "FZ",  "Cz",
]

# Channels with known eye-blink artifacts
ARTIFACT_CHANNELS: list[str] = ["FP1", "AF3"]

# P300 key electrodes (best discriminability)
P300_CHANNELS: list[str] = ["Cz", "Pz", "CP1", "CP2", "FC1", "FC2", "C3", "C4"]

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
BANDPASS_LO: float  = 0.1          # Hz
BANDPASS_HI: float  = 30.0         # Hz
FILTER_ORDER: int   = 4            # Butterworth order

EPOCH_TMIN: float   = -0.2         # seconds (pre-stimulus)
EPOCH_TMAX: float   =  0.8         # seconds (post-stimulus)
BASELINE_TMAX: float = 0.0         # baseline correction end (0 ms = stimulus onset)
AMPLITUDE_THRESHOLD: float = 100.0 # µV — epochs exceeding this are dropped

# ---------------------------------------------------------------------------
# Feature extraction parameters
# ---------------------------------------------------------------------------
P300_WINDOW_START: float = 0.25    # seconds
P300_WINDOW_END:   float = 0.50    # seconds
DOWNSAMPLE_FREQ:   float = 20.0    # Hz (for RawWindow extractor)
N_XDAWN_COMPONENTS: int  = 4       # xDAWN spatial filter components

# ---------------------------------------------------------------------------
# Model / training parameters
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5
RANDOM_STATE: int = 42

# EEGNet architecture
EEGNET_TEMPORAL_FILTERS: int = 8
EEGNET_DEPTH_MULTIPLIER:  int = 2
EEGNET_DROPOUT_RATE: float   = 0.5
EEGNET_EPOCHS: int           = 50
EEGNET_BATCH_SIZE: int       = 32
EEGNET_LR: float             = 1e-3
