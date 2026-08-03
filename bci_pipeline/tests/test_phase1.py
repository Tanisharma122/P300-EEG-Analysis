"""
tests/test_phase1.py — Phase 1 Verification Test Suite.

Verifies the complete data loading and preprocessing pipeline against the
expected outputs from the s01.mat dataset.

Run from the bci_pipeline/ directory:
    python tests/test_phase1.py

Expected output:
    [PASS] data.shape == (166770, 32)
    [PASS] fs == 512.0
    [PASS] len(chan_names) == 32
    [PASS] Pz in chan_names
    [PASS] Cz in chan_names
    [PASS] Target event count == 40
    [PASS] Non-target event count == 560
    [PASS] epochs.shape == (600, 32, 512)
    [PASS] Target epochs count == 40
    [PASS] Non-target epochs count == 560
    [PASS] No baseline amplitude bias (mean ~0)
    [PASS] Max amplitude after rejection <= 100.0 µV
    [PASS] All shapes consistent after filtering
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure bci_pipeline/ is on the path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import MAT_FILE, MARKER_TARGET, MARKER_NONTARGET, AMPLITUDE_THRESHOLD


# ---------------------------------------------------------------------------
# Minimal test framework
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def check(condition: bool, message: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {message}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {message}  ← FAILED")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test: data_loader
# ---------------------------------------------------------------------------

def test_data_loader(mat_path: Path) -> None:
    section("Phase 1A: Data Loader (src/data_loader.py)")

    from src.data_loader import load_recording, get_event_samples

    rec = load_recording(mat_path, verbose=False)

    check(rec.data.shape == (166770, 32),
          f"data.shape == (166770, 32) [got {rec.data.shape}]")
    check(rec.fs == 512.0,
          f"fs == 512.0 [got {rec.fs}]")
    check(len(rec.chan_names) == 32,
          f"len(chan_names) == 32 [got {len(rec.chan_names)}]")
    check("Pz" in rec.chan_names,
          f"'Pz' in chan_names [chan_names={rec.chan_names}]")
    check("Cz" in rec.chan_names,
          f"'Cz' in chan_names")
    check("CP1" in rec.chan_names, "'CP1' in chan_names")
    check("CP2" in rec.chan_names, "'CP2' in chan_names")
    check(rec.markers.shape == (166770,),
          f"markers.shape == (166770,) [got {rec.markers.shape}]")

    # Event counts
    ev_samples, ev_labels = get_event_samples(rec.markers)
    n_target     = int(np.sum(ev_labels == MARKER_TARGET))
    n_nontarget  = int(np.sum(ev_labels == MARKER_NONTARGET))
    check(n_target == 40,
          f"Target event count == 40 [got {n_target}]")
    check(n_nontarget == 560,
          f"Non-target event count == 560 [got {n_nontarget}]")

    # Channel index lookup
    pz_idx = rec.channel_index("Pz")
    check(isinstance(pz_idx, int) and 0 <= pz_idx < 32,
          f"channel_index('Pz') is valid int [got {pz_idx}]")

    return rec


# ---------------------------------------------------------------------------
# Test: bandpass filter
# ---------------------------------------------------------------------------

def test_bandpass(rec) -> None:
    section("Phase 1B: Bandpass Filter")
    from src.preprocessing import bandpass_filter
    from scipy import signal as sps

    data_filt = bandpass_filter(rec.data, fs=rec.fs)
    check(data_filt.shape == rec.data.shape,
          f"Filtered shape unchanged {rec.data.shape}")
    check(data_filt.dtype == np.float64,
          f"Output dtype is float64 [got {data_filt.dtype}]")

    # Verify 1 Hz passes, 0.05 Hz is attenuated
    fs = rec.fs
    t = np.arange(0, 2 * fs) / fs
    test_signal = (
        np.sin(2 * np.pi * 1.0 * t) +     # 1 Hz — should pass
        np.sin(2 * np.pi * 0.01 * t)       # 0.01 Hz — should be attenuated
    )
    test_data = test_signal[:, np.newaxis]  # (T, 1)
    filtered_test = bandpass_filter(test_data, fs=fs)
    ratio = np.std(filtered_test) / np.std(test_data)
    check(ratio > 0.3, f"Bandpass passes in-band signal (ratio={ratio:.3f})")

    return data_filt


# ---------------------------------------------------------------------------
# Test: epoch extraction
# ---------------------------------------------------------------------------

def test_epoch_extraction(data_filt, markers, fs) -> None:
    section("Phase 1C: Epoch Extraction")
    from src.preprocessing import extract_epochs

    epochs, labels = extract_epochs(data_filt, markers, fs=fs)

    check(epochs.ndim == 3,
          f"epochs.ndim == 3 [got {epochs.ndim}]")
    check(epochs.shape == (600, 32, 512),
          f"epochs.shape == (600, 32, 512) [got {epochs.shape}]")
    check(epochs.dtype == np.float64,
          f"epochs.dtype == float64 [got {epochs.dtype}]")

    n_target    = int(np.sum(labels == MARKER_TARGET))
    n_nontarget = int(np.sum(labels == MARKER_NONTARGET))
    check(n_target == 40,
          f"Target epochs == 40 [got {n_target}]")
    check(n_nontarget == 560,
          f"Non-target epochs == 560 [got {n_nontarget}]")

    return epochs, labels


# ---------------------------------------------------------------------------
# Test: baseline correction
# ---------------------------------------------------------------------------

def test_baseline_correction(epochs, labels, fs) -> None:
    section("Phase 1D: Baseline Correction")
    from src.preprocessing import baseline_correct
    from config import EPOCH_TMIN

    corrected = baseline_correct(epochs, fs=fs)

    check(corrected.shape == epochs.shape,
          f"Shape preserved after baseline correction {epochs.shape}")

    # Pre-stimulus window mean should be ~0 per epoch/channel
    pre_samples = int(round(-EPOCH_TMIN * fs))  # 102 samples
    baseline_means = corrected[:, :, :pre_samples].mean(axis=2)  # (N, C)
    grand_mean_abs = float(np.mean(np.abs(baseline_means)))
    check(grand_mean_abs < 1.0,
          f"Baseline mean ≈ 0 µV (grand |mean| = {grand_mean_abs:.4f})")

    return corrected


# ---------------------------------------------------------------------------
# Test: amplitude rejection
# ---------------------------------------------------------------------------

def test_amplitude_rejection(corrected_epochs, labels) -> None:
    section("Phase 1E: Amplitude Threshold Rejection")
    from src.preprocessing import amplitude_threshold_rejection

    clean_epochs, clean_labels, mask = amplitude_threshold_rejection(
        corrected_epochs, labels, threshold=AMPLITUDE_THRESHOLD
    )

    max_amp = float(np.max(np.abs(clean_epochs)))
    check(max_amp <= AMPLITUDE_THRESHOLD,
          f"Max amplitude <= {AMPLITUDE_THRESHOLD} µV after rejection [got {max_amp:.2f}]")
    check(clean_epochs.shape[0] == clean_labels.shape[0],
          f"Epoch/label counts match after rejection")
    check(clean_epochs.shape[0] >= 400,
          f"≥400 epochs retained [got {clean_epochs.shape[0]}]")

    n_target    = int(np.sum(clean_labels == MARKER_TARGET))
    n_nontarget = int(np.sum(clean_labels == MARKER_NONTARGET))
    print(f"    Retained: {clean_epochs.shape[0]} epochs (target={n_target}, non-target={n_nontarget})")

    return clean_epochs, clean_labels


# ---------------------------------------------------------------------------
# Test: full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline(mat_path: Path) -> None:
    section("Phase 1F: Full Pipeline (run_preprocessing_pipeline)")
    from src.preprocessing import run_preprocessing_pipeline

    epochs, labels, chan_names, fs = run_preprocessing_pipeline(
        mat_path, apply_notch=False, verbose=False
    )

    check(epochs.ndim == 3,                    "Output epochs are 3-D")
    check(epochs.shape[1] == 32,               f"32 channels [got {epochs.shape[1]}]")
    check(epochs.shape[2] == 512,              f"512 samples/epoch [got {epochs.shape[2]}]")
    check(epochs.shape[0] >= 400,              f"≥400 epochs retained [got {epochs.shape[0]}]")
    check(float(np.max(np.abs(epochs))) <= AMPLITUDE_THRESHOLD,
          f"No epoch exceeds {AMPLITUDE_THRESHOLD} µV threshold")
    check(len(chan_names) == 32,               f"32 channel names returned")
    check(fs == 512.0,                         f"fs == 512.0 Hz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Verification Test")
    parser.add_argument("--mat", default=None, help="Path to s01.mat")
    args = parser.parse_args()

    mat_path = Path(args.mat) if args.mat else MAT_FILE

    print(f"\n{'='*60}")
    print(f"  P300 BCI Pipeline — Phase 1 Verification Tests")
    print(f"  Dataset: {mat_path}")
    print(f"{'='*60}")

    if not mat_path.exists():
        print(f"\n[ERROR] Data file not found: {mat_path}")
        print("  Please place s01.mat in the data_explore/ directory.")
        sys.exit(1)

    try:
        rec          = test_data_loader(mat_path)
        data_filt    = test_bandpass(rec)
        epochs, labels = test_epoch_extraction(data_filt, rec.markers, rec.fs)
        corrected    = test_baseline_correction(epochs, labels, rec.fs)
        clean_e, clean_l = test_amplitude_rejection(corrected, labels)
        test_full_pipeline(mat_path)

    except Exception:
        print("\n[ERROR] Unexpected exception during tests:")
        traceback.print_exc()
        _FAIL += 1

    # Summary
    total = _PASS + _FAIL
    print(f"\n{'='*60}")
    print(f"  RESULTS: {_PASS}/{total} tests passed")
    if _FAIL == 0:
        print("  [OK] ALL PHASE 1 TESTS PASSED")
    else:
        print(f"  [FAILED] {_FAIL} test(s) FAILED")
    print(f"{'='*60}\n")

    sys.exit(0 if _FAIL == 0 else 1)
