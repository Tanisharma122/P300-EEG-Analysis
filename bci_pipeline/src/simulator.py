"""
src/simulator.py — Real-Time P300 BCI Stream Simulator.

Simulates a live EEG acquisition system by replaying a pre-recorded dataset
at native sampling rate (512 Hz). When a stimulus marker is detected, it
buffers the corresponding epoch, runs the trained pipeline inference, and
emits a P300 confidence score in [0.0, 1.0].

Architecture
------------
  EEGStreamSimulator
    ├── _ring_buffer : circular buffer holding `tmax - tmin` seconds of EEG
    ├── _marker_queue: detected stimulus events to process
    ├── _pipeline    : fitted sklearn / EEGNet pipeline
    └── run()        : main playback loop

Usage
-----
    from src.simulator import EEGStreamSimulator

    sim = EEGStreamSimulator(rec=recording, pipeline=fitted_pipeline)
    sim.run(max_events=20)
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        FS, EPOCH_TMIN, EPOCH_TMAX, MARKER_TARGET, MARKER_NONTARGET,
        BANDPASS_LO, BANDPASS_HI, FILTER_ORDER,
    )
except ImportError:
    FS = 512.0
    EPOCH_TMIN = -0.2
    EPOCH_TMAX = 0.8
    MARKER_TARGET = 1
    MARKER_NONTARGET = 2
    BANDPASS_LO = 0.1
    BANDPASS_HI = 30.0
    FILTER_ORDER = 4

logger = logging.getLogger(__name__)

# Type alias
OnPredictionCallback = Callable[[int, float, int, float], None]


class RingBuffer:
    """Fixed-size circular buffer for continuous EEG streaming.

    Parameters
    ----------
    n_samples : int
        Maximum number of samples to hold.
    n_channels : int
        Number of EEG channels.
    """

    def __init__(self, n_samples: int, n_channels: int) -> None:
        self._buf = np.zeros((n_samples, n_channels), dtype=np.float64)
        self._n   = n_samples
        self._idx = 0
        self._filled = False

    def push(self, sample: np.ndarray) -> None:
        """Push a single sample ``(n_channels,)`` into the buffer."""
        self._buf[self._idx] = sample
        self._idx = (self._idx + 1) % self._n
        if self._idx == 0:
            self._filled = True

    def push_batch(self, batch: np.ndarray) -> None:
        """Push a batch of samples ``(n_batch, n_channels)`` into the buffer."""
        for sample in batch:
            self.push(sample)

    def read(self) -> np.ndarray:
        """Return ordered buffer contents ``(n_samples, n_channels)``."""
        if self._filled:
            return np.roll(self._buf, -self._idx, axis=0)
        return self._buf[: self._idx].copy()

    @property
    def is_filled(self) -> bool:
        return self._filled


class EEGStreamSimulator:
    """Real-time P300 BCI stream simulator.

    Replays raw EEG data sample-by-sample (or in mini-batches) at simulated
    512 Hz, detects flash markers, buffers epochs, runs inference, and calls
    back with P300 confidence scores.

    Parameters
    ----------
    data : np.ndarray
        Continuous EEG, shape ``(n_samples, n_channels)`` in µV.
    markers : np.ndarray
        Per-sample marker array, shape ``(n_samples,)``.
    pipeline : Any
        Fitted sklearn-compatible pipeline. Must implement ``predict_proba``.
    fs : float
        Sampling frequency in Hz.
    epoch_tmin : float
        Epoch start relative to stimulus onset (default −0.2 s).
    epoch_tmax : float
        Epoch end relative to stimulus onset (default +0.8 s).
    speed_factor : float
        Playback speed multiplier (1.0 = real-time, 10.0 = 10× faster).
    on_prediction : callable | None
        Optional callback: ``on_prediction(event_idx, confidence, true_label, elapsed_s)``.
    """

    def __init__(
        self,
        data: np.ndarray,
        markers: np.ndarray,
        pipeline,
        fs: float = FS,
        epoch_tmin: float = EPOCH_TMIN,
        epoch_tmax: float = EPOCH_TMAX,
        speed_factor: float = 50.0,
        on_prediction: Optional[OnPredictionCallback] = None,
    ) -> None:
        self.data          = data
        self.markers       = markers
        self.pipeline      = pipeline
        self.fs            = fs
        self.epoch_tmin    = epoch_tmin
        self.epoch_tmax    = epoch_tmax
        self.speed_factor  = speed_factor
        self.on_prediction = on_prediction

        # Epoch geometry
        self._pre_samples  = int(round(-epoch_tmin * fs))
        self._post_samples = int(round(epoch_tmax * fs))
        self._epoch_len    = self._pre_samples + self._post_samples

        # Ring buffer holds enough history for one full epoch
        buffer_len = self._pre_samples + self._post_samples + 64  # a few extra
        self._ring = RingBuffer(buffer_len, data.shape[1])

        # Queue of (sample_idx, true_label) for pending epoch extraction
        self._pending: deque[tuple[int, int]] = deque()

        logger.info(
            "EEGStreamSimulator ready: %d total samples, epoch window [%g, %g] s, "
            "speed_factor=%.1f×",
            len(data), epoch_tmin, epoch_tmax, speed_factor,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_epoch(self, raw_epoch: np.ndarray) -> np.ndarray:
        """Apply bandpass filter + baseline correction to a single epoch.

        Parameters
        ----------
        raw_epoch : np.ndarray
            Shape ``(n_samples, n_channels)``.

        Returns
        -------
        np.ndarray
            Shape ``(1, n_channels, n_samples)`` — ready for pipeline.
        """
        from src.preprocessing import bandpass_filter, baseline_correct

        filtered = bandpass_filter(raw_epoch, fs=self.fs)
        epoch_3d = filtered.T[np.newaxis, :, :]   # (1, C, T)
        epoch_3d = baseline_correct(epoch_3d, fs=self.fs, tmin=self.epoch_tmin)
        return epoch_3d

    def _infer(self, epoch_3d: np.ndarray) -> float:
        """Run the fitted pipeline on a single epoch and return P300 confidence.

        Parameters
        ----------
        epoch_3d : np.ndarray
            Shape ``(1, n_channels, n_samples)``.

        Returns
        -------
        float
            Probability in [0.0, 1.0] that the epoch contains a P300.
        """
        try:
            proba = self.pipeline.predict_proba(epoch_3d)
            # proba[:, -1] is the target (P300) probability
            return float(proba[0, -1])
        except Exception as exc:
            logger.warning("Inference error: %s — returning 0.0", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        max_events: int = 50,
        start_sample: int = 0,
    ) -> list[dict]:
        """Run the stream simulation.

        Iterates through EEG samples, pushes them into the ring buffer,
        detects stimulus markers, waits until enough post-stimulus samples
        have been collected, extracts the epoch, runs inference, and
        reports the P300 confidence score.

        Parameters
        ----------
        max_events : int
            Stop after processing this many stimulus events.
        start_sample : int
            Sample index at which to begin playback.

        Returns
        -------
        list[dict]
            One entry per processed event:
            ``{'event_num', 'sample_idx', 'true_label', 'confidence', 'correct'}``.
        """
        results: list[dict] = []
        events_processed = 0
        pending: list[tuple[int, int, int]] = []  # (trigger_sample, true_label, target_ready_at)

        n_total = len(self.data)
        batch_size = max(1, int(self.fs // 10))   # 51 samples per step ≈ 10ms chunks
        step_sleep = (batch_size / self.fs) / self.speed_factor

        current_sample = start_sample
        print(f"\n[simulator] Starting stream at sample {start_sample} | "
              f"speed={self.speed_factor:.0f}× real-time | max_events={max_events}")
        print(f"{'─'*65}")
        print(f"  {'Event':>6}  {'SampleIdx':>10}  {'TrueLabel':>10}  "
              f"{'Confidence':>12}  {'Correct?':>8}")
        print(f"{'─'*65}")

        t_start = time.time()

        while current_sample < n_total and events_processed < max_events:
            # --- Push batch into ring buffer ---
            end_sample = min(current_sample + batch_size, n_total)
            self._ring.push_batch(self.data[current_sample:end_sample])

            # --- Detect new stimulus events in this batch ---
            for si in range(current_sample, end_sample):
                mv = int(self.markers[si])
                if mv in (MARKER_TARGET, MARKER_NONTARGET):
                    ready_at = si + self._post_samples
                    pending.append((si, mv, ready_at))

            # --- Process matured events (post-stimulus window collected) ---
            ready = [p for p in pending if p[2] <= end_sample]
            pending = [p for p in pending if p[2] > end_sample]

            for trigger_si, true_label, _ in ready:
                if events_processed >= max_events:
                    break

                s0 = trigger_si - self._pre_samples
                s1 = trigger_si + self._post_samples
                if s0 < 0 or s1 > n_total:
                    continue

                raw_epoch = self.data[s0:s1, :]          # (T, C)
                epoch_3d  = self._preprocess_epoch(raw_epoch)
                confidence = self._infer(epoch_3d)

                pred_label = MARKER_TARGET if confidence >= 0.5 else MARKER_NONTARGET
                correct = (pred_label == true_label)

                result = {
                    "event_num":  events_processed + 1,
                    "sample_idx": trigger_si,
                    "true_label": true_label,
                    "confidence": confidence,
                    "correct":    correct,
                }
                results.append(result)
                events_processed += 1

                label_str  = "TARGET" if true_label == MARKER_TARGET else "nontarget"
                correct_str = "✓" if correct else "✗"
                bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                print(
                    f"  {events_processed:>6}  {trigger_si:>10}  {label_str:>10}  "
                    f"{confidence:>8.4f}  {bar}  {correct_str}"
                )

                if self.on_prediction:
                    self.on_prediction(
                        events_processed, confidence, true_label,
                        time.time() - t_start,
                    )

            current_sample = end_sample
            time.sleep(step_sleep)

        elapsed = time.time() - t_start
        n_correct = sum(r["correct"] for r in results)
        print(f"{'─'*65}")
        print(f"[simulator] Done in {elapsed:.2f}s | "
              f"Processed {events_processed} events | "
              f"Accuracy = {n_correct}/{events_processed} "
              f"({100*n_correct/max(1, events_processed):.1f}%)\n")

        return results
