"""
Epoch + visualize the RSVP P300 dataset (s01.mat), using the confirmed structure:

  /RSVP/data           (166770, 32)   continuous EEG, samples x channels
  /RSVP/markers_target (166770,)      0 = no event, 1 = target, 2 = non-target
  /RSVP/srate           scalar        512 Hz
  /RSVP/chanlocs/labels (32,1)        channel names (object refs -> decode)

This script:
  1. Loads the continuous data + markers + channel names.
  2. Cuts epochs around EVERY marker (target=1 and non-target=2) using a
     window of -200ms to +800ms relative to stimulus onset.
  3. Plots:
       a) Raw signal snippet
       b) Grand-average ERP: target vs non-target (the core P300 plot)
       c) Class distribution (40 target vs 560 non-target -> imbalance)
       d) Per-channel target-minus-nontarget difference wave, using REAL
          channel names, so you can confirm Pz/Cz/CP1/CP2 etc. show the
          strongest effect
       e) PSD to check line noise

Run:  python epoch_and_visualize_p300.py s01.mat
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps


EPOCH_START_S = -0.2   # 200ms before stimulus
EPOCH_END_S = 0.8       # 800ms after stimulus
TARGET_CODE = 1
NONTARGET_CODE = 2
OUTPUT_DIR = "p300_figures"   # all plots get saved inside this folder


def decode_channel_labels(f):
    label_refs = f["/RSVP/chanlocs/labels"]
    names = []
    for i in range(label_refs.shape[0]):
        ref = label_refs[i, 0]
        char_codes = f[ref][()].flatten()
        names.append("".join(chr(c) for c in char_codes))
    return names


def load_all(path):
    with h5py.File(path, "r") as f:
        data = f["/RSVP/data"][()]                 # (166770, 32) samples x channels
        markers = f["/RSVP/markers_target"][()].flatten()
        fs = float(f["/RSVP/srate"][()].flatten()[0])
        chan_names = decode_channel_labels(f)
    return data, markers, fs, chan_names


def make_epochs(data, markers, fs, start_s, end_s):
    """data: (samples, channels). Returns epochs (n_events, channels, samples), labels (n_events,)"""
    pre = int(round(-start_s * fs))
    post = int(round(end_s * fs))
    epoch_len = pre + post

    event_idx = np.nonzero(markers)[0]
    event_val = markers[event_idx].astype(int)

    n_samples_total = data.shape[0]
    epochs = []
    labels = []
    dropped = 0
    for idx, val in zip(event_idx, event_val):
        s0 = idx - pre
        s1 = idx + post
        if s0 < 0 or s1 > n_samples_total:
            dropped += 1
            continue
        seg = data[s0:s1, :]          # (epoch_len, channels)
        epochs.append(seg.T)          # -> (channels, epoch_len)
        labels.append(val)

    print(f"Built {len(epochs)} epochs, dropped {dropped} near recording boundaries.")
    return np.array(epochs), np.array(labels)


def plot_raw_snippet(data, fs, chan_names, n_channels=4, n_seconds=3, out=None):
    n_samples = min(int(n_seconds * fs), data.shape[0])
    t = np.arange(n_samples) / fs
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, data[:n_samples, i], linewidth=0.8)
        ax.set_ylabel(chan_names[i])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw signal snippet (first few channels)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)


def plot_grand_average_erp(epochs, labels, fs, start_s, end_s, out=None):
    n_samples = epochs.shape[2]
    t = np.linspace(start_s, end_s, n_samples)
    target_avg = epochs[labels == TARGET_CODE].mean(axis=(0, 1))
    nontarget_avg = epochs[labels == NONTARGET_CODE].mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, target_avg, color="crimson", label=f"Target (n={np.sum(labels==TARGET_CODE)})")
    ax.plot(t, nontarget_avg, color="steelblue", label=f"Non-target (n={np.sum(labels==NONTARGET_CODE)})")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(0.25, 0.5, color="grey", alpha=0.15, label="Expected P300 window")
    ax.set_xlabel("Time relative to stimulus (s)")
    ax.set_ylabel("Amplitude (µV, averaged across all 32 channels)")
    ax.set_title("Grand-average ERP: Target vs Non-target")
    ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)


def plot_label_distribution(labels, out=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [np.sum(labels == NONTARGET_CODE), np.sum(labels == TARGET_CODE)]
    bars = ax.bar(["Non-target", "Target"], counts, color=["steelblue", "crimson"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title(f"Class imbalance ~1:{round(counts[0]/counts[1], 1)}")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)


def plot_channel_difference_wave(epochs, labels, fs, start_s, end_s, chan_names, out=None, top_n=8):
    target_avg = epochs[labels == TARGET_CODE].mean(axis=0)      # (channels, samples)
    nontarget_avg = epochs[labels == NONTARGET_CODE].mean(axis=0)
    diff = target_avg - nontarget_avg

    n_samples = diff.shape[1]
    t = np.linspace(start_s, end_s, n_samples)
    window_mask = (t >= 0.25) & (t <= 0.5)
    peak_scores = np.abs(diff[:, window_mask]).max(axis=1)
    top_channels = np.argsort(peak_scores)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 5))
    for ch in top_channels:
        ax.plot(t, diff[ch], label=chan_names[ch], linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(0.25, 0.5, color="grey", alpha=0.15)
    ax.set_xlabel("Time relative to stimulus (s)")
    ax.set_ylabel("Target − Non-target amplitude")
    ax.set_title(f"Top {top_n} channels by P300 discriminability")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)

    print("Top channels by peak target-vs-nontarget difference (0.25-0.5s):")
    for ch in top_channels:
        print(f"  {chan_names[ch]}: peak diff = {peak_scores[ch]:.4g}")


def plot_psd(data, fs, chan_names, n_channels=4, out=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_channels):
        f_axis, pxx = sps.welch(data[:, i], fs=fs, nperseg=min(2048, data.shape[0]))
        ax.semilogy(f_axis, pxx, label=chan_names[i])
    ax.axvline(50, color="grey", linestyle=":", label="50Hz")
    ax.axvline(60, color="grey", linestyle="--", label="60Hz")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power spectral density (line-noise check)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)


def main(path):
    # Name the output folder after the input file, e.g. "s01.mat" -> "p300_figures_s01"
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = f"{OUTPUT_DIR}_{base_name}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Figures will be saved to: {os.path.abspath(out_dir)}")

    def fig_path(name):
        return os.path.join(out_dir, f"{base_name}_{name}.png")

    print(f"Loading {path} ...")
    data, markers, fs, chan_names = load_all(path)
    print("Data shape (samples, channels):", data.shape)
    print("Sampling rate:", fs)
    print("Channels:", chan_names)

    epochs, labels = make_epochs(data, markers, fs, EPOCH_START_S, EPOCH_END_S)
    print("Epochs shape (n_events, channels, samples):", epochs.shape)
    print("Label counts -> target:", np.sum(labels == TARGET_CODE),
          "| non-target:", np.sum(labels == NONTARGET_CODE))

    plot_raw_snippet(data, fs, chan_names, out=fig_path("01_raw_snippet"))
    plot_grand_average_erp(epochs, labels, fs, EPOCH_START_S, EPOCH_END_S, out=fig_path("02_erp_grand_average"))
    plot_label_distribution(labels, out=fig_path("03_label_distribution"))
    plot_channel_difference_wave(epochs, labels, fs, EPOCH_START_S, EPOCH_END_S, chan_names, out=fig_path("04_channel_difference_wave"))
    plot_psd(data, fs, chan_names, out=fig_path("05_psd"))

    print(f"\nAll figures saved in: {os.path.abspath(out_dir)}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "s01.mat"
    main(path)