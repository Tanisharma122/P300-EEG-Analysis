# 🧠 P300 EEG Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research%20%2F%20Educational-green)](#-license)
[![Paradigm](https://img.shields.io/badge/Paradigm-RSVP%20Oddball-orange)](https://en.wikipedia.org/wiki/Oddball_paradigm)
[![EEG](https://img.shields.io/badge/EEG-32%20Channel-purple)](https://en.wikipedia.org/wiki/Electroencephalography)

A complete Python analysis pipeline for detecting and visualizing the **P300 Event-Related Potential (ERP)** from 32-channel EEG data recorded during an RSVP (Rapid Serial Visual Presentation) oddball paradigm. The dataset is stored in MATLAB v7.3 (HDF5) format.

---

## 📋 Overview

The **P300** is a positive ERP component peaking ~300 ms after a rare or meaningful stimulus. It is a cornerstone of Brain-Computer Interface (BCI) research and clinical neuroscience.

This project delivers a **complete exploratory analysis pipeline** for Subject 01, covering:

| Stage | What it does |
|---|---|
| 🔍 **Exploration** | Inspect the HDF5 file structure, keys, shapes, and dtypes |
| 🏷️ **Marker analysis** | Decode stimulus markers, channel labels, ISI statistics |
| ✂️ **Epoch extraction** | Cut −200 ms to +800 ms segments around every stimulus |
| 📊 **Visualization** | 5 publication-quality figures covering signal quality, ERP, and discriminability |
| 📝 **Conclusion** | Full analysis report with scientific interpretation |

### Key Results (Subject 01)

> **The P300 is real, strong, and scientifically valid in this dataset.**

- ✅ Clear P300 peak at **~10 µV** in the **300–500 ms** window
- ✅ Top discriminating electrodes: **Cz (16.2 µV), CP2 (15.7 µV), Pz (15.5 µV)** — matching neuroscience literature
- ✅ Clean recording: natural 1/f spectrum, **no line noise** (50 Hz / 60 Hz absent)
- ⚠️ Severe class imbalance: **40 targets vs 560 non-targets** (1:14 ratio)
- ⚠️ Minor frontal artifacts (FP1/AF3) — fixable with ICA

---

## 📁 Project Structure

`
P300 Detection/
│
├── 📁 data_explore/                   ← All analysis code & reports
│   ├── explore_dataset.py             # Initial HDF5 key explorer
│   ├── check_structure.py             # Print all groups, shapes & dtypes
│   ├── check_markers.py               # Inspect markers, channel names, ISI stats
│   ├── visulaize_p300_dataset.py      # Main script — epoch + generate all 5 figures
│   ├── ANALYSIS_CONCLUSION.md         # Full formatted analysis report
│   ├── ANALYSIS_CONCLUSION.txt        # Plain-text version of the report
│   └── .gitignore                     # Excludes .mat files and generated PNGs
│
├── 📁 p300_figures_s01/               ← Generated publication-quality figures
│   ├── s01_01_raw_snippet.png
│   ├── s01_02_erp_grand_average.png
│   ├── s01_03_label_distribution.png
│   ├── s01_04_channel_difference_wave.png
│   └── s01_05_psd.png
│
└── README.md                          ← This file
`

> ⚠️ **s01.mat (~265 MB) is excluded from this repo** due to GitHub file size limits.
> Download the dataset separately and place it in data_explore/ before running the scripts.

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| **File** | s01.mat |
| **Format** | MATLAB v7.3 (HDF5) |
| **HDF5 root** | /RSVP/ |
| **Signal** | /RSVP/data → (166770, 32) — samples × channels |
| **Markers** | /RSVP/markers_target → (166770,) — 0=baseline, 1=target, 2=non-target |
| **Sampling rate** | **512 Hz** |
| **Duration** | ~5.4 minutes |
| **Channels** | **32** (10-20 EEG cap) |

### Channel Layout

`
FP1  AF3  F7   F3   FC1  FC5  T7   C3
CP1  CP5  P7   P3   Pz   PO3  O1   Oz
O2   PO4  P4   P8   CP6  CP2  C4   T8
FC6  FC2  F4   F8   AF4  FP2  FZ   Cz
`

---

## 🚀 Getting Started

### Prerequisites

`ash
pip install h5py numpy scipy matplotlib
`

### Running the Scripts

All scripts are inside the data_explore/ folder:

**Step 1 — Explore the HDF5 structure:**
`ash
cd data_explore
python check_structure.py
`

**Step 2 — Inspect stimulus markers and channel labels:**
`ash
python check_markers.py
`

**Step 3 — Epoch the data and generate all 5 figures:**
`ash
python visulaize_p300_dataset.py s01.mat
`

Figures are saved to p300_figures_s01/ in the parent directory.

---

## 📊 Generated Figures

| Figure | Description |
|---|---|
| s01_01_raw_snippet.png | First 3 s of raw EEG — sanity check on signal quality |
| s01_02_erp_grand_average.png | **Core P300 plot** — Target vs Non-target grand average |
| s01_03_label_distribution.png | Class counts — shows the 1:14 imbalance |
| s01_04_channel_difference_wave.png | Top 8 electrodes ranked by P300 discriminability |
| s01_05_psd.png | Power spectral density — confirms no line noise |

---

## 🧠 Background: What is P300?

The P300 is an **endogenous ERP component** — it reflects cognitive processing, not just sensory response:

| Property | Detail |
|---|---|
| **Latency** | ~300 ms post-stimulus (can range 250–600 ms) |
| **Amplitude** | Higher for rare / target stimuli (oddball paradigm) |
| **Scalp topography** | Maximal at **Pz, Cz, CP1, CP2** (parieto-central midline) |
| **Applications** | P300-based speller BCIs, attention monitoring, clinical diagnosis |

In an **RSVP oddball paradigm**, images flash rapidly. Occasionally a target appears among non-targets. The brain generates a P300 only for targets — this involuntary neural signature is what BCI systems exploit to infer user intent without motor movement.

---

## ⚠️ Class Imbalance Warning

The dataset has a severe **1:14 target-to-non-target ratio**.

| Metric | Why it matters |
|---|---|
| Raw accuracy | ❌ Misleading — "always non-target" scores **93.3%** |
| **AUC-ROC** | ✅ Primary metric to use |
| F1-score | ✅ Good secondary metric |

Always use class_weight='balanced' or SMOTE when training classifiers on this data.

---

## 🛠️ Recommended Next Steps

### Preprocessing Pipeline

`
Raw EEG
  └─► Bandpass Filter (0.1–30 Hz)   → remove DC drift + high-freq noise
        └─► Baseline Correction      → subtract mean of −200ms to 0ms per epoch
              └─► ICA                → remove eye-blink components (FP1/AF3)
                    └─► Epoch Rejection → drop epochs with amplitude > 100 µV
                          └─► Clean Epochs ✅
`

### Classification Roadmap

| Classifier | Speed | Notes |
|---|---|---|
| **LDA** | ⚡ Fast | Classic P300 BCI algorithm |
| **SVM (RBF)** | Medium | Robust to small sample sizes |
| **EEGNet (CNN)** | Slow | ~90%+ AUC — state of the art |
| **SWLDA** | Fast | Used in BCI2000 P300 speller |

### Future Work

- [ ] Bandpass filtering (0.1–30 Hz) + baseline correction
- [ ] ICA artifact removal (eye-blinks on FP1/AF3)
- [ ] xDAWN spatial filtering for P300 amplification
- [ ] ML classification: LDA → SVM → EEGNet
- [ ] Cross-subject generalization (s02.mat, s03.mat, ...)
- [ ] Real-time BCI simulation

---

## 📝 Analysis Report

A full scientific analysis of Subject 01 is available in:

- [ANALYSIS_CONCLUSION.md](data_explore/ANALYSIS_CONCLUSION.md) — Formatted markdown with tables, findings, and pipeline
- [ANALYSIS_CONCLUSION.txt](data_explore/ANALYSIS_CONCLUSION.txt) — Plain-text version

---

## 📄 License

This project is for **educational and research purposes only**.
The EEG dataset (s01.mat) is not included — please refer to its original source for licensing terms.

---

*Analysis · Subject 01 · RSVP Oddball Paradigm · 32-channel EEG · 512 Hz · August 2026*
