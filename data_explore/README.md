# P300 EEG Analysis

A Python project for analyzing P300 Event-Related Potential (ERP) signals from EEG data, using MATLAB v7.3 `.mat` files as the dataset source.

## 📋 Overview

The **P300** is a well-known ERP component that occurs approximately 300ms after a rare or meaningful stimulus. It is widely used in Brain-Computer Interface (BCI) systems, clinical neuroscience, and cognitive research.

This project focuses on:
- Loading and exploring EEG data stored in MATLAB `.mat` (HDF5/v7.3) format
- Understanding dataset structure (keys, shapes, dtypes)
- Building the foundation for P300 detection and signal classification

## 📁 Project Structure

```
P300 Detection/
├── explore_dataset.py    # Explore keys/structure of the .mat file
├── 2.py                  # Recursive HDF5 dataset explorer
├── .gitignore            # Excludes large .mat data files
└── README.md             # Project documentation
```

## 🗃️ Dataset

The dataset used is a MATLAB v7.3 (HDF5) file: `s01.mat`

> ⚠️ **Note:** The raw `.mat` data file is **~265 MB** and is excluded from this repository due to GitHub's file size limits. Please acquire the dataset separately and place it in the project root before running the scripts.

The dataset likely follows the standard EEG-BCI format with:
- Multi-channel EEG recordings
- Stimulus event markers
- Trial/epoch metadata

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install h5py numpy
```

### Running the Scripts

**Explore the dataset keys:**
```bash
python explore_dataset.py
```

**Recursively inspect all HDF5 groups/datasets:**
```bash
python 2.py
```

## 🧠 Background: What is P300?

The P300 is an endogenous ERP component:
- **Latency:** ~300ms post-stimulus
- **Amplitude:** Higher for rare/target stimuli (oddball paradigm)
- **Application:** P300-based speller BCIs, attention monitoring, clinical diagnosis

## 🛠️ Future Work

- [ ] Epoch extraction around stimulus events
- [ ] Baseline correction and filtering (bandpass 0.1–30 Hz)
- [ ] Feature extraction (amplitude, latency)
- [ ] P300 classification using ML (LDA, SVM, CNNs)
- [ ] Cross-subject analysis

## 📄 License

This project is for educational and research purposes.
