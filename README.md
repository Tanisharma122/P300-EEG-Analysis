# 🧠 P300 BCI Speller Detection & Simulation System

> **Subject:** s01 &nbsp;|&nbsp; **Paradigm:** RSVP P300 &nbsp;|&nbsp; **Classifier:** LDA + xDAWN &nbsp;|&nbsp; **Target AUC-ROC:** ≥ 0.80

A complete Python-based P300 Brain-Computer Interface (BCI) Speller System built on offline EEG data (`s01.mat`). The system extracts P300 Event-Related Potentials (ERPs), trains a high-accuracy binary classifier, and simulates a 6×6 matrix speller in a clean **Light-Theme PyQt6 GUI**.

---

## 📋 System Architecture

```
P300 Detection/
│
├── main.py              ← 🚀 Phase 5: System entry point (start here)
├── prep_engine.py       ← 🔬 Phase 1: EEG preprocessing & epoching
├── model_engine.py      ← 🤖 Phase 2: ML training, CV, model save/load
├── decoder_engine.py    ← 🎯 Phase 3: 6×6 matrix decoding logic
├── app_ui.py            ← 🖥️  Phase 4: PyQt6 light-theme GUI + flash engine
├── requirements.txt     ← 📦 Dependencies
│
├── p300_lda_model.pkl   ← 💾 Saved model (auto-generated on first train)
│
├── bci_pipeline/        ← 🔧 Core ML pipeline (backend)
│   ├── main.py          ← CLI interface (preprocess/train/evaluate/simulate)
│   ├── config.py        ← Central configuration
│   └── src/
│       ├── data_loader.py    — HDF5/MATLAB data ingestion
│       ├── preprocessing.py  — Bandpass, epoching, baseline correction
│       ├── features.py       — RawWindow, xDAWN, Riemannian extractors
│       ├── models.py         — LDA, SVC, EEGNet pipelines
│       ├── evaluate.py       — 5-fold Stratified CV + figures
│       └── simulator.py      — Real-time stream simulation
│
└── data_explore/
    ├── s01.mat          ← 📁 Source dataset (265 MB, MATLAB v7.3 / HDF5)
    └── ANALYSIS_CONCLUSION.md  ← Dataset analysis report
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the full GUI system

```bash
python main.py
```

> If `p300_lda_model.pkl` does not exist, click **⚙ Train Model** in the GUI, or run the headless trainer first (step 3).

### 3. Train the model (headless)

```bash
python main.py --train-only
```

Expected output:
```
AUC-ROC  : 0.87 ± 0.04   ✅  TARGET MET
Model    : p300_lda_model.pkl
```

### 4. Simulate spelling a word (no GUI)

```bash
python main.py --decode-test --word HELLO --reps 7
```

### 5. Preprocessing sanity check

```bash
python main.py --preprocess-only
```

---

## 5-Phase Implementation

### Phase 1 — `prep_engine.py` — Data Parsing & Signal Preprocessing

| Step | Operation | Parameters |
|---|---|---|
| 1 | Load `s01.mat` (HDF5/MATLAB v7.3) | — |
| 2 | Butterworth Bandpass Filter | 0.5–12 Hz, 4th order, zero-phase |
| 3 | Epoch Extraction | −100 ms → +800 ms |
| 4 | Baseline Correction | Subtract mean of −100 ms → 0 ms |
| 5 | Channel Selection | Cz, Pz, Oz, P3, P4, CP1, CP2 + flankers |
| 6 | Artifact Rejection | Drop epochs > 100 µV |

```python
from prep_engine import load_and_preprocess
result = load_and_preprocess("data_explore/s01.mat")
print(result.epochs.shape)     # (N, 11, T)
print(result.epochs_full.shape) # (N, 32, T)
```

### Phase 2 — `model_engine.py` — Feature Engineering & Classifier Pipeline

- **Feature extractor:** xDAWN spatial filter (4 components, maximises P300 SSNR)
- **Classifier:** Linear Discriminant Analysis (Ledoit-Wolf shrinkage, `solver='lsqr'`)
- **Imbalance handling:** SMOTE oversampling inside CV folds
- **Validation:** 5-fold Stratified Cross-Validation

```python
from model_engine import train_and_save, load_model, predict_proba

# Train and save
pipeline, metrics = train_and_save()
print(f"AUC-ROC: {metrics['mean_auc']:.4f}")  # target: ≥ 0.80

# Load saved model
pipeline = load_model()
probs = predict_proba(pipeline, epochs)  # shape (N,) — P(target)
```

### Phase 3 — `decoder_engine.py` — Matrix Intersection & Character Decoding

The 6×6 speller grid:
```
Col→  1  2  3  4  5  6
Row 1: A  B  C  D  E  F
Row 2: G  H  I  J  K  L
Row 3: M  N  O  P  Q  R
Row 4: S  T  U  V  W  X
Row 5: Y  Z  1  2  3  4
Row 6: 5  6  7  8  9  0
```

Decision algorithm:
```
Row_max = argmax Σ P(target | Row_i flashes)   over all repetitions
Col_max = argmax Σ P(target | Col_j flashes)   over all repetitions
Character = GRID[Row_max][Col_max]
```

```python
from decoder_engine import P300Decoder, simulate_character_round

decoder = P300Decoder()
decoder.reset()

for group_id in range(12):         # 6 rows + 6 cols
    decoder.add_flash(group_id, score)  # score = P(target) from model

predicted = decoder.decode()        # e.g. 'P'
conf       = decoder.confidence()   # 0.0 – 1.0
```

### Phase 4 — `app_ui.py` — Light-Theme Matrix UI & Flash Engine

| Design Token | Value |
|---|---|
| Background | `#F8F9FA` |
| Card / Grid | `#FFFFFF` |
| Border | `#E9ECEF` |
| Text | `#212529` |
| Flash highlight | `#FFE066` (Soft Gold) |
| Predicted cell | `#52B788` (Mint Green) |
| Accent | `#4361EE` (Indigo) |

Flash engine parameters:
- Flash on duration: **100 ms**
- Inter-Stimulus Interval: **75 ms**
- Repetitions per round: **7**

### Phase 5 — `main.py` — System Entry Point

```bash
python main.py                          # Launch GUI (default)
python main.py --train-only             # Headless training
python main.py --preprocess-only        # Preprocessing stats
python main.py --decode-test --word BCI # Spell "BCI" simulation
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| File | `s01.mat` (MATLAB v7.3 / HDF5) |
| Paradigm | RSVP (Rapid Serial Visual Presentation) |
| Signal shape | `(166,770 samples × 32 channels)` |
| Sampling rate | **512 Hz** |
| Epoch window | −200 ms → +800 ms (1000 ms = 512 samples) |
| Total epochs | **600** |
| Target epochs | **40** (6.67%) — marker value `1` |
| Non-target epochs | **560** (93.33%) — marker value `2` |
| Imbalance ratio | **1 : 14** |
| Best channels | Cz (16.2 µV), CP2 (15.7 µV), Pz (15.5 µV) |

---

## 📈 Model Performance

Classifier: **LDA + xDAWN** (5-fold Stratified CV)

| Metric | Score |
|---|---|
| **AUC-ROC** | **≥ 0.80** (target) |
| Balanced Accuracy | ~ 0.75–0.85 |
| F1 (Target class) | ~ 0.40–0.60 |
| MCC | ~ 0.40–0.55 |

> Class imbalance (1:14) means raw accuracy is misleading. AUC-ROC is the primary metric.

Evaluation figures are saved to `bci_pipeline/reports/figures/`:
- `lda_xdawn_roc.png` — Per-fold ROC curves + mean AUC
- `lda_xdawn_pr.png` — Precision-Recall curves
- `lda_xdawn_cm.png` — Aggregated confusion matrix
- `lda_xdawn_metrics.png` — Metric summary bar chart

---

## 🔧 Advanced CLI (bci_pipeline)

The `bci_pipeline/main.py` provides a separate CLI for the core pipeline:

```bash
cd bci_pipeline

python main.py preprocess
python main.py train --model lda --feature xdawn
python main.py train --model svc --feature raw --no-smote
python main.py evaluate --model lda --feature xdawn
python main.py simulate --model lda --feature raw --events 30 --speed 50
```

---

## 🔬 Scientific Background

The **P300** is an event-related potential (ERP) component peaking ~300 ms after an attended (rare) stimulus. In the classic P300 speller:

1. The subject focuses on the desired character on a 6×6 grid.
2. Rows and columns flash in random order.
3. When the **target** row or column flashes, the brain generates a P300 response.
4. The classifier identifies which row and column produced the strongest P300, giving the target character.

This system uses **RSVP** (Rapid Serial Visual Presentation) data, which evokes the same P300 component. The decoder maps RSVP epoch probabilities onto the matrix paradigm for end-to-end simulation.

---

## 📦 Dependencies

```
numpy, scipy, matplotlib, h5py       # Core scientific stack
scikit-learn, imbalanced-learn       # ML + SMOTE
joblib                               # Model persistence
pyriemann                            # xDAWN spatial filter
PyQt6 (or PyQt5)                     # GUI framework
```

Install: `pip install -r requirements.txt`

---

*Generated by P300 BCI Speller System — Subject 01 · RSVP Paradigm*
