"""
app_ui.py — Phase 4: P300 BCI Speller — Light-Theme PyQt6 Matrix UI.

Implements the complete GUI for the P300 BCI Speller system with:
  - 6×6 character grid with animated row/column flash highlighting
  - Real-time classifier inference connected to the decoder engine
  - Predicted text output panel with confidence bar
  - Scrollable log console for live diagnostic messages
  - Start / Stop / Reset / Train controls

Design System (Light Theme)
----------------------------
  Background   : #F8F9FA
  Card/Grid    : #FFFFFF
  Border       : #E9ECEF
  Text         : #212529
  Accent       : #4361EE  (Indigo)
  Flash        : #FFE066  (Soft Gold)
  Predicted    : #52B788  (Mint Green)
  Danger       : #E63946  (Red)
  Font         : Segoe UI / system sans-serif

Running
-------
    python app_ui.py
    python app_ui.py --mat path/to/s01.mat --model p300_lda_model.pkl
"""

from __future__ import annotations

import sys
import os
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_BCI  = _ROOT / "bci_pipeline"
for _p in (str(_ROOT), str(_BCI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Qt import
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QTextEdit, QFrame, QProgressBar,
        QFileDialog, QSizePolicy, QScrollArea, QGroupBox, QStatusBar,
        QSplitter,
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QFontDatabase
    _QT_AVAILABLE = True
    _QT_VERSION   = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (                     # type: ignore[no-redef]
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QLabel, QPushButton, QTextEdit, QFrame, QProgressBar,
            QFileDialog, QSizePolicy, QScrollArea, QGroupBox, QStatusBar,
            QSplitter,
        )
        from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject  # type: ignore
        from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QFontDatabase  # type: ignore
        _QT_AVAILABLE = True
        _QT_VERSION   = 5
    except ImportError:
        _QT_AVAILABLE = False

from decoder_engine import (
    GRID, CHAR_TO_POS, N_ROWS, N_COLS, N_FLASH_GROUPS,
    row_group, col_group, P300Decoder,
)

logger = logging.getLogger("app_ui")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C_BG         = "#F8F9FA"
C_CARD       = "#FFFFFF"
C_BORDER     = "#E9ECEF"
C_TEXT       = "#212529"
C_TEXT_LIGHT = "#6C757D"
C_ACCENT     = "#4361EE"
C_ACCENT2    = "#7B2FBE"
C_FLASH      = "#FFE066"
C_FLASH_TEXT = "#212529"
C_PREDICTED  = "#52B788"
C_PREDICTED_T= "#FFFFFF"
C_DANGER     = "#E63946"
C_SUCCESS    = "#2D9E6A"
C_GRID_BTN   = "#FFFFFF"
C_GRID_HOVER = "#EDF2FF"
C_HEADER     = "#4361EE"

FLASH_MS  = 100    # ms — flash on duration
ISI_MS    = 75     # ms — inter-stimulus interval (flash off)
N_REPS    = 7      # flash repetitions per character round

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
STYLESHEET = f"""
/* ── Main window ── */
QMainWindow, QWidget#central {{
    background-color: {C_BG};
}}

/* ── Header bar ── */
QWidget#header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
    border-radius: 0px;
}}
QLabel#title {{
    color: white;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#subtitle {{
    color: rgba(255,255,255,0.8);
    font-size: 11px;
}}

/* ── Cards / panels ── */
QFrame#card {{
    background-color: {C_CARD};
    border: 1.5px solid {C_BORDER};
    border-radius: 12px;
}}

/* ── Grid character buttons ── */
QPushButton#gridBtn {{
    background-color: {C_GRID_BTN};
    color: {C_TEXT};
    border: 1.5px solid {C_BORDER};
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    min-width: 48px;
    min-height: 48px;
}}
QPushButton#gridBtn:hover {{
    background-color: {C_GRID_HOVER};
    border-color: {C_ACCENT};
}}

/* ── Control buttons ── */
QPushButton#btnPrimary {{
    background-color: {C_ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 20px;
}}
QPushButton#btnPrimary:hover {{
    background-color: #3551DD;
}}
QPushButton#btnPrimary:disabled {{
    background-color: #ADB5BD;
    color: white;
}}
QPushButton#btnDanger {{
    background-color: {C_DANGER};
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 20px;
}}
QPushButton#btnDanger:hover {{
    background-color: #C1121F;
}}
QPushButton#btnDanger:disabled {{
    background-color: #ADB5BD;
}}
QPushButton#btnSecondary {{
    background-color: {C_CARD};
    color: {C_TEXT};
    border: 1.5px solid {C_BORDER};
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 20px;
}}
QPushButton#btnSecondary:hover {{
    background-color: {C_GRID_HOVER};
    border-color: {C_ACCENT};
}}

/* ── Output text display ── */
QTextEdit#outputText {{
    background-color: {C_CARD};
    color: {C_TEXT};
    border: 2px solid {C_BORDER};
    border-radius: 10px;
    font-size: 28px;
    font-weight: 700;
    padding: 12px;
    letter-spacing: 4px;
}}

/* ── Log console ── */
QTextEdit#logConsole {{
    background-color: #212529;
    color: #ADB5BD;
    border: 1px solid #343A40;
    border-radius: 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 6px;
}}

/* ── Progress bar ── */
QProgressBar {{
    background-color: {C_BORDER};
    border: none;
    border-radius: 6px;
    height: 14px;
    text-align: center;
    font-size: 10px;
    font-weight: 600;
    color: {C_TEXT};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});
    border-radius: 6px;
}}

/* ── Status labels ── */
QLabel#statusOk  {{ color: {C_SUCCESS}; font-weight: 600; font-size: 12px; }}
QLabel#statusBad {{ color: {C_DANGER};  font-weight: 600; font-size: 12px; }}
QLabel#statusWarn{{ color: #F4A261;     font-weight: 600; font-size: 12px; }}

/* ── Section headers ── */
QLabel#sectionHeader {{
    color: {C_TEXT};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* ── Status bar ── */
QStatusBar {{
    background-color: {C_BORDER};
    color: {C_TEXT_LIGHT};
    font-size: 11px;
}}
"""


# ---------------------------------------------------------------------------
# Worker thread for model training
# ---------------------------------------------------------------------------
if _QT_AVAILABLE:
    class TrainWorker(QThread):
        """Background thread that runs model training without blocking the UI."""
        finished  = pyqtSignal(object, dict)   # (pipeline, metrics)
        error     = pyqtSignal(str)
        log_msg   = pyqtSignal(str)

        def __init__(self, mat_path: Path, model_path: Path) -> None:
            super().__init__()
            self.mat_path   = mat_path
            self.model_path = model_path

        def run(self) -> None:
            try:
                self.log_msg.emit("Loading & preprocessing EEG data...")
                from model_engine import train_and_save
                pipeline, metrics = train_and_save(
                    mat_path   = self.mat_path,
                    model_path = self.model_path,
                    verbose    = False,
                )
                self.finished.emit(pipeline, metrics)
            except Exception as exc:
                self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
if _QT_AVAILABLE:
    class P300SpellerApp(QMainWindow):
        """P300 BCI Speller — Main application window."""

        def __init__(
            self,
            mat_path:   Path = _ROOT / "data_explore" / "s01.mat",
            model_path: Path = _ROOT / "p300_lda_model.pkl",
        ) -> None:
            super().__init__()
            self.mat_path   = mat_path
            self.model_path = model_path

            # State
            self._pipeline       = None      # fitted sklearn pipeline
            self._prep_result    = None      # PreprocessResult
            self._decoder        = P300Decoder()
            self._flash_sequence: list[int] = []
            self._flash_idx      = 0
            self._rep_idx        = 0
            self._current_flash_group: Optional[int] = None
            self._running        = False
            self._output_text    = ""

            self._setup_ui()
            self._apply_styles()
            self._check_model_ready()

        # ── UI Construction ────────────────────────────────────────────────

        def _setup_ui(self) -> None:
            self.setWindowTitle("P300 BCI Speller System")
            self.setMinimumSize(1050, 680)
            self.resize(1200, 760)

            central = QWidget()
            central.setObjectName("central")
            self.setCentralWidget(central)
            root_layout = QVBoxLayout(central)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            # ── Header ──────────────────────────────────────────────
            header = self._make_header()
            root_layout.addWidget(header)

            # ── Toolbar ─────────────────────────────────────────────
            toolbar = self._make_toolbar()
            root_layout.addWidget(toolbar)

            # ── Splitter (Left: grid | Right: output + log) ─────────
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.setContentsMargins(12, 8, 12, 8)

            left_panel  = self._make_left_panel()
            right_panel = self._make_right_panel()

            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 5)
            splitter.setStretchFactor(1, 4)
            root_layout.addWidget(splitter, 1)

            # ── Status bar ──────────────────────────────────────────
            self._status_bar = QStatusBar()
            self.setStatusBar(self._status_bar)
            self._status_bar.showMessage("Ready — load or train a model to begin.")

            # ── Flash timer ─────────────────────────────────────────
            self._flash_timer = QTimer(self)
            self._flash_timer.timeout.connect(self._on_flash_tick)

        def _make_header(self) -> QWidget:
            header = QWidget()
            header.setObjectName("header")
            header.setFixedHeight(72)
            hl = QHBoxLayout(header)
            hl.setContentsMargins(24, 8, 24, 8)

            icon_lbl = QLabel("🧠")
            icon_lbl.setFont(QFont("Segoe UI Emoji", 28))

            title_col = QVBoxLayout()
            title_col.setSpacing(2)
            lbl_title = QLabel("P300 BCI Speller System")
            lbl_title.setObjectName("title")
            lbl_sub   = QLabel("RSVP · LDA + xDAWN · 6 × 6 Matrix Paradigm")
            lbl_sub.setObjectName("subtitle")
            title_col.addWidget(lbl_title)
            title_col.addWidget(lbl_sub)

            self._lbl_model_status = QLabel("Model: Not Loaded")
            self._lbl_model_status.setObjectName("statusBad")
            self._lbl_model_status.setStyleSheet(
                "color: rgba(255,255,255,0.85); font-weight:600; font-size:12px;")
            self._lbl_model_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            hl.addWidget(icon_lbl)
            hl.addSpacing(10)
            hl.addLayout(title_col)
            hl.addStretch()
            hl.addWidget(self._lbl_model_status)
            return header

        def _make_toolbar(self) -> QWidget:
            bar = QFrame()
            bar.setFrameShape(QFrame.Shape.NoFrame)
            bar.setStyleSheet(f"background:{C_CARD}; border-bottom:1px solid {C_BORDER};")
            bar.setFixedHeight(52)
            hl  = QHBoxLayout(bar)
            hl.setContentsMargins(16, 6, 16, 6)
            hl.setSpacing(10)

            # Dataset label
            self._lbl_dataset = QLabel(f"📁  {self.mat_path.name}")
            self._lbl_dataset.setStyleSheet(f"color:{C_TEXT_LIGHT}; font-size:12px;")

            # Buttons (short labels so they always fit in toolbar)
            self._btn_train  = self._make_btn("⚙ Train",   "btnSecondary")
            self._btn_load   = self._make_btn("📂 Load",    "btnSecondary")
            self._btn_start  = self._make_btn("▶ Start",   "btnPrimary")
            self._btn_stop   = self._make_btn("⏹ Stop",    "btnDanger")
            self._btn_reset  = self._make_btn("↺ Reset",   "btnSecondary")

            self._btn_train.setToolTip("Train LDA + xDAWN pipeline (5-fold CV, ~2 min)")
            self._btn_load.setToolTip("Load existing p300_lda_model.pkl")
            self._btn_start.setToolTip(f"Start {N_REPS}-round flash simulation")
            self._btn_stop.setToolTip("Stop current simulation")
            self._btn_reset.setToolTip("Clear output text and reset grid")

            self._btn_train.clicked.connect(self._on_train)
            self._btn_load.clicked.connect(self._on_load_model)
            self._btn_start.clicked.connect(self._on_start)
            self._btn_stop.clicked.connect(self._on_stop)
            self._btn_reset.clicked.connect(self._on_reset)

            self._btn_stop.setEnabled(False)
            self._btn_start.setEnabled(False)

            hl.addWidget(self._lbl_dataset)
            hl.addStretch()
            hl.addWidget(self._btn_train)
            hl.addWidget(self._btn_load)
            hl.addWidget(self._btn_start)
            hl.addWidget(self._btn_stop)
            hl.addWidget(self._btn_reset)
            return bar

        def _make_left_panel(self) -> QFrame:
            frame = QFrame()
            frame.setObjectName("card")
            vl = QVBoxLayout(frame)
            vl.setContentsMargins(16, 14, 16, 14)
            vl.setSpacing(10)

            hdr = QLabel("Stimulus Grid")
            hdr.setObjectName("sectionHeader")
            vl.addWidget(hdr)

            # Grid layout
            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            grid_layout.setSpacing(6)
            grid_layout.setContentsMargins(0, 0, 0, 0)

            self._grid_btns: dict[tuple[int,int], QPushButton] = {}
            for r, row in enumerate(GRID):
                for c, ch in enumerate(row):
                    btn = QPushButton(ch)
                    btn.setObjectName("gridBtn")
                    btn.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
                    btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    grid_layout.addWidget(btn, r, c)
                    self._grid_btns[(r, c)] = btn

            vl.addWidget(grid_widget, 1)

            # Flash info strip
            self._lbl_flash_info = QLabel("Flash: —")
            self._lbl_flash_info.setStyleSheet(
                f"color:{C_TEXT_LIGHT}; font-size:11px; font-style:italic;")
            vl.addWidget(self._lbl_flash_info)

            # ── BIG START SPELLING BUTTON (always visible below grid) ──
            self._btn_start_big = QPushButton("▶  Start Spelling Simulation")
            self._btn_start_big.setObjectName("btnPrimary")
            self._btn_start_big.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            self._btn_start_big.setFixedHeight(48)
            self._btn_start_big.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_start_big.setEnabled(False)
            self._btn_start_big.setToolTip(f"Run {N_REPS} repetitions × 12 flash groups")
            self._btn_start_big.clicked.connect(self._on_start)
            self._btn_start_big.setStyleSheet(
                f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                f"stop:0 {C_ACCENT},stop:1 {C_ACCENT2}); color:white; border:none;"
                f"border-radius:10px; font-size:14px; font-weight:700; padding:8px 20px; }}"
                f"QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                f"stop:0 #3551DD,stop:1 #6A20A8); }}"
                f"QPushButton:disabled {{ background:#ADB5BD; color:white; }}"
            )
            vl.addWidget(self._btn_start_big)

            return frame

        def _make_right_panel(self) -> QWidget:
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(4, 0, 0, 0)
            vl.setSpacing(10)

            # ── Output text panel ────────────────────────────────────
            out_frame = QFrame()
            out_frame.setObjectName("card")
            out_vl = QVBoxLayout(out_frame)
            out_vl.setContentsMargins(14, 12, 14, 12)
            out_vl.setSpacing(8)

            lbl_out = QLabel("Predicted Output")
            lbl_out.setObjectName("sectionHeader")
            out_vl.addWidget(lbl_out)

            self._txt_output = QTextEdit()
            self._txt_output.setObjectName("outputText")
            self._txt_output.setReadOnly(True)
            self._txt_output.setFixedHeight(80)
            self._txt_output.setPlaceholderText("_")
            out_vl.addWidget(self._txt_output)

            vl.addWidget(out_frame)

            # ── Confidence panel ─────────────────────────────────────
            conf_frame = QFrame()
            conf_frame.setObjectName("card")
            conf_vl = QVBoxLayout(conf_frame)
            conf_vl.setContentsMargins(14, 12, 14, 12)
            conf_vl.setSpacing(8)

            lbl_conf = QLabel("Detection Confidence")
            lbl_conf.setObjectName("sectionHeader")
            conf_vl.addWidget(lbl_conf)

            self._conf_bar = QProgressBar()
            self._conf_bar.setRange(0, 100)
            self._conf_bar.setValue(0)
            self._conf_bar.setFormat("%p%")
            self._conf_bar.setFixedHeight(22)
            conf_vl.addWidget(self._conf_bar)

            self._lbl_conf_detail = QLabel("—")
            self._lbl_conf_detail.setStyleSheet(
                f"color:{C_TEXT_LIGHT}; font-size:11px;")
            conf_vl.addWidget(self._lbl_conf_detail)

            vl.addWidget(conf_frame)

            # ── Last prediction detail ───────────────────────────────
            pred_frame = QFrame()
            pred_frame.setObjectName("card")
            pred_vl = QVBoxLayout(pred_frame)
            pred_vl.setContentsMargins(14, 12, 14, 12)
            pred_vl.setSpacing(6)

            lbl_pred = QLabel("Last Prediction")
            lbl_pred.setObjectName("sectionHeader")
            pred_vl.addWidget(lbl_pred)

            self._lbl_pred_char = QLabel("—")
            self._lbl_pred_char.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
            self._lbl_pred_char.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._lbl_pred_char.setStyleSheet(
                f"color:{C_ACCENT}; letter-spacing:6px;")
            pred_vl.addWidget(self._lbl_pred_char)

            hl_pos = QHBoxLayout()
            self._lbl_pred_row = QLabel("Row: —")
            self._lbl_pred_col = QLabel("Col: —")
            for lbl in (self._lbl_pred_row, self._lbl_pred_col):
                lbl.setStyleSheet(f"color:{C_TEXT_LIGHT}; font-size:11px;")
                hl_pos.addWidget(lbl)
            hl_pos.addStretch()
            pred_vl.addLayout(hl_pos)

            vl.addWidget(pred_frame)

            # ── Log console ──────────────────────────────────────────
            log_frame = QFrame()
            log_frame.setObjectName("card")
            log_vl = QVBoxLayout(log_frame)
            log_vl.setContentsMargins(14, 12, 14, 12)
            log_vl.setSpacing(6)

            hdr_row = QHBoxLayout()
            lbl_log = QLabel("System Log")
            lbl_log.setObjectName("sectionHeader")
            self._btn_clear_log = self._make_btn("Clear", "btnSecondary")
            self._btn_clear_log.setFixedHeight(24)
            self._btn_clear_log.setFixedWidth(60)
            self._btn_clear_log.clicked.connect(lambda: self._txt_log.clear())
            hdr_row.addWidget(lbl_log)
            hdr_row.addStretch()
            hdr_row.addWidget(self._btn_clear_log)
            log_vl.addLayout(hdr_row)

            self._txt_log = QTextEdit()
            self._txt_log.setObjectName("logConsole")
            self._txt_log.setReadOnly(True)
            self._txt_log.setMinimumHeight(120)
            log_vl.addWidget(self._txt_log)
            vl.addWidget(log_frame, 1)

            return container

        # ── Style helpers ──────────────────────────────────────────────────

        def _make_btn(self, text: str, obj_name: str) -> QPushButton:
            btn = QPushButton(text)
            btn.setObjectName(obj_name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFont(QFont("Segoe UI", 11))
            return btn

        def _apply_styles(self) -> None:
            self.setStyleSheet(STYLESHEET)

        # ── Model management ───────────────────────────────────────────────

        def _check_model_ready(self) -> None:
            """Auto-load model if pkl exists on startup."""
            if self.model_path.exists():
                try:
                    from model_engine import load_model
                    self._pipeline = load_model(self.model_path)
                    self._load_prep_data()
                    self._set_model_status("ready")
                    self._log(f"✅ Model auto-loaded: {self.model_path.name}")
                except Exception as exc:
                    self._log(f"⚠️  Could not auto-load model: {exc}")
                    self._set_model_status("error")
            else:
                self._log("ℹ️  No model found. Click '⚙ Train Model' to train.")
                self._set_model_status("missing")

        def _load_prep_data(self) -> None:
            """Load and cache preprocessed EEG data for simulation."""
            if self._prep_result is not None:
                return
            try:
                from prep_engine import load_and_preprocess
                self._log("Loading EEG data for simulation...")
                self._prep_result = load_and_preprocess(self.mat_path, verbose=False)
                self._log(f"✅ Data ready: {self._prep_result.summary()}")
                self._btn_start.setEnabled(True)
                self._btn_start_big.setEnabled(True)
            except Exception as exc:
                self._log(f"❌ Data load failed: {exc}")

        def _set_model_status(self, state: str) -> None:
            icons = {"ready": "✅", "missing": "⚪", "training": "⏳", "error": "❌"}
            texts = {
                "ready":    "Model: Ready ✅",
                "missing":  "Model: Not Found",
                "training": "Model: Training…",
                "error":    "Model: Error ❌",
            }
            self._lbl_model_status.setText(texts.get(state, state))
            if state == "ready":
                self._btn_start.setEnabled(True)
                self._btn_start_big.setEnabled(True)

        # ── Button handlers ────────────────────────────────────────────────

        def _on_train(self) -> None:
            if not self.mat_path.exists():
                self._log(f"❌ s01.mat not found: {self.mat_path}")
                return
            self._log("⚙  Starting model training (5-fold CV, LDA + xDAWN)…")
            self._set_model_status("training")
            self._btn_train.setEnabled(False)
            self._btn_start.setEnabled(False)
            self._btn_start_big.setEnabled(False)

            self._train_worker = TrainWorker(self.mat_path, self.model_path)
            self._train_worker.finished.connect(self._on_train_done)
            self._train_worker.error.connect(self._on_train_error)
            self._train_worker.log_msg.connect(self._log)
            self._train_worker.start()

        def _on_train_done(self, pipeline, metrics: dict) -> None:
            self._pipeline = pipeline
            self._set_model_status("ready")
            self._btn_train.setEnabled(True)
            auc = metrics.get("mean_auc", 0.0)
            status = "✅ Target met!" if auc >= 0.80 else "⚠️ Below 0.80"
            self._log(f"✅ Training complete — AUC-ROC: {auc:.4f}  {status}")
            self._log(f"   Model saved → {self.model_path.name}")
            self._load_prep_data()

        def _on_train_error(self, msg: str) -> None:
            self._log(f"❌ Training error: {msg}")
            self._set_model_status("error")
            self._btn_train.setEnabled(True)

        def _on_load_model(self) -> None:
            path_str, _ = QFileDialog.getOpenFileName(
                self, "Load P300 Model", str(_ROOT), "Pickle files (*.pkl)"
            )
            if not path_str:
                return
            try:
                from model_engine import load_model
                self._pipeline = load_model(path_str)
                self.model_path = Path(path_str)
                self._set_model_status("ready")
                self._log(f"✅ Model loaded: {Path(path_str).name}")
                self._load_prep_data()
            except Exception as exc:
                self._log(f"❌ Load failed: {exc}")

        def _on_start(self) -> None:
            if self._pipeline is None:
                self._log("⚠️  No model loaded. Train or load a model first.")
                return
            if self._prep_result is None:
                self._log("⚠️  EEG data not loaded. Wait for data to finish loading.")
                return

            self._running = True
            self._btn_start.setEnabled(False)
            self._btn_start_big.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._btn_train.setEnabled(False)
            self._btn_reset.setEnabled(False)

            self._decoder.reset()
            self._flash_sequence = list(range(N_FLASH_GROUPS))  # 0–11
            np.random.shuffle(self._flash_sequence)
            self._flash_idx = 0
            self._rep_idx   = 0

            self._log(f"\n▶  Starting {N_REPS}-repetition character round…")
            self._status_bar.showMessage(f"Simulation running — {N_REPS} repetitions × 12 groups")
            self._flash_timer.start(FLASH_MS + ISI_MS)

        def _on_stop(self) -> None:
            self._running = False
            self._flash_timer.stop()
            self._clear_flash()
            self._btn_start.setEnabled(True)
            self._btn_start_big.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._btn_train.setEnabled(True)
            self._btn_reset.setEnabled(True)
            self._log("⏹  Simulation stopped.")
            self._status_bar.showMessage("Simulation stopped.")

        def _on_reset(self) -> None:
            self._on_stop()
            self._output_text = ""
            self._txt_output.clear()
            self._conf_bar.setValue(0)
            self._lbl_conf_detail.setText("—")
            self._lbl_pred_char.setText("—")
            self._lbl_pred_row.setText("Row: —")
            self._lbl_pred_col.setText("Col: —")
            self._lbl_flash_info.setText("Flash: —")
            self._decoder.reset()
            self._clear_flash()
            self._log("↺  Reset complete.")
            self._status_bar.showMessage("Reset — Ready.")

        # ── Flash engine ───────────────────────────────────────────────────

        def _on_flash_tick(self) -> None:
            """Timer callback — advances the flash sequence one step."""
            if not self._running:
                return

            # Un-highlight previous group
            self._clear_flash()

            total_groups = len(self._flash_sequence)
            if self._flash_idx >= total_groups:
                # End of one repetition
                self._rep_idx  += 1
                self._flash_idx = 0
                np.random.shuffle(self._flash_sequence)   # re-randomise order

                if self._rep_idx >= N_REPS:
                    # All reps done → decode
                    self._finish_round()
                    return

            # Current group to flash
            gid = self._flash_sequence[self._flash_idx]
            self._current_flash_group = gid
            self._highlight_group(gid)

            group_type = "Row" if gid < N_ROWS else "Col"
            group_num  = gid + 1 if gid < N_ROWS else gid - N_ROWS + 1
            self._lbl_flash_info.setText(
                f"Flash: {group_type} {group_num}  |  Rep {self._rep_idx+1}/{N_REPS}")

            # Run inference on a real EEG epoch
            score = self._infer_flash(gid)
            self._decoder.add_flash(gid, score)

            self._flash_idx += 1

        def _highlight_group(self, gid: int) -> None:
            """Highlight all buttons in flash group ``gid``."""
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    btn = self._grid_btns[(r, c)]
                    in_group = (gid < N_ROWS and r == gid) or \
                               (gid >= N_ROWS and c == gid - N_ROWS)
                    if in_group:
                        btn.setStyleSheet(
                            f"background-color:{C_FLASH}; color:{C_FLASH_TEXT}; "
                            f"border:2px solid #F0C040; border-radius:8px; "
                            f"font-size:15px; font-weight:700;"
                        )
                    else:
                        btn.setStyleSheet("")  # reset to default

        def _clear_flash(self) -> None:
            """Remove all flash highlighting."""
            for btn in self._grid_btns.values():
                btn.setStyleSheet("")

        def _highlight_predicted(self, r: int, c: int) -> None:
            """Highlight the predicted character in mint green."""
            self._clear_flash()
            btn = self._grid_btns[(r, c)]
            btn.setStyleSheet(
                f"background-color:{C_PREDICTED}; color:{C_PREDICTED_T}; "
                f"border:2px solid {C_SUCCESS}; border-radius:8px; "
                f"font-size:15px; font-weight:700;"
            )

        def _infer_flash(self, gid: int) -> float:
            """Sample a real EEG epoch and run model inference."""
            try:
                from model_engine import predict_proba
                data = self._prep_result

                # Pick a random epoch — weighted toward targets for target groups
                # We don't know the true target group in simulation mode, so
                # we use uniform random sampling (realistic inference scenario)
                idx = np.random.randint(0, len(data.labels))
                epoch = data.epochs_full[idx : idx + 1]   # (1, C, T)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = float(predict_proba(self._pipeline, epoch)[0])
                return score
            except Exception as exc:
                logger.warning("Inference error: %s", exc)
                return 0.0

        def _finish_round(self) -> None:
            """Called when all repetitions are done — decode and display."""
            self._flash_timer.stop()
            self._clear_flash()

            pred_char = self._decoder.decode()
            conf      = self._decoder.confidence()
            pred_r, pred_c = self._decoder.predicted_position()

            # Highlight predicted cell
            self._highlight_predicted(pred_r, pred_c)

            # Update UI
            self._output_text += pred_char
            self._txt_output.setText(self._output_text)
            self._conf_bar.setValue(int(conf * 100))
            self._lbl_conf_detail.setText(
                f"Row score: {self._decoder.row_scores()[pred_r]:.3f}  |  "
                f"Col score: {self._decoder.col_scores()[pred_c]:.3f}"
            )
            self._lbl_pred_char.setText(pred_char)
            self._lbl_pred_row.setText(f"Row: {pred_r + 1}")
            self._lbl_pred_col.setText(f"Col: {pred_c + 1}")

            row_sc = np.round(self._decoder.row_scores(), 3)
            col_sc = np.round(self._decoder.col_scores(), 3)
            self._log(f"✅  Predicted: '{pred_char}'  "
                      f"(row={pred_r+1}, col={pred_c+1})  "
                      f"Confidence={conf:.3f}")
            self._log(f"   Row scores: {row_sc}")
            self._log(f"   Col scores: {col_sc}")

            self._status_bar.showMessage(
                f"Predicted: '{pred_char}'  |  Confidence: {conf:.1%}  "
                f"| Output: '{self._output_text}'"
            )

            # Re-enable controls after 1.5 s
            QTimer.singleShot(1500, self._after_round)

        def _after_round(self) -> None:
            self._running = False
            self._btn_start.setEnabled(True)
            self._btn_start_big.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._btn_train.setEnabled(True)
            self._btn_reset.setEnabled(True)
            self._decoder.reset()

        # ── Logging ────────────────────────────────────────────────────────

        def _log(self, msg: str) -> None:
            ts = datetime.now().strftime("%H:%M:%S")
            self._txt_log.append(f"<span style='color:#6C757D'>[{ts}]</span> {msg}")
            # Auto-scroll
            sb = self._txt_log.verticalScrollBar()
            sb.setValue(sb.maximum())


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def launch_app(
    mat_path:   Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> None:
    """Create and launch the P300 Speller GUI application."""
    if not _QT_AVAILABLE:
        print("ERROR: PyQt6 (or PyQt5) is not installed.")
        print("Install with:  pip install PyQt6")
        sys.exit(1)

    mat   = mat_path   or (_ROOT / "data_explore" / "s01.mat")
    model = model_path or (_ROOT / "p300_lda_model.pkl")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("P300 BCI Speller")
    app.setApplicationVersion("1.0")

    window = P300SpellerApp(mat_path=mat, model_path=model)
    window.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P300 BCI Speller — PyQt6 GUI",
    )
    parser.add_argument("--mat",   default=None, metavar="PATH",
                        help="Path to s01.mat")
    parser.add_argument("--model", default=None, metavar="PATH",
                        help="Path to p300_lda_model.pkl")
    args = parser.parse_args()

    mat   = Path(args.mat)   if args.mat   else None
    model = Path(args.model) if args.model else None
    launch_app(mat_path=mat, model_path=model)
