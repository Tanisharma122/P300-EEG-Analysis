"""
decoder_engine.py — Phase 3: P300 Matrix Intersection & Character Decoding.

Implements the classic P300 speller grid (6 rows × 6 columns = 36 characters)
and the decision algorithm that aggregates classifier output probabilities
across flash repetitions to identify the intended character.

Grid layout
-----------
    Col→  1    2    3    4    5    6
Row ↓
  1       A    B    C    D    E    F
  2       G    H    I    J    K    L
  3       M    N    O    P    Q    R
  4       S    T    U    V    W    X
  5       Y    Z    1    2    3    4
  6       5    6    7    8    9    0

Flash groups
------------
  Row groups   :  IDs 0–5   (each group = 6 characters in the same row)
  Column groups:  IDs 6–11  (each group = 6 characters in the same column)

Decision algorithm (Phase-3 spec)
----------------------------------
  1. For each flash repetition, the classifier returns P(target) ∈ [0,1].
  2. Probabilities are **accumulated** per group across all repetitions.
  3. Row_max  = argmax  of accumulated row  scores  → predicted row
  4. Col_max  = argmax  of accumulated col  scores  → predicted column
  5. Character = GRID[Row_max][Col_max]

Usage
-----
    from decoder_engine import P300Decoder

    decoder = P300Decoder()
    decoder.reset()

    # Feed flash scores one repetition set at a time
    for rep in range(n_reps):
        for group_id in range(12):          # 6 rows + 6 cols
            score = model_proba[...]        # float in [0,1]
            decoder.add_flash(group_id, score)

    predicted_char = decoder.decode()       # e.g. 'P'
    row, col       = decoder.predicted_position()

Running standalone (demo / smoke-test)
---------------------------------------
    python decoder_engine.py
"""

from __future__ import annotations

import logging
import random
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_BCI  = _ROOT / "bci_pipeline"
for _p in (str(_ROOT), str(_BCI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("decoder_engine")

# ---------------------------------------------------------------------------
# 6×6 P300 Speller Grid
# ---------------------------------------------------------------------------

#: Standard P300 speller character matrix (row-major, 0-indexed internally)
GRID: list[list[str]] = [
    ["A", "B", "C", "D", "E", "F"],   # Row 0 (display: Row 1)
    ["G", "H", "I", "J", "K", "L"],   # Row 1
    ["M", "N", "O", "P", "Q", "R"],   # Row 2
    ["S", "T", "U", "V", "W", "X"],   # Row 3
    ["Y", "Z", "1", "2", "3", "4"],   # Row 4
    ["5", "6", "7", "8", "9", "0"],   # Row 5
]

N_ROWS  = 6
N_COLS  = 6
N_FLASH_GROUPS = N_ROWS + N_COLS   # 12 total groups

#: Flat character list (left-to-right, top-to-bottom)
CHARS: list[str] = [ch for row in GRID for ch in row]

# Lookup: character → (row_idx, col_idx)  (0-indexed)
CHAR_TO_POS: dict[str, tuple[int, int]] = {
    ch: (r, c)
    for r, row in enumerate(GRID)
    for c, ch  in enumerate(row)
}

# Flash-group membership
# Group IDs 0–5  → Rows 0–5  (characters in the same row)
# Group IDs 6–11 → Cols 0–5  (characters in the same column)

def row_group(row_idx: int) -> int:
    """Return the flash-group ID for row ``row_idx`` (0-indexed)."""
    return row_idx                  # 0–5

def col_group(col_idx: int) -> int:
    """Return the flash-group ID for column ``col_idx`` (0-indexed)."""
    return N_ROWS + col_idx         # 6–11

def group_characters(group_id: int) -> list[str]:
    """Return the 6 characters that belong to flash-group ``group_id``."""
    if group_id < N_ROWS:
        return GRID[group_id][:]          # one full row
    else:
        col_idx = group_id - N_ROWS
        return [GRID[r][col_idx] for r in range(N_ROWS)]   # one full column

def char_groups(char: str) -> tuple[int, int]:
    """Return (row_group_id, col_group_id) for a character."""
    r, c = CHAR_TO_POS[char]
    return row_group(r), col_group(c)


# ---------------------------------------------------------------------------
# P300Decoder class
# ---------------------------------------------------------------------------

class P300Decoder:
    """Stateful P300 speller character decoder.

    Aggregates classifier P(target) scores across flash repetitions and
    identifies the target character by row × column intersection.

    Parameters
    ----------
    n_rows : int
        Number of rows in the grid (default 6).
    n_cols : int
        Number of columns in the grid (default 6).

    Examples
    --------
    >>> d = P300Decoder()
    >>> d.reset()
    >>> d.add_flash(group_id=0, score=0.85)   # Row 1 flashed, high P300
    >>> d.add_flash(group_id=7, score=0.72)   # Col 2 flashed, high P300
    >>> d.decode()
    'B'
    """

    def __init__(self, n_rows: int = N_ROWS, n_cols: int = N_COLS) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated scores — call before each new character round."""
        self._row_scores  = np.zeros(self.n_rows, dtype=np.float64)
        self._col_scores  = np.zeros(self.n_cols, dtype=np.float64)
        self._row_counts  = np.zeros(self.n_rows, dtype=np.int32)
        self._col_counts  = np.zeros(self.n_cols, dtype=np.int32)
        self._n_flashes   = 0
        logger.debug("Decoder reset.")

    # ------------------------------------------------------------------
    # Score accumulation
    # ------------------------------------------------------------------

    def add_flash(self, group_id: int, score: float) -> None:
        """Register a single flash event with its P300 classifier score.

        Parameters
        ----------
        group_id : int
            Flash group ID (0–5 = rows, 6–11 = columns).
        score : float
            Classifier output: P(target) ∈ [0.0, 1.0].
        """
        score = float(np.clip(score, 0.0, 1.0))
        if group_id < self.n_rows:
            self._row_scores[group_id] += score
            self._row_counts[group_id] += 1
        else:
            c = group_id - self.n_rows
            if 0 <= c < self.n_cols:
                self._col_scores[c] += score
                self._col_counts[c] += 1
        self._n_flashes += 1

    def add_flash_batch(
        self,
        group_ids: list[int] | np.ndarray,
        scores:    list[float] | np.ndarray,
    ) -> None:
        """Register a batch of flash events at once.

        Parameters
        ----------
        group_ids : array-like of int  Shape ``(N,)``
        scores    : array-like of float  Shape ``(N,)`` — P(target)
        """
        for gid, sc in zip(group_ids, scores):
            self.add_flash(int(gid), float(sc))

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def _mean_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mean row and column scores (averaged over repetitions)."""
        row_mean = np.where(
            self._row_counts > 0,
            self._row_scores / self._row_counts,
            0.0,
        )
        col_mean = np.where(
            self._col_counts > 0,
            self._col_scores / self._col_counts,
            0.0,
        )
        return row_mean, col_mean

    def predicted_position(self) -> tuple[int, int]:
        """Return the predicted (row_idx, col_idx) — 0-indexed.

        Uses argmax of mean accumulated scores for rows and columns.
        """
        row_mean, col_mean = self._mean_scores()
        pred_row = int(np.argmax(row_mean))
        pred_col = int(np.argmax(col_mean))
        return pred_row, pred_col

    def decode(self) -> str:
        """Return the predicted character.

        Algorithm
        ---------
        Row_max = argmax P(target | Row_i)   across all row flashes
        Col_max = argmax P(target | Col_j)   across all column flashes
        Character = GRID[Row_max][Col_max]
        """
        row_idx, col_idx = self.predicted_position()
        char = GRID[row_idx][col_idx]
        logger.debug(
            "Decode: row_scores=%s col_scores=%s → row=%d col=%d char='%s'",
            np.round(self._row_scores, 3),
            np.round(self._col_scores, 3),
            row_idx, col_idx, char,
        )
        return char

    def confidence(self) -> float:
        """Return a confidence score in [0, 1] for the current prediction.

        Computed as the product of the normalised row and column max scores.
        This is an indicator of how clearly the P300 peak stands out
        compared to distractor groups.
        """
        row_mean, col_mean = self._mean_scores()
        row_max = float(np.max(row_mean)) if row_mean.max() > 0 else 0.0
        col_max = float(np.max(col_mean)) if col_mean.max() > 0 else 0.0

        # Normalise by the mean of all scores
        row_total = float(np.sum(row_mean)) or 1.0
        col_total = float(np.sum(col_mean)) or 1.0

        row_conf = row_max / row_total
        col_conf = col_max / col_total
        # Geometric mean of row and column confidence
        return float(np.sqrt(row_conf * col_conf))

    @property
    def n_flashes(self) -> int:
        """Total number of flash events registered so far."""
        return self._n_flashes

    def row_scores(self) -> np.ndarray:
        """Mean accumulated score per row (shape ``(n_rows,)``)."""
        return self._mean_scores()[0]

    def col_scores(self) -> np.ndarray:
        """Mean accumulated score per column (shape ``(n_cols,)``)."""
        return self._mean_scores()[1]


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def simulate_character_round(
    target_char:  str,
    pipeline:     object,
    epochs_full:  np.ndarray,
    labels:       np.ndarray,
    n_reps:       int = 5,
    rng:          Optional[np.random.Generator] = None,
    verbose:      bool = True,
) -> tuple[str, float]:
    """Simulate a full character-prediction round using real EEG epochs.

    Maps real target/non-target epochs from the RSVP dataset onto the
    matrix flash paradigm:
    - For the row and column containing ``target_char``, we sample from
      actual **target** epochs (P300 present).
    - For all other rows and columns, we sample from **non-target** epochs.

    This is a valid simulation because the P300 signal in RSVP paradigms
    is functionally equivalent to the matrix paradigm — it reflects the
    brain's response to the attended stimulus.

    Parameters
    ----------
    target_char : str
        The character we assume the user is attending to.
    pipeline : sklearn pipeline
        Fitted classifier (from :func:`model_engine.load_model`).
    epochs_full : np.ndarray
        All preprocessed epochs, shape ``(N, C, T)``.
    labels : np.ndarray
        Labels array, shape ``(N,)`` — 1=target, 2=non-target.
    n_reps : int
        Number of flash repetitions per character round (default 5).
    rng : np.random.Generator | None
        Optional random number generator for reproducibility.
    verbose : bool
        Print round-by-round progress.

    Returns
    -------
    predicted_char : str
        The character predicted by the decoder.
    conf : float
        Confidence in [0, 1].
    """
    from model_engine import predict_proba

    if rng is None:
        rng = np.random.default_rng()

    if target_char.upper() not in CHAR_TO_POS:
        raise ValueError(f"Character '{target_char}' not in speller grid.")
    target_char = target_char.upper()
    tgt_row, tgt_col = CHAR_TO_POS[target_char]

    target_group_ids = {row_group(tgt_row), col_group(tgt_col)}

    # Epoch index pools
    tgt_idx = np.where(labels == 1)[0]
    nt_idx  = np.where(labels == 2)[0]

    if len(tgt_idx) == 0:
        raise ValueError("No target epochs available for simulation.")
    if len(nt_idx) == 0:
        raise ValueError("No non-target epochs available for simulation.")

    decoder = P300Decoder()
    decoder.reset()

    if verbose:
        print(f"\n[decoder] Simulating character '{target_char}' "
              f"(row={tgt_row+1}, col={tgt_col+1}) | {n_reps} reps × 12 groups")

    group_ids = list(range(N_FLASH_GROUPS))

    for rep in range(n_reps):
        rng.shuffle(group_ids)   # randomise flash order each rep
        for gid in group_ids:
            is_target_group = gid in target_group_ids
            # Sample real epoch for this flash
            if is_target_group:
                idx = rng.choice(tgt_idx)
            else:
                idx = rng.choice(nt_idx)

            epoch_3d = epochs_full[idx : idx + 1]   # (1, C, T)
            score    = float(predict_proba(pipeline, epoch_3d)[0])
            decoder.add_flash(gid, score)

    predicted_char = decoder.decode()
    conf           = decoder.confidence()

    if verbose:
        row_sc = decoder.row_scores()
        col_sc = decoder.col_scores()
        pred_r, pred_c = decoder.predicted_position()
        print(f"  Row scores : {np.round(row_sc, 3)}")
        print(f"  Col scores : {np.round(col_sc, 3)}")
        print(f"  → Predicted row {pred_r+1}, col {pred_c+1} = '{predicted_char}'")
        correct = "✅" if predicted_char == target_char else "❌"
        print(f"  {correct} Target='{target_char}'  Predicted='{predicted_char}'  "
              f"Confidence={conf:.3f}")

    return predicted_char, conf


def batch_simulate(
    pipeline:      object,
    epochs_full:   np.ndarray,
    labels:        np.ndarray,
    chars:         str = "HELLO",
    n_reps:        int = 7,
    seed:          int = 42,
    verbose:       bool = True,
) -> list[dict]:
    """Simulate spelling a word and return per-character results.

    Parameters
    ----------
    pipeline : sklearn pipeline
    epochs_full : np.ndarray  shape ``(N, C, T)``
    labels : np.ndarray  shape ``(N,)``
    chars : str  Word to spell (e.g. "HELLO").
    n_reps : int  Flash repetitions per character.
    seed : int  RNG seed.
    verbose : bool

    Returns
    -------
    list[dict]
        One dict per character with keys: ``target``, ``predicted``,
        ``correct``, ``confidence``.
    """
    rng = np.random.default_rng(seed)
    results = []
    for ch in chars.upper():
        if ch not in CHAR_TO_POS:
            logger.warning("Character '%s' not in grid — skipping.", ch)
            continue
        pred, conf = simulate_character_round(
            ch, pipeline, epochs_full, labels, n_reps=n_reps, rng=rng, verbose=verbose
        )
        results.append({
            "target":     ch,
            "predicted":  pred,
            "correct":    pred == ch,
            "confidence": conf,
        })

    if verbose and results:
        correct = sum(r["correct"] for r in results)
        predicted_word = "".join(r["predicted"] for r in results)
        print(f"\n[decoder] ═══════════════════════════════════════")
        print(f"  Target word    : {''.join(chars.upper())}")
        print(f"  Predicted word : {predicted_word}")
        print(f"  Accuracy       : {correct}/{len(results)} chars "
              f"({100*correct/len(results):.0f}%)")
        print(f"[decoder] ═══════════════════════════════════════\n")

    return results


# ---------------------------------------------------------------------------
# CLI / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P300 Phase-3: Matrix decoding logic demo",
    )
    parser.add_argument("--mat",   default=None, metavar="PATH",
                        help="Path to s01.mat")
    parser.add_argument("--model", default=None, metavar="PATH",
                        help="Path to p300_lda_model.pkl")
    parser.add_argument("--word",  default="P300", metavar="WORD",
                        help="Word to simulate spelling (default: P300)")
    parser.add_argument("--reps",  type=int, default=7,
                        help="Flash repetitions per character (default: 7)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n[decoder_engine] Grid layout:")
    print("  " + "─" * 35)
    for r, row in enumerate(GRID):
        print(f"  Row {r+1}: {' '.join(row)}")
    print("  " + "─" * 35)

    # Smoke-test decoder logic (no model required)
    print("\n[decoder_engine] Smoke-test: simulating 'P' with mock scores...")
    dec = P300Decoder()
    dec.reset()
    tgt_r, tgt_c = CHAR_TO_POS["P"]
    for gid in range(N_FLASH_GROUPS):
        if gid == row_group(tgt_r) or gid == col_group(tgt_c):
            dec.add_flash(gid, 0.90)   # strong P300 for target groups
        else:
            dec.add_flash(gid, 0.15)   # weak for non-targets
    result = dec.decode()
    print(f"  Predicted: '{result}' (expected 'P') — "
          f"{'✅' if result == 'P' else '❌'}")
    print(f"  Confidence: {dec.confidence():.3f}")

    # Full simulation if model exists
    mat_path   = Path(args.mat)   if args.mat   else _ROOT / "data_explore" / "s01.mat"
    model_path = Path(args.model) if args.model else _ROOT / "p300_lda_model.pkl"

    if model_path.exists() and mat_path.exists():
        print(f"\n[decoder_engine] Full simulation: '{args.word}'")
        from model_engine import load_model
        from prep_engine  import load_and_preprocess

        result = load_and_preprocess(mat_path, verbose=False)
        pipe   = load_model(model_path)

        batch_simulate(
            pipe, result.epochs_full, result.labels,
            chars=args.word, n_reps=args.reps, verbose=True,
        )
    else:
        print(f"\n[decoder_engine] Skipping full simulation "
              f"(model={model_path.exists()}, mat={mat_path.exists()})")
        print("  Run 'python model_engine.py' first to train the model.")
