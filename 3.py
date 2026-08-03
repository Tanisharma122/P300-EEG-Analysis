"""
Deep exploration script for MATLAB v7.3 (.mat = HDF5) P300 EEG files, e.g. s01.mat
from the "EEG Dataset for RSVP and P300 Speller BCI" (Figshare).

What this does beyond a basic key/shape dump:
  1. Recursively walks EVERY group/dataset and prints shape, dtype, attrs.
  2. Reads and previews actual values for small/likely-metadata datasets
     (labels, event markers, channel names, sampling rate, epoch indices).
  3. Tries to auto-classify each dataset as:
       - Raw/epoched EEG signal (large 2D/3D float array)
       - Label / target-nontarget vector (small int/bool array)
       - Channel names / montage info (string / object refs)
       - Sampling rate / scalar metadata
  4. Prints basic stats (min/max/mean/std, NaN count) for signal-like arrays
     so you can sanity-check units (µV vs V) and check for bad channels.
  5. Handles MATLAB's HDF5 quirks: object references, transposed dims
     (MATLAB is column-major, so arrays often come in reversed vs. how
     you'd index them in numpy/MNE), and char arrays stored as uint16 codes.

Run:  python explore_p300_dataset.py s01.mat
"""

import sys
import h5py
import numpy as np


def decode_matlab_string(dataset):
    """MATLAB HDF5 often stores strings as arrays of uint16 char codes,
    or as arrays of object references to such arrays. Try to decode."""
    try:
        data = dataset[()]
        if data.dtype.kind in ("u", "i"):
            # Likely char codes
            flat = data.flatten()
            try:
                s = "".join(chr(c) for c in flat if 0 < c < 0x110000)
                if s.isprintable() or s.strip():
                    return s
            except Exception:
                pass
        return None
    except Exception:
        return None


def summarize_numeric(data):
    """Return a compact stats string for a numeric numpy array."""
    try:
        arr = np.asarray(data)
        if arr.size == 0:
            return "empty array"
        flat = arr.astype(np.float64).flatten()
        nan_count = np.isnan(flat).sum()
        finite = flat[~np.isnan(flat)]
        if finite.size == 0:
            return f"all-NaN array, n={flat.size}"
        return (f"min={finite.min():.4g}  max={finite.max():.4g}  "
                f"mean={finite.mean():.4g}  std={finite.std():.4g}  "
                f"NaNs={nan_count}/{flat.size}")
    except Exception as e:
        return f"(could not compute stats: {e})"


def classify_dataset(name, shape, dtype):
    """Heuristic guess at what role this dataset plays in a P300 dataset."""
    lname = name.lower()
    ndim = len(shape)

    if any(k in lname for k in ("label", "target", "class", "marker",
                                 "event", "stim", "y")):
        return "LIKELY: labels / event markers / stimulus codes"
    if any(k in lname for k in ("chan", "montage", "electrode")):
        return "LIKELY: channel names / montage info"
    if any(k in lname for k in ("fs", "srate", "sampling", "freq")):
        return "LIKELY: sampling rate / frequency metadata"
    if any(k in lname for k in ("data", "eeg", "signal", "x", "epoch",
                                 "trial")):
        if ndim >= 2 and dtype.kind == "f":
            return "LIKELY: raw or epoched EEG signal array"
    if ndim == 0 or (ndim == 1 and shape[0] == 1):
        return "LIKELY: scalar metadata (e.g. a single parameter)"
    return "UNCLASSIFIED — inspect manually"


def explore_file(path, max_preview_elems=20, small_dataset_threshold=200):
    print("#" * 78)
    print(f"# DEEP EXPLORATION: {path}")
    print("#" * 78)

    with h5py.File(path, "r") as f:

        # --- Top level ---
        print("\n[Top-level keys]")
        for k in f.keys():
            print(" -", k, "|", type(f[k]))

        print("\n[Root file attributes]")
        for k, v in f.attrs.items():
            print(f"  {k} = {v}")

        # --- Full recursive walk ---
        def visit(name, obj):
            print("\n" + "=" * 78)
            print("Path :", name)
            print("Type :", type(obj).__name__)

            # Attributes on this node
            if len(obj.attrs) > 0:
                print("Attributes:")
                for ak, av in obj.attrs.items():
                    print(f"    {ak} = {av}")

            if isinstance(obj, h5py.Group):
                print(f"(Group with {len(obj.keys())} children: "
                      f"{list(obj.keys())[:10]}{'...' if len(obj.keys())>10 else ''})")
                return

            # It's a Dataset
            shape = obj.shape
            dtype = obj.dtype
            size = obj.size
            print("Shape :", shape)
            print("Dtype :", dtype)
            print("Size (elements):", size)
            print("Role guess:", classify_dataset(name, shape, dtype))

            # h5py object-reference datasets (common in MATLAB cell arrays)
            if dtype == h5py.special_dtype(ref=h5py.Reference) or \
               (dtype.kind == "O"):
                print("(Contains HDF5 object references — likely a MATLAB "
                      "cell array; dereference individually if needed.)")
                return

            # Try string decode for small char-like datasets
            if size <= small_dataset_threshold and dtype.kind in ("u", "i", "S"):
                decoded = decode_matlab_string(obj)
                if decoded:
                    print("Decoded string preview:", repr(decoded)[:200])

            # Numeric preview + stats
            if dtype.kind in ("f", "i", "u"):
                try:
                    data = obj[()]
                    flat_preview = np.asarray(data).flatten()[:max_preview_elems]
                    print("Value preview:", flat_preview)
                    if size > 1:
                        print("Stats:", summarize_numeric(data))
                    if size <= small_dataset_threshold:
                        print("Unique values:", np.unique(np.asarray(data)))
                except Exception as e:
                    print(f"(Could not read values: {e})")

        f.visititems(visit)

        # --- Targeted second pass: look for common P300 dataset names ---
        print("\n" + "#" * 78)
        print("# TARGETED CHECK: common P300 field names")
        print("#" * 78)
        common_names = ["data", "X", "y", "labels", "Label", "target",
                         "Target", "nontarget", "fs", "srate", "Fs",
                         "chan_names", "channels", "epochs", "trial",
                         "event", "stimulus", "markers"]

        def find_matches(name, obj):
            base = name.split("/")[-1]
            for c in common_names:
                if c.lower() == base.lower():
                    print(f"  MATCH: '/{name}'  ->  shape={getattr(obj, 'shape', None)}, "
                          f"dtype={getattr(obj, 'dtype', None)}")

        f.visititems(find_matches)

        print("\nDone. Remember: MATLAB HDF5 arrays are typically stored "
              "TRANSPOSED relative to how MATLAB displays them (row-major "
              "vs column-major), so a MATLAB [channels x samples] array "
              "often reads in h5py/numpy as [samples x channels] or "
              "reversed dims — verify with .shape before assuming axis order.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "s01.mat"
    explore_file(path)