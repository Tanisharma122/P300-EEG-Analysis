import h5py
import numpy as np

path = "s01.mat"

with h5py.File(path, "r") as f:
    srate = f["/RSVP/srate"][()]
    print("Sampling rate:", srate.flatten())

    markers = f["/RSVP/markers_target"][()].flatten()
    print("\nmarkers_target shape:", markers.shape)
    vals, counts = np.unique(markers, return_counts=True)
    print("Unique values in markers_target:", vals)
    print("Counts per value:", counts)

    nbTrials = f["/RSVP/nbTrials"][()].flatten()
    nbTrials_target = f["/RSVP/nbTrials_target"][()].flatten()
    nbTrials_nontarget = f["/RSVP/nbTrials_nontarget"][()].flatten()
    print("\nnbTrials:", nbTrials)
    print("nbTrials_target:", nbTrials_target)
    print("nbTrials_nontarget:", nbTrials_nontarget)

    # Decode channel labels (MATLAB cellstr -> HDF5 object references)
    print("\nChannel labels:")
    label_refs = f["/RSVP/chanlocs/labels"]
    decoded_labels = []
    for i in range(label_refs.shape[0]):
        ref = label_refs[i, 0]
        char_codes = f[ref][()].flatten()
        name = "".join(chr(c) for c in char_codes)
        decoded_labels.append(name)
    print(decoded_labels)

    # target / keyboard_response arrays (40,1) each
    target_arr = f["/RSVP/target"][()].flatten()
    kbd_arr = f["/RSVP/keyboard_response"][()].flatten()
    print("\ntarget (first 40):", target_arr)
    print("keyboard_response (first 40):", kbd_arr)

    # Where are the nonzero markers located (sample indices)?
    nonzero_idx = np.nonzero(markers)[0]
    print(f"\nNumber of nonzero marker samples: {len(nonzero_idx)}")
    print("First 20 nonzero marker sample indices:", nonzero_idx[:20])
    print("First 20 nonzero marker values:", markers[nonzero_idx][:20])
    if len(nonzero_idx) > 1:
        isis = np.diff(nonzero_idx)
        print("Inter-stimulus interval (samples) - first 20:", isis[:20])
        print("ISI stats: min=%d max=%d mean=%.1f" % (isis.min(), isis.max(), isis.mean()))