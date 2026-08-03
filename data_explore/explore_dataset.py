import h5py
import numpy as np

file_path = "s01.mat"

print("="*60)
print("Opening MATLAB v7.3 File")
print("="*60)

with h5py.File(file_path, 'r') as f:

    print("\nAvailable Keys:\n")

    for key in f.keys():
        print(key)