import h5py

path = "s01.mat"

with h5py.File(path, "r") as f:
    print("TOP-LEVEL KEYS:", list(f.keys()))
    print("=" * 60)

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"/{name}")
            print(f"    shape = {obj.shape}")
            print(f"    dtype = {obj.dtype}")
        else:
            print(f"/{name}  (GROUP, contains: {list(obj.keys())})")

    f.visititems(show)