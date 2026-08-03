import h5py

def explore(name, obj):
    print("="*70)
    print("Name :", name)
    print("Type :", type(obj))

    if isinstance(obj, h5py.Dataset):
        print("Shape :", obj.shape)
        print("Datatype :", obj.dtype)

with h5py.File("s01.mat", "r") as f:
    f.visititems(explore)