# SPDX-License-Identifier: MIT
"""Dump the structure of an OpenMC photon HDF5 file for reader scaffolding."""
import sys
import h5py

path = sys.argv[1] if len(sys.argv) > 1 else "data/endfb-vii.1-hdf5/photon/C.h5"


def walk(name, obj):
    if isinstance(obj, h5py.Group):
        attrs = dict(obj.attrs)
        print(f"G  {name}  attrs={attrs}")
    else:
        print(f"D  {name}  shape={obj.shape}  dtype={obj.dtype}")


with h5py.File(path, "r") as f:
    print("ROOT attrs:", dict(f.attrs))
    f.visititems(walk)
