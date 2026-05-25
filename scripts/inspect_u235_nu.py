# SPDX-License-Identifier: MIT
"""Inspect U-235 ν̄(E) tables and their interpolation codes."""
import h5py
import numpy as np

PATH = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-viii.1-hdf5/neutron/U235.h5"

INTERP_NAMES = {
    1: "histogram",
    2: "lin-lin",
    3: "lin-log",
    4: "log-lin",
    5: "log-log",
    6: "charged-particle",
}


def show_interp(dset, label):
    attrs = dict(dset.attrs)
    interp = attrs.get("interpolation", [])
    breakpoints = attrs.get("breakpoints", [])
    arr = np.array(dset)
    print(f"  {label}: shape={arr.shape}")
    print(f"    breakpoints: {breakpoints}")
    interp_arr = np.atleast_1d(interp).flatten()
    print(f"    interpolation codes (flat): {interp_arr}")
    for code in interp_arr:
        name = INTERP_NAMES.get(int(code), f"unknown({code})")
        print(f"      -> {name}")
    if arr.ndim > 1 and arr.shape[0] >= 2:
        print(f"    x[0..5]: {arr[0, :5]}")
        print(f"    y[0..5]: {arr[1, :5]}")
        print(f"    x[-3:]:  {arr[0, -3:]}")
        print(f"    y[-3:]:  {arr[1, -3:]}")
    elif arr.ndim == 1:
        print(f"    data[:5]: {arr[:5]}")
        print(f"    data[-3:]: {arr[-3:]}")


with h5py.File(PATH, "r") as f:
    nuc_name = list(f.keys())[0]
    r = f[nuc_name]["reactions"]["reaction_018"]  # MT=18 fission
    print(f"=== {nuc_name} MT=18 fission ν̄ inspection ===\n")

    for p in sorted(r.keys()):
        if not p.startswith("product_"):
            continue
        attrs = dict(r[p].attrs)
        particle = attrs.get("particle", b"")
        if hasattr(particle, "decode"):
            particle = particle.decode()
        if particle != "neutron":
            continue
        emission = attrs.get("emission_mode", b"")
        if hasattr(emission, "decode"):
            emission = emission.decode()
        print(f"--- {p}: emission={emission} ---")
        if "yield" in r[p]:
            yield_ds = r[p]["yield"]
            yield_attrs = dict(yield_ds.attrs)
            print(f"    yield type: {yield_attrs.get('type', 'unknown')}")
            print(f"    yield attrs: {yield_attrs}")
            if isinstance(yield_ds, h5py.Dataset):
                show_interp(yield_ds, "yield")
            else:
                print(f"    yield subkeys: {list(yield_ds.keys())}")
                for k in yield_ds.keys():
                    item = yield_ds[k]
                    if isinstance(item, h5py.Dataset):
                        show_interp(item, f"yield/{k}")
        print()
