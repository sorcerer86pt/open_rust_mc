# SPDX-License-Identifier: MIT
"""Inspect U-235 HDF5 file to see what interpolation Watt a(E) / b(E) use.

OpenMC's Tabulated1D honors the per-region `interpolation` attribute
on the HDF5 dataset (1=histogram, 2=lin-lin, 3=lin-log, 4=log-lin,
5=log-log). Our Rust `WattLaw::lookup_lin_lin` is hardcoded to lin-lin.
If U-235's Watt a/b use anything other than 2, our engine is biased.
"""
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


def walk_distribution(grp):
    """Recursively walk a distribution group, printing dataset interp."""
    attrs = dict(grp.attrs)
    dtype = attrs.get("type", b"")
    if hasattr(dtype, "decode"):
        dtype = dtype.decode()
    print(f"    type = {dtype}")
    print(f"    attrs = {attrs}")
    print(f"    subkeys = {list(grp.keys())}")
    for sub in grp.keys():
        item = grp[sub]
        if isinstance(item, h5py.Dataset):
            print(f"\n    > dataset {sub}:")
            show_interp(item, sub)
        else:
            print(f"\n    > subgroup {sub}: keys={list(item.keys())}")
            for ssub in item.keys():
                sitem = item[ssub]
                if isinstance(sitem, h5py.Dataset):
                    print(f"      > {sub}/{ssub}:")
                    show_interp(sitem, f"{sub}/{ssub}")


def show_interp(dset, label):
    attrs = dict(dset.attrs)
    interp = attrs.get("interpolation", np.array([]))
    breakpoints = attrs.get("breakpoints", np.array([]))
    arr = np.array(dset)
    print(f"  {label}: shape={arr.shape}")
    print(f"    breakpoints: {breakpoints}")
    print(f"    interpolation codes: {interp}")
    for code in np.atleast_1d(interp):
        name = INTERP_NAMES.get(int(code), f"unknown({code})")
        print(f"      → {name}")
    if arr.size > 0:
        print(f"    x[0..3]: {arr[0, :3] if arr.ndim > 1 else arr[:3]}")
        if arr.ndim > 1 and arr.shape[0] > 1:
            print(f"    y[0..3]: {arr[1, :3]}")


with h5py.File(PATH, "r") as f:
    nuc_name = list(f.keys())[0]
    r = f[nuc_name]["reactions"]["reaction_018"]  # MT=18 fission
    print(f"=== {nuc_name} MT=18 fission ===")
    print(f"reaction attrs: {dict(r.attrs)}")

    products = [k for k in r.keys() if k.startswith("product_")]
    print(f"products: {products}\n")

    for p in products:
        attrs = dict(r[p].attrs)
        particle = attrs.get("particle", b"")
        if hasattr(particle, "decode"):
            particle = particle.decode()
        if particle != "neutron":
            print(f"skipping {p} ({particle})")
            continue
        print(f"--- {p} (neutron product) ---")
        print(f"    product subkeys: {list(r[p].keys())}")
        print(f"    product attrs: {attrs}")
        # OpenMC HDF5 stores per-product distributions as distribution_0,
        # distribution_1, ... — one per region. n_distribution in attrs
        # tells us how many.
        dist_keys = [k for k in r[p].keys() if k.startswith("distribution_")]
        if not dist_keys:
            continue
        for dk in dist_keys:
            print(f"\n  >>> {dk} <<<")
            walk_distribution(r[p][dk])
        continue  # skip the rest of the loop
        dist = r[p]["distribution_0"]
        for dkey in dist.keys():
            dgrp = dist[dkey]
            dattrs = dict(dgrp.attrs)
            dtype = dattrs.get("type", b"")
            if hasattr(dtype, "decode"):
                dtype = dtype.decode()
            print(f"\n  distribution/{dkey}: type={dtype}")
            print(f"    attrs: {dattrs}")
            print(f"    subkeys: {list(dgrp.keys())}")
            # Walk subkeys looking for a/b (Watt) or other tabulated funcs
            for sub in dgrp.keys():
                ds = dgrp[sub]
                if isinstance(ds, h5py.Dataset):
                    print(f"\n    > subdataset {sub}:")
                    show_interp(ds, sub)
                else:
                    print(f"\n    > subgroup {sub}: keys={list(ds.keys())}")
                    for ssub in ds.keys():
                        sds = ds[ssub]
                        if isinstance(sds, h5py.Dataset):
                            print(f"\n      > {sub}/{ssub}:")
                            show_interp(sds, f"{sub}/{ssub}")
