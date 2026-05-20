# SPDX-License-Identifier: MIT
"""Probe U-233's reaction_018 layout to identify which ENDF law(s) the
prompt-neutron outgoing-energy distribution uses, and confirm whether
the engine's tabular-law reader is dropping it on the floor.

Usage:  python scripts/u233_layout_probe.py [data_dir]
"""
import sys, h5py, pathlib

data_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "data/endfb-vii.1-hdf5/neutron")

def dump_product(p, indent="    "):
    print(f"{indent}attrs: {dict(p.attrs)}")
    for name in p.keys():
        sub = p[name]
        print(f"{indent}- {name}  attrs={dict(sub.attrs)}")
        if isinstance(sub, h5py.Group):
            for sub_name in sub.keys():
                ssub = sub[sub_name]
                print(f"{indent}    - {sub_name}  attrs={dict(ssub.attrs)}")
                if isinstance(ssub, h5py.Group):
                    for sss_name in ssub.keys():
                        sssub = ssub[sss_name]
                        kind = "Group" if isinstance(sssub, h5py.Group) else f"Dataset shape={sssub.shape} dtype={sssub.dtype}"
                        print(f"{indent}        - {sss_name}  {kind}  attrs={dict(sssub.attrs)}")

for nuclide in ("U233", "U235"):
    f = h5py.File(data_dir / f"{nuclide}.h5", "r")
    print("="*72)
    print(f" {nuclide}: reactions/reaction_018")
    print("="*72)
    rxn = f[nuclide]["reactions"]["reaction_018"]
    print(f"  rxn attrs: {dict(rxn.attrs)}")
    for prod_name in sorted(rxn.keys()):
        if not prod_name.startswith("product_"):
            continue
        prod = rxn[prod_name]
        is_neutron = prod.attrs.get("particle", b"")
        if isinstance(is_neutron, bytes):
            is_neutron = is_neutron.decode()
        emission = prod.attrs.get("emission_mode", b"")
        if isinstance(emission, bytes):
            emission = emission.decode()
        print(f"\n  {prod_name}: particle={is_neutron!r} emission_mode={emission!r}")
        if is_neutron != "neutron":
            continue
        dump_product(prod)
    f.close()
