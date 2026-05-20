# SPDX-License-Identifier: MIT
"""PWR pin cell gamma-heating via the Python API.

Python counterpart to `src/bin/pwr_gamma_heating.rs`. Builds the same
UO₂ / He-gap / Zr-4 / H₂O-with-S(α,β) geometry, runs a neutron
k-eigenvalue phase that collects photon source events at each
capture/fission/inelastic site, then runs photon transport with
track-integrated CSDA electrons (+brems +MS +BB) through the same
CSG. Reports per-cell gamma-deposition fractions.

Reference result from the Rust binary and the OpenMC cross-code run:
  fuel 84.12 %, gap 0.00 %, clad 9.81 %, water 5.72 %

Usage:
  python rust_prototype/bindings/python/examples/pwr_gamma_heating.py \\
      data/endfb-vii.1-hdf5/neutron \\
      --photon-data data/endfb-vii.1-hdf5/photon
"""
from __future__ import annotations

import argparse
import sys

from open_rust_mc import (
    Material,
    PhotonMaterial,
    Scene,
    Settings,
    Sphere,
    XPlane,
    YPlane,
    ZCylinder,
    ZPlane,
    run_gamma_heating,
    uranium_oxide_material,
    water_material,
    zircaloy4_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="PWR pin gamma-heating via Python")
    parser.add_argument("data_dir", help="Neutron HDF5 data directory")
    parser.add_argument("--photon-data", required=True, help="Photon (per-element) HDF5 directory")
    parser.add_argument("--batches", type=int, default=150)
    parser.add_argument("--inactive", type=int, default=50)
    parser.add_argument("--particles", type=int, default=50_000)
    parser.add_argument("--photons", type=int, default=200_000)
    args = parser.parse_args()

    # ── Materials (neutron side) ──────────────────────────────────────
    # The helpers compute atom densities from macro density and
    # stoichiometry, and attach S(α,β) for water's hydrogen.
    fuel = uranium_oxide_material(
        "UO2 3.1%", density_g_per_cm3=10.4, enrichment=0.031,
        temperature=900.0, temp_idx=3,
    )
    clad = zircaloy4_material(
        density_g_per_cm3=6.55, temperature=600.0, temp_idx=2,
    )
    water = water_material(
        density_g_per_cm3=0.74, temperature=600.0, temp_idx=2,
        thermal_file="c_H_in_H2O.h5",
    )

    # ── Geometry ──────────────────────────────────────────────────────
    FUEL_OR, CLAD_IR, CLAD_OR, PITCH = 0.4096, 0.4180, 0.4750, 1.2600
    half = PITCH / 2.0

    scene = (
        Scene(args.data_dir)
        .set_photon_data_dir(args.photon_data)
        .add_material("fuel", fuel)
        .add_material("clad", clad)
        .add_material("water", water)
        # Cylinders and reflective lattice planes.
        .add_surface("fuel_or", ZCylinder(r=FUEL_OR))
        .add_surface("clad_ir", ZCylinder(r=CLAD_IR))
        .add_surface("clad_or", ZCylinder(r=CLAD_OR))
        .add_surface("xmin", XPlane(x0=-half, bc="reflective"))
        .add_surface("xmax", XPlane(x0=+half, bc="reflective"))
        .add_surface("ymin", YPlane(y0=-half, bc="reflective"))
        .add_surface("ymax", YPlane(y0=+half, bc="reflective"))
        .add_surface("zmin", ZPlane(z0=-half, bc="reflective"))
        .add_surface("zmax", ZPlane(z0=+half, bc="reflective"))
        .add_cell("fuel", "-fuel_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="fuel", temperature=900.0)
        .add_cell("gap", "+fuel_or -clad_ir +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill=None, temperature=600.0)
        .add_cell("clad", "+clad_ir -clad_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="clad", temperature=600.0)
        .add_cell("water", "+clad_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="water", temperature=600.0)
    )

    # ── Photon materials (same geometry, photon XS per element) ──────
    # Atom densities per element, computed to match the Rust binary's
    # UO2_MOL_DENSITY / ZR_ATOM_DENSITY / H2O_MOL_DENSITY constants.
    UO2_MOL_DENSITY = 2.319e-2
    ZR_ATOM_DENSITY = 4.324e-2
    H2O_MOL_DENSITY = 2.474e-2
    scene.add_photon_material(
        "fuel",
        PhotonMaterial(density_g_per_cm3=10.4)
            .add_element("U.h5", UO2_MOL_DENSITY)
            .add_element("O.h5", 2.0 * UO2_MOL_DENSITY),
    ).add_photon_material(
        "clad",
        PhotonMaterial(density_g_per_cm3=6.55).add_element("Zr.h5", ZR_ATOM_DENSITY),
    ).add_photon_material(
        "water",
        PhotonMaterial(density_g_per_cm3=0.74)
            .add_element("H.h5", 2.0 * H2O_MOL_DENSITY)
            .add_element("O.h5", H2O_MOL_DENSITY),
    )
    # "gap" has no photon material → treated as void for photons,
    # same as the neutron side.

    settings = Settings(
        batches=args.batches,
        inactive=args.inactive,
        particles=args.particles,
        seed=1,
    )

    print(f"Running PWR gamma-heating: neutrons {settings}, photons {args.photons}")
    result = run_gamma_heating(
        scene,
        neutron_settings=settings,
        n_photon_histories=args.photons,
    )

    print(f"\nk_eff    = {result.k_eff:.5f} ± {result.k_sigma:.5f}")
    print(f"neutron runtime: {result.neutron_runtime_seconds:.1f} s")
    print(f"photon  runtime: {result.photon_runtime_seconds:.1f} s "
          f"({result.photon_events} source events)")
    print(f"brems: {result.brems_photons_emitted} gamma, "
          f"{result.brems_energy_ev:.3e} eV "
          f"({100 * result.brems_energy_ev / max(result.total_source_energy_ev, 1):.2f} %)")
    print()
    print("-- Energy deposition by region --")
    print(f"  {'region':<6} {'deposited (eV)':>16} {'fraction':>10}")
    for name, ev, frac in zip(result.cell_names, result.deposition_ev, result.deposition_fraction):
        print(f"  {name:<6} {ev:>16.3e} {100 * frac:>9.3f} %")
    print(f"  {'escape':<6} {result.escaped_energy_ev:>16.3e} "
          f"{100 * result.escaped_energy_ev / result.total_source_energy_ev:>9.3f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
