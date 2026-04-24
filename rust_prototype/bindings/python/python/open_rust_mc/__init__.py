"""open-rust-mc: Python bindings for a pure-Rust Monte Carlo neutron transport engine.

This is a thin convenience wrapper around the native `_core` extension
compiled from `bindings/python/src/lib.rs`. All validation and
simulation logic lives on the Rust side — Python is only for scripting.

Example
-------
    >>> from open_rust_mc import Scene, Material, Sphere, Settings, run_eigenvalue
    >>> fuel = (Material("HEU", temperature=294.0)
    ...     .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
    ...     .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.43)
    ...     .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.49))
    >>> scene = (Scene("data/endfb-vii.1-hdf5/neutron")
    ...     .add_material("heu", fuel)
    ...     .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
    ...     .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
    ...     .add_cell("outside", region="+boundary"))
    >>> result = run_eigenvalue(scene, Settings(batches=50, inactive=10, particles=5000))
    >>> print(result.k_eff, result.k_sigma)
"""

from ._core import (
    # Surfaces
    Sphere,
    ZCylinder,
    XCylinder,
    YCylinder,
    XPlane,
    YPlane,
    ZPlane,
    # Entities
    Material,
    Settings,
    Scene,
    # Results
    EigenvalueResult,
    # Top-level functions
    run_eigenvalue,
)

__all__ = [
    "Sphere",
    "ZCylinder",
    "XCylinder",
    "YCylinder",
    "XPlane",
    "YPlane",
    "ZPlane",
    "Material",
    "Settings",
    "Scene",
    "EigenvalueResult",
    "run_eigenvalue",
]

__version__ = "0.1.0"
