#!/usr/bin/env python3
"""cclib-mojo quickstart: electron density of water (STO-3G) on a 3-D grid.

Self-contained: no cclib install and no Mojo toolchain required. The basis
data below is STO-3G for O and H (Hehre, Stewart, Pople, J. Chem. Phys. 51,
2657 (1969); values as shipped in PyQuante 1.6.5's basis library).

Run after `pip install cclib-mojo`:

    python quickstart.py

It prints the active backend (native Mojo kernel or NumPy fallback) and a
deterministic checksum of the density grid. With
CCLIB_MOJO_DISABLE_NATIVE=1 the fallback is forced; the checksum is the
same on either backend.
"""

import numpy as np

from cclib_mojo import backend_info, density_on_grid, wavefunction_on_grid

STO3G_H = [
    ("S", [(3.42525091, 0.15432897), (0.62391373, 0.53532814), (0.1688554, 0.44463454)]),
]
STO3G_O = [
    ("S", [(130.70932, 0.15432897), (23.808861, 0.53532814), (6.4436083, 0.44463454)]),
    ("S", [(5.0331513, -0.09996723), (1.1695961, 0.39951283), (0.380389, 0.70011547)]),
    ("P", [(5.0331513, 0.15591627), (1.1695961, 0.60768372), (0.380389, 0.39195739)]),
]

# Water, Angstrom (cclib convention).
gbasis = [STO3G_O, STO3G_H, STO3G_H]
atomcoords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])

# Fixed MO coefficients (7 basis functions: O 1s, 2s, 2px/y/z + H 1s x2).
# In a real workflow these come from ccdata.mocoeffs after parsing a logfile.
mocoeffs = np.array(
    [
        [-0.99421130, 0.23012347, 0.00103717, 0.10483291, 0.12991833, -0.13279467, -0.13279467],
        [-0.21441663, -0.89127605, -0.00290144, -0.29327618, -0.36335339, 0.37100119, 0.37100119],
        [0.0, 0.0, 0.61872341, 0.0, 0.0, 0.42073104, -0.42073104],
    ]
)

origin = (-3.0, -3.0, -3.0)
step = (0.25, 0.25, 0.25)
shape = (25, 25, 25)

info = backend_info()
backend = "native Mojo kernel" if info["native_available"] else "NumPy fallback"
print(f"cclib-mojo quickstart — backend: {backend}")

# One molecular orbital's amplitude (cclib's wavefunction() equivalent):
psi = wavefunction_on_grid(gbasis, atomcoords, mocoeffs[0], origin, step, shape)
print(f"wavefunction MO 0: shape {psi.shape}, psi(center) = {psi[12, 12, 12]:.8f}")

# Total density of the three MOs, sum_mo |psi_mo|^2
# (cclib's electrondensity_spin(); multiply by 2 for a closed shell):
rho = density_on_grid(gbasis, atomcoords, mocoeffs, origin, step, shape)
print(f"density of 3 MOs: shape {rho.shape}, integral ~= {rho.sum() * step[0]**3:.6f} A^-3 grid units")

# Deterministic checksum so the forced-fallback run can be compared exactly.
checksum = float(np.sum(rho * np.arange(rho.size).reshape(rho.shape)))
print(f"density checksum: {checksum:.10e}")
