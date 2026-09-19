"""Quickstart smoke for metpy-mojo (also the CI end-user wheel test).

Run after `pip install metpy-mojo`:

    python quickstart.py

Computes surface-based and most-unstable CAPE/CIN for a classic analytic
sounding and a small gridded field, and prints the active backend. In CI
this script is copied to a temp dir first, so sys.path[0] cannot shadow
the installed wheel with the in-repo package.
"""

import numpy as np

import metpy_mojo
from metpy_mojo import cape_cin, cape_cin_grid, parcel_diagnostics

# Classic midlatitude convective sounding (analytic, deterministic).
p = np.linspace(1000.0, 100.0, 91)
z = -np.log(p / 1000.0) * 7.5
T = np.maximum(300.0 - 6.5 * z, 222.0)
Td = np.maximum(294.0 - 5.0 * z, 200.0)

cape, cin = cape_cin(p, T, Td)
diag = parcel_diagnostics(p, T, Td)
cape_mu, cin_mu = cape_cin(p, T, Td, which="most_unstable")

print(f"surface-based:  CAPE = {cape:.1f} J/kg, CIN = {cin:.1f} J/kg")
print(f"  LCL {diag.lcl_pressure:.0f} hPa, LFC {diag.lfc_pressure:.0f} hPa, "
      f"EL {diag.el_pressure:.0f} hPa")
print(f"most-unstable:  CAPE = {cape_mu:.1f} J/kg, CIN = {cin_mu:.1f} J/kg")

# Gridded field: (nlev, 3, 4) with a latitudinal moisture gradient.
T3 = np.broadcast_to(T[:, None, None], (91, 3, 4)).copy()
Td3 = np.broadcast_to(Td[:, None, None], (91, 3, 4)).copy()
Td3 = np.minimum(T3 - 0.5, Td3 + np.linspace(0.0, 6.0, 4)[None, None, :])
cape2d, cin2d = cape_cin_grid(p, T3, Td3)
print(f"grid (3x4):     CAPE row 0 = {np.round(cape2d[0], 1)} J/kg")
print(f"cape checksum: {float(np.sum(cape2d)):.6f}")

info = metpy_mojo.backend_info()
backend = "native" if info["native_available"] else "fallback (pure Python)"
print(f"metpy-mojo {metpy_mojo.__version__} backend: {backend}")
if info["native_source"]:
    print(f"  kernel: {info['native_source']}")
