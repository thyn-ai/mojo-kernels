"""metpy-mojo: fast MetPy-compatible parcel CAPE/CIN for standard soundings.

Powered by a clean-room Mojo kernel where the platform supports it
(macOS arm64, Linux x86_64), with a vendored pure-Python fallback
everywhere else (including Windows).

    from metpy_mojo import cape_cin, cape_cin_grid

    cape, cin = cape_cin(p, T, Td)                      # surface parcel
    cape, cin = cape_cin(p, T, Td, which="most_unstable")
    cape2d, cin2d = cape_cin_grid(p_lev, T3d, Td3d)     # (nlev, y, x) fields

Units: hPa (strictly decreasing) and K; CAPE/CIN in J/kg (CIN <= 0).
Set METPY_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from metpy_mojo._native import backend_info, native_available
from metpy_mojo.core import ParcelDiagnostics, cape_cin, cape_cin_grid, parcel_diagnostics

__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "ParcelDiagnostics",
    "backend_info",
    "cape_cin",
    "cape_cin_grid",
    "native_available",
    "parcel_diagnostics",
    "__version__",
]
