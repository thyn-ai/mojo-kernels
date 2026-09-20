"""cclib-mojo: fast electron-density and wavefunction-on-grid evaluation.

A drop-in accelerator for cclib's ``cclib.method.volume`` path
(wavefunction / electron density on 3-D grids), powered by a Mojo kernel
where the platform supports it (macOS arm64, Linux x86_64), with a vendored
NumPy fallback everywhere else (including Windows). Also usable standalone:

    from cclib_mojo import density_on_grid, wavefunction_on_grid

    # gbasis/atomcoords/mocoeffs as parsed by cclib (Angstrom geometry):
    psi = wavefunction_on_grid(gbasis, atomcoords, mocoeffs,
                               origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                               shape=(51, 51, 51))
    rho = density_on_grid(gbasis, atomcoords, mocoeffs[:nocc],
                          origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                          shape=(51, 51, 51))   # sum_mo |psi_mo|^2

For cclib-API-shaped helpers see ``cclib_mojo.cclib_integration``.
Set CCLIB_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from cclib_mojo import cclib_integration
from cclib_mojo._basis import BasisError
from cclib_mojo._native import backend_info, native_available
from cclib_mojo.core import GridError, density_on_grid, wavefunction_on_grid

__version__ = "0.1.1"  # x-release-please-version
__all__ = [
    "BasisError",
    "GridError",
    "backend_info",
    "cclib_integration",
    "density_on_grid",
    "native_available",
    "wavefunction_on_grid",
    "__version__",
]
