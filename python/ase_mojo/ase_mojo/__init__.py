"""ase-mojo: fast periodic neighbor lists with an ASE-compatible API.

A drop-in accelerator for ``ase.neighborlist.primitive_neighbor_list`` /
``neighbor_list``, powered by a Mojo kernel where the platform supports it
(macOS arm64, Linux x86_64), with a vendored NumPy fallback everywhere else
(including Windows). ASE is not required to use this package (``Atoms``
objects are duck-typed), and ASE is never imported:

    import numpy as np
    from ase_mojo import primitive_neighbor_list, neighbor_list

    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])
    i, j, S = primitive_neighbor_list("ijS", [True, True, True], cell, pos, 1.5)

Set ASE_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from ase_mojo._native import NativeUnavailable, backend_info, native_available
from ase_mojo.core import neighbor_list, primitive_neighbor_list

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "NativeUnavailable",
    "backend_info",
    "native_available",
    "neighbor_list",
    "primitive_neighbor_list",
    "__version__",
]
