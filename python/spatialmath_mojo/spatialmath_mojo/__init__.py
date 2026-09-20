"""spatialmath-mojo: fast batched SE(3)/SO(3) pose algebra.

A batch accelerator for the pose operations of spatialmath-python (Peter
Corke's robotics toolbox): SE(3)/SO(3) composition, inverses, and point
transforms over whole pose batches in one call, powered by a Mojo kernel
where the platform supports it (macOS arm64, Linux x86_64), with a vendored
NumPy fallback everywhere else (including Windows):

    import numpy as np
    import spatialmath_mojo as smm

    A = np.stack([...])   # (n, 4, 4) float64 SE(3) poses
    B = np.stack([...])
    smm.compose(A, B)            # (n, 4, 4), per-pose A[i] @ B[i]
    smm.inverse(A)               # (n, 4, 4), per-pose trinv
    smm.transform(A, points)     # (n, m, 3), h2e(T[i] @ e2h(p[j]))

Set SPATIALMATH_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from spatialmath_mojo._native import backend_info, native_available
from spatialmath_mojo.core import compose, inverse, transform

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "backend_info",
    "compose",
    "inverse",
    "native_available",
    "transform",
    "__version__",
]
