"""dynesty-mojo: fast nested sampling with a Mojo bounding/proposal kernel.

A dynesty-shaped nested sampler whose bounding/proposal machinery —
single-ellipsoid fitting and uniform-in-ellipsoid batched proposals, the
per-iteration numerical work that dominates sampler overhead for cheap
likelihoods — runs in a Mojo kernel where the platform supports it (macOS
arm64, Linux x86_64), with a vendored NumPy fallback everywhere else
(including Windows). User likelihoods and prior transforms stay ordinary
Python callables, exactly as in dynesty:

    import numpy as np
    from dynesty_mojo import sample

    ndim = 5
    def loglike(v):
        return -0.5 * float(v @ v)
    def prior_transform(u):
        return 10.0 * u - 5.0

    res = sample(loglike, prior_transform, ndim, nlive=300, seed=42)
    print(res.logz, res.logz_err, res.weights @ res.samples)

Set DYNESTY_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from dynesty_mojo._native import backend_info, native_available
from dynesty_mojo.sampler import NSResults, SamplerError, sample

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "NSResults",
    "SamplerError",
    "backend_info",
    "native_available",
    "sample",
    "__version__",
]
