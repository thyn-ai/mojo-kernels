"""pykalman-mojo: a drop-in faster replacement for `pykalman`'s core surface.

Same KalmanFilter.filter()/.smooth() call shapes, same outputs — powered by
a Mojo kernel where the platform supports it (macOS arm64, Linux x86_64),
with a vendored pure-NumPy fallback everywhere else (including Windows).

    import numpy as np
    from pykalman_mojo import KalmanFilter

    kf = KalmanFilter(transition_matrices=[[1, 0.1], [0, 1]],
                      observation_matrices=[[1, 0]])
    filtered_means, filtered_covs = kf.filter(observations)
    smoothed_means, smoothed_covs = kf.smooth(observations)

    # or one-call:
    from pykalman_mojo import filter, smooth
    filtered_means, filtered_covs = filter(observations, **params)

Set PYKALMAN_MOJO_DISABLE_NATIVE=1 to force the pure-NumPy fallback.
"""

from pykalman_mojo._native import backend_info, native_available
from pykalman_mojo.core import KalmanFilter, filter, smooth

__version__ = "0.1.4"  # x-release-please-version
__all__ = [
    "KalmanFilter",
    "filter",
    "smooth",
    "backend_info",
    "native_available",
    "__version__",
]
