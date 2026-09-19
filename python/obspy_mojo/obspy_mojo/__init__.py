"""obspy-mojo: drop-in faster Konno-Ohmachi smoothing for ObsPy users.

Same function, same call shape, same results as
``obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing`` — powered by a
Mojo kernel where the platform supports it (macOS arm64, Linux x86_64), with
a vendored pure-NumPy fallback everywhere else (including Windows).

    import numpy as np
    from obspy_mojo import konno_ohmachi_smoothing

    smoothed = konno_ohmachi_smoothing(spectra, frequencies)   # b=40 default
    # identical (within documented tolerance) to
    # obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing(
    #     spectra, frequencies, bandwidth=40, count=1,
    #     enforce_no_matrix=False, max_memory_usage=512, normalize=False)

Set OBSPY_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from obspy_mojo._native import backend_info, native_available
from obspy_mojo.core import konno_ohmachi_smoothing

__version__ = "0.1.0"

__all__ = [
    "konno_ohmachi_smoothing",
    "backend_info",
    "native_available",
    "__version__",
]
