"""ruptures-mojo: a drop-in faster replacement for the `ruptures` package.

Same breakpoint sets, same call shapes — powered by a Mojo kernel where the
platform supports it (macOS arm64, Linux x86_64), with a vendored pure-Python
fallback everywhere else (including Windows).

    import ruptures_mojo

    bkps = ruptures_mojo.detect(signal, "dynp", n_bkps=3)       # == ruptures.Dynp
    bkps = ruptures_mojo.detect(signal, "pelt", pen=10)         # == ruptures.Pelt
    bkps = ruptures_mojo.detect(signal, "binseg", n_bkps=2)     # == ruptures.Binseg

Set RUPTURES_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from ruptures_mojo._native import backend_info, native_available
from ruptures_mojo._reference import BadSegmentationParameters
from ruptures_mojo.core import detect, last_backend

__version__ = "0.1.4"  # x-release-please-version
__all__ = [
    "BadSegmentationParameters",
    "backend_info",
    "detect",
    "last_backend",
    "native_available",
    "__version__",
]
