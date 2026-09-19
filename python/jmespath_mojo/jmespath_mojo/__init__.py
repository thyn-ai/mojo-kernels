"""jmespath-mojo: a drop-in faster replacement for the `jmespath` package.

Same call shape, same results, same errors — powered by a clean-room Mojo
kernel where the platform supports it (macOS arm64, Linux x86_64), with a
vendored pure-Python fallback everywhere else (including Windows).

    import jmespath_mojo

    jmespath_mojo.search("reservations[].instances[?state=='running'].id", doc)
    jmespath_mojo.search("sort_by(people, &age)[*].name", doc)

Set JMESPATH_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback;
inspect the active backend with jmespath_mojo.backend_info().
"""

from jmespath_mojo._native import backend_info, native_available
from jmespath_mojo.core import search

__version__ = "0.1.0"

__all__ = [
    "search",
    "backend_info",
    "native_available",
    "__version__",
]
