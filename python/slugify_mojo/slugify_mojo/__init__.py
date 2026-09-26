"""slugify-mojo: a drop-in faster replacement for the `python-slugify` package.

Same function, same call shape, byte-identical slugs — powered by a Mojo
kernel where the platform supports it (macOS arm64, Linux x86_64), with a
pure-Python fallback everywhere else (including Windows).

    from slugify_mojo import slugify, slugify_column

    slugify("Déjà Vu — Café naïve!")            # 'deja-vu-cafe-naive'
    slugify_column(["Hello World", "Mосква"])   # ['hello-world', 'moskva']

Set SLUGIFY_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from slugify_mojo._native import backend_info, native_available
from slugify_mojo.core import (
    DEFAULT_SEPARATOR,
    Algorithm,
    Backend,
    ReplacementStage,
    backend,
    slugify,
    slugify_column,
    smart_truncate,
)

__version__ = "0.1.5"  # x-release-please-version
__all__ = [
    "slugify",
    "slugify_column",
    "smart_truncate",
    "backend",
    "backend_info",
    "native_available",
    "Backend",
    "ReplacementStage",
    "Algorithm",
    "DEFAULT_SEPARATOR",
    "__version__",
]
