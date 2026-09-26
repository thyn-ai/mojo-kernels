"""mistune-mojo: a drop-in faster replacement for `mistune.markdown`.

Same call, same HTML — powered by a Mojo kernel where the platform supports
it (macOS arm64, Linux x86_64), with a vendored pure-Python engine everywhere
else (including Windows).

    import mistune_mojo

    mistune_mojo.markdown("# Hello *world*")
    # '<h1>Hello <em>world</em></h1>\n'   (same bytes as mistune.markdown)

Set MISTUNE_MOJO_DISABLE_NATIVE=1 to force the pure-Python engine.
"""

from mistune_mojo._native import backend_info, native_available
from mistune_mojo.core import markdown

__version__ = "0.1.5"  # x-release-please-version
__all__ = [
    "markdown",
    "backend_info",
    "native_available",
    "__version__",
]
