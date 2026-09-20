"""langdetect-mojo: a drop-in faster replacement for the `langdetect` package.

Same `detect()` / `detect_langs()` strings — powered by a Mojo kernel where
the platform supports it (macOS arm64, Linux x86_64), with a vendored
pure-Python fallback everywhere else (including Windows). The 55-language
profiles are read at runtime from the installed `langdetect` package.

    import langdetect_mojo

    langdetect_mojo.detect("War doesn't show who's right, just who's left.")
    # 'en'
    langdetect_mojo.detect_langs("War doesn't show who's right, just who's left.")
    # [en:0.9999966368178837]

Set LANGDETECT_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from langdetect_mojo._native import backend_info, native_available
from langdetect_mojo.core import (
    LangDetectError,
    Language,
    backend,
    detect,
    detect_langs,
    get_seed,
    set_seed,
)

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "LangDetectError",
    "Language",
    "backend",
    "backend_info",
    "detect",
    "detect_langs",
    "get_seed",
    "native_available",
    "set_seed",
    "__version__",
]
