"""difflib-mojo: difflib.SequenceMatcher and get_close_matches, accelerated.

A drop-in replacement for the hot parts of the standard library's difflib
(CPython 3.12 semantics), powered by a Mojo kernel where the platform
supports it (macOS arm64, Linux x86_64), with a vendored pure-Python engine
everywhere else (including Windows). Outputs — ratios, matching blocks,
opcodes, close-match ranking and tie order — are identical to the
reference's on both backends:

    import difflib_mojo

    difflib_mojo.SequenceMatcher(None, "abcd", "bcde").ratio()   # 0.75
    difflib_mojo.get_close_matches("appel", ["ape", "apple", "peach"])
    # ['apple', 'ape']

Batch entry points for the one-word-vs-many-candidates shape:

    difflib_mojo.get_close_matches_batch(words, candidates, n=5, cutoff=0.7,
                                         key=str.casefold)
    difflib_mojo.ratio_batch([(a, b) for a, b in pairs])

Set DIFFLIB_MOJO_DISABLE_NATIVE=1 to force the pure-Python engine.
"""

from difflib_mojo._native import backend_info, native_available
from difflib_mojo.core import (
    IS_CHARACTER_JUNK,
    IS_LINE_JUNK,
    Match,
    SequenceMatcher,
    get_close_matches,
    get_close_matches_batch,
    ratio_batch,
)

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "IS_CHARACTER_JUNK",
    "IS_LINE_JUNK",
    "Match",
    "SequenceMatcher",
    "backend_info",
    "get_close_matches",
    "get_close_matches_batch",
    "native_available",
    "ratio_batch",
    "__version__",
]
