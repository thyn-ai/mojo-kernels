"""oletools-mojo: a drop-in faster replacement for oletools'
`olevba.decompress_stream` — the MS-OVBA VBA decompression codec used to
extract macro source from Office documents.

Same call shape, same bytes out, same exception types — powered by a
clean-room Mojo kernel where the platform supports it (macOS arm64, Linux
x86_64), with a vendored pure-Python fallback everywhere else (including
Windows).

    from oletools_mojo import decompress_stream

    source = decompress_stream(compressed_container)   # bytes in, bytes out

To accelerate oletools' own VBA extraction (`olevba.VBA_Parser`) without
forking it, see `oletools_mojo.olevba_integration`.

Set OLETOOLS_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from __future__ import annotations

import os

from oletools_mojo import _native, _reference
from oletools_mojo._native import backend_info, native_available

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "decompress_stream",
    "get_backend",
    "backend_info",
    "native_available",
    "__version__",
]

# Cached native-availability probe (env disable always wins; failure to load
# is sticky for the process, like the other mojo-kernels wrappers).
_use_native: bool | None = None


def _backend() -> str:
    global _use_native
    if os.environ.get("OLETOOLS_MOJO_DISABLE_NATIVE") == "1":
        return "fallback"
    if _use_native is None:
        _use_native = _native.native_available()
    return "native" if _use_native else "fallback"


def get_backend() -> str:
    """The backend the next decompress call will use: "native" or "fallback"."""
    return _backend()


def decompress_stream(compressed_container) -> bytes:
    """Decompress an MS-OVBA CompressedContainer (VBA compressed source).

    Drop-in replacement for `oletools.olevba.decompress_stream`: accepts
    bytearray, bytes, or anything `bytearray()` accepts; returns the
    decompressed bytes. Output bytes and exception types
    (ValueError/IndexError/struct.error/TypeError) match the oracle
    exactly, on both backends.
    """
    # The oracle converts any non-bytearray input with bytearray(); keep the
    # identical conversion (including its TypeErrors for str/int input).
    if not isinstance(compressed_container, bytearray):
        compressed_container = bytearray(compressed_container)
    if _backend() == "native":
        return _native.decompress_native(bytes(compressed_container))
    return _reference.decompress_stream_reference(compressed_container)
