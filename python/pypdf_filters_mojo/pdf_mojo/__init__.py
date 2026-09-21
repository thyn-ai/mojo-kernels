"""pypdf-filters-mojo: Mojo-accelerated PNG-predictor and LZW decoders for
PDF streams, byte-compatible with pypdf's FlateDecode/LZWDecode filters.

Powered by a clean-room Mojo kernel where the platform supports it (macOS
arm64, Linux x86_64), with a vendored pure-Python fallback everywhere else
(including Windows). Both backends are byte-identical to the pypdf oracle.

    import zlib
    from pdf_mojo import decode_png_prediction, decode_lzw

    # Reconstruct an image XObject's flate stream with PNG predictors:
    raw = decode_png_prediction(
        zlib.decompress(stream_data),   # still predictor-coded bytes
        columns=640, colors=3, bits_per_component=8, predictor=15,
    )

    # Decode an LZW stream (EarlyChange=1 is the PDF default):
    raw = decode_lzw(lzw_data, early_change=1)

Set PDF_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from __future__ import annotations

import os

from pdf_mojo import _native, _reference
from pdf_mojo._native import backend_info, native_available
from pdf_mojo._reference import PdfFilterError

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "decode_png_prediction",
    "decode_lzw",
    "get_backend",
    "backend_info",
    "native_available",
    "PdfFilterError",
    "__version__",
]

_VALID_BPC = (1, 2, 4, 8, 16)
_VALID_PREDICTORS = (1, 2, 10, 11, 12, 13, 14, 15)

# Cached native-availability probe (env disable always wins; failure to load
# is sticky for the process, like the bm25-mojo wrapper).
_use_native: bool | None = None


def _backend() -> str:
    global _use_native
    if os.environ.get("PDF_MOJO_DISABLE_NATIVE") == "1":
        return "fallback"
    if _use_native is None:
        _use_native = _native.native_available()
    return "native" if _use_native else "fallback"


def get_backend() -> str:
    """The backend the next decode call will use: "native" or "fallback"."""
    return _backend()


def _as_bytes(data, name: str) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"{name} must be a bytes-like object, got {type(data).__name__}")


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def decode_png_prediction(
    data,
    columns: int,
    colors: int,
    bits_per_component: int,
    predictor: int = 15,
) -> bytes:
    """Reconstruct a PNG/TIFF-predicted (already inflated) stream.

    `data` is the predictor-coded content of a FlateDecode stream AFTER zlib
    inflation. `columns`, `colors`, and `bits_per_component` are the PDF
    decode parameters; `predictor` is 1 (none), 2 (TIFF), or 10-15 (PNG).
    Returns the reconstructed bytes (ragged final PNG rows are zero-padded
    to a full row, matching pypdf).

    Raises ValueError for invalid parameters and PdfFilterError for corrupt
    streams (e.g. a PNG row filter byte > 4).
    """
    data = _as_bytes(data, "data")
    columns = _as_int(columns, "columns")
    colors = _as_int(colors, "colors")
    bits_per_component = _as_int(bits_per_component, "bits_per_component")
    predictor = _as_int(predictor, "predictor")
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    if colors < 1:
        raise ValueError(f"colors must be >= 1, got {colors}")
    if bits_per_component not in _VALID_BPC:
        raise ValueError(
            f"bits_per_component must be one of {_VALID_BPC}, got {bits_per_component}"
        )
    if predictor not in _VALID_PREDICTORS:
        raise ValueError(f"predictor must be one of {_VALID_PREDICTORS}, got {predictor}")
    if _backend() == "native":
        return _native.decode_png_prediction_native(
            data, columns, colors, bits_per_component, predictor
        )
    return _reference.decode_png_prediction_reference(
        data, columns, colors, bits_per_component, predictor
    )


def decode_lzw(data, early_change: int = 1) -> bytes:
    """Decode an LZW stream (MSB-first codes, 9-12 bits, EarlyChange 0/1).

    `early_change` is the PDF /EarlyChange decode parameter (default 1).
    Note: pypdf 6.x grows the code width at table index (1 << width) - 1
    for BOTH EarlyChange values; this package matches the oracle
    byte-for-byte. Decoding stops at the EOD code or when the stream is
    exhausted; trailing garbage after EOD is ignored, matching pypdf.

    Raises ValueError for invalid parameters and PdfFilterError for corrupt
    streams (a non-literal code where a literal is required).
    """
    data = _as_bytes(data, "data")
    early_change = _as_int(early_change, "early_change")
    if early_change not in (0, 1):
        raise ValueError(f"early_change must be 0 or 1, got {early_change}")
    if _backend() == "native":
        return _native.decode_lzw_native(data, early_change)
    return _reference.decode_lzw_reference(data, early_change)
