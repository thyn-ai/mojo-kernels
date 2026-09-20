"""ctypes loader for the pdfmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$PDF_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``pdf_mojo/_native/``
    3. the repository development build output ``kernels/pypdf-filters/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``PDF_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  pdfmojo_abi_version(void)
    int64_t  pdfmojo_png_decode(const uint8_t* data, int64_t data_len,
                                int64_t columns, int64_t colors, int64_t bpc,
                                int64_t predictor, uint8_t* out_buf,
                                int64_t out_cap)
    int64_t  pdfmojo_lzw_decode(const uint8_t* data, int64_t data_len,
                                int64_t early_change, uint8_t** out_ptr)
    void     pdfmojo_free(uint8_t* ptr)

Negative return codes from the kernel: -1 invalid parameters, -2 malformed
LZW stream, -3 output buffer too small, -5 unsupported PNG row filter byte.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from pdf_mojo._reference import PdfFilterError

# Must equal ABI_VERSION in kernels/pypdf-filters/src/pdfmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "PDF_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PDF_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native pdfmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libpdfmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libpdfmojo.so"
    if sys.platform.startswith("win"):
        return "pdfmojo.dll"  # no Mojo toolchain builds this today
    return "libpdfmojo.so"


def _candidate_paths() -> list[tuple[str, str]]:
    """(source_label, path) candidates, in resolver order."""
    out: list[tuple[str, str]] = []
    override = os.environ.get(_ENV_LIB)
    if override:
        out.append((f"env {_ENV_LIB}", override))
    here = os.path.dirname(__file__)
    out.append(("bundled in wheel", os.path.join(here, "_native", _lib_basename())))
    out.append(
        (
            "repo-dev build output",
            os.path.abspath(
                os.path.join(
                    here, "..", "..", "..", "kernels", "pypdf-filters", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.pdfmojo_abi_version.argtypes = []
    lib.pdfmojo_abi_version.restype = ctypes.c_int32
    lib.pdfmojo_png_decode.argtypes = [
        ctypes.c_char_p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_int64,  # columns
        ctypes.c_int64,  # colors
        ctypes.c_int64,  # bits_per_component
        ctypes.c_int64,  # predictor
        ctypes.c_void_p,  # out_buf
        ctypes.c_int64,  # out_cap
    ]
    lib.pdfmojo_png_decode.restype = ctypes.c_int64
    lib.pdfmojo_lzw_decode.argtypes = [
        ctypes.c_char_p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_int64,  # early_change
        ctypes.POINTER(ctypes.c_void_p),  # out_ptr
    ]
    lib.pdfmojo_lzw_decode.restype = ctypes.c_int64
    lib.pdfmojo_free.argtypes = [ctypes.c_void_p]
    lib.pdfmojo_free.restype = None


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_SOURCE, _LOAD_ERROR
    if os.environ.get(_ENV_DISABLE) == "1":
        raise NativeUnavailable(f"native kernel disabled by {_ENV_DISABLE}=1")
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        errors: list[str] = []
        for label, path in _candidate_paths():
            try:
                if not path or not os.path.exists(path):
                    continue
                try:
                    lib = ctypes.CDLL(path)
                except OSError as exc:
                    errors.append(f"{label} ({path}): {exc}")
                    continue
                try:
                    _bind_abi(lib)
                    abi = int(lib.pdfmojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                _LIB, _LIB_SOURCE = lib, f"{label} ({path})"
                _LOAD_ERROR = None
                return lib
            except OSError as exc:
                errors.append(f"{label}: {exc}")
        _LOAD_ERROR = "; ".join(errors) or "no native kernel found on any resolver path"
        raise NativeUnavailable(_LOAD_ERROR)


def native_available() -> bool:
    """True if the native kernel can decode right now. Never raises."""
    try:
        _load()
        return True
    except NativeUnavailable:
        return False


def backend_info() -> dict:
    """Diagnostics for the active backend. Never raises."""
    info = {
        "native_available": False,
        "native_source": None,
        "abi_version_expected": ABI_VERSION,
        "abi_version_native": None,
        "disabled_by_env": os.environ.get(_ENV_DISABLE) == "1",
        "platform": sys.platform,
        "error": None,
    }
    try:
        lib = _load()
    except NativeUnavailable as exc:
        info["error"] = str(exc)
        return info
    info["native_available"] = True
    info["native_source"] = _LIB_SOURCE
    info["abi_version_native"] = int(lib.pdfmojo_abi_version())
    return info


def png_output_size(data_len: int, columns: int, colors: int, bpc: int, predictor: int) -> int:
    """Exact reconstructed size for `pdfmojo_png_decode` (validated inputs)."""
    if predictor in (1, 2):
        return data_len
    row_len = (colors * columns * bpc + 7) // 8
    stride = row_len + 1
    return ((data_len + stride - 1) // stride) * row_len


def decode_png_prediction_native(
    data: bytes, columns: int, colors: int, bpc: int, predictor: int
) -> bytes:
    """PNG/TIFF predictor reconstruction via the native kernel."""
    lib = _load()  # raises NativeUnavailable
    need = png_output_size(len(data), columns, colors, bpc, predictor)
    out = ctypes.create_string_buffer(max(need, 1))
    rc = lib.pdfmojo_png_decode(
        data, len(data), columns, colors, bpc, predictor,
        ctypes.cast(out, ctypes.c_void_p), need,
    )
    if rc == -5:
        raise PdfFilterError("unsupported PNG row filter byte (> 4)")
    if rc < 0:
        raise PdfFilterError(f"native kernel rejected the stream (status {rc})")
    return out.raw[:rc]


def decode_lzw_native(data: bytes, early_change: int) -> bytes:
    """LZW decode via the native kernel."""
    lib = _load()  # raises NativeUnavailable
    out_ptr = ctypes.c_void_p()
    rc = lib.pdfmojo_lzw_decode(data, len(data), early_change, ctypes.byref(out_ptr))
    if rc == -2:
        raise PdfFilterError("malformed LZW stream: non-literal code where a literal is required")
    if rc < 0:
        raise PdfFilterError(f"native kernel rejected the stream (status {rc})")
    try:
        return ctypes.string_at(out_ptr.value, rc)
    finally:
        lib.pdfmojo_free(out_ptr)
