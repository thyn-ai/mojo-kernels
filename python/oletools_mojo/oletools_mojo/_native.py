"""ctypes loader for the vbamojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$OLETOOLS_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``oletools_mojo/_native/``
    3. the repository development build output ``kernels/oletools-vba/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``OLETOOLS_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  vbamojo_abi_version(void)
    int64_t  vbamojo_decompress(const uint8_t* data, int64_t data_len,
                                uint8_t** out_ptr)
    void     vbamojo_free(uint8_t* ptr)

`vbamojo_decompress` returns the decompressed length (>= 0) with `out_ptr`
receiving a library-allocated buffer (free with `vbamojo_free`), or a
negative status code (and `out_ptr` left untouched):

    -1  empty input
    -2  invalid signature byte (!= 0x01)
    -3  invalid CompressedChunkSignature (!= 0b011)
    -4  RawChunk with CompressedChunkSize != 4098
    -5  CopyToken at difference == 0
    -6  truncated 16-bit read at the container end (chunk header or CopyToken)
    -7  copy source index out of range after negative-index wrap
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading

# Must equal ABI_VERSION in kernels/oletools-vba/src/vbamojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "OLETOOLS_MOJO_NATIVE_LIB"
_ENV_DISABLE = "OLETOOLS_MOJO_DISABLE_NATIVE"

# Kernel status codes (see module docstring).
_ERR_EMPTY = -1
_ERR_SIGNATURE = -2
_ERR_CHUNK_SIGNATURE = -3
_ERR_RAW_SIZE = -4
_ERR_COPYTOKEN_AT_ZERO = -5
_ERR_TRUNCATED_U16 = -6
_ERR_INDEX = -7


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native vbamojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libvbamojo.dylib"
    if sys.platform.startswith("linux"):
        return "libvbamojo.so"
    if sys.platform.startswith("win"):
        return "vbamojo.dll"  # no Mojo toolchain builds this today
    return "libvbamojo.so"


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
                    here, "..", "..", "..", "kernels", "oletools-vba", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.vbamojo_abi_version.argtypes = []
    lib.vbamojo_abi_version.restype = ctypes.c_int32
    lib.vbamojo_decompress.argtypes = [
        ctypes.c_char_p,  # data
        ctypes.c_int64,  # data_len
        ctypes.POINTER(ctypes.c_void_p),  # out_ptr
    ]
    lib.vbamojo_decompress.restype = ctypes.c_int64
    lib.vbamojo_free.argtypes = [ctypes.c_void_p]
    lib.vbamojo_free.restype = None


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
                    abi = int(lib.vbamojo_abi_version())
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
    """True if the native kernel can decompress right now. Never raises."""
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
    info["abi_version_native"] = int(lib.vbamojo_abi_version())
    return info


def decompress_native(data: bytes) -> bytes:
    """Decompress via the native kernel; raises oracle-identical errors."""
    lib = _load()  # raises NativeUnavailable
    out_ptr = ctypes.c_void_p()
    rc = lib.vbamojo_decompress(data, len(data), ctypes.byref(out_ptr))
    if rc >= 0:
        try:
            return ctypes.string_at(out_ptr.value, rc)
        finally:
            lib.vbamojo_free(out_ptr)
    if rc == _ERR_EMPTY:
        raise IndexError("bytearray index out of range")
    if rc == _ERR_SIGNATURE:
        raise ValueError("invalid signature byte {0:02X}".format(data[0]))
    if rc == _ERR_CHUNK_SIGNATURE:
        raise ValueError("Invalid CompressedChunkSignature in VBA compressed stream")
    if rc == _ERR_RAW_SIZE:
        # The oracle's message embeds the offending CompressedChunkSize,
        # which depends on full chunk-walk state. Error path only: let the
        # vendored reference re-raise the identical ValueError.
        from oletools_mojo._reference import decompress_stream_reference

        decompress_stream_reference(data)
        raise AssertionError("unreachable: reference did not raise")  # noqa: B904
    if rc == _ERR_COPYTOKEN_AT_ZERO:
        raise ValueError("math domain error")
    if rc == _ERR_TRUNCATED_U16:
        raise struct.error("unpack requires a buffer of 2 bytes")
    if rc == _ERR_INDEX:
        raise IndexError("bytearray index out of range")
    raise NativeUnavailable(f"native kernel returned an unknown status {rc}")
