"""ctypes loader for the msgpackmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$MSGPACK_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``msgpack_mojo/_native/``
    3. the repository development build output ``kernels/msgpack/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python byte engine. Set
``MSGPACK_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  msgpackmojo_abi_version(void)
    void*    msgpackmojo_pack(const uint8_t* stream, int64_t length, int64_t flags)
    void*    msgpackmojo_unpack(const uint8_t* data, int64_t length, int64_t flags)
    int32_t  msgpackmojo_result_status(void* handle)
    int64_t  msgpackmojo_result_error_pos(void* handle)
    int64_t  msgpackmojo_result_size(void* handle)
    uint8_t* msgpackmojo_result_data(void* handle)
    void     msgpackmojo_result_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/msgpack/src/msgpackmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; the
# wrapper falls back to the pure-Python engine.
ABI_VERSION = 1

# Result status codes, mirrored from the kernel.
STATUS_OK = 0
STATUS_FORMAT = 1  # malformed MessagePack (e.g. reserved prefix 0xc1)
STATUS_TRUNCATED = 2  # ran out of input mid-value
STATUS_EXTRA_DATA = 3  # trailing bytes after one complete value

# Pack flag bits (kernel-side): bit0 = use_bin_type, bit1 = use_single_float.
# Unpack flag bits: bit0 = raw.
PACK_USE_BIN_TYPE = 1
PACK_SINGLE_FLOAT = 2
UNPACK_RAW = 1

_ENV_LIB = "MSGPACK_MOJO_NATIVE_LIB"
_ENV_DISABLE = "MSGPACK_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native msgpackmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libmsgpackmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libmsgpackmojo.so"
    if sys.platform.startswith("win"):
        return "msgpackmojo.dll"  # no Mojo toolchain builds this today
    return "libmsgpackmojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "msgpack", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.msgpackmojo_abi_version.argtypes = []
    lib.msgpackmojo_abi_version.restype = ctypes.c_int32
    lib.msgpackmojo_pack.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]
    lib.msgpackmojo_pack.restype = ctypes.c_void_p
    lib.msgpackmojo_unpack.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]
    lib.msgpackmojo_unpack.restype = ctypes.c_void_p
    lib.msgpackmojo_result_status.argtypes = [ctypes.c_void_p]
    lib.msgpackmojo_result_status.restype = ctypes.c_int32
    lib.msgpackmojo_result_error_pos.argtypes = [ctypes.c_void_p]
    lib.msgpackmojo_result_error_pos.restype = ctypes.c_int64
    lib.msgpackmojo_result_size.argtypes = [ctypes.c_void_p]
    lib.msgpackmojo_result_size.restype = ctypes.c_int64
    lib.msgpackmojo_result_data.argtypes = [ctypes.c_void_p]
    lib.msgpackmojo_result_data.restype = ctypes.c_void_p
    lib.msgpackmojo_result_destroy.argtypes = [ctypes.c_void_p]
    lib.msgpackmojo_result_destroy.restype = None


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
                    abi = int(lib.msgpackmojo_abi_version())
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
    """True if the native kernel can run right now. Never raises."""
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
    info["abi_version_native"] = int(lib.msgpackmojo_abi_version())
    return info


def _read_result(handle: int) -> tuple[int, int, bytes]:
    """(status, error_pos, output bytes) from an owned result handle."""
    lib = _LIB
    assert lib is not None
    try:
        status = int(lib.msgpackmojo_result_status(handle))
        err_pos = int(lib.msgpackmojo_result_error_pos(handle))
        size = int(lib.msgpackmojo_result_size(handle))
        out = b""
        if size > 0:
            ptr = lib.msgpackmojo_result_data(handle)
            if not ptr:
                raise NativeUnavailable("native kernel returned a null data pointer")
            out = ctypes.string_at(ptr, size)
        return status, err_pos, out
    finally:
        lib.msgpackmojo_result_destroy(handle)


def pack_bytes(stream: bytes, flags: int) -> bytes:
    """Run one instruction stream through the native pack engine.

    Raises :class:`NativeUnavailable` if the kernel is missing or rejects the
    (wrapper-produced) stream; the caller then repacks with the pure-Python
    engine, so a kernel fault can never surface as corrupt output.
    """
    lib = _load()  # raises NativeUnavailable
    handle = lib.msgpackmojo_pack(stream, len(stream), flags)
    if not handle:
        raise NativeUnavailable("native kernel returned no result handle")
    status, _pos, out = _read_result(handle)
    if status != STATUS_OK:
        raise NativeUnavailable(f"native pack engine rejected the instruction stream ({status})")
    return out


def unpack_bytes(data: bytes, flags: int) -> tuple[int, int, bytes]:
    """Run one MessagePack buffer through the native unpack engine.

    Returns ``(status, error_byte_offset, record_stream)``: status 0 with the
    record stream on success, or one of the STATUS_* failure codes with the
    input offset where the failure was detected. Raises
    :class:`NativeUnavailable` only if the kernel itself is missing/broken.
    """
    lib = _load()  # raises NativeUnavailable
    handle = lib.msgpackmojo_unpack(data, len(data), flags)
    if not handle:
        raise NativeUnavailable("native kernel returned no result handle")
    return _read_result(handle)
