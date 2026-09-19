"""ctypes loader for the tomlmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$TOML_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``toml_mojo/_native/``
    3. the repository development build output ``kernels/toml/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the stdlib ``tomllib`` parser. Set
``TOML_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  tomlmojo_abi_version(void)
    void*    tomlmojo_parse(const uint8_t* data, int64_t length)
    int32_t  tomlmojo_result_status(void* handle)
    int64_t  tomlmojo_result_error_pos(void* handle)
    int64_t  tomlmojo_result_size(void* handle)
    uint8_t* tomlmojo_result_data(void* handle)
    void     tomlmojo_result_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/toml/src/tomlmojo.mojo. A mismatch means
# the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "TOML_MOJO_NATIVE_LIB"
_ENV_DISABLE = "TOML_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native tomlmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libtomlmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libtomlmojo.so"
    if sys.platform.startswith("win"):
        return "tomlmojo.dll"  # no Mojo toolchain builds this today
    return "libtomlmojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "toml", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.tomlmojo_abi_version.argtypes = []
    lib.tomlmojo_abi_version.restype = ctypes.c_int32
    lib.tomlmojo_parse.argtypes = [ctypes.c_char_p, ctypes.c_int64]
    lib.tomlmojo_parse.restype = ctypes.c_void_p
    lib.tomlmojo_result_status.argtypes = [ctypes.c_void_p]
    lib.tomlmojo_result_status.restype = ctypes.c_int32
    lib.tomlmojo_result_error_pos.argtypes = [ctypes.c_void_p]
    lib.tomlmojo_result_error_pos.restype = ctypes.c_int64
    lib.tomlmojo_result_size.argtypes = [ctypes.c_void_p]
    lib.tomlmojo_result_size.restype = ctypes.c_int64
    lib.tomlmojo_result_data.argtypes = [ctypes.c_void_p]
    lib.tomlmojo_result_data.restype = ctypes.c_void_p
    lib.tomlmojo_result_destroy.argtypes = [ctypes.c_void_p]
    lib.tomlmojo_result_destroy.restype = None


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
                    abi = int(lib.tomlmojo_abi_version())
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
    """True if the native kernel can parse right now. Never raises."""
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
    info["abi_version_native"] = int(lib.tomlmojo_abi_version())
    return info


def parse_bytes(data: bytes) -> tuple[int, int, bytes]:
    """Run one document through the native kernel.

    Returns ``(status, error_byte_offset, record_stream)``: status 0 with the
    output stream on success, status 1 with the parse-error byte offset
    otherwise. Raises :class:`NativeUnavailable` if the kernel is missing.
    """
    lib = _load()  # raises NativeUnavailable
    handle = lib.tomlmojo_parse(data, ctypes.c_int64(len(data)))
    if not handle:
        raise NativeUnavailable("native kernel returned no result handle")
    try:
        status = int(lib.tomlmojo_result_status(handle))
        err_pos = int(lib.tomlmojo_result_error_pos(handle))
        size = int(lib.tomlmojo_result_size(handle))
        stream = b""
        if size > 0:
            ptr = lib.tomlmojo_result_data(handle)
            if not ptr:
                raise NativeUnavailable("native kernel returned a null data pointer")
            stream = ctypes.string_at(ptr, size)
        return status, err_pos, stream
    finally:
        lib.tomlmojo_result_destroy(handle)
