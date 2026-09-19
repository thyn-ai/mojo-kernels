"""ctypes loader for the uprootmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$UPROOT_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``uproot_mojo/_native/``
    3. the repository development build output ``kernels/uproot-branches/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python walker. Set
``UPROOT_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t uprootmojo_abi_version(void)
    int32_t uprootmojo_scan(const uint8_t* data, int64_t data_len,
                            const int64_t* borders, int64_t n_entries,
                            int32_t mode, int32_t itemsize,
                            int64_t* out_mode, int64_t* out_total_items,
                            int64_t* out_total_bytes)
    int32_t uprootmojo_fill(const uint8_t* data, int64_t data_len,
                            const int64_t* borders, int64_t n_entries,
                            int32_t mode, int32_t itemsize,
                            int64_t* out_offsets, uint8_t* out_content,
                            int64_t* out_string_offsets,
                            int64_t total_items, int64_t total_bytes)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

from uproot_mojo._fallback import MODE_STR_VEC, OK, error_name
from uproot_mojo._rootfile import BasketDataError

# Must equal ABI_VERSION in kernels/uproot-branches/src/uprootmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "UPROOT_MOJO_NATIVE_LIB"
_ENV_DISABLE = "UPROOT_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native uprootmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libuprootmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libuprootmojo.so"
    if sys.platform.startswith("win"):
        return "uprootmojo.dll"  # no Mojo toolchain builds this today
    return "libuprootmojo.so"


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
                    here, "..", "..", "..", "kernels", "uproot-branches", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.uprootmojo_abi_version.argtypes = []
    lib.uprootmojo_abi_version.restype = ctypes.c_int32
    lib.uprootmojo_scan.argtypes = [
        u8p,  # data
        ctypes.c_int64,  # data_len
        i64p,  # borders[n_entries+1]
        ctypes.c_int64,  # n_entries
        ctypes.c_int32,  # mode
        ctypes.c_int32,  # itemsize
        i64p,  # out_mode
        i64p,  # out_total_items
        i64p,  # out_total_bytes
    ]
    lib.uprootmojo_scan.restype = ctypes.c_int32
    lib.uprootmojo_fill.argtypes = [
        u8p,  # data
        ctypes.c_int64,  # data_len
        i64p,  # borders[n_entries+1]
        ctypes.c_int64,  # n_entries
        ctypes.c_int32,  # mode
        ctypes.c_int32,  # itemsize
        i64p,  # out_offsets[n_entries+1]
        u8p,  # out_content[total_bytes]
        i64p,  # out_string_offsets[total_items+1] (STR_VEC only)
        ctypes.c_int64,  # total_items
        ctypes.c_int64,  # total_bytes
    ]
    lib.uprootmojo_fill.restype = ctypes.c_int32


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
                    abi = int(lib.uprootmojo_abi_version())
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
    """True if the native kernel can walk baskets right now. Never raises."""
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
    info["abi_version_native"] = int(lib.uprootmojo_abi_version())
    return info


def _raise_for_code(code: int, where: str) -> None:
    if code != OK:
        raise BasketDataError(
            code, f"{where}: basket walk failed with {error_name(code)} ({code})"
        )


def walk_basket(data: bytes, borders: np.ndarray, mode: int, itemsize: int, where: str):
    """Scan + fill one basket with the native kernel.

    Same signature/semantics as ``uproot_mojo._fallback.walk_basket``:
    returns (detected_mode, offsets, content, string_offsets).
    """
    lib = _load()  # raises NativeUnavailable
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)

    data_arr = np.frombuffer(data, dtype=np.uint8)
    borders64 = np.ascontiguousarray(borders, dtype=np.int64)
    n_entries = np.int64(len(borders64) - 1)
    out_mode = np.zeros(1, dtype=np.int64)
    out_total_items = np.zeros(1, dtype=np.int64)
    out_total_bytes = np.zeros(1, dtype=np.int64)

    rc = lib.uprootmojo_scan(
        data_arr.ctypes.data_as(u8p),
        ctypes.c_int64(len(data_arr)),
        borders64.ctypes.data_as(i64p),
        n_entries,
        ctypes.c_int32(mode),
        ctypes.c_int32(itemsize),
        out_mode.ctypes.data_as(i64p),
        out_total_items.ctypes.data_as(i64p),
        out_total_bytes.ctypes.data_as(i64p),
    )
    _raise_for_code(rc, where)
    detected = int(out_mode[0])
    total_items, total_bytes = int(out_total_items[0]), int(out_total_bytes[0])

    offsets = np.zeros(n_entries + 1, dtype=np.int64)
    content = np.zeros(total_bytes, dtype=np.uint8)
    string_offsets = np.zeros(total_items + 1, dtype=np.int64)

    rc = lib.uprootmojo_fill(
        data_arr.ctypes.data_as(u8p),
        ctypes.c_int64(len(data_arr)),
        borders64.ctypes.data_as(i64p),
        n_entries,
        ctypes.c_int32(detected),
        ctypes.c_int32(itemsize),
        offsets.ctypes.data_as(i64p),
        content.ctypes.data_as(u8p),
        string_offsets.ctypes.data_as(i64p),
        ctypes.c_int64(total_items),
        ctypes.c_int64(total_bytes),
    )
    _raise_for_code(rc, where)
    return (
        detected,
        offsets,
        content.tobytes(),
        string_offsets if detected == MODE_STR_VEC else None,
    )
