"""ctypes loader for the dateutilmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$DATEUTIL_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``dateutil_mojo/_native/``
    3. the repository development build output ``kernels/dateutil-parse/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the pure-Python reference parser. Set
``DATEUTIL_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  dateutilmojo_abi_version(void)
    int32_t  dateutilmojo_parse_batch(const uint8_t* data,
                                      const int64_t* offsets,  // n+1
                                      int64_t n,
                                      uint32_t flags,   // bit0 dayfirst, bit1 yearfirst
                                      const int32_t* defaults, // y,mo,d,h,mi,s,us + pivot_base
                                      int32_t* out,     // n*9: status,y,mo,d,h,mi,s,us,tzsec
                                      int64_t out_len)

Per row: status 0 = parsed (tzsec == INT32_MIN -> naive, else offset seconds;
0 -> UTC), status 1 = outside the native fast path, the caller re-parses with
the Python reference. The native parser only accepts shapes whose semantics
are verified against the oracle; everything else is status 1.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from array import array

# Must equal ABI_VERSION in kernels/dateutil-parse/src/dateutilmojo.mojo.
ABI_VERSION = 1

NAIVE_SENTINEL = -(2**31)  # INT32_MIN: row parsed, no timezone

_FLAG_DAYFIRST = 1
_FLAG_YEARFIRST = 2

_ENV_LIB = "DATEUTIL_MOJO_NATIVE_LIB"
_ENV_DISABLE = "DATEUTIL_MOJO_DISABLE_NATIVE"

_ROW = 9  # int32 fields per output row


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native dateutilmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libdateutilmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libdateutilmojo.so"
    if sys.platform.startswith("win"):
        return "dateutilmojo.dll"  # no Mojo toolchain builds this today
    return "libdateutilmojo.so"


def _candidate_paths() -> list[tuple[str, str]]:
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
                    here, "..", "..", "..", "kernels", "dateutil-parse", "build",
                    _lib_basename(),
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.dateutilmojo_abi_version.argtypes = []
    lib.dateutilmojo_abi_version.restype = ctypes.c_int32
    lib.dateutilmojo_parse_batch.argtypes = [
        u8p,  # data
        i64p,  # offsets (n+1)
        ctypes.c_int64,  # n
        ctypes.c_uint32,  # flags
        i32p,  # defaults (8)
        i32p,  # out (n*9)
        ctypes.c_int64,  # out_len
    ]
    lib.dateutilmojo_parse_batch.restype = ctypes.c_int32


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None
_PARSE_FN = None  # cached bound entry point after first load


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_SOURCE, _LOAD_ERROR, _PARSE_FN
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
                    abi = int(lib.dateutilmojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                _LIB, _LIB_SOURCE = lib, f"{label} ({path})"
                _PARSE_FN = lib.dateutilmojo_parse_batch
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
    info["abi_version_native"] = int(lib.dateutilmojo_abi_version())
    return info


def parse_batch_native(
    strings: list[str],
    *,
    dayfirst: bool,
    yearfirst: bool,
    defaults: tuple[int, int, int, int, int, int, int],
    pivot_base: int,
) -> list[int] | None:
    """Run the native batch parser.

    Returns the flat int32 output: n rows of 9 ints
    (status, y, mo, d, h, mi, s, us, tzsec). Raises NativeUnavailable if the
    kernel or the arguments are rejected.
    """
    lib = _load()  # raises NativeUnavailable
    n = len(strings)
    if n == 0:
        return []
    parts: list[bytes] = []
    offsets = array("q", [0])
    for s in strings:
        b = s.encode("utf-8")
        parts.append(b)
        offsets.append(offsets[-1] + len(b))
    data = b"".join(parts)
    flags = (_FLAG_DAYFIRST if dayfirst else 0) | (_FLAG_YEARFIRST if yearfirst else 0)
    defs = (ctypes.c_int32 * 8)(*defaults, pivot_base)
    offs_buf = (ctypes.c_int64 * (n + 1))(*offsets)
    out = (ctypes.c_int32 * (_ROW * n))()
    fn = _PARSE_FN if _PARSE_FN is not None else lib.dateutilmojo_parse_batch
    rc = fn(
        ctypes.cast(ctypes.c_char_p(data), ctypes.POINTER(ctypes.c_uint8)),
        offs_buf,
        n,
        flags,
        defs,
        out,
        _ROW * n,
    )
    if rc != 0:
        raise NativeUnavailable(f"native batch parse rejected the input (status {rc})")
    return list(out)
