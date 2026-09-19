"""ctypes loader for the cronitermojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$CRONITER_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``croniter_mojo/_native/``
    3. the repository development build output ``kernels/croniter/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the pure-Python engine.
Set ``CRONITER_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  cronitermojo_abi_version(void)
    void*    cronitermojo_create(uint64_t minutes, uint32_t hours,
                                 uint32_t doms, int32_t dom_has_l,
                                 uint32_t months, uint32_t dows,
                                 int32_t dom_restricted, int32_t dow_restricted,
                                 int32_t day_or, int32_t dow_has_hash,
                                 const int32_t* hash_dow,
                                 const int32_t* hash_nth, int32_t n_hash)
    int32_t  cronitermojo_seek(void* handle, int32_t y, int32_t mo, int32_t d,
                               int32_t h, int32_t mi, int32_t direction,
                               int32_t year_bound, int32_t* out_ymdhm)
    void     cronitermojo_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from ._fields import Schedule

# Must equal ABI_VERSION in kernels/croniter/src/cronitermojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "CRONITER_MOJO_NATIVE_LIB"
_ENV_DISABLE = "CRONITER_MOJO_DISABLE_NATIVE"

_DIR_NEXT = 1
_DIR_PREV = -1

_RC_MATCH = 0
_RC_NO_MATCH_WITHIN_BOUND = 1


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native cronitermojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libcronitermojo.dylib"
    if sys.platform.startswith("linux"):
        return "libcronitermojo.so"
    if sys.platform.startswith("win"):
        return "cronitermojo.dll"  # no Mojo toolchain builds this today
    return "libcronitermojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "croniter", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.cronitermojo_abi_version.argtypes = []
    lib.cronitermojo_abi_version.restype = ctypes.c_int32
    lib.cronitermojo_create.argtypes = [
        ctypes.c_uint64,  # minutes
        ctypes.c_uint32,  # hours
        ctypes.c_uint32,  # doms
        ctypes.c_int32,  # dom_has_l
        ctypes.c_uint32,  # months
        ctypes.c_uint32,  # dows
        ctypes.c_int32,  # dom_restricted
        ctypes.c_int32,  # dow_restricted
        ctypes.c_int32,  # day_or
        ctypes.c_int32,  # dow_has_hash
        i32p,  # hash_dow[n_hash]
        i32p,  # hash_nth[n_hash]
        ctypes.c_int32,  # n_hash
    ]
    lib.cronitermojo_create.restype = ctypes.c_void_p
    lib.cronitermojo_seek.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,  # year
        ctypes.c_int32,  # month
        ctypes.c_int32,  # day
        ctypes.c_int32,  # hour
        ctypes.c_int32,  # minute
        ctypes.c_int32,  # direction
        ctypes.c_int32,  # year_bound
        i32p,  # out_ymdhm[5]
    ]
    lib.cronitermojo_seek.restype = ctypes.c_int32
    lib.cronitermojo_destroy.argtypes = [ctypes.c_void_p]
    lib.cronitermojo_destroy.restype = None


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
                    abi = int(lib.cronitermojo_abi_version())
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
    """True if the native kernel can seek right now. Never raises."""
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
    info["abi_version_native"] = int(lib.cronitermojo_abi_version())
    return info


class NativeEngine:
    """Native wall-clock seek over a parsed schedule. Not thread-safe to close twice."""

    def __init__(self, schedule: Schedule) -> None:
        lib = _load()  # raises NativeUnavailable
        n_hash = len(schedule.hash_specs)
        hash_dow = hash_nth = None
        if n_hash:
            hash_dow = (ctypes.c_int32 * n_hash)(*(hd for hd, _ in schedule.hash_specs))
            hash_nth = (ctypes.c_int32 * n_hash)(*(hn for _, hn in schedule.hash_specs))
        handle = lib.cronitermojo_create(
            ctypes.c_uint64(schedule.minutes),
            ctypes.c_uint32(schedule.hours),
            ctypes.c_uint32(schedule.doms),
            ctypes.c_int32(1 if schedule.dom_has_l else 0),
            ctypes.c_uint32(schedule.months),
            ctypes.c_uint32(schedule.dows),
            ctypes.c_int32(1 if schedule.dom_restricted else 0),
            ctypes.c_int32(1 if schedule.dow_restricted else 0),
            ctypes.c_int32(1 if schedule.day_or else 0),
            ctypes.c_int32(1 if schedule.dow_has_hash else 0),
            hash_dow,
            hash_nth,
            ctypes.c_int32(n_hash),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the schedule; falling back to the "
                "pure-Python engine"
            )
        # The kernel copies every buffer; the ctypes arrays may be freed by the GC.
        self._lib = lib
        self._handle = handle

    def _seek(self, direction: int, y: int, mo: int, d: int, h: int, mi: int, year_bound: int):
        if self._handle is None:
            raise NativeUnavailable("native schedule is closed")
        out = (ctypes.c_int32 * 5)()
        rc = self._lib.cronitermojo_seek(
            self._handle,
            ctypes.c_int32(y),
            ctypes.c_int32(mo),
            ctypes.c_int32(d),
            ctypes.c_int32(h),
            ctypes.c_int32(mi),
            ctypes.c_int32(direction),
            ctypes.c_int32(year_bound),
            out,
        )
        if rc == _RC_MATCH:
            return (out[0], out[1], out[2], out[3], out[4])
        if rc == _RC_NO_MATCH_WITHIN_BOUND:
            return None
        raise NativeUnavailable(f"native seek failed with status {rc}")

    def seek_next(self, y: int, mo: int, d: int, h: int, mi: int, year_bound: int):
        return self._seek(_DIR_NEXT, y, mo, d, h, mi, year_bound)

    def seek_prev(self, y: int, mo: int, d: int, h: int, mi: int, year_bound: int):
        return self._seek(_DIR_PREV, y, mo, d, h, mi, year_bound)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.cronitermojo_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
