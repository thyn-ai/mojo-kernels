"""ctypes loader for the pymavmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$PYMAVLINK_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``pymavlink_mojo/_native/``
    3. the repository development build output ``kernels/pymavlink/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python parser. Set
``PYMAVLINK_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

The native :func:`parse` returns the same flat arena layout as
:mod:`pymavlink_mojo._reference` (see :mod:`pymavlink_mojo._layout`); the
dialect tables are passed through the ABI on every call so the kernel and
the fallback share one source of truth (:mod:`pymavlink_mojo._dialect`).

Stable C ABI (v1)::

    int32_t  pymavmojo_abi_version(void)
    void*    pymavmojo_parse(const uint8_t* data, int64_t len, int32_t tlog_mode,
                             int64_t n_msg, const uint32_t* msg_ids,
                             const uint8_t* msg_crc_extra,
                             const uint16_t* msg_csize,
                             const uint8_t* msg_nflds,
                             const uint32_t* msg_foff,
                             const uint16_t* fld_woff,
                             const uint8_t* fld_type,
                             const uint16_t* fld_alen,
                             int32_t* out_err)
    int64_t  pymavmojo_msg_count(void* h)
    int64_t  pymavmojo_error_count(void* h)
    int64_t  pymavmojo_consumed(void* h)
    int32_t  pymavmojo_arrays(void* h, const int64_t** recs, const int64_t** fi,
                              const double** ff, const uint8_t** fu,
                              const uint8_t** stream,
                              int64_t* n_recs, int64_t* n_fi,
                              int64_t* n_ff, int64_t* n_fu, int64_t* n_stream)
    void     pymavmojo_free(void* h)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

from pymavlink_mojo._dialect import MESSAGES
from pymavlink_mojo.errors import NativeUnavailable

# Must equal ABI_VERSION in kernels/pymavlink/src/pymavmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree.
ABI_VERSION = 1

_ENV_LIB = "PYMAVLINK_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PYMAVLINK_MOJO_DISABLE_NATIVE"

_u8p = ctypes.POINTER(ctypes.c_uint8)
_u16p = ctypes.POINTER(ctypes.c_uint16)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)
_f64p = ctypes.POINTER(ctypes.c_double)
_i64pp = ctypes.POINTER(_i64p)
_f64pp = ctypes.POINTER(_f64p)
_u8pp = ctypes.POINTER(_u8p)


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libpymavmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libpymavmojo.so"
    if sys.platform.startswith("win"):
        return "pymavmojo.dll"  # no Mojo toolchain builds this today
    return "libpymavmojo.so"


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
                    here, "..", "..", "..", "kernels", "pymavlink", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.pymavmojo_abi_version.argtypes = []
    lib.pymavmojo_abi_version.restype = ctypes.c_int32
    lib.pymavmojo_parse.argtypes = [
        _u8p,  # data
        ctypes.c_int64,  # length
        ctypes.c_int32,  # tlog_mode
        ctypes.c_int64,  # n_msg
        _u32p,  # msg_ids
        _u8p,  # msg_crc_extra
        _u16p,  # msg_csize
        _u8p,  # msg_nflds
        _u32p,  # msg_foff
        _u16p,  # fld_woff
        _u8p,  # fld_type
        _u16p,  # fld_alen
        _i32p,  # out_err
    ]
    lib.pymavmojo_parse.restype = ctypes.c_void_p
    lib.pymavmojo_msg_count.argtypes = [ctypes.c_void_p]
    lib.pymavmojo_msg_count.restype = ctypes.c_int64
    lib.pymavmojo_error_count.argtypes = [ctypes.c_void_p]
    lib.pymavmojo_error_count.restype = ctypes.c_int64
    lib.pymavmojo_consumed.argtypes = [ctypes.c_void_p]
    lib.pymavmojo_consumed.restype = ctypes.c_int64
    lib.pymavmojo_arrays.argtypes = [
        ctypes.c_void_p,
        _i64pp,
        _i64pp,
        _f64pp,
        _u8pp,
        _u8pp,
        _i64p,
        _i64p,
        _i64p,
        _i64p,
        _i64p,
    ]
    lib.pymavmojo_arrays.restype = ctypes.c_int32
    lib.pymavmojo_free.argtypes = [ctypes.c_void_p]
    lib.pymavmojo_free.restype = None


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
            if not path or not os.path.exists(path):
                continue
            try:
                lib = ctypes.CDLL(path)
            except OSError as exc:
                errors.append(f"{label} ({path}): {exc}")
                continue
            try:
                _bind_abi(lib)
                abi = int(lib.pymavmojo_abi_version())
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
    info["abi_version_native"] = int(lib.pymavmojo_abi_version())
    return info


class _DialectTables:
    """The bundled dialect tables as ctypes arrays (built once, cached).

    Passing them through the ABI on every call keeps _dialect.py the single
    source of truth for the native kernel, the fallback, and the wrapper.
    """

    def __init__(self) -> None:
        n = len(MESSAGES)
        self.msg_ids = (ctypes.c_uint32 * n)(*(m[0] for m in MESSAGES))
        self.msg_crc_extra = (ctypes.c_uint8 * n)(*(m[2] for m in MESSAGES))
        self.msg_csize = (ctypes.c_uint16 * n)(*(m[3] for m in MESSAGES))
        self.msg_nflds = (ctypes.c_uint8 * n)(*(len(m[4]) for m in MESSAGES))
        foffs: list[int] = []
        woff: list[int] = []
        wtype: list[int] = []
        walen: list[int] = []
        for m in MESSAGES:
            foffs.append(len(woff))
            for _name, code, alen, off in m[4]:
                woff.append(off)
                wtype.append(code)
                walen.append(alen)
        nf = len(woff)
        self.msg_foff = (ctypes.c_uint32 * n)(*foffs)
        self.fld_woff = (ctypes.c_uint16 * nf)(*woff)
        self.fld_type = (ctypes.c_uint8 * nf)(*wtype)
        self.fld_alen = (ctypes.c_uint16 * nf)(*walen)
        self.n_msg = n


_TABLES: _DialectTables | None = None


def _tables() -> _DialectTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = _DialectTables()
    return _TABLES


def parse(data: bytes, tlog_mode: bool):
    """Parse a whole buffer on the native kernel.

    Returns ``(recs, fi, ff, fu, src, consumed, n_errors)`` in the shared
    arena layout (see :mod:`pymavlink_mojo._layout`); ``src`` is the byte
    string that record offsets slice (the input for raw streams, the
    kernel's logical parse stream for tlogs). The returned arrays are
    private copies (the kernel handle is freed before returning).
    Raises :class:`NativeUnavailable` on kernel/load errors only.
    """
    lib = _load()  # raises NativeUnavailable
    t = _tables()
    n = len(data)
    # ctypes needs an addressable buffer even for empty input.
    buf = ctypes.create_string_buffer(bytes(data), max(n, 1))
    err = ctypes.c_int32(0)
    handle = lib.pymavmojo_parse(
        ctypes.cast(buf, _u8p),
        n,
        1 if tlog_mode else 0,
        t.n_msg,
        t.msg_ids,
        t.msg_crc_extra,
        t.msg_csize,
        t.msg_nflds,
        t.msg_foff,
        t.fld_woff,
        t.fld_type,
        t.fld_alen,
        ctypes.byref(err),
    )
    if not handle:
        raise NativeUnavailable(f"native parse failed with status {err.value}")
    try:
        consumed = int(lib.pymavmojo_consumed(handle))
        n_errors = int(lib.pymavmojo_error_count(handle))
        recs = _i64p()
        fi = _i64p()
        ff = _f64p()
        fu = _u8p()
        stream = _u8p()
        n_recs = ctypes.c_int64(0)
        n_fi = ctypes.c_int64(0)
        n_ff = ctypes.c_int64(0)
        n_fu = ctypes.c_int64(0)
        n_stream = ctypes.c_int64(0)
        rc = lib.pymavmojo_arrays(
            handle,
            ctypes.byref(recs),
            ctypes.byref(fi),
            ctypes.byref(ff),
            ctypes.byref(fu),
            ctypes.byref(stream),
            ctypes.byref(n_recs),
            ctypes.byref(n_fi),
            ctypes.byref(n_ff),
            ctypes.byref(n_fu),
            ctypes.byref(n_stream),
        )
        if rc != 0:
            raise NativeUnavailable(f"native array export failed with status {rc}")
        # Copy out of the kernel-owned arenas before freeing the handle.
        recs_arr = np.frombuffer(
            ctypes.string_at(recs, n_recs.value * 8), dtype=np.int64
        ).copy()
        fi_arr = np.frombuffer(ctypes.string_at(fi, n_fi.value * 8), dtype=np.int64).copy()
        ff_arr = np.frombuffer(ctypes.string_at(ff, n_ff.value * 8), dtype=np.float64).copy()
        fu_bytes = ctypes.string_at(fu, n_fu.value) if n_fu.value else b""
        # tlog parses point record offsets into the kernel's logical stream;
        # raw parses point into the input buffer itself.
        if tlog_mode:
            src = ctypes.string_at(stream, n_stream.value) if n_stream.value else b""
        else:
            src = data
        return recs_arr, fi_arr, ff_arr, fu_bytes, src, consumed, n_errors
    finally:
        lib.pymavmojo_free(handle)
