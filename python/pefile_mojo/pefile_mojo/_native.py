"""ctypes loader for the pefilemojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$PEFILE_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``pefile_mojo/_native/``
    3. the repository development build output ``kernels/pefile/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``PEFILE_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  pefilemojo_abi_version(void)
    int32_t  pefilemojo_checksum(const uint8_t* data, int64_t len,
                                 int64_t checksum_offset, uint64_t* out)
    int32_t  pefilemojo_imports_count(data, len, hdr..., sections..., counts[3])
    int32_t  pefilemojo_imports_fill(data, len, hdr..., sections..., caps, outs...)

All pointer arguments must be valid (never NULL) when the associated
count/length is positive; the wrapper allocates one-slot dummies for empty
buffers. The kernel never retains the passed buffers.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/pefile/src/pefilemojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back to the pure-Python reference.
ABI_VERSION = 1

_ENV_LIB = "PEFILE_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PEFILE_MOJO_DISABLE_NATIVE"

# Kernel status codes.
_ERR_OK = 0


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native pefilemojo kernel could not be found, loaded, or verified."""


class NativeCallError(RuntimeError):  # noqa: N818
    """The native kernel rejected an input it was contractually given."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libpefilemojo.dylib"
    if sys.platform.startswith("linux"):
        return "libpefilemojo.so"
    if sys.platform.startswith("win"):
        return "pefilemojo.dll"  # no Mojo toolchain builds this today
    return "libpefilemojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "pefile", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.pefilemojo_abi_version.argtypes = []
    lib.pefilemojo_abi_version.restype = ctypes.c_int32
    lib.pefilemojo_checksum.argtypes = [u8p, ctypes.c_int64, ctypes.c_int64, u64p]
    lib.pefilemojo_checksum.restype = ctypes.c_int32
    _hdr = [
        u8p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_int32,  # pe_type
        ctypes.c_uint64,  # image_base
        ctypes.c_int64,  # import_dir_rva
        ctypes.c_int64,  # section_alignment
        ctypes.c_int64,  # file_alignment
        ctypes.c_int64,  # header_len
        ctypes.c_int64,  # n_sections
        u32p,  # sec_vaddr
        u32p,  # sec_vsize
        u32p,  # sec_rawsize
        u32p,  # sec_rawptr
    ]
    lib.pefilemojo_imports_count.argtypes = _hdr + [i64p]
    lib.pefilemojo_imports_count.restype = ctypes.c_int32
    lib.pefilemojo_imports_fill.argtypes = _hdr + [
        ctypes.c_int64,  # cap_desc
        ctypes.c_int64,  # cap_syms
        ctypes.c_int64,  # cap_pool
        i64p,  # desc_dll_off
        i64p,  # desc_dll_len
        i64p,  # desc_sym_start
        i64p,  # desc_sym_count
        u8p,  # pool
        i64p,  # sym_name_off
        i64p,  # sym_name_len
        i64p,  # sym_ordinal
        i64p,  # sym_hint
        u64p,  # sym_address
        u64p,  # sym_bound
    ]
    lib.pefilemojo_imports_fill.restype = ctypes.c_int32


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
                abi = int(lib.pefilemojo_abi_version())
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
    """True if the native kernel can be used right now. Never raises."""
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
    info["abi_version_native"] = int(lib.pefilemojo_abi_version())
    return info


def _as_u8_buffer(data: bytes | bytearray | memoryview) -> tuple:
    """Pin the caller's bytes for the duration of one kernel call."""
    view = memoryview(data)
    if view.ndim != 1:
        raise ValueError("data must be a flat bytes-like object")
    if len(view) == 0:
        return (ctypes.c_uint8 * 1)(), 0
    if view.readonly:
        buf = (ctypes.c_uint8 * len(view)).from_buffer_copy(view)
        return buf, len(view)
    buf = (ctypes.c_uint8 * len(view)).from_buffer(view)
    return buf, len(view)


def native_checksum(data: bytes | bytearray | memoryview, checksum_offset: int) -> int:
    """PE checksum via the native kernel. Raises NativeUnavailable on load
    failure, NativeCallError on contract violations."""
    lib = _load()
    buf, n = _as_u8_buffer(data)
    out = ctypes.c_uint64(0)
    rc = lib.pefilemojo_checksum(buf, n, checksum_offset, ctypes.byref(out))
    if rc != _ERR_OK:
        raise NativeCallError(f"pefilemojo_checksum failed with status {rc}")
    return int(out.value)


def _dummy_u32() -> ctypes.Array:
    return (ctypes.c_uint32 * 1)(0)


def native_parse_imports(header: dict) -> dict:
    """Two-phase native import walk.

    ``header`` is the wrapper-parsed header bundle (see core._parse_headers).
    Returns the flat result dict (arrays + string pool); the wrapper turns it
    into dataclasses. Raises NativeUnavailable / NativeCallError.
    """
    lib = _load()
    buf, n = _as_u8_buffer(header["data"])
    sections = header["sections"]
    n_sec = len(sections)
    if n_sec:
        va = (ctypes.c_uint32 * n_sec)(*[s[0] for s in sections])
        vs = (ctypes.c_uint32 * n_sec)(*[s[1] for s in sections])
        rs = (ctypes.c_uint32 * n_sec)(*[s[2] for s in sections])
        rp = (ctypes.c_uint32 * n_sec)(*[s[3] for s in sections])
    else:
        va = vs = rs = rp = _dummy_u32()
    hdr = (
        buf,
        n,
        header["pe_type"],
        header["image_base"],
        header["import_dir_rva"],
        header["section_alignment"],
        header["file_alignment"],
        header["header_len"],
        n_sec,
        va,
        vs,
        rs,
        rp,
    )
    counts = (ctypes.c_int64 * 3)()
    rc = lib.pefilemojo_imports_count(*hdr, counts)
    if rc != _ERR_OK:
        raise NativeCallError(f"pefilemojo_imports_count failed with status {rc}")
    n_desc, n_syms, pool_bytes = (max(int(c), 0) for c in counts)

    i64 = ctypes.c_int64
    u64 = ctypes.c_uint64
    desc_dll_off = (i64 * max(n_desc, 1))()
    desc_dll_len = (i64 * max(n_desc, 1))()
    desc_sym_start = (i64 * max(n_desc, 1))()
    desc_sym_count = (i64 * max(n_desc, 1))()
    pool = (ctypes.c_uint8 * max(pool_bytes, 1))()
    sym_name_off = (i64 * max(n_syms, 1))()
    sym_name_len = (i64 * max(n_syms, 1))()
    sym_ordinal = (i64 * max(n_syms, 1))()
    sym_hint = (i64 * max(n_syms, 1))()
    sym_address = (u64 * max(n_syms, 1))()
    sym_bound = (u64 * max(n_syms, 1))()
    rc = lib.pefilemojo_imports_fill(
        *hdr,
        n_desc,
        n_syms,
        pool_bytes,
        desc_dll_off,
        desc_dll_len,
        desc_sym_start,
        desc_sym_count,
        pool,
        sym_name_off,
        sym_name_len,
        sym_ordinal,
        sym_hint,
        sym_address,
        sym_bound,
    )
    if rc != _ERR_OK:
        raise NativeCallError(f"pefilemojo_imports_fill failed with status {rc}")
    return {
        "n_desc": n_desc,
        "n_syms": n_syms,
        "desc_dll_off": desc_dll_off,
        "desc_dll_len": desc_dll_len,
        "desc_sym_start": desc_sym_start,
        "desc_sym_count": desc_sym_count,
        "pool": pool,
        "sym_name_off": sym_name_off,
        "sym_name_len": sym_name_len,
        "sym_ordinal": sym_ordinal,
        "sym_hint": sym_hint,
        "sym_address": sym_address,
        "sym_bound": sym_bound,
    }
