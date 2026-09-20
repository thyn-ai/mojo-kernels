"""ctypes loader for the volatility3hivemojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$VOL_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``vol_mojo/_native/``
    3. the repository development build output ``kernels/volatility3-hive/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``VOL_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t volatility3hivemojo_abi_version(void)

    int32_t volatility3hivemojo_hive_walk_count(
        data, data_len, root_cell, storage_len, volatile_len,
        out_n_nodes, out_n_name_bytes)
    int32_t volatility3hivemojo_hive_walk_fill(
        data, data_len, root_cell, storage_len, volatile_len,
        cap_nodes, cap_name_bytes,
        out_offsets, out_name_off, out_name_len, out_name_bytes, out_n_nodes)

    int32_t volatility3hivemojo_pool_scan_count(
        data, data_len, n_constraints, tag_blob, tag_off, tag_len,
        size_min, size_max, page_type, index_min, index_max,
        alignment, layout, vista_semantics, out_n_hits)
    int32_t volatility3hivemojo_pool_scan_fill(
        ... same ..., cap_hits, out_offsets, out_tag_idx, out_n_hits)

Status codes: 0 ok; 1 invalid arguments; 2 malformed hive (fatal, mirrors the
oracle raising); 4 output capacity exceeded; 5 iteration guard tripped.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/volatility3-hive/src/volatility3hivemojo.mojo.
# A mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

# Kernel status codes shared with core.py.
STATUS_OK = 0
STATUS_INVALID = 1
STATUS_MALFORMED = 2
STATUS_CAPACITY = 4
STATUS_GUARD = 5

LAYOUT_X64 = 0
LAYOUT_X86 = 1

_ENV_LIB = "VOL_MOJO_NATIVE_LIB"
_ENV_DISABLE = "VOL_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native volatility3hivemojo kernel could not be found, loaded, or verified."""


class NativeKernelError(RuntimeError):  # noqa: N818
    """The native kernel reported a failure status for a call."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libvolatility3hivemojo.dylib"
    if sys.platform.startswith("linux"):
        return "libvolatility3hivemojo.so"
    if sys.platform.startswith("win"):
        return "volatility3hivemojo.dll"  # no Mojo toolchain builds this today
    return "libvolatility3hivemojo.so"


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
                    here, "..", "..", "..", "kernels", "volatility3-hive", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.volatility3hivemojo_abi_version.argtypes = []
    lib.volatility3hivemojo_abi_version.restype = ctypes.c_int32

    hive_common = [
        u8p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_uint64,  # root_cell
        ctypes.c_uint64,  # storage_len
        ctypes.c_uint64,  # volatile_len
    ]
    lib.volatility3hivemojo_hive_walk_count.argtypes = hive_common + [
        i64p,  # out_n_nodes
        i64p,  # out_n_name_bytes
    ]
    lib.volatility3hivemojo_hive_walk_count.restype = ctypes.c_int32
    lib.volatility3hivemojo_hive_walk_fill.argtypes = hive_common + [
        ctypes.c_int64,  # cap_nodes
        ctypes.c_int64,  # cap_name_bytes
        i64p,  # out_offsets
        i64p,  # out_name_off
        i32p,  # out_name_len
        u8p,  # out_name_bytes
        i64p,  # out_n_nodes
    ]
    lib.volatility3hivemojo_hive_walk_fill.restype = ctypes.c_int32

    pool_common = [
        u8p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_int64,  # n_constraints
        u8p,  # tag_blob
        i32p,  # tag_off
        i32p,  # tag_len
        i64p,  # size_min
        i64p,  # size_max
        i32p,  # page_type
        i64p,  # index_min
        i64p,  # index_max
        ctypes.c_int64,  # alignment
        ctypes.c_int32,  # layout
        ctypes.c_int32,  # vista_semantics
    ]
    lib.volatility3hivemojo_pool_scan_count.argtypes = pool_common + [
        i64p,  # out_n_hits
    ]
    lib.volatility3hivemojo_pool_scan_count.restype = ctypes.c_int32
    lib.volatility3hivemojo_pool_scan_fill.argtypes = pool_common + [
        ctypes.c_int64,  # cap_hits
        i64p,  # out_offsets
        i32p,  # out_tag_idx
        i64p,  # out_n_hits
    ]
    lib.volatility3hivemojo_pool_scan_fill.restype = ctypes.c_int32


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
                    abi = int(lib.volatility3hivemojo_abi_version())
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
    """True if the native kernel can evaluate right now. Never raises."""
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
    info["abi_version_native"] = int(lib.volatility3hivemojo_abi_version())
    return info


def _check_status(rc: int, what: str) -> None:
    """Map a kernel status to an exception (STATUS_OK is a no-op)."""
    if rc == STATUS_OK:
        return
    if rc == STATUS_MALFORMED:
        raise NativeKernelError(f"{what}: malformed hive image (status 2)")
    if rc == STATUS_CAPACITY:
        raise NativeKernelError(f"{what}: output capacity exceeded (status 4)")
    if rc == STATUS_GUARD:
        raise NativeKernelError(f"{what}: iteration guard tripped (status 5)")
    raise NativeKernelError(f"{what}: kernel status {rc}")


def walk_hive_native(
    data: bytes,
    root_cell: int,
    storage_len: int,
    volatile_len: int,
) -> tuple[list[int], list[bytes]]:
    """Run the hive walk on the native kernel.

    Returns (node_offsets, name_bytes_per_node). Raises NativeUnavailable if
    the kernel cannot be used at all and NativeKernelError on a failure
    status; the caller maps status 2 onto the shared HiveError path.
    """
    lib = _load()  # raises NativeUnavailable
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    n = ctypes.c_int64(0)
    nb = ctypes.c_int64(0)
    rc = lib.volatility3hivemojo_hive_walk_count(
        buf,
        ctypes.c_int64(len(data)),
        ctypes.c_uint64(root_cell),
        ctypes.c_uint64(storage_len),
        ctypes.c_uint64(volatile_len),
        ctypes.byref(n),
        ctypes.byref(nb),
    )
    _check_status(rc, "hive walk count")
    count, nbytes = n.value, nb.value
    offsets = (ctypes.c_int64 * max(count, 1))()
    name_off = (ctypes.c_int64 * max(count, 1))()
    name_len = (ctypes.c_int32 * max(count, 1))()
    name_blob = (ctypes.c_uint8 * max(nbytes, 1))()
    out_n = ctypes.c_int64(0)
    rc = lib.volatility3hivemojo_hive_walk_fill(
        buf,
        ctypes.c_int64(len(data)),
        ctypes.c_uint64(root_cell),
        ctypes.c_uint64(storage_len),
        ctypes.c_uint64(volatile_len),
        ctypes.c_int64(count),
        ctypes.c_int64(nbytes),
        offsets,
        name_off,
        name_len,
        name_blob,
        ctypes.byref(out_n),
    )
    _check_status(rc, "hive walk fill")
    blob = bytes(name_blob)
    names = [
        blob[name_off[i] : name_off[i] + name_len[i]] for i in range(out_n.value)
    ]
    return [int(offsets[i]) for i in range(out_n.value)], names


def scan_pool_headers_native(
    data: bytes,
    constraints_sorted: list[tuple[bytes, int, int, int, int, int]],
    alignment: int,
    layout: int,
    vista_semantics: bool,
) -> list[tuple[int, int]]:
    """Run the pool-header scan on the native kernel.

    ``constraints_sorted`` holds (tag, size_min, size_max, page_type,
    index_min, index_max) rows pre-sorted with descending tag length; bounds
    of 0 mean "no bound" and page_type 0 means "no pool-type test". Returns
    (header_offset, constraint_index) pairs in scan order. Raises
    NativeUnavailable if the kernel cannot be used and NativeKernelError on
    a failure status.
    """
    lib = _load()  # raises NativeUnavailable
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    blob_bytes = b"".join(row[0] for row in constraints_sorted) or b"\x00"
    tag_blob = (ctypes.c_uint8 * len(blob_bytes)).from_buffer_copy(blob_bytes)
    n_con = len(constraints_sorted)
    tag_off = (ctypes.c_int32 * max(n_con, 1))()
    tag_len = (ctypes.c_int32 * max(n_con, 1))()
    size_min = (ctypes.c_int64 * max(n_con, 1))()
    size_max = (ctypes.c_int64 * max(n_con, 1))()
    page_type = (ctypes.c_int32 * max(n_con, 1))()
    index_min = (ctypes.c_int64 * max(n_con, 1))()
    index_max = (ctypes.c_int64 * max(n_con, 1))()
    cursor = 0
    for i, row in enumerate(constraints_sorted):
        tag_off[i] = cursor
        tag_len[i] = len(row[0])
        size_min[i] = row[1]
        size_max[i] = row[2]
        page_type[i] = row[3]
        index_min[i] = row[4]
        index_max[i] = row[5]
        cursor += len(row[0])

    common = (
        buf,
        ctypes.c_int64(len(data)),
        ctypes.c_int64(n_con),
        tag_blob,
        tag_off,
        tag_len,
        size_min,
        size_max,
        page_type,
        index_min,
        index_max,
        ctypes.c_int64(alignment),
        ctypes.c_int32(layout),
        ctypes.c_int32(1 if vista_semantics else 0),
    )
    n_hits = ctypes.c_int64(0)
    rc = lib.volatility3hivemojo_pool_scan_count(*common, ctypes.byref(n_hits))
    _check_status(rc, "pool scan count")
    offsets = (ctypes.c_int64 * max(n_hits.value, 1))()
    tag_idx = (ctypes.c_int32 * max(n_hits.value, 1))()
    out_n = ctypes.c_int64(0)
    rc = lib.volatility3hivemojo_pool_scan_fill(
        *common, ctypes.c_int64(n_hits.value), offsets, tag_idx, ctypes.byref(out_n)
    )
    _check_status(rc, "pool scan fill")
    return [(int(offsets[i]), int(tag_idx[i])) for i in range(out_n.value)]
