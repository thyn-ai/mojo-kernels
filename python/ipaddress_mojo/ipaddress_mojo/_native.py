"""ctypes loader for the ipaddressmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$IPADDRESS_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``ipaddress_mojo/_native/``
    3. the repository development build output ``kernels/ipaddress/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, the batch API falls back to the pure-Python /
NumPy implementations below (silently, with identical results). Set
``IPADDRESS_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  ipaddressmojo_abi_version(void)
    int32_t  ipaddressmojo_parse_batch(buf, offsets, n, out_version,
                                       out_hi, out_lo, out_scope_off,
                                       out_scope_len, out_err)
    int32_t  ipaddressmojo_contains_any_v4(net_lo, net_hi, n_nets,
                                           addrs, n_addrs, out)
    int32_t  ipaddressmojo_contains_any_v6(net_lo_hi, net_lo_lo, net_hi_hi,
                                           net_hi_lo, n_nets, addr_hi, addr_lo,
                                           n_addrs, out)
    int64_t  ipaddressmojo_collapse_v4(in_lo, in_plen, n, out_lo, out_plen)
    int64_t  ipaddressmojo_collapse_v6(in_lo_hi, in_lo_lo, in_plen, n,
                                       out_lo_hi, out_lo_lo, out_plen)

The kernel only decides accept/reject on malformed input (out_err != 0); the
wrapper re-parses the first failing item with the pure-Python reference so
raised exceptions are byte-identical to the stdlib's on every backend.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/ipaddress/src/ipaddressmojo.mojo.
ABI_VERSION = 1

_ENV_LIB = "IPADDRESS_MOJO_NATIVE_LIB"
_ENV_DISABLE = "IPADDRESS_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native ipaddressmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libipaddressmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libipaddressmojo.so"
    if sys.platform.startswith("win"):
        return "ipaddressmojo.dll"  # no Mojo toolchain builds this today
    return "libipaddressmojo.so"


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
                    here, "..", "..", "..", "kernels", "ipaddress", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.ipaddressmojo_abi_version.argtypes = []
    lib.ipaddressmojo_abi_version.restype = ctypes.c_int32
    lib.ipaddressmojo_parse_batch.argtypes = [
        u8p,  # buf
        i64p,  # offsets [n+1]
        ctypes.c_int64,  # n
        i32p,  # out_version [n]
        u64p,  # out_hi [n]
        u64p,  # out_lo [n]
        i32p,  # out_scope_off [n] (-1 = none)
        i32p,  # out_scope_len [n]
        i32p,  # out_err [n]
    ]
    lib.ipaddressmojo_parse_batch.restype = ctypes.c_int32
    lib.ipaddressmojo_contains_any_v4.argtypes = [
        u32p, u32p, ctypes.c_int64, u32p, ctypes.c_int64, u8p,
    ]
    lib.ipaddressmojo_contains_any_v4.restype = ctypes.c_int32
    lib.ipaddressmojo_contains_any_v6.argtypes = [
        u64p, u64p, u64p, u64p, ctypes.c_int64, u64p, u64p, ctypes.c_int64, u8p,
    ]
    lib.ipaddressmojo_contains_any_v6.restype = ctypes.c_int32
    lib.ipaddressmojo_collapse_v4.argtypes = [u32p, i32p, ctypes.c_int64, u32p, i32p]
    lib.ipaddressmojo_collapse_v4.restype = ctypes.c_int64
    lib.ipaddressmojo_collapse_v6.argtypes = [
        u64p, u64p, i32p, ctypes.c_int64, u64p, u64p, i32p,
    ]
    lib.ipaddressmojo_collapse_v6.restype = ctypes.c_int64


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
                    abi = int(lib.ipaddressmojo_abi_version())
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


def try_load() -> ctypes.CDLL | None:
    """The native kernel, or None when unavailable (never raises)."""
    try:
        return _load()
    except NativeUnavailable:
        return None


def native_available() -> bool:
    """True if the native kernel can serve right now. Never raises."""
    return try_load() is not None


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
    info["abi_version_native"] = int(lib.ipaddressmojo_abi_version())
    return info


# ---------------------------------------------------------------------------
# Batch parse (native kernel; caller re-parses failures for exact errors)
# ---------------------------------------------------------------------------


def parse_many(lib: ctypes.CDLL, items: list) -> list:
    """Native parse of a list of inputs; returns address objects.

    Non-string items are parsed by the pure-Python reference (they are cheap
    and rare in bulk workloads). The first kernel-reported failure is
    re-parsed with the reference so the raised exception matches the stdlib
    exactly.
    """
    from ipaddress_mojo.core import IPv4Address, IPv6Address, ip_address

    n = len(items)
    results: list = [None] * n
    str_positions: list[int] = []
    chunks: list[bytes] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            str_positions.append(i)
            chunks.append(item.encode("utf-8"))
        else:
            results[i] = ip_address(item)
    n_str = len(chunks)
    if n_str == 0:
        return results
    buf = b"".join(chunks)
    offsets = np.zeros(n_str + 1, dtype=np.int64)
    pos = 0
    for k, chunk in enumerate(chunks):
        pos += len(chunk)
        offsets[k + 1] = pos
    out_version = np.zeros(n_str, dtype=np.int32)
    out_hi = np.zeros(n_str, dtype=np.uint64)
    out_lo = np.zeros(n_str, dtype=np.uint64)
    out_scope_off = np.zeros(n_str, dtype=np.int32)
    out_scope_len = np.zeros(n_str, dtype=np.int32)
    out_err = np.zeros(n_str, dtype=np.int32)

    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    buf_arr = np.frombuffer(buf, dtype=np.uint8) if buf else np.zeros(1, dtype=np.uint8)
    rc = lib.ipaddressmojo_parse_batch(
        buf_arr.ctypes.data_as(u8p),
        offsets.ctypes.data_as(i64p),
        n_str,
        out_version.ctypes.data_as(i32p),
        out_hi.ctypes.data_as(u64p),
        out_lo.ctypes.data_as(u64p),
        out_scope_off.ctypes.data_as(i32p),
        out_scope_len.ctypes.data_as(i32p),
        out_err.ctypes.data_as(i32p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native batch parse failed with status {rc}")
    bad = np.nonzero(out_err)[0]
    if bad.size:
        # Re-parse the first failing item with the reference: the exception
        # it raises is byte-identical to the stdlib's on every backend.
        ip_address(items[str_positions[int(bad[0])]])
        raise AssertionError("reference parse accepted a kernel-rejected item")
    for k, i in enumerate(str_positions):
        version = int(out_version[k])
        if version == 4:
            results[i] = IPv4Address(int(out_lo[k]))
        else:
            soff = int(out_scope_off[k])
            scope = None
            if soff >= 0:
                slen = int(out_scope_len[k])
                scope = chunks[k][soff : soff + slen].decode("utf-8")
            results[i] = IPv6Address._from_parts(
                (int(out_hi[k]) << 64) | int(out_lo[k]), scope
            )
    return results


# ---------------------------------------------------------------------------
# Batch membership (native kernel or NumPy fallback; identical results)
# ---------------------------------------------------------------------------


def _split_by_version(networks, addresses):
    """Partition into v4/v6 (net_lo, net_hi) and (value, out_index) groups."""
    v4_lo: list[int] = []
    v4_hi: list[int] = []
    v6_lo: list[int] = []
    v6_hi: list[int] = []
    for net in networks:
        if net._version == 4:
            v4_lo.append(net.network_address._ip)
            v4_hi.append(net.broadcast_address._ip)
        else:
            v6_lo.append(net.network_address._ip)
            v6_hi.append(net.broadcast_address._ip)
    v4_addr: list[int] = []
    v4_idx: list[int] = []
    v6_addr: list[int] = []
    v6_idx: list[int] = []
    for i, addr in enumerate(addresses):
        if addr._version == 4:
            v4_addr.append(addr._ip)
            v4_idx.append(i)
        else:
            v6_addr.append(addr._ip)
            v6_idx.append(i)
    return (v4_lo, v4_hi, v4_addr, v4_idx), (v6_lo, v6_hi, v6_addr, v6_idx)


def _contains_any_np_1d(net_lo: np.ndarray, net_hi: np.ndarray, addrs: np.ndarray) -> np.ndarray:
    """sort + prefix-max interval stabbing (NumPy fallback for v4)."""
    out = np.zeros(addrs.shape[0], dtype=bool)
    if net_lo.shape[0] == 0 or addrs.shape[0] == 0:
        return out
    order = np.argsort(net_lo, kind="stable")
    lo = net_lo[order]
    maxhi = np.maximum.accumulate(net_hi[order])
    idx = np.searchsorted(lo, addrs, side="right") - 1
    mask = idx >= 0
    out[mask] = maxhi[idx[mask]] >= addrs[mask]
    return out


def _pairs_to_void(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    arr = np.empty(hi.shape[0], dtype=[("hi", "<u8"), ("lo", "<u8")])
    arr["hi"] = hi
    arr["lo"] = lo
    return arr


def _contains_any_np_2d(
    net_lo_hi: np.ndarray, net_lo_lo: np.ndarray,
    net_hi_hi: np.ndarray, net_hi_lo: np.ndarray,
    addr_hi: np.ndarray, addr_lo: np.ndarray,
) -> np.ndarray:
    """Same stabbing for 128-bit keys held as (hi, lo) uint64 pairs."""
    out = np.zeros(addr_hi.shape[0], dtype=bool)
    n_nets = net_lo_hi.shape[0]
    if n_nets == 0 or addr_hi.shape[0] == 0:
        return out
    lo_void = _pairs_to_void(net_lo_hi, net_lo_lo)
    order = np.argsort(lo_void, kind="stable")
    lo_sorted = lo_void[order]
    hi_hi_s = net_hi_hi[order]
    hi_lo_s = net_hi_lo[order]
    # Prefix lexicographic max of (hi, lo). numpy has no running-max ufunc for
    # 128-bit keys, so this one pass stays scalar (still O(n); the stabbing
    # queries below stay vectorized).
    mh_hi = np.empty(n_nets, dtype=np.uint64)
    mh_lo = np.empty(n_nets, dtype=np.uint64)
    b_hi = 0
    b_lo = 0
    for i in range(n_nets):
        h_hi = int(hi_hi_s[i])
        h_lo = int(hi_lo_s[i])
        if i == 0 or h_hi > b_hi or (h_hi == b_hi and h_lo > b_lo):
            b_hi = h_hi
            b_lo = h_lo
        mh_hi[i] = b_hi
        mh_lo[i] = b_lo
    addr_void = _pairs_to_void(addr_hi, addr_lo)
    idx = np.searchsorted(lo_sorted, addr_void, side="right") - 1
    mask = idx >= 0
    ii = idx[mask]
    out[mask] = (mh_hi[ii] > addr_hi[mask]) | (
        (mh_hi[ii] == addr_hi[mask]) & (mh_lo[ii] >= addr_lo[mask])
    )
    return out


def contains_many(networks: list, addresses: list) -> np.ndarray:
    """bool per address: contained in ANY network (cross-version never)."""
    out = np.zeros(len(addresses), dtype=bool)
    (v4_lo, v4_hi, v4_addr, v4_idx), (v6_lo, v6_hi, v6_addr, v6_idx) = (
        _split_by_version(networks, addresses)
    )
    lib = try_load()

    if v4_addr:
        v4_lo_a = np.asarray(v4_lo, dtype=np.uint32)
        v4_hi_a = np.asarray(v4_hi, dtype=np.uint32)
        v4_addr_a = np.asarray(v4_addr, dtype=np.uint32)
        if lib is not None:
            res = np.zeros(v4_addr_a.shape[0], dtype=np.uint8)
            u32p = ctypes.POINTER(ctypes.c_uint32)
            u8p = ctypes.POINTER(ctypes.c_uint8)
            rc = lib.ipaddressmojo_contains_any_v4(
                v4_lo_a.ctypes.data_as(u32p),
                v4_hi_a.ctypes.data_as(u32p),
                v4_lo_a.shape[0],
                v4_addr_a.ctypes.data_as(u32p),
                v4_addr_a.shape[0],
                res.ctypes.data_as(u8p),
            )
            if rc != 0:
                raise NativeUnavailable(f"native contains_any_v4 failed: {rc}")
            out[np.asarray(v4_idx)] = res.astype(bool)
        else:
            out[np.asarray(v4_idx)] = _contains_any_np_1d(v4_lo_a, v4_hi_a, v4_addr_a)

    if v6_addr:
        lo_hi = np.asarray([v >> 64 for v in v6_lo], dtype=np.uint64)
        lo_lo = np.asarray([v & 0xFFFFFFFFFFFFFFFF for v in v6_lo], dtype=np.uint64)
        hi_hi = np.asarray([v >> 64 for v in v6_hi], dtype=np.uint64)
        hi_lo = np.asarray([v & 0xFFFFFFFFFFFFFFFF for v in v6_hi], dtype=np.uint64)
        a_hi = np.asarray([v >> 64 for v in v6_addr], dtype=np.uint64)
        a_lo = np.asarray([v & 0xFFFFFFFFFFFFFFFF for v in v6_addr], dtype=np.uint64)
        if lib is not None:
            res = np.zeros(a_hi.shape[0], dtype=np.uint8)
            u64p = ctypes.POINTER(ctypes.c_uint64)
            u8p = ctypes.POINTER(ctypes.c_uint8)
            rc = lib.ipaddressmojo_contains_any_v6(
                lo_hi.ctypes.data_as(u64p),
                lo_lo.ctypes.data_as(u64p),
                hi_hi.ctypes.data_as(u64p),
                hi_lo.ctypes.data_as(u64p),
                lo_hi.shape[0],
                a_hi.ctypes.data_as(u64p),
                a_lo.ctypes.data_as(u64p),
                a_hi.shape[0],
                res.ctypes.data_as(u8p),
            )
            if rc != 0:
                raise NativeUnavailable(f"native contains_any_v6 failed: {rc}")
            out[np.asarray(v6_idx)] = res.astype(bool)
        else:
            out[np.asarray(v6_idx)] = _contains_any_np_2d(
                lo_hi, lo_lo, hi_hi, hi_lo, a_hi, a_lo
            )
    return out


# ---------------------------------------------------------------------------
# Batch collapse (native kernel or reference merge; identical results)
# ---------------------------------------------------------------------------


def collapse(items: list[tuple[int, int]], bits: int, version: int):
    """Collapse [(base_int, prefixlen)] into the canonical minimal cover."""
    from ipaddress_mojo import _reference

    n = len(items)
    if n == 0:
        return []
    lib = try_load()
    if lib is None:
        return _reference.collapse_intervals(items, bits)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    i32p = ctypes.POINTER(ctypes.c_int32)
    if version == 4:
        in_lo = np.asarray([b for b, _ in items], dtype=np.uint32)
        in_plen = np.asarray([p for _, p in items], dtype=np.int32)
        out_lo = np.empty(n, dtype=np.uint32)
        out_plen = np.empty(n, dtype=np.int32)
        count = lib.ipaddressmojo_collapse_v4(
            in_lo.ctypes.data_as(u32p),
            in_plen.ctypes.data_as(i32p),
            n,
            out_lo.ctypes.data_as(u32p),
            out_plen.ctypes.data_as(i32p),
        )
        if count < 0:
            raise NativeUnavailable(f"native collapse_v4 failed: {count}")
        return [(int(out_lo[i]), int(out_plen[i])) for i in range(count)]
    in_lo_hi = np.asarray([b >> 64 for b, _ in items], dtype=np.uint64)
    in_lo_lo = np.asarray([b & 0xFFFFFFFFFFFFFFFF for b, _ in items], dtype=np.uint64)
    in_plen = np.asarray([p for _, p in items], dtype=np.int32)
    out_lo_hi = np.empty(n, dtype=np.uint64)
    out_lo_lo = np.empty(n, dtype=np.uint64)
    out_plen = np.empty(n, dtype=np.int32)
    count = lib.ipaddressmojo_collapse_v6(
        in_lo_hi.ctypes.data_as(u64p),
        in_lo_lo.ctypes.data_as(u64p),
        in_plen.ctypes.data_as(i32p),
        n,
        out_lo_hi.ctypes.data_as(u64p),
        out_lo_lo.ctypes.data_as(u64p),
        out_plen.ctypes.data_as(i32p),
    )
    if count < 0:
        raise NativeUnavailable(f"native collapse_v6 failed: {count}")
    return [
        ((int(out_lo_hi[i]) << 64) | int(out_lo_lo[i]), int(out_plen[i]))
        for i in range(count)
    ]
