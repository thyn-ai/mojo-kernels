"""ctypes loader for the difflibmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$DIFFLIB_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``difflib_mojo/_native/``
    3. the repository development build output ``kernels/difflib/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python engine. Set
``DIFFLIB_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  difflibmojo_abi_version(void)
    int64_t  difflibmojo_matching_blocks(a, la, b, lb, b_junk, b_allowed,
                                         out, out_cap)
    int32_t  difflibmojo_find_longest_match(a, la, b, lb, b_junk, b_allowed,
                                            alo, ahi, blo, bhi, out3)
    double   difflibmojo_quick_ratio(a, la, b, lb)
    int32_t  difflibmojo_close_matches_batch(words, words_off, nw,
                                             cands, cands_off, nc,
                                             cutoff, autojunk,
                                             pass_out, ratio_out)

Sequences cross the ABI as int32 codepoint arrays (the wrapper encodes
``str`` as UTF-32-LE); ``b_junk``/``b_allowed`` are per-position uint8 masks
the wrapper derives from the isjunk callable and the autojunk popularity
rule. See kernels/difflib/src/difflibmojo.mojo for the exact contract.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/difflib/src/difflibmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back to the pure-Python engine.
ABI_VERSION = 1

_ENV_LIB = "DIFFLIB_MOJO_NATIVE_LIB"
_ENV_DISABLE = "DIFFLIB_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native difflibmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libdifflibmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libdifflibmojo.so"
    if sys.platform.startswith("win"):
        return "difflibmojo.dll"  # no Mojo toolchain builds this today
    return "libdifflibmojo.so"


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
                    here, "..", "..", "..", "kernels", "difflib", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.difflibmojo_abi_version.argtypes = []
    lib.difflibmojo_abi_version.restype = ctypes.c_int32
    lib.difflibmojo_matching_blocks.argtypes = [
        i32p,  # a codepoints[la]
        ctypes.c_int64,  # la
        i32p,  # b codepoints[lb]
        ctypes.c_int64,  # lb
        u8p,  # b_junk[lb]
        u8p,  # b_allowed[lb]
        i64p,  # out[out_cap]
        ctypes.c_int64,  # out_cap
    ]
    lib.difflibmojo_matching_blocks.restype = ctypes.c_int64
    lib.difflibmojo_find_longest_match.argtypes = [
        i32p,  # a codepoints[la]
        ctypes.c_int64,  # la
        i32p,  # b codepoints[lb]
        ctypes.c_int64,  # lb
        u8p,  # b_junk[lb]
        u8p,  # b_allowed[lb]
        ctypes.c_int64,  # alo
        ctypes.c_int64,  # ahi
        ctypes.c_int64,  # blo
        ctypes.c_int64,  # bhi
        i64p,  # out3[3]
    ]
    lib.difflibmojo_find_longest_match.restype = ctypes.c_int32
    lib.difflibmojo_quick_ratio.argtypes = [
        i32p,  # a codepoints[la]
        ctypes.c_int64,  # la
        i32p,  # b codepoints[lb]
        ctypes.c_int64,  # lb
    ]
    lib.difflibmojo_quick_ratio.restype = ctypes.c_double
    lib.difflibmojo_close_matches_batch.argtypes = [
        i32p,  # words codepoints, flattened
        i64p,  # words_off[nw+1]
        ctypes.c_int64,  # nw
        i32p,  # cands codepoints, flattened
        i64p,  # cands_off[nc+1]
        ctypes.c_int64,  # nc
        ctypes.c_double,  # cutoff
        ctypes.c_int32,  # autojunk
        u8p,  # pass_out[nw*nc]
        f64p,  # ratio_out[nw*nc]
    ]
    lib.difflibmojo_close_matches_batch.restype = ctypes.c_int32


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
                    abi = int(lib.difflibmojo_abi_version())
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
    info["abi_version_native"] = int(lib.difflibmojo_abi_version())
    return info


# ---------------------------------------------------------------------------
# Typed entry points (all raise NativeUnavailable on any kernel failure)
# ---------------------------------------------------------------------------

_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)
_u8p = ctypes.POINTER(ctypes.c_uint8)
_f64p = ctypes.POINTER(ctypes.c_double)


def _cp(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    if arr.shape[0] == 0:
        # ctypes still hands over a valid (never dereferenced) pointer.
        return np.empty(0, dtype=np.int32)
    return arr


def _mask(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.uint8)


def matching_blocks(
    a_cp: np.ndarray,
    b_cp: np.ndarray,
    b_junk: np.ndarray,
    b_allowed: np.ndarray,
) -> np.ndarray:
    """Raw matching-block triples (discovery order), int64 (n, 3).

    The caller sorts, collapses adjacent blocks and appends the
    (len(a), len(b), 0) sentinel — the same post-processing the reference
    performs. Raises NativeUnavailable on any failure.
    """
    lib = _load()
    a_cp = _cp(a_cp)
    b_cp = _cp(b_cp)
    b_junk = _mask(b_junk)
    b_allowed = _mask(b_allowed)
    la, lb = a_cp.shape[0], b_cp.shape[0]
    cap = 3 * (min(la, lb) + 1)
    out = np.empty(cap, dtype=np.int64)
    n = lib.difflibmojo_matching_blocks(
        a_cp.ctypes.data_as(_i32p),
        ctypes.c_int64(la),
        b_cp.ctypes.data_as(_i32p),
        ctypes.c_int64(lb),
        b_junk.ctypes.data_as(_u8p),
        b_allowed.ctypes.data_as(_u8p),
        out.ctypes.data_as(_i64p),
        ctypes.c_int64(cap),
    )
    if n < 0:
        raise NativeUnavailable(f"native matching_blocks failed with status {n}")
    return out[: 3 * n].reshape(n, 3)


def find_longest_match(
    a_cp: np.ndarray,
    b_cp: np.ndarray,
    b_junk: np.ndarray,
    b_allowed: np.ndarray,
    alo: int,
    ahi: int,
    blo: int,
    bhi: int,
) -> tuple[int, int, int]:
    """One find_longest_match call; returns the (i, j, k) triple."""
    lib = _load()
    a_cp = _cp(a_cp)
    b_cp = _cp(b_cp)
    b_junk = _mask(b_junk)
    b_allowed = _mask(b_allowed)
    out3 = np.zeros(3, dtype=np.int64)
    rc = lib.difflibmojo_find_longest_match(
        a_cp.ctypes.data_as(_i32p),
        ctypes.c_int64(a_cp.shape[0]),
        b_cp.ctypes.data_as(_i32p),
        ctypes.c_int64(b_cp.shape[0]),
        b_junk.ctypes.data_as(_u8p),
        b_allowed.ctypes.data_as(_u8p),
        ctypes.c_int64(alo),
        ctypes.c_int64(ahi),
        ctypes.c_int64(blo),
        ctypes.c_int64(bhi),
        out3.ctypes.data_as(_i64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native find_longest_match failed with status {rc}")
    return int(out3[0]), int(out3[1]), int(out3[2])


def quick_ratio(a_cp: np.ndarray, b_cp: np.ndarray) -> float:
    """Multiset-intersection upper bound, identical to quick_ratio()."""
    lib = _load()
    a_cp = _cp(a_cp)
    b_cp = _cp(b_cp)
    return float(
        lib.difflibmojo_quick_ratio(
            a_cp.ctypes.data_as(_i32p),
            ctypes.c_int64(a_cp.shape[0]),
            b_cp.ctypes.data_as(_i32p),
            ctypes.c_int64(b_cp.shape[0]),
        )
    )


def close_matches_batch(
    words_flat: np.ndarray,
    words_off: np.ndarray,
    cands_flat: np.ndarray,
    cands_off: np.ndarray,
    cutoff: float,
    autojunk: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Score nw words against nc candidates with the get_close_matches
    three-tier filter. Returns (pass[nw, nc] uint8, ratio[nw, nc] float64);
    ratio[w, c] is valid exactly when pass[w, c] is 1."""
    lib = _load()
    words_flat = _cp(words_flat)
    cands_flat = _cp(cands_flat)
    words_off = np.ascontiguousarray(words_off, dtype=np.int64)
    cands_off = np.ascontiguousarray(cands_off, dtype=np.int64)
    nw = words_off.shape[0] - 1
    nc = cands_off.shape[0] - 1
    passes = np.zeros(nw * nc, dtype=np.uint8)
    ratios = np.zeros(nw * nc, dtype=np.float64)
    rc = lib.difflibmojo_close_matches_batch(
        words_flat.ctypes.data_as(_i32p),
        words_off.ctypes.data_as(_i64p),
        ctypes.c_int64(nw),
        cands_flat.ctypes.data_as(_i32p),
        cands_off.ctypes.data_as(_i64p),
        ctypes.c_int64(nc),
        ctypes.c_double(cutoff),
        ctypes.c_int32(1 if autojunk else 0),
        passes.ctypes.data_as(_u8p),
        ratios.ctypes.data_as(_f64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native close_matches_batch failed with status {rc}")
    return passes.reshape(nw, nc), ratios.reshape(nw, nc)
