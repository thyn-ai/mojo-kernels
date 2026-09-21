"""ctypes loader for the statsmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$STATISTICS_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``statistics_mojo/_native/``
    3. the repository development build output ``kernels/statistics/build/``

If the library cannot be found, fails to load, reports an ABI version this
package does not understand, or fails the load-time self-test,
:class:`NativeUnavailable` is raised and the caller falls back to the stdlib
reference implementation. Set ``STATISTICS_MOJO_DISABLE_NATIVE=1`` to force
that fallback (used by the differential test suite).

Stable C ABI (v1)::

    int32_t statsmojo_abi_version(void)
    int32_t statsmojo_ssum_f64(const double* x, const int64_t* offsets,
                               int64_t n_cols, int32_t mode, double c,
                               uint64_t* out, int32_t* status)
    int32_t statsmojo_ssum_i64(const int64_t* x, const int64_t* offsets,
                               int64_t n_cols, int32_t mode, uint64_t* out)
    int32_t statsmojo_fsum_f64(const double* x, const int64_t* offsets,
                               int64_t n_cols, double* out, int32_t* status)
    int32_t statsmojo_sort_f64(double* x, const int64_t* offsets, int64_t n_cols)
    int32_t statsmojo_sort_i64(int64_t* x, const int64_t* offsets, int64_t n_cols)
    int32_t statsmojo_ssum_list_f64(uint64_t list_addr, uint64_t float_type,
                                    uint64_t int_type, int32_t mode, double c,
                                    uint64_t* out, int32_t* status)
    int32_t statsmojo_ssum_list_i64(uint64_t list_addr, uint64_t int_type,
                                    int32_t mode, uint64_t* out, int32_t* status)
    int32_t statsmojo_fsum_list_f64(uint64_t list_addr, uint64_t float_type,
                                    uint64_t int_type, double* out, int32_t* status)
    int32_t statsmojo_sort_list_f64(uint64_t list_addr, int64_t cap,
                                    uint64_t float_type, double* out, int32_t* status)
    int32_t statsmojo_sort_list_i64(uint64_t list_addr, int64_t cap,
                                    uint64_t int_type, int64_t* out, int32_t* status)

The ``*_list_*`` entry points read CPython object internals (PyListObject
ob_item, PyFloatObject.ob_fval, compact PyLongObject digits) and MUST be
called with the GIL held — they are bound through a separate
:class:`ctypes.PyDLL` handle. Every element's ob_type is checked against the
exact type object; any mismatch aborts the column with ST_TYPE and the
wrapper reroutes the column to a safe path. The layout assumptions are
verified by a load-time self-test; if it fails the whole native backend is
treated as unavailable.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from fractions import Fraction

import numpy as np

# Must equal ABI_VERSION in kernels/statistics/src/statsmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

# Superaccumulator layout (mirrors the kernel; also reported by
# statsmojo_words_per_col at load time and cross-checked).
WORDS_X = 36
WORDS_SQ = 68
LANES = 4
LANE_WORDS = 2 * WORDS_X + WORDS_SQ  # 140
WORDS_PER_COL = LANES * LANE_WORDS  # 560
OFFSET_X = 1074
OFFSET_SQ = 2148

# Per-column status codes (mirror the kernel).
ST_OK = 0
ST_OVERFLOW = 1
ST_INF_MIX = 2
ST_TYPE = 3
ST_NONFINITE = 4
ST_SIGNED_ZERO = 5

# ssum modes (mirror the kernel).
MODE_SUM = 0
MODE_SUM_SQ = 1
MODE_PRODUCTS = 2

_ENV_LIB = "STATISTICS_MOJO_NATIVE_LIB"
_ENV_DISABLE = "STATISTICS_MOJO_DISABLE_NATIVE"

_TWO_1074 = 1 << OFFSET_X
_TWO_2148 = 1 << OFFSET_SQ

# CPython 3.12 object layout offsets (verified by _self_test at load).
_LIST_OB_SIZE = 16
_LIST_OB_ITEM = 24


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native statsmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libstatsmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libstatsmojo.so"
    if sys.platform.startswith("win"):
        return "statsmojo.dll"  # no Mojo toolchain builds this today
    return "libstatsmojo.so"


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
                    here, "..", "..", "..", "kernels", "statistics", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL, pydll: ctypes.PyDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.statsmojo_abi_version.argtypes = []
    lib.statsmojo_abi_version.restype = ctypes.c_int32
    lib.statsmojo_words_per_col.argtypes = []
    lib.statsmojo_words_per_col.restype = ctypes.c_int32
    lib.statsmojo_ssum_f64.argtypes = [
        f64p, i64p, ctypes.c_int64, ctypes.c_int32, ctypes.c_double, u64p, i32p
    ]
    lib.statsmojo_ssum_f64.restype = ctypes.c_int32
    lib.statsmojo_ssum_i64.argtypes = [i64p, i64p, ctypes.c_int64, ctypes.c_int32, u64p]
    lib.statsmojo_ssum_i64.restype = ctypes.c_int32
    lib.statsmojo_fsum_f64.argtypes = [f64p, i64p, ctypes.c_int64, f64p, i32p]
    lib.statsmojo_fsum_f64.restype = ctypes.c_int32
    lib.statsmojo_sort_f64.argtypes = [f64p, i64p, ctypes.c_int64, i32p]
    lib.statsmojo_sort_f64.restype = ctypes.c_int32
    lib.statsmojo_sort_i64.argtypes = [i64p, i64p, ctypes.c_int64]
    lib.statsmojo_sort_i64.restype = ctypes.c_int32
    # List-unboxing entry points: bound on the PyDLL handle so the GIL stays
    # held for the whole call (the kernel reads live CPython objects).
    pydll.statsmojo_ssum_list_f64.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int32, ctypes.c_double, u64p, i32p,
    ]
    pydll.statsmojo_ssum_list_f64.restype = ctypes.c_int32
    pydll.statsmojo_ssum_list_i64.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, u64p, i32p
    ]
    pydll.statsmojo_ssum_list_i64.restype = ctypes.c_int32
    pydll.statsmojo_fsum_list_f64.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, f64p, i32p
    ]
    pydll.statsmojo_fsum_list_f64.restype = ctypes.c_int32
    pydll.statsmojo_sort_list_f64.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint64, f64p, i32p
    ]
    pydll.statsmojo_sort_list_f64.restype = ctypes.c_int32
    pydll.statsmojo_sort_list_i64.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint64, i64p, i32p
    ]
    pydll.statsmojo_sort_list_i64.restype = ctypes.c_int32


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_PYLIB: ctypes.PyDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _assemble(words: np.ndarray, squares: bool) -> tuple[int, int]:
    """Python big ints (scaled_sum, scaled_sum_sq) from one column's words.

    The exact sum is scaled_sum * 2**-OFFSET_X and the exact sum of squares
    is scaled_sum_sq * 2**-OFFSET_SQ.
    """
    b = words.tobytes()
    lane_bytes = LANE_WORDS * 8
    pos = neg = sq = 0
    for lane in range(LANES):
        chunk = b[lane * lane_bytes: (lane + 1) * lane_bytes]
        pos += int.from_bytes(chunk[: WORDS_X * 8], "little")
        neg += int.from_bytes(chunk[WORDS_X * 8: 2 * WORDS_X * 8], "little")
        if squares:
            sq += int.from_bytes(chunk[2 * WORDS_X * 8:], "little")
    return pos - neg, sq


def _self_test(lib: ctypes.CDLL, pydll: ctypes.PyDLL) -> None:
    """Verify ABI layout and every CPython-layout assumption, end to end.

    Runs one small known-answer computation per entry point (flat and
    list-unboxing) and compares against exact Python arithmetic. Any
    disagreement means the kernel cannot be trusted on this interpreter;
    raise NativeUnavailable so every caller falls back.
    """
    f64p = ctypes.POINTER(ctypes.c_double)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)

    def fail(what: str) -> None:
        raise NativeUnavailable(f"native self-test failed: {what}")

    out = np.zeros(WORDS_PER_COL, dtype=np.uint64)
    status = np.zeros(1, dtype=np.int32)

    # Flat f64 exact sums.
    arr = np.array([1.5, -2.25, 3.25], dtype=np.float64)
    offs = np.array([0, 3], dtype=np.int64)
    rc = lib.statsmojo_ssum_f64(
        arr.ctypes.data_as(f64p), offs.ctypes.data_as(i64p), 1,
        MODE_SUM_SQ, 0.0, out.ctypes.data_as(u64p), status.ctypes.data_as(i32p),
    )
    a1, a2 = _assemble(out, True)
    if rc != 0 or status[0] != ST_OK:
        fail("statsmojo_ssum_f64 rc/status")
    if Fraction(a1, _TWO_1074) != Fraction(5, 2):
        fail("flat f64 sum")
    if Fraction(a2, _TWO_2148) != Fraction(9, 4) + Fraction(81, 16) + Fraction(169, 16):
        fail("flat f64 sum of squares")

    # Flat i64 exact sums.
    iarr = np.array([3, -7, 2**40], dtype=np.int64)
    rc = lib.statsmojo_ssum_i64(
        iarr.ctypes.data_as(i64p), offs.ctypes.data_as(i64p), 1,
        MODE_SUM_SQ, out.ctypes.data_as(u64p),
    )
    a1, a2 = _assemble(out, True)
    if rc != 0 or a1 != (3 - 7 + 2**40) * _TWO_1074:
        fail("flat i64 sum")
    if a2 != (9 + 49 + 2**80) * _TWO_2148:
        fail("flat i64 sum of squares")

    # Flat fsum (CPython semantics, including the overflow quirk).
    fout = np.zeros(1, dtype=np.float64)
    rc = lib.statsmojo_fsum_f64(
        arr.ctypes.data_as(f64p), offs.ctypes.data_as(i64p), 1,
        fout.ctypes.data_as(f64p), status.ctypes.data_as(i32p),
    )
    if rc != 0 or status[0] != ST_OK or fout[0] != math.fsum(arr.tolist()):
        fail("flat fsum")
    ov = np.array([1e308, 1e308, -1e308], dtype=np.float64)
    rc = lib.statsmojo_fsum_f64(
        ov.ctypes.data_as(f64p), offs.ctypes.data_as(i64p), 1,
        fout.ctypes.data_as(f64p), status.ctypes.data_as(i32p),
    )
    if status[0] != ST_OVERFLOW:
        fail("fsum intermediate-overflow detection")

    # Flat sorts.
    s = np.array([3.0, -1.0, 2.0], dtype=np.float64)
    lib.statsmojo_sort_f64(
        s.ctypes.data_as(f64p), offs.ctypes.data_as(i64p), 1, status.ctypes.data_as(i32p)
    )
    if s.tolist() != [-1.0, 2.0, 3.0] or status[0] != ST_OK:
        fail("flat f64 sort")
    si = np.array([5, -3, 2**40], dtype=np.int64)
    lib.statsmojo_sort_i64(si.ctypes.data_as(i64p), offs.ctypes.data_as(i64p), 1)
    if si.tolist() != [-3, 5, 2**40]:
        fail("flat i64 sort")

    # List-unboxing paths (guarded: any layout drift must fail here, never
    # produce wrong numbers).
    xs = [1.5, -2.25, 3.25]
    rc = pydll.statsmojo_ssum_list_f64(
        id(xs), id(float), id(int), MODE_SUM_SQ, 0.0,
        out.ctypes.data_as(u64p), status.ctypes.data_as(i32p),
    )
    a1, a2 = _assemble(out, True)
    if rc != 0 or status[0] != ST_OK:
        fail("list f64 rc/status")
    if Fraction(a1, _TWO_1074) != Fraction(5, 2):
        fail("list f64 sum (PyFloat layout)")
    if Fraction(a2, _TWO_2148) != Fraction(9, 4) + Fraction(81, 16) + Fraction(169, 16):
        fail("list f64 sum of squares (PyFloat layout)")

    xi = [0, 1, -1, 2**30, -(2**30), 2**45, -(2**45), 2**53 + 7]
    rc = pydll.statsmojo_ssum_list_i64(
        id(xi), id(int), MODE_SUM_SQ,
        out.ctypes.data_as(u64p), status.ctypes.data_as(i32p),
    )
    a1, a2 = _assemble(out, True)
    if rc != 0 or status[0] != ST_OK:
        fail("list i64 rc/status")
    if a1 != sum(xi) * _TWO_1074 or a2 != sum(v * v for v in xi) * _TWO_2148:
        fail("list i64 sums (PyLong layout)")

    rc = pydll.statsmojo_fsum_list_f64(
        id(xs), id(float), id(int),
        fout.ctypes.data_as(f64p), status.ctypes.data_as(i32p),
    )
    if rc != 0 or status[0] != ST_OK or fout[0] != math.fsum(xs):
        fail("list fsum")

    sout = np.zeros(3, dtype=np.float64)
    xs2 = [3.0, -1.0, 2.0]
    rc = pydll.statsmojo_sort_list_f64(
        id(xs2), 3, id(float), sout.ctypes.data_as(f64p), status.ctypes.data_as(i32p),
    )
    if rc != 0 or status[0] != ST_OK or sout.tolist() != [-1.0, 2.0, 3.0]:
        fail("list f64 sort")

    siout = np.zeros(4, dtype=np.int64)
    xi2 = [5, -3, 2**40, 1]
    rc = pydll.statsmojo_sort_list_i64(
        id(xi2), 4, id(int), siout.ctypes.data_as(i64p), status.ctypes.data_as(i32p),
    )
    if rc != 0 or status[0] != ST_OK or siout.tolist() != [-3, 1, 5, 2**40]:
        fail("list i64 sort")


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, ABI-handshake, and self-test the kernel. Never caches failure."""
    global _LIB, _PYLIB, _LIB_SOURCE, _LOAD_ERROR
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
                    pydll = ctypes.PyDLL(path)
                    _bind_abi(lib, pydll)
                    abi = int(lib.statsmojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                if int(lib.statsmojo_words_per_col()) != WORDS_PER_COL:
                    errors.append(f"{label} ({path}): accumulator layout mismatch")
                    continue
                try:
                    _self_test(lib, pydll)
                except NativeUnavailable as exc:
                    errors.append(f"{label} ({path}): {exc}")
                    continue
                _LIB, _PYLIB, _LIB_SOURCE = lib, pydll, f"{label} ({path})"
                _LOAD_ERROR = None
                return lib
            except OSError as exc:
                errors.append(f"{label}: {exc}")
        _LOAD_ERROR = "; ".join(errors) or "no native kernel found on any resolver path"
        raise NativeUnavailable(_LOAD_ERROR)


def native_available() -> bool:
    """True if the native kernel can compute right now. Never raises."""
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
    info["abi_version_native"] = int(lib.statsmojo_abi_version())
    return info


def _f64p(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _i64p(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int64):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _u64p(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint64):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _i32p(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ---------------------------------------------------------------------------
# Low-level native primitives. Each returns None (or per-column Nones) when
# the data is not eligible for the fast path (status != ST_OK); the caller
# then recomputes those columns on a safe path. NativeUnavailable propagates.
# ---------------------------------------------------------------------------


def ssum_f64(
    flat: np.ndarray, offsets: np.ndarray, mode: int, c: float = 0.0
) -> list[tuple[int, int] | None]:
    """Exact sums per column: (scaled_sum, scaled_sum_sq) or None per column."""
    lib = _load()
    n_cols = offsets.shape[0] - 1
    out = np.zeros((n_cols, WORDS_PER_COL), dtype=np.uint64)
    status = np.zeros(n_cols, dtype=np.int32)
    rc = lib.statsmojo_ssum_f64(
        _f64p(flat), _i64p(offsets), n_cols, mode, ctypes.c_double(c),
        _u64p(out), _i32p(status),
    )
    if rc != 0:
        raise NativeUnavailable(f"native ssum_f64 failed with status {rc}")
    squares = mode == MODE_SUM_SQ
    return [
        _assemble(out[col], squares) if status[col] == ST_OK else None
        for col in range(n_cols)
    ]


def ssum_i64(flat: np.ndarray, offsets: np.ndarray, mode: int) -> list[tuple[int, int]]:
    """Exact sums per int64 column (always succeeds for in-range data)."""
    lib = _load()
    n_cols = offsets.shape[0] - 1
    out = np.zeros((n_cols, WORDS_PER_COL), dtype=np.uint64)
    rc = lib.statsmojo_ssum_i64(_i64p(flat), _i64p(offsets), n_cols, mode, _u64p(out))
    if rc != 0:
        raise NativeUnavailable(f"native ssum_i64 failed with status {rc}")
    squares = mode == MODE_SUM_SQ
    return [_assemble(out[col], squares) for col in range(n_cols)]


def ssum_list_f64(data: list, mode: int, c: float = 0.0) -> tuple[int, int] | None:
    """Exact sums for one list column (floats + small ints); None if ineligible."""
    pydll = _ensure_pydll()
    out = np.zeros(WORDS_PER_COL, dtype=np.uint64)
    status = np.zeros(1, dtype=np.int32)
    rc = pydll.statsmojo_ssum_list_f64(
        id(data), id(float), id(int), mode, ctypes.c_double(c),
        _u64p(out), _i32p(status),
    )
    if rc != 0:
        raise NativeUnavailable(f"native ssum_list_f64 failed with status {rc}")
    if status[0] != ST_OK:
        return None
    return _assemble(out, mode == MODE_SUM_SQ)


def ssum_list_i64(data: list, mode: int) -> tuple[int, int] | None:
    """Exact sums for one list column of compact ints; None if ineligible."""
    pydll = _ensure_pydll()
    out = np.zeros(WORDS_PER_COL, dtype=np.uint64)
    status = np.zeros(1, dtype=np.int32)
    rc = pydll.statsmojo_ssum_list_i64(
        id(data), id(int), mode, _u64p(out), _i32p(status)
    )
    if rc != 0:
        raise NativeUnavailable(f"native ssum_list_i64 failed with status {rc}")
    if status[0] != ST_OK:
        return None
    return _assemble(out, mode == MODE_SUM_SQ)


def _ensure_pydll() -> ctypes.PyDLL:
    _load()
    assert _PYLIB is not None
    return _PYLIB


def _raise_fsum_error(status: int) -> None:
    if status == ST_OVERFLOW:
        raise OverflowError("intermediate overflow in fsum")
    if status == ST_INF_MIX:
        raise ValueError("-inf + inf in fsum")
    raise NativeUnavailable(f"native fsum failed with status {status}")


def fsum_f64(flat: np.ndarray, offsets: np.ndarray) -> list[float]:
    """CPython-semantics fsum per column; raises the fsum errors in column order."""
    lib = _load()
    n_cols = offsets.shape[0] - 1
    out = np.zeros(n_cols, dtype=np.float64)
    status = np.zeros(n_cols, dtype=np.int32)
    rc = lib.statsmojo_fsum_f64(_f64p(flat), _i64p(offsets), n_cols, _f64p(out), _i32p(status))
    if rc != 0:
        raise NativeUnavailable(f"native fsum_f64 failed with status {rc}")
    for col in range(n_cols):
        if status[col] != ST_OK:
            _raise_fsum_error(int(status[col]))
    return out.tolist()


def fsum_list(data: list) -> float | None:
    """CPython-semantics fsum of one list column; None if ineligible."""
    pydll = _ensure_pydll()
    out = np.zeros(1, dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)
    rc = pydll.statsmojo_fsum_list_f64(
        id(data), id(float), id(int), _f64p(out), _i32p(status)
    )
    if rc != 0:
        raise NativeUnavailable(f"native fsum_list_f64 failed with status {rc}")
    if status[0] == ST_TYPE:
        return None
    if status[0] != ST_OK:
        _raise_fsum_error(int(status[0]))
    return float(out[0])


def sort_f64_inplace(flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Ascending sort of each column segment, in place.

    Returns the per-column status array: ST_OK, or ST_NONFINITE /
    ST_SIGNED_ZERO for columns the kernel left unsorted (reroute those to
    the reference path).
    """
    lib = _load()
    n_cols = offsets.shape[0] - 1
    status = np.zeros(n_cols, dtype=np.int32)
    rc = lib.statsmojo_sort_f64(_f64p(flat), _i64p(offsets), n_cols, _i32p(status))
    if rc != 0:
        raise NativeUnavailable(f"native sort_f64 failed with status {rc}")
    return status


def sort_i64_inplace(flat: np.ndarray, offsets: np.ndarray) -> None:
    """Ascending sort of each int64 column segment, in place."""
    lib = _load()
    rc = lib.statsmojo_sort_i64(_i64p(flat), _i64p(offsets), offsets.shape[0] - 1)
    if rc != 0:
        raise NativeUnavailable(f"native sort_i64 failed with status {rc}")


def sorted_list_f64(data: list) -> np.ndarray | None:
    """Sorted float64 copy of an all-float list; None if ineligible."""
    pydll = _ensure_pydll()
    out = np.zeros(len(data), dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)
    rc = pydll.statsmojo_sort_list_f64(
        id(data), len(data), id(float), _f64p(out), _i32p(status)
    )
    if rc != 0:
        raise NativeUnavailable(f"native sort_list_f64 failed with status {rc}")
    if status[0] != ST_OK:
        return None
    return out


def sorted_list_i64(data: list) -> np.ndarray | None:
    """Sorted int64 copy of an all-compact-int list; None if ineligible."""
    pydll = _ensure_pydll()
    out = np.zeros(len(data), dtype=np.int64)
    status = np.zeros(1, dtype=np.int32)
    rc = pydll.statsmojo_sort_list_i64(
        id(data), len(data), id(int), _i64p(out), _i32p(status)
    )
    if rc != 0:
        raise NativeUnavailable(f"native sort_list_i64 failed with status {rc}")
    if status[0] != ST_OK:
        return None
    return out
