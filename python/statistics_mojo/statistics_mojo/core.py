"""Drop-in statistics functions matching CPython 3.12 stdlib `statistics`.

Same call shapes, same values, same types, same errors — powered by a Mojo
kernel where the platform supports it (macOS arm64, Linux x86_64), with a
pure-Python fallback (delegation to the stdlib module itself) everywhere
else. The differential test suite asserts exact agreement with the stdlib on
both backends, including exact-rational semantics for int inputs, float64
bit-exactness for float inputs, and identical StatisticsError messages.

How parity is achieved (see kernels/statistics/src/statsmojo.mojo):

* mean / variance / stdev (+ population variants): the stdlib expands every
  value to an exact rational and sums exactly. For float64 (and int64) data
  the kernel does the same summation in wide fixed-point superaccumulators
  (one bit position per possible binary exponent), so the sums are the same
  exact rationals; the final division/conversion happens in Python big-int
  arithmetic and is correctly rounded exactly like the stdlib's. stdev uses a
  correctly-rounded integer square root of the exact rational mean square
  deviation, the algorithm the stdlib documents.
* fmean: the kernel replicates CPython's math.fsum algorithm step for step
  (two-sum partials, intermediate-overflow and -inf+inf detection, half-even
  final rounding), so results are bit-identical, quirks included.
* median / quantiles: an LSD radix sort replaces the stdlib's timsort. Data
  whose ordering is identity- rather than value-defined (NaN, mixed +0.0 and
  -0.0) is routed to the stdlib path, as is every non-float64/int64 input
  type (Fraction, Decimal, mixed, arbitrary iterables).

Only homogeneous float64/int64-shaped data is accelerated; everything else
runs on the stdlib implementation on both backends (see the README for the
exact support matrix).
"""

from __future__ import annotations

import math
import statistics as _stdlib
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

from statistics_mojo import _native, _reference
from statistics_mojo._native import NativeUnavailable

__all__ = [
    "mean",
    "fmean",
    "median",
    "median_low",
    "median_high",
    "mode",
    "multimode",
    "variance",
    "stdev",
    "pvariance",
    "pstdev",
    "quantiles",
    "mean_batch",
    "fmean_batch",
    "median_batch",
    "variance_batch",
    "stdev_batch",
    "pvariance_batch",
    "pstdev_batch",
    "quantiles_batch",
]

StatisticsError = _stdlib.StatisticsError

_TWO_1074 = 1 << 1074
_TWO_2148 = 1 << 2148
_MAX_EXACT_INT = 1 << 53  # largest |int| that converts to float64 exactly

_MODE_SUM = _native.MODE_SUM
_MODE_SUM_SQ = _native.MODE_SUM_SQ
_MODE_PRODUCTS = _native.MODE_PRODUCTS

# Input kinds for the dispatch.
_FLOAT = "float"
_INT = "int"
_MIX_SMALL = "mix_small"
_OTHER = "other"

_SQRT_BIT_WIDTH = 2 * sys.float_info.mant_dig + 3  # 109 for binary64


# ---------------------------------------------------------------------------
# Exact rational finalization (single O(1) tail of the stdlib algorithms)
# ---------------------------------------------------------------------------


def _isqrt_rto(n: int, m: int) -> int:
    """Square root of n/m rounded to the nearest integer, ties to odd."""
    a = math.isqrt(n // m)
    return a | (a * a * m != n)


def _float_sqrt_of_frac(n: int, m: int) -> float:
    """Square root of n/m as a float, correctly rounded (stdlib algorithm)."""
    q = (n.bit_length() - m.bit_length() - _SQRT_BIT_WIDTH) // 2
    if q >= 0:
        numerator = _isqrt_rto(n, m << (2 * q)) << q
        denominator = 1
    else:
        numerator = _isqrt_rto(n << (-2 * q), m)
        denominator = 1 << (-q)
    return numerator / denominator


def _finalize_mean(a1: int, n: int, as_int: bool):
    """mean from the scaled exact sum; mirrors the stdlib's _convert."""
    frac = Fraction(a1, n * _TWO_1074)
    if as_int and frac.denominator == 1:
        return frac.numerator
    return float(frac)


def _finalize_spread(a1: int, a2: int, n: int, ddof: int, as_int: bool, sqrt: bool):
    """variance/stdev (and population variants) from exact scaled sums."""
    mss = Fraction(n * a2 - a1 * a1, n * _TWO_2148 * ddof)
    if sqrt:
        return _float_sqrt_of_frac(mss.numerator, mss.denominator)
    if as_int and mss.denominator == 1:
        return mss.numerator
    return float(mss)


def _finalize_products(a1: int, ddof: int, sqrt: bool):
    """variance/stdev with a caller-given centre (float64-rounded products)."""
    mss = Fraction(a1, _TWO_1074 * ddof)
    if sqrt:
        return _float_sqrt_of_frac(mss.numerator, mss.denominator)
    return float(mss)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _attempt(fn, *args):
    """Call a native primitive; None if the kernel is unavailable right now.

    Per-column ineligibility is reported by the primitive itself (None /
    per-column Nones); only kernel-unavailable raises NativeUnavailable.
    """
    try:
        return fn(*args)
    except NativeUnavailable:
        return None


def _classify_seq(seq) -> str:
    """One-pass type classification of a materialized sequence."""
    types = set(map(type, seq))
    if types <= {float}:
        return _FLOAT
    if types <= {int, bool}:
        return _INT
    if types <= {float, int, bool}:
        for v in seq:
            if type(v) is not float and abs(v) > _MAX_EXACT_INT:
                return _OTHER
        return _MIX_SMALL
    return _OTHER


def _as_i64(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Contiguous int64 view of a bool/int ndarray; False if out of range."""
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
        return arr, False
    return np.ascontiguousarray(arr, dtype=np.int64).ravel(), True


def _single_offsets(n: int) -> np.ndarray:
    return np.array([0, n], dtype=np.int64)


def _concat_f64(cols: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(cols) + 1, dtype=np.int64)
    np.cumsum([c.size for c in cols], out=offsets[1:])
    flat = np.concatenate(
        [np.ascontiguousarray(c, dtype=np.float64).ravel() for c in cols]
    )
    return flat, offsets


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def mean(data):
    """Arithmetic mean of data, exactly as CPython 3.12 `statistics.mean`."""
    if type(data) is list:
        n = len(data)
        if n == 0:
            raise StatisticsError("mean requires at least one data point")
        r = _attempt(_native.ssum_list_i64, data, _MODE_SUM)
        if r is not None:
            return _finalize_mean(r[0], n, as_int=True)
        r = _attempt(_native.ssum_list_f64, data, _MODE_SUM)
        if r is not None:
            return _finalize_mean(r[0], n, as_int=False)
        return _reference.mean(data)
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            return _reference.mean(data)
        n = data.size
        if n == 0:
            raise StatisticsError("mean requires at least one data point")
        if data.dtype == np.float64:
            a = np.ascontiguousarray(data)
            r = _attempt(_native.ssum_f64, a, _single_offsets(n), _MODE_SUM)
            if r is not None and r[0] is not None:
                return _finalize_mean(r[0][0], n, as_int=False)
            return _reference.mean(data)
        # Non-float64 ndarrays: the stdlib coerces to the numpy scalar type
        # (np.int64(Fraction) truncates!), which the native path does not
        # reproduce — route them to the stdlib implementation.
        return _reference.mean(data)
    seq = data if isinstance(data, tuple) else list(data)
    if not seq:
        raise StatisticsError("mean requires at least one data point")
    kind = _classify_seq(seq)
    if kind == _FLOAT or kind == _MIX_SMALL:
        a = np.asarray(seq, dtype=np.float64)
        r = _attempt(_native.ssum_f64, a, _single_offsets(len(seq)), _MODE_SUM)
        if r is not None and r[0] is not None:
            return _finalize_mean(r[0][0], len(seq), as_int=False)
        return _reference.mean(seq)
    if kind == _INT:
        try:
            a = np.asarray(seq, dtype=np.int64)
        except (OverflowError, ValueError):
            return _reference.mean(seq)
        r = _attempt(_native.ssum_i64, a, _single_offsets(len(seq)), _MODE_SUM)
        if r is not None:
            return _finalize_mean(r[0][0], len(seq), as_int=True)
        return _reference.mean(seq)
    return _reference.mean(seq)


# ---------------------------------------------------------------------------
# fmean
# ---------------------------------------------------------------------------


def fmean(data, weights=None):
    """Fast floating-point mean, bit-identical to CPython 3.12 `statistics.fmean`."""
    if weights is not None:
        # math.sumprod-based path; the stdlib computation is used as-is.
        return _reference.fmean(data, weights)
    if type(data) is list:
        n = len(data)
        if n == 0:
            raise StatisticsError("fmean requires at least one data point")
        r = _attempt(_native.fsum_list, data)
        if r is not None:
            return r / n
        return _reference.fmean(data)
    if isinstance(data, np.ndarray) and data.ndim == 1 and data.dtype.kind in "biuf":
        n = data.size
        if n == 0:
            raise StatisticsError("fmean requires at least one data point")
        # int/uint -> float64 conversion is correctly rounded, exactly the
        # conversion fsum applies per element.
        a = np.ascontiguousarray(data, dtype=np.float64)
        totals = _attempt(_native.fsum_f64, a, _single_offsets(n))
        if totals is None:
            return _reference.fmean(data)
        return totals[0] / n
    return _reference.fmean(data)


# ---------------------------------------------------------------------------
# median family
# ---------------------------------------------------------------------------


def _sort_inplace_native(arr: np.ndarray, offsets: np.ndarray, kind: str) -> bool:
    """In-place native sort; False when the kernel is unavailable or the data
    is identity-ordered (NaN / mixed signed zeros), i.e. reroute to stdlib."""
    try:
        if kind == _FLOAT:
            status = _native.sort_f64_inplace(arr, offsets)
            return bool((status == _native.ST_OK).all())
        _native.sort_i64_inplace(arr, offsets)
    except NativeUnavailable:
        return False
    return True


def _sorted_native(seq) -> tuple[np.ndarray | None, str | None]:
    """(sorted values, kind) via the native radix sort, or (None, kind/None).

    Eligibility: homogeneous float64 (no NaN, no mixed +0.0/-0.0 — the kernel
    detects both and reports them) or compact int64 data. Anything else
    returns None for the values and the caller reroutes to the stdlib path.
    """
    if type(seq) is list:
        r = _attempt(_native.sorted_list_i64, seq)
        if r is not None:
            return r, _INT
        r = _attempt(_native.sorted_list_f64, seq)
        if r is not None:
            return r, _FLOAT
        return None, None
    if isinstance(seq, np.ndarray):
        if seq.ndim != 1 or seq.dtype != np.float64:
            # Only float64 ndarrays are sorted natively. Integer ndarrays:
            # the stdlib's numpy-scalar arithmetic (wrapping adds/multiplies
            # in the interpolation) is identity-defined, so they route to the
            # stdlib path.
            return None, None
        a = np.ascontiguousarray(seq)
        b = a.copy()  # the kernel sorts in place; never mutate caller data
        if not _sort_inplace_native(b, _single_offsets(b.size), _FLOAT):
            return None, None
        return b, _FLOAT
    kind = _classify_seq(seq)
    if kind == _FLOAT:
        a = np.asarray(seq, dtype=np.float64)
        if not _sort_inplace_native(a, _single_offsets(a.size), _FLOAT):
            return None, None
        return a, _FLOAT
    if kind == _INT:
        try:
            a = np.asarray(seq, dtype=np.int64)
        except (OverflowError, ValueError):
            return None, _INT
        if not _sort_inplace_native(a, _single_offsets(a.size), _INT):
            return None, None
        return a, _INT
    return None, None


def median(data):
    """Median (middle value) of data, exactly as CPython 3.12 `statistics.median`."""
    seq = data if isinstance(data, (list, tuple, np.ndarray)) else list(data)
    n = len(seq)
    if n == 0:
        raise StatisticsError("no median for empty data")
    sv, kind = _sorted_native(seq)
    if sv is None:
        return _reference.median(seq)
    i = n // 2
    pick = int if kind == _INT else float
    if n % 2:
        return pick(sv[i])
    return (pick(sv[i - 1]) + pick(sv[i])) / 2


def median_low(data):
    """Low median of data, exactly as CPython 3.12 `statistics.median_low`."""
    seq = data if isinstance(data, (list, tuple, np.ndarray)) else list(data)
    n = len(seq)
    if n == 0:
        raise StatisticsError("no median for empty data")
    sv, kind = _sorted_native(seq)
    if sv is None:
        return _reference.median_low(seq)
    i = n // 2 if n % 2 else n // 2 - 1
    pick = int if kind == _INT else float
    return pick(sv[i])


def median_high(data):
    """High median of data, exactly as CPython 3.12 `statistics.median_high`."""
    seq = data if isinstance(data, (list, tuple, np.ndarray)) else list(data)
    n = len(seq)
    if n == 0:
        raise StatisticsError("no median for empty data")
    sv, kind = _sorted_native(seq)
    if sv is None:
        return _reference.median_high(seq)
    pick = int if kind == _INT else float
    return pick(sv[n // 2])


# ---------------------------------------------------------------------------
# mode / multimode (Counter semantics; identical on both backends)
# ---------------------------------------------------------------------------


def mode(data):
    """Most common data point (first encountered on ties), as `statistics.mode`."""
    pairs = Counter(iter(data)).most_common(1)
    try:
        return pairs[0][0]
    except IndexError:
        raise StatisticsError("no mode for empty data") from None


def multimode(data):
    """List of most common values in first-encountered order, as `statistics.multimode`."""
    counts = Counter(iter(data))
    if not counts:
        return []
    maxcount = max(counts.values())
    return [value for value, count in counts.items() if count == maxcount]


# ---------------------------------------------------------------------------
# variance / stdev family
# ---------------------------------------------------------------------------


def _spread(data, ref_fn, ddof_of_n, min_n, err_msg, sqrt, xbar):
    """Shared dispatch for variance/stdev/pvariance/pstdev (xbar=None only)."""
    if type(data) is list:
        n = len(data)
        if n < min_n:
            raise StatisticsError(err_msg)
        r = _attempt(_native.ssum_list_i64, data, _MODE_SUM_SQ)
        if r is not None:
            return _finalize_spread(r[0], r[1], n, ddof_of_n(n), as_int=True, sqrt=sqrt)
        r = _attempt(_native.ssum_list_f64, data, _MODE_SUM_SQ)
        if r is not None:
            return _finalize_spread(r[0], r[1], n, ddof_of_n(n), as_int=False, sqrt=sqrt)
        return ref_fn(data, xbar)
    if isinstance(data, np.ndarray):
        if data.ndim != 1 or data.dtype != np.float64:
            # Non-float64 ndarrays coerce to numpy scalar types in the stdlib
            # (truncating np.int64 conversion); route to the stdlib path.
            return ref_fn(data, xbar)
        n = data.size
        if n < min_n:
            raise StatisticsError(err_msg)
        a = np.ascontiguousarray(data)
        r = _attempt(_native.ssum_f64, a, _single_offsets(n), _MODE_SUM_SQ)
        if r is not None and r[0] is not None:
            return _finalize_spread(
                r[0][0], r[0][1], n, ddof_of_n(n), as_int=False, sqrt=sqrt
            )
        return ref_fn(data, xbar)
    seq = data if isinstance(data, tuple) else list(data)
    if len(seq) < min_n:
        raise StatisticsError(err_msg)
    kind = _classify_seq(seq)
    if kind == _FLOAT or kind == _MIX_SMALL:
        a = np.asarray(seq, dtype=np.float64)
        r = _attempt(_native.ssum_f64, a, _single_offsets(len(seq)), _MODE_SUM_SQ)
        if r is not None and r[0] is not None:
            return _finalize_spread(
                r[0][0], r[0][1], len(seq), ddof_of_n(len(seq)), as_int=False, sqrt=sqrt
            )
        return ref_fn(seq, xbar)
    if kind == _INT:
        try:
            a = np.asarray(seq, dtype=np.int64)
        except (OverflowError, ValueError):
            return ref_fn(seq, xbar)
        r = _attempt(_native.ssum_i64, a, _single_offsets(len(seq)), _MODE_SUM_SQ)
        if r is not None:
            return _finalize_spread(
                r[0][0], r[0][1], len(seq), ddof_of_n(len(seq)), as_int=True, sqrt=sqrt
            )
        return ref_fn(seq, xbar)
    return ref_fn(seq, xbar)


def _spread_c(data, ref_fn, c, ddof_of_n, min_n, err_msg, sqrt):
    """variance/stdev with a caller-given centre (the stdlib's xbar/mu path).

    The stdlib rounds each product (x - c) * (x - c) to float64 and sums the
    products exactly; the kernel does the same (mode 2). Only float64-shaped
    data with a finite float/int centre is accelerated; all-int data (whose
    products stay integral in the stdlib) and every other type route to the
    stdlib path.
    """
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        return ref_fn(data, c)
    try:
        cf = float(c)
    except OverflowError:
        return ref_fn(data, c)
    if not math.isfinite(cf):
        return ref_fn(data, c)
    if type(data) is list:
        n = len(data)
        if n < min_n:
            raise StatisticsError(err_msg)
        if isinstance(c, int):
            # An integral centre keeps the products of all-int data integral
            # in the stdlib (T=int), so that data is not float64-shaped here.
            probe = _attempt(_native.ssum_list_i64, data, _MODE_SUM)
            if probe is not None:
                return ref_fn(data, c)
        r = _attempt(_native.ssum_list_f64, data, _MODE_PRODUCTS, cf)
        if r is not None:
            return _finalize_products(r[0], ddof_of_n(n), sqrt)
        return ref_fn(data, c)
    if isinstance(data, np.ndarray):
        if data.ndim != 1 or data.dtype != np.float64:
            return ref_fn(data, c)
        n = data.size
        if n < min_n:
            raise StatisticsError(err_msg)
        a = np.ascontiguousarray(data)
        r = _attempt(_native.ssum_f64, a, _single_offsets(n), _MODE_PRODUCTS, cf)
        if r is not None and r[0] is not None:
            return _finalize_products(r[0][0], ddof_of_n(n), sqrt)
        return ref_fn(data, c)
    seq = data if isinstance(data, tuple) else list(data)
    if len(seq) < min_n:
        raise StatisticsError(err_msg)
    kind = _classify_seq(seq)
    if kind == _FLOAT or kind == _MIX_SMALL:
        a = np.asarray(seq, dtype=np.float64)
        r = _attempt(_native.ssum_f64, a, _single_offsets(len(seq)), _MODE_PRODUCTS, cf)
        if r is not None and r[0] is not None:
            return _finalize_products(r[0][0], ddof_of_n(len(seq)), sqrt)
        return ref_fn(seq, c)
    return ref_fn(seq, c)


def variance(data, xbar=None):
    """Sample variance of data, exactly as CPython 3.12 `statistics.variance`."""
    msg = "variance requires at least two data points"
    if xbar is None:
        return _spread(data, _reference.variance, lambda n: n - 1, 2, msg, sqrt=False, xbar=None)
    return _spread_c(data, _reference.variance, xbar, lambda n: n - 1, 2, msg, sqrt=False)


def stdev(data, xbar=None):
    """Sample standard deviation, exactly as CPython 3.12 `statistics.stdev`."""
    msg = "stdev requires at least two data points"
    if xbar is None:
        return _spread(data, _reference.stdev, lambda n: n - 1, 2, msg, sqrt=True, xbar=None)
    return _spread_c(data, _reference.stdev, xbar, lambda n: n - 1, 2, msg, sqrt=True)


def pvariance(data, mu=None):
    """Population variance, exactly as CPython 3.12 `statistics.pvariance`."""
    msg = "pvariance requires at least one data point"
    if mu is None:
        return _spread(data, _reference.pvariance, lambda n: n, 1, msg, sqrt=False, xbar=None)
    return _spread_c(data, _reference.pvariance, mu, lambda n: n, 1, msg, sqrt=False)


def pstdev(data, mu=None):
    """Population standard deviation, exactly as CPython 3.12 `statistics.pstdev`."""
    msg = "pstdev requires at least one data point"
    if mu is None:
        return _spread(data, _reference.pstdev, lambda n: n, 1, msg, sqrt=True, xbar=None)
    return _spread_c(data, _reference.pstdev, mu, lambda n: n, 1, msg, sqrt=True)


# ---------------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------------


def quantiles(data, *, n=4, method="exclusive"):
    """Cut points dividing data into n intervals, as `statistics.quantiles`."""
    if n < 1:
        raise StatisticsError("n must be at least 1")
    seq = data if isinstance(data, (list, tuple, np.ndarray)) else list(data)
    ld = len(seq)
    if ld < 2:
        raise StatisticsError("must have at least two data points")
    if method == "inclusive":
        m = ld - 1
    elif method == "exclusive":
        m = ld + 1
    else:
        raise ValueError(f"Unknown method: {method!r}")
    sv, kind = _sorted_native(seq)
    if sv is None:
        return _reference.quantiles(seq, n=n, method=method)
    pick = int if kind == _INT else float
    result = []
    if method == "inclusive":
        for i in range(1, n):
            j, delta = divmod(i * m, n)
            interpolated = (pick(sv[j]) * (n - delta) + pick(sv[j + 1]) * delta) / n
            result.append(interpolated)
    else:
        for i in range(1, n):
            j = i * m // n  # rescale i to m/n
            j = 1 if j < 1 else ld - 1 if j > ld - 1 else j  # clamp to 1 .. ld-1
            delta = i * m - j * n  # exact integer math
            interpolated = (pick(sv[j - 1]) * (n - delta) + pick(sv[j]) * delta) / n
            result.append(interpolated)
    return result


# ---------------------------------------------------------------------------
# Column batch API: one call, many columns, per-column stdlib semantics
# ---------------------------------------------------------------------------


def _batch_columns(columns):
    return [
        c if type(c) is list or isinstance(c, np.ndarray) else list(c)
        for c in columns
    ]


def mean_batch(columns):
    """mean() over each column; one kernel pass for all-list / all-ndarray batches."""
    cols = _batch_columns(columns)
    if all(type(c) is list for c in cols):
        out = []
        for c in cols:
            if not c:
                raise StatisticsError("mean requires at least one data point")
            r = _attempt(_native.ssum_list_i64, c, _MODE_SUM)
            if r is not None:
                out.append(_finalize_mean(r[0], len(c), as_int=True))
                continue
            r = _attempt(_native.ssum_list_f64, c, _MODE_SUM)
            if r is not None:
                out.append(_finalize_mean(r[0], len(c), as_int=False))
                continue
            out.append(_reference.mean(c))
        return out
    if all(isinstance(c, np.ndarray) and c.dtype == np.float64 and c.ndim == 1 for c in cols):
        for c in cols:
            if c.size == 0:
                raise StatisticsError("mean requires at least one data point")
        flat, offsets = _concat_f64(cols)
        r = _attempt(_native.ssum_f64, flat, offsets, _MODE_SUM)
        if r is None:
            return [_reference.mean(c) for c in cols]
        return [
            _finalize_mean(a[0], c.size, as_int=False) if a is not None else _reference.mean(c)
            for a, c in zip(r, cols)
        ]
    return [mean(c) for c in cols]


def fmean_batch(columns):
    """fmean() over each column (unweighted; weighted data is out of batch scope)."""
    cols = _batch_columns(columns)
    if all(type(c) is list for c in cols):
        out = []
        for c in cols:
            if not c:
                raise StatisticsError("fmean requires at least one data point")
            r = _attempt(_native.fsum_list, c)
            out.append(r / len(c) if r is not None else _reference.fmean(c))
        return out
    if all(isinstance(c, np.ndarray) and c.dtype == np.float64 and c.ndim == 1 for c in cols):
        for c in cols:
            if c.size == 0:
                raise StatisticsError("fmean requires at least one data point")
        flat, offsets = _concat_f64(cols)
        totals = _attempt(_native.fsum_f64, flat, offsets)
        if totals is None:
            return [_reference.fmean(c) for c in cols]
        return [t / c.size for t, c in zip(totals, cols)]
    return [fmean(c) for c in cols]


def _spread_batch(columns, pop, sqrt):
    cols = _batch_columns(columns)
    if pop:
        min_n, msg = 1, (
            "pstdev requires at least one data point"
            if sqrt
            else "pvariance requires at least one data point"
        )
    else:
        min_n, msg = 2, (
            "stdev requires at least two data points"
            if sqrt
            else "variance requires at least two data points"
        )
    ddof_of_n = (lambda n: n) if pop else (lambda n: n - 1)
    ref = (
        _reference.pstdev
        if sqrt and pop
        else _reference.stdev
        if sqrt
        else _reference.pvariance
        if pop
        else _reference.variance
    )
    if all(type(c) is list for c in cols):
        out = []
        for c in cols:
            n = len(c)
            if n < min_n:
                raise StatisticsError(msg)
            r = _attempt(_native.ssum_list_i64, c, _MODE_SUM_SQ)
            if r is not None:
                out.append(
                    _finalize_spread(r[0], r[1], n, ddof_of_n(n), as_int=True, sqrt=sqrt)
                )
                continue
            r = _attempt(_native.ssum_list_f64, c, _MODE_SUM_SQ)
            if r is not None:
                out.append(
                    _finalize_spread(r[0], r[1], n, ddof_of_n(n), as_int=False, sqrt=sqrt)
                )
                continue
            out.append(ref(c, None))
        return out
    if all(isinstance(c, np.ndarray) and c.dtype == np.float64 and c.ndim == 1 for c in cols):
        for c in cols:
            if c.size < min_n:
                raise StatisticsError(msg)
        flat, offsets = _concat_f64(cols)
        r = _attempt(_native.ssum_f64, flat, offsets, _MODE_SUM_SQ)
        if r is None:
            return [ref(c, None) for c in cols]
        out = []
        for a, c in zip(r, cols):
            if a is None:
                out.append(ref(c, None))
            else:
                out.append(
                    _finalize_spread(
                        a[0], a[1], c.size, ddof_of_n(c.size), as_int=False, sqrt=sqrt
                    )
                )
        return out
    fn = pstdev if sqrt and pop else stdev if sqrt else pvariance if pop else variance
    return [fn(c) for c in cols]


def variance_batch(columns):
    """variance() over each column; one kernel pass for uniform batches."""
    return _spread_batch(columns, pop=False, sqrt=False)


def stdev_batch(columns):
    """stdev() over each column; one kernel pass for uniform batches."""
    return _spread_batch(columns, pop=False, sqrt=True)


def pvariance_batch(columns):
    """pvariance() over each column; one kernel pass for uniform batches."""
    return _spread_batch(columns, pop=True, sqrt=False)


def pstdev_batch(columns):
    """pstdev() over each column; one kernel pass for uniform batches."""
    return _spread_batch(columns, pop=True, sqrt=True)


def median_batch(columns):
    """median() over each column, with per-column stdlib semantics."""
    cols = _batch_columns(columns)
    out = []
    for seq in cols:
        if isinstance(seq, np.ndarray):
            out.append(median(seq))
            continue
        n = len(seq)
        if n == 0:
            raise StatisticsError("no median for empty data")
        sv, kind = _sorted_native(seq)
        if sv is None:
            out.append(_reference.median(seq))
            continue
        i = n // 2
        pick = int if kind == _INT else float
        out.append(pick(sv[i]) if n % 2 else (pick(sv[i - 1]) + pick(sv[i])) / 2)
    return out


def quantiles_batch(columns, *, n=4, method="exclusive"):
    """quantiles() over each column, with per-column stdlib semantics."""
    cols = _batch_columns(columns)
    # The stdlib checks n before anything else; method is validated per
    # column after that column's length check (same order as scalar calls).
    if n < 1:
        raise StatisticsError("n must be at least 1")
    return [quantiles(c, n=n, method=method) for c in cols]
