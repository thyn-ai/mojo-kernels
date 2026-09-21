"""Differential tests: statistics_mojo must match the stdlib `statistics` module
value-for-value, type-for-type, and error-for-error.

Run twice by `scripts/test_all_statistics.sh`: once against the native Mojo
kernel and once with STATISTICS_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback, which delegates to the stdlib module). The oracle is the Python
3.12 standard library `statistics` module itself.

Gate: exact equality — same value, same type (int vs float vs Fraction vs
Decimal), same exception type and message. For float results this is
bit-for-bit (repr equality); the exact-rational semantics of the stdlib are
replicated, not approximated. The single documented exception: ndarray
inputs yield builtin float/int results where the stdlib yields numpy scalar
types, so ndarray cases gate on value and repr-of-value only.
"""

from __future__ import annotations

import math
import os
import statistics
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import statistics_mojo as sm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_same(name, got, want):
    """Exact agreement: type, value, and repr (catches -0.0 and int/float)."""
    assert type(got) is type(want), (
        f"{name}: type {type(got).__name__} != {type(want).__name__}: {got!r} vs {want!r}"
    )
    if isinstance(got, list):
        assert len(got) == len(want), name
        for i, (g, w) in enumerate(zip(got, want)):
            check_same(f"{name}[{i}]", g, w)
        return
    if isinstance(got, float) and math.isnan(got):
        assert math.isnan(want), f"{name}: {got!r} vs {want!r}"
        return
    assert got == want, f"{name}: {got!r} != {want!r}"
    assert repr(got) == repr(want), f"{name}: repr {got!r} != {want!r}"


def check_raises(name, fn, *args, **kwargs):
    """Same exception type and message from statistics_mojo and statistics."""
    fn_m = getattr(sm, fn)
    fn_s = getattr(statistics, fn)
    try:
        got = fn_m(*args, **kwargs)
    except Exception as e:
        got_exc = (type(e).__name__, str(e))
    else:
        got_exc = ("NO ERROR", repr(got))
    try:
        want = fn_s(*args, **kwargs)
    except Exception as e:
        want_exc = (type(e).__name__, str(e))
    else:
        want_exc = ("NO ERROR", repr(want))
    assert got_exc == want_exc, f"{name}: {got_exc} != {want_exc}"


def check_call(name, fn_name, data, *args, **kwargs):
    """Same result (exact) or same exception from statistics_mojo.fn and statistics.fn."""
    try:
        got = getattr(sm, fn_name)(data, *args, **kwargs)
    except Exception as e:
        got_exc = (type(e).__name__, str(e))
    else:
        got_exc = None
    try:
        want = getattr(statistics, fn_name)(data, *args, **kwargs)
    except Exception as e:
        want_exc = (type(e).__name__, str(e))
    else:
        want_exc = None
    if got_exc is not None or want_exc is not None:
        assert got_exc == want_exc, f"{name}: {got_exc} != {want_exc}"
        return
    check_same(name, got, want)


# ---------------------------------------------------------------------------
# Seeded data generators (deterministic on any machine)
# ---------------------------------------------------------------------------


def make_floats(seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n).tolist()


def make_ints(seed: int, n: int, lo=-1000, hi=1000) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi + 1, n).tolist()


FLOAT_CASES = [
    pytest.param(make_floats(1, 100), id="normal100"),
    pytest.param(make_floats(2, 1000), id="normal1000"),
    pytest.param(make_floats(3, 2), id="two"),
    pytest.param(make_floats(4, 3), id="three"),
    pytest.param([0.1] * 10, id="point-one-ten"),
    pytest.param([1e308, -1e308, 1e308], id="overflow-cancel"),
    pytest.param([1e-300, 1e300, -1e-300, -1e300] * 25, id="wide-range"),
    pytest.param([5e-324, -5e-324, 1.0], id="subnormal"),
    pytest.param([1.5, -2.25, 3.25, 0.0, -0.0], id="neg-zero-only-mix"),
    pytest.param([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5], id="docstring"),
    pytest.param([7.0], id="single"),
    pytest.param([1e16, 1.0, 1e-16] * 40, id="halfway"),
]

INT_CASES = [
    pytest.param(make_ints(5, 100), id="ints100"),
    pytest.param(make_ints(6, 1000), id="ints1000"),
    pytest.param([1, 2, 3], id="ints-whole-mean"),
    pytest.param([1, 2], id="ints-half-mean"),
    pytest.param([1, 2, 4, 8], id="ints-pow2"),
    pytest.param([2**53 - 1, -(2**53) + 1, 7], id="ints-53bit"),
    pytest.param([2**60 + 3, -(2**60) - 3, 11, 2**60 - 9], id="ints-60bit"),
    pytest.param([2**70, 2**71, 2**69], id="ints-70bit"),
    pytest.param([0] * 5, id="ints-zeros"),
    pytest.param([True, False, True], id="bools"),
    pytest.param(list(range(100)), id="ints-range100"),
    pytest.param([5], id="ints-single"),
]

MIXED_CASES = [
    pytest.param([1, 2.5, 3], id="mixed-small"),
    pytest.param([2**53, 1.5, -2**53], id="mixed-53bit"),
    pytest.param([2**60, 1.5], id="mixed-big"),
    pytest.param([Fraction(1, 3), Fraction(2, 3)], id="fractions"),
    pytest.param([Fraction(1, 6), Fraction(1, 2), Fraction(5, 3)], id="fractions-var"),
    pytest.param([Decimal("0.5"), Decimal("0.75"), Decimal("0.625"), Decimal("0.375")], id="decimals"),
    pytest.param(
        [Decimal("27.5"), Decimal("30.25"), Decimal("30.25"), Decimal("34.5"), Decimal("41.75")],
        id="decimals-var",
    ),
    pytest.param([Decimal("0.1"), Decimal("0.2"), Decimal("0.3")], id="decimals-tenth"),
    pytest.param([1.0, math.inf], id="float-inf"),
    pytest.param([math.nan, 1.0], id="float-nan"),
    pytest.param([1.0, math.inf, 3.0], id="float-inf-3"),
    pytest.param([math.inf, -math.inf, 1.0], id="float-inf-mix"),
    pytest.param([Fraction(1, 2), 0.5, 1], id="fraction-float-mix"),
    pytest.param([1, Fraction(1, 3)], id="int-fraction-mix"),
]

SPREAD_FNS = ["variance", "stdev", "pvariance", "pstdev"]
CENTRAL_FNS = ["mean", "fmean", "median", "median_low", "median_high"]


# ---------------------------------------------------------------------------
# Central tendency + spread, all data shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data", FLOAT_CASES + INT_CASES + MIXED_CASES)
@pytest.mark.parametrize("fn", ["mean", "fmean"])
def test_mean_family(data, fn):
    check_call(f"{fn}({type(data).__name__})", fn, data)


@pytest.mark.parametrize("data", FLOAT_CASES + INT_CASES + MIXED_CASES)
@pytest.mark.parametrize("fn", SPREAD_FNS)
def test_spread_family(data, fn):
    check_call(f"{fn}", fn, data)


@pytest.mark.parametrize("data", FLOAT_CASES + INT_CASES + MIXED_CASES)
@pytest.mark.parametrize("fn", ["median", "median_low", "median_high"])
def test_median_family(data, fn):
    check_call(fn, fn, data)


@pytest.mark.parametrize("data", FLOAT_CASES + INT_CASES + MIXED_CASES)
@pytest.mark.parametrize("fn", ["mode", "multimode"])
def test_mode_family(data, fn):
    check_call(fn, fn, data)


# ---------------------------------------------------------------------------
# Variance with a caller-given centre (xbar / mu)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,c",
    [
        ([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5], 2.0),
        ([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5], 2),
        ([1.5, 2.5, 3.5], 0.0),
        ([1, 2, 3], 2),            # all-int data, int centre: integral products
        ([1, 2, 3], 2.0),          # all-int data, float centre
        ([1, 2, 3], 2.5),
        ([1, 2.5, 3], 1),          # mixed data, int centre
        ([1, 2.5, 3], 1.25),
        (make_floats(7, 500), 0.125),
        (make_floats(8, 500), -3.75),
        ([1e16, 1.0, 1e-16], 1e16),
        ([1e308, 1e308], 1e308),   # zero deviation, huge centre
        ([1e308, -1e308], 0.0),    # products overflow to inf
        ([1.0, 2.0], math.inf),
        ([1.0, 2.0], math.nan),
        ([1.0, 2.0], Fraction(3, 2)),
        ([1.0, 2.0], Decimal("1.5")),
        ([Fraction(1, 2), Fraction(1, 3)], Fraction(5, 12)),
        ([Decimal("1.5"), Decimal("2.5")], Decimal("2.0")),
        ([1, 2, 3], True),
        ([1.5, 2.5], True),
        ([2**70, 2**71], 2**70),
    ],
)
@pytest.mark.parametrize("fn", SPREAD_FNS)
def test_spread_with_centre(data, c, fn):
    check_call(f"{fn}(c={c!r})", fn, data, c)


def test_spread_centre_matches_mean():
    # stdlib documents that passing the computed mean as xbar/mu is allowed;
    # results must match the oracle for that call shape too.
    data = make_floats(9, 300)
    mean = statistics.mean(data)
    for fn in SPREAD_FNS:
        check_call(f"{fn}(mean)", fn, data, mean)


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data", FLOAT_CASES + INT_CASES + MIXED_CASES[:4])
@pytest.mark.parametrize("n", [1, 2, 4, 10, 100])
@pytest.mark.parametrize("method", ["exclusive", "inclusive"])
def test_quantiles(data, n, method):
    if len(data) < 2:
        check_raises("quantiles-short", "quantiles", data, n=n, method=method)
        return
    check_call(f"quantiles(n={n},{method})", "quantiles", data, n=n, method=method)


def test_quantiles_errors():
    check_raises("quantiles n=0", "quantiles", [1.0, 2.0], n=0)
    check_raises("quantiles n=-3", "quantiles", [1.0, 2.0], n=-3)
    check_raises("quantiles empty", "quantiles", [], n=4)
    check_raises("quantiles single", "quantiles", [1.0], n=4)
    check_raises("quantiles bogus", "quantiles", [1.0, 2.0, 3.0], method="bogus")
    check_raises("quantiles bogus short", "quantiles", [1.0], method="bogus")
    check_raises("quantiles bogus empty n0", "quantiles", [], n=0, method="bogus")
    check_raises("quantiles n float", "quantiles", [1.0, 2.0, 3.0], n=4.0)


def test_quantiles_sorted_input_flag():
    # A sorted input must not change the result (the kernel re-sorts anyway).
    data = make_floats(10, 250)
    assert sm.quantiles(data, n=100) == sm.quantiles(sorted(data), n=100)


# ---------------------------------------------------------------------------
# Signed zeros and NaN ordering (identity-defined cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        [0.0, -0.0, 1.0],
        [-0.0, 0.0, 1.0],
        [0.0, 0.0, -0.0, -0.0, 2.0],
        [1.0, math.nan, 2.0],
        [math.nan, 3.0, 1.0, 2.0],
        [0.0, -0.0, 1.0, -1.0],
    ],
)
@pytest.mark.parametrize("fn", ["median", "median_low", "median_high"])
def test_identity_defined_order(data, fn):
    check_call(f"{fn}({data})", fn, data)


@pytest.mark.parametrize(
    "data",
    [
        [0.0, -0.0, 1.0, 2.0, 3.0],
        [-0.0, 0.0, 1.0, 2.0, 3.0],
        [1.0, math.nan, 2.0, 3.0, 4.0],
    ],
)
@pytest.mark.parametrize("method", ["exclusive", "inclusive"])
def test_quantiles_identity_defined(data, method):
    check_call(f"quantiles({data},{method})", "quantiles", data, n=4, method=method)


# ---------------------------------------------------------------------------
# Error cases (messages must match exactly)
# ---------------------------------------------------------------------------


def test_error_messages():
    check_raises("mean []", "mean", [])
    check_raises("fmean []", "fmean", [])
    check_raises("median []", "median", [])
    check_raises("median_low []", "median_low", [])
    check_raises("median_high []", "median_high", [])
    check_raises("variance []", "variance", [])
    check_raises("variance [1]", "variance", [1.0])
    check_raises("variance [1] c", "variance", [1.0], 2.0)
    check_raises("stdev []", "stdev", [])
    check_raises("stdev [1]", "stdev", [1.0])
    check_raises("pvariance []", "pvariance", [])
    check_raises("pstdev []", "pstdev", [])
    check_raises("mode []", "mode", [])
    check_raises("mean str", "mean", ["a"])
    check_raises("variance str", "variance", ["a", "b"])
    check_raises("fmean overflow", "fmean", [1e308, 1e308, -1e308])
    check_raises("fmean inf-mix", "fmean", [math.inf, -math.inf])
    check_raises("fmean overflow ndarray", "fmean", np.array([1e308, 1e308, -1e308]))
    check_raises("fmean inf-mix ndarray", "fmean", np.array([math.inf, -math.inf]))
    check_raises("mean Decimal+float", "mean", [Decimal("1"), 1.5])
    check_raises("variance Decimal+float", "variance", [Decimal("1"), 1.5, 2.5])
    check_raises("mean Fraction+Decimal", "mean", [Fraction(1, 2), Decimal("0.5")])
    check_raises("median mixed unsortable", "median", [1, "a"])
    check_raises("quantiles mixed unsortable", "quantiles", [1, "a", 2])


def test_multimode_empty():
    check_call("multimode []", "multimode", [])


# ---------------------------------------------------------------------------
# Input shapes: tuple, range, generator, ndarray
# ---------------------------------------------------------------------------


def test_iterables():
    # Generators are single-use: each side gets a fresh one.
    assert sm.mean(x * 0.5 for x in range(10)) == statistics.mean(x * 0.5 for x in range(10))
    assert sm.variance(x * 0.25 for x in range(20)) == statistics.variance(x * 0.25 for x in range(20))
    assert sm.median(x for x in range(9)) == statistics.median(x for x in range(9))
    assert sm.fmean(x * 0.5 for x in range(10)) == statistics.fmean(x * 0.5 for x in range(10))
    check_call("mean range", "mean", range(7))
    check_call("variance range", "variance", range(50))
    check_call("quantiles range", "quantiles", range(30), n=10)
    check_call("mean tuple", "mean", (1.5, 2.5, 3.0))
    check_call("median tuple", "median", (3, 1, 2))
    check_call("variance tuple", "variance", (1.5, 2.5, 3.0, 4.5))
    check_call("quantiles tuple", "quantiles", (3.0, 1.0, 2.0, 4.0))


@pytest.mark.parametrize("fn", CENTRAL_FNS + SPREAD_FNS)
def test_ndarray_float(fn):
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 1.0, 500)
    got = getattr(sm, fn)(data)
    want = getattr(statistics, fn)(data)
    # Documented difference: builtin types for ndarray input (stdlib yields
    # numpy scalars); values must agree bit-for-bit.
    assert float(got) == float(want), fn
    assert repr(float(got)) == repr(float(want)), fn


@pytest.mark.parametrize("fn", ["mean", "variance", "stdev", "median", "quantiles"])
def test_ndarray_int(fn):
    # Integer ndarrays route to the stdlib path (numpy scalar coercion
    # semantics), so results are identical including the numpy scalar types
    # — and including the stdlib's numpy-specific AttributeError quirks.
    data = np.arange(1, 101, dtype=np.int64)
    try:
        got = getattr(sm, fn)(data)
    except Exception as e:
        got = f"RAISE {type(e).__name__}: {e}"
    try:
        want = getattr(statistics, fn)(data)
    except Exception as e:
        want = f"RAISE {type(e).__name__}: {e}"
    assert repr(got) == repr(want), fn


def test_ndarray_uint64_huge():
    data = np.array([2**64 - 1, 2**64 - 3, 5], dtype=np.uint64)
    got = sm.mean(data)
    want = statistics.mean(data)
    assert repr(got) == repr(want)
    got = sm.variance(data)
    want = statistics.variance(data)
    assert repr(got) == repr(want)


def test_ndarray_noncontiguous():
    base = np.arange(200, dtype=np.float64)
    data = base[::2]
    for fn in ["mean", "variance", "stdev", "median"]:
        got = getattr(sm, fn)(data)
        want = getattr(statistics, fn)(data)
        # float64 ndarrays are accelerated; builtin float vs np.float64.
        assert repr(float(got)) == repr(float(want)), fn
    got = sm.quantiles(data, n=10)
    want = statistics.quantiles(data, n=10)
    assert [repr(float(g)) for g in got] == [repr(float(w)) for w in want]


# ---------------------------------------------------------------------------
# Exactness spot checks (documented exact-rational semantics)
# ---------------------------------------------------------------------------


def test_exact_rational_semantics():
    # mean of ints is an int when the division is exact, float otherwise.
    assert type(sm.mean([1, 2, 3])) is int
    assert type(sm.mean([1, 2])) is float
    # variance of Decimal is a Decimal with context-rounded division.
    v = sm.variance([Decimal("27.5"), Decimal("30.25"), Decimal("30.25"), Decimal("34.5"), Decimal("41.75")])
    assert type(v) is Decimal
    assert v == Decimal("31.01875")
    # variance of Fractions is an exact Fraction.
    assert sm.variance([Fraction(1, 6), Fraction(1, 2), Fraction(5, 3)]) == Fraction(67, 108)
    assert type(sm.variance([Fraction(1, 6), Fraction(1, 2), Fraction(5, 3)])) is Fraction
    # stdev of Decimals is a correctly rounded Decimal.
    dec_data = [Decimal("1.5"), Decimal("2.5"), Decimal("2.5"), Decimal("2.75"), Decimal("3.25"), Decimal("4.75")]
    s = sm.stdev(dec_data)
    assert type(s) is Decimal
    assert s == statistics.stdev(dec_data)
    # _sum's exactness: cancellation that builtin sum loses is kept.
    data = [1e50, 1.0, -1e50] * 1000
    assert sm.mean(data) == statistics.mean(data)
    assert sm.mean(data) == Fraction(1000, 3000) or sm.mean(data) == 1.0 / 3


def test_fsum_quirks_replicated():
    # CPython fsum is order-dependent on intermediate overflow; replicate.
    check_raises("fsum overflow order", "fmean", [1e308, 1e308, -1e308])
    check_call("fsum cancel order", "fmean", [1e308, -1e308, 1e308])
    check_call("fsum halfway", "fmean", [1e-16, 1.0, 1e16])


# ---------------------------------------------------------------------------
# Column batch API
# ---------------------------------------------------------------------------


def _batch_check(fn_batch, fn_scalar, cols, **kwargs):
    try:
        got = fn_batch(cols, **kwargs)
    except Exception as e:
        got_exc = (type(e).__name__, str(e))
    else:
        got_exc = None
    try:
        want = [fn_scalar(c, **kwargs) for c in cols]
    except Exception as e:
        want_exc = (type(e).__name__, str(e))
    else:
        want_exc = None
    if got_exc is not None or want_exc is not None:
        assert got_exc == want_exc, f"{got_exc} != {want_exc}"
        return
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        if isinstance(w, np.generic) or (isinstance(w, list) and w and isinstance(w[0], np.generic)):
            # stdlib-on-ndarray yields numpy scalar types; values must agree.
            if isinstance(w, list):
                assert [float(x) for x in g] == [float(x) for x in w], f"batch[{i}]"
            else:
                assert float(g) == float(w), f"batch[{i}]"
            continue
        check_same(f"batch[{i}]", g, w)


def test_batch_lists():
    cols = [
        make_floats(20, 100),
        make_ints(21, 100),
        [1, 2, 3],
        [Fraction(1, 2), Fraction(1, 3)],
        [Decimal("0.5"), Decimal("1.5")],
        [1, 2.5, 3],
        [1.0, math.inf],
        [0.1] * 10,
    ]
    _batch_check(sm.mean_batch, statistics.mean, cols)
    _batch_check(sm.fmean_batch, statistics.fmean, cols)
    _batch_check(sm.variance_batch, statistics.variance, [c for c in cols if len(c) >= 2])
    _batch_check(sm.stdev_batch, statistics.stdev, [c for c in cols if len(c) >= 2])
    _batch_check(sm.pvariance_batch, statistics.pvariance, cols)
    _batch_check(sm.pstdev_batch, statistics.pstdev, cols)
    _batch_check(sm.median_batch, statistics.median, cols)
    _batch_check(sm.quantiles_batch, statistics.quantiles, cols, n=10)
    _batch_check(sm.quantiles_batch, statistics.quantiles, cols, n=4, method="inclusive")


def test_batch_ndarrays():
    rng = np.random.default_rng(22)
    cols = [rng.normal(0.0, 1.0, n) for n in (100, 1000, 50)]
    _batch_check(sm.mean_batch, statistics.mean, cols)
    _batch_check(sm.fmean_batch, statistics.fmean, cols)
    _batch_check(sm.variance_batch, statistics.variance, cols)
    _batch_check(sm.stdev_batch, statistics.stdev, cols)
    _batch_check(sm.pvariance_batch, statistics.pvariance, cols)
    _batch_check(sm.pstdev_batch, statistics.pstdev, cols)
    _batch_check(sm.median_batch, statistics.median, cols)
    _batch_check(sm.quantiles_batch, statistics.quantiles, cols, n=100)


def test_batch_mixed_shapes():
    cols = [[1.5, 2.5], np.array([1.5, 2.5, 3.5]), (1, 2, 3), range(5)]
    _batch_check(sm.mean_batch, statistics.mean, cols)
    _batch_check(sm.variance_batch, statistics.variance, cols)


def test_batch_errors():
    def expect_same_error(fn_batch, fn_scalar, cols, *args, **kwargs):
        try:
            got = fn_batch(cols, *args, **kwargs)
        except Exception as e:
            got_exc = (type(e).__name__, str(e))
        else:
            got_exc = ("NO ERROR", repr(got))
        try:
            for c in cols:
                fn_scalar(c, *args, **kwargs)
        except Exception as e:
            want_exc = (type(e).__name__, str(e))
        else:
            want_exc = ("NO ERROR", None)
        assert got_exc == want_exc, f"{got_exc} != {want_exc}"

    expect_same_error(sm.mean_batch, statistics.mean, [[1.0], []])
    expect_same_error(sm.variance_batch, statistics.variance, [[1.0, 2.0], [3.0]])
    expect_same_error(sm.fmean_batch, statistics.fmean, [[]])
    expect_same_error(sm.quantiles_batch, statistics.quantiles, [[1.0, 2.0]], n=0)
    expect_same_error(sm.quantiles_batch, statistics.quantiles, [[1.0, 2.0]], method="bogus")
    expect_same_error(sm.quantiles_batch, statistics.quantiles, [[1.0]])
    expect_same_error(sm.median_batch, statistics.median, [[1.0], []])


def test_batch_empty():
    assert sm.mean_batch([]) == []
    assert sm.variance_batch([]) == []
    assert sm.quantiles_batch([], n=100) == []


# ---------------------------------------------------------------------------
# Large-data parity (the accelerated path at scale)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", CENTRAL_FNS + SPREAD_FNS + ["quantiles"])
def test_large_float_list(fn):
    data = make_floats(30, 50_000)
    if fn == "quantiles":
        check_call("large quantiles", fn, data, n=100)
    else:
        check_call(f"large {fn}", fn, data)


@pytest.mark.parametrize("fn", ["mean", "variance", "stdev", "median", "quantiles"])
def test_large_int_list(fn):
    data = make_ints(31, 50_000)
    if fn == "quantiles":
        check_call("large int quantiles", fn, data, n=100)
    else:
        check_call(f"large int {fn}", fn, data)


def test_native_backend_actually_serves(monkeypatch):
    """On the native run, hot paths must not touch the stdlib fallback."""
    if os.environ.get("STATISTICS_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback run")
    assert sm.native_available()

    def boom(*args, **kwargs):
        raise AssertionError("stdlib fallback engaged on the native path")

    data = make_floats(40, 1000)
    idata = make_ints(41, 1000)
    monkeypatch.setattr(sm._reference, "mean", boom)
    monkeypatch.setattr(sm._reference, "variance", boom)
    monkeypatch.setattr(sm._reference, "median", boom)
    monkeypatch.setattr(sm._reference, "quantiles", boom)
    monkeypatch.setattr(sm._reference, "fmean", boom)
    sm.mean(data)
    sm.mean(idata)
    sm.variance(data)
    sm.stdev(idata)
    sm.median(data)
    sm.median(idata)
    sm.quantiles(data, n=10)
    sm.fmean(data)
    sm.mean_batch([data, data])
    sm.variance_batch([data, data])


def test_fallback_used_for_exotic_inputs(monkeypatch):
    """Decimal/Fraction data must route to the stdlib implementation."""
    if os.environ.get("STATISTICS_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback run")
    called = []
    real = statistics.mean

    def spy(data):
        called.append(1)
        return real(data)

    monkeypatch.setattr(sm._reference, "mean", spy)
    sm.mean([Decimal("0.5"), Decimal("1.5")])
    sm.mean([Fraction(1, 3), Fraction(2, 3)])
    assert called, "exotic inputs must route to the stdlib path"
