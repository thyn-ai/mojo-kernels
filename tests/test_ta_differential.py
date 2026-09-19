"""Differential tests: ta_mojo must match pandas-ta-classic element-for-element.

Run twice by `scripts/test_all_ta.sh`: once against the native Mojo kernel and
once with TA_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback). Both
backends must agree with the oracle within 1e-10 everywhere, with exactly
equal NaN masks (in practice the agreement is bit-exact or within a few ulps,
~1e-14).

The oracle is the published PyPI package `pandas-ta-classic` (the maintained
successor to twopirllc/pandas-ta), exercised through its pandas code path:
TA-Lib must NOT be installed in the test environment, otherwise the oracle
switches implementations and this suite is meaningless (guarded below).

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible on any
machine.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="oracle dependency")
ta = pytest.importorskip("pandas_ta_classic", reason="differential oracle")

import ta_mojo

ATOL = 1e-10  # documented tolerance; measured agreement is ~1e-14


# ---------------------------------------------------------------- fixtures


def make_close(seed: int, n: int) -> np.ndarray:
    """Deterministic random-walk close series (float64)."""
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))


def make_ohlc(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic high/low/close with high >= close >= low."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    high = close + rng.uniform(0.0, 1.5, n)
    low = close - rng.uniform(0.0, 1.5, n)
    return high, low, close


def with_nan(x: np.ndarray, leading: int = 0, interior: tuple[int, ...] = ()) -> np.ndarray:
    out = x.copy()
    out[:leading] = np.nan
    for i in interior:
        out[i] = np.nan
    return out


def assert_parity(ours: np.ndarray, ref) -> None:
    """Exact NaN-mask equality plus values within the documented tolerance."""
    ref_arr = np.asarray(ref, dtype=np.float64)
    assert isinstance(ours, np.ndarray), f"expected np.ndarray, got {type(ours)}"
    assert ours.dtype == np.float64, f"expected float64, got {ours.dtype}"
    assert ours.shape == ref_arr.shape
    ours_nan = np.isnan(ours)
    ref_nan = np.isnan(ref_arr)
    assert np.array_equal(ours_nan, ref_nan), (
        f"NaN masks differ at {np.flatnonzero(ours_nan != ref_nan)[:10]}"
    )
    valid = ~ref_nan
    if valid.any():
        np.testing.assert_allclose(ours[valid], ref_arr[valid], rtol=0, atol=ATOL)


def macd_cols(fast: int, slow: int, signal: int) -> tuple[str, str, str]:
    if slow < fast:
        fast, slow = slow, fast
    base = f"_{fast}_{slow}_{signal}"
    return f"MACD{base}", f"MACDs{base}", f"MACDh{base}"


def ref_macd(close, fast: int, slow: int, signal: int):
    df = ta.macd(pd.Series(close), fast=fast, slow=slow, signal=signal)
    line_col, signal_col, hist_col = macd_cols(fast, slow, signal)
    return df[line_col], df[signal_col], df[hist_col]


# ---------------------------------------------------------------- guards


def test_oracle_uses_pandas_path():
    # The oracle switches to TA-Lib when it is installed; this suite targets
    # the pandas implementation that pip installs deliver by default.
    assert ta.Imports["talib"] is False, "TA-Lib present: oracle would not use its pandas path"


def test_expected_backend():
    info = ta_mojo.backend_info()
    if os.environ.get("TA_MOJO_DISABLE_NATIVE") == "1":
        assert info["native_available"] is False
    else:
        # The native run requires a built kernel (kernels/ta/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == 1


# ---------------------------------------------------------------- ema


@pytest.mark.parametrize("length", [3, 10, 50])
@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("sma", [True, False])
def test_ema_parity(length, adjust, sma):
    close = make_close(seed=101, n=2000)
    ours = ta_mojo.ema(close, length, adjust=adjust, sma=sma)
    ref = ta.ema(pd.Series(close), length=length, adjust=adjust, sma=sma)
    assert_parity(ours, ref)


def test_ema_leading_and_interior_nan():
    close = with_nan(make_close(seed=103, n=400), leading=5, interior=(77, 120, 200))
    for adjust in (False, True):
        for sma in (True, False):
            ours = ta_mojo.ema(close, 10, adjust=adjust, sma=sma)
            ref = ta.ema(pd.Series(close), length=10, adjust=adjust, sma=sma)
            assert_parity(ours, ref)


def test_ema_constant_series():
    close = np.full(300, 42.0)
    for adjust in (False, True):
        assert_parity(ta_mojo.ema(close, 10, adjust=adjust), ta.ema(pd.Series(close), 10, adjust=adjust))


def test_ema_length_one():
    close = make_close(seed=105, n=200)
    assert_parity(ta_mojo.ema(close, 1), ta.ema(pd.Series(close), 1))


def test_ema_shorter_than_length_returns_all_nan():
    close = make_close(seed=106, n=7)
    ours = ta_mojo.ema(close, 10)
    assert ours.shape == (7,) and np.isnan(ours).all()
    # The oracle declines this input entirely; an all-NaN array is our
    # array-valued equivalent.
    assert ta.ema(pd.Series(close), 10) is None


# ---------------------------------------------------------------- rsi


@pytest.mark.parametrize("length", [1, 2, 14, 27])
@pytest.mark.parametrize("drift", [1, 2])
def test_rsi_parity(length, drift):
    close = make_close(seed=111, n=2000)
    ours = ta_mojo.rsi(close, length, drift=drift)
    ref = ta.rsi(pd.Series(close), length=length, drift=drift)
    assert_parity(ours, ref)


def test_rsi_scalar():
    close = make_close(seed=112, n=500)
    assert_parity(ta_mojo.rsi(close, 14, scalar=1.0), ta.rsi(pd.Series(close), 14, scalar=1.0))


def test_rsi_leading_and_interior_nan():
    close = with_nan(make_close(seed=113, n=400), leading=3, interior=(50, 51, 150))
    assert_parity(ta_mojo.rsi(close, 14), ta.rsi(pd.Series(close), length=14))


def test_rsi_constant_series_is_nan_after_seed():
    close = np.full(200, 7.0)
    ours = ta_mojo.rsi(close, 14)
    ref = ta.rsi(pd.Series(close), length=14)
    # up == down == 0 after the seed, so 0/0 -> NaN on both sides.
    assert_parity(ours, ref)
    assert np.isnan(ours).all()


def test_rsi_monotone_up_is_100():
    close = np.arange(1.0, 201.0)  # every diff positive, down avg exactly 0
    ours = ta_mojo.rsi(close, 14)
    ref = ta.rsi(pd.Series(close), length=14)
    assert_parity(ours, ref)
    assert np.allclose(ours[14:], 100.0, rtol=0, atol=ATOL)


# ---------------------------------------------------------------- atr


@pytest.mark.parametrize("length", [1, 14, 30])
@pytest.mark.parametrize("drift", [1, 3])
def test_atr_parity(length, drift):
    high, low, close = make_ohlc(seed=121, n=2000)
    ours = ta_mojo.atr(high, low, close, length, drift=drift)
    ref = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=length, drift=drift)
    assert_parity(ours, ref)


def test_atr_flat_bars_use_epsilon_range():
    # high == low on every bar: the zero range becomes machine epsilon.
    close = make_close(seed=122, n=300)
    flat = close.copy()
    ours = ta_mojo.atr(flat, flat, close, 14)
    ref = ta.atr(pd.Series(flat), pd.Series(flat), pd.Series(close), length=14)
    assert_parity(ours, ref)


def test_atr_leading_and_interior_nan():
    high, low, close = make_ohlc(seed=123, n=400)
    close = with_nan(close, leading=4, interior=(100,))
    high = with_nan(high, leading=4, interior=(100,))
    low = with_nan(low, leading=4, interior=(100,))
    ours = ta_mojo.atr(high, low, close, 14)
    ref = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=14)
    assert_parity(ours, ref)


# ---------------------------------------------------------------- macd


@pytest.mark.parametrize(
    "fast,slow,signal",
    [(12, 26, 9), (8, 21, 5), (26, 12, 9), (5, 13, 4), (1, 2, 1), (1, 1, 1)],
)
def test_macd_parity(fast, slow, signal):
    close = make_close(seed=131, n=2000)
    ours = ta_mojo.macd(close, fast, slow, signal)
    ref_line, ref_signal, ref_hist = ref_macd(close, fast, slow, signal)
    assert_parity(ours.macd, ref_line)
    assert_parity(ours.signal, ref_signal)
    assert_parity(ours.histogram, ref_hist)


def test_macd_leading_and_interior_nan():
    close = with_nan(make_close(seed=133, n=400), leading=6, interior=(150,))
    ours = ta_mojo.macd(close, 12, 26, 9)
    ref_line, ref_signal, ref_hist = ref_macd(close, 12, 26, 9)
    assert_parity(ours.macd, ref_line)
    assert_parity(ours.signal, ref_signal)
    assert_parity(ours.histogram, ref_hist)


def test_macd_constant_series():
    close = np.full(300, 11.0)
    ours = ta_mojo.macd(close, 12, 26, 9)
    ref_line, ref_signal, ref_hist = ref_macd(close, 12, 26, 9)
    assert_parity(ours.macd, ref_line)
    assert_parity(ours.signal, ref_signal)
    assert_parity(ours.histogram, ref_hist)


def test_macd_shorter_than_slow_returns_all_nan():
    close = make_close(seed=134, n=20)
    ours = ta_mojo.macd(close, 12, 26, 9)
    for arr in ours:
        assert arr.shape == (20,) and np.isnan(arr).all()
    assert ta.macd(pd.Series(close), fast=12, slow=26, signal=9) is None


# ------------------------------------------------------------ input handling


def test_input_validation():
    close = make_close(seed=141, n=100)
    with pytest.raises(ValueError):
        ta_mojo.ema(close.reshape(10, 10), 10)  # not 1-D
    with pytest.raises(ValueError):
        ta_mojo.ema(close, 0)  # non-positive length
    with pytest.raises(ValueError):
        ta_mojo.rsi(close, -3)
    with pytest.raises(ValueError):
        ta_mojo.rsi(close, 14, scalar=float("nan"))
    with pytest.raises(ValueError):
        ta_mojo.atr(close[:50], close, close, 14)  # length mismatch
    with pytest.raises(ValueError):
        ta_mojo.macd(close, 12, 26, 0)
    with pytest.raises(ValueError):
        ta_mojo.ema(object(), 10)  # not array-like floats


def test_array_like_inputs_and_empty():
    assert_parity(ta_mojo.ema([1.0, 2.0, 3.0, 4.0, 5.0], 2), ta.ema(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 2))
    assert_parity(ta_mojo.ema(range(1, 30), 5), ta.ema(pd.Series(range(1, 30), dtype=float), 5))
    empty = ta_mojo.ema(np.array([]), 10)
    assert empty.shape == (0,) and empty.dtype == np.float64
    assert ta_mojo.macd(np.array([])).macd.shape == (0,)


# ------------------------------------------------------- cross-backend unity


def test_native_and_fallback_agree(monkeypatch):
    # The Mojo kernel and the vendored fallback execute the same IEEE-754
    # float64 operations in the same order, but compiled code (the Mojo
    # kernel, like the oracle's own Cython wheels) may contract a*b+c into a
    # fused multiply-add while CPython bytecode cannot — so the backends can
    # differ by ~1 ulp (~1e-14). They must still have exactly equal NaN masks
    # and agree far below the documented 1e-10 tolerance.
    high, low, close = make_ohlc(seed=151, n=500)
    results = {}
    for disabled in (None, "1"):
        if disabled is None:
            monkeypatch.delenv("TA_MOJO_DISABLE_NATIVE", raising=False)
        else:
            monkeypatch.setenv("TA_MOJO_DISABLE_NATIVE", disabled)
        m = ta_mojo.macd(close, 12, 26, 9)
        results[disabled] = (
            ta_mojo.ema(close, 10),
            ta_mojo.ema(close, 10, adjust=True),
            ta_mojo.ema(close, 10, sma=False),
            ta_mojo.rsi(close, 14),
            ta_mojo.atr(high, low, close, 14),
            m.macd,
            m.signal,
            m.histogram,
        )
    for native_arr, fallback_arr in zip(results[None], results["1"]):
        assert np.array_equal(np.isnan(native_arr), np.isnan(fallback_arr))
        np.testing.assert_allclose(native_arr, fallback_arr, rtol=0, atol=1e-12)


# ------------------------------------------------------ agreement measurement


def test_measured_agreement_well_below_tolerance(capsys):
    """Report the real max abs diff over a battery; it must sit far under 1e-10."""
    worst = 0.0

    def track(ours, ref):
        nonlocal worst
        ref_arr = np.asarray(ref, dtype=np.float64)
        valid = ~np.isnan(ref_arr)
        if valid.any():
            worst = max(worst, float(np.max(np.abs(ours[valid] - ref_arr[valid]))))

    for seed in (161, 162, 163):
        high, low, close = make_ohlc(seed=seed, n=3000)
        sc, sh, sl = pd.Series(close), pd.Series(high), pd.Series(low)
        track(ta_mojo.ema(close, 10), ta.ema(sc, 10))
        track(ta_mojo.ema(close, 10, adjust=True), ta.ema(sc, 10, adjust=True))
        track(ta_mojo.rsi(close, 14), ta.rsi(sc, 14))
        track(ta_mojo.atr(high, low, close, 14), ta.atr(sh, sl, sc, 14))
        m = ta_mojo.macd(close, 12, 26, 9)
        rl, rs, rh = ref_macd(close, 12, 26, 9)
        track(m.macd, rl)
        track(m.signal, rs)
        track(m.histogram, rh)
    with capsys.disabled():
        print(f"\nmeasured max|diff| over agreement battery: {worst:.3e}")
    assert worst <= 1e-12, f"agreement {worst:.3e} worse than the expected ulp level"
