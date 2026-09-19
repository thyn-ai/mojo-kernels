"""Public indicator API: ema, rsi, atr, macd — array-like in, float64 out.

Each function first tries the native Mojo kernel and transparently falls back
to the vendored pure-Python reference when the kernel is unavailable; both
backends are covered by the same differential suite against the
pandas-ta-classic oracle (documented tolerance 1e-10, NaN masks exactly equal;
measured agreement is at the 1e-14 level).

Semantics matched to the oracle (pandas path, no TA-Lib):

* ema: SMA-seeded warmup (``sma=True``, the oracle default): seed = mean of
  the first ``length`` values at index ``length - 1``, NaN before it; then the
  exponential recursion with ``alpha = 2 / (length + 1)`` in the reference
  ``adjust`` weighting (``adjust=False`` default; ``adjust=True`` supported).
* rsi: Wilder smoothing (RMA, SMA-seeded, ``alpha = 1 / length``) of the
  clipped one-bar differences; ``rsi = scalar * up / (up + |down|)``.
* atr: true range (zero high-low bars become machine epsilon; the first bar
  is NaN) smoothed with the same RMA.
* macd: plain-SMA-seeded EMAs seeded at ``fast - 1`` / ``slow - 1``; signal
  line seeded ``signal - 1`` bars after the MACD line's first value.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ta_mojo import _native, _reference


class MacdResult(NamedTuple):
    """MACD output trio: the MACD line, the signal line, and their difference."""

    macd: np.ndarray
    signal: np.ndarray
    histogram: np.ndarray


def _as_f64(name: str, value) -> np.ndarray:
    """Coerce array-like input to a contiguous float64 1-D array."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D array-like of floats: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape!r}")
    return np.ascontiguousarray(arr)


def _pos_int(name: str, value, default: int) -> int:
    """Positive-int parameter: None falls back to the oracle default.

    Unlike the oracle (which silently replaces an invalid value with the
    default), a non-positive or non-integer value is a caller error here.
    """
    if value is None:
        return default
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive int, got {value!r}") from exc
    if iv < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return iv


def _rma(x: np.ndarray, length: int) -> np.ndarray:
    """Backend-dispatched Wilder RMA (internal)."""
    try:
        return _native.rma(x, length)
    except _native.NativeUnavailable:
        return _reference.rma(x, length)


def ema(close, length=10, *, adjust=False, sma=True) -> np.ndarray:
    """Exponential Moving Average of `close`. Returns float64 ndarray.

    `sma=True` (the oracle default) seeds the recursion with the SMA of the
    first `length` values (NaN warmup prefix of `length - 1`); `sma=False`
    runs the plain recursion from the first value (no NaN prefix).
    """
    x = _as_f64("close", close)
    len_ = _pos_int("length", length, 10)
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n < len_:
        # The oracle declines a series shorter than `length`; the array-valued
        # equivalent is an all-NaN result.
        return np.full(n, np.nan, dtype=np.float64)
    try:
        return _native.ema(x, len_, bool(adjust), bool(sma))
    except _native.NativeUnavailable:
        return _reference.ema(x, len_, bool(adjust), bool(sma))


def rsi(close, length=14, *, scalar=100.0, drift=1) -> np.ndarray:
    """Wilder's Relative Strength Index of `close`. Returns float64 ndarray."""
    x = _as_f64("close", close)
    len_ = _pos_int("length", length, 14)
    drift_ = _pos_int("drift", drift, 1)
    scalar_ = float(scalar)
    if not math.isfinite(scalar_):
        raise ValueError(f"scalar must be finite, got {scalar!r}")
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n < len_:
        return np.full(n, np.nan, dtype=np.float64)
    diff = np.empty(n, dtype=np.float64)
    diff[:drift_] = np.nan
    diff[drift_:] = x[drift_:] - x[: n - drift_]
    # NaN comparisons are False, so NaN differences stay NaN in both clips.
    positive = np.where(diff < 0, 0.0, diff)
    negative = np.where(diff > 0, 0.0, diff)
    positive_avg = _rma(np.ascontiguousarray(positive), len_)
    negative_avg = _rma(np.ascontiguousarray(negative), len_)
    # Same operation order as the oracle: (scalar * up) / (up + |down|).
    # Flat bars make up + |down| == 0; the oracle yields NaN there without a
    # numpy warning, so mirror that (0/0 -> NaN) quietly.
    with np.errstate(invalid="ignore", divide="ignore"):
        return scalar_ * positive_avg / (positive_avg + np.abs(negative_avg))


def atr(high, low, close, length=14, *, drift=1) -> np.ndarray:
    """Average True Range (Wilder smoothing). Returns float64 ndarray."""
    h = _as_f64("high", high)
    l = _as_f64("low", low)
    c = _as_f64("close", close)
    len_ = _pos_int("length", length, 14)
    drift_ = _pos_int("drift", drift, 1)
    if not (h.shape[0] == l.shape[0] == c.shape[0]):
        raise ValueError(
            "high, low and close must have the same length, got "
            f"{h.shape[0]}, {l.shape[0]} and {c.shape[0]}"
        )
    n = h.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n < len_:
        return np.full(n, np.nan, dtype=np.float64)
    try:
        tr = _native.true_range(h, l, c, drift_)
    except _native.NativeUnavailable:
        tr = _reference.true_range(h, l, c, drift_)
    return _rma(np.ascontiguousarray(tr), len_)


def macd(close, fast=12, slow=26, signal=9) -> MacdResult:
    """Moving Average Convergence/Divergence of `close`.

    Returns a :class:`MacdResult` of three float64 ndarrays (macd, signal,
    histogram). If ``slow < fast`` the two are swapped, like the oracle.
    """
    x = _as_f64("close", close)
    fast_ = _pos_int("fast", fast, 12)
    slow_ = _pos_int("slow", slow, 26)
    signal_ = _pos_int("signal", signal, 9)
    if slow_ < fast_:
        fast_, slow_ = slow_, fast_
    n = x.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return MacdResult(empty, empty.copy(), empty.copy())
    if n < max(fast_, slow_, signal_):
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        return MacdResult(nan_arr, nan_arr.copy(), nan_arr.copy())
    try:
        m, s, h = _native.macd(x, fast_, slow_, signal_)
    except _native.NativeUnavailable:
        m, s, h = _reference.macd(x, fast_, slow_, signal_)
    return MacdResult(m, s, h)
