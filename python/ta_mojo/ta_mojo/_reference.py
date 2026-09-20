"""Vendored pure-Python reference indicators (EMA, RMA, True Range, MACD).

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``TA_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
published indicator definitions (Wilder, "New Concepts in Technical Trading
Systems", 1978, for RSI/ATR/RMA; Appel for MACD) written to be observably
identical to the pandas-based reference stack the differential suite checks
against (pandas ``ewm().mean()`` as used by the pandas-ta-classic indicator
functions): same SMA-seeded warmup, same NaN-prefix semantics, same IEEE-754
float64 operation order (Python floats are C doubles), so both backends agree
bit-for-bit in practice.

Only the recurrences live here; input validation and the numpy element-wise
glue (diff/clip/combine) are shared with the native path in `ta_mojo.core`,
so the two backends can never disagree about anything but the kernel itself.
"""

from __future__ import annotations

import math

import numpy as np

NAN = math.nan
EPS = float(np.finfo(np.float64).eps)  # 2**-52, for the TR zero-bar rule


def _ewm_tail(
    x: list[float],
    n: int,
    com: float,
    adjust: bool,
    out: np.ndarray,
    start: int,
    weighted: float,
    old_wt: float,
    nobs: int,
) -> None:
    """Continue the reference ewm().mean() recursion from index `start`.

    Replicates the float64 operation order of the reference implementation
    (ignore_na=False, normalize=True, min_periods=1): one fused
    multiply-and-divide weighted update per observation, the constant-series
    shortcut, and the com == 1 irregular-interval weight for the non-adjusted
    variant.
    """
    alpha = 1.0 / (1.0 + com)
    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha
    w = weighted
    wt = old_wt
    nobs_i = nobs
    for i in range(start, n):
        cur = x[i]
        is_obs = cur == cur  # NaN check; +/-inf counts as an observation
        if is_obs:
            nobs_i += 1
        if w == w:
            # ignore_na is False for every supported indicator: the weight
            # decays on every bar, observed or not.
            wt *= old_wt_factor
            if is_obs:
                if w != cur:
                    if not adjust and com == 1.0:
                        # irregular-interval refresh of the inbound weight
                        new_wt = 1.0 - wt
                    w = wt * w + new_wt * cur
                    w /= wt + new_wt
                if adjust:
                    wt += new_wt
                else:
                    wt = 1.0
        elif is_obs:
            w = cur
        out[i] = w if nobs_i >= 1 else NAN


def _ewm_mean(x: list[float], com: float, adjust: bool, out: np.ndarray) -> None:
    """Whole-series reference ewm().mean() (min_periods=1, ignore_na=False)."""
    first = x[0]
    nobs = 1 if first == first else 0
    out[0] = first if nobs >= 1 else NAN
    _ewm_tail(x, len(x), com, adjust, out, 1, first, 1.0, nobs)


def _sma_seeded_ewm(
    x: list[float], length: int, com: float, adjust: bool, out: np.ndarray
) -> None:
    """SMA-seeded exponentially weighted recursion (TA-Lib-style warmup).

    The seed is the NaN-skipping mean of the first `length` values starting
    at the first non-NaN position, placed at fv + length - 1; earlier
    positions are NaN. If fewer than `length` values follow the first valid
    one, the result is NaN everywhere (the seed does not exist).
    """
    n = len(x)
    fv = 0
    while fv < n and x[fv] != x[fv]:
        fv += 1
    if fv + length > n:
        out[:] = NAN
        return
    s = 0.0
    cnt = 0
    for i in range(fv, fv + length):
        v = x[i]
        if v == v:
            s += v
            cnt += 1
    seed = s / float(cnt)  # cnt >= 1 (x[fv] is not NaN)
    out[: fv + length - 1] = NAN
    out[fv + length - 1] = seed
    _ewm_tail(x, n, com, adjust, out, fv + length, seed, 1.0, 1)


def _ema_aligned(
    x: list[float], m: int, period: int, seed_end: int, out: np.ndarray
) -> None:
    """Plain-SMA-seeded EMA: seed = mean(x[seed_end-period+1 .. seed_end])
    (NOT NaN-skipping), then out[i] = k*x[i] + (1-k)*out[i-1] with
    k = 2/(period+1). Positions before seed_end are NaN; a NaN seed or NaN
    input propagates through the rest of the series."""
    k = 2.0 / (period + 1)
    start = seed_end - period + 1
    if start < 0 or seed_end >= m:
        out[:] = NAN
        return
    s = 0.0
    for i in range(start, seed_end + 1):
        s += x[i]
    out[:seed_end] = NAN
    out[seed_end] = s / float(period)
    prev = out[seed_end]
    for i in range(seed_end + 1, m):
        prev = k * x[i] + (1.0 - k) * prev
        out[i] = prev


def ema(x: np.ndarray, length: int, adjust: bool, sma: bool) -> np.ndarray:
    """EMA of a float64 series; NaN warmup prefix of length-1 when sma=True."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    xs = x.tolist()  # Python floats: exact IEEE-754 double ops in the loops
    com = (float(length) - 1.0) / 2.0
    if sma:
        _sma_seeded_ewm(xs, length, com, adjust, out)
    else:
        _ewm_mean(xs, com, adjust, out)
    return out


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """Wilder's moving average (alpha = 1/length, SMA-seeded)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    xs = x.tolist()
    # Round-trip through center-of-mass exactly like the reference stack:
    # ewm(alpha=a) stores com = (1-a)/a and recomputes alpha = 1/(1+com).
    alpha = 1.0 / float(length)
    com = (1.0 - alpha) / alpha
    _sma_seeded_ewm(xs, length, com, False, out)
    return out


def true_range(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, drift: int
) -> np.ndarray:
    """True range: NaN-skipping row max of |h-l|, |h-prev_c|, |prev_c-l|.

    An exact zero high-low range is replaced by machine epsilon; the first
    (first all-valid row) + drift bars are NaN.
    """
    n = h.shape[0]
    hl = h - l
    hl = np.where(hl == 0.0, EPS, hl)  # -0.0 == 0.0, so it is replaced too
    pc = np.empty(n, dtype=np.float64)
    pc[:drift] = NAN
    pc[drift:] = c[: n - drift]
    # np.fmax skips NaN operands; all-NaN stays NaN. Max is exact in float64.
    tr = np.fmax(np.fmax(np.abs(hl), np.abs(h - pc)), np.abs(pc - l))
    finite = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
    start = int(np.argmax(finite)) if bool(finite.any()) else n
    tr[: min(start + drift, n)] = NAN
    return tr


def macd(
    x: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (fast <= slow after the caller's swap) -> (macd, signal, hist).

    Leading non-finite rows (NaN or +/-inf) are skipped and padded back with
    NaN. The fast/slow EMAs are plain-SMA-seeded at fast-1 / slow-1; the
    signal EMA is seeded at (slow-1) + (signal-1) from the MACD line.
    """
    n = x.shape[0]
    out_macd = np.full(n, NAN, dtype=np.float64)
    out_signal = np.full(n, NAN, dtype=np.float64)
    out_hist = np.full(n, NAN, dtype=np.float64)
    if n == 0:
        return out_macd, out_signal, out_hist
    finite = np.isfinite(x)
    lead = int(np.argmax(finite)) if bool(finite.any()) else n
    m = n - lead
    if m == 0:
        return out_macd, out_signal, out_hist
    xs = x[lead:].tolist()
    fe = np.empty(m, dtype=np.float64)
    se = np.empty(m, dtype=np.float64)
    mc = np.empty(m, dtype=np.float64)
    _ema_aligned(xs, m, fast, fast - 1, fe)
    _ema_aligned(xs, m, slow, slow - 1, se)
    for i in range(m):
        mc[i] = fe[i] - se[i]
    _ema_aligned(mc.tolist(), m, signal, (slow - 1) + (signal - 1), se)
    out_macd[lead:] = mc
    out_signal[lead:] = se
    out_hist[lead:] = mc - se
    return out_macd, out_signal, out_hist
