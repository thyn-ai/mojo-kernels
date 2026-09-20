"""Clean-room technical-analysis indicator kernels (EMA, Wilder RMA, True
Range, MACD) on float64 series.

Written fresh from the published indicator definitions (Wilder, "New Concepts
in Technical Trading Systems", 1978, for RSI/ATR/RMA; Appel for MACD). The
loop-carried recurrences deliberately replicate, step for step, the IEEE-754
float64 operation order of the reference software stack they are checked
against (pandas `ewm().mean()` with `adjust`/`min_periods` as used by the
pandas-ta-classic indicator functions), so outputs agree with that oracle to
within 1e-10 (bit-identical in practice). No third-party Mojo code is used or
adapted.

Exported C ABI (stateless: every call computes one whole series):

    int32_t tamojo_abi_version(void)
    int32_t tamojo_ema(x, n, length, adjust, sma, dst)
    int32_t tamojo_rma(x, n, length, dst)
    int32_t tamojo_true_range(h, l, c, n, drift, dst)
    int32_t tamojo_macd(x, n, fast, slow, signal, out_macd, out_signal,
                        out_hist)

EMA/RMA seeding (matches the oracle's TA-Lib-style behaviour): the first
`length` values from the first non-NaN position are averaged (NaN-skipping)
into an SMA seed placed at position `fv + length - 1`; everything before it is
NaN. From the seed on, the exponentially weighted recursion runs with
alpha = 2/(length+1) for EMA and alpha = 1/length for RMA, including the
incremental-weight bookkeeping (`old_wt`/`new_wt`), the constant-series
shortcut, and the com == 1 irregular-interval branch of the reference ewm
implementation. MACD uses the plain (non-NaN-skipping) SMA seed and the
`k*x + (1-k)*prev` recurrence, with the signal line seeded `signal-1` bars
after the MACD line's first value.
"""

from std.math import isfinite, nan
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spelling (untracked origin: the caller owns every buffer).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]

comptime NAN = nan[DType.float64]()
# np.finfo(float).eps, used for the true-range zero-bar rule (2**-52 exactly).
comptime EPS = Float64(2.220446049250313e-16)


def _abs(v: Float64) -> Float64:
    return v if v >= 0.0 else -v


def _ewm_tail(
    x: F64Ptr,
    n: Int,
    com: Float64,
    adjust: Bool,
    dst: F64Ptr,
    start: Int,
    weighted: Float64,
    old_wt: Float64,
    nobs: Int,
):
    """Continue the reference ewm().mean() recursion from index `start`.

    Replicates the float64 operation order of the reference implementation
    (ignore_na=False, normalize=True, min_periods=1): one fused
    multiply-and-divide weighted update per observation, the constant-series
    shortcut (an unchanged estimate when `weighted == cur`), and the
    com == 1 irregular-interval weight for the non-adjusted variant.
    """
    var alpha = 1.0 / (1.0 + com)
    var old_wt_factor = 1.0 - alpha
    var new_wt = 1.0 if adjust else alpha
    var w = weighted
    var wt = old_wt
    var nobs_i = nobs
    for i in range(start, n):
        var cur = x[unsafe_offset=i]
        var is_obs = cur == cur  # NaN check; +/-inf counts as an observation
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
        dst[unsafe_offset=i] = w if nobs_i >= 1 else NAN


def _ewm_mean(x: F64Ptr, n: Int, com: Float64, adjust: Bool, dst: F64Ptr):
    """Whole-series reference ewm().mean() (min_periods=1, ignore_na=False)."""
    var first = x[unsafe_offset=0]
    var nobs = 1 if first == first else 0
    dst[unsafe_offset=0] = first if nobs >= 1 else NAN
    _ewm_tail(x, n, com, adjust, dst, 1, first, 1.0, nobs)


def _sma_seeded_ewm(
    x: F64Ptr, n: Int, length: Int, com: Float64, adjust: Bool, dst: F64Ptr
):
    """SMA-seeded exponentially weighted recursion (TA-Lib-style warmup).

    The seed is the NaN-skipping mean of the first `length` values starting
    at the first non-NaN position, placed at fv + length - 1; earlier
    positions are NaN. If fewer than `length` values follow the first valid
    one, the result is NaN everywhere (the seed does not exist).
    """
    var fv = 0
    while fv < n and x[unsafe_offset=fv] != x[unsafe_offset=fv]:
        fv += 1
    if fv + length > n:
        for i in range(n):
            dst[unsafe_offset=i] = NAN
        return
    var s = Float64(0.0)
    var cnt = 0
    for i in range(fv, fv + length):
        var v = x[unsafe_offset=i]
        if v == v:
            s += v
            cnt += 1
    var seed = s / Float64(cnt)  # cnt >= 1 (x[fv] is not NaN)
    for i in range(fv + length - 1):
        dst[unsafe_offset=i] = NAN
    dst[unsafe_offset=fv + length - 1] = seed
    _ewm_tail(x, n, com, adjust, dst, fv + length, seed, 1.0, 1)


def _ema_aligned(
    x: F64Ptr, m: Int, period: Int, seed_end: Int, dst: F64Ptr
):
    """Plain-SMA-seeded EMA: seed = mean(x[seed_end-period+1 .. seed_end])
    (NOT NaN-skipping), then dst[i] = k*x[i] + (1-k)*dst[i-1] with
    k = 2/(period+1). Positions before seed_end are NaN; a NaN seed or NaN
    input propagates through the rest of the series."""
    var k = 2.0 / Float64(period + 1)
    var start = seed_end - period + 1
    if start < 0 or seed_end >= m:
        for i in range(m):
            dst[unsafe_offset=i] = NAN
        return
    var s = Float64(0.0)
    for i in range(start, seed_end + 1):
        s += x[unsafe_offset=i]
    for i in range(seed_end):
        dst[unsafe_offset=i] = NAN
    dst[unsafe_offset=seed_end] = s / Float64(period)
    for i in range(seed_end + 1, m):
        dst[unsafe_offset=i] = k * x[unsafe_offset=i] + (1.0 - k) * dst[
            unsafe_offset=i - 1
        ]


@export
def tamojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def tamojo_ema(
    x: F64Ptr,
    n: Int64,
    length: Int64,
    adjust: Int32,
    sma: Int32,
    dst: F64Ptr,
) abi("C") -> Int32:
    """EMA of x[0..n) into dst[0..n). Returns 0, or 2 on invalid arguments."""
    if n < 0 or length < 1:
        return 2
    if n == 0:
        return 0
    var com = (Float64(length) - 1.0) / 2.0
    if sma != 0:
        _sma_seeded_ewm(x, Int(n), Int(length), com, adjust != 0, dst)
    else:
        _ewm_mean(x, Int(n), com, adjust != 0, dst)
    return 0


@export
def tamojo_rma(
    x: F64Ptr,
    n: Int64,
    length: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """Wilder's moving average (alpha = 1/length, SMA-seeded) into dst[0..n)."""
    if n < 0 or length < 1:
        return 2
    if n == 0:
        return 0
    # Round-trip through center-of-mass exactly like the reference stack:
    # ewm(alpha=a) stores com = (1-a)/a and recomputes alpha = 1/(1+com).
    var alpha = 1.0 / Float64(length)
    var com = (1.0 - alpha) / alpha
    _sma_seeded_ewm(x, Int(n), Int(length), com, False, dst)
    return 0


@export
def tamojo_true_range(
    h: F64Ptr,
    l: F64Ptr,
    c: F64Ptr,
    n: Int64,
    drift: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """True range of (h, l, c) into dst[0..n).

    tr[i] = max(|h-l|, |h-c[i-drift]|, |c[i-drift]-l|), NaN-skipping per bar;
    an exact zero high-low range is replaced by machine epsilon; the first
    (first all-valid row) + drift bars are NaN.
    """
    if n < 0 or drift < 1:
        return 2
    var dn = Int(n)
    var dd = Int(drift)
    for i in range(dn):
        var hi = h[unsafe_offset=i]
        var lo = l[unsafe_offset=i]
        var hl = hi - lo
        if hl == 0.0:
            hl = EPS
        var m = _abs(hl)
        if i >= dd:
            var pc = c[unsafe_offset=i - dd]
            var b = _abs(hi - pc)
            var cc = _abs(pc - lo)
            # NaN-skipping max (order-independent for float64).
            if b == b and (m != m or b > m):
                m = b
            if cc == cc and (m != m or cc > m):
                m = cc
        dst[unsafe_offset=i] = m
    # Blank the leading bars: TR needs a previous close.
    var start = dn
    for i in range(dn):
        var hi = h[unsafe_offset=i]
        var lo = l[unsafe_offset=i]
        var cl = c[unsafe_offset=i]
        if hi == hi and lo == lo and cl == cl:
            start = i
            break
    var blank = start + dd
    if blank > dn:
        blank = dn
    for i in range(blank):
        dst[unsafe_offset=i] = NAN
    return 0


@export
def tamojo_macd(
    x: F64Ptr,
    n: Int64,
    fast: Int64,
    slow: Int64,
    signal: Int64,
    out_macd: F64Ptr,
    out_signal: F64Ptr,
    out_hist: F64Ptr,
) abi("C") -> Int32:
    """MACD (fast, slow, signal) of x[0..n) into three dst[0..n) buffers.

    Leading non-finite rows (NaN or +/-inf) are skipped and padded back with
    NaN. The fast/slow EMAs are plain-SMA-seeded at fast-1 / slow-1; the
    signal EMA is seeded at (slow-1) + (signal-1) from the MACD line.
    """
    if n < 0 or fast < 1 or slow < 1 or signal < 1:
        return 2
    var dn = Int(n)
    if dn == 0:
        return 0
    var f = Int(fast)
    var s = Int(slow)
    var g = Int(signal)

    # Skip the leading non-finite run (np.isfinite semantics: NaN and inf).
    var lead = 0
    while lead < dn and not isfinite(x[unsafe_offset=lead]):
        lead += 1
    for i in range(lead):
        out_macd[unsafe_offset=i] = NAN
        out_signal[unsafe_offset=i] = NAN
        out_hist[unsafe_offset=i] = NAN
    var m = dn - lead
    if m == 0:
        return 0

    # Scratch buffers over the trimmed series x[lead .. dn).
    var xs = unsafe_alloc[Float64](m)
    var fe = unsafe_alloc[Float64](m)
    var se = unsafe_alloc[Float64](m)
    for i in range(m):
        xs[unsafe_offset=i] = x[unsafe_offset=i + lead]
    _ema_aligned(xs, m, f, f - 1, fe)
    _ema_aligned(xs, m, s, s - 1, se)
    # MACD line; xs is reused for it (the close copy is no longer needed).
    var mc = xs
    for i in range(m):
        mc[unsafe_offset=i] = fe[unsafe_offset=i] - se[unsafe_offset=i]
        out_macd[unsafe_offset=i + lead] = mc[unsafe_offset=i]
    # Signal line from the MACD line, seeded signal-1 bars after slow-1.
    _ema_aligned(mc, m, g, (s - 1) + (g - 1), se)
    for i in range(m):
        out_signal[unsafe_offset=i + lead] = se[unsafe_offset=i]
        out_hist[unsafe_offset=i + lead] = mc[unsafe_offset=i] - se[
            unsafe_offset=i
        ]
    xs.unsafe_free()
    fe.unsafe_free()
    se.unsafe_free()
    return 0
