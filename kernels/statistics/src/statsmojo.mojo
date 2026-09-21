"""Clean-room exact statistics kernels matching CPython 3.12 stdlib semantics.

Written fresh from the documented behaviour of the CPython 3.12 `statistics`
and `math` modules (the test oracle). No third-party Mojo code is used or
adapted; no CPython code is compiled — every kernel below is an independent
implementation whose outputs are verified bit-for-bit against the oracle by
the differential test suite.

Three kernel families:

1. Exact sums (superaccumulator). The stdlib computes mean/variance/stdev by
   expanding every value to an exact rational and summing exactly. For binary
   floats every such rational has a power-of-two denominator, so the exact
   sum is equivalently a wide fixed-point (Kulisch-style) accumulator: each
   value m * 2**q is added at bit offset q into a ~2300-bit two's-complement-
   free accumulator pair (positive and negative parts kept apart, so no sign
   extension is ever needed). Squares m**2 * 2**(2q) go into a second, wider
   accumulator. Both are exact for every finite float64 input and are summed
   order-independently, so the caller may split the work into independent
   lanes. The wrapper reads the words back as Python big ints and finishes
   the (single, O(1)) rational division exactly like the stdlib does.
   Integer columns (CPython compact ints) use the same accumulators.
   A "products" mode replicates the stdlib's xbar-given variance path, where
   each term (x - c) * (x - c) is first rounded to float64 and only then
   summed exactly.

2. fsum. A step-for-step reimplementation of the documented CPython fsum
   algorithm (non-overlapping partials with two-sum compression, immediate
   OverflowError on intermediate overflow, -inf + inf detection, and the
   final correctly-rounded reduction with the half-even fix-up), so results
   — including its order-dependent overflow quirks — match bit-for-bit.

3. Radix sort. LSD radix sort (6 passes x 11 bits) on the standard
   order-preserving key transform of float64/int64, used to accelerate
   median/quantiles. NaN and mixed-signed-zero inputs are detected by the
   caller or the list entry points and routed to the reference path (Python
   sort order is only value-defined there), so the kernel never sees them.

CPython list unboxing (optional fast path, guarded by a load-time self-test
in the Python wrapper): the *_list_* entry points take a CPython list's
ob_item array and read PyFloatObject.ob_fval / compact PyLongObject digits
directly. Every element's ob_type is checked against the exact type address;
any mismatch, non-compact int, or out-of-range value aborts the column with
a status code and the wrapper recomputes that column on a safe path. These
entry points MUST be called with the GIL held (the wrapper uses PyDLL).

Exported C ABI (v1):

    int32_t statsmojo_abi_version(void)
    int32_t statsmojo_ssum_f64(const double* x, const int64_t* offsets,
                               int64_t n_cols, int32_t mode, double c,
                               uint64_t* oacc, int32_t* status)
    int32_t statsmojo_ssum_i64(const int64_t* x, const int64_t* offsets,
                               int64_t n_cols, int32_t mode, uint64_t* oacc)
    int32_t statsmojo_fsum_f64(const double* x, const int64_t* offsets,
                               int64_t n_cols, double* oacc, int32_t* status)
    int32_t statsmojo_sort_f64(double* x, const int64_t* offsets,
                               int64_t n_cols, int32_t* status)
    int32_t statsmojo_sort_i64(int64_t* x, const int64_t* offsets,
                               int64_t n_cols)
    int32_t statsmojo_ssum_list_f64(uint64_t list_addr, uint64_t float_type,
                                    uint64_t int_type, int32_t mode, double c,
                                    uint64_t* out, int32_t* status)
    int32_t statsmojo_ssum_list_i64(uint64_t list_addr, uint64_t int_type,
                                    int32_t mode, uint64_t* out, int32_t* status)
    int32_t statsmojo_fsum_list_f64(uint64_t list_addr, uint64_t float_type,
                                    uint64_t int_type, double* out, int32_t* status)
    int32_t statsmojo_sort_list_f64(uint64_t list_addr, int64_t cap,
                                    uint64_t float_type, double* out,
                                    int32_t* status)
    int32_t statsmojo_sort_list_i64(uint64_t list_addr, int64_t cap,
                                    uint64_t int_type, int64_t* out,
                                    int32_t* status)

status codes (per column): 0 ok; 1 intermediate overflow in fsum;
2 -inf + inf in fsum; 3 element type not handled by the fast path;
4 non-finite value where the exact path requires finite; 5 both +0.0 and
-0.0 present (sort order of signed zeros is identity-defined upstream).

mode for ssum: 0 = exact sum only; 1 = exact sum and exact sum of squares;
2 = exact sum of the float64-rounded products (x - c) * (x - c).

`out` for ssum holds WORDS_PER_COL (560) uint64 words per column:
LANES (4) consecutive lanes, each lane = 36 words positive-sum accumulator,
36 words negative-sum accumulator, 68 words square accumulator. With
OFFSET_X = 1074 the exact sum is (pos - neg) * 2**-1074; with
OFFSET_SQ = 2148 the exact sum of squares is sq * 2**-2148.
"""

from std.memory import Pointer, bitcast
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns every buffer).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]

# Per-column status codes (mirrored in python/statistics_mojo/_native.py).
comptime ST_OK: Int32 = 0
comptime ST_OVERFLOW: Int32 = 1
comptime ST_INF_MIX: Int32 = 2
comptime ST_TYPE: Int32 = 3
comptime ST_NONFINITE: Int32 = 4
comptime ST_SIGNED_ZERO: Int32 = 5

# ssum modes.
comptime MODE_SUM: Int32 = 0
comptime MODE_SUM_SQ: Int32 = 1
comptime MODE_PRODUCTS: Int32 = 2

# Superaccumulator geometry (see module docstring).
comptime OFFSET_X: Int = 1074
comptime WORDS_X: Int = 36
comptime OFFSET_SQ: Int = 2148
comptime WORDS_SQ: Int = 68
comptime LANES: Int = 4
comptime LANE_WORDS: Int = 2 * WORDS_X + WORDS_SQ  # 140
comptime WORDS_PER_COL: Int = LANES * LANE_WORDS  # 560

comptime U64_SIGN: UInt64 = 0x8000000000000000
comptime EXP_MASK: UInt64 = 0x7FF0000000000000
comptime MANT_MASK: UInt64 = 0x000FFFFFFFFFFFFF
comptime ABS_MASK: UInt64 = 0x7FFFFFFFFFFFFFFF


def _fabs(x: Float64) -> Float64:
    return bitcast[DType.float64](bitcast[DType.uint64](x) & ABS_MASK)


def _is_finite(x: Float64) -> Bool:
    return (bitcast[DType.uint64](x) & EXP_MASK) != EXP_MASK


def _is_inf(x: Float64) -> Bool:
    var b = bitcast[DType.uint64](x)
    return (b & EXP_MASK) == EXP_MASK and (b & MANT_MASK) == 0


def _is_nan_bits(b: UInt64) -> Bool:
    return (b & ABS_MASK) > EXP_MASK


# ---------------------------------------------------------------------------
# Superaccumulator primitives
# ---------------------------------------------------------------------------


def _acc_add(acc: U64Ptr, w: Int, v: UInt64):
    """acc[w] += v with unsigned carry ripple (headroom is guaranteed)."""
    var old = acc[unsafe_offset=w]
    var new = old + v
    acc[unsafe_offset=w] = new
    if new >= old:
        return
    var i = w + 1
    while True:
        var o = acc[unsafe_offset=i]
        var n2 = o + 1
        acc[unsafe_offset=i] = n2
        if n2 != 0:
            return
        i += 1


def _acc_add_shifted(acc: U64Ptr, m: UInt64, bit: Int):
    """Add m * 2**bit into the accumulator (m fits in one word)."""
    if m == 0:
        return
    var w = bit >> 6
    var s = bit & 63
    if s == 0:
        _acc_add(acc, w, m)
    else:
        _acc_add(acc, w, m << UInt64(s))
        _acc_add(acc, w + 1, m >> UInt64(64 - s))


def _acc_add128_shifted(acc: U64Ptr, hi: UInt64, lo: UInt64, bit: Int):
    """Add (hi * 2**64 + lo) * 2**bit into the accumulator."""
    var w = bit >> 6
    var s = bit & 63
    if s == 0:
        if lo != 0:
            _acc_add(acc, w, lo)
        if hi != 0:
            _acc_add(acc, w + 1, hi)
    else:
        _acc_add(acc, w, lo << UInt64(s))
        _acc_add(acc, w + 1, (lo >> UInt64(64 - s)) | (hi << UInt64(s)))
        var top = hi >> UInt64(64 - s)
        if top != 0:
            _acc_add(acc, w + 2, top)


def _mul_sq(m: UInt64) -> Tuple[UInt64, UInt64]:
    """(hi, lo) of m * m for m < 2**63, via 32/63-bit limb split."""
    var a = m >> 32
    var b = m & 0xFFFFFFFF
    var aa = a * a  # < 2**62
    var bb = b * b  # < 2**64
    var ab = a * b  # a < 2**31, b < 2**32 -> ab < 2**63
    # m**2 = aa * 2**64 + ab * 2**33 + bb
    var lo = bb + (ab << 33)
    var carry: UInt64 = 1 if lo < bb else 0
    var hi = aa + (ab >> 31) + carry
    return (hi, lo)


def _decompose(x: Float64) -> Tuple[UInt64, Int, Bool]:
    """Finite float64 -> (m, q, neg) with x = +/-m * 2**q, m a 53-bit int."""
    var b = bitcast[DType.uint64](x)
    var e = Int((b >> 52) & 0x7FF)
    var mant = b & MANT_MASK
    var q: Int
    if e == 0:
        q = -1074
    else:
        q = e - 1075
        mant |= 0x10000000000000
    return (mant, q, (b >> 63) != 0)


def _zero(oacc: U64Ptr):
    for i in range(WORDS_PER_COL):
        oacc[unsafe_offset=i] = 0


def _accum_f64(v: Float64, mode: Int32, c: Float64, lane: Int, oacc: U64Ptr) -> Int32:
    """Accumulate one float64 into lane `lane`; status code or ST_OK.
    Non-finite v (or a non-finite product) is reported, never accumulated."""
    var base = lane * LANE_WORDS
    if mode == MODE_PRODUCTS:
        var d = v - c
        var p = d * d
        if not _is_finite(p):
            return ST_NONFINITE
        var f = _decompose(p)
        _acc_add_shifted(oacc.unsafe_offset(base), f[0], f[1] + OFFSET_X)
        return ST_OK
    if not _is_finite(v):
        return ST_NONFINITE
    var f = _decompose(v)
    if f[2]:
        _acc_add_shifted(oacc.unsafe_offset(base + WORDS_X), f[0], f[1] + OFFSET_X)
    else:
        _acc_add_shifted(oacc.unsafe_offset(base), f[0], f[1] + OFFSET_X)
    if mode == MODE_SUM_SQ:
        var sq = _mul_sq(f[0])
        _acc_add128_shifted(
            oacc.unsafe_offset(base + 2 * WORDS_X), sq[0], sq[1], 2 * f[1] + OFFSET_SQ
        )
    return ST_OK


def _accum_i64(v: Int64, mode: Int32, lane: Int, oacc: U64Ptr):
    """Accumulate one int64 into lane `lane` (two's-complement magnitude)."""
    var base = lane * LANE_WORDS
    var neg = v < 0
    var mag = bitcast[DType.uint64](v)
    if neg:
        mag = ~mag + 1
    if neg:
        _acc_add_shifted(oacc.unsafe_offset(base + WORDS_X), mag, OFFSET_X)
    else:
        _acc_add_shifted(oacc.unsafe_offset(base), mag, OFFSET_X)
    if mode == MODE_SUM_SQ:
        var sq = _mul_sq(mag)
        _acc_add128_shifted(oacc.unsafe_offset(base + 2 * WORDS_X), sq[0], sq[1], OFFSET_SQ)


# ---------------------------------------------------------------------------
# CPython list unboxing (GIL held by the caller; guarded by load-time probe)
# ---------------------------------------------------------------------------

# PyObject* element access: ob_type at +8; PyFloatObject.ob_fval at +16;
# PyLongObject lv_tag at +16 (sign in low 2 bits: 0 pos, 1 zero, 2 neg;
# digit count in bits 3+), 30-bit little-endian digits at +24.


def _load_u64(addr: UInt64) -> UInt64:
    return U64Ptr(unsafe_from_address=Int(addr))[]


def _load_u32(addr: UInt64) -> UInt32:
    return U32Ptr(unsafe_from_address=Int(addr))[]


def _load_f64(addr: UInt64) -> Float64:
    return F64Ptr(unsafe_from_address=Int(addr))[]


def _unbox_small_int(obj: UInt64) -> Tuple[UInt64, Bool, Bool]:
    """Compact PyLong -> (magnitude, negative, ok). ok=False if > 2 digits."""
    var tag = _load_u64(obj + 16)
    var sign = tag & 3
    if sign == 1:
        return (0, False, True)  # zero
    var ndig = Int(tag >> 3)
    if ndig > 2:
        return (0, False, False)
    var mag = UInt64(_load_u32(obj + 24))
    if ndig == 2:
        mag |= UInt64(_load_u32(obj + 28)) << 30
    return (mag, sign == 2, True)


def _unbox_list_f64(
    items: U64Ptr, i: Int, float_type: UInt64, int_type: UInt64
) -> Tuple[Float64, Int32]:
    """Element i as float64. Ints convert exactly (|v| <= 2**53 required,
    matching the rational-exactness of the stdlib's mixed int/float sums);
    anything else yields ST_TYPE so the caller reroutes the column."""
    var obj = items[unsafe_offset=i]
    var tpe = _load_u64(obj + 8)
    if tpe == float_type:
        return (_load_f64(obj + 16), ST_OK)
    if tpe == int_type:
        var r = _unbox_small_int(obj)
        if not r[2] or r[0] > (1 << 53):
            return (0.0, ST_TYPE)
        var v = Float64(Int64(r[0]))
        if r[1]:
            v = -v
        return (v, ST_OK)
    return (0.0, ST_TYPE)


def _unbox_list_fsum(
    items: U64Ptr, i: Int, float_type: UInt64, int_type: UInt64
) -> Tuple[Float64, Int32]:
    """Element i as float64 for fsum: compact ints convert like
    PyFloat_AsDouble (round-to-nearest, any compact magnitude)."""
    var obj = items[unsafe_offset=i]
    var tpe = _load_u64(obj + 8)
    if tpe == float_type:
        return (_load_f64(obj + 16), ST_OK)
    if tpe == int_type:
        var r = _unbox_small_int(obj)
        if not r[2]:
            return (0.0, ST_TYPE)
        var v = Float64(Int64(r[0]))
        if r[1]:
            v = -v
        return (v, ST_OK)
    return (0.0, ST_TYPE)


# ---------------------------------------------------------------------------
# Exact-sum column kernels
# ---------------------------------------------------------------------------


def _ssum_f64_col(
    x: F64Ptr, n: Int, mode: Int32, c: Float64, oacc: U64Ptr
) -> Int32:
    _zero(oacc)
    for i in range(n):
        var st = _accum_f64(x[unsafe_offset=i], mode, c, i & (LANES - 1), oacc)
        if st != ST_OK:
            return st
    return ST_OK


def _ssum_i64_col(x: I64Ptr, n: Int, mode: Int32, oacc: U64Ptr):
    _zero(oacc)
    for i in range(n):
        _accum_i64(x[unsafe_offset=i], mode, i & (LANES - 1), oacc)


def _ssum_list_f64_col(
    items: U64Ptr,
    n: Int,
    float_type: UInt64,
    int_type: UInt64,
    mode: Int32,
    c: Float64,
    oacc: U64Ptr,
) -> Int32:
    _zero(oacc)
    for i in range(n):
        var u = _unbox_list_f64(items, i, float_type, int_type)
        if u[1] != ST_OK:
            return u[1]
        var st = _accum_f64(u[0], mode, c, i & (LANES - 1), oacc)
        if st != ST_OK:
            return st
    return ST_OK


def _ssum_list_i64_col(
    items: U64Ptr, n: Int, int_type: UInt64, mode: Int32, oacc: U64Ptr
) -> Int32:
    _zero(oacc)
    for i in range(n):
        var obj = items[unsafe_offset=i]
        if _load_u64(obj + 8) != int_type:
            return ST_TYPE
        var r = _unbox_small_int(obj)
        if not r[2]:
            return ST_TYPE
        var v = Int64(r[0])
        if r[1]:
            v = -v
        _accum_i64(v, mode, i & (LANES - 1), oacc)
    return ST_OK


# ---------------------------------------------------------------------------
# fsum (documented CPython algorithm, replicated step for step)
# ---------------------------------------------------------------------------


def _fsum_core(x: F64Ptr, n: Int, p: F64Ptr) -> Tuple[Float64, Int32]:
    var np = 0
    var special_sum = 0.0
    var inf_sum = 0.0
    for k in range(n):
        var xv = x[unsafe_offset=k]
        var xsave = xv
        var i = 0
        var j = 0
        while j < np:
            var y = p[unsafe_offset=j]
            if _fabs(xv) < _fabs(y):
                var t = xv
                xv = y
                y = t
            var hi = xv + y
            var lo = y - (hi - xv)
            if lo != 0.0:
                p[unsafe_offset=i] = lo
                i += 1
            xv = hi
            j += 1
        np = i
        if xv != 0.0:
            if not _is_finite(xv):
                if _is_finite(xsave):
                    return (0.0, ST_OVERFLOW)
                if _is_inf(xsave):
                    inf_sum += xsave
                special_sum += xsave
                np = 0
            else:
                p[unsafe_offset=np] = xv
                np += 1
    if special_sum != 0.0:
        if inf_sum != inf_sum:
            return (0.0, ST_INF_MIX)
        return (special_sum, ST_OK)
    var result = 0.0
    if np > 0:
        np -= 1
        result = p[unsafe_offset=np]
        var lo = 0.0
        while np > 0:
            var xx = result
            np -= 1
            var y = p[unsafe_offset=np]
            result = xx + y
            var yr = result - xx
            lo = y - yr
            if lo != 0.0:
                break
        if np > 0 and (
            (lo < 0.0 and p[unsafe_offset=np - 1] < 0.0)
            or (lo > 0.0 and p[unsafe_offset=np - 1] > 0.0)
        ):
            var y = lo * 2.0
            var xx = result + y
            var yr = xx - result
            if y == yr:
                result = xx
    return (result, ST_OK)


# ---------------------------------------------------------------------------
# Radix sort (LSD, 6 passes x 11 bits, stable)
# ---------------------------------------------------------------------------


def _key_f64(b: UInt64) -> UInt64:
    if (b >> 63) != 0:
        return ~b
    return b | U64_SIGN


def _sort_u64(
    src0: U64Ptr, n: Int, tmp: U64Ptr, counts: U32Ptr, float_keys: Bool
):
    var src = src0
    var dst = tmp
    for p in range(6):
        var shift = UInt64(p * 11)
        for i in range(2048):
            counts[unsafe_offset=i] = 0
        for i in range(n):
            var b = src[unsafe_offset=i]
            var k = _key_f64(b) if float_keys else b ^ U64_SIGN
            counts[unsafe_offset=Int((k >> shift) & 2047)] += 1
        var total: UInt32 = 0
        for bkt in range(2048):
            var cnt = counts[unsafe_offset=bkt]
            counts[unsafe_offset=bkt] = total
            total += cnt
        for i in range(n):
            var b = src[unsafe_offset=i]
            var k = _key_f64(b) if float_keys else b ^ U64_SIGN
            var bkt = Int((k >> shift) & 2047)
            dst[unsafe_offset=Int(counts[unsafe_offset=bkt])] = b
            counts[unsafe_offset=bkt] += 1
        var t = src
        src = dst
        dst = t
    # 6 (even) passes return the data into src0.


# ---------------------------------------------------------------------------
# Exported C ABI
# ---------------------------------------------------------------------------


@export
def statsmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def statsmojo_words_per_col() abi("C") -> Int32:
    return Int32(WORDS_PER_COL)


@export
def statsmojo_ssum_f64(
    x: F64Ptr,
    offsets: I64Ptr,
    n_cols: Int64,
    mode: Int32,
    c: Float64,
    oacc: U64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """Exact per-column sums; out has WORDS_PER_COL words per column.
    mode 2 uses `c` (the float64-rounded-products path)."""
    if n_cols < 0 or (mode < 0 or mode > 2):
        return 2
    for col in range(Int(n_cols)):
        var start = offsets[unsafe_offset=col]
        var stop = offsets[unsafe_offset=col + 1]
        var st = _ssum_f64_col(
            x.unsafe_offset(start), Int(stop - start), mode, c, oacc.unsafe_offset(col * WORDS_PER_COL)
        )
        status[unsafe_offset=col] = st
    return 0


@export
def statsmojo_ssum_i64(
    x: I64Ptr,
    offsets: I64Ptr,
    n_cols: Int64,
    mode: Int32,
    oacc: U64Ptr,
) abi("C") -> Int32:
    """Exact per-column sums for int64 columns (modes 0 and 1 only)."""
    if n_cols < 0 or mode < 0 or mode > 1:
        return 2
    for col in range(Int(n_cols)):
        var start = offsets[unsafe_offset=col]
        var stop = offsets[unsafe_offset=col + 1]
        _ssum_i64_col(
            x.unsafe_offset(start), Int(stop - start), mode, oacc.unsafe_offset(col * WORDS_PER_COL)
        )
    return 0


@export
def statsmojo_fsum_f64(
    x: F64Ptr,
    offsets: I64Ptr,
    n_cols: Int64,
    oacc: F64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """CPython-semantics fsum per column (bit-exact, quirks included)."""
    if n_cols < 0:
        return 2
    var max_len: Int = 0
    for col in range(Int(n_cols)):
        var l = Int(offsets[unsafe_offset=col + 1] - offsets[unsafe_offset=col])
        if l > max_len:
            max_len = l
    var p = unsafe_alloc[Float64](max_len if max_len > 0 else 1)
    for col in range(Int(n_cols)):
        var start = offsets[unsafe_offset=col]
        var stop = offsets[unsafe_offset=col + 1]
        var r = _fsum_core(x.unsafe_offset(start), Int(stop - start), p)
        oacc[unsafe_offset=col] = r[0]
        status[unsafe_offset=col] = r[1]
    p.unsafe_free()
    return 0


@export
def statsmojo_sort_f64(
    x: F64Ptr, offsets: I64Ptr, n_cols: Int64, status: I32Ptr
) abi("C") -> Int32:
    """Ascending sort of each column segment, in place. Columns containing
    NaN, or both +0.0 and -0.0, are reported via status (ST_NONFINITE /
    ST_SIGNED_ZERO) and left unsorted; the wrapper reroutes those columns to
    the reference path (their order is identity-defined upstream)."""
    if n_cols < 0:
        return 2
    var max_len: Int = 0
    for col in range(Int(n_cols)):
        var l = Int(offsets[unsafe_offset=col + 1] - offsets[unsafe_offset=col])
        if l > max_len:
            max_len = l
    if max_len == 0:
        return 0
    var tmp = unsafe_alloc[UInt64](max_len)
    var counts = unsafe_alloc[UInt32](2048)
    var xu = x.unsafe_bitcast[UInt64]()
    for col in range(Int(n_cols)):
        var start = offsets[unsafe_offset=col]
        var stop = offsets[unsafe_offset=col + 1]
        var dn = Int(stop - start)
        var st: Int32 = ST_OK
        var seen_pos_zero = False
        var seen_neg_zero = False
        for i in range(dn):
            var b = xu[unsafe_offset=i + Int(start)]
            if _is_nan_bits(b):
                st = ST_NONFINITE
                break
            if b == 0:
                seen_pos_zero = True
            elif b == U64_SIGN:
                seen_neg_zero = True
        if st == ST_OK and seen_pos_zero and seen_neg_zero:
            st = ST_SIGNED_ZERO
        if st == ST_OK:
            _sort_u64(xu.unsafe_offset(start), dn, tmp, counts, True)
        status[unsafe_offset=col] = st
    tmp.unsafe_free()
    counts.unsafe_free()
    return 0


@export
def statsmojo_sort_i64(
    x: I64Ptr, offsets: I64Ptr, n_cols: Int64
) abi("C") -> Int32:
    """Ascending sort of each int64 column segment, in place."""
    if n_cols < 0:
        return 2
    var max_len: Int = 0
    for col in range(Int(n_cols)):
        var l = Int(offsets[unsafe_offset=col + 1] - offsets[unsafe_offset=col])
        if l > max_len:
            max_len = l
    if max_len == 0:
        return 0
    var tmp = unsafe_alloc[UInt64](max_len)
    var counts = unsafe_alloc[UInt32](2048)
    var xu = x.unsafe_bitcast[UInt64]()
    for col in range(Int(n_cols)):
        var start = offsets[unsafe_offset=col]
        var stop = offsets[unsafe_offset=col + 1]
        _sort_u64(xu.unsafe_offset(start), Int(stop - start), tmp, counts, False)
    tmp.unsafe_free()
    counts.unsafe_free()
    return 0


# --- List-unboxing entry points (MUST be called with the GIL held) ---
#
# These take the ADDRESS OF A CPython list object (not its ob_item buffer)
# and read ob_size (+16) and ob_item (+24) themselves. Because the caller
# (ctypes PyDLL) holds the GIL for the whole call, the list cannot be mutated
# or reallocated while the kernel reads it; reading the pointers here, rather
# than in Python bytecode before the call, closes the resize race. The sort
# entries additionally take the caller's output-buffer capacity and bail with
# ST_TYPE if the live ob_size exceeds it (the wrapper then reroutes).


def _list_items(list_addr: UInt64) -> Tuple[U64Ptr, Int]:
    var ob_size = _load_u64(list_addr + 16)
    var items = U64Ptr(unsafe_from_address=Int(_load_u64(list_addr + 24)))
    return (items, Int(ob_size))


@export
def statsmojo_ssum_list_f64(
    list_addr: UInt64,
    float_type: UInt64,
    int_type: UInt64,
    mode: Int32,
    c: Float64,
    oacc: U64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """Exact sums for one CPython list column (floats, plus compact ints
    whose magnitude is <= 2**53). status is a single per-column code."""
    if mode < 0 or mode > 2:
        return 2
    var li = _list_items(list_addr)
    status[unsafe_offset=0] = _ssum_list_f64_col(
        li[0], li[1], float_type, int_type, mode, c, oacc
    )
    return 0


@export
def statsmojo_ssum_list_i64(
    list_addr: UInt64,
    int_type: UInt64,
    mode: Int32,
    oacc: U64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """Exact sums for one CPython list column of compact ints."""
    if mode < 0 or mode > 1:
        return 2
    var li = _list_items(list_addr)
    status[unsafe_offset=0] = _ssum_list_i64_col(li[0], li[1], int_type, mode, oacc)
    return 0


@export
def statsmojo_fsum_list_f64(
    list_addr: UInt64,
    float_type: UInt64,
    int_type: UInt64,
    oacc: F64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """CPython-semantics fsum of one list column (floats and compact ints)."""
    var li = _list_items(list_addr)
    var dn = li[1]
    var p = unsafe_alloc[Float64](dn if dn > 0 else 1)
    var buf = unsafe_alloc[Float64](dn if dn > 0 else 1)
    var st: Int32 = ST_OK
    for i in range(dn):
        var u = _unbox_list_fsum(li[0], i, float_type, int_type)
        if u[1] != ST_OK:
            st = u[1]
            break
        buf[unsafe_offset=i] = u[0]
    if st == ST_OK:
        var r = _fsum_core(buf, dn, p)
        oacc[unsafe_offset=0] = r[0]
        st = r[1]
    status[unsafe_offset=0] = st
    p.unsafe_free()
    buf.unsafe_free()
    return 0


@export
def statsmojo_sort_list_f64(
    list_addr: UInt64,
    cap: Int64,
    float_type: UInt64,
    oacc: F64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """Unbox + ascending sort of one all-float list column into oacc[0..n).
    Rejects non-float elements (ints included), NaN, and mixed +0.0/-0.0
    (status codes 3/4/5) so the wrapper can route those columns to the
    reference path."""
    if cap < 0:
        return 2
    var li = _list_items(list_addr)
    var dn = li[1]
    if Int64(dn) > cap:
        status[unsafe_offset=0] = ST_TYPE
        return 0
    var out_u = oacc.unsafe_bitcast[UInt64]()
    var seen_pos_zero = False
    var seen_neg_zero = False
    for i in range(dn):
        var obj = li[0][unsafe_offset=i]
        if _load_u64(obj + 8) != float_type:
            status[unsafe_offset=0] = ST_TYPE
            return 0
        var b = _load_u64(obj + 16)
        if _is_nan_bits(b):
            status[unsafe_offset=0] = ST_NONFINITE
            return 0
        if b == 0:
            seen_pos_zero = True
        elif b == U64_SIGN:
            seen_neg_zero = True
        out_u[unsafe_offset=i] = b
    if seen_pos_zero and seen_neg_zero:
        status[unsafe_offset=0] = ST_SIGNED_ZERO
        return 0
    if dn > 1:
        var tmp = unsafe_alloc[UInt64](dn)
        var counts = unsafe_alloc[UInt32](2048)
        _sort_u64(out_u, dn, tmp, counts, True)
        tmp.unsafe_free()
        counts.unsafe_free()
    status[unsafe_offset=0] = ST_OK
    return 0


@export
def statsmojo_sort_list_i64(
    list_addr: UInt64,
    cap: Int64,
    int_type: UInt64,
    oacc: I64Ptr,
    status: I32Ptr,
) abi("C") -> Int32:
    """Unbox + ascending sort of one compact-int list column into oacc[0..n)."""
    if cap < 0:
        return 2
    var li = _list_items(list_addr)
    var dn = li[1]
    if Int64(dn) > cap:
        status[unsafe_offset=0] = ST_TYPE
        return 0
    var out_u = oacc.unsafe_bitcast[UInt64]()
    for i in range(dn):
        var obj = li[0][unsafe_offset=i]
        if _load_u64(obj + 8) != int_type:
            status[unsafe_offset=0] = ST_TYPE
            return 0
        var r = _unbox_small_int(obj)
        if not r[2]:
            status[unsafe_offset=0] = ST_TYPE
            return 0
        var v = Int64(r[0])
        if r[1]:
            v = -v
        out_u[unsafe_offset=i] = bitcast[DType.uint64](v)
    if dn > 1:
        var tmp = unsafe_alloc[UInt64](dn)
        var counts = unsafe_alloc[UInt32](2048)
        _sort_u64(out_u, dn, tmp, counts, False)
        tmp.unsafe_free()
        counts.unsafe_free()
    status[unsafe_offset=0] = ST_OK
    return 0
