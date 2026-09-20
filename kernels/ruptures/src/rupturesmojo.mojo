"""Clean-room change-point detection kernel (Dynp / Pelt / Binseg for 1-D
signals, l2 and l1 cost models).

Written fresh from the textbook algorithms (Killick et al., PELT, JASA 2012;
Truong et al., "Selective review of offline change point detection methods",
Signal Processing 167 (2020); Sen & Srivastava, binary segmentation 1975).
No third-party Mojo code is used or adapted.

The kernel reproduces the *observable* behaviour of the published `ruptures`
package (PyPI ruptures 1.1.10) breakpoint-for-breakpoint on finite 1-D
float64 signals: identical candidate enumeration, identical float64 operation
order, identical tie-breaking, so breakpoint sets are integer equal.

Exactness contract:

- l2 segment cost of x[a:b] (length m) is var * m evaluated the way NumPy
  evaluates `x[a:b].var() * m`: mean = pairwise_sum(x)/m, then
  var = pairwise_sum((x - mean)^2)/m, cost = var * m, where pairwise_sum is
  NumPy's pairwise summation order for contiguous float64 (8-accumulator
  blocks up to 128 elements, binary recursion above). Bit-exact by
  construction; verified element-for-element against NumPy.
- l1 segment cost is sum |x - median(x)| with NumPy median semantics
  (middle value for odd m; (a + b) / 2 of the two middle values for even m),
  summed with the same pairwise summation.
- Dynp: G[k][t] = min over admissible last breakpoints b of
  G[k-1][b] + cost(b, t); totals are left-to-right sums; ties keep the
  smallest b (strict < scan in ascending order).
- Pelt: F[t] optimal penalized total for x[0:t]; candidates scanned in
  ascending t; ties keep the smallest t; prune t when
  total_t > F[bkp] + pen.
- Binseg: greedy max-gain split; within a segment gain ties keep the largest
  candidate (tuple max), across segments ties keep the leftmost (first max).

Exported C ABI (batch-shaped: whole signal in, whole breakpoint list out):

    int32_t rupturesmojo_abi_version(void)
    int32_t rupturesmojo_detect(signal, n, method, model, min_size, jump,
                                n_bkps, pen, out_bkps, out_cap)

`rupturesmojo_detect` returns the number of breakpoints written to `out_bkps`
(>= 0) or a negative status:
    -1  impossible segmentation configuration (BadSegmentationParameters)
    -2  out_cap too small
    -3  invalid arguments
    -4  empty candidate set (matches Python's ValueError from min())
    -5  problem too large for the native path (caller falls back)
    -6  scratch allocation failed (caller falls back)
"""

from std.collections import Dict
from std.math import abs, inf, nan
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Detection methods.
comptime METHOD_DYNP: Int32 = 0
comptime METHOD_PELT: Int32 = 1
comptime METHOD_BINSEG: Int32 = 2

# Cost models.
comptime MODEL_L2: Int32 = 0
comptime MODEL_L1: Int32 = 1

# Status codes (negative), mirrored in the Python wrapper.
comptime STATUS_BAD_SEGMENTATION: Int32 = -1
comptime STATUS_OUTPUT_CAPACITY: Int32 = -2
comptime STATUS_INVALID_ARGUMENT: Int32 = -3
comptime STATUS_EMPTY_CANDIDATES: Int32 = -4
comptime STATUS_TOO_LARGE: Int32 = -5
comptime STATUS_ALLOC_FAILED: Int32 = -6

# Dynp cost-matrix size caps. The matrix holds G*(G-1)/2 float64 cells for a
# grid of G = n//jump + 2 candidate positions; the cap bounds it at 144 MB.
comptime DYNP_MAX_GRID: Int = 6000

# IEEE-754 positive infinity / NaN, matching Python float semantics.
comptime INF: Float64 = inf[DType.float64]()
comptime NAN: Float64 = nan[DType.float64]()

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


# ---------------------------------------------------------------------------
# NumPy pairwise summation order for contiguous float64, bit-exact.
# ---------------------------------------------------------------------------


def pw_sum(x: F64Ptr, start: Int, n: Int) -> Float64:
    """Sum x[start:start+n] in NumPy's pairwise reduction order."""
    if n < 8:
        var res = 0.0
        for i in range(start, start + n):
            res += x[unsafe_offset=i]
        return res
    elif n <= 128:
        # 8 scalar accumulators r[0..7] held as 4 SIMD pairs; lane j of the
        # flattened tuple is accumulator r[j], so the block loop and the final
        # combine tree reproduce NumPy's order exactly.
        var r0 = x.unsafe_load[width=2](start)
        var r1 = x.unsafe_load[width=2](start + 2)
        var r2 = x.unsafe_load[width=2](start + 4)
        var r3 = x.unsafe_load[width=2](start + 6)
        var i = start + 8
        var stop = start + n - (n % 8)
        while i < stop:
            r0 += x.unsafe_load[width=2](i)
            r1 += x.unsafe_load[width=2](i + 2)
            r2 += x.unsafe_load[width=2](i + 4)
            r3 += x.unsafe_load[width=2](i + 6)
            i += 8
        var res = ((r0[0] + r0[1]) + (r1[0] + r1[1])) + (
            (r2[0] + r2[1]) + (r3[0] + r3[1])
        )
        while i < start + n:
            res += x[unsafe_offset=i]
            i += 1
        return res
    else:
        var n2 = n // 2
        n2 -= n2 % 8
        return pw_sum(x, start, n2) + pw_sum(x, start + n2, n - n2)


# ---------------------------------------------------------------------------
# Segment cost functions (scratch buffers are caller-provided, sized >= n).
# ---------------------------------------------------------------------------


def cost_l2(x: F64Ptr, a: Int, b: Int, scratch: F64Ptr) -> Float64:
    """Compute var(x[a:b]) * (b - a) with NumPy's two-pass float64 semantics."""
    var m = b - a
    var mf = Float64(m)
    var mean = pw_sum(x, a, m) / mf
    for i in range(m):
        var d = x[unsafe_offset=a + i] - mean
        scratch[unsafe_offset=i] = d * d
    var variance = pw_sum(scratch, 0, m) / mf
    return variance * mf


def _partition(p: F64Ptr, lo: Int, hi: Int) -> Int:
    """Median-of-three + Lomuto partition of p[lo:hi] (hi exclusive, hi-lo > 2).

    Returns the pivot's final index: p[lo:i] < p[i] <= p[i+1:hi].
    """
    var mid = lo + (hi - lo) // 2
    # Order p[lo] <= p[mid] <= p[hi-1]; the median is the pivot.
    if p[unsafe_offset=mid] < p[unsafe_offset=lo]:
        var t0 = p[unsafe_offset=lo]
        p[unsafe_offset=lo] = p[unsafe_offset=mid]
        p[unsafe_offset=mid] = t0
    if p[unsafe_offset=hi - 1] < p[unsafe_offset=lo]:
        var t1 = p[unsafe_offset=lo]
        p[unsafe_offset=lo] = p[unsafe_offset=hi - 1]
        p[unsafe_offset=hi - 1] = t1
    if p[unsafe_offset=hi - 1] < p[unsafe_offset=mid]:
        var t2 = p[unsafe_offset=mid]
        p[unsafe_offset=mid] = p[unsafe_offset=hi - 1]
        p[unsafe_offset=hi - 1] = t2
    var pivot = p[unsafe_offset=mid]
    # Move the pivot out of the scan, then Lomuto-partition.
    var t3 = p[unsafe_offset=mid]
    p[unsafe_offset=mid] = p[unsafe_offset=hi - 2]
    p[unsafe_offset=hi - 2] = t3
    var i = lo
    for k in range(lo, hi - 2):
        if p[unsafe_offset=k] < pivot:
            var t4 = p[unsafe_offset=i]
            p[unsafe_offset=i] = p[unsafe_offset=k]
            p[unsafe_offset=k] = t4
            i += 1
    p[unsafe_offset=hi - 2] = p[unsafe_offset=i]
    p[unsafe_offset=i] = pivot
    return i


def _insertion_sort(p: F64Ptr, lo: Int, hi: Int) -> None:
    for i in range(lo + 1, hi):
        var key = p[unsafe_offset=i]
        var j = i - 1
        while j >= lo and p[unsafe_offset=j] > key:
            p[unsafe_offset=j + 1] = p[unsafe_offset=j]
            j -= 1
        p[unsafe_offset=j + 1] = key


def _select_kth(p: F64Ptr, lo: Int, hi: Int, k: Int) -> None:
    """Quickselect: p[k] becomes the k-th smallest value of p[lo:hi], with
    p[lo:k] <= p[k] <= p[k+1:hi] (NumPy `partition` semantics: the selected
    values are identical to a full sort's). Deterministic, O(m) average."""
    var lo_ = lo
    var hi_ = hi
    while hi_ - lo_ > 16:
        var i = _partition(p, lo_, hi_)
        if k == i:
            return
        elif k < i:
            hi_ = i
        else:
            lo_ = i + 1
    _insertion_sort(p, lo_, hi_)


def cost_l1(
    x: F64Ptr, a: Int, b: Int, sel_buf: F64Ptr, dev_buf: F64Ptr
) -> Float64:
    """Compute sum |x[a:b] - median(x[a:b])| with NumPy median/sum semantics.

    The median is the exact middle value for odd m, (v1 + v2) / 2 of the two
    middle values for even m — the same values np.median produces.
    """
    var m = b - a
    for i in range(m):
        sel_buf[unsafe_offset=i] = x[unsafe_offset=a + i]
    var med: Float64
    if m % 2 == 1:
        _select_kth(sel_buf, 0, m, (m - 1) // 2)
        med = sel_buf[unsafe_offset=(m - 1) // 2]
    else:
        var k1 = m // 2 - 1
        _select_kth(sel_buf, 0, m, k1)
        # The second middle value is the minimum of the upper part.
        var v2 = sel_buf[unsafe_offset=k1 + 1]
        for i in range(k1 + 2, m):
            if sel_buf[unsafe_offset=i] < v2:
                v2 = sel_buf[unsafe_offset=i]
        med = (sel_buf[unsafe_offset=k1] + v2) / 2.0
    for i in range(m):
        dev_buf[unsafe_offset=i] = abs(x[unsafe_offset=a + i] - med)
    return pw_sum(dev_buf, 0, m)


def seg_cost(
    model: Int32, x: F64Ptr, a: Int, b: Int, buf1: F64Ptr, buf2: F64Ptr
) -> Float64:
    if model == MODEL_L1:
        return cost_l1(x, a, b, buf1, buf2)
    return cost_l2(x, a, b, buf1)


# ---------------------------------------------------------------------------
# Shared feasibility check (mirrors ruptures.utils.sanity_check).
# ---------------------------------------------------------------------------


def sanity_check(n_samples: Int, n_bkps: Int, jump: Int, min_size: Int) -> Bool:
    var n_adm_bkps = n_samples // jump
    if n_bkps > n_adm_bkps:
        return False
    var ceil_ms_jump = (min_size + jump - 1) // jump
    if n_bkps * ceil_ms_jump * jump + min_size > n_samples:
        return False
    return True


# ---------------------------------------------------------------------------
# Dynp: exact optimal partition via dynamic programming.
# ---------------------------------------------------------------------------


def dynp_detect(
    model: Int32,
    x: F64Ptr,
    n: Int,
    min_size: Int,
    jump: Int,
    n_bkps: Int,
    out_bkps: I64Ptr,
    out_cap: Int,
) -> Int32:
    if out_cap < n_bkps + 1:
        return STATUS_OUTPUT_CAPACITY
    # Grid of candidate positions: multiples of jump in [0, n], plus n.
    var n_grid = n // jump + 1
    if n % jump != 0:
        n_grid += 1
    if n_grid > DYNP_MAX_GRID:
        return STATUS_TOO_LARGE

    var pos = unsafe_alloc[Int64](n_grid)
    var pos_of = unsafe_alloc[Int64](n + 1)
    for i in range(n + 1):
        pos_of[unsafe_offset=i] = -1
    var gi = 0
    var p = 0
    while p <= n:
        pos[unsafe_offset=gi] = Int64(p)
        pos_of[unsafe_offset=p] = Int64(gi)
        gi += 1
        p += jump
    if pos[unsafe_offset=gi - 1] != Int64(n):
        pos[unsafe_offset=gi] = Int64(n)
        pos_of[unsafe_offset=n] = Int64(gi)
        gi += 1

    # Packed upper-triangular cost matrix: T(i, j) = cost(pos[i], pos[j])
    # for i < j, laid out row-major as i*(2*G - i - 1)/2 + (j - i - 1).
    var n_cells = n_grid * (n_grid - 1) // 2
    var matrix = unsafe_alloc[Float64](n_cells)
    var buf1 = unsafe_alloc[Float64](n)
    var buf2 = unsafe_alloc[Float64](n)

    var row_base = 0
    for i in range(n_grid):
        var pi = Int(pos[unsafe_offset=i])
        for j in range(i + 1, n_grid):
            var pj = Int(pos[unsafe_offset=j])
            var cell: Float64
            if pj - pi >= min_size:
                cell = seg_cost(model, x, pi, pj, buf1, buf2)
            else:
                cell = INF
            matrix[unsafe_offset=row_base + (j - i - 1)] = cell
        row_base += n_grid - i - 1

    # G[k][i]: optimal total cost for x[0:pos[i]] with k breakpoints.
    var width = n_grid
    var G = unsafe_alloc[Float64]((n_bkps + 1) * width)
    var parent = unsafe_alloc[Int64]((n_bkps + 1) * width)

    for i in range(width):
        var t = Int(pos[unsafe_offset=i])
        if t >= min_size:
            # cost(0, t): matrix cell (0, i) when i > 0; i == 0 has t == 0
            G[unsafe_offset=i] = matrix[unsafe_offset=i - 1] if i > 0 else INF
        else:
            G[unsafe_offset=i] = INF
        parent[unsafe_offset=i] = -1

    for k in range(1, n_bkps + 1):
        var row_prev = (k - 1) * width
        var row_cur = k * width
        for i in range(width):
            var t = Int(pos[unsafe_offset=i])
            var best = INF
            var best_b = -1
            # Candidate last breakpoints: nonzero multiples of jump below t,
            # ascending; strict < keeps the smallest on ties.
            var b = jump
            while b < t:
                if t - b >= min_size and sanity_check(b, k - 1, jump, min_size):
                    var j = Int(pos_of[unsafe_offset=b])
                    var cell = matrix[
                        unsafe_offset=j * (2 * n_grid - j - 1) // 2 + (i - j - 1)
                    ]
                    var total = G[unsafe_offset=row_prev + j] + cell
                    if total < best:
                        best = total
                        best_b = b
                b += jump
            G[unsafe_offset=row_cur + i] = best
            parent[unsafe_offset=row_cur + i] = Int64(best_b)

    # Backtrack from (n_bkps, n).
    var status: Int32
    if parent[unsafe_offset=n_bkps * width + (n_grid - 1)] < 0:
        status = STATUS_BAD_SEGMENTATION
    else:
        var chain = List[Int64]()
        var i = n_grid - 1
        var k = n_bkps
        while k > 0:
            var b = parent[unsafe_offset=k * width + i]
            chain.append(b)
            i = Int(pos_of[unsafe_offset=Int(b)])
            k -= 1
        var idx = 0
        for c in range(len(chain) - 1, -1, -1):
            out_bkps[unsafe_offset=idx] = chain[c]
            idx += 1
        out_bkps[unsafe_offset=idx] = Int64(n)
        status = Int32(n_bkps + 1)

    pos.unsafe_free()
    pos_of.unsafe_free()
    matrix.unsafe_free()
    buf1.unsafe_free()
    buf2.unsafe_free()
    G.unsafe_free()
    parent.unsafe_free()
    return status


# ---------------------------------------------------------------------------
# Pelt: penalized optimal partition with pruning.
# ---------------------------------------------------------------------------


def pelt_detect(
    model: Int32,
    x: F64Ptr,
    n: Int,
    min_size: Int,
    jump: Int,
    pen: Float64,
    out_bkps: I64Ptr,
    out_cap: Int,
) -> Int32:
    var buf1 = unsafe_alloc[Float64](n)
    var buf2 = unsafe_alloc[Float64](n)
    var F = unsafe_alloc[Float64](n + 1)
    var defined = unsafe_alloc[UInt8](n + 1)
    var parent = unsafe_alloc[Int64](n + 1)
    for i in range(n + 1):
        F[unsafe_offset=i] = 0.0
        defined[unsafe_offset=i] = 0
        parent[unsafe_offset=i] = -1
    F[unsafe_offset=0] = 0.0
    defined[unsafe_offset=0] = 1

    var admissible = List[Int]()
    var status = Int32(0)

    # ind: multiples of jump in [0, n) that are >= min_size, plus n.
    var ind = List[Int]()
    var k0 = (min_size + jump - 1) // jump * jump  # smallest multiple >= min_size
    var b0 = k0
    while b0 < n:
        ind.append(b0)
        b0 += jump
    ind.append(n)

    for bi in range(len(ind)):
        var bkp = ind[bi]
        var new_adm_pt = ((bkp - min_size) // jump) * jump
        admissible.append(new_adm_pt)

        var best_total = INF
        var best_t = -1
        var totals = List[Float64]()  # aligned with `admissible`; NaN = skipped
        for ai in range(len(admissible)):
            var t = admissible[ai]
            if defined[unsafe_offset=t] == 0:
                totals.append(NAN)  # skipped; the prune step drops it
                continue
            var total = F[unsafe_offset=t] + (
                seg_cost(model, x, t, bkp, buf1, buf2) + pen
            )
            totals.append(total)
            if total < best_total:  # strict: smallest t wins ties
                best_total = total
                best_t = t
        if best_t < 0:
            status = STATUS_EMPTY_CANDIDATES
            break
        F[unsafe_offset=bkp] = best_total
        defined[unsafe_offset=bkp] = 1
        parent[unsafe_offset=bkp] = Int64(best_t)
        # Prune starts whose candidate total exceeds optimum + pen.
        var kept = List[Int]()
        for ai in range(len(admissible)):
            var tot = totals[ai]
            if tot == tot and tot <= best_total + pen:  # tot==tot drops NaN skips
                kept.append(admissible[ai])
        admissible = kept^

    if status == 0:
        # Backtrack from n.
        var chain = List[Int64]()
        var t = n
        while parent[unsafe_offset=t] > 0:
            chain.append(parent[unsafe_offset=t])
            t = Int(parent[unsafe_offset=t])
        var count = len(chain) + 1
        if count > out_cap:
            status = STATUS_OUTPUT_CAPACITY
        else:
            var idx = 0
            for c in range(len(chain) - 1, -1, -1):
                out_bkps[unsafe_offset=idx] = chain[c]
                idx += 1
            out_bkps[unsafe_offset=idx] = Int64(n)
            status = Int32(count)

    buf1.unsafe_free()
    buf2.unsafe_free()
    F.unsafe_free()
    defined.unsafe_free()
    parent.unsafe_free()
    return status


# ---------------------------------------------------------------------------
# Binseg: greedy binary segmentation.
# ---------------------------------------------------------------------------


def single_bkp(
    model: Int32,
    x: F64Ptr,
    start: Int,
    end: Int,
    jump: Int,
    min_size: Int,
    buf1: F64Ptr,
    buf2: F64Ptr,
) -> Tuple[Int, Float64]:
    """Best split of x[start:end]: (bkp, gain); (-1, 0.0) if none exists.

    Gain ties resolve to the largest candidate breakpoint (tuple max).
    """
    var segment_cost = seg_cost(model, x, start, end, buf1, buf2)
    var best_gain = 0.0
    var best_bkp = -1
    var b = start
    while b < end:
        if b - start >= min_size and end - b >= min_size:
            var gain = (
                segment_cost
                - seg_cost(model, x, start, b, buf1, buf2)
                - seg_cost(model, x, b, end, buf1, buf2)
            )
            if best_bkp < 0 or gain > best_gain or (
                gain == best_gain and b > best_bkp
            ):
                best_gain = gain
                best_bkp = b
        b += jump
    if best_bkp < 0:
        return (-1, 0.0)
    return (best_bkp, best_gain)


def binseg_detect(
    model: Int32,
    x: F64Ptr,
    n: Int,
    min_size: Int,
    jump: Int,
    n_bkps: Int,
    out_bkps: I64Ptr,
    out_cap: Int,
) -> Int32:
    var buf1 = unsafe_alloc[Float64](n)
    var buf2 = unsafe_alloc[Float64](n)
    var bkps = List[Int]()
    bkps.append(n)
    # Memo of single_bkp results per (start, end) segment.
    var memo_bkp = Dict[UInt64, Int64]()
    var memo_gain = Dict[UInt64, Float64]()

    while True:
        # Best split across current segments: max gain, leftmost on ties.
        var best_bkp = -1
        var best_gain = 0.0
        var first = True
        var prev = 0
        for si in range(len(bkps)):
            var end = bkps[si]
            var start = prev
            prev = end
            var key = (UInt64(start) << 32) | UInt64(end)
            var sb: Int
            var sg: Float64
            if key in memo_bkp:
                sb = Int(memo_bkp.get(key, Int64(-1)))
                sg = memo_gain.get(key, 0.0)
            else:
                var res = single_bkp(model, x, start, end, jump, min_size, buf1, buf2)
                sb = res[0]
                sg = res[1]
                memo_bkp[key] = Int64(sb)
                memo_gain[key] = sg
            # (None, 0) participates with gain 0, like the reference.
            var cand_gain = sg
            var cand_bkp = sb
            if first or cand_gain > best_gain:
                first = False
                best_gain = cand_gain
                best_bkp = cand_bkp
        if best_bkp < 0:  # all configurations explored
            break
        if len(bkps) - 1 >= n_bkps:
            break
        # Insert sorted (bkps stays ascending).
        var ins = len(bkps)
        for si in range(len(bkps)):
            if bkps[si] > best_bkp:
                ins = si
                break
        bkps.insert(ins, best_bkp)

    var status: Int32
    if len(bkps) > out_cap:
        status = STATUS_OUTPUT_CAPACITY
    else:
        for si in range(len(bkps)):
            out_bkps[unsafe_offset=si] = Int64(bkps[si])
        status = Int32(len(bkps))

    buf1.unsafe_free()
    buf2.unsafe_free()
    return status


# ---------------------------------------------------------------------------
# Exported C ABI.
# ---------------------------------------------------------------------------


@export
def rupturesmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def rupturesmojo_detect(
    signal: F64Ptr,
    n: Int64,
    method: Int32,
    model: Int32,
    min_size: Int64,
    jump: Int64,
    n_bkps: Int64,
    pen: Float64,
    out_bkps: I64Ptr,
    out_cap: Int64,
) abi("C") -> Int32:
    """Run one whole detection on a finite 1-D float64 signal.

    `out_bkps` must hold at least out_cap int64 slots; the returned count is
    the number of breakpoints written (always ending with n).
    """
    if n < 1 or jump < 1 or min_size < 1:
        return STATUS_INVALID_ARGUMENT
    if model != MODEL_L2 and model != MODEL_L1:
        return STATUS_INVALID_ARGUMENT
    if method != METHOD_DYNP and method != METHOD_PELT and method != METHOD_BINSEG:
        return STATUS_INVALID_ARGUMENT
    if n_bkps < 0:
        return STATUS_INVALID_ARGUMENT

    var nn = Int(n)
    var ms = Int(min_size)
    var jp = Int(jump)
    var kb = Int(n_bkps)
    # Effective min_size: max(requested, cost model minimum).
    var cost_min = 1 if model == MODEL_L2 else 2
    if cost_min > ms:
        ms = cost_min

    if method == METHOD_DYNP:
        if not sanity_check(nn, kb, jp, ms):
            return STATUS_BAD_SEGMENTATION
        if kb == 0:
            if out_cap < 1:
                return STATUS_OUTPUT_CAPACITY
            out_bkps[unsafe_offset=0] = n
            return 1
        return dynp_detect(model, signal, nn, ms, jp, kb, out_bkps, Int(out_cap))
    elif method == METHOD_PELT:
        if not sanity_check(nn, 0, jp, ms):
            return STATUS_BAD_SEGMENTATION
        return pelt_detect(model, signal, nn, ms, jp, pen, out_bkps, Int(out_cap))
    else:
        if not sanity_check(nn, kb, jp, ms):
            return STATUS_BAD_SEGMENTATION
        return binseg_detect(model, signal, nn, ms, jp, kb, out_bkps, Int(out_cap))
