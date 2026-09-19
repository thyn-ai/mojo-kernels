"""Clean-room Ckmeans.1d.dp kernel: optimal 1-D k-means by dynamic programming.

Written fresh from the published algorithm (Wang & Song, "Ckmeans.1d.dp:
Optimal k-means Clustering in One Dimension by Dynamic Programming", The R
Journal Vol. 3/2, 2011, including the 3.4.6 divide-and-conquer fill that
narrows each cell's split-point scan with already-computed backtrack
bounds). No third-party Mojo code is used or adapted. The kernel replicates
the observable behavior of the simple-statistics `ckmeans` reference —
including its exact tie resolution (a descending split-point scan with
strict `<` comparisons, so on equal cost the largest split point wins), its
median-shifted cumulative sums, and its two-branch within-cluster
sum-of-squares formula — so cluster assignments agree exactly, not
approximately.

All floating-point arithmetic is IEEE-754 float64 in the reference's
operation order (cumulative sums accumulated sequentially, `(count * mu) *
mu` multiplication order, negative-ssq clamped to zero with NaN passing
through), so every argmin comparison sees bit-identical operands.

Exported C ABI (single-shot; no state is kept across calls):

    int32_t ckmeansmojo_abi_version(void)
    int32_t ckmeansmojo_cluster(const float64* sorted, int32_t n, int32_t k,
                                int32_t* out_lefts /* [k] */)
    int32_t ckmeansmojo_cluster_unsorted(const float64* data, int32_t n,
                                         int32_t k, int32_t* out_lefts,
                                         float64* out_sorted /* [n] */)

`out_lefts[c]` is the start index (into the sorted array) of cluster c;
cluster c covers sorted[out_lefts[c] .. out_lefts[c+1] - 1] with
out_lefts[k] implicitly n. The unsorted entry additionally stable-sorts the
input natively (bit-identical order to the reference's stable Array sort;
the wrapper routes NaN-containing data to the sorted entry instead, where
the sort kept the engine's own comparator semantics). Both return 0 on
success, 2 on n < 1, 3 on k outside [1, n]; the unsorted entry returns 10
when the data has a single unique value.
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the input and
# output buffers; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]


def _ssq(j: Int32, i: Int32, sums: F64Ptr, sums_of_squares: F64Ptr) -> Float64:
    """Within-cluster sum of squared deviations for sorted[j .. i], from
    median-shifted cumulative sums. Two branches exactly like the reference
    (j > 0 subtracts the prefix through j-1; j == 0 uses the whole prefix),
    with negative results clamped to 0 (NaN compares false and passes
    through, as in the reference)."""
    var sji: Float64
    if j > 0:
        var count = Float64(i - j + 1)
        var muji = (sums[unsafe_offset=Int(i)] - sums[unsafe_offset=Int(j - 1)]) / count
        sji = (
            sums_of_squares[unsafe_offset=Int(i)]
            - sums_of_squares[unsafe_offset=Int(j - 1)]
            - count * muji * muji
        )
    else:
        var sum_i = sums[unsafe_offset=Int(i)]
        sji = sums_of_squares[unsafe_offset=Int(i)] - (sum_i * sum_i) / Float64(i + 1)
    if sji < 0.0:
        return Float64(0.0)
    return sji


def _fill_matrix_column(
    i_min: Int32,
    i_max: Int32,
    cluster: Int32,
    n: Int32,
    prev_cost: F64Ptr,
    cur_cost: F64Ptr,
    prev_bt: I32Ptr,
    cur_bt: I32Ptr,
    sums: F64Ptr,
    sums_of_squares: F64Ptr,
):
    """Fill cost/backtrack cells for `cluster` over i in [i_min, i_max],
    midpoint-first, recursing left then right. Each cell scans candidate
    split points j descending from jhigh to jlow, where the bounds come from
    already-computed backtrack cells (optimal split points are monotone in
    i); strict `<` updates plus the `>=` early break make the largest
    cost-minimizing j win every tie, and re-reading the raised jlow each
    iteration matches the reference's shrinking-lower-bound loop."""
    if i_min > i_max:
        return

    var i = (i_min + i_max) // 2

    # Initial candidate: cluster [i..i] alone appended to the optimum over
    # the first i values (ssq(i, i) == 0, so the cost is the previous row's).
    cur_cost[unsafe_offset=Int(i)] = prev_cost[unsafe_offset=Int(i - 1)]
    cur_bt[unsafe_offset=Int(i)] = i

    var jlow = cluster
    if i_min > cluster:
        var bt_low = cur_bt[unsafe_offset=Int(i_min - 1)]
        if bt_low > jlow:
            jlow = bt_low
    var bt_prev = prev_bt[unsafe_offset=Int(i)]
    if bt_prev > jlow:
        jlow = bt_prev

    var jhigh = i - 1
    if i_max < n - 1:
        var bt_high = cur_bt[unsafe_offset=Int(i_max + 1)]
        if bt_high < jhigh:
            jhigh = bt_high

    var j = jhigh
    while j >= jlow:
        var sji = _ssq(j, i, sums, sums_of_squares)
        var prev_at_jlow = prev_cost[unsafe_offset=Int(jlow - 1)]
        if sji + prev_at_jlow >= cur_cost[unsafe_offset=Int(i)]:
            break

        # Examine the lower bound of the cluster border, then shrink it.
        var sjlowi = _ssq(jlow, i, sums, sums_of_squares)
        var ssqjlow = sjlowi + prev_at_jlow
        if ssqjlow < cur_cost[unsafe_offset=Int(i)]:
            cur_cost[unsafe_offset=Int(i)] = ssqjlow
            cur_bt[unsafe_offset=Int(i)] = jlow
        jlow += 1

        var ssqj = sji + prev_cost[unsafe_offset=Int(j - 1)]
        if ssqj < cur_cost[unsafe_offset=Int(i)]:
            cur_cost[unsafe_offset=Int(i)] = ssqj
            cur_bt[unsafe_offset=Int(i)] = j
        j -= 1

    _fill_matrix_column(
        i_min, i - 1, cluster, n, prev_cost, cur_cost, prev_bt, cur_bt, sums, sums_of_squares
    )
    _fill_matrix_column(
        i + 1, i_max, cluster, n, prev_cost, cur_cost, prev_bt, cur_bt, sums, sums_of_squares
    )


def _stable_merge_sort(data: F64Ptr, n: Int32, sorted_out: F64Ptr):
    """Bottom-up stable merge sort of `data` into `sorted_out`, numeric ascending.

    The input is finite (the wrapper routes NaN-containing data to its JS
    sort path). Stability matters: -0 and +0 compare equal, and keeping
    their original relative order reproduces the reference's stable Array
    sort exactly, so downstream sums see bit-identical operands."""
    var buf = unsafe_alloc[Float64](Int(n))
    var i = Int32(0)
    while i < n:
        sorted_out[unsafe_offset=Int(i)] = data[unsafe_offset=Int(i)]
        i += 1

    var src = sorted_out
    var dst = buf
    var width = Int32(1)
    while width < n:
        var run = Int32(0)
        while run < n:
            var mid = run + width
            var end = run + 2 * width
            if mid > n:
                mid = n
            if end > n:
                end = n
            var left = run
            var right = mid
            var at = run
            while left < mid and right < end:
                if src[unsafe_offset=Int(left)] <= src[unsafe_offset=Int(right)]:
                    dst[unsafe_offset=Int(at)] = src[unsafe_offset=Int(left)]
                    left += 1
                else:
                    dst[unsafe_offset=Int(at)] = src[unsafe_offset=Int(right)]
                    right += 1
                at += 1
            while left < mid:
                dst[unsafe_offset=Int(at)] = src[unsafe_offset=Int(left)]
                left += 1
                at += 1
            while right < end:
                dst[unsafe_offset=Int(at)] = src[unsafe_offset=Int(right)]
                right += 1
                at += 1
            run += 2 * width
        var tmp = src
        src = dst
        dst = tmp
        width *= 2

    # After the swaps the sorted run may live in `buf`; land it in `sorted_out`.
    if src != sorted_out:
        i = 0
        while i < n:
            sorted_out[unsafe_offset=Int(i)] = src[unsafe_offset=Int(i)]
            i += 1
    buf.unsafe_free()


def _count_unique(sorted_data: F64Ptr, n: Int32) -> Int32:
    """Unique-value count of the sorted array, matching the reference's
    `!==` walk (with finite data, IEEE `!=` is the same test)."""
    var count = Int32(0)
    var last = Float64(0.0)
    var i = Int32(0)
    while i < n:
        var v = sorted_data[unsafe_offset=Int(i)]
        if i == 0 or v != last:
            last = v
            count += 1
        i += 1
    return count


def _run_dp(sorted_data: F64Ptr, n: Int32, k: Int32, out_lefts: I32Ptr):
    """Optimal 1-D k-means cluster boundaries for the sorted input: the
    median-shifted DP over cluster rows, then the backtrack walk. Allocates
    and frees its own scratch."""
    var sums = unsafe_alloc[Float64](Int(n))
    var sums_of_squares = unsafe_alloc[Float64](Int(n))
    # Two cost rows suffice (a fill reads only the previous row); the
    # backtrack matrix is kept whole for the boundary walk.
    var cost = unsafe_alloc[Float64](Int(n) * 2)
    var backtrack = unsafe_alloc[Int32](Int(n) * Int(k))

    # Zero the backtrack matrix (uncomputed cells must read as 0).
    var z = 0
    var bt_total = Int(n) * Int(k)
    while z < bt_total:
        backtrack[unsafe_offset=z] = 0
        z += 1

    # Shift values by the middle element to improve numeric stability,
    # exactly like the reference (index floor(n/2) of the sorted array).
    var shift = sorted_data[unsafe_offset=Int(n // 2)]
    var i = Int32(0)
    while i < n:
        var shifted = sorted_data[unsafe_offset=Int(i)] - shift
        if i == 0:
            sums[unsafe_offset=0] = shifted
            sums_of_squares[unsafe_offset=0] = shifted * shifted
        else:
            sums[unsafe_offset=Int(i)] = sums[unsafe_offset=Int(i - 1)] + shifted
            sums_of_squares[unsafe_offset=Int(i)] = (
                sums_of_squares[unsafe_offset=Int(i - 1)] + shifted * shifted
            )
        i += 1

    # Cluster row 0: one cluster covering sorted[0..i].
    i = 0
    while i < n:
        cost[unsafe_offset=Int(i)] = _ssq(0, i, sums, sums_of_squares)
        backtrack[unsafe_offset=Int(i)] = 0
        i += 1

    # Remaining rows; the last row only needs cell n-1.
    var cluster = Int32(1)
    while cluster < k:
        var i_min = cluster
        if cluster == k - 1:
            i_min = n - 1
        var prev_row = (cluster - 1) % 2
        var cur_row = cluster % 2
        _fill_matrix_column(
            i_min,
            n - 1,
            cluster,
            n,
            cost.unsafe_offset(Int(prev_row) * Int(n)),
            cost.unsafe_offset(Int(cur_row) * Int(n)),
            backtrack.unsafe_offset(Int(cluster - 1) * Int(n)),
            backtrack.unsafe_offset(Int(cluster) * Int(n)),
            sums,
            sums_of_squares,
        )
        cluster += 1

    # Backtrack the cluster boundaries from the bottom-right corner.
    var cluster_right = n - 1
    cluster = k - 1
    while cluster >= 0:
        var cluster_left = backtrack[unsafe_offset=Int(cluster) * Int(n) + Int(cluster_right)]
        out_lefts[unsafe_offset=Int(cluster)] = cluster_left
        if cluster > 0:
            cluster_right = cluster_left - 1
        cluster -= 1

    sums.unsafe_free()
    sums_of_squares.unsafe_free()
    cost.unsafe_free()
    backtrack.unsafe_free()


@export
def ckmeansmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def ckmeansmojo_cluster(
    sorted_data: F64Ptr, n: Int32, k: Int32, out_lefts: I32Ptr
) abi("C") -> Int32:
    """Cluster boundaries for an already-sorted input (the NaN path: the
    wrapper sorted with the engine's own comparator semantics).

    `out_lefts` must hold k slots. Returns 0 on success, 2 if n < 1, 3 if k
    is outside [1, n]."""
    if n < 1:
        return 2
    if k < 1 or k > n:
        return 3
    _run_dp(sorted_data, n, k, out_lefts)
    return 0


@export
def ckmeansmojo_cluster_unsorted(
    data: F64Ptr, n: Int32, k: Int32, out_lefts: I32Ptr, out_sorted: F64Ptr
) abi("C") -> Int32:
    """Full pipeline for finite data: stable-sort `data` into `out_sorted`
    (bit-identical order to the reference's stable Array sort), then run the
    DP and write the cluster-left boundaries.

    Returns 0 on success; 10 when the sorted data has a single unique value
    (the caller returns one cluster holding `out_sorted`, and `out_lefts` is
    untouched); 2 if n < 1; 3 if k is outside [1, n]."""
    if n < 1:
        return 2
    if k < 1 or k > n:
        return 3
    _stable_merge_sort(data, n, out_sorted)
    if _count_unique(out_sorted, n) == 1:
        return 10
    _run_dp(out_sorted, n, k, out_lefts)
    return 0
