"""Clean-room Levenshtein / Damerau-Levenshtein edit-distance kernel.

Written fresh from the textbook dynamic program for edit distance and the
published Lowrance-Wagner recurrence for unrestricted Damerau-Levenshtein
(transposition with intervening edits), plus the Optimal String Alignment
(OSA, "restricted") adjacent-transposition variant. No third-party Mojo code
is used or adapted.

The kernel operates on UTF-16 code units, exactly like JavaScript strings
(`String.length`, `charCodeAt`, `charAt` are all UTF-16 based), so distances
are identical to the JS reference with no index conversion. Astral characters
(e.g. emoji) occupy two code units, precisely mirroring the JS behavior.

Exported C ABI (stateless: one call computes one pair's distance; all scratch
is allocated and freed inside the call):

    int32_t  naturalmojo_abi_version(void)
    float64  naturalmojo_distance(source, source_len, target, target_len,
                                  insertion_cost, deletion_cost,
                                  substitution_cost, transposition_cost,
                                  damerau, restricted)
             -> the edit distance (integer-valued when all costs are
                integers), or -1.0 on invalid input (negative length)

All arithmetic is IEEE-754 float64 in the same operation order as the JS
reference implementation, so results agree bit-for-bit (distances are
compared for exact equality in the differential suite). With the default
unit costs every value is a small non-negative integer.

Semantics (matching the documented behavior of the reference package):

    D[0][0] = 0;  D[i][0] = i * deletion_cost;  D[0][j] = j * insertion_cost
    D[i][j] = min(
        D[i][j-1]   + insertion_cost,                        # insertion
        D[i-1][j]   + deletion_cost,                         # deletion
        D[i-1][j-1] + (s[i] != t[j] ? substitution_cost : 0) # substitution
        # restricted (OSA), when i,j > 1 and s[i-1..i] == reversed t[j-1..j]:
        D[i-2][j-2] + transposition_cost
        # unrestricted (Lowrance-Wagner), when i,j > 1 and l = last row with
        # s[l] == t[j] (per the reference's row-scoped last-occurrence map)
        # and k = last column < j in this row with s[i] == t[k]:
        D[l-1][k-1] + (i-l-1)*deletion_cost + (j-k-1)*insertion_cost
                      + transposition_cost)

Scratch strategy: plain Levenshtein and OSA keep three rolling rows
(OSA needs D[i-2]); unrestricted Damerau needs D[l-1][k-1] for an arbitrary
past row l, so it keeps the full (n+1) x (m+1) float64 matrix plus a
65536-entry last-occurrence map over UTF-16 code units (zeroed per call).
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]

# One Int32 slot per UTF-16 code unit for the unrestricted variant's
# last-occurrence map (row numbers are 1-based; 0 means "not seen").
comptime ALPHABET_SIZE = 65536


def _min3(a: Float64, b: Float64, c: Float64) -> Float64:
    var m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


@export
def naturalmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def naturalmojo_distance(
    source: U16Ptr,
    source_len: Int32,
    target: U16Ptr,
    target_len: Int32,
    insertion_cost: Float64,
    deletion_cost: Float64,
    substitution_cost: Float64,
    transposition_cost: Float64,
    damerau: Int32,
    restricted: Int32,
) abi("C") -> Float64:
    """Compute the edit distance between two UTF-16 strings.

    damerau=0: plain Levenshtein (insert/delete/substitute).
    damerau=1, restricted=1: OSA (adjacent transposition, no substring
        edited twice).
    damerau=1, restricted=0: unrestricted Damerau-Levenshtein
        (Lowrance-Wagner transpositions).

    Returns -1.0 on invalid input (negative length). The caller owns the
    input buffers; all scratch is allocated and freed inside this call.
    """
    if source_len < 0 or target_len < 0:
        return -1.0

    var n = Int(source_len)
    var m = Int(target_len)

    if damerau != 0 and restricted == 0:
        return _distance_unrestricted(
            source,
            n,
            target,
            m,
            insertion_cost,
            deletion_cost,
            substitution_cost,
            transposition_cost,
        )
    return _distance_rolling(
        source,
        n,
        target,
        m,
        insertion_cost,
        deletion_cost,
        substitution_cost,
        transposition_cost,
        damerau != 0,
    )


def _distance_rolling(
    source: U16Ptr,
    n: Int,
    target: U16Ptr,
    m: Int,
    insertion_cost: Float64,
    deletion_cost: Float64,
    substitution_cost: Float64,
    transposition_cost: Float64,
    osa: Bool,
) -> Float64:
    """Plain Levenshtein / OSA over three rolling rows (row i-2 is needed
    only for the OSA transposition term)."""
    var width = m + 1
    var row_im2 = unsafe_alloc[Float64](width)
    var row_im1 = unsafe_alloc[Float64](width)
    var row_cur = unsafe_alloc[Float64](width)

    # Row 0: D[0][j] = j * insertion_cost.
    for j in range(width):
        row_cur[unsafe_offset=j] = Float64(j) * insertion_cost

    var result = row_cur[unsafe_offset=m]
    for i in range(1, n + 1):
        # Rotate the row buffers: cur becomes i-1, im1 becomes i-2.
        var tmp = row_im2
        row_im2 = row_im1
        row_im1 = row_cur
        row_cur = tmp
        row_cur[unsafe_offset=0] = Float64(i) * deletion_cost
        var s_i = source[unsafe_offset=i - 1]
        for j in range(1, m + 1):
            var cost_insert = row_cur[unsafe_offset=j - 1] + insertion_cost
            var cost_delete = row_im1[unsafe_offset=j] + deletion_cost
            var cost_substitute = row_im1[unsafe_offset=j - 1]
            var t_j = target[unsafe_offset=j - 1]
            if s_i != t_j:
                cost_substitute += substitution_cost
            var best = _min3(cost_insert, cost_delete, cost_substitute)
            if osa and i > 1 and j > 1:
                # Adjacent transposition: s[i-2..i] is the reverse of
                # t[j-2..j] (1-based string positions).
                if s_i == target[unsafe_offset=j - 2] and source[
                    unsafe_offset=i - 2
                ] == t_j:
                    var cost_transpose = row_im2[unsafe_offset=j - 2] + transposition_cost
                    if cost_transpose < best:
                        best = cost_transpose
            row_cur[unsafe_offset=j] = best
        result = row_cur[unsafe_offset=m]

    row_im2.unsafe_free()
    row_im1.unsafe_free()
    row_cur.unsafe_free()
    return result


def _distance_unrestricted(
    source: U16Ptr,
    n: Int,
    target: U16Ptr,
    m: Int,
    insertion_cost: Float64,
    deletion_cost: Float64,
    substitution_cost: Float64,
    transposition_cost: Float64,
) -> Float64:
    """Unrestricted Damerau-Levenshtein (Lowrance-Wagner). Needs the full
    matrix: the transposition term references D[l-1][k-1] for the last row l
    whose source character equals the current target character.

    `last_row` maps a UTF-16 code unit to the last 1-based matrix row whose
    source character is that unit (0 = not seen). Mirroring the reference,
    the entry for the current row's source character is (re)written with the
    current row at every cell, and `last_col_match` is the last column of the
    current row whose target character equals the current source character
    (0 = none yet). The map is reset afterwards only for touched entries.
    """
    var width = m + 1
    var cells = (n + 1) * width
    var matrix = unsafe_alloc[Float64](cells)
    var last_row = unsafe_alloc[Int32](ALPHABET_SIZE)
    for i in range(ALPHABET_SIZE):
        last_row[unsafe_offset=i] = 0

    # Initialize row 0 and column 0.
    for j in range(width):
        matrix[unsafe_offset=j] = Float64(j) * insertion_cost
    for i in range(1, n + 1):
        matrix[unsafe_offset=i * width] = Float64(i) * deletion_cost

    for i in range(1, n + 1):
        var last_col_match = 0
        var s_i = source[unsafe_offset=i - 1]
        var row_base = i * width
        var prev_base = (i - 1) * width
        for j in range(1, m + 1):
            var cost_insert = matrix[unsafe_offset=row_base + j - 1] + insertion_cost
            var cost_delete = matrix[unsafe_offset=prev_base + j] + deletion_cost
            var cost_substitute = matrix[unsafe_offset=prev_base + j - 1]
            var t_j = target[unsafe_offset=j - 1]
            if s_i != t_j:
                cost_substitute += substitution_cost
            var best = _min3(cost_insert, cost_delete, cost_substitute)

            if i > 1 and j > 1 and last_col_match > 0:
                var l = Int(last_row[unsafe_offset=Int(t_j)])
                if l > 0:
                    var cost_transpose = (
                        matrix[unsafe_offset=(l - 1) * width + last_col_match - 1]
                        + Float64(i - l - 1) * deletion_cost
                        + Float64(j - last_col_match - 1) * insertion_cost
                        + transposition_cost
                    )
                    if cost_transpose < best:
                        best = cost_transpose

            matrix[unsafe_offset=row_base + j] = best
            last_row[unsafe_offset=Int(s_i)] = Int32(i)
            if s_i == t_j:
                last_col_match = j

    var result = matrix[unsafe_offset=n * width + m]

    # Reset only the touched map entries (one per source position), so the
    # next call on re-used memory never observes stale rows. The map is
    # freed here; the reset keeps the routine safe if allocation strategies
    # change.
    for i in range(1, n + 1):
        last_row[unsafe_offset=Int(source[unsafe_offset=i - 1])] = 0

    matrix.unsafe_free()
    last_row.unsafe_free()
    return result
