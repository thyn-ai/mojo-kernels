"""Clean-room Ratcliff/Obershelp "gestalt pattern matching" kernel.

Re-implements, from the published algorithm description and the documented
public behavior of CPython 3.12's difflib.SequenceMatcher, the hot loops of
the standard library's sequence matcher:

  * find_longest_match: the O(len(a) x len(b)) longest-junk-free-match DP
    over a CSR index of b, with CPython-compatible tie breaking (strictly
    greater k wins; scanning i ascending and, per i, j ascending) and the
    two-phase match extension (first over non-junk elements, then over
    adjacent matching junk).
  * get_matching_blocks: the divide-and-conquer driver around it.
  * quick_ratio: multiset-intersection cardinality of a and b.
  * close-matches batching: one call scores many candidates against one
    word with get_close_matches' three-tier filter (real_quick_ratio,
    quick_ratio, ratio) and its autojunk popularity heuristic.

No third-party Mojo code is used or adapted; nothing here is derived from
CPython source. Sequences arrive as int32 codepoint arrays (the Python
wrapper encodes str -> UTF-32-LE); junk/popularity classification of b's
elements arrives as per-position byte masks computed by the wrapper (or, in
the batch entry point, is recomputed in-kernel from the autojunk rule:
popular = occurring more than len(b)//100 + 1 times when len(b) >= 200).

DP scratch uses generation stamps (per-call salt windows), so neither rows
nor calls pay an O(len(b)) reset. All ratio arithmetic is IEEE-754 float64
in the reference's operation order: 2.0 * matches / length, or 1.0 when the
sequences have no elements between them.

Exported C ABI (v1):

    int32_t  difflibmojo_abi_version(void)

    // Raw (unsorted, uncollapsed) matching-block triples for a pair.
    // outp receives ntriples*3 int64 (i, j, k); returns the triple count,
    // or -1 when out_cap < 3 * (min(la, lb) + 1). The wrapper sorts,
    // collapses adjacent blocks and appends the (la, lb, 0) sentinel.
    int64_t  difflibmojo_matching_blocks(
                 const int32_t* a, int64_t la,
                 const int32_t* b, int64_t lb,
                 const uint8_t* b_junk,     // [lb] 1 when b[j] is junk
                 const uint8_t* b_allowed,  // [lb] 1 when b[j]'s element is
                                            // in b2j (not junk, not popular)
                 int64_t* outp, int64_t out_cap)

    // One find_longest_match call; out3 receives (i, j, k).
    // Returns 0, or 2 on an out-of-range window.
    int32_t  difflibmojo_find_longest_match(
                 const int32_t* a, int64_t la,
                 const int32_t* b, int64_t lb,
                 const uint8_t* b_junk, const uint8_t* b_allowed,
                 int64_t alo, int64_t ahi, int64_t blo, int64_t bhi,
                 int64_t* out3)

    // Multiset-intersection upper bound, identical to quick_ratio().
    float64  difflibmojo_quick_ratio(const int32_t* a, int64_t la,
                                     const int32_t* b, int64_t lb)

    // Batch close-matches scoring: nw words (b-side) x nc candidates
    // (a-side), flattened codepoints + CSR offsets. For every pair applies
    // get_close_matches' filter chain at the given cutoff; pass_out[w*nc+c]
    // is 1 and ratio_out[w*nc+c] holds the ratio iff all three tiers pass.
    // autojunk != 0 applies the popularity heuristic to each word.
    // Returns 0 on success, 2 on invalid arguments.
    int32_t  difflibmojo_close_matches_batch(
                 const int32_t* words, const int64_t* words_off, int64_t nw,
                 const int32_t* cands, const int64_t* cands_off, int64_t nc,
                 float64 cutoff, int32_t autojunk,
                 uint8_t* pass_out, float64* ratio_out)
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates). IdxPtr is the
# kernel-internal index scratch (Int is 64-bit on every supported target).
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime IdxPtr = Pointer[Int, MutUntrackedOrigin]


struct Match3:
    """One matching block: a[i:i+k] == b[j:j+k]."""

    var i: Int
    var j: Int
    var k: Int

    def __init__(out self, i: Int, j: Int, k: Int):
        self.i = i
        self.j = j
        self.k = k


def _alloc_idx(n: Int) -> IdxPtr:
    """Zeroed Int scratch (at least one slot, so empty is never NULL)."""
    var m = n if n > 0 else 1
    var p = unsafe_alloc[Int](m)
    for i in range(m):
        p[unsafe_offset=i] = 0
    return p


def _alloc_u8(n: Int) -> U8Ptr:
    """Zeroed UInt8 scratch (at least one slot)."""
    var m = n if n > 0 else 1
    var p = unsafe_alloc[UInt8](m)
    for i in range(m):
        p[unsafe_offset=i] = 0
    return p


struct BSide:
    """Index of the b sequence plus scratch space for the match DP.

    CSR layout: for codepoint c, the ascending b-positions whose element is
    allowed (neither junk nor autojunk-popular) are
    pos[off[c] .. off[c+1]] — the same ascending order the reference's
    b2j lists produce. full[c] counts ALL occurrences of c in b (junk or
    not) for the quick_ratio multiset intersection. junk[j] is 1 when
    b[j] is a junk element. jlen/stamp are the row-DP scratch, salted per
    find_longest_match call so no array ever needs clearing; aval/astamp
    are the same trick for quick_ratio's avail bookkeeping.
    """

    var max_char: Int
    var off: IdxPtr  # max_char + 2 entries
    var pos: IdxPtr  # lb entries
    var full: IdxPtr  # max_char + 1 entries
    var junk: U8Ptr  # lb entries (owned copy)
    # Row-DP scratch, double-buffered by row parity: writes for row i go to
    # buffer (i-alo)&1 while reads hit buffer (i-1-alo)&1, so a write never
    # clobbers the previous row's value the way a single array would.
    var jlen0: IdxPtr  # lb entries
    var jlen1: IdxPtr  # lb entries
    var stamp0: IdxPtr  # lb entries
    var stamp1: IdxPtr  # lb entries
    var salt: Int
    var aval: IdxPtr  # max_char + 1 entries
    var astamp: IdxPtr  # max_char + 1 entries
    var asalt: Int

    def __init__(
        out self,
        max_char: Int,
        off: IdxPtr,
        pos: IdxPtr,
        full: IdxPtr,
        junk: U8Ptr,
        jlen0: IdxPtr,
        jlen1: IdxPtr,
        stamp0: IdxPtr,
        stamp1: IdxPtr,
        aval: IdxPtr,
        astamp: IdxPtr,
    ):
        self.max_char = max_char
        self.off = off
        self.pos = pos
        self.full = full
        self.junk = junk
        self.jlen0 = jlen0
        self.jlen1 = jlen1
        self.stamp0 = stamp0
        self.stamp1 = stamp1
        self.salt = 0
        self.aval = aval
        self.astamp = astamp
        self.asalt = 0

    def free(mut self):
        self.off.unsafe_free()
        self.pos.unsafe_free()
        self.full.unsafe_free()
        self.junk.unsafe_free()
        self.jlen0.unsafe_free()
        self.jlen1.unsafe_free()
        self.stamp0.unsafe_free()
        self.stamp1.unsafe_free()
        self.aval.unsafe_free()
        self.astamp.unsafe_free()


def _build_bside(
    b: I32Ptr,
    lb: Int,
    junk_src: Optional[U8Ptr],
    allowed_src: Optional[U8Ptr],
    autojunk: Bool,
) -> BSide:
    """Build the CSR index of b. When both masks are present they arrive as
    per-position flags from the wrapper; otherwise junk is empty and the
    autojunk popularity rule is applied in-kernel."""
    var maxc = 0
    for j in range(lb):
        if Int(b[unsafe_offset=j]) > maxc:
            maxc = Int(b[unsafe_offset=j])

    var full = _alloc_idx(maxc + 1)
    for j in range(lb):
        full[unsafe_offset=Int(b[unsafe_offset=j])] += 1

    var junk = _alloc_u8(lb)
    var allowed = _alloc_u8(lb)
    if junk_src and allowed_src:
        var js = junk_src.value()
        var al = allowed_src.value()
        for j in range(lb):
            junk[unsafe_offset=j] = js[unsafe_offset=j]
            allowed[unsafe_offset=j] = al[unsafe_offset=j]
    elif autojunk and lb >= 200:
        # Popular elements (more than 1% + 1 of b) leave the index.
        var ntest = lb // 100 + 1
        for j in range(lb):
            allowed[unsafe_offset=j] = (
                1 if full[unsafe_offset=Int(b[unsafe_offset=j])] <= ntest else 0
            )
    else:
        for j in range(lb):
            allowed[unsafe_offset=j] = 1

    var off = _alloc_idx(maxc + 2)
    for j in range(lb):
        if allowed[unsafe_offset=j] != 0:
            off[unsafe_offset=Int(b[unsafe_offset=j]) + 1] += 1
    for c in range(maxc + 1):
        off[unsafe_offset=c + 1] += off[unsafe_offset=c]

    var pos = _alloc_idx(lb)
    var cursor = _alloc_idx(maxc + 1)
    for c in range(maxc + 1):
        cursor[unsafe_offset=c] = off[unsafe_offset=c]
    for j in range(lb):
        if allowed[unsafe_offset=j] != 0:
            var c = Int(b[unsafe_offset=j])
            pos[unsafe_offset=cursor[unsafe_offset=c]] = j
            cursor[unsafe_offset=c] += 1
    cursor.unsafe_free()
    allowed.unsafe_free()

    return BSide(
        maxc,
        off,
        pos,
        full,
        junk,
        _alloc_idx(lb),
        _alloc_idx(lb),
        _alloc_idx(lb),
        _alloc_idx(lb),
        _alloc_idx(maxc + 1),
        _alloc_idx(maxc + 1),
    )


def _flm(
    mut bs: BSide,
    a: I32Ptr,
    b: I32Ptr,
    alo: Int,
    ahi: Int,
    blo: Int,
    bhi: Int,
) -> Match3:
    """Longest junk-free matching block in a[alo:ahi] vs b[blo:bhi],
    with the reference's tie breaking and junk extension."""
    var besti = alo
    var bestj = blo
    var bestsize = 0
    var base = bs.salt

    for i in range(alo, ahi):
        var ci = Int(a[unsafe_offset=i])
        if ci < 0 or ci > bs.max_char:
            continue
        var rowstamp = base + (i - alo) + 1
        var cur = bs.stamp0
        var prev = bs.stamp1
        var curv = bs.jlen0
        var prevv = bs.jlen1
        if ((i - alo) & 1) == 1:
            cur = bs.stamp1
            prev = bs.stamp0
            curv = bs.jlen1
            prevv = bs.jlen0
        var lo = bs.off[unsafe_offset=ci]
        var hi = bs.off[unsafe_offset=ci + 1]
        for p in range(lo, hi):
            var j = bs.pos[unsafe_offset=p]
            if j < blo:
                continue
            if j >= bhi:
                break
            var k = 1
            if j > 0 and prev[unsafe_offset=j - 1] == rowstamp - 1:
                k = prevv[unsafe_offset=j - 1] + 1
            curv[unsafe_offset=j] = k
            cur[unsafe_offset=j] = rowstamp
            if k > bestsize:
                besti = i - k + 1
                bestj = j - k + 1
                bestsize = k

    # The salt window this call used can never collide with a later call.
    bs.salt = base + (ahi - alo) + 1

    # Extend the best by non-junk elements on each end (this is how
    # autojunk-popular elements, absent from the index, still join a match).
    while (
        besti > alo
        and bestj > blo
        and bs.junk[unsafe_offset=bestj - 1] == 0
        and a[unsafe_offset=besti - 1] == b[unsafe_offset=bestj - 1]
    ):
        besti -= 1
        bestj -= 1
        bestsize += 1
    while (
        besti + bestsize < ahi
        and bestj + bestsize < bhi
        and bs.junk[unsafe_offset=bestj + bestsize] == 0
        and a[unsafe_offset=besti + bestsize]
        == b[unsafe_offset=bestj + bestsize]
    ):
        bestsize += 1

    # ... then suck up matching junk adjacent to the interesting match.
    while (
        besti > alo
        and bestj > blo
        and bs.junk[unsafe_offset=bestj - 1] != 0
        and a[unsafe_offset=besti - 1] == b[unsafe_offset=bestj - 1]
    ):
        besti -= 1
        bestj -= 1
        bestsize += 1
    while (
        besti + bestsize < ahi
        and bestj + bestsize < bhi
        and bs.junk[unsafe_offset=bestj + bestsize] != 0
        and a[unsafe_offset=besti + bestsize]
        == b[unsafe_offset=bestj + bestsize]
    ):
        bestsize += 1

    return Match3(besti, bestj, bestsize)


def _blocks(
    mut bs: BSide,
    a: I32Ptr,
    la: Int,
    b: I32Ptr,
    lb: Int,
    mut triples: List[Int],
    collect: Bool,
) -> Int:
    """Divide-and-conquer driver: returns the sum of block sizes (the M in
    ratio = 2.0*M / T) and, when collect, appends every raw (i, j, k)
    triple to `triples` in discovery order."""
    var stack = List[Int]()
    stack.append(0)
    stack.append(la)
    stack.append(0)
    stack.append(lb)
    var sumk = 0
    while len(stack) > 0:
        var bhi = stack.pop()
        var blo = stack.pop()
        var ahi = stack.pop()
        var alo = stack.pop()
        var m = _flm(bs, a, b, alo, ahi, blo, bhi)
        if m.k > 0:
            sumk += m.k
            if collect:
                triples.append(m.i)
                triples.append(m.j)
                triples.append(m.k)
            if alo < m.i and blo < m.j:
                stack.append(alo)
                stack.append(m.i)
                stack.append(blo)
                stack.append(m.j)
            if m.i + m.k < ahi and m.j + m.k < bhi:
                stack.append(m.i + m.k)
                stack.append(ahi)
                stack.append(m.j + m.k)
                stack.append(bhi)
    return sumk


def _quick_matches(mut bs: BSide, a: I32Ptr, la: Int) -> Int:
    """Multiset-intersection cardinality of a and b (quick_ratio's M)."""
    bs.asalt += 1
    var salt = bs.asalt
    var matches = 0
    for i in range(la):
        var c = Int(a[unsafe_offset=i])
        if c < 0 or c > bs.max_char:
            continue  # not in b at all: consumes nothing, matches nothing
        var numb = bs.full[unsafe_offset=c]
        if bs.astamp[unsafe_offset=c] == salt:
            numb = bs.aval[unsafe_offset=c]
        bs.astamp[unsafe_offset=c] = salt
        bs.aval[unsafe_offset=c] = numb - 1
        if numb > 0:
            matches += 1
    return matches


def _calc_ratio(m: Int, t: Int) -> Float64:
    """2.0*M / T, or 1.0 when a and b are both empty — the reference's rule."""
    if t == 0:
        return 1.0
    return 2.0 * Float64(m) / Float64(t)


@export
def difflibmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def difflibmojo_matching_blocks(
    a: I32Ptr,
    la: Int64,
    b: I32Ptr,
    lb: Int64,
    b_junk: U8Ptr,
    b_allowed: U8Ptr,
    outp: I64Ptr,
    out_cap: Int64,
) abi("C") -> Int64:
    var cap_needed = 3 * ((la if la < lb else lb) + 1)
    if la < 0 or lb < 0 or out_cap < cap_needed:
        return -1
    var bs = _build_bside(
        b, Int(lb), Optional[U8Ptr](b_junk), Optional[U8Ptr](b_allowed), False
    )
    var triples = List[Int]()
    _ = _blocks(bs, a, Int(la), b, Int(lb), triples, True)
    var n = len(triples) // 3
    for t in range(n):
        outp[unsafe_offset=3 * t] = Int64(triples[3 * t])
        outp[unsafe_offset=3 * t + 1] = Int64(triples[3 * t + 1])
        outp[unsafe_offset=3 * t + 2] = Int64(triples[3 * t + 2])
    bs.free()
    return Int64(n)


@export
def difflibmojo_find_longest_match(
    a: I32Ptr,
    la: Int64,
    b: I32Ptr,
    lb: Int64,
    b_junk: U8Ptr,
    b_allowed: U8Ptr,
    alo: Int64,
    ahi: Int64,
    blo: Int64,
    bhi: Int64,
    out3: I64Ptr,
) abi("C") -> Int32:
    if (
        la < 0
        or lb < 0
        or alo < 0
        or blo < 0
        or alo > ahi
        or blo > bhi
        or ahi > la
        or bhi > lb
    ):
        return 2
    var bs = _build_bside(
        b, Int(lb), Optional[U8Ptr](b_junk), Optional[U8Ptr](b_allowed), False
    )
    var m = _flm(bs, a, b, Int(alo), Int(ahi), Int(blo), Int(bhi))
    out3[unsafe_offset=0] = Int64(m.i)
    out3[unsafe_offset=1] = Int64(m.j)
    out3[unsafe_offset=2] = Int64(m.k)
    bs.free()
    return 0


@export
def difflibmojo_quick_ratio(
    a: I32Ptr, la: Int64, b: I32Ptr, lb: Int64
) abi("C") -> Float64:
    if la < 0 or lb < 0:
        return 0.0
    var bs = _build_bside(b, Int(lb), None, None, False)
    var m = _quick_matches(bs, a, Int(la))
    bs.free()
    return _calc_ratio(m, Int(la + lb))


@export
def difflibmojo_close_matches_batch(
    words: I32Ptr,
    words_off: I64Ptr,
    nw: Int64,
    cands: I32Ptr,
    cands_off: I64Ptr,
    nc: Int64,
    cutoff: Float64,
    autojunk: Int32,
    pass_out: U8Ptr,
    ratio_out: F64Ptr,
) abi("C") -> Int32:
    if nw < 0 or nc < 0:
        return 2
    for w in range(Int(nw)):
        var bw0 = Int(words_off[unsafe_offset=w])
        var lb = Int(words_off[unsafe_offset=w + 1]) - bw0
        var b = words.unsafe_offset(bw0)
        var bs = _build_bside(b, lb, None, None, autojunk != 0)
        for c in range(Int(nc)):
            var ca0 = Int(cands_off[unsafe_offset=c])
            var la = Int(cands_off[unsafe_offset=c + 1]) - ca0
            var a = cands.unsafe_offset(ca0)
            var t = la + lb
            var idx = w * Int(nc) + c
            pass_out[unsafe_offset=idx] = 0
            # Tier 1: real_quick_ratio — 2.0*min(la, lb) / (la + lb).
            var rqr = _calc_ratio(la if la < lb else lb, t)
            if rqr < cutoff:
                continue
            # Tier 2: quick_ratio — multiset intersection.
            var qr = _calc_ratio(_quick_matches(bs, a, la), t)
            if qr < cutoff:
                continue
            # Tier 3: full ratio from the matching blocks.
            var sink = List[Int]()
            var m = _blocks(bs, a, la, b, lb, sink, False)
            var ratio = _calc_ratio(m, t)
            if ratio >= cutoff:
                pass_out[unsafe_offset=idx] = 1
                ratio_out[unsafe_offset=idx] = ratio
        bs.free()
    return 0
