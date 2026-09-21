"""Clean-room BM25 scoring kernel (Okapi / BM25L / BM25Plus).

Written fresh from the textbook BM25 family of ranking functions
(Trotman et al., "Improvements to BM25 and Language Models Examined").
No third-party Mojo code is used or adapted.

Exported C ABI v3 (batch-shaped: the index is built once from doc-major
term-frequency streams, then each call scores one whole token-id query — or
one whole batch of queries — against the whole corpus):

    int32_t  bm25mojo_abi_version(void)              -> 3
    void*    bm25mojo_index_create(n_docs, doc_len, avgdl, k1, b, delta,
                                   variant, n_terms, doc_offsets, doc_tids,
                                   doc_freqs, idf)
    int32_t  bm25mojo_vocab_attach(handle, toks, width, n_terms)
    int32_t  bm25mojo_score(handle, qids, n_query, out_scores)
    int32_t  bm25mojo_score_batch(handle, qids_flat, query_offsets,
                                  n_queries, out_panel)
    int32_t  bm25mojo_score_batch_str(handle, toks, width, counts,
                                      n_queries, out_panel)
    void     bm25mojo_index_destroy(handle)

ABI v2 changes versus v1 (motivated by benchmarks/AUTOPSY-bm25s.md):

* Index-time baking. The kernel builds the CSR postings itself and bakes the
  full per-posting contribution `idf * tf-component` into the CSR values, in
  the same IEEE-754 float64 operation order the v1 kernel used at query time
  (which itself matched the reference NumPy expression order). Query-time
  scoring of Okapi/BM25L is then a pure gather-add — bit-identical results,
  with no division, no doc_len gather, and no idf lookup on the hot path.
* No score-buffer zeroing in the kernel. The caller passes a zeroed buffer
  (numpy calloc), so per-query cost is O(postings), not O(n_docs).
* A batch entry point scores many queries in one FFI call; each row is
  computed exactly as the single-query path computes it (bit-identical).

Score formulas (reference operation order, PyPI rank_bm25 0.2.2):

    Okapi: score[d] += idf * (qf * (k1 + 1) / (qf + k1 * (1 - b + b * dl / avgdl)))
    Plus:  score[d] += idf * (delta + qf * (k1 + 1) / (k1 * (1 - b + b * dl / avgdl) + qf))
    L:     score[d] += idf * qf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)
                            with ctd = qf / (1 - b + b * dl / avgdl)

BM25Plus gives even unposted documents a per-term floor idf * delta. The
baked posting value is the posted excess `full - floor`; per query the floor
sum F (accumulated in query-token order) is written with one dense SIMD fill,
and posted excess is gathered on top. When idf * delta is not finite
(|idf * delta| overflows to +-inf or is NaN), the reference's per-document
value for that term is the same non-finite constant for EVERY document
(posted or not, because delta + frac rounds to delta), so the term is flagged
at index time and contributes that constant via the dense fill; its postings
are skipped exactly as the reference's single evaluation requires.

ABI v3 additions (motivated by the live-Space Xeon arena: at 5k docs the
batch call is dominated by the Python vocab dict's cache footprint — the
mapping itself is microseconds, the evicted dict's misses are not):

* An optional native vocabulary map. `bm25mojo_vocab_attach` hands the index
  its terms as fixed-width NUL-padded byte slots ('S' array layout, terms in
  id order) once, and the kernel builds an open-addressing FNV-1a hash map
  (a few hundred KB of dense C arrays instead of a megabyte-scale Python
  dict of per-object pointers).
* `bm25mojo_score_batch_str` then takes the query batch as one fixed-width
  byte-slot array plus per-query token counts: token -> id mapping, per-query
  offset accumulation and scoring all happen natively in a single FFI call.
  Tokens not found in the map contribute exactly 0 and are skipped — the same
  result the id-mapping path computes by dropping them, bit-identical. A slot
  whose bytes after the first NUL are not all zero cannot be a NUL-padded
  fixed-width string (it is an embedded-NUL token the map cannot represent
  safely); the call then fails with status 3 and the wrapper falls back to
  its id-mapping path for that batch.
"""

from std.math import isfinite
from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 3

# Fixed-width 'S' token slots wider than this are refused (the Python wrapper
# falls back to its id-mapping path for such batches).
comptime MAX_TOKEN_WIDTH: Int64 = 4096

# Score-formula variants. The idf values always come from the caller.
comptime VARIANT_OKAPI: Int32 = 0
comptime VARIANT_L: Int32 = 1
comptime VARIANT_PLUS: Int32 = 2

# Native SIMD width for float64 on the build target (2 on NEON, 4 on AVX2).
comptime WIDTH = simd_width_of[DType.float64]()

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


struct BM25Index(Copyable, Movable):
    """Owned native index: CSR postings with baked float64 contributions."""

    var n_docs: Int64
    var avgdl: Float64
    var k1: Float64
    var b: Float64
    var delta: Float64
    var variant: Int32
    var n_terms: Int64
    var nnz: Int64
    var offsets: I64Ptr  # [n_terms + 1] CSR row offsets
    var docs: I32Ptr  # [nnz] posting doc ids (ascending per term)
    var weights: F64Ptr  # [nnz] baked contribution (full - floor), see module docs
    var idf: F64Ptr  # [n_terms]
    var overflow: Pointer[Bool, MutUntrackedOrigin]  # [n_terms] Plus only
    var floor0: F64Ptr  # [n_terms] Plus only: idf * delta per term
    # Optional native vocabulary map (ABI v3, bm25mojo_vocab_attach).
    var vocab_keys: Pointer[UInt8, MutUntrackedOrigin]  # [vocab_cap * vocab_width]
    var vocab_lens: I32Ptr  # [vocab_cap] token byte length (0 = empty slot)
    var vocab_tids: I32Ptr  # [vocab_cap] term id
    var vocab_cap: Int64  # power-of-two table capacity (0 = not attached)
    var vocab_width: Int64  # fixed slot width of the attached vocabulary

    def __init__(
        out self,
        n_docs: Int64,
        avgdl: Float64,
        k1: Float64,
        b: Float64,
        delta: Float64,
        variant: Int32,
        n_terms: Int64,
        nnz: Int64,
        offsets: I64Ptr,
        docs: I32Ptr,
        weights: F64Ptr,
        idf: F64Ptr,
        overflow: Pointer[Bool, MutUntrackedOrigin],
        floor0: F64Ptr,
    ):
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.variant = variant
        self.n_terms = n_terms
        self.nnz = nnz
        self.offsets = offsets
        self.docs = docs
        self.weights = weights
        self.idf = idf
        self.overflow = overflow
        self.floor0 = floor0
        # Placeholder pointer targets: the vocab fields are only ever
        # dereferenced when vocab_cap != 0 (every access is guarded), so any
        # valid pointer does as the "not attached" initial value — Mojo
        # pointers are non-nullable by design.
        self.vocab_keys = offsets.unsafe_bitcast[UInt8]()
        self.vocab_lens = offsets.unsafe_bitcast[Int32]()
        self.vocab_tids = offsets.unsafe_bitcast[Int32]()
        self.vocab_cap = 0
        self.vocab_width = 0


def _term_score[
    width: Int
](
    variant: Int32,
    qf: SIMD[DType.float64, width],
    dl: SIMD[DType.float64, width],
    idf: Float64,
    k1: Float64,
    b: Float64,
    avgdl: Float64,
    delta: Float64,
) -> SIMD[DType.float64, width]:
    """One BM25 term contribution per lane, in reference operation order."""
    var norm = SIMD[DType.float64, width](1.0 - b) + (
        SIMD[DType.float64, width](b) * dl
    ) / SIMD[DType.float64, width](avgdl)
    if variant == VARIANT_PLUS:
        var den_p = SIMD[DType.float64, width](k1) * norm + qf
        return SIMD[DType.float64, width](idf) * (
            SIMD[DType.float64, width](delta)
            + (qf * SIMD[DType.float64, width](k1 + 1.0)) / den_p
        )
    elif variant == VARIANT_L:
        # PyPI rank_bm25 0.2.2 BM25L semantics: the leading qf factor zeroes
        # unposted documents, so there is no dense floor for this variant.
        var ctd = qf / norm
        return (
            SIMD[DType.float64, width](idf)
            * qf
            * SIMD[DType.float64, width](k1 + 1.0)
            * (ctd + SIMD[DType.float64, width](delta))
        ) / (
            SIMD[DType.float64, width](k1)
            + ctd
            + SIMD[DType.float64, width](delta)
        )
    else:  # VARIANT_OKAPI
        var den = qf + SIMD[DType.float64, width](k1) * norm
        return SIMD[DType.float64, width](idf) * (
            (qf * SIMD[DType.float64, width](k1 + 1.0)) / den
        )


@export
def bm25mojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def bm25mojo_index_create(
    n_docs: Int64,
    doc_len: F64Ptr,
    avgdl: Float64,
    k1: Float64,
    b: Float64,
    delta: Float64,
    variant: Int32,
    n_terms: Int64,
    doc_offsets: I64Ptr,
    doc_tids: I32Ptr,
    doc_freqs: F64Ptr,
    idf: F64Ptr,
) abi("C") -> Handle:
    """Build a baked CSR index from doc-major term streams; NULL on invalid input.

    `doc_offsets` has n_docs + 1 ascending entries into `doc_tids`/`doc_freqs`
    (the unique terms of each document, any order). All buffers are validated
    and copied; the caller may free them on return.
    """
    if (
        n_docs <= 0
        or n_terms < 0
        or variant < VARIANT_OKAPI
        or variant > VARIANT_PLUS
    ):
        return None
    var ntok = doc_offsets[unsafe_offset=Int(n_docs)]
    if ntok < 0:
        return None

    # --- pass 1: document frequencies (also validates term ids) ---
    var df = unsafe_alloc[Int64](Int(n_terms) + 1)
    for t in range(Int(n_terms) + 1):
        df[unsafe_offset=t] = 0
    var prev_off = Int64(0)
    for d in range(Int(n_docs)):
        var off = doc_offsets[unsafe_offset=d]
        if off != prev_off:
            df.unsafe_free()
            return None  # offsets must be contiguous and ascending
        var end = doc_offsets[unsafe_offset=d + 1]
        if end < off or end > ntok:
            df.unsafe_free()
            return None
        prev_off = end
        for e in range(Int(off), Int(end)):
            var t = Int(doc_tids[unsafe_offset=e])
            if t < 0 or t >= Int(n_terms):
                df.unsafe_free()
                return None
            df[unsafe_offset=t] += 1

    # --- CSR offsets (exclusive prefix sum over df) ---
    var offsets = unsafe_alloc[Int64](Int(n_terms) + 1)
    offsets[unsafe_offset=0] = 0
    for t in range(Int(n_terms)):
        offsets[unsafe_offset=t + 1] = offsets[unsafe_offset=t] + df[
            unsafe_offset=t
        ]
    var nnz = offsets[unsafe_offset=Int(n_terms)]
    if nnz != ntok:
        df.unsafe_free()
        offsets.unsafe_free()
        return None  # ntok must equal the total posting count

    # --- scatter docs into CSR (ascending per term: docs arrive in order) ---
    var docs = unsafe_alloc[Int32](Int(nnz))
    var cursor = unsafe_alloc[Int64](Int(n_terms))
    for t in range(Int(n_terms)):
        cursor[unsafe_offset=t] = offsets[unsafe_offset=t]
    for d in range(Int(n_docs)):
        var e = Int(doc_offsets[unsafe_offset=d])
        var end = Int(doc_offsets[unsafe_offset=d + 1])
        while e < end:
            var t = Int(doc_tids[unsafe_offset=e])
            var pos = cursor[unsafe_offset=t]
            cursor[unsafe_offset=t] = pos + 1
            docs[unsafe_offset=Int(pos)] = Int32(d)
            e += 1
    cursor.unsafe_free()

    # --- per-term floors (Plus) and overflow flags ---
    var idf_copy = unsafe_alloc[Float64](Int(n_terms))
    if n_terms > 0:
        unsafe_memcpy(dest=idf_copy, src=idf, count=Int(n_terms))
    var floor0 = unsafe_alloc[Float64](Int(n_terms))
    var overflow = unsafe_alloc[Bool](Int(n_terms))
    for t in range(Int(n_terms)):
        var f = Float64(0.0)
        if variant == VARIANT_PLUS:
            f = idf_copy[unsafe_offset=t] * delta
        floor0[unsafe_offset=t] = f
        overflow[unsafe_offset=t] = not isfinite(f)

    # --- bake weights = full - floor per posting (reference op order) ---
    # Copy raw frequencies into a temp CSR-ordered buffer (reusing df as the
    # scatter cursor, reset to the row offsets), then evaluate the
    # contribution per posting with SIMD over each term's contiguous run.
    var freqs = unsafe_alloc[Float64](Int(nnz))
    for t in range(Int(n_terms)):
        df[unsafe_offset=t] = offsets[unsafe_offset=t]
    for d in range(Int(n_docs)):
        var e = Int(doc_offsets[unsafe_offset=d])
        var end = Int(doc_offsets[unsafe_offset=d + 1])
        while e < end:
            var t = Int(doc_tids[unsafe_offset=e])
            freqs[unsafe_offset=Int(df[unsafe_offset=t])] = doc_freqs[
                unsafe_offset=e
            ]
            df[unsafe_offset=t] += 1
            e += 1
    df.unsafe_free()
    var weights = unsafe_alloc[Float64](Int(nnz))
    for t in range(Int(n_terms)):
        var idf_t = idf_copy[unsafe_offset=t]
        var floor_t = floor0[unsafe_offset=t]
        var j = Int(offsets[unsafe_offset=t])
        var end = Int(offsets[unsafe_offset=t + 1])
        while j + WIDTH <= end:
            var qf = freqs.unsafe_load[width=WIDTH](j)
            var dl = SIMD[DType.float64, WIDTH]()
            for lane in range(WIDTH):
                dl[lane] = doc_len[unsafe_offset=Int(docs[unsafe_offset=j + lane])]
            var w = _term_score[WIDTH](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            ) - SIMD[DType.float64, WIDTH](floor_t)
            weights.unsafe_store(j, w)
            j += WIDTH
        while j < end:
            var qf = SIMD[DType.float64, 1](freqs[unsafe_offset=j])
            var dl = SIMD[DType.float64, 1](
                doc_len[unsafe_offset=Int(docs[unsafe_offset=j])]
            )
            var w = _term_score[1](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            ) - SIMD[DType.float64, 1](floor_t)
            weights[unsafe_offset=j] = w[0]
            j += 1
    freqs.unsafe_free()

    var idx = unsafe_alloc[BM25Index](1)
    idx[] = BM25Index(
        n_docs,
        avgdl,
        k1,
        b,
        delta,
        variant,
        n_terms,
        nnz,
        offsets,
        docs,
        weights,
        idf_copy,
        overflow,
        floor0,
    )
    return idx.unsafe_bitcast[UInt8]()


def _score_one(
    idx: Pointer[BM25Index, MutUntrackedOrigin],
    qids: I32Ptr,
    n_query: Int64,
    out_scores: F64Ptr,
) -> Int32:
    """Accumulate one query into `out_scores` (must be pre-zeroed)."""
    var n_docs = Int(idx[].n_docs)
    var variant = idx[].variant
    var n_terms = Int(idx[].n_terms)
    var offsets = idx[].offsets
    var docs = idx[].docs
    var weights = idx[].weights
    var overflow = idx[].overflow
    var floor0 = idx[].floor0

    if variant == VARIANT_PLUS:
        # Dense floor: F = sum of per-term floors in query-token order, then
        # one write-only SIMD fill (bit-identical to zero + F). Overflowing
        # terms contribute their non-finite constant here; their postings are
        # skipped below because the reference's per-document value for them
        # is that same constant everywhere.
        var floor_total = Float64(0.0)
        for qi in range(Int(n_query)):
            var t = Int(qids[unsafe_offset=qi])
            if t < 0 or t >= n_terms:
                return 2
            floor_total += floor0[unsafe_offset=t]
        var fill = SIMD[DType.float64, WIDTH](floor_total)
        var d = 0
        while d + WIDTH <= n_docs:
            out_scores.unsafe_store(d, fill)
            d += WIDTH
        while d < n_docs:
            out_scores[unsafe_offset=d] = floor_total
            d += 1
        for qi in range(Int(n_query)):
            var t = Int(qids[unsafe_offset=qi])
            if overflow[unsafe_offset=t]:
                continue
            var j = Int(offsets[unsafe_offset=t])
            var end = Int(offsets[unsafe_offset=t + 1])
            while j < end:
                out_scores[unsafe_offset=Int(docs[unsafe_offset=j])] += weights[
                    unsafe_offset=j
                ]
                j += 1
    else:
        for qi in range(Int(n_query)):
            var t = Int(qids[unsafe_offset=qi])
            if t < 0 or t >= n_terms:
                return 2
            var j = Int(offsets[unsafe_offset=t])
            var end = Int(offsets[unsafe_offset=t + 1])
            while j < end:
                out_scores[unsafe_offset=Int(docs[unsafe_offset=j])] += weights[
                    unsafe_offset=j
                ]
                j += 1
    return 0


@export
def bm25mojo_score(
    handle: Handle,
    qids: I32Ptr,
    n_query: Int64,
    out_scores: F64Ptr,
) abi("C") -> Int32:
    """Score one query over the whole corpus into the PRE-ZEROED `out_scores`.

    The kernel does not zero the buffer (the caller's numpy calloc already
    guarantees zeros, so per-query cost is O(postings), not O(n_docs)).
    Returns 0 on success, 1 on a NULL handle, 2 on an out-of-range term id
    or query length.
    """
    if not handle:
        return 1
    if n_query < 0:
        return 2
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    return _score_one(idx, qids, n_query, out_scores)


@export
def bm25mojo_score_batch(
    handle: Handle,
    qids_flat: I32Ptr,
    query_offsets: I64Ptr,
    n_queries: Int64,
    out_panel: F64Ptr,
) abi("C") -> Int32:
    """Score a whole batch of queries in one FFI call.

    `qids_flat`/`query_offsets` are the concatenated per-query term-id lists
    (offsets has n_queries + 1 entries); `out_panel` is a PRE-ZEROED
    n_queries x n_docs float64 panel, row-major. Each row is computed exactly
    as `bm25mojo_score` computes it, so batch results are bit-identical to
    per-query calls.
    """
    if not handle:
        return 1
    if n_queries < 0:
        return 2
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    var n_docs = idx[].n_docs
    var prev = Int64(0)
    for q in range(Int(n_queries)):
        var start = query_offsets[unsafe_offset=q]
        var end = query_offsets[unsafe_offset=q + 1]
        if start != prev or end < start:
            return 2
        prev = end
        var rc = _score_one(
            idx,
            qids_flat.unsafe_offset(Int(start)),
            end - start,
            out_panel.unsafe_offset(Int(Int64(q) * n_docs)),
        )
        if rc != 0:
            return rc
    return 0


# ---------------------------------------------------------------------------
# ABI v3: native vocabulary map + string-token batch scoring
# ---------------------------------------------------------------------------


def _fnv1a(bytes: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt64:
    """FNV-1a 64-bit over the token's content bytes."""
    var h = UInt64(14695981039346656037)
    for i in range(n):
        h ^= UInt64(Int(bytes[unsafe_offset=i]))
        h *= UInt64(1099511628211)
    return h


def _slot_scan(slot: Pointer[UInt8, MutUntrackedOrigin], width: Int) -> Int:
    """Content length of a NUL-padded fixed-width slot, or -1 when unsafe.

    The content ends at the first NUL (or spans the slot). Any nonzero byte
    after the first NUL means the token contains an embedded NUL, which the
    native map cannot represent without ambiguity; the caller must refuse it
    (the wrapper falls back to its id-mapping path for that batch/vocab).
    """
    var first_zero = -1
    for i in range(width):
        if slot[unsafe_offset=i] == 0:
            first_zero = i
            break
    if first_zero < 0:
        return width
    for i in range(first_zero + 1, width):
        if slot[unsafe_offset=i] != 0:
            return -1
    return first_zero


def _vocab_lookup(
    idx: Pointer[BM25Index, MutUntrackedOrigin],
    tok: Pointer[UInt8, MutUntrackedOrigin],
    length: Int,
) -> Int32:
    """Token -> term id via the attached map; -1 when absent (unseen term)."""
    var cap = idx[].vocab_cap
    var lens = idx[].vocab_lens
    var tids = idx[].vocab_tids
    var keys = idx[].vocab_keys
    var width = Int(idx[].vocab_width)
    var mask = Int(cap - 1)
    var h = _fnv1a(tok, length)
    var i = Int(h % UInt64(cap))
    while lens[unsafe_offset=i] != 0:
        if Int(lens[unsafe_offset=i]) == length:
            var key = keys.unsafe_offset(i * width)
            var same = True
            for k in range(length):
                if key[unsafe_offset=k] != tok[unsafe_offset=k]:
                    same = False
                    break
            if same:
                return tids[unsafe_offset=i]
        i = (i + 1) & mask
    return -1


@export
def bm25mojo_vocab_attach(
    handle: Handle,
    toks: Pointer[UInt8, MutUntrackedOrigin],
    width: Int64,
    n_terms: Int64,
) abi("C") -> Int32:
    """Attach the index vocabulary and build the native token->id map.

    `toks` is n_terms fixed-width NUL-padded byte slots ('S' array layout)
    listing every term in id order. Returns 0 on success, 1 on a NULL handle,
    2 on invalid arguments, 3 on an embedded-NUL (unmappable) vocab token,
    4 when a map is already attached.
    """
    if not handle:
        return 1
    if width <= 0 or width > MAX_TOKEN_WIDTH or n_terms < 0:
        return 2
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    if idx[].vocab_cap != 0:
        return 4
    if n_terms != idx[].n_terms:
        return 2
    var cap = Int64(16)
    while cap < n_terms * 2:
        cap *= 2
    var lens = unsafe_alloc[Int32](Int(cap))
    for i in range(Int(cap)):
        lens[unsafe_offset=i] = 0
    var tids = unsafe_alloc[Int32](Int(cap))
    var keys = unsafe_alloc[UInt8](Int(cap * width))
    var mask = Int(cap - 1)
    for t in range(Int(n_terms)):
        var slot = toks.unsafe_offset(Int(Int64(t) * width))
        var L = _slot_scan(slot, Int(width))
        if L < 0:
            keys.unsafe_free()
            lens.unsafe_free()
            tids.unsafe_free()
            return 3
        var h = _fnv1a(slot, L)
        var i = Int(h % UInt64(cap))
        while lens[unsafe_offset=i] != 0:
            i = (i + 1) & mask
        var dst = keys.unsafe_offset(i * Int(width))
        for k in range(Int(width)):
            dst[unsafe_offset=k] = slot[unsafe_offset=k]
        lens[unsafe_offset=i] = Int32(L)
        tids[unsafe_offset=i] = Int32(t)
    idx[].vocab_keys = keys
    idx[].vocab_lens = lens
    idx[].vocab_tids = tids
    idx[].vocab_cap = cap
    idx[].vocab_width = width
    return 0


def _accumulate(
    idx: Pointer[BM25Index, MutUntrackedOrigin],
    t: Int,
    out_scores: F64Ptr,
):
    """Add one term's baked posting contributions into `out_scores`."""
    var j = Int(idx[].offsets[unsafe_offset=t])
    var end = Int(idx[].offsets[unsafe_offset=t + 1])
    var docs = idx[].docs
    var weights = idx[].weights
    while j < end:
        out_scores[unsafe_offset=Int(docs[unsafe_offset=j])] += weights[
            unsafe_offset=j
        ]
        j += 1


@export
def bm25mojo_score_batch_str(
    handle: Handle,
    toks: Pointer[UInt8, MutUntrackedOrigin],
    width: Int64,
    counts: I32Ptr,
    n_queries: Int64,
    out_panel: F64Ptr,
) abi("C") -> Int32:
    """Score a batch of queries given as fixed-width 'S' token slots.

    `toks` holds the concatenated per-query token slots (each `width` bytes,
    NUL-padded); `counts[q]` is query q's token count; `out_panel` is a
    PRE-ZEROED n_queries x n_docs float64 panel. Tokens absent from the
    attached map contribute exactly 0 (skipped), and each row is computed in
    query-token order — both exactly as the id-mapping path computes them, so
    results are bit-identical to `bm25mojo_score_batch` over the mapped ids.
    Returns 0 on success, 1 on a NULL handle, 2 on invalid arguments, 3 on an
    embedded-NUL (unmappable) query token, 4 when no map is attached.
    """
    if not handle:
        return 1
    if n_queries < 0 or width <= 0 or width > MAX_TOKEN_WIDTH:
        return 2
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    if idx[].vocab_cap == 0:
        return 4
    var n_docs = Int(idx[].n_docs)
    var variant = idx[].variant
    var overflow = idx[].overflow
    var floor0 = idx[].floor0
    var base = Int64(0)
    for q in range(Int(n_queries)):
        var cnt = Int(counts[unsafe_offset=q])
        if cnt < 0:
            return 2
        var row = out_panel.unsafe_offset(q * n_docs)
        if variant == VARIANT_PLUS:
            # Floor sum over FOUND tokens only, in query-token order (exactly
            # the filtered-id order of the v2 path), then one dense SIMD fill
            # and the posted excess on top.
            var floor_total = Float64(0.0)
            for j in range(cnt):
                var slot = toks.unsafe_offset(Int((base + Int64(j)) * width))
                var L = _slot_scan(slot, Int(width))
                if L < 0:
                    return 3
                var t = _vocab_lookup(idx, slot, L)
                if t >= 0:
                    floor_total += floor0[unsafe_offset=Int(t)]
            var fill = SIMD[DType.float64, WIDTH](floor_total)
            var d = 0
            while d + WIDTH <= n_docs:
                row.unsafe_store(d, fill)
                d += WIDTH
            while d < n_docs:
                row[unsafe_offset=d] = floor_total
                d += 1
            for j in range(cnt):
                var slot = toks.unsafe_offset(Int((base + Int64(j)) * width))
                var L = _slot_scan(slot, Int(width))
                var t = _vocab_lookup(idx, slot, L)
                if t < 0 or overflow[unsafe_offset=Int(t)]:
                    continue
                _accumulate(idx, Int(t), row)
        else:
            for j in range(cnt):
                var slot = toks.unsafe_offset(Int((base + Int64(j)) * width))
                var L = _slot_scan(slot, Int(width))
                if L < 0:
                    return 3
                var t = _vocab_lookup(idx, slot, L)
                if t < 0:
                    continue
                _accumulate(idx, Int(t), row)
        base += Int64(cnt)
    return 0


@export
def bm25mojo_index_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    idx[].offsets.unsafe_free()
    idx[].docs.unsafe_free()
    idx[].weights.unsafe_free()
    idx[].idf.unsafe_free()
    idx[].overflow.unsafe_free()
    idx[].floor0.unsafe_free()
    if idx[].vocab_cap != 0:
        idx[].vocab_keys.unsafe_free()
        idx[].vocab_lens.unsafe_free()
        idx[].vocab_tids.unsafe_free()
    idx.unsafe_free()
