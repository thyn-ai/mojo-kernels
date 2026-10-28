"""Clean-room BM25 scoring kernel (Okapi / BM25L / BM25Plus).

Written fresh from the textbook BM25 family of ranking functions
(Trotman et al., "Improvements to BM25 and Language Models Examined").
No third-party Mojo code is used or adapted.

Exported C ABI v4:

    int32_t  bm25mojo_abi_version(void)              -> 4
    void*    bm25mojo_build_index_corpus(n_docs, tokens, doc_offsets,
                                         variant, k1, b, delta, epsilon)
    void*    bm25mojo_build_index_tids(n_docs, token_tids, doc_offsets,
                                       variant, k1, b, delta, epsilon)
    void     bm25mojo_build_result_free(result)
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

ABI v4 adds native corpus ingestion. The wrapper can hand the kernel either a
raw UTF-8 token stream (``bm25mojo_build_index_corpus``) or a pre-tokenized
stream of term ids (``bm25mojo_build_index_tids``). The term-id path moves
vocabulary assignment to Python (where hashable-token dicts are extremely fast)
and lets the kernel focus on frequency counting and CSR preparation, which is
the faster shape for the typical str-token corpus.

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
and posted excess is gathered on top. That decomposition is the reference's
sum of per-term values `idf * (delta + frac)` reassociated as
`(sum of floors) + sum of (full - floor)`: identical in exact arithmetic, and
identical in IEEE-754 float64 whenever no intermediate in either order
overflows. Reassociation is NOT overflow-safe, though: the floor sum can
overflow to +-inf while the reference's interleaved sum stays finite (or hits
the opposite infinity, making the kernel's fill + excess a NaN where the
reference is +-inf). The kernel therefore bounds the magnitudes before
scoring a query the fast way: with `bound[t] = |floor_t| + max_j |excess_j|`
per term (any non-finite piece poisoning the bound to +inf), a query whose
bound sum (in query-token order) reaches SAFE_BOUND (= Float64 max / 2) is
scored term by term in the reference's own order instead — every document
receives the term's value (the baked `full` where posted, the floor
elsewhere) in query-token order, bit-identical to the reference's dense
evaluation. Below the bound every partial sum of either order stays below
SAFE_BOUND, so neither order can produce an infinity or NaN the other does
not; the only remaining difference between the orders is rounding (the
ill-conditioned-norm known issue). In-domain parameters never reach the
bound (floors and excesses are single-digit), so the fast path always runs
there. Where the reference itself evaluates 0/0 for a document without the
term (the degenerate-nan known issue: k1 * norm == 0), the precise path still
adds the floor; those inputs are exactly the harness's degenerate class.

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

from std.math import inf, isfinite
from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 4

# Fast-path magnitude bound for the BM25Plus floor decomposition (see module
# docs): Float64 max / 2. A query whose per-term |floor| + max|excess| sum
# stays below it cannot overflow in either evaluation order.
comptime SAFE_BOUND: Float64 = 8.9884656743115795e307

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
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


# ---------------------------------------------------------------------------
# ABI v4: native corpus ingestion
# ---------------------------------------------------------------------------


struct BuildResult(Copyable, Movable):
    """Output of bm25mojo_build_index_corpus; caller copies then frees."""

    var status: Int32
    var n_docs: Int64
    var n_terms: Int64
    var nnz: Int64
    var avgdl: Float64
    var doc_len: F64Ptr
    var doc_offsets: I64Ptr
    var doc_tids: I32Ptr
    var doc_freqs: F64Ptr
    var df: I64Ptr
    var vocab_buf: U8Ptr
    var vocab_lens: I32Ptr

    def __init__(
        out self,
        status: Int32,
        n_docs: Int64,
        n_terms: Int64,
        nnz: Int64,
        avgdl: Float64,
        doc_len: F64Ptr,
        doc_offsets: I64Ptr,
        doc_tids: I32Ptr,
        doc_freqs: F64Ptr,
        df: I64Ptr,
        vocab_buf: U8Ptr,
        vocab_lens: I32Ptr,
    ):
        self.status = status
        self.n_docs = n_docs
        self.n_terms = n_terms
        self.nnz = nnz
        self.avgdl = avgdl
        self.doc_len = doc_len
        self.doc_offsets = doc_offsets
        self.doc_tids = doc_tids
        self.doc_freqs = doc_freqs
        self.df = df
        self.vocab_buf = vocab_buf
        self.vocab_lens = vocab_lens


def _next_pow2(n: Int64) -> Int64:
    if n <= 1:
        return 1
    var p = Int64(1)
    while p < n:
        p <<= 1
    return p


def _fnv1a_bytes(bytes: U8Ptr, n: Int) -> UInt64:
    """FNV-1a 64-bit over `n` bytes."""
    var h = UInt64(14695981039346656037)
    for i in range(n):
        h ^= UInt64(Int(bytes[unsafe_offset=i]))
        h *= UInt64(1099511628211)
    return h


def _token_equal(
    tokens: U8Ptr,
    off_a: Int64,
    len_a: Int,
    off_b: Int64,
    len_b: Int,
) -> Bool:
    if len_a != len_b:
        return False
    for k in range(len_a):
        if tokens[unsafe_offset=Int(off_a + Int64(k))] != tokens[
            unsafe_offset=Int(off_b + Int64(k))
        ]:
            return False
    return True


struct BuildScratch:
    """Transient state for the two-pass native corpus builder."""

    var cap: Int64
    var used: Pointer[Int8, MutUntrackedOrigin]
    var keys_off: I64Ptr
    var keys_len: I32Ptr
    var values: I32Ptr
    var term_off: I64Ptr
    var term_len: I32Ptr
    var token_tids: I32Ptr
    var df: I64Ptr
    var count: I64Ptr

    def __init__(
        out self,
        cap: Int64,
        total_tokens: Int64,
        n_docs: Int64,
    ):
        self.cap = cap
        self.used = unsafe_alloc[Int8](Int(cap))
        self.keys_off = unsafe_alloc[Int64](Int(cap))
        self.keys_len = unsafe_alloc[Int32](Int(cap))
        self.values = unsafe_alloc[Int32](Int(cap))
        self.term_off = unsafe_alloc[Int64](Int(total_tokens))
        self.term_len = unsafe_alloc[Int32](Int(total_tokens))
        self.token_tids = unsafe_alloc[Int32](Int(total_tokens))
        self.df = unsafe_alloc[Int64](Int(total_tokens))
        self.count = unsafe_alloc[Int64](Int(total_tokens))
        for i in range(Int(cap)):
            self.used[unsafe_offset=i] = 0

    def free(self):
        self.used.unsafe_free()
        self.keys_off.unsafe_free()
        self.keys_len.unsafe_free()
        self.values.unsafe_free()
        self.term_off.unsafe_free()
        self.term_len.unsafe_free()
        self.token_tids.unsafe_free()
        self.df.unsafe_free()
        self.count.unsafe_free()


def _lookup_or_insert_bytes(
    tokens: U8Ptr,
    off: Int64,
    L: Int,
    mut scratch: BuildScratch,
    mut n_terms: Int64,
) -> Int32:
    """Map one token byte slice to a term id, creating the entry if needed."""
    var h = _fnv1a_bytes(tokens.unsafe_offset(Int(off)), L)
    var mask = Int(scratch.cap - 1)
    var i = Int(h % UInt64(scratch.cap))
    while True:
        if scratch.used[unsafe_offset=i] == 0:
            scratch.used[unsafe_offset=i] = 1
            scratch.keys_off[unsafe_offset=i] = off
            scratch.keys_len[unsafe_offset=i] = Int32(L)
            scratch.values[unsafe_offset=i] = Int32(n_terms)
            scratch.term_off[unsafe_offset=Int(n_terms)] = off
            scratch.term_len[unsafe_offset=Int(n_terms)] = Int32(L)
            n_terms += 1
            return scratch.values[unsafe_offset=i]
        if Int(scratch.keys_len[unsafe_offset=i]) == L:
            if _token_equal(
                tokens,
                scratch.keys_off[unsafe_offset=i],
                L,
                off,
                L,
            ):
                return scratch.values[unsafe_offset=i]
        i = (i + 1) & mask


def _build_index_corpus_impl(
    n_docs: Int64,
    tokens: U8Ptr,
    doc_offsets: I64Ptr,
    variant: Int32,
    k1: Float64,
    b: Float64,
    delta: Float64,
    epsilon: Float64,
) -> Handle:
    if (
        n_docs <= 0
        or variant < VARIANT_OKAPI
        or variant > VARIANT_PLUS
    ):
        return None

    var total_bytes = doc_offsets[unsafe_offset=Int(n_docs)]
    if total_bytes < 0:
        return None

    # Validate byte-offset monotonicity.
    var prev = Int64(0)
    for d in range(Int(n_docs) + 1):
        var off = doc_offsets[unsafe_offset=d]
        if off < prev or off > total_bytes:
            return None
        prev = off

    # There cannot be more tokens than bytes, so size transient buffers for the
    # worst case and trim later. The hash table is sized for that same bound.
    var max_tokens = total_bytes
    if max_tokens < 1:
        max_tokens = 1
    var cap = _next_pow2(max_tokens * 2)
    if cap < 16:
        cap = 16
    var scratch = BuildScratch(cap, max_tokens, n_docs)

    var doc_token_offsets = unsafe_alloc[Int64](Int(n_docs) + 1)
    doc_token_offsets[unsafe_offset=0] = 0

    # Pass 1: read each document's length-prefixed UTF-8 tokens and assign term
    # ids in first-appearance order. doc_token_offsets records token boundaries.
    var n_terms = Int64(0)
    var total_tokens = Int64(0)
    for d in range(Int(n_docs)):
        var start = doc_offsets[unsafe_offset=d]
        var end = doc_offsets[unsafe_offset=d + 1]
        var i = start
        while i < end:
            if i + 4 > end:
                doc_token_offsets.unsafe_free()
                scratch.free()
                return None
            var L = Int(tokens[unsafe_offset=Int(i)])
            L |= Int(tokens[unsafe_offset=Int(i + 1)]) << 8
            L |= Int(tokens[unsafe_offset=Int(i + 2)]) << 16
            L |= Int(tokens[unsafe_offset=Int(i + 3)]) << 24
            var tok_start = i + 4
            var tok_end = tok_start + Int64(L)
            if tok_end > end or tok_end < tok_start:
                doc_token_offsets.unsafe_free()
                scratch.free()
                return None
            scratch.token_tids[unsafe_offset=Int(total_tokens)] = _lookup_or_insert_bytes(
                tokens, tok_start, L, scratch, n_terms
            )
            total_tokens += 1
            i = tok_end
        doc_token_offsets[unsafe_offset=d + 1] = total_tokens

    if n_terms == 0:
        # rank_bm25 raises ZeroDivisionError on an empty vocabulary; force the
        # Python wrapper down its reference build path by reporting failure.
        doc_token_offsets.unsafe_free()
        scratch.free()
        return None

    # Pass 2: per-document frequencies and document frequencies.
    for t in range(Int(n_terms)):
        scratch.df[unsafe_offset=t] = 0
        scratch.count[unsafe_offset=t] = 0

    var doc_len = unsafe_alloc[Float64](Int(n_docs))
    var doc_offsets_out = unsafe_alloc[Int64](Int(n_docs) + 1)
    var tids_capacity = Int(total_tokens)
    if tids_capacity < 1:
        tids_capacity = 1
    var doc_tids = unsafe_alloc[Int32](tids_capacity)
    var doc_freqs = unsafe_alloc[Float64](tids_capacity)
    var pos = Int64(0)
    doc_offsets_out[unsafe_offset=0] = 0

    for d in range(Int(n_docs)):
        var start = doc_token_offsets[unsafe_offset=d]
        var end = doc_token_offsets[unsafe_offset=d + 1]
        doc_len[unsafe_offset=d] = Float64(end - start)

        # Count frequencies; increment df on the first sighting in this doc.
        for e in range(Int(start), Int(end)):
            var t = Int(scratch.token_tids[unsafe_offset=e])
            if scratch.count[unsafe_offset=t] == 0:
                scratch.df[unsafe_offset=t] += 1
            scratch.count[unsafe_offset=t] += 1

        # Emit unique (tid, freq) pairs for this doc in first-appearance order.
        for e in range(Int(start), Int(end)):
            var t = Int(scratch.token_tids[unsafe_offset=e])
            var c = scratch.count[unsafe_offset=t]
            if c > 0:
                doc_tids[unsafe_offset=Int(pos)] = Int32(t)
                doc_freqs[unsafe_offset=Int(pos)] = Float64(c)
                pos += 1
                scratch.count[unsafe_offset=t] = 0
        doc_offsets_out[unsafe_offset=d + 1] = pos

    doc_token_offsets.unsafe_free()
    var nnz = pos

    # Compute avgdl.
    var total_len = Float64(0.0)
    for d in range(Int(n_docs)):
        total_len += doc_len[unsafe_offset=d]
    var avgdl = total_len / Float64(n_docs)

    # Copy document frequencies into the result. Idfs are intentionally left to
    # the Python wrapper, which evaluates them with the same ``math.log``
    # implementation as rank_bm25 and applies the Okapi epsilon floor; that is
    # the only way to keep the ``idf`` dict bit-identical with the reference.
    var df = unsafe_alloc[Int64](Int(n_terms))
    for t in range(Int(n_terms)):
        df[unsafe_offset=t] = scratch.df[unsafe_offset=t]

    # Pack vocabulary terms into a contiguous byte buffer.
    var vocab_bytes = Int64(0)
    for t in range(Int(n_terms)):
        vocab_bytes += Int64(scratch.term_len[unsafe_offset=t])
    var vocab_buf_size = Int(vocab_bytes)
    if vocab_buf_size < 1:
        vocab_buf_size = 1
    var vocab_buf = unsafe_alloc[UInt8](vocab_buf_size)
    var vocab_lens = unsafe_alloc[Int32](Int(n_terms))
    var write = 0
    for t in range(Int(n_terms)):
        var L = Int(scratch.term_len[unsafe_offset=t])
        var off = Int(scratch.term_off[unsafe_offset=t])
        for k in range(L):
            vocab_buf[unsafe_offset=write + k] = tokens[unsafe_offset=off + k]
        vocab_lens[unsafe_offset=t] = Int32(L)
        write += L

    scratch.free()

    var result = unsafe_alloc[BuildResult](1)
    result[] = BuildResult(
        Int32(0),
        n_docs,
        n_terms,
        nnz,
        avgdl,
        doc_len,
        doc_offsets_out,
        doc_tids,
        doc_freqs,
        df,
        vocab_buf,
        vocab_lens,
    )
    return result.unsafe_bitcast[UInt8]()


@export
def bm25mojo_build_index_corpus(
    n_docs: Int64,
    tokens: U8Ptr,
    doc_offsets: I64Ptr,
    variant: Int32,
    k1: Float64,
    b: Float64,
    delta: Float64,
    epsilon: Float64,
) abi("C") -> Handle:
    """Native corpus ingestion: length-prefixed UTF-8 token stream -> index data.

    `tokens` is the concatenation of every document's tokens, each encoded as a
    4-byte little-endian length followed by that many UTF-8 bytes; `doc_offsets`
    has n_docs + 1 byte offsets into `tokens`. On success returns a BuildResult
    pointer (status == 0) whose arrays the caller must copy and then free with
    bm25mojo_build_result_free. On any error returns NULL.
    """
    return _build_index_corpus_impl(
        n_docs, tokens, doc_offsets, variant, k1, b, delta, epsilon
    )


def _build_index_tids_impl(
    n_docs: Int64,
    token_tids: I32Ptr,
    doc_offsets: I64Ptr,
    variant: Int32,
    k1: Float64,
    b: Float64,
    delta: Float64,
    epsilon: Float64,
) -> Handle:
    """Native corpus ingestion from pre-assigned term ids.

    Python assigns term ids in first-appearance order, so the kernel only has
    to count per-document frequencies and document frequencies. No byte parsing,
    no hash table, no vocabulary buffer.
    """
    if (
        n_docs <= 0
        or variant < VARIANT_OKAPI
        or variant > VARIANT_PLUS
    ):
        return None

    var total_tokens = doc_offsets[unsafe_offset=Int(n_docs)]
    if total_tokens < 0:
        return None

    # Validate token-offset monotonicity.
    var prev = Int64(0)
    for d in range(Int(n_docs) + 1):
        var off = doc_offsets[unsafe_offset=d]
        if off < prev or off > total_tokens:
            return None
        prev = off

    # Discover n_terms from the provided tids (they must be dense 0..n_terms-1).
    var n_terms = Int64(0)
    for i in range(Int(total_tokens)):
        var t = Int(token_tids[unsafe_offset=i])
        if t < 0:
            return None
        if Int64(t) >= n_terms:
            n_terms = Int64(t) + 1

    if n_terms == 0:
        return None

    var doc_len = unsafe_alloc[Float64](Int(n_docs))
    var doc_offsets_out = unsafe_alloc[Int64](Int(n_docs) + 1)
    var tids_capacity = Int(total_tokens)
    if tids_capacity < 1:
        tids_capacity = 1
    var doc_tids = unsafe_alloc[Int32](tids_capacity)
    var doc_freqs = unsafe_alloc[Float64](tids_capacity)

    # Per-term transient counters (zeroed once, reused per document).
    var count = unsafe_alloc[Int64](Int(n_terms))
    for t in range(Int(n_terms)):
        count[unsafe_offset=t] = 0
    var df = unsafe_alloc[Int64](Int(n_terms))
    for t in range(Int(n_terms)):
        df[unsafe_offset=t] = 0

    var pos = Int64(0)
    doc_offsets_out[unsafe_offset=0] = 0

    for d in range(Int(n_docs)):
        var start = doc_offsets[unsafe_offset=d]
        var end = doc_offsets[unsafe_offset=d + 1]
        doc_len[unsafe_offset=d] = Float64(end - start)

        # Count frequencies; increment df on the first sighting in this doc.
        for e in range(Int(start), Int(end)):
            var t = Int(token_tids[unsafe_offset=e])
            if count[unsafe_offset=t] == 0:
                df[unsafe_offset=t] += 1
            count[unsafe_offset=t] += 1

        # Emit unique (tid, freq) pairs for this doc in first-appearance order.
        for e in range(Int(start), Int(end)):
            var t = Int(token_tids[unsafe_offset=e])
            var c = count[unsafe_offset=t]
            if c > 0:
                doc_tids[unsafe_offset=Int(pos)] = Int32(t)
                doc_freqs[unsafe_offset=Int(pos)] = Float64(c)
                pos += 1
                count[unsafe_offset=t] = 0
        doc_offsets_out[unsafe_offset=d + 1] = pos

    var nnz = pos

    # Compute avgdl.
    var total_len = Float64(0.0)
    for d in range(Int(n_docs)):
        total_len += doc_len[unsafe_offset=d]
    var avgdl = total_len / Float64(n_docs)

    # Dummy vocabulary buffer: the term-id path owns the vocabulary on the
    # Python side, so the kernel returns an empty buffer and zero-length terms.
    var vocab_buf = unsafe_alloc[UInt8](1)
    var vocab_lens = unsafe_alloc[Int32](Int(n_terms))
    for t in range(Int(n_terms)):
        vocab_lens[unsafe_offset=t] = 0

    count.unsafe_free()

    var result = unsafe_alloc[BuildResult](1)
    result[] = BuildResult(
        Int32(0),
        n_docs,
        n_terms,
        nnz,
        avgdl,
        doc_len,
        doc_offsets_out,
        doc_tids,
        doc_freqs,
        df,
        vocab_buf,
        vocab_lens,
    )
    return result.unsafe_bitcast[UInt8]()


@export
def bm25mojo_build_index_tids(
    n_docs: Int64,
    token_tids: I32Ptr,
    doc_offsets: I64Ptr,
    variant: Int32,
    k1: Float64,
    b: Float64,
    delta: Float64,
    epsilon: Float64,
) abi("C") -> Handle:
    """Native corpus ingestion: term-id stream -> index data.

    `token_tids` is the concatenation of every document's term ids (int32);
    `doc_offsets` has n_docs + 1 token offsets. Term ids must be a dense
    0..n_terms-1 range assigned in first-appearance order. On success returns a
    BuildResult pointer (status == 0) whose arrays the caller must copy and then
    free with bm25mojo_build_result_free. On any error returns NULL.
    """
    return _build_index_tids_impl(
        n_docs, token_tids, doc_offsets, variant, k1, b, delta, epsilon
    )


@export
def bm25mojo_build_result_free(result: Handle) abi("C"):
    """Free a BuildResult and the arrays it owns."""
    if not result:
        return
    var r = result.value().unsafe_bitcast[BuildResult]()
    r[].doc_len.unsafe_free()
    r[].doc_offsets.unsafe_free()
    r[].doc_tids.unsafe_free()
    r[].doc_freqs.unsafe_free()
    r[].df.unsafe_free()
    r[].vocab_buf.unsafe_free()
    r[].vocab_lens.unsafe_free()
    r.unsafe_free()


# ---------------------------------------------------------------------------
# ABI v3: baked scoring index
# ---------------------------------------------------------------------------


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
    var bound: F64Ptr  # [n_terms] Plus only: |floor| + max|excess| (see module docs)
    var floor0: F64Ptr  # [n_terms] Plus only: idf * delta per term
    var fulls: F64Ptr  # [nnz] Plus only: baked per-posting idf * (delta + frac)
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
        bound: F64Ptr,
        floor0: F64Ptr,
        fulls: F64Ptr,
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
        self.bound = bound
        self.floor0 = floor0
        self.fulls = fulls
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

    # --- per-term floors and magnitude bounds (Plus) ---
    var idf_copy = unsafe_alloc[Float64](Int(n_terms))
    if n_terms > 0:
        unsafe_memcpy(dest=idf_copy, src=idf, count=Int(n_terms))
    var floor0 = unsafe_alloc[Float64](Int(n_terms))
    var bound = unsafe_alloc[Float64](Int(n_terms))
    for t in range(Int(n_terms)):
        var f = Float64(0.0)
        if variant == VARIANT_PLUS:
            f = idf_copy[unsafe_offset=t] * delta
        floor0[unsafe_offset=t] = f
        bound[unsafe_offset=t] = Float64(0.0)  # finalized after the bake below

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
    # Plus also keeps the reference's per-posting value itself, for the
    # precise path (aliased placeholder for the other variants, never read).
    var fulls = weights
    if variant == VARIANT_PLUS:
        fulls = unsafe_alloc[Float64](Int(nnz))
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
            var full = _term_score[WIDTH](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            )
            if variant == VARIANT_PLUS:
                fulls.unsafe_store(j, full)
            var w = full - SIMD[DType.float64, WIDTH](floor_t)
            weights.unsafe_store(j, w)
            j += WIDTH
        while j < end:
            var qf = SIMD[DType.float64, 1](freqs[unsafe_offset=j])
            var dl = SIMD[DType.float64, 1](
                doc_len[unsafe_offset=Int(docs[unsafe_offset=j])]
            )
            var full = _term_score[1](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            )
            if variant == VARIANT_PLUS:
                fulls[unsafe_offset=j] = full[0]
            var w = full - SIMD[DType.float64, 1](floor_t)
            weights[unsafe_offset=j] = w[0]
            j += 1
    if variant == VARIANT_PLUS:
        # bound[t] = |floor_t| + max_j |excess_j|; any non-finite piece
        # poisons the bound (+inf here, NaN via abs(NaN) + m for a NaN floor),
        # and a non-finite bound never passes the query-time SAFE_BOUND check.
        #
        # Conditioning guard (issue #58): a posting whose length normaliser
        # (1 - b) + (b * dl) / avgdl or outer denominator k1 * norm + qf
        # cancels with amplification >= 1e13 (the fuzz harness's own
        # KnownIssue threshold, 1 / SCORE_RTOL) carries rounding residue in
        # place of a score, and the decomposed floor + (full - floor) order
        # rounds that residue differently from the reference's single
        # evaluation idf * (delta + frac). Poison the bound for such terms so
        # every query containing them takes the precise (reference-order)
        # path: kernel and harness then agree by construction, since the
        # decomposed order only ever runs where parity is checkable.
        # In-domain parameters never fire (every summand shares a sign, so
        # the amplification is at most 1).
        var AMP_LIMIT = Float64(1.0e13)
        var TINY = Float64(2.2250738585072014e-308)
        var one_minus_b = 1.0 - b
        for t in range(Int(n_terms)):
            var m = Float64(0.0)
            var ill = False
            var j = Int(offsets[unsafe_offset=t])
            var end = Int(offsets[unsafe_offset=t + 1])
            while j < end:
                var aw = abs(weights[unsafe_offset=j])
                if not isfinite(aw):
                    m = inf[DType.float64]()
                elif aw > m:
                    m = aw
                if not ill:
                    var dl_j = doc_len[unsafe_offset=Int(docs[unsafe_offset=j])]
                    var length_term = b * dl_j / avgdl
                    var norm_j = one_minus_b + length_term
                    var norm_big = abs(one_minus_b)
                    if abs(length_term) > norm_big:
                        norm_big = abs(length_term)
                    var norm_small = abs(norm_j)
                    if norm_small < TINY:
                        norm_small = TINY
                    var k1_norm = k1 * norm_j
                    var qf_j = freqs[unsafe_offset=j]
                    var den_p = k1_norm + qf_j
                    var den_big = abs(k1_norm)
                    if abs(qf_j) > den_big:
                        den_big = abs(qf_j)
                    var den_small = abs(den_p)
                    if den_small < TINY:
                        den_small = TINY
                    if norm_big / norm_small >= AMP_LIMIT or den_big / den_small >= AMP_LIMIT:
                        ill = True
                j += 1
            if ill:
                bound[unsafe_offset=t] = inf[DType.float64]()
            else:
                bound[unsafe_offset=t] = abs(floor0[unsafe_offset=t]) + m
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
        bound,
        floor0,
        fulls,
    )
    return idx.unsafe_bitcast[UInt8]()


def _plus_term_precise(
    idx: Pointer[BM25Index, MutUntrackedOrigin],
    t: Int,
    out_scores: F64Ptr,
):
    """Add one BM25Plus term to `out_scores` in reference evaluation order.

    Every document receives the term's value — the baked `full` where the
    term is posted, the floor `idf * delta` where it is not — exactly as the
    reference's dense per-term pass computes it (same IEEE-754 operation
    order per document; the caller iterates the query in token order, so the
    per-document accumulation order matches too). Used when the magnitude
    bound cannot rule out overflow in the decomposed order.
    """
    var n_docs = Int(idx[].n_docs)
    var floor_t = idx[].floor0[unsafe_offset=t]
    var j = Int(idx[].offsets[unsafe_offset=t])
    var end = Int(idx[].offsets[unsafe_offset=t + 1])
    var d = 0
    while j < end:
        var doc = Int(idx[].docs[unsafe_offset=j])
        while d < doc:
            out_scores[unsafe_offset=d] += floor_t
            d += 1
        out_scores[unsafe_offset=doc] += idx[].fulls[unsafe_offset=j]
        d = doc + 1
        j += 1
    while d < n_docs:
        out_scores[unsafe_offset=d] += floor_t
        d += 1


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
    var bound = idx[].bound
    var floor0 = idx[].floor0

    if variant == VARIANT_PLUS:
        # The decomposed fast order (one dense fill of the floor sum, then
        # gather-add of the posted excess) is exact whenever neither it nor
        # the reference's interleaved order can overflow; the per-query bound
        # sum proves that below SAFE_BOUND (see module docs). Above it the
        # query is scored term by term in the reference's own order.
        var floor_total = Float64(0.0)
        var bound_total = Float64(0.0)
        for qi in range(Int(n_query)):
            var t = Int(qids[unsafe_offset=qi])
            if t < 0 or t >= n_terms:
                return 2
            floor_total += floor0[unsafe_offset=t]
            bound_total += bound[unsafe_offset=t]
        if bound_total < SAFE_BOUND:
            # Dense floor: F = sum of per-term floors in query-token order,
            # then one write-only SIMD fill (bit-identical to zero + F).
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
                var j = Int(offsets[unsafe_offset=t])
                var end = Int(offsets[unsafe_offset=t + 1])
                while j < end:
                    out_scores[unsafe_offset=Int(docs[unsafe_offset=j])] += weights[
                        unsafe_offset=j
                    ]
                    j += 1
        else:
            for qi in range(Int(n_query)):
                _plus_term_precise(idx, Int(qids[unsafe_offset=qi]), out_scores)
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
    var bound = idx[].bound
    var floor0 = idx[].floor0
    var base = Int64(0)
    for q in range(Int(n_queries)):
        var cnt = Int(counts[unsafe_offset=q])
        if cnt < 0:
            return 2
        var row = out_panel.unsafe_offset(q * n_docs)
        if variant == VARIANT_PLUS:
            # Floor sum and bound sum over FOUND tokens only, in query-token
            # order (exactly the filtered-id order of the v2 path); then the
            # same fast-or-precise dispatch as `_score_one`.
            var floor_total = Float64(0.0)
            var bound_total = Float64(0.0)
            for j in range(cnt):
                var slot = toks.unsafe_offset(Int((base + Int64(j)) * width))
                var L = _slot_scan(slot, Int(width))
                if L < 0:
                    return 3
                var t = _vocab_lookup(idx, slot, L)
                if t >= 0:
                    floor_total += floor0[unsafe_offset=Int(t)]
                    bound_total += bound[unsafe_offset=Int(t)]
            if bound_total < SAFE_BOUND:
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
                    if t < 0:
                        continue
                    _accumulate(idx, Int(t), row)
            else:
                for j in range(cnt):
                    var slot = toks.unsafe_offset(Int((base + Int64(j)) * width))
                    var L = _slot_scan(slot, Int(width))
                    var t = _vocab_lookup(idx, slot, L)
                    if t >= 0:
                        _plus_term_precise(idx, Int(t), row)
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
    idx[].bound.unsafe_free()
    idx[].floor0.unsafe_free()
    if idx[].variant == VARIANT_PLUS:
        idx[].fulls.unsafe_free()  # allocated only for Plus (placeholder else)
    if idx[].vocab_cap != 0:
        idx[].vocab_keys.unsafe_free()
        idx[].vocab_lens.unsafe_free()
        idx[].vocab_tids.unsafe_free()
    idx.unsafe_free()
