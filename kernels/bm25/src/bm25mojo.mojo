"""Clean-room BM25 scoring kernel (Okapi / BM25L / BM25Plus).

Written fresh from the textbook BM25 family of ranking functions
(Trotman et al., "Improvements to BM25 and Language Models Examined").
No third-party Mojo code is used or adapted.

Exported C ABI (batch-shaped: the index is built once from CSR postings,
then each call scores one whole token-id query against the whole corpus):

    int32_t  bm25mojo_abi_version(void)
    void*    bm25mojo_index_create(n_docs, doc_len, avgdl, k1, b, delta,
                                   variant, n_terms, offsets, docs, freqs, idf)
    int32_t  bm25mojo_score(handle, qids, n_query, out_scores)
    void     bm25mojo_index_destroy(handle)

The inverse document frequency is computed by the Python wrapper (it owns the
vocabulary); this kernel receives one idf value per term and evaluates only the
per-document accumulation, vectorized across postings with SIMD:

    Okapi: score[d] += idf * (qf * (k1 + 1) / (qf + k1 * (1 - b + b * dl / avgdl)))
    Plus:  score[d] += idf * (delta + qf * (k1 + 1) / (k1 * (1 - b + b * dl / avgdl) + qf))
    L:     score[d] += idf * qf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)
                            with ctd = qf / (1 - b + b * dl / avgdl)

All arithmetic is IEEE-754 float64 in the same operation order as the
reference NumPy implementation (PyPI rank_bm25 0.2.2), with one SIMD lane per
posting, so scores match the reference element-wise (bit-identical in
practice). Accumulation per document is sequential in query-term order. For
BM25Plus, whose delta term gives even unposted documents a constant per-term
floor, the floor is added with a dense SIMD pass per query term and posted
documents receive only their excess over it (Okapi/BM25L have a zero floor).
When that floor is not finite (|idf * delta| overflows float64, or is NaN),
the excess would be inf - inf = NaN, so the term is instead evaluated once per
document in reference order, exactly as the reference does.
"""

from std.math import isfinite
from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

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
    """Owned, native copy of one corpus index (CSR postings + per-doc length)."""

    var n_docs: Int64
    var avgdl: Float64
    var k1: Float64
    var b: Float64
    var delta: Float64
    var variant: Int32
    var n_terms: Int64
    var nnz: Int64
    var doc_len: F64Ptr  # [n_docs]
    var offsets: I64Ptr  # [n_terms + 1] CSR row offsets
    var docs: I32Ptr  # [nnz] posting doc ids (ascending per term)
    var freqs: F64Ptr  # [nnz] posting term frequencies
    var idf: F64Ptr  # [n_terms]

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
        doc_len: F64Ptr,
        offsets: I64Ptr,
        docs: I32Ptr,
        freqs: F64Ptr,
        idf: F64Ptr,
    ):
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.variant = variant
        self.n_terms = n_terms
        self.nnz = nnz
        self.doc_len = doc_len
        self.offsets = offsets
        self.docs = docs
        self.freqs = freqs
        self.idf = idf


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
    offsets: I64Ptr,
    docs: I32Ptr,
    freqs: F64Ptr,
    idf: F64Ptr,
) abi("C") -> Handle:
    """Copy the caller's buffers into a native index; NULL on invalid input."""
    if (
        n_docs <= 0
        or n_terms < 0
        or variant < VARIANT_OKAPI
        or variant > VARIANT_PLUS
    ):
        return None

    var nnz = Int64(0)
    if n_terms > 0:
        nnz = offsets[unsafe_offset=Int(n_terms)]
        if nnz < 0:
            return None

    var dl_copy = unsafe_alloc[Float64](Int(n_docs))
    unsafe_memcpy(dest=dl_copy, src=doc_len, count=Int(n_docs))
    var off_copy = unsafe_alloc[Int64](Int(n_terms) + 1)
    unsafe_memcpy(dest=off_copy, src=offsets, count=Int(n_terms) + 1)
    var idf_copy = unsafe_alloc[Float64](Int(n_terms))
    if n_terms > 0:
        unsafe_memcpy(dest=idf_copy, src=idf, count=Int(n_terms))
    var docs_copy = unsafe_alloc[Int32](Int(nnz))
    var freqs_copy = unsafe_alloc[Float64](Int(nnz))
    if nnz > 0:
        unsafe_memcpy(dest=docs_copy, src=docs, count=Int(nnz))
        unsafe_memcpy(dest=freqs_copy, src=freqs, count=Int(nnz))

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
        dl_copy,
        off_copy,
        docs_copy,
        freqs_copy,
        idf_copy,
    )
    return idx.unsafe_bitcast[UInt8]()


@export
def bm25mojo_score(
    handle: Handle,
    qids: I32Ptr,
    n_query: Int64,
    out_scores: F64Ptr,
) abi("C") -> Int32:
    """Accumulate one query's scores over the whole corpus into `out_scores`.

    `out_scores` must hold n_docs float64 slots; it is zeroed here first.
    Returns 0 on success, 1 on a NULL handle, 2 on a term id or query
    length outside the indexed range.
    """
    if not handle:
        return 1
    if n_query < 0:
        return 2
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    var n_docs = Int(idx[].n_docs)
    for d in range(n_docs):
        out_scores[unsafe_offset=d] = 0.0

    var k1 = idx[].k1
    var b = idx[].b
    var avgdl = idx[].avgdl
    var delta = idx[].delta
    var variant = idx[].variant
    var n_terms = Int(idx[].n_terms)
    var doc_len = idx[].doc_len
    var offsets = idx[].offsets
    var docs = idx[].docs
    var freqs = idx[].freqs
    var idf = idx[].idf

    for qi in range(Int(n_query)):
        var t = Int(qids[unsafe_offset=qi])
        if t < 0 or t >= n_terms:
            return 2
        var idf_t = idf[unsafe_offset=t]
        var i = Int(offsets[unsafe_offset=t])
        var end = Int(offsets[unsafe_offset=t + 1])
        # BM25Plus adds a constant floor to EVERY document per query term (its
        # delta term is non-zero even when qf == 0). Add that floor densely,
        # then add only the excess over the floor for posted docs. For Okapi
        # and BM25L the floor is exactly 0, so the dense pass is skipped and
        # subtracting it below is an exact no-op (x - 0.0 == x).
        var floor_t = Float64(0.0)
        if variant == VARIANT_PLUS:
            floor_t = idf_t * delta
        if not isfinite(floor_t):
            # |idf * delta| overflowed to +-inf (or is NaN). Adding that floor
            # densely and then a posted document's "excess over the floor"
            # would be inf - inf = NaN, where the reference's single
            # evaluation idf * (delta + ...) is +-inf. Evaluate the reference
            # expression once per document instead -- posted documents with
            # their qf, every other document with qf = 0 -- walking the
            # corpus in tandem with the ascending posting list. Only
            # out-of-domain parameters (|delta| near 1.8e308) reach this
            # path, so it is scalar.
            var d = 0
            while d < n_docs:
                var qf = SIMD[DType.float64, 1](0.0)
                if i < end and Int(docs[unsafe_offset=i]) == d:
                    qf = SIMD[DType.float64, 1](freqs[unsafe_offset=i])
                    i += 1
                var dl = SIMD[DType.float64, 1](doc_len[unsafe_offset=d])
                out_scores[unsafe_offset=d] += _term_score[1](
                    variant, qf, dl, idf_t, k1, b, avgdl, delta
                )[0]
                d += 1
            continue
        if floor_t != 0.0:
            var floor_v = SIMD[DType.float64, WIDTH](floor_t)
            var d = 0
            while d + WIDTH <= n_docs:
                out_scores.unsafe_store(
                    d, out_scores.unsafe_load[width=WIDTH](d) + floor_v
                )
                d += WIDTH
            while d < n_docs:
                out_scores[unsafe_offset=d] += floor_t
                d += 1
        # Vector body: one SIMD lane per posting.
        while i + WIDTH <= end:
            var qf = freqs.unsafe_load[width=WIDTH](i)
            var dl = SIMD[DType.float64, WIDTH]()
            for lane in range(WIDTH):
                dl[lane] = doc_len[unsafe_offset=Int(docs[unsafe_offset=i + lane])]
            var contrib = _term_score[WIDTH](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            ) - SIMD[DType.float64, WIDTH](floor_t)
            for lane in range(WIDTH):
                out_scores[unsafe_offset=Int(docs[unsafe_offset=i + lane])] += contrib[
                    lane
                ]
            i += WIDTH
        # Scalar tail (width-1 keeps one code path for the formula).
        while i < end:
            var qf = SIMD[DType.float64, 1](freqs[unsafe_offset=i])
            var dl = SIMD[DType.float64, 1](
                doc_len[unsafe_offset=Int(docs[unsafe_offset=i])]
            )
            var contrib = _term_score[1](
                variant, qf, dl, idf_t, k1, b, avgdl, delta
            ) - SIMD[DType.float64, 1](floor_t)
            out_scores[unsafe_offset=Int(docs[unsafe_offset=i])] += contrib[0]
            i += 1
    return 0


@export
def bm25mojo_index_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var idx = handle.value().unsafe_bitcast[BM25Index]()
    idx[].doc_len.unsafe_free()
    idx[].offsets.unsafe_free()
    idx[].docs.unsafe_free()
    idx[].freqs.unsafe_free()
    idx[].idf.unsafe_free()
    idx.unsafe_free()
