"""Clean-room fuzzy/prefix term-resolution + BM25 scoring kernel replicating
MiniSearch's observable semantics.

Written fresh from the textbook bounded Levenshtein dynamic program (two-row,
early exit when a row's minimum exceeds the edit budget; insertion / deletion /
substitution cost 1, no transposition) and from black-box observed scoring
behavior of the reference engine. No third-party Mojo or JS code is used or
adapted.

The kernel operates on UTF-16 code units, exactly like JavaScript strings, so
term matching is byte-identical to the JS reference (astral characters occupy
two code units and match as two independent units).

Exported C ABI (batch-shaped: the index is built once from flat buffers that
the kernel copies; each query then runs as one fuzzy/prefix scan per query
term plus one scoring accumulation over the resolved variants' posting lists):

    int32_t  msmojo_abi_version(void)
    void*    msmojo_index_create(vocab_chars, vocab_offsets, n_terms,
                                 post_offsets, post_doc, post_field, post_tf,
                                 docfield_len, avgdl, n_docs, n_fields)
    void     msmojo_index_destroy(handle)
    int32_t  msmojo_fuzzy_scan(handle, query, qlen, max_edits,
                               out_ids, out_dist, cap)
    int32_t  msmojo_prefix_scan(handle, query, qlen, out_ids, cap)
    int32_t  msmojo_score(handle, qv_offsets, qv_ids, qv_weight, qv_idf, qv_qt,
                          n_raw_q, field_boost, allowed_fields_mask,
                          out_scores, out_mask, out_records, record_cap)

Scan order contract (the wrapper depends on it for exact `terms` ordering;
the wrapper stores the vocabulary in reverse-map radix DFS order):
  - msmojo_fuzzy_scan emits matching (term_id, distance) pairs in DESCENDING
    term_id order (== forward-map radix DFS, the reference fuzzy order),
    including distance 0.
  - msmojo_prefix_scan emits matching term_ids in ASCENDING term_id order
    (== reverse-map radix DFS, the reference prefix order), including the
    exact term when present.

Scoring contract: for every raw query term occurrence, for every resolved
variant (in the order given), the kernel walks the variant's posting list
(sorted by (doc, field)) and accumulates
    weight * boost[field] * idf(variant, field) * F(tf, dl/avgdl)
into out_scores[doc], sets bit qv_qt[q] in out_mask[doc], and emits one
(doc, qt, term, field_mask) record per matching doc (records arrive in
(raw_qt, variant, doc) order). The per-(variant, field) idf is computed by
the caller (JavaScript Math.log is correctly rounded; the Mojo stdlib log is
not accurate enough for the 1e-9 gate) and passed in as qv_idf; F replicates
the reference's observed BM25 tf component:
    F   = (45*tf + 3 + 7*r) / (25*tf + 9 + 21*r), r = dl / avgdl
All floating-point arithmetic is IEEE-754 float64, so scores agree with the
JS reference to within a few ulps (validated far inside the 1e-9 gate).
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
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# Upper bound on raw query terms per query (wrapper enforces n_distinct <= 32
# for the Int32 match bitmask; raw occurrences may repeat).
comptime MAX_RAW_Q = 64


struct MSIndex(Copyable, Movable):
    """Owned, native copy of one inverted index: vocabulary, per-term document
    frequency, CSR posting lists, per-(doc, field) unique-term counts, and
    per-field average field lengths."""

    var n_terms: Int32
    var vocab_chars: U16Ptr  # [total_chars] flat UTF-16 code units
    var vocab_offsets: I32Ptr  # [n_terms + 1]
    var post_offsets: I32Ptr  # [n_terms + 1]
    var post_doc: I32Ptr  # [n_postings]
    var post_field: I32Ptr  # [n_postings]
    var post_tf: I32Ptr  # [n_postings]
    var docfield_len: I32Ptr  # [n_docs * n_fields] unique-term counts
    var avgdl: F64Ptr  # [n_fields]
    var n_docs: Int32
    var n_fields: Int32
    var max_term_len: Int32

    def __init__(
        out self,
        n_terms: Int32,
        vocab_chars: U16Ptr,
        vocab_offsets: I32Ptr,
        post_offsets: I32Ptr,
        post_doc: I32Ptr,
        post_field: I32Ptr,
        post_tf: I32Ptr,
        docfield_len: I32Ptr,
        avgdl: F64Ptr,
        n_docs: Int32,
        n_fields: Int32,
        max_term_len: Int32,
    ):
        self.n_terms = n_terms
        self.vocab_chars = vocab_chars
        self.vocab_offsets = vocab_offsets
        self.post_offsets = post_offsets
        self.post_doc = post_doc
        self.post_field = post_field
        self.post_tf = post_tf
        self.docfield_len = docfield_len
        self.avgdl = avgdl
        self.n_docs = n_docs
        self.n_fields = n_fields
        self.max_term_len = max_term_len


def _bounded_levenshtein(
    query: U16Ptr,
    qlen: Int32,
    term: U16Ptr,
    tlen: Int32,
    max_edits: Int32,
    row_a: I32Ptr,
    row_b: I32Ptr,
) -> Int32:
    """Levenshtein distance(query, term) with ins/del/sub cost 1 (no
    transposition), or -1 when the distance exceeds max_edits. Two-row DP
    with an early exit as soon as some row's minimum exceeds the budget."""
    var dlen = tlen - qlen
    if dlen < 0:
        dlen = -dlen
    if dlen > max_edits:
        return -1

    var prev = row_a
    var cur = row_b
    var j = Int32(0)
    while j <= tlen:
        prev[unsafe_offset=Int(j)] = j
        j += 1

    var i = Int32(1)
    while i <= qlen:
        cur[unsafe_offset=0] = i
        var row_min = i
        var qc = query[unsafe_offset=Int(i - 1)]
        j = 1
        while j <= tlen:
            var cost = Int32(0)
            if term[unsafe_offset=Int(j - 1)] != qc:
                cost = 1
            var v = prev[unsafe_offset=Int(j)] + 1
            var w = cur[unsafe_offset=Int(j - 1)] + 1
            if w < v:
                v = w
            w = prev[unsafe_offset=Int(j - 1)] + cost
            if w < v:
                v = w
            cur[unsafe_offset=Int(j)] = v
            if v < row_min:
                row_min = v
            j += 1
        if row_min > max_edits:
            return -1
        var tmp = prev
        prev = cur
        cur = tmp
        i += 1

    var d = prev[unsafe_offset=Int(tlen)]
    if d > max_edits:
        return -1
    return d


@export
def msmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def msmojo_index_create(
    vocab_chars: U16Ptr,
    vocab_offsets: I32Ptr,
    n_terms: Int32,
    post_offsets: I32Ptr,
    post_doc: I32Ptr,
    post_field: I32Ptr,
    post_tf: I32Ptr,
    docfield_len: I32Ptr,
    avgdl: F64Ptr,
    n_docs: Int32,
    n_fields: Int32,
) abi("C") -> Handle:
    """Copy the caller's flat index buffers into a native index; NULL on
    invalid input. All buffers are copied; the caller may free theirs.
    Postings must be sorted by (doc, field); term_ids are vocabulary
    insertion order."""
    if n_terms < 0 or n_docs < 0 or n_fields < 1:
        return None

    var total_chars = Int64(0)
    var max_term_len = Int32(0)
    if n_terms > 0:
        total_chars = Int64(vocab_offsets[unsafe_offset=Int(n_terms)])
        if total_chars < 0 or total_chars > 0x7FFFFFFF:
            return None
        var t = Int32(0)
        var prev = vocab_offsets[unsafe_offset=0]
        while t < n_terms:
            var cur = vocab_offsets[unsafe_offset=Int(t + 1)]
            var length = cur - prev
            if length < 0:
                return None
            if length > max_term_len:
                max_term_len = length
            prev = cur
            t += 1

    var n_postings = Int64(0)
    if n_terms > 0:
        n_postings = Int64(post_offsets[unsafe_offset=Int(n_terms)])
        if n_postings < 0 or n_postings > 0x7FFFFFFF:
            return None

    var vocab_chars_copy = unsafe_alloc[UInt16](Int(total_chars) + 1)
    if total_chars > 0:
        unsafe_memcpy(dest=vocab_chars_copy, src=vocab_chars, count=Int(total_chars))
    var vocab_offsets_copy = unsafe_alloc[Int32](Int(n_terms) + 1)
    unsafe_memcpy(dest=vocab_offsets_copy, src=vocab_offsets, count=Int(n_terms) + 1)
    var post_offsets_copy = unsafe_alloc[Int32](Int(n_terms) + 1)
    unsafe_memcpy(dest=post_offsets_copy, src=post_offsets, count=Int(n_terms) + 1)
    var post_doc_copy = unsafe_alloc[Int32](Int(n_postings) + 1)
    var post_field_copy = unsafe_alloc[Int32](Int(n_postings) + 1)
    var post_tf_copy = unsafe_alloc[Int32](Int(n_postings) + 1)
    if n_postings > 0:
        unsafe_memcpy(dest=post_doc_copy, src=post_doc, count=Int(n_postings))
        unsafe_memcpy(dest=post_field_copy, src=post_field, count=Int(n_postings))
        unsafe_memcpy(dest=post_tf_copy, src=post_tf, count=Int(n_postings))
    var docfield_len_copy = unsafe_alloc[Int32](Int(n_docs) * Int(n_fields) + 1)
    if n_docs > 0:
        unsafe_memcpy(
            dest=docfield_len_copy, src=docfield_len, count=Int(n_docs) * Int(n_fields)
        )
    var avgdl_copy = unsafe_alloc[Float64](Int(n_fields))
    unsafe_memcpy(dest=avgdl_copy, src=avgdl, count=Int(n_fields))

    var idx = unsafe_alloc[MSIndex](1)
    idx[] = MSIndex(
        n_terms,
        vocab_chars_copy,
        vocab_offsets_copy,
        post_offsets_copy,
        post_doc_copy,
        post_field_copy,
        post_tf_copy,
        docfield_len_copy,
        avgdl_copy,
        n_docs,
        n_fields,
        max_term_len,
    )
    return idx.unsafe_bitcast[UInt8]()


@export
def msmojo_index_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var idx = handle.value().unsafe_bitcast[MSIndex]()
    idx[].vocab_chars.unsafe_free()
    idx[].vocab_offsets.unsafe_free()
    idx[].post_offsets.unsafe_free()
    idx[].post_doc.unsafe_free()
    idx[].post_field.unsafe_free()
    idx[].post_tf.unsafe_free()
    idx[].docfield_len.unsafe_free()
    idx[].avgdl.unsafe_free()
    idx.unsafe_free()


@export
def msmojo_fuzzy_scan(
    handle: Handle,
    query: U16Ptr,
    qlen: Int32,
    max_edits: Int32,
    out_ids: I32Ptr,
    out_dist: I32Ptr,
    cap: Int32,
) abi("C") -> Int32:
    """Scan the whole vocabulary in DESCENDING term_id order, emitting
    (term_id, distance) for every term whose bounded Levenshtein distance to
    the query is <= max_edits (distance 0 included). The caller stores the
    vocabulary in reverse-map radix DFS order, so descending ids enumerate
    matches in the reference's forward-map fuzzy-traversal order. Returns the
    match count, or a negative status on invalid input. `cap` must be >=
    n_terms."""
    if not handle:
        return -1
    if qlen < 0 or max_edits < 0 or cap < 0:
        return -2
    var idx = handle.value().unsafe_bitcast[MSIndex]()
    var n_terms = idx[].n_terms
    if cap < n_terms:
        return -3
    if n_terms == 0:
        return 0

    var row_a = unsafe_alloc[Int32](Int(idx[].max_term_len) + 2)
    var row_b = unsafe_alloc[Int32](Int(idx[].max_term_len) + 2)

    var count = Int32(0)
    var t = n_terms
    while t > 0:
        t -= 1
        var start = idx[].vocab_offsets[unsafe_offset=Int(t)]
        var end = idx[].vocab_offsets[unsafe_offset=Int(t + 1)]
        var d = _bounded_levenshtein(
            query,
            qlen,
            idx[].vocab_chars.unsafe_offset(Int(start)),
            end - start,
            max_edits,
            row_a,
            row_b,
        )
        if d >= 0:
            out_ids[unsafe_offset=Int(count)] = t
            out_dist[unsafe_offset=Int(count)] = d
            count += 1

    row_a.unsafe_free()
    row_b.unsafe_free()
    return count


@export
def msmojo_prefix_scan(
    handle: Handle,
    query: U16Ptr,
    qlen: Int32,
    out_ids: I32Ptr,
    cap: Int32,
) abi("C") -> Int32:
    """Scan the whole vocabulary in ASCENDING term_id order, emitting every
    term of length >= qlen whose first qlen code units equal the query (the
    exact term itself included). The caller stores the vocabulary in
    reverse-map radix DFS order, so ascending ids enumerate matches in the
    reference's prefix-traversal order (the exact term, when present, is
    pulled to the front by the caller). Returns the match count, or a
    negative status on invalid input. `cap` must be >= n_terms."""
    if not handle:
        return -1
    if qlen < 0 or cap < 0:
        return -2
    var idx = handle.value().unsafe_bitcast[MSIndex]()
    var n_terms = idx[].n_terms
    if cap < n_terms:
        return -3
    if n_terms == 0:
        return 0

    var count = Int32(0)
    var t = Int32(0)
    while t < n_terms:
        var start = idx[].vocab_offsets[unsafe_offset=Int(t)]
        var end = idx[].vocab_offsets[unsafe_offset=Int(t + 1)]
        var tlen = end - start
        if tlen >= qlen:
            var k = Int32(0)
            var matched = True
            while k < qlen:
                if idx[].vocab_chars[unsafe_offset=Int(start + k)] != query[
                    unsafe_offset=Int(k)
                ]:
                    matched = False
                    break
                k += 1
            if matched:
                out_ids[unsafe_offset=Int(count)] = t
                count += 1
        t += 1
    return count


@export
def msmojo_score(
    handle: Handle,
    qv_offsets: I32Ptr,
    qv_ids: I32Ptr,
    qv_weight: F64Ptr,
    qv_idf: F64Ptr,
    qv_qt: I32Ptr,
    n_raw_q: Int32,
    field_boost: F64Ptr,
    allowed_fields_mask: Int32,
    out_scores: F64Ptr,
    out_mask: I32Ptr,
    out_records: I32Ptr,
    record_cap: Int32,
) abi("C") -> Int32:
    """Accumulate one query's BM25 scores over the resolved variants.

    qv_* describe the variants of each raw query term as a CSR list
    (qv_offsets has n_raw_q + 1 entries); qv_qt[q] is the distinct query-term
    index of raw occurrence q (0..31); qv_idf[v * n_fields + f] is the
    caller-computed idf of variant v in field f. For every variant the kernel
    walks its posting list, and for every posting whose field is allowed
    accumulates
        weight * boost[field] * idf * F(tf, dl/avgdl)
    into out_scores[doc], sets bit qv_qt[q] in out_mask[doc], and emits one
    4-Int32 record (doc, qt, term_id, field_mask) per matching doc (postings
    are (doc, field)-sorted, so a doc's fields merge into one record).

    out_scores / out_mask hold n_docs slots (zeroed by this call);
    out_records holds 4 * record_cap slots. Returns the record count, -4 when
    record_cap is exceeded (outputs are undefined then), or another negative
    status on invalid input.
    """
    if not handle:
        return -1
    if n_raw_q < 1 or n_raw_q > MAX_RAW_Q or record_cap < 0:
        return -2
    var idx = handle.value().unsafe_bitcast[MSIndex]()
    var n_docs = idx[].n_docs
    var n_fields = idx[].n_fields

    var d = Int32(0)
    while d < n_docs:
        out_scores[unsafe_offset=Int(d)] = 0.0
        out_mask[unsafe_offset=Int(d)] = 0
        d += 1

    var n_records = Int32(0)
    var q = Int32(0)
    while q < n_raw_q:
        var qt = qv_qt[unsafe_offset=Int(q)]
        if qt < 0 or qt > 31:
            return -2
        var qt_bit = Int32(1) << qt
        var v = qv_offsets[unsafe_offset=Int(q)]
        var v_end = qv_offsets[unsafe_offset=Int(q + 1)]
        while v < v_end:
            var term = qv_ids[unsafe_offset=Int(v)]
            if term < 0 or term >= idx[].n_terms:
                return -2
            var weight = qv_weight[unsafe_offset=Int(v)]
            var p = idx[].post_offsets[unsafe_offset=Int(term)]
            var p_end = idx[].post_offsets[unsafe_offset=Int(term + 1)]
            var cur_doc = Int32(-1)
            var cur_fmask = Int32(0)
            while p < p_end:
                var field = idx[].post_field[unsafe_offset=Int(p)]
                if (allowed_fields_mask & (Int32(1) << field)) != 0:
                    var doc = idx[].post_doc[unsafe_offset=Int(p)]
                    var tf = Float64(idx[].post_tf[unsafe_offset=Int(p)])
                    var idf = qv_idf[unsafe_offset=Int(v * n_fields + field)]
                    var dl = Float64(
                        idx[].docfield_len[unsafe_offset=Int(doc * n_fields + field)]
                    )
                    var r = dl / idx[].avgdl[unsafe_offset=Int(field)]
                    var big_f = (45.0 * tf + 3.0 + 7.0 * r) / (
                        25.0 * tf + 9.0 + 21.0 * r
                    )
                    out_scores[unsafe_offset=Int(doc)] = out_scores[
                        unsafe_offset=Int(doc)
                    ] + weight * field_boost[unsafe_offset=Int(field)] * idf * big_f
                    out_mask[unsafe_offset=Int(doc)] = out_mask[
                        unsafe_offset=Int(doc)
                    ] | qt_bit
                    if doc != cur_doc:
                        if cur_doc >= 0:
                            if n_records >= record_cap:
                                return -4
                            var base = n_records * 4
                            out_records[unsafe_offset=Int(base)] = cur_doc
                            out_records[unsafe_offset=Int(base + 1)] = qt
                            out_records[unsafe_offset=Int(base + 2)] = term
                            out_records[unsafe_offset=Int(base + 3)] = cur_fmask
                            n_records += 1
                        cur_doc = doc
                        cur_fmask = 0
                    cur_fmask = cur_fmask | (Int32(1) << field)
                p += 1
            if cur_doc >= 0:
                if n_records >= record_cap:
                    return -4
                var base = n_records * 4
                out_records[unsafe_offset=Int(base)] = cur_doc
                out_records[unsafe_offset=Int(base + 1)] = qt
                out_records[unsafe_offset=Int(base + 2)] = term
                out_records[unsafe_offset=Int(base + 3)] = cur_fmask
                n_records += 1
            v += 1
        q += 1
    return n_records
