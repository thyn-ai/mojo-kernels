"""Clean-room Bitap fuzzy-search kernel replicating Fuse.js v7 observable semantics.

Written fresh from the published behavior of the Bitap/Shift-And fuzzy string
matching algorithm (a textbook, word-parallel bitmask dynamic program) and the
documented scoring rules of fuzzy search: an exact-match pre-pass, an
error-level loop with a binary-search window, and a location/distance score.
No third-party Mojo code is used or adapted.

The kernel operates on UTF-16 code units, exactly like JavaScript strings
(`String.length`, `charCodeAt`, `charAt`, `indexOf` are all UTF-16 based), so
scores and match spans are bit-identical to the JS reference without any index
conversion. Astral characters (e.g. emoji) occupy two code units and match as
two independent units, precisely mirroring the JS engine's behavior.

Exported C ABI (batch-shaped: the index is built once from a flat UTF-16
buffer; each query runs as begin -> range* -> end so the caller may drive the
per-text loops from multiple OS threads — see src/shim.c, which spawns
pthreads calling fusemojo_search_range over disjoint text ranges):

    int32_t  fusemojo_abi_version(void)
    void*    fusemojo_index_create(chars, offsets, n_texts, location, distance,
                                   threshold, min_match_char_length,
                                   find_all_matches, ignore_location,
                                   compute_matches, include_matches)
    void*    fusemojo_search_begin(handle, pattern, pattern_len,
                                   location_offset, exact_check, out_scores,
                                   out_is_match, out_idx_offsets, n_jobs)
    int32_t  fusemojo_search_range(ctx, job_id, start, end)
    int32_t  fusemojo_search_end(ctx)            -> total index pairs, <0 on error
    void     fusemojo_copy_indices(handle, out_pairs)
    void     fusemojo_index_destroy(handle)

Threading contract: jobs process disjoint, ascending text ranges and write
only their own texts' output slots and their own per-job stash, so results
are identical regardless of scheduling. Per-text index-pair counts land in
out_idx_offsets[t+1]; fusemojo_search_end turns them into CSR offsets and
concatenates the per-job stashes in job order (== text order) into the
index's stash, read back through fusemojo_copy_indices.

All integer arithmetic is 32-bit two's complement, matching JavaScript's
bitwise operators (ToInt32 coercion). All floating-point arithmetic is
IEEE-754 float64 in the same operation order as the JS reference, so scores
agree bit-for-bit.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# JavaScript's MAX_BITS: patterns longer than 32 code units are chunked by the
# caller; one kernel call always handles a single chunk of 1..32 code units.
comptime MAX_PATTERN_LEN: Int32 = 32

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# Size of the pattern-alphabet table: one Int32 mask per UTF-16 code unit.
comptime ALPHABET_SIZE = 65536


struct FuseIndex(Copyable, Movable):
    """Owned, native copy of one text collection plus the fixed match options."""

    var n_texts: Int32
    var chars: U16Ptr  # [total_chars] flat UTF-16 code units
    var offsets: I32Ptr  # [n_texts + 1] code-unit offsets per text
    var max_text_len: Int32
    var location: Int32
    var distance: Float64
    var threshold: Float64
    var min_match_char_length: Int32
    var find_all_matches: Int32  # 0/1
    var ignore_location: Int32  # 0/1
    var compute_matches: Int32  # 0/1 (minMatchCharLength > 1 || includeMatches)
    var include_matches: Int32  # 0/1
    # Final [start, end] Int32 pairs of the most recent completed search,
    # read back through fusemojo_copy_indices. Grown geometrically.
    var stash: I32Ptr
    var stash_cap: Int32  # capacity in Int32 slots
    var stash_len: Int32  # used Int32 slots (2 * total_pairs)

    def __init__(
        out self,
        n_texts: Int32,
        chars: U16Ptr,
        offsets: I32Ptr,
        max_text_len: Int32,
        location: Int32,
        distance: Float64,
        threshold: Float64,
        min_match_char_length: Int32,
        find_all_matches: Int32,
        ignore_location: Int32,
        compute_matches: Int32,
        include_matches: Int32,
        stash: I32Ptr,
        stash_cap: Int32,
    ):
        self.n_texts = n_texts
        self.chars = chars
        self.offsets = offsets
        self.max_text_len = max_text_len
        self.location = location
        self.distance = distance
        self.threshold = threshold
        self.min_match_char_length = min_match_char_length
        self.find_all_matches = find_all_matches
        self.ignore_location = ignore_location
        self.compute_matches = compute_matches
        self.include_matches = include_matches
        self.stash = stash
        self.stash_cap = stash_cap
        self.stash_len = 0


struct JobState(Copyable, Movable):
    """Per-thread scratch for one range job: ping-pong bit arrays, match mask,
    and a growable stash of index pairs produced by this job's texts."""

    var buf_a: I32Ptr  # [scratch_len]
    var buf_b: I32Ptr  # [scratch_len]
    var match_mask: U8Ptr  # [max_text_len + 1]
    var stash: I32Ptr
    var stash_cap: Int32
    var stash_len: Int32

    def __init__(
        out self,
        buf_a: I32Ptr,
        buf_b: I32Ptr,
        match_mask: U8Ptr,
        stash: I32Ptr,
        stash_cap: Int32,
    ):
        self.buf_a = buf_a
        self.buf_b = buf_b
        self.match_mask = match_mask
        self.stash = stash
        self.stash_cap = stash_cap
        self.stash_len = 0


struct SearchCtx(Copyable, Movable):
    """Everything one query needs, shared read-only across range jobs except
    the per-job JobState blocks (indexed by job_id)."""

    var idx: Pointer[FuseIndex, MutUntrackedOrigin]
    var pattern: U16Ptr  # [pattern_len] owned copy (caller buffer is transient)
    var pattern_len: Int32
    var location_offset: Int32
    var exact_check: Int32
    var out_scores: F64Ptr
    var out_is_match: I32Ptr
    var out_idx_offsets: I32Ptr
    var alphabet: I32Ptr  # [ALPHABET_SIZE]
    var n_jobs: Int32
    var jobs: Pointer[JobState, MutUntrackedOrigin]  # [n_jobs]

    def __init__(
        out self,
        idx: Pointer[FuseIndex, MutUntrackedOrigin],
        pattern: U16Ptr,
        pattern_len: Int32,
        location_offset: Int32,
        exact_check: Int32,
        out_scores: F64Ptr,
        out_is_match: I32Ptr,
        out_idx_offsets: I32Ptr,
        alphabet: I32Ptr,
        n_jobs: Int32,
        jobs: Pointer[JobState, MutUntrackedOrigin],
    ):
        self.idx = idx
        self.pattern = pattern
        self.pattern_len = pattern_len
        self.location_offset = location_offset
        self.exact_check = exact_check
        self.out_scores = out_scores
        self.out_is_match = out_is_match
        self.out_idx_offsets = out_idx_offsets
        self.alphabet = alphabet
        self.n_jobs = n_jobs
        self.jobs = jobs


def _compute_score(
    pattern_len: Int32,
    errors: Int32,
    current_location: Int32,
    expected_location: Int32,
    distance: Float64,
    ignore_location: Int32,
) -> Float64:
    """Match-quality score in [0, inf): accuracy plus location proximity.

    Identical operation order to the JS reference:
    accuracy = errors / pattern.length; if ignoreLocation return accuracy;
    proximity = |expected - current|; if distance == 0: proximity ? 1 :
    accuracy; else accuracy + proximity / distance.
    """
    var accuracy = Float64(errors) / Float64(pattern_len)
    if ignore_location != 0:
        return accuracy
    var proximity = expected_location - current_location
    if proximity < 0:
        proximity = -proximity
    if distance == 0.0:
        if proximity != 0:
            return Float64(1.0)
        return accuracy
    return accuracy + Float64(proximity) / distance


def _index_of(
    chars: U16Ptr,
    text_off: Int32,
    text_len: Int32,
    pattern: U16Ptr,
    pattern_len: Int32,
    from_index: Int32,
) -> Int32:
    """First position >= from_index where `pattern` occurs in the text, -1 if
    absent. Same observable result as JS `String.prototype.indexOf`."""
    var pos = from_index
    if pos < 0:
        pos = 0
    var last = text_len - pattern_len
    var first = pattern[unsafe_offset=0]
    # Skip-scan on the first code unit, then verify the tail.
    while pos <= last:
        while pos <= last and chars[unsafe_offset=Int(text_off + pos)] != first:
            pos += 1
        if pos > last:
            break
        var k = Int32(1)
        while k < pattern_len:
            if chars[unsafe_offset=Int(text_off + pos + k)] != pattern[
                unsafe_offset=Int(k)
            ]:
                break
            k += 1
        if k == pattern_len:
            return pos
        pos += 1
    return Int32(-1)


def _emit_pair(job: Pointer[JobState, MutUntrackedOrigin], start: Int32, end: Int32):
    """Append one [start, end] pair to the job's stash, growing it x2."""
    if job[].stash_len + 2 > job[].stash_cap:
        var new_cap = job[].stash_cap * 2
        var new_stash = unsafe_alloc[Int32](Int(new_cap))
        unsafe_memcpy(dest=new_stash, src=job[].stash, count=Int(job[].stash_len))
        job[].stash.unsafe_free()
        job[].stash = new_stash
        job[].stash_cap = new_cap
    job[].stash[unsafe_offset=Int(job[].stash_len)] = start
    job[].stash[unsafe_offset=Int(job[].stash_len + 1)] = end
    job[].stash_len += 2


def _convert_mask_to_indices(
    job: Pointer[JobState, MutUntrackedOrigin],
    match_mask: U8Ptr,
    text_len: Int32,
    min_match_char_length: Int32,
) -> Int32:
    """Turn a 0/1 match mask into [start, end] runs of length >= min length.

    Returns the number of pairs emitted (caller uses 0 to veto the match,
    mirroring the JS reference's empty-indices rule)."""
    var emitted = Int32(0)
    var start = Int32(-1)
    var i = Int32(0)
    while i < text_len:
        var m = match_mask[unsafe_offset=Int(i)]
        if m != 0 and start == -1:
            start = i
        elif m == 0 and start != -1:
            var end = i - 1
            if end - start + 1 >= min_match_char_length:
                _emit_pair(job, start, end)
                emitted += 1
            start = -1
        i += 1
    # Trailing run: matchmask[i-1] truthy implies start was set.
    if text_len > 0 and match_mask[unsafe_offset=Int(text_len - 1)] != 0:
        if text_len - start >= min_match_char_length:
            _emit_pair(job, start, text_len - 1)
            emitted += 1
    return emitted


def _bitap_search_one(
    idx: Pointer[FuseIndex, MutUntrackedOrigin],
    job: Pointer[JobState, MutUntrackedOrigin],
    text_index: Int32,
    pattern: U16Ptr,
    pattern_len: Int32,
    location_offset: Int32,
    exact_check: Int32,
    alphabet: I32Ptr,
    out_scores: F64Ptr,
    out_is_match: I32Ptr,
    out_idx_offsets: I32Ptr,
):
    """Replicate one text's Bitap search, writing score/isMatch/pair-count.

    `buf_a`/`buf_b` are ping-pong bit-array scratch of size >= text_len +
    pattern_len + 2. JavaScript's sparse bit arrays read holes back as
    undefined (which bitwise operators coerce to 0), so this kernel zeroes
    both buffers over the reachable window on the first error-level
    iteration: any slot not written during the immediately preceding
    iteration (below an early-break point, or outside a shrinking window)
    then correctly reads as 0, because a buffer is only ever read as `last`
    right after it was written as `cur`.
    """
    var text_off = idx[].offsets[unsafe_offset=Int(text_index)]
    var text_len = idx[].offsets[unsafe_offset=Int(text_index + 1)] - text_off
    var chars = idx[].chars
    var compute_matches = idx[].compute_matches
    var min_match_char_length = idx[].min_match_char_length
    var distance = idx[].distance
    var ignore_location = idx[].ignore_location
    var buf_a = job[].buf_a
    var buf_b = job[].buf_b
    var match_mask = job[].match_mask

    # Exact-equality fast path (whole pattern === whole text): score exactly 0,
    # always a match, indices [[0, len-1]] only when includeMatches.
    if exact_check != 0 and pattern_len == text_len:
        var same = Int32(1)
        var k = Int32(0)
        while k < pattern_len:
            if chars[unsafe_offset=Int(text_off + k)] != pattern[
                unsafe_offset=Int(k)
            ]:
                same = 0
                break
            k += 1
        if same != 0:
            out_scores[unsafe_offset=Int(text_index)] = Float64(0.0)
            out_is_match[unsafe_offset=Int(text_index)] = 1
            var base = Int32(0)
            if idx[].include_matches != 0:
                _emit_pair(job, Int32(0), text_len - 1)
                base += 1
            out_idx_offsets[unsafe_offset=Int(text_index + 1)] = base
            return

    # Clamp the expected location into [0, text_len].
    var expected_location = idx[].location + location_offset
    if expected_location < 0:
        expected_location = 0
    if expected_location > text_len:
        expected_location = text_len

    var current_threshold = idx[].threshold
    var best_location = expected_location

    if compute_matches != 0:
        # JS: matchMask = Array(textLen) (holes read as falsy) -> zero the mask.
        var z = Int32(0)
        while z < text_len:
            match_mask[unsafe_offset=Int(z)] = 0
            z += 1

    # Exact-substring pre-pass: tighten the threshold with every exact hit at
    # or after expected_location.
    var index = _index_of(
        chars, text_off, text_len, pattern, pattern_len, best_location
    )
    while index > -1:
        var score = _compute_score(
            pattern_len, Int32(0), index, expected_location, distance, ignore_location
        )
        if score < current_threshold:
            current_threshold = score
        best_location = index + pattern_len
        if compute_matches != 0:
            var k = Int32(0)
            while k < pattern_len:
                match_mask[unsafe_offset=Int(index + k)] = 1
                k += 1
        index = _index_of(
            chars, text_off, text_len, pattern, pattern_len, best_location
        )

    best_location = Int32(-1)
    var final_score = Float64(1.0)
    var bin_max = pattern_len + text_len
    var mask = Int32(1) << (pattern_len - 1)

    var cur = buf_a
    var last = buf_b

    # Both ping-pong buffers are zeroed over the reachable window on the first
    # error-level iteration (see below), once the first window is known.
    var i = Int32(0)
    while i < pattern_len:
        # Binary search: how far from expected_location may the match stray at
        # this error level before the score exceeds the current threshold.
        var bin_min = Int32(0)
        var bin_mid = bin_max
        while bin_min < bin_mid:
            var score = _compute_score(
                pattern_len, i, expected_location + bin_mid, expected_location, distance, ignore_location
            )
            if score <= current_threshold:
                bin_min = bin_mid
            else:
                bin_max = bin_mid
            bin_mid = (bin_max - bin_min) // 2 + bin_min
        bin_max = bin_mid

        var start = expected_location - bin_mid + 1
        if start < 1:
            start = 1
        var finish: Int32
        if idx[].find_all_matches != 0:
            finish = text_len
        else:
            finish = expected_location + bin_mid
            if finish > text_len:
                finish = text_len
            finish += pattern_len

        if i == 0:
            # Zero both ping-pong buffers over the reachable window so that
            # slots a later iteration reads but the previous iteration never
            # wrote (JS sparse array holes) come back as 0. Windows are
            # nested non-increasing across iterations, so [0, finish_0 + 2)
            # covers every later read.
            var zb = Int32(0)
            var zero_end = finish + 2
            while zb < zero_end:
                buf_a[unsafe_offset=Int(zb)] = 0
                buf_b[unsafe_offset=Int(zb)] = 0
                zb += 1

        # (1 << i) - 1 computed at 64-bit width: for i == 31 the JS reference
        # produces ToInt32((1 << 31) - 1) == 2147483647, the same value.
        cur[unsafe_offset=Int(finish + 1)] = Int32((Int64(1) << Int64(i)) - 1)

        var j = finish
        while j >= start:
            var current_location = j - 1
            var char_match = Int32(0)
            if current_location < text_len:
                char_match = alphabet[
                    unsafe_offset=Int(chars[unsafe_offset=Int(text_off + current_location)])
                ]
                if compute_matches != 0:
                    if char_match != 0:
                        match_mask[unsafe_offset=Int(current_location)] = 1
                    else:
                        match_mask[unsafe_offset=Int(current_location)] = 0
            # Positions >= text_len read a zero alphabet entry in JS
            # (charAt returns ''), and their matchMask writes are zeros past
            # the end, which cannot affect the extracted indices.
            var v = ((cur[unsafe_offset=Int(j + 1)] << 1) | 1) & char_match
            if i != 0:
                v |= (
                    ((last[unsafe_offset=Int(j + 1)] | last[unsafe_offset=Int(j)]) << 1)
                    | 1
                    | last[unsafe_offset=Int(j + 1)]
                )
            cur[unsafe_offset=Int(j)] = v
            if (v & mask) != 0:
                final_score = _compute_score(
                    pattern_len, i, current_location, expected_location, distance, ignore_location
                )
                if final_score <= current_threshold:
                    current_threshold = final_score
                    best_location = current_location
                    if best_location <= expected_location:
                        break
                    start = 2 * expected_location - best_location
                    if start < 1:
                        start = 1
            j -= 1

        var give_up = _compute_score(
            pattern_len, i + 1, expected_location, expected_location, distance, ignore_location
        )
        if give_up > current_threshold:
            break

        var tmp = cur
        cur = last
        last = tmp
        i += 1

    # Math.max(0.001, finalScore) applies whether or not the text matched.
    var out_score = final_score
    if out_score < 0.001:
        out_score = 0.001
    out_scores[unsafe_offset=Int(text_index)] = out_score

    var is_match = Int32(0)
    if best_location >= 0:
        is_match = 1
    var base = Int32(0)
    if compute_matches != 0:
        var emitted = _convert_mask_to_indices(
            job, match_mask, text_len, min_match_char_length
        )
        if emitted == 0:
            # Empty indices veto the match (the clamped score is still
            # reported and still folded into the caller's chunk average).
            is_match = 0
        base += emitted
    out_is_match[unsafe_offset=Int(text_index)] = is_match
    out_idx_offsets[unsafe_offset=Int(text_index + 1)] = base


@export
def fusemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def fusemojo_index_create(
    chars: U16Ptr,
    offsets: I32Ptr,
    n_texts: Int32,
    location: Int32,
    distance: Float64,
    threshold: Float64,
    min_match_char_length: Int32,
    find_all_matches: Int32,
    ignore_location: Int32,
    compute_matches: Int32,
    include_matches: Int32,
) abi("C") -> Handle:
    """Copy the caller's flat UTF-16 buffer into a native index; NULL on
    invalid input. All buffers are copied; the caller may free theirs."""
    if n_texts < 0 or min_match_char_length < 0:
        return None

    var total_chars = Int64(0)
    var max_text_len = Int32(0)
    if n_texts > 0:
        total_chars = Int64(offsets[unsafe_offset=Int(n_texts)])
        if total_chars < 0 or total_chars > 0x7FFFFFFF:
            return None
        var t = Int32(0)
        var prev = offsets[unsafe_offset=0]
        while t < n_texts:
            var cur = offsets[unsafe_offset=Int(t + 1)]
            var length = cur - prev
            if length < 0:
                return None
            if length > max_text_len:
                max_text_len = length
            prev = cur
            t += 1

    var chars_copy = unsafe_alloc[UInt16](Int(total_chars) + 1)
    if total_chars > 0:
        unsafe_memcpy(dest=chars_copy, src=chars, count=Int(total_chars))
    var offsets_copy = unsafe_alloc[Int32](Int(n_texts) + 1)
    unsafe_memcpy(dest=offsets_copy, src=offsets, count=Int(n_texts) + 1)

    var stash = unsafe_alloc[Int32](8192)
    var idx = unsafe_alloc[FuseIndex](1)
    idx[] = FuseIndex(
        n_texts,
        chars_copy,
        offsets_copy,
        max_text_len,
        location,
        distance,
        threshold,
        min_match_char_length,
        find_all_matches,
        ignore_location,
        compute_matches,
        include_matches,
        stash,
        8192,
    )
    return idx.unsafe_bitcast[UInt8]()


@export
def fusemojo_search_begin(
    handle: Handle,
    pattern: U16Ptr,
    pattern_len: Int32,
    location_offset: Int32,
    exact_check: Int32,
    out_scores: F64Ptr,
    out_is_match: I32Ptr,
    out_idx_offsets: I32Ptr,
    n_jobs: Int32,
) abi("C") -> Handle:
    """Set up one query over one pattern chunk (1..32 UTF-16 code units).

    Copies the pattern, builds the pattern alphabet, and allocates per-job
    scratch for up to `n_jobs` range jobs. `out_scores`/`out_is_match` must
    hold n_texts slots, `out_idx_offsets` n_texts + 1 slots; the output
    buffers stay owned (and pinned) by the caller for the whole query.
    Returns an opaque context, or NULL on invalid input.
    """
    if not handle:
        return None
    if pattern_len < 1 or pattern_len > MAX_PATTERN_LEN:
        return None
    if n_jobs < 1:
        return None
    var idx = handle.value().unsafe_bitcast[FuseIndex]()
    var max_text_len = idx[].max_text_len

    var pattern_copy = unsafe_alloc[UInt16](Int(pattern_len))
    unsafe_memcpy(dest=pattern_copy, src=pattern, count=Int(pattern_len))

    # Pattern alphabet: one Int32 mask per UTF-16 code unit.
    var alphabet = unsafe_alloc[Int32](ALPHABET_SIZE)
    var a = 0
    while a < ALPHABET_SIZE:
        alphabet[unsafe_offset=a] = 0
        a += 1
    var p = Int32(0)
    while p < pattern_len:
        var c = Int(pattern_copy[unsafe_offset=Int(p)])
        alphabet[unsafe_offset=c] = alphabet[unsafe_offset=c] | (
            Int32(1) << (pattern_len - p - 1)
        )
        p += 1

    # Per-job scratch: ping-pong bit arrays sized for the largest possible
    # window, a match mask, and a stash for produced index pairs.
    var scratch_len = Int(max_text_len) + Int(pattern_len) + 4
    var jobs = unsafe_alloc[JobState](Int(n_jobs))
    var job = Int32(0)
    while job < n_jobs:
        jobs[unsafe_offset=Int(job)] = JobState(
            unsafe_alloc[Int32](scratch_len),
            unsafe_alloc[Int32](scratch_len),
            unsafe_alloc[UInt8](Int(max_text_len) + 1),
            unsafe_alloc[Int32](4096),
            4096,
        )
        job += 1

    out_idx_offsets[unsafe_offset=0] = 0

    var ctx = unsafe_alloc[SearchCtx](1)
    ctx[] = SearchCtx(
        idx,
        pattern_copy,
        pattern_len,
        location_offset,
        exact_check,
        out_scores,
        out_is_match,
        out_idx_offsets,
        alphabet,
        n_jobs,
        jobs,
    )
    return ctx.unsafe_bitcast[UInt8]()


@export
def fusemojo_search_range(
    ctx_handle: Handle, job_id: Int32, start: Int32, end: Int32
) abi("C") -> Int32:
    """Score texts [start, end) of the context's query. Safe to run
    concurrently for disjoint ranges with distinct job ids; results are
    identical regardless of scheduling. Returns 0 on success."""
    if not ctx_handle:
        return 1
    var ctx = ctx_handle.value().unsafe_bitcast[SearchCtx]()
    if job_id < 0 or job_id >= ctx[].n_jobs:
        return 2
    var idx = ctx[].idx
    if start < 0:
        return 3
    var stop = end
    if stop > idx[].n_texts:
        stop = idx[].n_texts
    var job = ctx[].jobs.unsafe_offset(Int(job_id))
    var t = start
    while t < stop:
        _bitap_search_one(
            idx,
            job,
            t,
            ctx[].pattern,
            ctx[].pattern_len,
            ctx[].location_offset,
            ctx[].exact_check,
            ctx[].alphabet,
            ctx[].out_scores,
            ctx[].out_is_match,
            ctx[].out_idx_offsets,
        )
        t += 1
    return 0


@export
def fusemojo_search_end(ctx_handle: Handle) abi("C") -> Int32:
    """Finish the query: fold per-text pair counts into CSR offsets, concat
    the per-job stashes (job order == text order) into the index stash, free
    the context, and return the total pair count (<0 on a NULL context)."""
    if not ctx_handle:
        return -1
    var ctx = ctx_handle.value().unsafe_bitcast[SearchCtx]()
    var idx = ctx[].idx
    var n_texts = idx[].n_texts

    # Prefix-sum the per-text counts into CSR offsets and size the stash.
    var total = Int32(0)
    var t = Int32(0)
    while t < n_texts:
        total += ctx[].out_idx_offsets[unsafe_offset=Int(t + 1)]
        ctx[].out_idx_offsets[unsafe_offset=Int(t + 1)] = total
        t += 1

    var need = total * 2
    if need > idx[].stash_cap:
        var new_cap = idx[].stash_cap
        while new_cap < need:
            new_cap *= 2
        var new_stash = unsafe_alloc[Int32](Int(new_cap))
        idx[].stash.unsafe_free()
        idx[].stash = new_stash
        idx[].stash_cap = new_cap
    idx[].stash_len = need

    # Concatenate per-job stashes in job order (jobs cover disjoint ascending
    # ranges, so this is ascending text order).
    var write_at = Int32(0)
    var job = Int32(0)
    while job < ctx[].n_jobs:
        var len = ctx[].jobs[unsafe_offset=Int(job)].stash_len
        if len > 0:
            unsafe_memcpy(
                dest=idx[].stash.unsafe_offset(Int(write_at)),
                src=ctx[].jobs[unsafe_offset=Int(job)].stash,
                count=Int(len),
            )
            write_at += len
        job += 1

    # Free the context.
    job = 0
    while job < ctx[].n_jobs:
        ctx[].jobs[unsafe_offset=Int(job)].buf_a.unsafe_free()
        ctx[].jobs[unsafe_offset=Int(job)].buf_b.unsafe_free()
        ctx[].jobs[unsafe_offset=Int(job)].match_mask.unsafe_free()
        ctx[].jobs[unsafe_offset=Int(job)].stash.unsafe_free()
        job += 1
    ctx[].jobs.unsafe_free()
    ctx[].alphabet.unsafe_free()
    ctx[].pattern.unsafe_free()
    ctx.unsafe_free()
    return total


@export
def fusemojo_copy_indices(handle: Handle, out_pairs: I32Ptr) abi("C"):
    """Copy the stashed [start, end] pairs (2 * total_pairs Int32 values)
    from the most recent completed search into `out_pairs`."""
    if not handle:
        return
    var idx = handle.value().unsafe_bitcast[FuseIndex]()
    if idx[].stash_len > 0:
        unsafe_memcpy(dest=out_pairs, src=idx[].stash, count=Int(idx[].stash_len))


@export
def fusemojo_index_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var idx = handle.value().unsafe_bitcast[FuseIndex]()
    idx[].chars.unsafe_free()
    idx[].offsets.unsafe_free()
    idx[].stash.unsafe_free()
    idx.unsafe_free()
