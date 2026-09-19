"""Clean-room sacrebleu statistics kernel (BLEU + chrF n-gram counting).

Written fresh from the published metric definitions (Papineni et al. 2002
BLEU sufficient statistics with clipped multi-reference n-gram counts;
Popovic 2015 chrF character/word n-gram precision-recall). No third-party
Mojo code is used or adapted. Tokenization happens in the Python wrapper;
this kernel receives integer-encoded sentences and performs only the
counting work (the hot nested Counter loops of the reference package).

Both entry points are batch-shaped and stateless: they accumulate corpus
statistics for sentence pairs [i0, i1) into caller-owned output buffers and
return 0 on success. If a sentence pair exceeds the per-pair packing limits
(>65534 distinct tokens/words or >1022 distinct characters in one pair,
which cannot produce an exact 64-bit n-gram key), the function stops and
returns (pair_index + 1); the wrapper then handles that single pair in
Python and calls again for the remainder. All counting is integer-exact.

Exported C ABI (v1):

    int32_t sacremojo_abi_version(void)
    int32_t sacremojo_bleu_stats(hyp_ids, hyp_off, ref_ids, ref_off,
                                 seg_index, i0, i1, stats)
        stats: int64[10] = correct[4], total[4], sys_len, ref_len
    int32_t sacremojo_chrf_stats(hypc, hypc_off, refc, refc_off, seg_index,
                                 hypw, hypw_off, refw, refw_off, i0, i1,
                                 char_order, word_order, beta,
                                 out_m, out_h, out_r)
        out_*: int64[char_order + word_order]

For chrF the kernel also selects, per sentence pair, the reference with the
highest sentence-level F score (ties keep the earliest) and accumulates
that reference's per-order statistics, mirroring the wrapper's pure-Python
fallback operation order exactly (IEEE-754 float64, no FMA contraction).
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Validated parameter bounds (the wrapper routes anything outside these
# ranges to its pure-Python fallback path).
comptime MAX_CHAR_ORDER: Int64 = 6
comptime MAX_WORD_ORDER: Int64 = 4
comptime MAX_ORDERS: Int64 = MAX_CHAR_ORDER + MAX_WORD_ORDER

# Per-pair distinct-id ceilings (local id + 1 must fit its key slot).
comptime MAX_LOCAL16: Int64 = 65535  # 16-bit slots (BLEU tokens, chrF words)
comptime MAX_LOCAL10: Int64 = 1023  # 10-bit slots (chrF characters)

comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


def _hash64(x: UInt64) -> UInt64:
    """splitmix64-style finalizer; deterministic across platforms."""
    var h = x ^ (x >> 33)
    h = h * 0xFF51AFD7ED558CCD
    h = h ^ (h >> 33)
    return h


def _next_pow2(x: Int64) -> Int64:
    var p: Int64 = 64
    while p < x:
        p = p * 2
    return p


def _alloc_u32_map(
    cap: Int64,
) -> Tuple[U32Ptr, U32Ptr, U32Ptr]:
    var keys = unsafe_alloc[UInt32](Int(cap))
    var vals = unsafe_alloc[UInt32](Int(cap))
    var gens = unsafe_alloc[UInt32](Int(cap))
    for i in range(Int(cap)):
        gens[unsafe_offset=i] = 0
    return keys, vals, gens


def _alloc_u64_map(
    cap: Int64,
) -> Tuple[U64Ptr, U32Ptr, U32Ptr]:
    var keys = unsafe_alloc[UInt64](Int(cap))
    var vals = unsafe_alloc[UInt32](Int(cap))
    var gens = unsafe_alloc[UInt32](Int(cap))
    for i in range(Int(cap)):
        gens[unsafe_offset=i] = 0
    return keys, vals, gens


struct Workspace(Copyable, Movable):
    """Grow-only scratch buffers shared across sentence pairs of one call.

    All maps are generation-stamped open-addressing tables, so clearing a
    map for a new (pair, order) is O(1) (bump the generation counter).
    """

    # local re-id map: key = global id + 1 (u32), value = local id (u32)
    var lkeys: U32Ptr
    var lvals: U32Ptr
    var lgens: U32Ptr
    var lcap: Int64
    var lgen: UInt32
    var lsize: Int64
    # n-gram count map (hypothesis): key = packed u64, value = count
    var ckeys: U64Ptr
    var cvals: U32Ptr
    var cgens: U32Ptr
    var ccap: Int64
    var cgen: UInt32
    # n-gram count map (current reference): key = packed u64, value = count
    var rkeys: U64Ptr
    var rvals: U32Ptr
    var rgens: U32Ptr
    var rcap: Int64
    var rgen: UInt32
    # BLEU reference max-count map: key = packed u64, value = max count
    var mkeys: U64Ptr
    var mvals: U32Ptr
    var mgens: U32Ptr
    var mcap: Int64
    var mgen: UInt32
    # re-ided token scratch: hyp / all refs of the current pair
    var hloc: U32Ptr
    var hloc_cap: Int64
    var rloc: U32Ptr
    var rloc_cap: Int64
    # re-ided word scratch (chrF word n-grams; second id namespace)
    var hloc_w: U32Ptr
    var hloc_w_cap: Int64
    var rloc_w: U32Ptr
    var rloc_w_cap: Int64
    # per-(pair, ref) order statistics and the winning reference's copy
    var cur: I64Ptr  # [3 * MAX_ORDERS] m, hh, rr for the current ref
    var best: I64Ptr  # [3 * MAX_ORDERS] m, hh, rr for the winning ref
    # hypothesis distinct n-gram (key, count) sections per order
    var hkeys: U64Ptr
    var hkeys_cap: Int64
    var hcounts: U32Ptr
    var hoff: I64Ptr  # [MAX_ORDERS + 1] section offsets

    def __init__(out self):
        var l = _alloc_u32_map(64)
        self.lkeys = l[0].copy()
        self.lvals = l[1].copy()
        self.lgens = l[2].copy()
        self.lcap = 64
        self.lgen = 0
        self.lsize = 0
        var c = _alloc_u64_map(64)
        self.ckeys = c[0].copy()
        self.cvals = c[1].copy()
        self.cgens = c[2].copy()
        self.ccap = 64
        self.cgen = 0
        var r = _alloc_u64_map(64)
        self.rkeys = r[0].copy()
        self.rvals = r[1].copy()
        self.rgens = r[2].copy()
        self.rcap = 64
        self.rgen = 0
        var m = _alloc_u64_map(64)
        self.mkeys = m[0].copy()
        self.mvals = m[1].copy()
        self.mgens = m[2].copy()
        self.mcap = 64
        self.mgen = 0
        self.hloc = unsafe_alloc[UInt32](64)
        self.hloc_cap = 64
        self.rloc = unsafe_alloc[UInt32](64)
        self.rloc_cap = 64
        self.hloc_w = unsafe_alloc[UInt32](64)
        self.hloc_w_cap = 64
        self.rloc_w = unsafe_alloc[UInt32](64)
        self.rloc_w_cap = 64
        self.cur = unsafe_alloc[Int64](Int(3 * MAX_ORDERS))
        self.best = unsafe_alloc[Int64](Int(3 * MAX_ORDERS))
        self.hkeys = unsafe_alloc[UInt64](64)
        self.hkeys_cap = 64
        self.hcounts = unsafe_alloc[UInt32](64)
        self.hoff = unsafe_alloc[Int64](Int(MAX_ORDERS + 1))

    def free(mut self):
        self.lkeys.unsafe_free()
        self.lvals.unsafe_free()
        self.lgens.unsafe_free()
        self.ckeys.unsafe_free()
        self.cvals.unsafe_free()
        self.cgens.unsafe_free()
        self.rkeys.unsafe_free()
        self.rvals.unsafe_free()
        self.rgens.unsafe_free()
        self.mkeys.unsafe_free()
        self.mvals.unsafe_free()
        self.mgens.unsafe_free()
        self.hloc.unsafe_free()
        self.rloc.unsafe_free()
        self.hloc_w.unsafe_free()
        self.rloc_w.unsafe_free()
        self.cur.unsafe_free()
        self.best.unsafe_free()
        self.hkeys.unsafe_free()
        self.hcounts.unsafe_free()
        self.hoff.unsafe_free()

    def ensure_map_capacity(mut self, total_tokens: Int64):
        """Size all four maps for one sentence pair (called at pair start)."""
        var need = _next_pow2(4 * total_tokens + 16)
        if need > self.lcap:
            self.lkeys.unsafe_free()
            self.lvals.unsafe_free()
            self.lgens.unsafe_free()
            var l = _alloc_u32_map(need)
            self.lkeys = l[0].copy()
            self.lvals = l[1].copy()
            self.lgens = l[2].copy()
            self.lcap = need
        if need > self.ccap:
            self.ckeys.unsafe_free()
            self.cvals.unsafe_free()
            self.cgens.unsafe_free()
            var c = _alloc_u64_map(need)
            self.ckeys = c[0].copy()
            self.cvals = c[1].copy()
            self.cgens = c[2].copy()
            self.ccap = need
        if need > self.rcap:
            self.rkeys.unsafe_free()
            self.rvals.unsafe_free()
            self.rgens.unsafe_free()
            var r = _alloc_u64_map(need)
            self.rkeys = r[0].copy()
            self.rvals = r[1].copy()
            self.rgens = r[2].copy()
            self.rcap = need
        if need > self.mcap:
            self.mkeys.unsafe_free()
            self.mvals.unsafe_free()
            self.mgens.unsafe_free()
            var m = _alloc_u64_map(need)
            self.mkeys = m[0].copy()
            self.mvals = m[1].copy()
            self.mgens = m[2].copy()
            self.mcap = need

    def ensure_scratch_capacity(
        mut self, hyp_tokens: Int64, ref_tokens: Int64
    ):
        if hyp_tokens > self.hloc_cap:
            self.hloc.unsafe_free()
            self.hloc_cap = _next_pow2(hyp_tokens + 16)
            self.hloc = unsafe_alloc[UInt32](Int(self.hloc_cap))
        if ref_tokens > self.rloc_cap:
            self.rloc.unsafe_free()
            self.rloc_cap = _next_pow2(ref_tokens + 16)
            self.rloc = unsafe_alloc[UInt32](Int(self.rloc_cap))

    def ensure_word_scratch_capacity(
        mut self, hyp_words: Int64, ref_words: Int64
    ):
        if hyp_words > self.hloc_w_cap:
            self.hloc_w.unsafe_free()
            self.hloc_w_cap = _next_pow2(hyp_words + 16)
            self.hloc_w = unsafe_alloc[UInt32](Int(self.hloc_w_cap))
        if ref_words > self.rloc_w_cap:
            self.rloc_w.unsafe_free()
            self.rloc_w_cap = _next_pow2(ref_words + 16)
            self.rloc_w = unsafe_alloc[UInt32](Int(self.rloc_w_cap))

    def ensure_hkey_capacity(mut self, total: Int64):
        if total > self.hkeys_cap:
            self.hkeys.unsafe_free()
            self.hcounts.unsafe_free()
            self.hkeys_cap = _next_pow2(total + 16)
            self.hkeys = unsafe_alloc[UInt64](Int(self.hkeys_cap))
            self.hcounts = unsafe_alloc[UInt32](Int(self.hkeys_cap))

    def local_id(mut self, gid: UInt32, max_id: Int64) -> Int64:
        """Map a global id to its per-pair local id; -1 when over the cap."""
        var slot = Int(_hash64(UInt64(gid + 1)) & UInt64(self.lcap - 1))
        while True:
            if self.lgens[unsafe_offset=slot] != self.lgen:
                if self.lsize >= max_id:
                    return -1
                self.lgens[unsafe_offset=slot] = self.lgen
                self.lkeys[unsafe_offset=slot] = gid + 1
                self.lvals[unsafe_offset=slot] = UInt32(self.lsize)
                self.lsize += 1
                return self.lsize - 1
            if self.lkeys[unsafe_offset=slot] == gid + 1:
                return Int64(self.lvals[unsafe_offset=slot])
            slot = (slot + 1) & Int(self.lcap - 1)

    def count_inc(mut self, key: UInt64):
        var slot = Int(_hash64(key) & UInt64(self.ccap - 1))
        while True:
            if self.cgens[unsafe_offset=slot] != self.cgen:
                self.cgens[unsafe_offset=slot] = self.cgen
                self.ckeys[unsafe_offset=slot] = key
                self.cvals[unsafe_offset=slot] = 1
                return
            if self.ckeys[unsafe_offset=slot] == key:
                self.cvals[unsafe_offset=slot] += 1
                return
            slot = (slot + 1) & Int(self.ccap - 1)

    def ref_inc(mut self, key: UInt64):
        var slot = Int(_hash64(key) & UInt64(self.rcap - 1))
        while True:
            if self.rgens[unsafe_offset=slot] != self.rgen:
                self.rgens[unsafe_offset=slot] = self.rgen
                self.rkeys[unsafe_offset=slot] = key
                self.rvals[unsafe_offset=slot] = 1
                return
            if self.rkeys[unsafe_offset=slot] == key:
                self.rvals[unsafe_offset=slot] += 1
                return
            slot = (slot + 1) & Int(self.rcap - 1)

    def max_update(mut self, key: UInt64, val: UInt32):
        var slot = Int(_hash64(key) & UInt64(self.mcap - 1))
        while True:
            if self.mgens[unsafe_offset=slot] != self.mgen:
                self.mgens[unsafe_offset=slot] = self.mgen
                self.mkeys[unsafe_offset=slot] = key
                self.mvals[unsafe_offset=slot] = val
                return
            if self.mkeys[unsafe_offset=slot] == key:
                if val > self.mvals[unsafe_offset=slot]:
                    self.mvals[unsafe_offset=slot] = val
                return
            slot = (slot + 1) & Int(self.mcap - 1)

    def max_get(mut self, key: UInt64) -> UInt32:
        var slot = Int(_hash64(key) & UInt64(self.mcap - 1))
        while True:
            if self.mgens[unsafe_offset=slot] != self.mgen:
                return 0
            if self.mkeys[unsafe_offset=slot] == key:
                return self.mvals[unsafe_offset=slot]
            slot = (slot + 1) & Int(self.mcap - 1)

    def ref_get(mut self, key: UInt64) -> UInt32:
        var slot = Int(_hash64(key) & UInt64(self.rcap - 1))
        while True:
            if self.rgens[unsafe_offset=slot] != self.rgen:
                return 0
            if self.rkeys[unsafe_offset=slot] == key:
                return self.rvals[unsafe_offset=slot]
            slot = (slot + 1) & Int(self.rcap - 1)


def _pack16(locs: U32Ptr, i: Int64, n: Int64) -> UInt64:
    """Pack n 16-bit local ids (id+1) starting at i into a u64 key."""
    var key = UInt64(0)
    for j in range(Int(n)):
        key = (key << 16) | UInt64(locs[unsafe_offset=Int(i) + j] + 1)
    return key


def _pack10(locs: U32Ptr, i: Int64, n: Int64) -> UInt64:
    """Pack n 10-bit local ids (id+1) starting at i into a u64 key."""
    var key = UInt64(0)
    for j in range(Int(n)):
        key = (key << 10) | UInt64(locs[unsafe_offset=Int(i) + j] + 1)
    return key


@export
def sacremojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def sacremojo_bleu_stats(
    hyp_ids: U32Ptr,
    hyp_off: I64Ptr,
    ref_ids: U32Ptr,
    ref_off: I64Ptr,
    seg_index: I64Ptr,
    i0: Int64,
    i1: Int64,
    stats: I64Ptr,
) abi("C") -> Int32:
    """Accumulate BLEU sufficient statistics for pairs [i0, i1) into stats.

    stats layout: correct[0..3], total[0..3], sys_len, ref_len. Returns 0 on
    success, or (pair_index + 1) when a pair has more than MAX_LOCAL16
    distinct tokens (the caller must process that pair itself and retry
    with i0 = pair_index + 1).
    """
    var ws = Workspace()
    for p in range(i0, i1):
        var hs = hyp_off[unsafe_offset=Int(p)]
        var he = hyp_off[unsafe_offset=Int(p + 1)]
        var hl = he - hs
        var s0 = seg_index[unsafe_offset=Int(p)]
        var s1 = seg_index[unsafe_offset=Int(p + 1)]
        if s1 <= s0:
            continue  # pair with no references: contributes nothing
        # closest reference length (ties keep the shorter one)
        var best_rl: Int64 = -1
        var total_ref: Int64 = 0
        for s in range(s0, s1):
            var rl = ref_off[unsafe_offset=Int(s + 1)] - ref_off[unsafe_offset=Int(s)]
            total_ref += rl
            var d = rl - hl
            if d < 0:
                d = -d
            var bd: Int64 = 0
            if best_rl >= 0:
                bd = best_rl - hl
                if bd < 0:
                    bd = -bd
            if best_rl < 0 or d < bd or (d == bd and rl < best_rl):
                best_rl = rl
        stats[unsafe_offset=8] += hl
        stats[unsafe_offset=9] += best_rl

        ws.ensure_map_capacity(hl + total_ref)
        ws.ensure_scratch_capacity(hl, total_ref)
        ws.lgen += 1
        ws.lsize = 0
        # re-id the hypothesis and all references of this pair
        var overflow = False
        for i in range(Int(hl)):
            var lid = ws.local_id(hyp_ids[unsafe_offset=Int(hs) + i], MAX_LOCAL16)
            if lid < 0:
                overflow = True
                break
            ws.hloc[unsafe_offset=i] = UInt32(lid)
        if not overflow:
            var rbase = ref_off[unsafe_offset=Int(s0)]
            for i in range(Int(total_ref)):
                var lid = ws.local_id(ref_ids[unsafe_offset=Int(rbase) + i], MAX_LOCAL16)
                if lid < 0:
                    overflow = True
                    break
                ws.rloc[unsafe_offset=i] = UInt32(lid)
        if overflow:
            ws.free()
            return Int32(p + 1)

        for n in range(Int64(1), Int64(5)):
            var hn = hl - n + 1
            if hn > 0:
                stats[unsafe_offset=4 + (n - 1)] += hn
                ws.cgen += 1
                for i in range(Int(hn)):
                    ws.count_inc(_pack16(ws.hloc, Int64(i), n))
            ws.mgen += 1
            for s in range(s0, s1):
                var rs = ref_off[unsafe_offset=Int(s)] - ref_off[unsafe_offset=Int(s0)]
                var rl = ref_off[unsafe_offset=Int(s + 1)] - ref_off[unsafe_offset=Int(s)]
                var rn = rl - n + 1
                if rn <= 0:
                    continue
                ws.rgen += 1
                for i in range(Int(rn)):
                    ws.ref_inc(_pack16(ws.rloc, rs + Int64(i), n))
                # merge this reference's counts into the max map
                for slot in range(Int(ws.rcap)):
                    if ws.rgens[unsafe_offset=slot] == ws.rgen:
                        ws.max_update(
                            ws.rkeys[unsafe_offset=slot],
                            ws.rvals[unsafe_offset=slot],
                        )
            if hn > 0:
                # clipped correct counts over hypothesis distinct n-grams
                var correct: Int64 = 0
                for slot in range(Int(ws.ccap)):
                    if ws.cgens[unsafe_offset=slot] == ws.cgen:
                        var hc = ws.cvals[unsafe_offset=slot]
                        var rc = ws.max_get(ws.ckeys[unsafe_offset=slot])
                        correct += Int64(hc if hc < rc else rc)
                stats[unsafe_offset=n - 1] += correct
    ws.free()
    return 0


def _chrf_fscore(
    cur: I64Ptr, orders: Int64, beta: Float64
) -> Float64:
    """Sentence-level chrF F (x100), mirroring the wrapper's op order.

    `cur` sections: match at [0, MAX_ORDERS), hyp totals at [MAX_ORDERS,
    2*MAX_ORDERS), ref totals at [2*MAX_ORDERS, 3*MAX_ORDERS).
    """
    var np: Int64 = 0
    var sp = Float64(0.0)
    var sr = Float64(0.0)
    for j in range(orders):
        var h = cur[unsafe_offset=MAX_ORDERS + j]
        var r = cur[unsafe_offset=2 * MAX_ORDERS + j]
        if h > 0 and r > 0:
            np += 1
            sp += Float64(cur[unsafe_offset=j]) / Float64(h)
            sr += Float64(cur[unsafe_offset=j]) / Float64(r)
    if np == 0:
        return 0.0
    var ap = sp / Float64(np)
    var ar = sr / Float64(np)
    var bs = beta * beta
    var den = bs * ap + ar
    if den == 0.0:
        return 0.0
    return (1.0 + bs) * ap * ar / den * 100.0


@export
def sacremojo_chrf_stats(
    hypc: U32Ptr,
    hypc_off: I64Ptr,
    refc: U32Ptr,
    refc_off: I64Ptr,
    seg_index: I64Ptr,
    hypw: U32Ptr,
    hypw_off: I64Ptr,
    refw: U32Ptr,
    refw_off: I64Ptr,
    i0: Int64,
    i1: Int64,
    char_order: Int64,
    word_order: Int64,
    beta: Float64,
    out_m: I64Ptr,
    out_h: I64Ptr,
    out_r: I64Ptr,
) abi("C") -> Int32:
    """Accumulate chrF per-order (match, hyp_total, ref_total) for [i0, i1).

    Per sentence pair, the reference with the highest sentence-level F
    score (computed from per-order stats, ties keep the earliest) is
    selected; its statistics are accumulated with the rule:
      - both totals > 0: accumulate match, hyp_total, ref_total
      - only ref_total > 0: accumulate ref_total
      - otherwise: accumulate nothing
    Character ids use 10-bit local packing, word ids 16-bit packing;
    a pair exceeding the per-pair distinct limits aborts with
    (pair_index + 1). Returns 2 on out-of-range parameters.
    """
    if (
        char_order < 1
        or char_order > MAX_CHAR_ORDER
        or word_order < 0
        or word_order > MAX_WORD_ORDER
        or i1 < i0
    ):
        return 2
    var orders = char_order + word_order
    var ws = Workspace()
    for p in range(i0, i1):
        var hs = hypc_off[unsafe_offset=Int(p)]
        var he = hypc_off[unsafe_offset=Int(p + 1)]
        var hl = he - hs
        var hws = hypw_off[unsafe_offset=Int(p)]
        var hwe = hypw_off[unsafe_offset=Int(p + 1)]
        var hwl = hwe - hws
        var s0 = seg_index[unsafe_offset=Int(p)]
        var s1 = seg_index[unsafe_offset=Int(p + 1)]
        if s1 <= s0:
            continue  # pair with no references: contributes nothing
        var total_ref_chars: Int64 = 0
        var total_ref_words: Int64 = 0
        for s in range(s0, s1):
            total_ref_chars += refc_off[unsafe_offset=Int(s + 1)] - refc_off[unsafe_offset=Int(s)]
            total_ref_words += refw_off[unsafe_offset=Int(s + 1)] - refw_off[unsafe_offset=Int(s)]

        var max_tokens = hl + total_ref_chars
        if hwl + total_ref_words > max_tokens:
            max_tokens = hwl + total_ref_words
        ws.ensure_map_capacity(max_tokens)
        ws.ensure_scratch_capacity(hl, total_ref_chars)
        ws.ensure_word_scratch_capacity(hwl, total_ref_words)

        # ---- re-id characters (10-bit local packing) ----
        ws.lgen += 1
        ws.lsize = 0
        var overflow = False
        for i in range(Int(hl)):
            var lid = ws.local_id(hypc[unsafe_offset=Int(hs) + i], MAX_LOCAL10)
            if lid < 0:
                overflow = True
                break
            ws.hloc[unsafe_offset=i] = UInt32(lid)
        if not overflow:
            var rcbase = refc_off[unsafe_offset=Int(s0)]
            for i in range(Int(total_ref_chars)):
                var lid = ws.local_id(refc[unsafe_offset=Int(rcbase) + i], MAX_LOCAL10)
                if lid < 0:
                    overflow = True
                    break
                ws.rloc[unsafe_offset=i] = UInt32(lid)
        # ---- re-id words (16-bit local packing, new generation) ----
        if not overflow and word_order > 0:
            ws.lgen += 1
            ws.lsize = 0
            for i in range(Int(hwl)):
                var lid = ws.local_id(hypw[unsafe_offset=Int(hws) + i], MAX_LOCAL16)
                if lid < 0:
                    overflow = True
                    break
                ws.hloc_w[unsafe_offset=i] = UInt32(lid)
            if not overflow:
                var rwbase = refw_off[unsafe_offset=Int(s0)]
                for i in range(Int(total_ref_words)):
                    var lid = ws.local_id(refw[unsafe_offset=Int(rwbase) + i], MAX_LOCAL16)
                    if lid < 0:
                        overflow = True
                        break
                    ws.rloc_w[unsafe_offset=i] = UInt32(lid)
        if overflow:
            ws.free()
            return Int32(p + 1)

        # ---- count hypothesis n-grams once per order and compact ----
        ws.ensure_hkey_capacity(char_order * hl + word_order * hwl)
        var total_distinct: Int64 = 0
        for n in range(Int64(1), char_order + 1):
            var j = n - 1
            ws.hoff[unsafe_offset=j] = total_distinct
            var hn = hl - n + 1
            ws.cgen += 1
            if hn > 0:
                for i in range(Int(hn)):
                    ws.count_inc(_pack10(ws.hloc, Int64(i), n))
                for slot in range(Int(ws.ccap)):
                    if ws.cgens[unsafe_offset=slot] == ws.cgen:
                        ws.hkeys[unsafe_offset=Int(total_distinct)] = ws.ckeys[unsafe_offset=slot]
                        ws.hcounts[unsafe_offset=Int(total_distinct)] = ws.cvals[unsafe_offset=slot]
                        total_distinct += 1
            ws.hoff[unsafe_offset=j + 1] = total_distinct
        for n in range(Int64(1), word_order + 1):
            var j = char_order + n - 1
            ws.hoff[unsafe_offset=j] = total_distinct
            var hn = hwl - n + 1
            ws.cgen += 1
            if hn > 0:
                for i in range(Int(hn)):
                    ws.count_inc(_pack16(ws.hloc_w, Int64(i), n))
                for slot in range(Int(ws.ccap)):
                    if ws.cgens[unsafe_offset=slot] == ws.cgen:
                        ws.hkeys[unsafe_offset=Int(total_distinct)] = ws.ckeys[unsafe_offset=slot]
                        ws.hcounts[unsafe_offset=Int(total_distinct)] = ws.cvals[unsafe_offset=slot]
                        total_distinct += 1
            ws.hoff[unsafe_offset=j + 1] = total_distinct

        # ---- per reference: per-order stats, combined-F selection ----
        var best_f = Float64(-1.0)
        var have_best = False
        for s in range(s0, s1):
            var rcs = refc_off[unsafe_offset=Int(s)] - refc_off[unsafe_offset=Int(s0)]
            var rcl = refc_off[unsafe_offset=Int(s + 1)] - refc_off[unsafe_offset=Int(s)]
            var rws = refw_off[unsafe_offset=Int(s)] - refw_off[unsafe_offset=Int(s0)]
            var rwl = refw_off[unsafe_offset=Int(s + 1)] - refw_off[unsafe_offset=Int(s)]
            for n in range(Int64(1), char_order + 1):
                var j = n - 1
                var hn = hl - n + 1
                var rn = rcl - n + 1
                var hh = hn if hn > 0 else Int64(0)
                var rr = rn if rn > 0 else Int64(0)
                ws.cur[unsafe_offset=MAX_ORDERS + j] = hh
                ws.cur[unsafe_offset=2 * MAX_ORDERS + j] = rr
                var m: Int64 = 0
                if hh > 0 and rr > 0:
                    ws.rgen += 1
                    for i in range(Int(rn)):
                        ws.ref_inc(_pack10(ws.rloc, rcs + Int64(i), n))
                    for e in range(Int(ws.hoff[unsafe_offset=j]), Int(ws.hoff[unsafe_offset=j + 1])):
                        var hc = ws.hcounts[unsafe_offset=e]
                        var rc = ws.ref_get(ws.hkeys[unsafe_offset=e])
                        m += Int64(hc if hc < rc else rc)
                ws.cur[unsafe_offset=j] = m
            for n in range(Int64(1), word_order + 1):
                var j = char_order + n - 1
                var hn = hwl - n + 1
                var rn = rwl - n + 1
                var hh = hn if hn > 0 else Int64(0)
                var rr = rn if rn > 0 else Int64(0)
                ws.cur[unsafe_offset=MAX_ORDERS + j] = hh
                ws.cur[unsafe_offset=2 * MAX_ORDERS + j] = rr
                var m: Int64 = 0
                if hh > 0 and rr > 0:
                    ws.rgen += 1
                    for i in range(Int(rn)):
                        ws.ref_inc(_pack16(ws.rloc_w, rws + Int64(i), n))
                    for e in range(Int(ws.hoff[unsafe_offset=j]), Int(ws.hoff[unsafe_offset=j + 1])):
                        var hc = ws.hcounts[unsafe_offset=e]
                        var rc = ws.ref_get(ws.hkeys[unsafe_offset=e])
                        m += Int64(hc if hc < rc else rc)
                ws.cur[unsafe_offset=j] = m
            var f = _chrf_fscore(ws.cur, orders, beta)
            if not have_best or f > best_f:
                best_f = f
                have_best = True
                for j in range(3 * MAX_ORDERS):
                    ws.best[unsafe_offset=j] = ws.cur[unsafe_offset=j]

        # ---- accumulate the winning reference's statistics ----
        for j in range(orders):
            var hh = ws.best[unsafe_offset=MAX_ORDERS + j]
            var rr = ws.best[unsafe_offset=2 * MAX_ORDERS + j]
            if hh > 0 and rr > 0:
                out_m[unsafe_offset=j] += ws.best[unsafe_offset=j]
                out_h[unsafe_offset=j] += hh
                out_r[unsafe_offset=j] += rr
            elif rr > 0:
                out_r[unsafe_offset=j] += rr
    ws.free()
    return 0
