"""Clean-room IP address bulk kernels: batch parse, batch membership, batch collapse.

Written fresh from RFC 791 / RFC 8200 (address formats), RFC 4632 (CIDR),
RFC 5952 (canonical text form) and the documented behavior of the CPython
3.12 standard-library `ipaddress` module, whose observable accept/reject
semantics this kernel reproduces exactly (the Python wrapper re-parses any
rejected item with the pure-Python reference, so exception types/messages
are byte-identical to the stdlib's on every backend). No third-party Mojo
code is used or adapted.

Exported C ABI (v1):

    int32_t  ipaddressmojo_abi_version(void)
    int32_t  ipaddressmojo_parse_batch(buf, offsets, n,
                                       out_version, out_hi, out_lo,
                                       out_scope_off, out_scope_len, out_err)
    int32_t  ipaddressmojo_contains_any_v4(net_lo, net_hi, n_nets,
                                           addrs, n_addrs, out)
    int32_t  ipaddressmojo_contains_any_v6(net_lo_hi, net_lo_lo, net_hi_hi,
                                           net_hi_lo, n_nets,
                                           addr_hi, addr_lo, n_addrs, out)
    int64_t  ipaddressmojo_collapse_v4(in_lo, in_plen, n, out_lo, out_plen)
    int64_t  ipaddressmojo_collapse_v6(in_lo_hi, in_lo_lo, in_plen, n,
                                       out_lo_hi, out_lo_lo, out_plen)

Batch membership builds a sort + prefix-max interval-stabbing structure:
with networks sorted by base address, an address is contained in *any*
network [lo_i, hi_i] with lo_i <= a iff max(hi_0..hi_i) >= a. Deterministic
heapsort (no introsort worst cases) and a branch-free binary search.

Batch collapse produces the canonical minimal CIDR cover (unique, so the
result is identical to the reference's merge by construction).
"""

from std.collections import List
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]

comptime ERR_OK: Int32 = 0
comptime ERR_ARGS: Int32 = 1

comptime U64_MAX: UInt64 = 0xFFFFFFFFFFFFFFFF


@export
def ipaddressmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------


def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


def _is_hex(b: UInt8) -> Bool:
    return (b >= 48 and b <= 57) or (b >= 97 and b <= 102) or (b >= 65 and b <= 70)


def _hex_val(b: UInt8) -> UInt32:
    if b <= 57:
        return UInt32(b - 48)
    if b >= 97:
        return UInt32(b - 87)
    return UInt32(b - 55)


def _find(data: U8Ptr, start: Int64, end: Int64, b: UInt8) -> Int64:
    """First index of byte b in [start, end), or -1."""
    for i in range(Int(start), Int(end)):
        if data[unsafe_offset=i] == b:
            return Int64(i)
    return -1


def _count_double_colon(data: U8Ptr, start: Int64, end: Int64) -> Int64:
    """Non-overlapping occurrences of '::' (Python str.count semantics)."""
    var count = Int64(0)
    var i = start
    while i + 1 < end:
        if data[unsafe_offset=Int(i)] == 58 and data[unsafe_offset=Int(i + 1)] == 58:
            count += 1
            i += 2
        else:
            i += 1
    return count


def _has_triple_colon(data: U8Ptr, start: Int64, end: Int64) -> Bool:
    var i = start
    while i + 2 < end:
        if (
            data[unsafe_offset=Int(i)] == 58
            and data[unsafe_offset=Int(i + 1)] == 58
            and data[unsafe_offset=Int(i + 2)] == 58
        ):
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# IPv4 parser (stdlib accept/reject semantics)
# ---------------------------------------------------------------------------


def _parse_v4(data: U8Ptr, start: Int64, end: Int64, mut out_val: UInt64) -> Bool:
    """Parse dotted-quad bytes [start, end). False = stdlib would reject."""
    if start == end:
        return False
    var value = UInt64(0)
    var n_parts = Int64(0)
    var part_start = start
    var i = start
    while i <= end:
        if i == end or data[unsafe_offset=Int(i)] == 46:  # '.'
            var plen = i - part_start
            if plen == 0:
                return False  # empty octet
            if plen > 3:
                return False  # octet has more than 3 characters
            var first = data[unsafe_offset=Int(part_start)]
            var v = UInt64(0)
            for k in range(Int(part_start), Int(i)):
                var b = data[unsafe_offset=k]
                if not _is_digit(b):
                    return False  # non-decimal digit
                v = v * 10 + UInt64(b - 48)
                if v > 255:
                    return False  # octet too big (saturating early is safe)
            if first == 48 and plen != 1:
                return False  # leading zeros
            value = (value << 8) | v
            n_parts += 1
            part_start = i + 1
        i += 1
    if n_parts != 4:
        return False
    out_val = value
    return True


# ---------------------------------------------------------------------------
# IPv6 parser (stdlib accept/reject semantics, including %scope ids)
# ---------------------------------------------------------------------------


def _parse_v6(
    data: U8Ptr,
    start: Int64,
    end: Int64,
    mut out_hi: UInt64,
    mut out_lo: UInt64,
    mut scope_off: Int32,
    mut scope_len: Int32,
) -> Bool:
    """Parse an IPv6 string (with optional %scope). False = stdlib rejects."""
    scope_off = -1
    scope_len = 0
    var addr_end = end
    var pct = _find(data, start, end, 37)  # '%'
    if pct >= 0:
        # Scope must be non-empty and contain no further '%'.
        if pct + 1 == end:
            return False
        if _find(data, pct + 1, end, 37) >= 0:
            return False
        scope_off = Int32(pct + 1 - start)
        scope_len = Int32(end - pct - 1)
        addr_end = pct
    if addr_end == start:
        return False  # address part empty

    # '::' multiplicity: more than one occurrence, or any ':::' present.
    if _count_double_colon(data, start, addr_end) > 1:
        return False
    if _has_triple_colon(data, start, addr_end):
        return False

    # Count colons and parts; locate the last part for embedded-v4 handling.
    var n_colons = Int64(0)
    var i = start
    while i < addr_end:
        if data[unsafe_offset=Int(i)] == 58:
            n_colons += 1
        i += 1
    var n_parts = n_colons + 1
    if n_parts < 3:
        return False

    # Embedded IPv4 is only allowed in the last part; it is parsed and
    # expanded to two hextets BEFORE the colon/parts count checks (so it
    # moves the "at most 8 colons" boundary by one, like the reference).
    var last_start = addr_end
    var j = addr_end
    while j > start:
        j -= 1
        if data[unsafe_offset=Int(j)] == 58:
            last_start = j + 1
            break
    var has_v4 = _find(data, last_start, addr_end, 46) >= 0
    var v4_val = UInt64(0)
    if has_v4:
        if not _parse_v4(data, last_start, addr_end, v4_val):
            return False

    if n_colons + (Int64(1) if has_v4 else Int64(0)) > 8:
        return False

    var has_dc = _count_double_colon(data, start, addr_end) == 1
    if has_dc:
        # The '::' must stand for at least one hextet: at most 7 other parts
        # (an embedded v4 counts as two).
        var n_other = Int64(0)
        var p = start
        var part_start = start
        while p <= addr_end:
            if p == addr_end or data[unsafe_offset=Int(p)] == 58:
                if p > part_start:
                    n_other += 1
                part_start = p + 1
            p += 1
        if has_v4:
            n_other += 1
        if n_other > 7:
            return False
    else:
        var total = n_parts + (Int64(1) if has_v4 else Int64(0))
        if total != 8:
            return False

    # Leading/trailing single ':' checks.
    if data[unsafe_offset=Int(start)] == 58 and not (
        start + 1 < addr_end and data[unsafe_offset=Int(start + 1)] == 58
    ):
        return False
    if data[unsafe_offset=Int(addr_end - 1)] == 58 and not (
        addr_end - 2 >= start and data[unsafe_offset=Int(addr_end - 2)] == 58
    ):
        return False

    # Validate hextets and assemble. Empty parts are the '::' marker (one or
    # two consecutive empties, position of the zero fill); the embedded v4
    # expands to two hextets. Any validation failure rejects the string.
    var hextets = List[UInt32]()
    var fill_at = Int64(-1)
    var p = start
    var part_start = start
    while p <= addr_end:
        if p == addr_end or data[unsafe_offset=Int(p)] == 58:
            var pe = p
            var ps = part_start
            if pe == ps:
                if fill_at < 0:
                    fill_at = Int64(len(hextets))
            elif has_v4 and ps == last_start:
                hextets.append(UInt32((v4_val >> 16) & 0xFFFF))
                hextets.append(UInt32(v4_val & 0xFFFF))
            else:
                if pe - ps > 4:
                    return False
                var v = UInt32(0)
                for k in range(Int(ps), Int(pe)):
                    var b = data[unsafe_offset=k]
                    if not _is_hex(b):
                        return False
                    v = (v << 4) | _hex_val(b)
                hextets.append(v)
            part_start = p + 1
        p += 1
    if len(hextets) > 8:
        return False  # defensive; the count checks above already reject

    var fill = 8 - len(hextets)
    var value_hi = UInt64(0)
    var value_lo = UInt64(0)
    for k in range(8):
        var h = UInt32(0)
        if fill_at < 0:
            h = hextets[k]
        elif k < Int(fill_at):
            h = hextets[k]
        elif k < Int(fill_at) + fill:
            h = 0
        else:
            h = hextets[k - fill]
        if k < 4:
            value_hi = (value_hi << 16) | UInt64(h)
        else:
            value_lo = (value_lo << 16) | UInt64(h)
    out_hi = value_hi
    out_lo = value_lo
    return True


# ---------------------------------------------------------------------------
# Batch parse entry point
# ---------------------------------------------------------------------------


@export
def ipaddressmojo_parse_batch(
    buf: U8Ptr,
    offsets: I64Ptr,
    n: Int64,
    out_version: I32Ptr,
    out_hi: U64Ptr,
    out_lo: U64Ptr,
    out_scope_off: I32Ptr,
    out_scope_len: I32Ptr,
    out_err: I32Ptr,
) abi("C") -> Int32:
    """Parse n items; version is chosen by presence of ':' (v6) like the
    stdlib factory (a v4 string can never contain ':'). out_err[i] != 0 marks
    a stdlib-rejected item; the wrapper re-parses it for the exact error."""
    if n < 0:
        return ERR_ARGS
    for i in range(Int(n)):
        var start = offsets[unsafe_offset=i]
        var end = offsets[unsafe_offset=i + 1]
        var ok = False
        var hi = UInt64(0)
        var lo = UInt64(0)
        var scope_off = Int32(-1)
        var scope_len = Int32(0)
        var version = Int32(4)
        if _find(buf, start, end, 58) >= 0:
            version = 6
            if _find(buf, start, end, 47) < 0:  # '/' never valid in an address
                ok = _parse_v6(buf, start, end, hi, lo, scope_off, scope_len)
        else:
            if _find(buf, start, end, 47) < 0:
                ok = _parse_v4(buf, start, end, lo)
        out_version[unsafe_offset=i] = version
        out_hi[unsafe_offset=i] = hi
        out_lo[unsafe_offset=i] = lo
        out_scope_off[unsafe_offset=i] = scope_off
        out_scope_len[unsafe_offset=i] = scope_len
        out_err[unsafe_offset=i] = Int32(0) if ok else Int32(1)
    return ERR_OK


# ---------------------------------------------------------------------------
# Heapsort (deterministic; no quicksort worst cases on sorted input)
# ---------------------------------------------------------------------------


def _sift_down_u64(keys: U64Ptr, start: Int64, end: Int64):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= end:
            break
        if child + 1 < end and keys[unsafe_offset=Int(child)] < keys[
            unsafe_offset=Int(child + 1)
        ]:
            child += 1
        if keys[unsafe_offset=Int(root)] < keys[unsafe_offset=Int(child)]:
            var tmp = keys[unsafe_offset=Int(root)]
            keys[unsafe_offset=Int(root)] = keys[unsafe_offset=Int(child)]
            keys[unsafe_offset=Int(child)] = tmp
            root = child
        else:
            break


def _heap_sort_u64(keys: U64Ptr, n: Int64):
    var start = n >> 1
    while start > 0:
        start -= 1
        _sift_down_u64(keys, start, n)
    var end = n
    while end > 1:
        end -= 1
        var tmp = keys[unsafe_offset=0]
        keys[unsafe_offset=0] = keys[unsafe_offset=Int(end)]
        keys[unsafe_offset=Int(end)] = tmp
        _sift_down_u64(keys, 0, end)


# ---------------------------------------------------------------------------
# contains_any: sort + prefix-max interval stabbing
# ---------------------------------------------------------------------------


@export
def ipaddressmojo_contains_any_v4(
    net_lo: U32Ptr,
    net_hi: U32Ptr,
    n_nets: Int64,
    addrs: U32Ptr,
    n_addrs: Int64,
    out_hits: U8Ptr,
) abi("C") -> Int32:
    if n_nets < 0 or n_addrs < 0:
        return ERR_ARGS
    if n_nets == 0:
        for i in range(Int(n_addrs)):
            out_hits[unsafe_offset=i] = 0
        return ERR_OK
    # Composite sort keys: (lo << 32) | hi, sorted ascending, then the hi
    # lane is replaced by the running max so a single binary search answers
    # "contained in any network".
    var keys = unsafe_alloc[UInt64](Int(n_nets))
    for i in range(Int(n_nets)):
        keys[unsafe_offset=i] = (UInt64(net_lo[unsafe_offset=i]) << 32) | UInt64(
            net_hi[unsafe_offset=i]
        )
    _heap_sort_u64(keys, n_nets)
    var best = UInt32(0)
    for i in range(Int(n_nets)):
        var k = keys[unsafe_offset=i]
        var hi = UInt32(k & 0xFFFFFFFF)
        if i == 0 or hi > best:
            best = hi
        keys[unsafe_offset=i] = (k & 0xFFFFFFFF00000000) | UInt64(best)
    for a in range(Int(n_addrs)):
        var v = UInt64(addrs[unsafe_offset=a])
        # Rightmost key with lo <= v: binary search on the hi lane (lo part).
        var lo_idx = Int64(-1)
        var lo_bound = Int64(0)
        var hi_bound = n_nets
        while hi_bound - lo_bound > 0:
            var mid = (lo_bound + hi_bound) >> 1
            if (keys[unsafe_offset=Int(mid)] >> 32) <= v:
                lo_idx = mid
                lo_bound = mid + 1
            else:
                hi_bound = mid
        out_hits[unsafe_offset=a] = UInt8(
            1 if lo_idx >= 0 and UInt32(keys[unsafe_offset=Int(lo_idx)] & 0xFFFFFFFF) >= UInt32(v) else 0
        )
    keys.unsafe_free()
    return ERR_OK


# u128 helpers: values are (hi, lo) pairs of UInt64.


def _sift_down_u128(
    k0: U64Ptr,
    k1: U64Ptr,
    k2: U64Ptr,
    k3: U64Ptr,
    start: Int64,
    end: Int64,
):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= end:
            break
        if child + 1 < end and _less_u128(k0, k1, k2, k3, child, child + 1):
            child += 1
        if _less_u128(k0, k1, k2, k3, root, child):
            _swap_u128(k0, k1, k2, k3, root, child)
            root = child
        else:
            break


def _less_u128(
    k0: U64Ptr, k1: U64Ptr, k2: U64Ptr, k3: U64Ptr, a: Int64, b: Int64
) -> Bool:
    """Lexicographic compare over four lanes (lo_hi, lo_lo, hi_hi, hi_lo)."""
    var a0 = k0[unsafe_offset=Int(a)]
    var b0 = k0[unsafe_offset=Int(b)]
    if a0 != b0:
        return a0 < b0
    var a1 = k1[unsafe_offset=Int(a)]
    var b1 = k1[unsafe_offset=Int(b)]
    if a1 != b1:
        return a1 < b1
    var a2 = k2[unsafe_offset=Int(a)]
    var b2 = k2[unsafe_offset=Int(b)]
    if a2 != b2:
        return a2 < b2
    return k3[unsafe_offset=Int(a)] < k3[unsafe_offset=Int(b)]


def _swap_u128(
    k0: U64Ptr, k1: U64Ptr, k2: U64Ptr, k3: U64Ptr, a: Int64, b: Int64
):
    var t0 = k0[unsafe_offset=Int(a)]
    var t1 = k1[unsafe_offset=Int(a)]
    var t2 = k2[unsafe_offset=Int(a)]
    var t3 = k3[unsafe_offset=Int(a)]
    k0[unsafe_offset=Int(a)] = k0[unsafe_offset=Int(b)]
    k1[unsafe_offset=Int(a)] = k1[unsafe_offset=Int(b)]
    k2[unsafe_offset=Int(a)] = k2[unsafe_offset=Int(b)]
    k3[unsafe_offset=Int(a)] = k3[unsafe_offset=Int(b)]
    k0[unsafe_offset=Int(b)] = t0
    k1[unsafe_offset=Int(b)] = t1
    k2[unsafe_offset=Int(b)] = t2
    k3[unsafe_offset=Int(b)] = t3


def _heap_sort_u128(k0: U64Ptr, k1: U64Ptr, k2: U64Ptr, k3: U64Ptr, n: Int64):
    var start = n >> 1
    while start > 0:
        start -= 1
        _sift_down_u128(k0, k1, k2, k3, start, n)
    var end = n
    while end > 1:
        end -= 1
        _swap_u128(k0, k1, k2, k3, 0, end)
        _sift_down_u128(k0, k1, k2, k3, 0, end)


@export
def ipaddressmojo_contains_any_v6(
    net_lo_hi: U64Ptr,
    net_lo_lo: U64Ptr,
    net_hi_hi: U64Ptr,
    net_hi_lo: U64Ptr,
    n_nets: Int64,
    addr_hi: U64Ptr,
    addr_lo: U64Ptr,
    n_addrs: Int64,
    out_hits: U8Ptr,
) abi("C") -> Int32:
    if n_nets < 0 or n_addrs < 0:
        return ERR_ARGS
    if n_nets == 0:
        for i in range(Int(n_addrs)):
            out_hits[unsafe_offset=i] = 0
        return ERR_OK
    # Sort the four input lanes in place (caller's arrays are scratch).
    _heap_sort_u128(net_lo_hi, net_lo_lo, net_hi_hi, net_hi_lo, n_nets)
    # Prefix-max over the hi pair, in place.
    var best_hi = net_hi_hi[unsafe_offset=0]
    var best_lo = net_hi_lo[unsafe_offset=0]
    for i in range(1, Int(n_nets)):
        var h_hi = net_hi_hi[unsafe_offset=i]
        var h_lo = net_hi_lo[unsafe_offset=i]
        if h_hi > best_hi or (h_hi == best_hi and h_lo > best_lo):
            best_hi = h_hi
            best_lo = h_lo
        else:
            net_hi_hi[unsafe_offset=i] = best_hi
            net_hi_lo[unsafe_offset=i] = best_lo
    for a in range(Int(n_addrs)):
        var v_hi = addr_hi[unsafe_offset=a]
        var v_lo = addr_lo[unsafe_offset=a]
        var idx = Int64(-1)
        var lo_bound = Int64(0)
        var hi_bound = n_nets
        while hi_bound - lo_bound > 0:
            var mid = (lo_bound + hi_bound) >> 1
            var m_hi = net_lo_hi[unsafe_offset=Int(mid)]
            var m_lo = net_lo_lo[unsafe_offset=Int(mid)]
            if m_hi < v_hi or (m_hi == v_hi and m_lo <= v_lo):
                idx = mid
                lo_bound = mid + 1
            else:
                hi_bound = mid
        var hit = False
        if idx >= 0:
            var b_hi = net_hi_hi[unsafe_offset=Int(idx)]
            var b_lo = net_hi_lo[unsafe_offset=Int(idx)]
            hit = b_hi > v_hi or (b_hi == v_hi and b_lo >= v_lo)
        out_hits[unsafe_offset=a] = UInt8(1) if hit else UInt8(0)
    return ERR_OK


# ---------------------------------------------------------------------------
# collapse: sort + canonical merge
# ---------------------------------------------------------------------------


@export
def ipaddressmojo_collapse_v4(
    in_lo: U32Ptr,
    in_plen: I32Ptr,
    n: Int64,
    out_lo: U32Ptr,
    out_plen: I32Ptr,
) abi("C") -> Int64:
    """Merge (base, prefixlen) into the canonical minimal cover. The output
    (never longer than the input) is written into out_*; returns the count."""
    if n < 0:
        return -1
    if n == 0:
        return 0
    # Sort by (base, prefixlen) via composite u64 keys, then merge on a
    # stack kept directly in the output arrays.
    var keys = unsafe_alloc[UInt64](Int(n))
    for i in range(Int(n)):
        keys[unsafe_offset=i] = UInt64(in_lo[unsafe_offset=i]) * 64 + UInt64(
            in_plen[unsafe_offset=i]
        )
    _heap_sort_u64(keys, n)
    var top = Int64(-1)  # stack top index in out_*
    var st_lo = List[UInt64]()
    var st_plen = List[Int64]()
    var st_hi = List[UInt64]()
    for i in range(Int(n)):
        var k = keys[unsafe_offset=i]
        var cur_lo = UInt64(k // 64)
        var cur_plen = Int64(k % 64)
        var cur_hi = cur_lo + (UInt64(1) << UInt64(32 - cur_plen)) - 1
        var alive = True
        while len(st_lo) > 0:
            var t_lo = st_lo[len(st_lo) - 1]
            var t_plen = st_plen[len(st_plen) - 1]
            var t_hi = st_hi[len(st_hi) - 1]
            if cur_lo >= t_lo and cur_hi <= t_hi:
                alive = False  # contained: drop
                break
            if (
                t_plen == cur_plen
                and t_hi < 0xFFFFFFFF
                and t_hi + 1 == cur_lo
                and t_lo % (UInt64(1) << UInt64(32 - t_plen + 1)) == 0
            ):
                # Siblings: merge into the parent and re-check.
                _ = st_lo.pop()
                _ = st_plen.pop()
                _ = st_hi.pop()
                cur_plen = t_plen - 1
                cur_hi = t_lo + (UInt64(1) << UInt64(32 - cur_plen)) - 1
                cur_lo = t_lo
                continue
            break
        if alive:
            st_lo.append(cur_lo)
            st_plen.append(cur_plen)
            st_hi.append(cur_hi)
    var count = Int64(len(st_lo))
    for i in range(Int(count)):
        out_lo[unsafe_offset=i] = UInt32(st_lo[i])
        out_plen[unsafe_offset=i] = Int32(st_plen[i])
    keys.unsafe_free()
    return count


# u128 collapse: sort (lo_hi, lo_lo, plen) with three lanes.


def _less3(
    k0: U64Ptr, k1: U64Ptr, k2: I32Ptr, a: Int64, b: Int64
) -> Bool:
    var a0 = k0[unsafe_offset=Int(a)]
    var b0 = k0[unsafe_offset=Int(b)]
    if a0 != b0:
        return a0 < b0
    var a1 = k1[unsafe_offset=Int(a)]
    var b1 = k1[unsafe_offset=Int(b)]
    if a1 != b1:
        return a1 < b1
    return k2[unsafe_offset=Int(a)] < k2[unsafe_offset=Int(b)]


def _swap3(k0: U64Ptr, k1: U64Ptr, k2: I32Ptr, a: Int64, b: Int64):
    var t0 = k0[unsafe_offset=Int(a)]
    var t1 = k1[unsafe_offset=Int(a)]
    var t2 = k2[unsafe_offset=Int(a)]
    k0[unsafe_offset=Int(a)] = k0[unsafe_offset=Int(b)]
    k1[unsafe_offset=Int(a)] = k1[unsafe_offset=Int(b)]
    k2[unsafe_offset=Int(a)] = k2[unsafe_offset=Int(b)]
    k0[unsafe_offset=Int(b)] = t0
    k1[unsafe_offset=Int(b)] = t1
    k2[unsafe_offset=Int(b)] = t2


def _sift3(k0: U64Ptr, k1: U64Ptr, k2: I32Ptr, start: Int64, end: Int64):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= end:
            break
        if child + 1 < end and _less3(k0, k1, k2, child, child + 1):
            child += 1
        if _less3(k0, k1, k2, root, child):
            _swap3(k0, k1, k2, root, child)
            root = child
        else:
            break


def _heap_sort3(k0: U64Ptr, k1: U64Ptr, k2: I32Ptr, n: Int64):
    var start = n >> 1
    while start > 0:
        start -= 1
        _sift3(k0, k1, k2, start, n)
    var end = n
    while end > 1:
        end -= 1
        _swap3(k0, k1, k2, 0, end)
        _sift3(k0, k1, k2, 0, end)


def _u128_add_size(hi: UInt64, lo: UInt64, shift: Int64) -> Tuple[UInt64, UInt64]:
    """(hi, lo) + (2^shift - 1), 128-bit. shift in [0, 128]."""
    var add_hi = UInt64(0)
    var add_lo = UInt64(0)
    if shift >= 128:
        add_hi = U64_MAX
        add_lo = U64_MAX
    elif shift >= 64:
        add_hi = (UInt64(1) << UInt64(shift - 64)) - 1
        add_lo = U64_MAX
    else:
        add_lo = (UInt64(1) << UInt64(shift)) - 1
    var new_lo = lo + add_lo
    var carry = UInt64(1) if new_lo < lo else UInt64(0)
    return (hi + add_hi + carry, new_lo)


def _u128_aligned(hi: UInt64, lo: UInt64, shift: Int64) -> Bool:
    """(hi, lo) % 2^shift == 0. shift in [1, 128]."""
    if shift >= 128:
        return hi == 0 and lo == 0
    if shift >= 64:
        if lo != 0:
            return False
        return hi % (UInt64(1) << UInt64(shift - 64)) == 0
    return lo % (UInt64(1) << UInt64(shift)) == 0


@export
def ipaddressmojo_collapse_v6(
    in_lo_hi: U64Ptr,
    in_lo_lo: U64Ptr,
    in_plen: I32Ptr,
    n: Int64,
    out_lo_hi: U64Ptr,
    out_lo_lo: U64Ptr,
    out_plen: I32Ptr,
) abi("C") -> Int64:
    if n < 0:
        return -1
    if n == 0:
        return 0
    # Sort the input lanes in place (caller's arrays are scratch).
    _heap_sort3(in_lo_hi, in_lo_lo, in_plen, n)
    var st_hi_hi = List[UInt64]()
    var st_hi_lo = List[UInt64]()
    var st_lo_hi = List[UInt64]()
    var st_lo_lo = List[UInt64]()
    var st_plen = List[Int64]()
    for i in range(Int(n)):
        var cur_lo_hi = in_lo_hi[unsafe_offset=i]
        var cur_lo_lo = in_lo_lo[unsafe_offset=i]
        var cur_plen = Int64(in_plen[unsafe_offset=i])
        var hi_pair = _u128_add_size(cur_lo_hi, cur_lo_lo, 128 - cur_plen)
        var cur_hi_hi = hi_pair[0]
        var cur_hi_lo = hi_pair[1]
        var alive = True
        while len(st_lo_hi) > 0:
            var t_lo_hi = st_lo_hi[len(st_lo_hi) - 1]
            var t_lo_lo = st_lo_lo[len(st_lo_lo) - 1]
            var t_hi_hi = st_hi_hi[len(st_hi_hi) - 1]
            var t_hi_lo = st_hi_lo[len(st_hi_lo) - 1]
            var t_plen = st_plen[len(st_plen) - 1]
            var contained = (
                (cur_lo_hi > t_lo_hi or (cur_lo_hi == t_lo_hi and cur_lo_lo >= t_lo_lo))
                and (
                    cur_hi_hi < t_hi_hi
                    or (cur_hi_hi == t_hi_hi and cur_hi_lo <= t_hi_lo)
                )
            )
            if contained:
                alive = False
                break
            var t_full = t_hi_hi == U64_MAX and t_hi_lo == U64_MAX
            var next_pair = _u128_add_size(t_hi_hi, t_hi_lo, 0)
            var adjacent = next_pair[0] == cur_lo_hi and next_pair[1] == cur_lo_lo
            if (
                t_plen == cur_plen
                and not t_full
                and adjacent
                and _u128_aligned(t_lo_hi, t_lo_lo, 128 - t_plen + 1)
            ):
                _ = st_lo_hi.pop()
                _ = st_lo_lo.pop()
                _ = st_hi_hi.pop()
                _ = st_hi_lo.pop()
                _ = st_plen.pop()
                cur_plen = t_plen - 1
                cur_lo_hi = t_lo_hi
                cur_lo_lo = t_lo_lo
                hi_pair = _u128_add_size(t_lo_hi, t_lo_lo, 128 - cur_plen)
                cur_hi_hi = hi_pair[0]
                cur_hi_lo = hi_pair[1]
                continue
            break
        if alive:
            st_lo_hi.append(cur_lo_hi)
            st_lo_lo.append(cur_lo_lo)
            st_hi_hi.append(cur_hi_hi)
            st_hi_lo.append(cur_hi_lo)
            st_plen.append(cur_plen)
    var count = Int64(len(st_plen))
    for i in range(Int(count)):
        out_lo_hi[unsafe_offset=i] = st_lo_hi[i]
        out_lo_lo[unsafe_offset=i] = st_lo_lo[i]
        out_plen[unsafe_offset=i] = Int32(st_plen[i])
    return count
