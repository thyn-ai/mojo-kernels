"""Clean-room Windows registry-hive walker + pool-header constraint validator.

Written fresh against the documented on-disk formats and the publicly
described scanner semantics (the differential test-suite in this repository
checks the outputs against the volatility3 oracle, but no volatility3 code is
used or reproduced here):

  * Hive walk (scope: flat .dat hive images): pre-order traversal of the
    _CM_KEY_NODE tree starting at the base block's root cell, mirroring the
    traversal a `RegistryHive.visit_nodes` visitor observes. Per visited key
    node the kernel emits the node's cell-space offset (cell index + 4) and
    its name bytes (the Name field, NameLength bytes, latin-1 on the Python
    side). Subkey enumeration follows the two SubKeyLists slots in order;
    each slot may point to an "ri"/"lh"/"lf" index cell which is flattened
    depth-first in list order ("lh"/"lf" entries are (offset, hash) pairs and
    only the offset is followed). Entries that do not translate or do not
    decode are skipped silently, exactly like the reference traversal;
    entries whose cell type is not a key node or index are likewise skipped.
    A node whose own fields cannot be decoded (truncated hive) aborts the
    walk with a fatal status, which the Python wrapper surfaces as a
    HiveError (the reference raises an exception on the same input).

  * Pool-header validation: multi-pattern tag scan (leftmost, longest match
    at each position, resume after the matched tag) followed by per-hit
    constraint validation — block-size bounds (alignment * BlockSize against
    [size_min, size_max]), pool-type bitmask (FREE / NONPAGED / PAGED with
    pre-Vista or Vista+ semantics) and PoolIndex bounds. Header layouts:
    the x64 tables (byte1 = PoolIndex, byte2 = BlockSize, byte3 = PoolType,
    tag at +4) and the x86 table (9-bit BlockSize / 7-bit PoolIndex /
    7-bit PoolType packed in two little-endian u16 words, tag at +4).
    Header offsets are normalized through the scan layer's address mask
    ((1 << ceil(log2(size - 1))) - 1) exactly like the reference's object
    factory — a negative raw offset (tag in the first 4 bytes) wraps — and
    a needed field read that then falls outside the buffer skips the hit,
    mirroring the reference's InvalidAddressException handling; a hit whose
    constraint needs no fields always passes (with the masked offset).

Exported C ABI (batch-shaped: whole walk / whole scan per call):

    int32_t volatility3hivemojo_abi_version(void)

    int32_t volatility3hivemojo_hive_walk_count(
        data, data_len, root_cell, storage_len, volatile_len,
        out_n_nodes, out_n_name_bytes)

    int32_t volatility3hivemojo_hive_walk_fill(
        data, data_len, root_cell, storage_len, volatile_len,
        cap_nodes, cap_name_bytes,
        out_offsets, out_name_off, out_name_len, out_name_bytes, out_n_nodes)

        data          the flat .dat hive image (regf base block + hbins)
        root_cell     root cell index (from the base block; caller-side)
        storage_len   non-volatile storage length (invalid-cell bound)
        volatile_len  volatile storage length (0 for flat .dat files)
        out_offsets   int64[cap_nodes]  node struct offsets (cell index + 4)
        out_name_off  int64[cap_nodes]  byte offset of each name in the blob
        out_name_len  int32[cap_nodes]  length of each name
        out_name_bytes uint8[cap_name_bytes]  concatenated name bytes

        status 0 ok, 1 invalid arguments, 2 malformed hive (fatal; the
        reference raises on the same input), 4 output capacity exceeded,
        5 iteration guard tripped (pathological/cyclic image)

    int32_t volatility3hivemojo_pool_scan_count(
        data, data_len, n_constraints, tag_blob, tag_off, tag_len,
        size_min, size_max, page_type, index_min, index_max,
        alignment, layout, vista_semantics, out_n_hits)

    int32_t volatility3hivemojo_pool_scan_fill(
        ... same inputs ..., cap_hits, out_offsets, out_tag_idx, out_n_hits)

        constraints   pre-sorted by the caller with descending tag length
                      (guarantees the longest-match-at-position semantics);
                      size/index bounds of 0 mean "no bound" (matching the
                      reference's truthiness tests), page_type 0 means no
                      pool-type test
        layout        0 = x64 tables, 1 = x86 table
        out_offsets   int64[cap_hits]  header offsets ((tag offset - 4)
                      normalized through the scan layer's address mask)
        out_tag_idx   int32[cap_hits]  index into the given constraint arrays

        status 0 ok, 1 invalid arguments, 4 output capacity exceeded
"""

from std.collections import List
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library never retains pointers between calls).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]

# Cell signatures as little-endian u16 values ("nk" == b"n" | b"k" << 8).
comptime SIG_NK: UInt64 = 0x6B6E
comptime SIG_LF: UInt64 = 0x666C
comptime SIG_LH: UInt64 = 0x686C
comptime SIG_RI: UInt64 = 0x6972

comptime HIVE_BINS_BASE: UInt64 = 0x1000
comptime ADDR_MASK: UInt64 = 0x7FFFFFFF
comptime VOLATILE_BIT: UInt64 = 0x80000000

# _CM_KEY_NODE field offsets within the cell data (after the 4-byte size).
comptime NK_SUBKEY_LISTS: UInt64 = 0x1C
comptime NK_NAME_LENGTH: UInt64 = 0x48
comptime NK_NAME: UInt64 = 0x4C

# Guard against pathological/cyclic images: bound the total number of
# processed walk tasks (the reference would raise RecursionError or loop).
comptime OPS_CAP: UInt64 = 1 << 30

comptime POOL_LAYOUT_X64: Int32 = 0
comptime POOL_LAYOUT_X86: Int32 = 1
comptime POOL_TAG_OFFSET: Int64 = 4

comptime PAGE_TYPE_PAGED: Int32 = 1
comptime PAGE_TYPE_NONPAGED: Int32 = 2
comptime PAGE_TYPE_FREE: Int32 = 4


@export
def volatility3hivemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def read_u16le(data: U8Ptr, off: UInt64) -> UInt64:
    return UInt64(data[off]) | (UInt64(data[off + 1]) << 8)


def read_u32le(data: U8Ptr, off: UInt64) -> UInt64:
    return (
        UInt64(data[off])
        | (UInt64(data[off + 1]) << 8)
        | (UInt64(data[off + 2]) << 16)
        | (UInt64(data[off + 3]) << 32)
    )


def translate(
    off: UInt64,
    need: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    data_len: UInt64,
) -> UInt64:
    """Cell-space offset -> file offset; returns ~0 when not readable.

    Mirrors the reference layer's address check (offset above the storage
    length of the addressed storage class is invalid) for a flat hive image,
    where the hive bins map 1:1 after the 4 KiB base block.
    """
    var limit = storage_len
    if off & VOLATILE_BIT != 0:
        limit = volatile_len
    if (off & ADDR_MASK) > limit:
        return ~UInt64(0)
    var foff = HIVE_BINS_BASE + (off & ADDR_MASK)
    if foff + need > data_len or foff + need < foff:
        return ~UInt64(0)
    return foff


def push_entries(
    data: U8Ptr,
    data_len: UInt64,
    cell: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    mut dstack: List[UInt64],
    mut ops: UInt64,
) -> Int32:
    """Push an index cell's entries onto the task stack (reversed).

    The cell must decode as an ri/lh/lf index; if its Count/List fields are
    not readable the walk is fatal (the reference raises on the same input),
    reported as status 2.
    """
    var fsig = translate(cell + 4, 4, storage_len, volatile_len, data_len)
    if fsig == ~UInt64(0):
        return 2
    var sig = read_u16le(data, fsig)
    var stride: UInt64 = 0
    if sig == SIG_RI:
        stride = 4
    elif sig == SIG_LH or sig == SIG_LF:
        stride = 8
    else:
        return 0
    var count = read_u16le(data, fsig + 2)
    var entries_base = fsig + 4
    for j in range(count):
        ops += 1
        if ops > OPS_CAP:
            return 5
        var eoff = entries_base + UInt64(j) * stride
        if eoff + 4 > data_len:
            return 2
        dstack.append(read_u32le(data, eoff))
    # Reverse the just-pushed run in place so the LIFO pops in list order.
    var n = Int(count)
    var base = len(dstack) - n
    for k in range(n // 2):
        var tmp = dstack[base + k]
        dstack[base + k] = dstack[base + n - 1 - k]
        dstack[base + n - 1 - k] = tmp
    return 0


def descend(
    data: U8Ptr,
    data_len: UInt64,
    list_cell: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    maxaddr: UInt64,
    mut children: List[UInt64],
    mut ops: UInt64,
) -> Int32:
    """Flatten one SubKeyLists slot into `children` (in reference order).

    Returns 0 on success (unreadable/unknown cells are skipped), 2 when a
    reached index cell cannot be decoded (the reference raises there), 5
    when the iteration guard trips.
    """
    var ftop = translate(list_cell + 4, 2, storage_len, volatile_len, data_len)
    if ftop == ~UInt64(0):
        return 0  # the reference's signature read fails soft here: skip
    var top_sig = read_u16le(data, ftop)
    if top_sig != SIG_RI and top_sig != SIG_LH and top_sig != SIG_LF:
        return 0

    var dstack = List[UInt64]()
    var rc = push_entries(
        data, data_len, list_cell, storage_len, volatile_len, dstack, ops
    )
    if rc != 0:
        return rc
    while len(dstack) > 0:
        var e = dstack.pop()
        ops += 1
        if ops > OPS_CAP:
            return 5
        # Entries outside the address space, unreadable, or of an unexpected
        # cell type are skipped by the reference.
        if (e & ADDR_MASK) > maxaddr:
            continue
        var fe = translate(e + 4, 2, storage_len, volatile_len, data_len)
        if fe == ~UInt64(0):
            continue
        var esig = read_u16le(data, fe)
        if esig == SIG_NK:
            children.append(e)
        elif esig == SIG_RI or esig == SIG_LH or esig == SIG_LF:
            rc = push_entries(
                data, data_len, e, storage_len, volatile_len, dstack, ops
            )
            if rc != 0:
                return rc
        # anything else (vk/sk/db/unknown) is skipped by the reference
    return 0


def walk_core(
    data: U8Ptr,
    data_len: UInt64,
    root_cell: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    count_only: Bool,
    cap_nodes: UInt64,
    cap_name_bytes: UInt64,
    out_offsets: I64Ptr,
    out_name_off: I64Ptr,
    out_name_len: I32Ptr,
    out_name_bytes: U8Ptr,
    mut n_nodes: UInt64,
    mut n_name_bytes: UInt64,
) -> Int32:
    if data_len < HIVE_BINS_BASE:
        return 1
    var maxaddr = VOLATILE_BIT | volatile_len
    var stack = List[UInt64]()
    var children = List[UInt64]()
    stack.append(root_cell)
    n_nodes = 0
    n_name_bytes = 0
    var ops: UInt64 = 0
    while len(stack) > 0:
        var cell = stack.pop()
        ops += 1
        if ops > OPS_CAP:
            return 5
        # The node to visit must be a decodable key node; anything else is
        # fatal here (the reference raises for a bad visited node).
        var fsig = translate(cell + 4, 2, storage_len, volatile_len, data_len)
        if fsig == ~UInt64(0):
            return 2
        if read_u16le(data, fsig) != SIG_NK:
            return 2
        var fsl = translate(
            cell + 4 + NK_SUBKEY_LISTS, 8, storage_len, volatile_len, data_len
        )
        if fsl == ~UInt64(0):
            return 2
        var fnl = translate(
            cell + 4 + NK_NAME_LENGTH, 2, storage_len, volatile_len, data_len
        )
        if fnl == ~UInt64(0):
            return 2
        var name_len = read_u16le(data, fnl)
        var fname = translate(
            cell + 4 + NK_NAME, name_len, storage_len, volatile_len, data_len
        )
        if fname == ~UInt64(0):
            return 2
        if not count_only:
            if n_nodes >= cap_nodes or n_name_bytes + name_len > cap_name_bytes:
                return 4
            out_offsets[n_nodes] = Int64(cell + 4)
            out_name_off[n_nodes] = Int64(n_name_bytes)
            out_name_len[n_nodes] = Int32(name_len)
            for k in range(name_len):
                out_name_bytes[n_name_bytes + k] = data[fname + k]
        n_nodes += 1
        n_name_bytes += name_len
        # Children: SubKeyLists[0] then SubKeyLists[1], each flattened.
        children.clear()
        for idx in range(2):
            var sl = read_u32le(data, fsl + UInt64(idx) * 4)
            var rc = descend(
                data,
                data_len,
                sl,
                storage_len,
                volatile_len,
                maxaddr,
                children,
                ops,
            )
            if rc != 0:
                return rc
        # Push reversed so the stack pops children in traversal order.
        for i in range(len(children) - 1, -1, -1):
            stack.append(children[i])
    return 0


@export
def volatility3hivemojo_hive_walk_count(
    data: U8Ptr,
    data_len: Int64,
    root_cell: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    out_n_nodes: I64Ptr,
    out_n_name_bytes: I64Ptr,
) abi("C") -> Int32:
    """First pass: count nodes and total name bytes. Status codes per header."""
    if data_len <= 0:
        return 1
    # Scratch outputs (never written in count mode; valid pointers required).
    var s64a = unsafe_alloc[Int64](1)
    var s64b = unsafe_alloc[Int64](1)
    var s32 = unsafe_alloc[Int32](1)
    var s8 = unsafe_alloc[UInt8](1)
    var n_nodes: UInt64 = 0
    var n_bytes: UInt64 = 0
    var rc = walk_core(
        data,
        UInt64(data_len),
        root_cell,
        storage_len,
        volatile_len,
        True,
        0,
        0,
        s64a,
        s64b,
        s32,
        s8,
        n_nodes,
        n_bytes,
    )
    s64a.free()
    s64b.free()
    s32.free()
    s8.free()
    if rc != 0:
        return rc
    out_n_nodes[] = Int64(n_nodes)
    out_n_name_bytes[] = Int64(n_bytes)
    return 0


@export
def volatility3hivemojo_hive_walk_fill(
    data: U8Ptr,
    data_len: Int64,
    root_cell: UInt64,
    storage_len: UInt64,
    volatile_len: UInt64,
    cap_nodes: Int64,
    cap_name_bytes: Int64,
    out_offsets: I64Ptr,
    out_name_off: I64Ptr,
    out_name_len: I32Ptr,
    out_name_bytes: U8Ptr,
    out_n_nodes: I64Ptr,
) abi("C") -> Int32:
    """Second pass: fill the walk output. Capacities must cover the count pass."""
    if data_len <= 0 or cap_nodes < 0 or cap_name_bytes < 0:
        return 1
    var n_nodes: UInt64 = 0
    var n_bytes: UInt64 = 0
    var rc = walk_core(
        data,
        UInt64(data_len),
        root_cell,
        storage_len,
        volatile_len,
        False,
        UInt64(cap_nodes),
        UInt64(cap_name_bytes),
        out_offsets,
        out_name_off,
        out_name_len,
        out_name_bytes,
        n_nodes,
        n_bytes,
    )
    if rc != 0:
        return rc
    out_n_nodes[] = Int64(n_nodes)
    return 0


def pool_scan_core(
    data: U8Ptr,
    data_len: UInt64,
    n_constraints: UInt64,
    tag_blob: U8Ptr,
    tag_off: I32Ptr,
    tag_len: I32Ptr,
    size_min: I64Ptr,
    size_max: I64Ptr,
    page_type: I32Ptr,
    index_min: I64Ptr,
    index_max: I64Ptr,
    alignment: UInt64,
    layout: Int32,
    vista_semantics: Int32,
    count_only: Bool,
    cap_hits: UInt64,
    out_offsets: I64Ptr,
    out_tag_idx: I32Ptr,
    mut n_hits: UInt64,
) -> Int32:
    n_hits = 0
    if alignment == 0:
        return 1
    # The reference constructs headers on a flat layer whose address mask is
    # (1 << ceil(log2(size - 1))) - 1; object offsets are normalized through
    # it. Buffers smaller than 2 bytes cannot back such a layer.
    if data_len < 2:
        return 1
    var pow2: UInt64 = 1
    while pow2 < data_len - 1:
        pow2 <<= 1
    var addr_mask = pow2 - 1
    # First-byte filter: a position cannot start a tag unless its byte is
    # some tag's first byte. For <= 64 constraints a 256-entry bitmask
    # (bit i = constraint i, rows pre-sorted by descending tag length)
    # skips impossible positions in one lookup; larger sets use the plain
    # candidate loop. Output semantics are unchanged.
    var fbm = unsafe_alloc[UInt64](256)
    for k in range(256):
        fbm[k] = 0
    var use_bitmap = n_constraints <= 64
    if use_bitmap:
        for i in range(n_constraints):
            var first = tag_blob[UInt64(Int(tag_off[i]))]
            fbm[first] |= UInt64(1) << UInt64(i)
    var pos: UInt64 = 0
    while pos < data_len:
        # Longest tag matching at pos (caller pre-sorted by descending tag
        # length, so the first match found is the longest), mirroring the
        # reference scanner's leftmost non-overlapping multi-pattern pass.
        var matched: Int64 = -1
        if use_bitmap and fbm[data[pos]] == 0:
            pos += 1
            continue
        for i in range(n_constraints):
            if use_bitmap and (fbm[data[pos]] & (UInt64(1) << UInt64(i))) == 0:
                continue
            var tl = UInt64(Int(tag_len[i]))
            if tl == 0 or pos + tl > data_len:
                continue
            var toff = UInt64(Int(tag_off[i]))
            var same = True
            for k in range(tl):
                if data[pos + k] != tag_blob[toff + k]:
                    same = False
                    break
            if same:
                matched = Int64(i)
                break
        if matched < 0:
            pos += 1
            continue
        var i = UInt64(matched)
        var tl = UInt64(Int(tag_len[i]))
        # Per-hit header validation. Header = tag offset - 4, normalized
        # through the layer's address mask exactly like the reference
        # (a negative or oversized raw offset wraps; reads that then fall
        # outside the buffer skip the hit, like the reference's
        # InvalidAddressException handling).
        var h = (pos - UInt64(POOL_TAG_OFFSET)) & addr_mask
        var smin = size_min[i]
        var smax = size_max[i]
        var ptype_mask = page_type[i]
        var imin = index_min[i]
        var imax = index_max[i]
        var need_block = smin != 0 or smax != 0
        var need_type = ptype_mask != 0
        var need_index = imin != 0 or imax != 0
        # u16 member reads: BlockSize/PoolType word at relative offset 2
        # (member offsets are masked too), PoolIndex word at offset 0.
        var roff2 = (h + 2) & addr_mask
        var passes = True
        if (need_block or need_type) and roff2 + 2 > data_len:
            passes = False
        if passes and need_index and h + 2 > data_len:
            passes = False
        if passes and need_block:
            var block_size: UInt64 = 0
            if layout == POOL_LAYOUT_X64:
                block_size = UInt64(data[roff2])
            else:
                block_size = read_u16le(data, roff2) & 0x1FF
            var total = alignment * block_size
            if smin != 0 and total < UInt64(smin):
                passes = False
            if passes and smax != 0 and total > UInt64(smax):
                passes = False
        if passes and need_type:
            var pool_type: UInt64 = 0
            if layout == POOL_LAYOUT_X64:
                pool_type = UInt64(data[roff2 + 1])
            else:
                pool_type = (read_u16le(data, roff2) >> 9) & 0x7F
            var is_free = pool_type == 0
            var is_paged: Bool
            var is_nonpaged: Bool
            if vista_semantics != 0:
                is_paged = pool_type % 2 == 1
                is_nonpaged = pool_type % 2 == 0 and pool_type > 0
            else:
                is_paged = pool_type % 2 == 0 and pool_type > 0
                is_nonpaged = pool_type % 2 == 1
            var type_ok = False
            if ptype_mask & PAGE_TYPE_FREE != 0 and is_free:
                type_ok = True
            elif ptype_mask & PAGE_TYPE_NONPAGED != 0 and is_nonpaged:
                type_ok = True
            elif ptype_mask & PAGE_TYPE_PAGED != 0 and is_paged:
                type_ok = True
            if not type_ok:
                passes = False
        if passes and need_index:
            var pool_index: UInt64 = 0
            if layout == POOL_LAYOUT_X64:
                pool_index = UInt64(data[h + 1])
            else:
                pool_index = (read_u16le(data, h) >> 9) & 0x7F
            if imin != 0 and pool_index < UInt64(imin):
                passes = False
            if passes and imax != 0 and pool_index > UInt64(imax):
                passes = False
        if passes:
            if not count_only:
                if n_hits >= cap_hits:
                    fbm.free()
                    return 4
                out_offsets[n_hits] = Int64(h)
                out_tag_idx[n_hits] = Int32(i)
            n_hits += 1
        pos += tl
    fbm.free()
    return 0


@export
def volatility3hivemojo_pool_scan_count(
    data: U8Ptr,
    data_len: Int64,
    n_constraints: Int64,
    tag_blob: U8Ptr,
    tag_off: I32Ptr,
    tag_len: I32Ptr,
    size_min: I64Ptr,
    size_max: I64Ptr,
    page_type: I32Ptr,
    index_min: I64Ptr,
    index_max: I64Ptr,
    alignment: Int64,
    layout: Int32,
    vista_semantics: Int32,
    out_n_hits: I64Ptr,
) abi("C") -> Int32:
    """First pass: count passing hits. Status codes per header."""
    if data_len < 0 or n_constraints < 0 or alignment <= 0:
        return 1
    if layout != POOL_LAYOUT_X64 and layout != POOL_LAYOUT_X86:
        return 1
    var s64 = unsafe_alloc[Int64](1)
    var s32 = unsafe_alloc[Int32](1)
    var n_hits: UInt64 = 0
    var rc = pool_scan_core(
        data,
        UInt64(data_len),
        UInt64(n_constraints),
        tag_blob,
        tag_off,
        tag_len,
        size_min,
        size_max,
        page_type,
        index_min,
        index_max,
        UInt64(alignment),
        layout,
        vista_semantics,
        True,
        0,
        s64,
        s32,
        n_hits,
    )
    s64.free()
    s32.free()
    if rc != 0:
        return rc
    out_n_hits[] = Int64(n_hits)
    return 0


@export
def volatility3hivemojo_pool_scan_fill(
    data: U8Ptr,
    data_len: Int64,
    n_constraints: Int64,
    tag_blob: U8Ptr,
    tag_off: I32Ptr,
    tag_len: I32Ptr,
    size_min: I64Ptr,
    size_max: I64Ptr,
    page_type: I32Ptr,
    index_min: I64Ptr,
    index_max: I64Ptr,
    alignment: Int64,
    layout: Int32,
    vista_semantics: Int32,
    cap_hits: Int64,
    out_offsets: I64Ptr,
    out_tag_idx: I32Ptr,
    out_n_hits: I64Ptr,
) abi("C") -> Int32:
    """Second pass: fill the hit output. Capacity must cover the count pass."""
    if data_len < 0 or n_constraints < 0 or alignment <= 0 or cap_hits < 0:
        return 1
    if layout != POOL_LAYOUT_X64 and layout != POOL_LAYOUT_X86:
        return 1
    var n_hits: UInt64 = 0
    var rc = pool_scan_core(
        data,
        UInt64(data_len),
        UInt64(n_constraints),
        tag_blob,
        tag_off,
        tag_len,
        size_min,
        size_max,
        page_type,
        index_min,
        index_max,
        UInt64(alignment),
        layout,
        vista_semantics,
        False,
        UInt64(cap_hits),
        out_offsets,
        out_tag_idx,
        n_hits,
    )
    if rc != 0:
        return rc
    out_n_hits[] = Int64(n_hits)
    return 0
