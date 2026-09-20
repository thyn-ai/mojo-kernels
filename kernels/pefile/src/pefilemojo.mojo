"""Clean-room PE checksum and import-directory walker.

Written fresh from the Microsoft PE/COFF specification and the documented
behavior of the reference implementation (PyPI pefile 2024.8.26), whose
observable semantics this kernel reproduces bit-for-bit on well-formed files:

* ``generate_checksum`` — the dword one's-complement sum over the image with
  the CheckSum field skipped and the file zero-padded to a dword boundary,
  folded to 16 bits, plus the original file length.
* ``parse_import_directory`` / ``parse_imports`` — the import descriptor
  walk, ILT/IAT thunk-table walks, ordinal/hint/name resolution, bound-IAT
  detection, and pefile's name/dll charset validation and skip rules.

No third-party Mojo code is used or adapted.

Exported C ABI (v1):

    int32_t  pefilemojo_abi_version(void)
    int32_t  pefilemojo_checksum(data, len, checksum_offset, out_u64)
    int32_t  pefilemojo_imports_count(data, len, hdr..., sections..., out_counts[3])
    int32_t  pefilemojo_imports_fill(data, len, hdr..., sections..., caps, outs...)

The import walk is two-phase: ``count`` sizes the result (descriptors,
symbols, string-pool bytes), the caller allocates, and ``fill`` re-runs the
same deterministic walk writing into the caller's buffers.

Checksum exactness note: the reference folds the running sum mod (2**32 - 1)
per dword and keeps the "negative zero" representative 0xFFFFFFFF when the
total is a nonzero multiple of 2**32 - 1, which then folds to 0xFFFF instead
of 0. This kernel computes the provably identical result: exact block sums in
u64, exact reduction T = S mod (2**32 - 1), a nonzero-dword flag, and 0xFFFF
substituted iff T == 0 and any dword was nonzero. The differential suite
asserts bit-equality against the reference on every fixture.
"""

from std.collections import List
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]

# PE format constants.
comptime OPTIONAL_HEADER_MAGIC_PE: Int32 = 0x10B
comptime OPTIONAL_HEADER_MAGIC_PE_PLUS: Int32 = 0x20B
comptime IMAGE_ORDINAL_FLAG: UInt64 = 0x80000000
comptime IMAGE_ORDINAL_FLAG64: UInt64 = 0x8000000000000000
comptime MAX_IMPORT_SYMBOLS: Int64 = 0x2000
comptime MAX_IMPORT_NAME_LENGTH: Int64 = 0x200
comptime MAX_DLL_LENGTH: Int64 = 0x200
comptime MAX_ADDRESS_SPREAD: UInt64 = 128 * 1024 * 1024
comptime MAX_REPEATED_ADDRESSES: Int64 = 15
comptime ADDR_4GB: UInt64 = 0x100000000
comptime BOUND_NONE: UInt64 = 0xFFFFFFFFFFFFFFFF

comptime ERR_OK: Int32 = 0
comptime ERR_ARGS: Int32 = 1
comptime ERR_CAPACITY: Int32 = 2

# One's-complement modulus for the checksum fold (2**32 - 1).
comptime M32: UInt64 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


@export
def pefilemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def pefilemojo_checksum(
    data: U8Ptr,
    data_len: Int64,
    checksum_offset: Int64,
    out_val: U64Ptr,
) abi("C") -> Int32:
    """PE checksum: bit-exact with pefile.PE.generate_checksum's dword loop.

    ``checksum_offset`` is the file offset of the OptionalHeader.CheckSum
    field; that dword is skipped. The image is zero-padded to a dword
    boundary; the result is the 16-bit folded sum plus the original length.
    """
    if data_len < 0 or checksum_offset < 0:
        return ERR_ARGS

    var n_full = data_len >> 2
    var rem = data_len & 3
    var n_dwords = n_full + (Int64(1) if rem != 0 else Int64(0))
    var skip = checksum_offset >> 2

    var total = UInt64(0)  # running value of S mod (2**32 - 1), in [0, M32)
    var any_nonzero = False

    # Blocked accumulation: each block sums at most 2**20 dwords, so the u64
    # block sum stays below 2**52 and is reduced exactly afterwards.
    comptime BLOCK = Int64(1) << 20
    var i = Int64(0)
    while i < n_dwords:
        var block_end = i + BLOCK
        if block_end > n_dwords:
            block_end = n_dwords
        var block = UInt64(0)
        for j in range(i, block_end):
            if j == skip:
                continue
            var dword = UInt64(0)
            var base = j << 2
            if j < n_full:
                dword = (
                    UInt64(data[unsafe_offset=Int(base)])
                    | (UInt64(data[unsafe_offset=Int(base + 1)]) << 8)
                    | (UInt64(data[unsafe_offset=Int(base + 2)]) << 16)
                    | (UInt64(data[unsafe_offset=Int(base + 3)]) << 24)
                )
            else:
                # Zero-padded tail dword (pefile pads with b"\0" * (4 - rem)).
                for k in range(Int(rem)):
                    dword |= UInt64(data[unsafe_offset=Int(base + Int64(k))]) << (
                        8 * UInt64(k)
                    )
            block += dword
            if dword != 0:
                any_nonzero = True
        # Reduce block (< 2**52) and total+block (< 2**33) mod (2**32 - 1).
        block = (block & M32) + (block >> 32)
        block = (block & M32) + (block >> 32)
        if block >= M32:
            block -= M32
        total += block
        total = (total & M32) + (total >> 32)
        if total >= M32:
            total -= M32
        i = block_end

    # 16-bit one's-complement fold, exactly the reference's sequence.
    var c = total
    if c == 0 and any_nonzero:
        # The reference's running value is 0xFFFFFFFF here ("negative zero"),
        # which folds to 0xFFFF rather than 0.
        c = M32
    c = (c & 0xFFFF) + (c >> 16)
    c = c + (c >> 16)
    c = c & 0xFFFF
    out_val[unsafe_offset=0] = c + UInt64(data_len)
    return ERR_OK


# ---------------------------------------------------------------------------
# Section model (pefile SectionStructure semantics)
# ---------------------------------------------------------------------------


struct Section(Copyable, ImplicitlyCopyable, Movable):
    var vaddr: Int64  # VirtualAddress (raw)
    var vsize: Int64  # Misc_VirtualSize
    var rawsize: Int64  # SizeOfRawData
    var rawptr: Int64  # PointerToRawData (raw)
    var vaddr_adj: Int64  # adjust_SectionAlignment(VirtualAddress)
    var rawptr_adj: Int64  # adjust_PointerToRawData + LdrpWx86 rule
    var next_vaddr: Int64  # next section's raw VirtualAddress; -1 = none

    def __init__(
        out self,
        vaddr: Int64,
        vsize: Int64,
        rawsize: Int64,
        rawptr: Int64,
        vaddr_adj: Int64,
        rawptr_adj: Int64,
        next_vaddr: Int64,
    ):
        self.vaddr = vaddr
        self.vsize = vsize
        self.rawsize = rawsize
        self.rawptr = rawptr
        self.vaddr_adj = vaddr_adj
        self.rawptr_adj = rawptr_adj
        self.next_vaddr = next_vaddr


def _adjust_section_alignment(val: Int64, sa_in: Int64, fa: Int64) -> Int64:
    var sa = sa_in
    if sa < 0x1000:
        sa = fa
    if sa != 0 and val % sa != 0:
        return sa * (val // sa)
    return val


def _load_sections(
    n_sections: Int64,
    sec_vaddr: U32Ptr,
    sec_vsize: U32Ptr,
    sec_rawsize: U32Ptr,
    sec_rawptr: U32Ptr,
    section_alignment: Int64,
    file_alignment: Int64,
) -> List[Section]:
    """Sections must already be sorted by VirtualAddress (wrapper's job)."""
    var out = List[Section]()
    for i in range(Int(n_sections)):
        var vaddr = Int64(sec_vaddr[unsafe_offset=i])
        var rawptr = Int64(sec_rawptr[unsafe_offset=i])
        var vadj = _adjust_section_alignment(vaddr, section_alignment, file_alignment)
        var radj = rawptr & ~0x1FF
        if section_alignment < 0x1000 and rawptr == vaddr:
            radj = vaddr
        var nxt = Int64(-1)
        if i + 1 < Int(n_sections):
            nxt = Int64(sec_vaddr[unsafe_offset=i + 1])
        out.append(
            Section(
                vaddr,
                Int64(sec_vsize[unsafe_offset=i]),
                Int64(sec_rawsize[unsafe_offset=i]),
                rawptr,
                vadj,
                radj,
                nxt,
            )
        )
    return out^


def _contains_rva(s: Section, rva: Int64, data_len: Int64) -> Bool:
    var size: Int64
    if data_len - s.rawptr_adj < s.rawsize:
        size = s.vsize
    else:
        size = s.rawsize if s.rawsize > s.vsize else s.vsize
    if (
        s.next_vaddr >= 0
        and s.next_vaddr > s.vaddr
        and s.vaddr_adj + size > s.next_vaddr
    ):
        size = s.next_vaddr - s.vaddr_adj
    return s.vaddr_adj <= rva and rva < s.vaddr_adj + size


def _find_section(sections: List[Section], rva: Int64, data_len: Int64) -> Int64:
    for i in range(len(sections)):
        if _contains_rva(sections[i], rva, data_len):
            return Int64(i)
    return Int64(-1)


def _clamp_slice(a: Int64, b: Int64, data_len: Int64) -> Tuple[Int64, Int64]:
    """Python bytes[a:b] clamping (inputs are non-negative here)."""
    var lo = a if a < data_len else data_len
    var hi = b if b < data_len else data_len
    if hi < lo:
        hi = lo
    return Tuple(lo, hi)


def _section_get_data(
    s: Section, rva: Int64, length: Int64, data_len: Int64
) -> Tuple[Int64, Int64]:
    var offset = (rva - s.vaddr_adj) + s.rawptr_adj
    var end = offset + length
    # pefile trims against the UNADJUSTED PointerToRawData here.
    if end > s.rawptr + s.rawsize:
        end = s.rawptr + s.rawsize
    return _clamp_slice(offset, end, data_len)


struct DataSlice(Copyable, ImplicitlyCopyable, Movable):
    var start: Int64
    var end: Int64
    var ok: Bool

    def __init__(out self, start: Int64, end: Int64, ok: Bool):
        self.start = start
        self.end = end
        self.ok = ok


def _get_data(
    sections: List[Section],
    rva: Int64,
    length: Int64,
    data_len: Int64,
    header_len: Int64,
) -> DataSlice:
    """pefile PE.get_data: fails (PEFormatError) only when no section holds
    the RVA and the RVA lies past the end of the file."""
    var idx = _find_section(sections, rva, data_len)
    if idx >= 0:
        var sl = _section_get_data(sections[Int(idx)], rva, length, data_len)
        return DataSlice(sl[0], sl[1], True)
    if rva < data_len:
        # pefile tries self.header first, then the whole file; both are plain
        # slices of the same bytes, so one clamped slice covers both.
        var sl = _clamp_slice(rva, rva + length, data_len)
        return DataSlice(sl[0], sl[1], True)
    return DataSlice(0, 0, False)


def _get_string_at_rva(
    sections: List[Section],
    data: U8Ptr,
    rva: Int64,
    max_length: Int64,
    data_len: Int64,
) -> Tuple[Int64, Int64]:
    """Never fails (pefile semantics); NUL-trims within the fetched window."""
    var sl: Tuple[Int64, Int64]
    var idx = _find_section(sections, rva, data_len)
    if idx >= 0:
        sl = _section_get_data(sections[Int(idx)], rva, max_length, data_len)
    else:
        sl = _clamp_slice(rva, rva + max_length, data_len)
    var start = sl[0]
    var end = sl[1]
    var p = start
    while p < end:
        if data[unsafe_offset=Int(p)] == 0:
            end = p
            break
        p += 1
    return Tuple(start, end)


def _get_offset_from_rva(
    sections: List[Section], rva: Int64, data_len: Int64
) -> Tuple[Int64, Bool]:
    var idx = _find_section(sections, rva, data_len)
    if idx >= 0:
        var s = sections[Int(idx)]
        return Tuple(rva - s.vaddr_adj + s.rawptr_adj, True)
    if rva < data_len:
        return Tuple(rva, True)
    return Tuple(Int64(0), False)


# ---------------------------------------------------------------------------
# Charset validation (pefile is_valid_function_name / is_valid_dos_filename)
# ---------------------------------------------------------------------------


def _is_fn_char(c: UInt8) -> Bool:
    # a-z A-Z 0-9 plus "._?@$()<>"  (pefile allowed_function_name/allowed_extra)
    if c >= 97 and c <= 122:
        return True
    if c >= 65 and c <= 90:
        return True
    if c >= 48 and c <= 57:
        return True
    return (
        c == 0x2E  # .
        or c == 0x5F  # _
        or c == 0x3F  # ?
        or c == 0x40  # @
        or c == 0x24  # $
        or c == 0x28  # (
        or c == 0x29  # )
        or c == 0x3C  # <
        or c == 0x3E  # >
    )


def _is_dll_char(c: UInt8) -> Bool:
    # a-z A-Z 0-9 plus "!#$%&'()-@^_`{}~+,.;=[]:" plus path separators \/
    if _is_fn_char(c):
        return True
    return (
        c == 0x21  # !
        or c == 0x23  # #
        or c == 0x25  # %
        or c == 0x26  # &
        or c == 0x27  # '
        or c == 0x2D  # -
        or c == 0x5E  # ^
        or c == 0x60  # `
        or c == 0x7B  # {
        or c == 0x7D  # }
        or c == 0x7E  # ~
        or c == 0x2B  # +
        or c == 0x2C  # ,
        or c == 0x3B  # ;
        or c == 0x3D  # =
        or c == 0x5B  # [
        or c == 0x5D  # ]
        or c == 0x3A  # :
        or c == 0x5C  # backslash
        or c == 0x2F  # slash
    )


def _all_chars(data: U8Ptr, start: Int64, end: Int64, fn_mode: Bool) -> Bool:
    for p in range(Int(start), Int(end)):
        var c = data[unsafe_offset=p]
        if fn_mode:
            if not _is_fn_char(c):
                return False
        else:
            if not _is_dll_char(c):
                return False
    return True


# ---------------------------------------------------------------------------
# The import walk
# ---------------------------------------------------------------------------


struct WalkParams(Movable):
    var data: U8Ptr
    var data_len: Int64
    var pe_type: Int32
    var image_base: UInt64
    var import_dir_rva: Int64
    var header_len: Int64
    var sections: List[Section]

    def __init__(
        out self,
        data: U8Ptr,
        data_len: Int64,
        pe_type: Int32,
        image_base: UInt64,
        import_dir_rva: Int64,
        header_len: Int64,
        var sections: List[Section],
    ):
        self.data = data
        self.data_len = data_len
        self.pe_type = pe_type
        self.image_base = image_base
        self.import_dir_rva = import_dir_rva
        self.header_len = header_len
        self.sections = sections^


struct WalkOut(Movable):
    """Result sink: counting (fill=False) or writing caller buffers."""

    var n_desc: Int64
    var n_syms: Int64
    var pool_bytes: Int64
    var cap_desc: Int64
    var cap_syms: Int64
    var cap_pool: Int64
    var overflow: Bool
    var desc_dll_off: I64Ptr
    var desc_dll_len: I64Ptr
    var desc_sym_start: I64Ptr
    var desc_sym_count: I64Ptr
    var pool: U8Ptr
    var sym_name_off: I64Ptr
    var sym_name_len: I64Ptr
    var sym_ordinal: I64Ptr
    var sym_hint: I64Ptr
    var sym_address: U64Ptr
    var sym_bound: U64Ptr
    # Name length of every emitted symbol, in order; used to roll a
    # descriptor's symbols back when pefile would drop the descriptor.
    var emitted_lens: List[Int64]

    def __init__(
        out self,
        cap_desc: Int64,
        cap_syms: Int64,
        cap_pool: Int64,
        desc_dll_off: I64Ptr,
        desc_dll_len: I64Ptr,
        desc_sym_start: I64Ptr,
        desc_sym_count: I64Ptr,
        pool: U8Ptr,
        sym_name_off: I64Ptr,
        sym_name_len: I64Ptr,
        sym_ordinal: I64Ptr,
        sym_hint: I64Ptr,
        sym_address: U64Ptr,
        sym_bound: U64Ptr,
    ):
        self.n_desc = 0
        self.n_syms = 0
        self.pool_bytes = 0
        self.cap_desc = cap_desc
        self.cap_syms = cap_syms
        self.cap_pool = cap_pool
        self.overflow = False
        self.desc_dll_off = desc_dll_off
        self.desc_dll_len = desc_dll_len
        self.desc_sym_start = desc_sym_start
        self.desc_sym_count = desc_sym_count
        self.pool = pool
        self.sym_name_off = sym_name_off
        self.sym_name_len = sym_name_len
        self.sym_ordinal = sym_ordinal
        self.sym_hint = sym_hint
        self.sym_address = sym_address
        self.sym_bound = sym_bound
        self.emitted_lens = List[Int64]()


def _rollback_to(mut out: WalkOut, sym_start: Int64):
    """Drop every symbol emitted since sym_start, refunding pool bytes."""
    while out.n_syms > sym_start:
        var l = out.emitted_lens.pop()
        if l > 0:
            out.pool_bytes -= l
        out.n_syms -= 1


def _u16le(data: U8Ptr, off: Int64) -> Int64:
    return Int64(data[unsafe_offset=Int(off)]) | (
        Int64(data[unsafe_offset=Int(off + 1)]) << 8
    )


def _u32le(data: U8Ptr, off: Int64) -> Int64:
    return (
        Int64(data[unsafe_offset=Int(off)])
        | (Int64(data[unsafe_offset=Int(off + 1)]) << 8)
        | (Int64(data[unsafe_offset=Int(off + 2)]) << 16)
        | (Int64(data[unsafe_offset=Int(off + 3)]) << 24)
    )


def _u64le(data: U8Ptr, off: Int64) -> UInt64:
    return (
        UInt64(data[unsafe_offset=Int(off)])
        | (UInt64(data[unsafe_offset=Int(off + 1)]) << 8)
        | (UInt64(data[unsafe_offset=Int(off + 2)]) << 16)
        | (UInt64(data[unsafe_offset=Int(off + 3)]) << 24)
        | (UInt64(data[unsafe_offset=Int(off + 4)]) << 32)
        | (UInt64(data[unsafe_offset=Int(off + 5)]) << 40)
        | (UInt64(data[unsafe_offset=Int(off + 6)]) << 48)
        | (UInt64(data[unsafe_offset=Int(off + 7)]) << 56)
    )


struct AddrSet(Movable):
    """Membership + spread tracking for one thunk table (pefile AddressSet)."""

    var items: List[UInt64]
    var lo: UInt64
    var hi: UInt64
    var has_any: Bool

    def __init__(out self):
        self.items = List[UInt64]()
        self.lo = 0
        self.hi = 0
        self.has_any = False

    def contains(self, v: UInt64) -> Bool:
        for i in range(len(self.items)):
            if self.items[i] == v:
                return True
        return False

    def add(mut self, v: UInt64):
        self.items.append(v)
        if not self.has_any:
            self.lo = v
            self.hi = v
            self.has_any = True
        else:
            if v < self.lo:
                self.lo = v
            if v > self.hi:
                self.hi = v

    def diff(self) -> UInt64:
        if not self.has_any:
            return 0
        return self.hi - self.lo


def _walk_thunk_table(
    params: WalkParams,
    start_rva: Int64,
    max_length: Int64,
    mut total_symbols: Int64,
    mut fatal: Bool,
) -> List[UInt64]:
    """pefile get_import_table. fatal=True <=> pefile returned None."""
    fatal = False
    var table = List[UInt64]()
    var ordinal_flag = IMAGE_ORDINAL_FLAG
    var thunk_size = Int64(4)
    if params.pe_type == OPTIONAL_HEADER_MAGIC_PE_PLUS:
        ordinal_flag = IMAGE_ORDINAL_FLAG64
        thunk_size = 8
    var repeated_address = Int64(0)
    var set32 = AddrSet()
    var set64 = AddrSet()
    var rva = start_rva
    while rva != 0:
        if rva >= start_rva + max_length:
            break
        if total_symbols > MAX_IMPORT_SYMBOLS:
            break
        total_symbols += 1
        if repeated_address >= MAX_REPEATED_ADDRESSES:
            return List[UInt64]()  # pefile: return [] (empty, not fatal)
        if set32.diff() > MAX_ADDRESS_SPREAD or set64.diff() > MAX_ADDRESS_SPREAD:
            return List[UInt64]()
        var got = _get_data(
            params.sections, rva, thunk_size, params.data_len, params.header_len
        )
        if not got.ok or got.end - got.start != thunk_size:
            fatal = True  # pefile: return None
            return List[UInt64]()
        var aod: UInt64
        if thunk_size == 4:
            aod = UInt64(_u32le(params.data, got.start))
        else:
            aod = _u64le(params.data, got.start)
        # Overlap check: AddressOfData inside the scanned table => stop.
        if aod != 0 and aod >= UInt64(start_rva) and aod <= UInt64(rva):
            break
        if aod != 0:
            if (aod & ordinal_flag) != 0:
                # Literal 0x7FFFFFFF mask for both PE types (pefile behavior).
                if (aod & 0x7FFFFFFF) > 0xFFFF:
                    return List[UInt64]()  # corrupt: return []
            else:
                if aod >= ADDR_4GB:
                    if set64.contains(aod):
                        repeated_address += 1
                    set64.add(aod)
                else:
                    if set32.contains(aod):
                        repeated_address += 1
                    set32.add(aod)
        else:
            break  # all-zero terminator
        rva += thunk_size
        table.append(aod)
    return table^


def _parse_imports_of_descriptor(
    params: WalkParams,
    oft: Int64,
    ft: Int64,
    max_length: Int64,
    mut total_symbols: Int64,
    mut out: WalkOut,
    fill: Bool,
) -> Bool:
    """pefile parse_imports for one descriptor. True iff any symbol survives
    (False <=> pefile's import_data was empty or parsing raised)."""
    var ilt_fatal = False
    var iat_fatal = False
    var ilt = _walk_thunk_table(params, oft, max_length, total_symbols, ilt_fatal)
    var iat = _walk_thunk_table(params, ft, max_length, total_symbols, iat_fatal)
    if len(ilt) == 0 and len(iat) == 0:
        return False
    var sym_start = out.n_syms

    var ordinal_flag = IMAGE_ORDINAL_FLAG
    var address_mask = UInt64(0x7FFFFFFF)
    var imp_offset = Int64(4)
    if params.pe_type == OPTIONAL_HEADER_MAGIC_PE_PLUS:
        ordinal_flag = IMAGE_ORDINAL_FLAG64
        address_mask = 0x7FFFFFFFFFFFFFFF
        imp_offset = 8

    var use_ilt = len(ilt) > 0
    var n_table = len(ilt) if use_ilt else len(iat)
    var num_invalid = Int64(0)
    var idx = Int64(0)
    while idx < Int64(n_table):
        var aod = ilt[Int(idx)] if use_ilt else iat[Int(idx)]
        var imp_ord = Int64(-1)
        var imp_hint = Int64(-1)
        var name_start = Int64(-1)
        var name_end = Int64(-1)
        var name_invalid = False
        var fatal = False
        # Table entries are never zero (the walk stops at the terminator).
        if (aod & ordinal_flag) != 0:
            imp_ord = Int64(aod & 0xFFFF)
        else:
            var hint_rva = Int64(aod & address_mask)
            var hg = _get_data(
                params.sections, hint_rva, 2, params.data_len, params.header_len
            )
            if not hg.ok:
                fatal = True
            else:
                if hg.end - hg.start >= 2:
                    imp_hint = _u16le(params.data, hg.start)
                var ns = _get_string_at_rva(
                    params.sections,
                    params.data,
                    Int64(aod) + 2,
                    MAX_IMPORT_NAME_LENGTH,
                    params.data_len,
                )
                name_start = ns[0]
                name_end = ns[1]
                if _all_chars(params.data, name_start, name_end, True):
                    # get_offset_from_rva can still fail (PEFormatError).
                    var off = _get_offset_from_rva(
                        params.sections, Int64(aod) + 2, params.data_len
                    )
                    if not off[1]:
                        fatal = True
                else:
                    name_invalid = True  # pefile substitutes b"*invalid*"
        var imp_address = UInt64(ft) + params.image_base + UInt64(idx * imp_offset)
        var bound = BOUND_NONE
        if len(ilt) > 0 and len(iat) > 0 and idx < Int64(len(iat)):
            if ilt[Int(idx)] != iat[Int(idx)]:
                bound = iat[Int(idx)]
        if fatal:
            # pefile raises PEFormatError("Invalid entries, aborting parsing.");
            # the directory walk then treats the whole descriptor as empty.
            _rollback_to(out, sym_start)
            return False
        if name_invalid:
            # pefile: skip b"*invalid*" names, abort after 1000+ pure garbage.
            if num_invalid > 1000 and num_invalid == idx:
                _rollback_to(out, sym_start)
                return False
            num_invalid += 1
            idx += 1
            continue
        var has_name = name_start >= 0 and name_end > name_start
        if imp_ord > 0 or has_name:
            # pefile: `if imp_ord or imp_name:` (ordinal 0 and empty name drop)
            var pool_off = Int64(-1)
            var name_len = Int64(-1)
            if has_name:
                pool_off = out.pool_bytes
                name_len = name_end - name_start
            # Capacity contract: the caller sizes buffers from the count
            # phase's FINAL counters. A descriptor's symbols may transiently
            # exceed them here and then be rolled back (pefile drops the
            # descriptor); only rolled-back data ever exceeds the final
            # counters, so writes beyond capacity are skipped safely and the
            # final counters are validated at the end of the fill walk.
            if fill and out.n_syms < out.cap_syms and out.pool_bytes + (
                name_len if name_len > 0 else Int64(0)
            ) <= out.cap_pool:
                var i = out.n_syms
                out.sym_name_off[unsafe_offset=Int(i)] = pool_off
                out.sym_name_len[unsafe_offset=Int(i)] = name_len
                out.sym_ordinal[unsafe_offset=Int(i)] = imp_ord
                out.sym_hint[unsafe_offset=Int(i)] = imp_hint
                out.sym_address[unsafe_offset=Int(i)] = imp_address
                out.sym_bound[unsafe_offset=Int(i)] = bound
                if name_len > 0:
                    for k in range(Int(name_len)):
                        out.pool[unsafe_offset=Int(pool_off + Int64(k))] = params.data[
                            unsafe_offset=Int(name_start + Int64(k))
                        ]
            out.emitted_lens.append(name_len if name_len > 0 else Int64(0))
            out.n_syms += 1
            if name_len > 0:
                out.pool_bytes += name_len
        idx += 1

    # pefile: `if not import_data: error_count += 1; continue`
    return out.n_syms > sym_start


def _walk_imports(params: WalkParams, mut out: WalkOut, fill: Bool) -> Bool:
    """The full parse_import_directory walk. False <=> capacity overflow."""
    var rva = params.import_dir_rva
    var error_count = Int64(0)
    var total_symbols = Int64(0)

    while True:
        var got = _get_data(params.sections, rva, 20, params.data_len, params.header_len)
        if not got.ok:
            break
        if got.end - got.start < 20:
            break  # truncated descriptor (pefile raises struct.error; see docs)
        var base = got.start
        var all_zero = True
        for k in range(20):
            if params.data[unsafe_offset=Int(base + Int64(k))] != 0:
                all_zero = False
                break
        if all_zero:
            break
        var oft = _u32le(params.data, base)
        var name_rva = _u32le(params.data, base + 12)
        var ft = _u32le(params.data, base + 16)
        var file_offset = _get_offset_from_rva(params.sections, rva, params.data_len)[0]
        rva += 20
        var max_len = params.data_len - file_offset
        if rva > oft or rva > ft:
            var a = rva - oft
            var b = rva - ft
            max_len = a if a > b else b

        var sym_start = out.n_syms
        var had = _parse_imports_of_descriptor(
            params, oft, ft, max_len, total_symbols, out, fill
        )
        if out.overflow:
            return False
        if error_count > 5:
            break
        if not had:
            error_count += 1
            continue

        var ds = _get_string_at_rva(
            params.sections, params.data, name_rva, MAX_DLL_LENGTH, params.data_len
        )
        var dll_start = ds[0]
        var dll_end = ds[1]
        var dll_valid = _all_chars(params.data, dll_start, dll_end, False)
        var dll_len = dll_end - dll_start
        if not dll_valid:
            dll_len = 9  # len(b"*invalid*")
        if dll_len == 0:
            # pefile: `if dll:` — an empty name drops the whole descriptor.
            _rollback_to(out, sym_start)
            continue
        var pool_off = out.pool_bytes
        # Same capacity contract as for symbols: only rolled-back data ever
        # exceeds the count phase's final counters; skip the write, keep the
        # state updates in lockstep, and validate final counters after the
        # walk. (Descriptors are never rolled back, so a descriptor row
        # beyond capacity always means count/fill divergence.)
        if fill and out.n_desc < out.cap_desc and out.pool_bytes + dll_len <= out.cap_pool:
            var i = out.n_desc
            out.desc_dll_off[unsafe_offset=Int(i)] = pool_off
            out.desc_dll_len[unsafe_offset=Int(i)] = dll_len
            out.desc_sym_start[unsafe_offset=Int(i)] = sym_start
            out.desc_sym_count[unsafe_offset=Int(i)] = out.n_syms - sym_start
            if dll_valid:
                for k in range(Int(dll_len)):
                    out.pool[unsafe_offset=Int(pool_off + Int64(k))] = params.data[
                        unsafe_offset=Int(dll_start + Int64(k))
                    ]
            else:
                var invalid = "*invalid*".as_bytes()
                for k in range(9):
                    out.pool[unsafe_offset=Int(pool_off + Int64(k))] = invalid[k]
        out.n_desc += 1
        out.pool_bytes += dll_len
    if fill and (
        out.n_desc > out.cap_desc
        or out.n_syms > out.cap_syms
        or out.pool_bytes > out.cap_pool
    ):
        out.overflow = True
    return not out.overflow


def _make_params(
    data: U8Ptr,
    data_len: Int64,
    pe_type: Int32,
    image_base: UInt64,
    import_dir_rva: Int64,
    section_alignment: Int64,
    file_alignment: Int64,
    header_len: Int64,
    n_sections: Int64,
    sec_vaddr: U32Ptr,
    sec_vsize: U32Ptr,
    sec_rawsize: U32Ptr,
    sec_rawptr: U32Ptr,
    mut err: Int32,
) -> WalkParams:
    err = ERR_OK
    # Pointer arguments are never null when the associated count/length is
    # positive; the Python wrapper guarantees that (it allocates 1-slot
    # dummies for empty buffers). Mojo Pointers are non-null by design.
    if data_len < 0 or n_sections < 0 or import_dir_rva < 0 or header_len < 0:
        err = ERR_ARGS
    if header_len > data_len:
        err = ERR_ARGS
    var sections = _load_sections(
        n_sections,
        sec_vaddr,
        sec_vsize,
        sec_rawsize,
        sec_rawptr,
        section_alignment,
        file_alignment,
    )
    return WalkParams(
        data,
        data_len,
        pe_type,
        image_base,
        import_dir_rva,
        header_len,
        sections^,
    )


@export
def pefilemojo_imports_count(
    data: U8Ptr,
    data_len: Int64,
    pe_type: Int32,
    image_base: UInt64,
    import_dir_rva: Int64,
    section_alignment: Int64,
    file_alignment: Int64,
    header_len: Int64,
    n_sections: Int64,
    sec_vaddr: U32Ptr,
    sec_vsize: U32Ptr,
    sec_rawsize: U32Ptr,
    sec_rawptr: U32Ptr,
    out_counts: I64Ptr,
) abi("C") -> Int32:
    """Phase 1: size the import parse. out_counts = [n_desc, n_syms, pool_bytes]."""
    var err = Int32(0)
    var params = _make_params(
        data,
        data_len,
        pe_type,
        image_base,
        import_dir_rva,
        section_alignment,
        file_alignment,
        header_len,
        n_sections,
        sec_vaddr,
        sec_vsize,
        sec_rawsize,
        sec_rawptr,
        err,
    )
    if err != ERR_OK:
        return err
    # Count phase writes no output buffers (every write is guarded by `fill`),
    # but Mojo pointers are non-nullable, so the dummy outputs are carved from
    # one small scratch allocation that is freed before returning.
    var scratch = unsafe_alloc[UInt8](8 * 14)
    var d64 = scratch.unsafe_bitcast[Int64]()
    var du64 = scratch.unsafe_bitcast[UInt64]()
    var out = WalkOut(
        0,
        0,
        0,
        d64,
        d64.unsafe_offset(1),
        d64.unsafe_offset(2),
        d64.unsafe_offset(3),
        scratch,
        d64.unsafe_offset(4),
        d64.unsafe_offset(5),
        d64.unsafe_offset(6),
        d64.unsafe_offset(7),
        du64.unsafe_offset(8),
        du64.unsafe_offset(9),
    )
    _ = _walk_imports(params, out, False)
    out_counts[unsafe_offset=0] = out.n_desc
    out_counts[unsafe_offset=1] = out.n_syms
    out_counts[unsafe_offset=2] = out.pool_bytes
    scratch.unsafe_free()
    return ERR_OK


@export
def pefilemojo_imports_fill(
    data: U8Ptr,
    data_len: Int64,
    pe_type: Int32,
    image_base: UInt64,
    import_dir_rva: Int64,
    section_alignment: Int64,
    file_alignment: Int64,
    header_len: Int64,
    n_sections: Int64,
    sec_vaddr: U32Ptr,
    sec_vsize: U32Ptr,
    sec_rawsize: U32Ptr,
    sec_rawptr: U32Ptr,
    cap_desc: Int64,
    cap_syms: Int64,
    cap_pool: Int64,
    desc_dll_off: I64Ptr,
    desc_dll_len: I64Ptr,
    desc_sym_start: I64Ptr,
    desc_sym_count: I64Ptr,
    pool: U8Ptr,
    sym_name_off: I64Ptr,
    sym_name_len: I64Ptr,
    sym_ordinal: I64Ptr,
    sym_hint: I64Ptr,
    sym_address: U64Ptr,
    sym_bound: U64Ptr,
) abi("C") -> Int32:
    """Phase 2: re-run the deterministic walk into caller-sized buffers."""
    if cap_desc < 0 or cap_syms < 0 or cap_pool < 0:
        return ERR_ARGS
    # Output pointers are guaranteed non-null by the wrapper (1-slot dummies
    # for zero capacities); see _make_params.
    var err = Int32(0)
    var params = _make_params(
        data,
        data_len,
        pe_type,
        image_base,
        import_dir_rva,
        section_alignment,
        file_alignment,
        header_len,
        n_sections,
        sec_vaddr,
        sec_vsize,
        sec_rawsize,
        sec_rawptr,
        err,
    )
    if err != ERR_OK:
        return err
    var out = WalkOut(
        cap_desc,
        cap_syms,
        cap_pool,
        desc_dll_off,
        desc_dll_len,
        desc_sym_start,
        desc_sym_count,
        pool,
        sym_name_off,
        sym_name_len,
        sym_ordinal,
        sym_hint,
        sym_address,
        sym_bound,
    )
    if not _walk_imports(params, out, True):
        return ERR_CAPACITY
    return ERR_OK
