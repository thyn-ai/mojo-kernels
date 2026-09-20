"""Pure-Python reference implementation of the pefilemojo kernel semantics.

This is the fallback backend: it runs everywhere CPython runs and implements
exactly the same observable behavior as the Mojo kernel (the differential
suite asserts agreement with PyPI pefile 2024.8.26 for both backends):

* :func:`checksum` — the PE/COFF CheckSum dword one's-complement loop, with
  the CheckSum field skipped, zero padding to a dword boundary, the 16-bit
  fold, and the file length added. Bit-exact, including the "negative zero"
  case where a nonzero total congruent to 0 mod (2**32 - 1) folds to 0xFFFF.
* :func:`parse_imports` — the import descriptor/thunk-table walk with
  pefile's section model, charset validation, skip rules, and bound-IAT
  detection.

Written fresh from the PE/COFF specification and the documented behavior of
the reference implementation. No third-party code is used or adapted.
"""

from __future__ import annotations

OPTIONAL_HEADER_MAGIC_PE_PLUS = 0x20B
_IMAGE_ORDINAL_FLAG = 0x80000000
_IMAGE_ORDINAL_FLAG64 = 0x8000000000000000
_MAX_IMPORT_SYMBOLS = 0x2000
_MAX_IMPORT_NAME_LENGTH = 0x200
_MAX_DLL_LENGTH = 0x200
_MAX_ADDRESS_SPREAD = 128 * 2**20
_MAX_REPEATED_ADDRESSES = 15
_ADDR_4GB = 2**32
_M32 = 0xFFFFFFFF  # 2**32 - 1, the one's-complement modulus

_FUNCTION_NAME_CHARS = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._?@$()<>"
)
_DLL_NAME_CHARS = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b"!#$%&'()-@^_`{}~+,.;=[]:\\/"
)


def checksum(data: bytes | bytearray | memoryview, checksum_offset: int) -> int:
    """Bit-exact PE checksum (see module docstring for the fold semantics)."""
    view = memoryview(data)
    n = len(view)
    remainder = n % 4
    n_dwords = n // 4 + (1 if remainder else 0)
    skip = checksum_offset // 4

    total = 0  # running S mod (2**32 - 1), always in [0, 2**32 - 1)
    any_nonzero = False
    i = 0
    while i < n_dwords:
        block_end = min(i + (1 << 20), n_dwords)
        block = 0
        for j in range(i, block_end):
            if j == skip:
                continue
            base = j * 4
            if j < n // 4:
                dword = int.from_bytes(view[base : base + 4], "little")
            else:
                dword = int.from_bytes(view[base:n], "little")
            block += dword
            if dword:
                any_nonzero = True
        # Exact reduction mod (2**32 - 1); block < 2**52, total < 2**33.
        block = (block & _M32) + (block >> 32)
        block = (block & _M32) + (block >> 32)
        if block >= _M32:
            block -= _M32
        total += block
        total = (total & _M32) + (total >> 32)
        if total >= _M32:
            total -= _M32
        i = block_end

    c = total
    if c == 0 and any_nonzero:
        c = _M32  # "negative zero" folds to 0xFFFF, not 0
    c = (c & 0xFFFF) + (c >> 16)
    c = c + (c >> 16)
    c = c & 0xFFFF
    return c + n


class _Section:
    """pefile SectionStructure semantics (adjusted addresses, raw bounds)."""

    __slots__ = ("vaddr", "vsize", "rawsize", "rawptr", "vaddr_adj", "rawptr_adj",
                 "next_vaddr")

    def __init__(self, vaddr, vsize, rawsize, rawptr, section_alignment, file_alignment,
                 next_vaddr):
        self.vaddr = vaddr
        self.vsize = vsize
        self.rawsize = rawsize
        self.rawptr = rawptr
        self.vaddr_adj = _adjust_section_alignment(vaddr, section_alignment,
                                                   file_alignment)
        rawptr_adj = rawptr & ~0x1FF
        if section_alignment < 0x1000 and rawptr == vaddr:
            rawptr_adj = vaddr
        self.rawptr_adj = rawptr_adj
        self.next_vaddr = next_vaddr

    def contains_rva(self, rva, data_len):
        if data_len - self.rawptr_adj < self.rawsize:
            size = self.vsize
        else:
            size = max(self.rawsize, self.vsize)
        if (self.next_vaddr is not None and self.next_vaddr > self.vaddr
                and self.vaddr_adj + size > self.next_vaddr):
            size = self.next_vaddr - self.vaddr_adj
        return self.vaddr_adj <= rva < self.vaddr_adj + size

    def get_data(self, rva, length, data_len):
        offset = (rva - self.vaddr_adj) + self.rawptr_adj
        end = offset + length
        # pefile trims against the UNADJUSTED PointerToRawData here.
        if end > self.rawptr + self.rawsize:
            end = self.rawptr + self.rawsize
        return _clamp_slice(offset, end, data_len)


def _adjust_section_alignment(val, section_alignment, file_alignment):
    sa = section_alignment
    if sa < 0x1000:
        sa = file_alignment
    if sa and val % sa:
        return sa * (val // sa)
    return val


def _clamp_slice(a, b, data_len):
    lo = min(a, data_len)
    hi = min(b, data_len)
    return lo, max(lo, hi)


def make_sections(section_rows, section_alignment, file_alignment):
    """section_rows: sorted (vaddr, vsize, rawsize, rawptr) tuples."""
    out = []
    n = len(section_rows)
    for i, (vaddr, vsize, rawsize, rawptr) in enumerate(section_rows):
        next_vaddr = section_rows[i + 1][0] if i + 1 < n else None
        out.append(
            _Section(vaddr, vsize, rawsize, rawptr, section_alignment, file_alignment,
                     next_vaddr)
        )
    return out


class _Walker:
    """The import-directory walk over one image (pefile parse semantics)."""

    def __init__(self, data, pe_type, image_base, import_dir_rva, sections, header_len):
        self.data = memoryview(data)
        self.data_len = len(self.data)
        self.pe_type = pe_type
        self.image_base = image_base
        self.import_dir_rva = import_dir_rva
        self.sections = sections
        self.header_len = header_len
        self.total_symbols = 0

    # -- data access --------------------------------------------------------

    def _find_section(self, rva):
        for s in self.sections:
            if s.contains_rva(rva, self.data_len):
                return s
        return None

    def get_data(self, rva, length):
        """pefile PE.get_data -> (start, end) slice, or None (PEFormatError)."""
        s = self._find_section(rva)
        if s is not None:
            return s.get_data(rva, length, self.data_len)
        if rva < self.data_len:
            return _clamp_slice(rva, rva + length, self.data_len)
        return None

    def get_string_at_rva(self, rva, max_length):
        s = self._find_section(rva)
        if s is not None:
            start, end = s.get_data(rva, max_length, self.data_len)
        else:
            start, end = _clamp_slice(rva, rva + max_length, self.data_len)
        nul = bytes(self.data[start:end]).find(b"\0")
        if nul >= 0:
            end = start + nul
        return start, end

    def get_offset_from_rva(self, rva):
        s = self._find_section(rva)
        if s is not None:
            return rva - s.vaddr_adj + s.rawptr_adj
        if rva < self.data_len:
            return rva
        return None

    # -- thunk tables ---------------------------------------------------------

    def walk_thunk_table(self, start_rva, max_length):
        """pefile get_import_table -> (aods, ok). ok=False <=> None (fatal)."""
        table = []
        ordinal_flag = _IMAGE_ORDINAL_FLAG
        thunk_size = 4
        if self.pe_type == OPTIONAL_HEADER_MAGIC_PE_PLUS:
            ordinal_flag = _IMAGE_ORDINAL_FLAG64
            thunk_size = 8
        repeated_address = 0
        set32, set64 = set(), set()
        rva = start_rva
        while rva:
            if rva >= start_rva + max_length:
                break
            if self.total_symbols > _MAX_IMPORT_SYMBOLS:
                break
            self.total_symbols += 1
            if repeated_address >= _MAX_REPEATED_ADDRESSES:
                return [], True
            if _spread(set32) > _MAX_ADDRESS_SPREAD or _spread(set64) > _MAX_ADDRESS_SPREAD:
                return [], True
            got = self.get_data(rva, thunk_size)
            if got is None or got[1] - got[0] != thunk_size:
                return [], False
            aod = int.from_bytes(self.data[got[0] : got[1]], "little")
            if aod and start_rva <= aod <= rva:
                break
            if aod:
                if aod & ordinal_flag:
                    # Literal 32-bit mask for both PE types (pefile behavior).
                    if aod & 0x7FFFFFFF > 0xFFFF:
                        return [], True
                else:
                    the_set = set64 if aod >= _ADDR_4GB else set32
                    if aod in the_set:
                        repeated_address += 1
                    the_set.add(aod)
            else:
                break
            rva += thunk_size
            table.append(aod)
        return table, True

    # -- one descriptor -------------------------------------------------------

    def parse_descriptor_imports(self, oft, ft, max_length):
        """pefile parse_imports -> list of symbol dicts (possibly empty)."""
        ilt, _ = self.walk_thunk_table(oft, max_length)
        iat, _ = self.walk_thunk_table(ft, max_length)
        if not ilt and not iat:
            return []
        table = ilt if ilt else iat

        ordinal_flag = _IMAGE_ORDINAL_FLAG
        address_mask = 0x7FFFFFFF
        imp_offset = 4
        if self.pe_type == OPTIONAL_HEADER_MAGIC_PE_PLUS:
            ordinal_flag = _IMAGE_ORDINAL_FLAG64
            address_mask = 0x7FFFFFFFFFFFFFFF
            imp_offset = 8

        symbols = []
        num_invalid = 0
        for idx, aod in enumerate(table):
            imp_ord = None
            imp_hint = None
            name_start = name_end = None
            name_invalid = False
            fatal = False
            # Table entries are never zero (the walk stops at the terminator).
            if aod & ordinal_flag:
                imp_ord = aod & 0xFFFF
            else:
                hint_rva = aod & address_mask
                hg = self.get_data(hint_rva, 2)
                if hg is None:
                    fatal = True
                else:
                    if hg[1] - hg[0] >= 2:
                        imp_hint = int.from_bytes(self.data[hg[0] : hg[0] + 2], "little")
                    name_start, name_end = self.get_string_at_rva(
                        aod + 2, _MAX_IMPORT_NAME_LENGTH
                    )
                    raw = bytes(self.data[name_start:name_end])
                    if all(c in _FUNCTION_NAME_CHARS for c in raw):
                        if self.get_offset_from_rva(aod + 2) is None:
                            fatal = True
                    else:
                        name_invalid = True  # pefile substitutes b"*invalid*"
            imp_address = ft + self.image_base + idx * imp_offset
            bound = None
            if ilt and iat and idx < len(iat) and ilt[idx] != iat[idx]:
                bound = iat[idx]
            if fatal:
                # pefile raises PEFormatError; the directory walk then treats
                # the whole descriptor as having no imports.
                return []
            if name_invalid:
                if num_invalid > 1000 and num_invalid == idx:
                    return []
                num_invalid += 1
                continue
            has_name = name_start is not None and name_end > name_start
            if imp_ord or has_name:
                # pefile: `if imp_ord or imp_name:` (ordinal 0 / empty name drop)
                symbols.append(
                    {
                        "name": bytes(self.data[name_start:name_end]) if has_name else None,
                        "ordinal": imp_ord,
                        "hint": imp_hint,
                        "address": imp_address,
                        "bound": bound,
                    }
                )
        return symbols

    # -- the directory walk ---------------------------------------------------

    def walk(self):
        """pefile parse_import_directory -> list of (dll, [symbols])."""
        out = []
        rva = self.import_dir_rva
        error_count = 0
        while True:
            got = self.get_data(rva, 20)
            if got is None:
                break
            if got[1] - got[0] < 20:
                break  # truncated (pefile raises struct.error; documented gap)
            base = got[0]
            if all(b == 0 for b in self.data[base : base + 20]):
                break
            oft = int.from_bytes(self.data[base : base + 4], "little")
            name_rva = int.from_bytes(self.data[base + 12 : base + 16], "little")
            ft = int.from_bytes(self.data[base + 16 : base + 20], "little")
            file_offset = self.get_offset_from_rva(rva)
            rva += 20
            max_len = self.data_len - file_offset
            if rva > oft or rva > ft:
                max_len = max(rva - oft, rva - ft)

            import_data = self.parse_descriptor_imports(oft, ft, max_len)
            if error_count > 5:
                break
            if not import_data:
                error_count += 1
                continue

            dll_start, dll_end = self.get_string_at_rva(name_rva, _MAX_DLL_LENGTH)
            raw_dll = bytes(self.data[dll_start:dll_end])
            if not all(c in _DLL_NAME_CHARS for c in raw_dll):
                raw_dll = b"*invalid*"
            if not raw_dll:
                continue  # pefile: `if dll:` drops the descriptor
            out.append((raw_dll, import_data))
        return out


def _spread(addr_set):
    if len(addr_set) < 2:
        return 0
    return max(addr_set) - min(addr_set)


def parse_imports(header):
    """Run the fallback walk on a wrapper-parsed header bundle (core._parse_headers)."""
    sections = make_sections(
        header["sections"], header["section_alignment"], header["file_alignment"]
    )
    walker = _Walker(
        header["data"],
        header["pe_type"],
        header["image_base"],
        header["import_dir_rva"],
        sections,
        header["header_len"],
    )
    return walker.walk()
