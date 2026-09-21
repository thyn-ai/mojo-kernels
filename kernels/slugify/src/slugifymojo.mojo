"""Clean-room slugify scoring kernel (transliteration + ASCII slug filter).

Written fresh from the observed behavior of the PyPI `python-slugify` 9.1.0
pipeline on ASCII text: every codepoint is mapped through the text-unidecode
transliteration table, HTML entity references are decoded, and the result is
reduced to runs of [-a-zA-Z0-9] joined by single dashes with no leading or
trailing dash. No third-party Mojo code is used or adapted; the
transliteration table and the entity-name table are not compiled in either —
they are loaded at runtime by the Python wrapper from the installed
`text-unidecode` package data and the standard library's
`html.entities.name2codepoint`, and handed over as a binary blob (see
python/slugify_mojo/slugify_mojo/_data.py for the format).

Determinism: the kernel is a pure function of (blob, input bytes, flags).
The Python wrapper owns every Unicode-aware step the oracle delegates to
CPython (NFKD normalization, case folding, custom regex patterns, the
allow_unicode path); the kernel only ever sees text where the remaining work
is byte-defined. Entity decoding runs on the transliterated output, which is
pure ASCII by construction, so byte-level scanning is exactly equivalent to
the reference's str-level regular expressions there.

Exported C ABI (v1):

    int32_t  slugifymojo_abi_version(void)
    void*    slugifymojo_table_create(const uint8_t* blob, int64_t blob_len)
    int32_t  slugifymojo_transform(void* handle, const uint8_t* in,
                                   int64_t in_len, uint8_t* out,
                                   int64_t out_cap, int64_t* out_len,
                                   uint32_t flags)
    int32_t  slugifymojo_transform_batch(void* handle, const uint8_t* in,
                                         const int64_t* in_offsets,
                                         int64_t n_items, uint8_t* out,
                                         int64_t* out_offsets,
                                         int64_t out_cap,
                                         const uint32_t* item_flags)
    void     slugifymojo_table_destroy(void* handle)

transform returns 0 on success, 1 when out_cap is too small (out_len then
holds the required size), 2 on invalid arguments. transform_batch packs item
outputs back to back and fills out_offsets[0..n_items]; on overflow it
returns 1 with out_offsets[n_items] holding the required total. Each batch
item carries its own flags word.

flags (stages always run in this fixed order):
    bit 0  TRANS       transliterate UTF-8 input through the table
    bit 1  ENT_NAMED   decode &name; references (name in the blob table)
    bit 2  ENT_DECIMAL decode &#123; references
    bit 3  ENT_HEX     decode &#x1a; references (uppercase X only when
                       NUMERIC_LEGACY is clear)
    bit 4  NUMERIC_LEGACY  legacy numeric semantics: one invalid numeric
                       reference abandons the whole pass (all references of
                       that base stay literal) and surrogate values are
                       emitted; otherwise each invalid reference stays
                       literal on its own and surrogates are never emitted
    bit 5  FILTER      quote stripping, digit-comma joining, disallowed-run
                       collapsing, dash dedup, dash trimming (byte level)
    bit 6  QUOTE_DASH  fold apostrophe runs to single dashes before the
                       transliteration stage; the wrapper selects it only
                       where no normalization step can reorder around it
    bit 7  LOWER       fold ASCII A-Z to a-z during FILTER; the wrapper
                       selects it only for pure-ASCII inputs, where
                       CPython's str.lower() reduces to that mapping
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Blob header layout (little-endian u64 words).
comptime BLOB_MAGIC: UInt64 = 0x534C4D4F4A4F3031  # "SLMOJO01"
comptime BLOB_VERSION: UInt64 = 1
comptime HDR_WORDS: Int64 = 16
comptime H_N_TRANS: Int64 = 2
comptime H_OFF_TRANS_BYTES: Int64 = 3
comptime H_TRANS_BYTES_LEN: Int64 = 4
comptime H_N_ENT: Int64 = 5
comptime H_OFF_ENT_NAME_OFF: Int64 = 6
comptime H_OFF_ENT_NAMES: Int64 = 7
comptime H_ENT_NAMES_LEN: Int64 = 8
comptime H_OFF_ENT_CPS: Int64 = 9

# transform() flags
comptime F_TRANS: UInt32 = 1
comptime F_ENT_NAMED: UInt32 = 2
comptime F_ENT_DECIMAL: UInt32 = 4
comptime F_ENT_HEX: UInt32 = 8
comptime F_NUMERIC_LEGACY: UInt32 = 16
comptime F_FILTER: UInt32 = 32
comptime F_QUOTE_DASH: UInt32 = 64
comptime F_LOWER: UInt32 = 128

# CPython's int(string) digit ceiling for base 10 (sys.get_int_max_str_digits
# default): more digits than this raises ValueError, which the reference
# catches. Hex parsing is a power-of-2 base and is exempt there; the value
# range check below governs it instead.
comptime MAX_DECIMAL_DIGITS: Int64 = 4300
comptime MAX_CODEPOINT: UInt32 = 0x10FFFF

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


struct SlugTable:
    """Kernel-owned copy of the transliteration and entity tables."""

    var n_trans: Int64
    var trans_off: U32Ptr  # n_trans + 1 offsets into trans_bytes
    var trans_bytes: U8Ptr
    var n_ent: Int64
    var ent_name_off: U32Ptr  # n_ent + 1 offsets into ent_names
    var ent_names: U8Ptr  # concatenated entity names, sorted bytewise
    var ent_cps: U32Ptr  # n_ent codepoints, parallel to the sorted names

    def __init__(
        out self,
        n_trans: Int64,
        trans_off: U32Ptr,
        trans_bytes: U8Ptr,
        n_ent: Int64,
        ent_name_off: U32Ptr,
        ent_names: U8Ptr,
        ent_cps: U32Ptr,
    ):
        self.n_trans = n_trans
        self.trans_off = trans_off
        self.trans_bytes = trans_bytes
        self.n_ent = n_ent
        self.ent_name_off = ent_name_off
        self.ent_names = ent_names
        self.ent_cps = ent_cps

    def free(mut self):
        self.trans_off.unsafe_free()
        self.trans_bytes.unsafe_free()
        self.ent_name_off.unsafe_free()
        self.ent_names.unsafe_free()
        self.ent_cps.unsafe_free()


# --------------------------------------------------------------------------
# Blob parsing
# --------------------------------------------------------------------------


def _read_u64(blob: U8Ptr, off: Int64) -> UInt64:
    # `off` is always a multiple of 8 by blob construction
    return blob.unsafe_bitcast[UInt64]()[unsafe_offset=Int(off // 8)]


def _read_u32(blob: U8Ptr, off: Int64) -> UInt32:
    # `off` is always a multiple of 4 by blob construction
    return blob.unsafe_bitcast[UInt32]()[unsafe_offset=Int(off // 4)]


def _copy_u32_table(blob: U8Ptr, off: Int64, n: Int64) -> U32Ptr:
    var dst = unsafe_alloc[UInt32](Int(n))
    for i in range(Int(n)):
        dst[unsafe_offset=i] = _read_u32(blob, off + Int64(i) * 4)
    return dst


def _copy_u8_table(blob: U8Ptr, off: Int64, n: Int64) -> U8Ptr:
    var dst = unsafe_alloc[UInt8](Int(n))
    for i in range(Int(n)):
        dst[unsafe_offset=i] = blob[unsafe_offset=Int(off) + i]
    return dst


def _parse_blob(blob: U8Ptr, blob_len: Int64) -> Handle:
    """Validate and copy the wrapper's table blob; NULL when malformed.

    All validation happens against the caller's blob first; only after every
    check passes does the kernel allocate and copy, so a rejected blob never
    leaves partial state behind.
    """
    if blob_len < HDR_WORDS * 8:
        return None
    if _read_u64(blob, 0) != BLOB_MAGIC or _read_u64(blob, 8) != BLOB_VERSION:
        return None
    var n_trans = _read_u64(blob, H_N_TRANS * 8)
    var trans_bytes_len = _read_u64(blob, H_TRANS_BYTES_LEN * 8)
    var n_ent = _read_u64(blob, H_N_ENT * 8)
    var ent_names_len = _read_u64(blob, H_ENT_NAMES_LEN * 8)
    if n_trans <= 0 or n_trans > 1 << 21 or n_ent < 0 or n_ent > 1 << 16:
        return None
    if trans_bytes_len < 0 or trans_bytes_len > 1 << 28:
        return None
    if ent_names_len < 0 or ent_names_len > 1 << 24:
        return None

    var off_trans_bytes = Int64(_read_u64(blob, H_OFF_TRANS_BYTES * 8))
    var off_ent_name_off = Int64(_read_u64(blob, H_OFF_ENT_NAME_OFF * 8))
    var off_ent_names = Int64(_read_u64(blob, H_OFF_ENT_NAMES * 8))
    var off_ent_cps = Int64(_read_u64(blob, H_OFF_ENT_CPS * 8))
    var offs = List[Int64]()
    var lens = List[Int64]()
    offs.append(off_trans_bytes)
    lens.append(Int64(trans_bytes_len))
    offs.append(off_ent_name_off)
    lens.append(Int64(n_ent + 1) * 4)
    offs.append(off_ent_names)
    lens.append(Int64(ent_names_len))
    offs.append(off_ent_cps)
    lens.append(Int64(n_ent) * 4)
    for s in range(len(offs)):
        if offs[s] < HDR_WORDS * 8 or offs[s] % 8 != 0:
            return None
        if offs[s] + lens[s] > blob_len:
            return None

    # Content validation straight from the caller's blob (no allocation yet):
    # the raw table must hold exactly n_trans NUL-separated entries (n_trans
    # - 1 separators; the last entry runs to the end of the payload), every
    # byte must be ASCII (the kernel's filter stage assumes transliteration
    # output is pure ASCII), entity names must be strictly ascending (binary
    # search) and codepoints in range.
    var nul_count = Int64(0)
    for i in range(Int(trans_bytes_len)):
        var b = blob[unsafe_offset=Int(off_trans_bytes) + i]
        if b == 0:
            nul_count += 1
        elif b >= 0x80:
            return None
    if nul_count != Int64(n_trans) - 1:
        return None
    if n_ent > 0:
        if _read_u32(blob, off_ent_name_off) != 0:
            return None
        if _read_u32(blob, off_ent_name_off + Int64(n_ent) * 4) != UInt32(ent_names_len):
            return None
        for i in range(1, Int(n_ent + 1)):
            if _read_u32(blob, off_ent_name_off + Int64(i - 1) * 4) > _read_u32(
                blob, off_ent_name_off + Int64(i) * 4
            ):
                return None
        for i in range(1, Int(n_ent)):
            var prev_lo = _read_u32(blob, off_ent_name_off + Int64(i - 1) * 4)
            var prev_hi = _read_u32(blob, off_ent_name_off + Int64(i) * 4)
            var cur_lo = prev_hi
            var cur_hi = _read_u32(blob, off_ent_name_off + Int64(i + 1) * 4)
            # strict lexicographic increase, byte by byte
            var shorter = Int(prev_hi - prev_lo)
            if Int(cur_hi - cur_lo) < shorter:
                shorter = Int(cur_hi - cur_lo)
            var cmp = 0
            for k in range(shorter):
                var a = blob[unsafe_offset=Int(off_ent_names + Int64(prev_lo)) + k]
                var b = blob[unsafe_offset=Int(off_ent_names + Int64(cur_lo)) + k]
                if a < b:
                    cmp = -1
                    break
                if a > b:
                    cmp = 1
                    break
            if cmp == 0:
                if prev_hi - prev_lo < cur_hi - cur_lo:
                    cmp = -1
                elif prev_hi - prev_lo > cur_hi - cur_lo:
                    cmp = 1
            if cmp >= 0:
                return None
        for i in range(Int(n_ent)):
            if _read_u32(blob, off_ent_cps + Int64(i) * 4) > MAX_CODEPOINT:
                return None

    # Everything checks out: copy into kernel-owned storage, indexing the
    # NUL-separated entries as we copy. trans_off[k] is the start of entry
    # k; entry k spans [trans_off[k], trans_off[k+1] - 1) — the byte before
    # the next start is its NUL separator — with a virtual NUL at the end.
    var trans_off = unsafe_alloc[UInt32](Int(n_trans) + 1)
    var trans_bytes = unsafe_alloc[UInt8](Int(trans_bytes_len))
    var entry = Int64(0)
    trans_off[unsafe_offset=0] = 0
    for i in range(Int(trans_bytes_len)):
        var b = blob[unsafe_offset=Int(off_trans_bytes) + i]
        trans_bytes[unsafe_offset=i] = b
        if b == 0:
            entry += 1
            trans_off[unsafe_offset=Int(entry)] = UInt32(i + 1)
    trans_off[unsafe_offset=Int(n_trans)] = UInt32(trans_bytes_len) + 1
    var slot = unsafe_alloc[SlugTable](1)
    slot[] = SlugTable(
        Int64(n_trans),
        trans_off,
        trans_bytes,
        Int64(n_ent),
        _copy_u32_table(blob, off_ent_name_off, Int64(n_ent) + 1),
        _copy_u8_table(blob, off_ent_names, Int64(ent_names_len)),
        _copy_u32_table(blob, off_ent_cps, Int64(n_ent)),
    )
    return slot.unsafe_bitcast[UInt8]()


# --------------------------------------------------------------------------
# Byte helpers
# --------------------------------------------------------------------------


def _is_alnum_ascii(b: UInt8) -> Bool:
    return (b >= 0x30 and b <= 0x39) or (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A)


def _is_digit_ascii(b: UInt8) -> Bool:
    return b >= 0x30 and b <= 0x39


def _emit_utf8(mut out: List[UInt8], cp: UInt32):
    """Append cp as UTF-8. Values in 0xD800..0xDFFF use the three-byte form,
    matching CPython's surrogatepass encoding (the reference materializes
    such values as lone surrogates in a str)."""
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


# --------------------------------------------------------------------------
# Stage 1: transliteration through the blob table
# --------------------------------------------------------------------------


def _transliterate(t: SlugTable, src: List[UInt8]) -> List[UInt8]:
    """Map every codepoint through the table: cp 0 -> NUL byte, table hits ->
    their bytes, codepoints past the table -> dropped (the reference catches
    IndexError and skips them)."""
    var out = List[UInt8](capacity=len(src) * 2 + 16)
    var i = 0
    var n = len(src)
    while i < n:
        var b0 = src[i]
        var cp: UInt32
        var size: Int
        if b0 < 0x80:
            cp = UInt32(b0)
            size = 1
        elif b0 < 0xE0:
            cp = UInt32(b0 & 0x1F) << 6
            if i + 1 < n:
                cp |= UInt32(src[i + 1] & 0x3F)
            size = 2
        elif b0 < 0xF0:
            cp = UInt32(b0 & 0x0F) << 12
            if i + 1 < n:
                cp |= UInt32(src[i + 1] & 0x3F) << 6
            if i + 2 < n:
                cp |= UInt32(src[i + 2] & 0x3F)
            size = 3
        else:
            cp = UInt32(b0 & 0x07) << 18
            if i + 1 < n:
                cp |= UInt32(src[i + 1] & 0x3F) << 12
            if i + 2 < n:
                cp |= UInt32(src[i + 2] & 0x3F) << 6
            if i + 3 < n:
                cp |= UInt32(src[i + 3] & 0x3F)
            size = 4
        i += size
        if cp == 0:
            out.append(0)
            continue
        if Int64(cp) > t.n_trans:
            continue
        var lo = t.trans_off[unsafe_offset=Int(cp - 1)]
        var hi = t.trans_off[unsafe_offset=Int(cp)] - 1
        for k in range(Int(lo), Int(hi)):
            out.append(t.trans_bytes[unsafe_offset=k])
    return out^


# --------------------------------------------------------------------------
# Stage 2: entity reference decoding (byte level; input is ASCII here)
# --------------------------------------------------------------------------


def _lookup_entity(t: SlugTable, src: List[UInt8], lo: Int, hi: Int) -> Int:
    """Binary search src[lo:hi] in the sorted entity-name table; returns the
    codepoint or -1. Names are ASCII, so byte comparison is exact."""
    var a = Int(0)
    var b = Int(t.n_ent) - 1
    var span = hi - lo
    while a <= b:
        var mid = (a + b) // 2
        var m_lo = Int(t.ent_name_off[unsafe_offset=mid])
        var m_hi = Int(t.ent_name_off[unsafe_offset=mid + 1])
        var m_span = m_hi - m_lo
        var shorter = span if span < m_span else m_span
        var cmp = 0
        for k in range(shorter):
            var x = src[lo + k]
            var y = t.ent_names[unsafe_offset=m_lo + k]
            if x < y:
                cmp = -1
                break
            if x > y:
                cmp = 1
                break
        if cmp == 0:
            if span < m_span:
                cmp = -1
            elif span > m_span:
                cmp = 1
        if cmp < 0:
            b = mid - 1
        elif cmp > 0:
            a = mid + 1
        else:
            return Int(t.ent_cps[unsafe_offset=mid])
    return -1


def _decode_named(t: SlugTable, src: List[UInt8]) -> List[UInt8]:
    """&name; -> the named codepoint. A failed '&' is emitted and scanning
    resumes one byte later, exactly like the reference regex advancing."""
    var out = List[UInt8](capacity=len(src) + 16)
    var i = 0
    var n = len(src)
    while i < n:
        if src[i] != 0x26:  # '&'
            out.append(src[i])
            i += 1
            continue
        var j = i + 1
        while j < n and _is_alnum_ascii(src[j]):
            j += 1
        if j < n and src[j] == 0x3B and j > i + 1:  # ';'
            var cp = _lookup_entity(t, src, i + 1, j)
            if cp >= 0:
                _emit_utf8(out, UInt32(cp))
                i = j + 1
                continue
        out.append(0x26)
        i += 1
    return out^


def _hex_val(b: UInt8) -> Int:
    if b >= 0x30 and b <= 0x39:
        return Int(b - 0x30)
    if b >= 0x61 and b <= 0x66:
        return Int(b - 0x61) + 10
    if b >= 0x41 and b <= 0x46:
        return Int(b - 0x41) + 10
    return -1


def _decode_numeric(mut buf: List[UInt8], base: Int, legacy: Bool):
    """Decode &#123; (base 10) or &#x1a; (base 16) references, in place.

    Legacy semantics (the reference wraps the whole substitution in one
    try/except): a single reference whose int()/chr() would raise abandons
    the entire pass — every reference of this base stays literal — and
    surrogate values are emitted (surrogatepass 3-byte form). Modern
    semantics: an invalid reference stays literal by itself and scanning
    resumes after it; surrogates are never emitted.
    """
    var src = buf.copy()
    var out = List[UInt8](capacity=len(src) + 16)
    var i = 0
    var n = len(src)
    while i < n:
        var matched = False
        var value: UInt32 = 0
        var end = i
        if src[i] == 0x26 and i + 2 < n and src[i + 1] == 0x23:  # '&#'
            var j = i + 2
            if base == 16:
                var xc = src[j]
                if xc == 0x78 or (not legacy and xc == 0x58):  # 'x' / 'X'
                    j += 1
                else:
                    j = -1
            if j >= 0:
                var k = j
                if base == 10:
                    while k < n and _is_digit_ascii(src[k]):
                        k += 1
                else:
                    while k < n and _hex_val(src[k]) >= 0:
                        k += 1
                if k > j and k < n and src[k] == 0x3B:  # ';'
                    var ndig = Int64(k - j)
                    var ok = True
                    if base == 10 and ndig > MAX_DECIMAL_DIGITS:
                        ok = False
                    else:
                        var acc: UInt64 = 0
                        for d in range(j, k):
                            var dv = _hex_val(src[d]) if base == 16 else Int(src[d] - 0x30)
                            acc = acc * UInt64(base) + UInt64(dv)
                            if acc > UInt64(MAX_CODEPOINT):
                                ok = False
                                break
                        if ok:
                            value = UInt32(acc)
                    if ok and not legacy and value >= 0xD800 and value <= 0xDFFF:
                        ok = False
                    if ok:
                        matched = True
                        end = k + 1
                    else:
                        if legacy:
                            # Abandon the whole pass: buf stays untouched.
                            return
                        # Modern: leave this reference literal, skip past it.
                        for d in range(i, k + 1):
                            out.append(src[d])
                        i = k + 1
        if matched:
            _emit_utf8(out, value)
            i = end
        elif src[i] == 0x26 and i + 2 < n and src[i + 1] == 0x23:
            # '&#' that never formed a reference: emit the '&' and move on,
            # like the regex advancing one character.
            out.append(0x26)
            i += 1
        elif src[i] != 0x26:
            out.append(src[i])
            i += 1
        else:
            out.append(src[i])
            i += 1
    buf = out^


# --------------------------------------------------------------------------
# Stage 0 (optional): apostrophe runs to a single dash (byte level; the
# wrapper only selects it where no normalization step can reorder around it)
# --------------------------------------------------------------------------


def _quote_dash(src: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(src) + 16)
    var i = 0
    var n = len(src)
    while i < n:
        if src[i] == 0x27:
            out.append(0x2D)
            while i < n and src[i] == 0x27:
                i += 1
        else:
            out.append(src[i])
            i += 1
    return out^


# --------------------------------------------------------------------------
# Stage 3: ASCII slug filter (quote strip, digit commas, disallowed runs)
# --------------------------------------------------------------------------


def _filter_ascii(src: List[UInt8], lower: Bool) -> List[UInt8]:
    # 1. remove every apostrophe (the reference's second quote substitution);
    # optionally fold ASCII case. The wrapper selects `lower` only when the
    # input is pure ASCII, where CPython's str.lower() reduces to exactly
    # this A-Z mapping.
    var no_quotes = List[UInt8](capacity=len(src) + 16)
    for i in range(len(src)):
        var b = src[i]
        if b != 0x27:
            if lower and b >= 0x41 and b <= 0x5A:
                b += 0x20
            no_quotes.append(b)
    # 2. remove commas sitting between two ASCII digits. The reference regex
    # uses Unicode digit classes, but on this stage's input every decimal
    # digit that could survive the later disallowed-character filter is
    # ASCII, and a comma between non-ASCII digits sits inside a disallowed
    # run either way, so the byte-level check is exactly equivalent here.
    var no_commas = List[UInt8](capacity=len(no_quotes) + 16)
    var m = len(no_quotes)
    for i in range(m):
        if (
            no_quotes[i] == 0x2C
            and i > 0
            and i + 1 < m
            and _is_digit_ascii(no_quotes[i - 1])
            and _is_digit_ascii(no_quotes[i + 1])
        ):
            continue
        no_commas.append(no_quotes[i])
    # 3. collapse every maximal run of (disallowed byte | '-') into a single
    # dash, dropping leading/trailing runs: the fused effect of the
    # disallowed-chars substitution, the dash dedup and the dash strip.
    var out = List[UInt8](capacity=len(no_commas) + 16)
    var in_run = False
    for i in range(len(no_commas)):
        var b = no_commas[i]
        if _is_alnum_ascii(b):
            if in_run and len(out) > 0:
                out.append(0x2D)
            in_run = False
            out.append(b)
        else:  # '-' or a disallowed byte: same run class after the filter
            in_run = True
    return out^


# --------------------------------------------------------------------------
# Transform driver
# --------------------------------------------------------------------------


def _transform(t: SlugTable, src: U8Ptr, n: Int64, flags: UInt32) -> List[UInt8]:
    var buf = List[UInt8](capacity=Int(n) + 16)
    for i in range(Int(n)):
        buf.append(src[unsafe_offset=i])
    if flags & F_QUOTE_DASH:
        buf = _quote_dash(buf^)
    if flags & F_TRANS:
        buf = _transliterate(t, buf^)
    if flags & F_ENT_NAMED:
        buf = _decode_named(t, buf^)
    if flags & F_ENT_DECIMAL:
        _decode_numeric(buf, 10, flags & F_NUMERIC_LEGACY != 0)
    if flags & F_ENT_HEX:
        _decode_numeric(buf, 16, flags & F_NUMERIC_LEGACY != 0)
    if flags & F_FILTER:
        buf = _filter_ascii(buf^, flags & F_LOWER != 0)
    return buf^


# --------------------------------------------------------------------------
# Exported C ABI
# --------------------------------------------------------------------------


@export
def slugifymojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def slugifymojo_table_create(blob: U8Ptr, blob_len: Int64) abi("C") -> Handle:
    if blob_len < 0:
        return None
    return _parse_blob(blob, blob_len)


@export
def slugifymojo_transform(
    handle: Handle,
    in_bytes: U8Ptr,
    in_len: Int64,
    out_bytes: U8Ptr,
    out_cap: Int64,
    out_len: I64Ptr,
    flags: UInt32,
) abi("C") -> Int32:
    if not handle or in_len < 0 or out_cap < 0:
        return 2
    var t = handle.value().unsafe_bitcast[SlugTable]()
    var buf = _transform(t[], in_bytes, in_len, flags)
    out_len[unsafe_offset=0] = Int64(len(buf))
    if Int64(len(buf)) > out_cap:
        return 1
    for i in range(len(buf)):
        out_bytes[unsafe_offset=i] = buf[i]
    return 0


@export
def slugifymojo_transform_batch(
    handle: Handle,
    in_bytes: U8Ptr,
    in_offsets: I64Ptr,
    n_items: Int64,
    out_bytes: U8Ptr,
    out_offsets: I64Ptr,
    out_cap: Int64,
    item_flags: U32Ptr,
) abi("C") -> Int32:
    if not handle or n_items < 0 or out_cap < 0:
        return 2
    var t = handle.value().unsafe_bitcast[SlugTable]()
    var pos = Int64(0)
    out_offsets[unsafe_offset=0] = 0
    for item in range(Int(n_items)):
        var lo = in_offsets[unsafe_offset=item]
        var hi = in_offsets[unsafe_offset=item + 1]
        if hi < lo:
            return 2
        var buf = _transform(
            t[],
            in_bytes.unsafe_offset(Int(lo)),
            hi - lo,
            item_flags[unsafe_offset=item],
        )
        pos += Int64(len(buf))
        out_offsets[unsafe_offset=item + 1] = pos
    if pos > out_cap:
        return 1
    pos = 0
    for item in range(Int(n_items)):
        var lo = in_offsets[unsafe_offset=item]
        var hi = in_offsets[unsafe_offset=item + 1]
        var buf = _transform(
            t[],
            in_bytes.unsafe_offset(Int(lo)),
            hi - lo,
            item_flags[unsafe_offset=item],
        )
        for i in range(len(buf)):
            out_bytes[unsafe_offset=Int(pos) + i] = buf[i]
        pos += Int64(len(buf))
    return 0


@export
def slugifymojo_table_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var t = handle.value().unsafe_bitcast[SlugTable]()
    t[].free()
    t.unsafe_free()
