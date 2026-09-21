"""Clean-room phone number parsing/validation/formatting kernel.

Written fresh against the behavior of the public libphonenumber metadata
(Apache-2.0 data, see kernels/phonenumbers/data/NOTICE) and the observable
behavior of the PyPI `phonenumbers` 9.0.39 reference. No libphonenumber code
is used or adapted: the matching engine is a priority-ordered Thompson NFA
simulation (Pike VM) over digit-mask bytecode compiled by
kernels/phonenumbers/gen_tables.py, and the parse/validate/format pipeline
mirrors the documented libphonenumber semantics, pinned empirically against
the oracle (see python/phonenumbers_mojo/phonenumbers_mojo/_fallback.py for
the executable specification this kernel ports to native code).

Input charset: the kernel handles ASCII input only. The Python wrapper scans
each input and routes anything containing non-ASCII bytes to the pure-Python
fallback engine, so results are always oracle-identical either way.

Exported C ABI (v1):

    int32_t  phonenumbersmojo_abi_version(void)
    void*    phonenumbersmojo_metadata_create(const uint8_t* blob, int64_t blob_len)
    int32_t  phonenumbersmojo_parse(void* handle, const uint8_t* text, int64_t text_len,
                                    int32_t region_idx, PNMOut* out)
    int32_t  phonenumbersmojo_validate(void* handle, int32_t cc, const uint8_t* nsn,
                                       int64_t nsn_len, int32_t* out_flags)
    int32_t  phonenumbersmojo_format(void* handle, int32_t cc, const uint8_t* nsn,
                                     int64_t nsn_len, const uint8_t* ext, int64_t ext_len,
                                     int32_t fmt, uint8_t* out, int64_t out_cap)
    int32_t  phonenumbersmojo_validate_batch(void* handle, const uint8_t* packed,
                                             const int64_t* offsets, const int32_t* region_idx,
                                             int64_t n, uint8_t* out_valid, int32_t* out_status)
    void     phonenumbersmojo_metadata_destroy(void* handle)

PNMOut layout (matches the ctypes wrapper): 9 x int32 (status, error_type,
cc, ccs, leading_zeros, valid, possible, nsn_len, ext_len) followed by
char nsn[24] and char ext[24]. status: 0 ok, 1 parse error (error_type is the
oracle's NumberParseException code; ccs selects the message variant), 3 the
input was not ASCII (wrapper must use its fallback engine for this input).

Blob format: see gen_tables.py (PNM1 header, program/string/region offset
tables, cc map). Programs are digit-mask NFA bytecode:

    op 0 CLASS (x = 10-bit digit mask)   op 3 SAVE  (x = capture slot)
    op 1 SPLIT (x, y)                    op 4 MATCH
    op 2 JMP   (x)                       op 5 EOL
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime OP_CLS: Int32 = 0
comptime OP_SPLIT: Int32 = 1
comptime OP_JMP: Int32 = 2
comptime OP_SAVE: Int32 = 3
comptime OP_MATCH: Int32 = 4
comptime OP_EOL: Int32 = 5

comptime FULL: Int32 = 0
comptime PREFIX: Int32 = 1

comptime NO_LEN: UInt32 = 0xFFFFFFFF
comptime MAX_NSN: Int = 17
comptime MAX_INPUT: Int = 250

# NumberParseException error types (oracle parity)
comptime E_INVALID_CC: Int32 = 0
comptime E_NOT_A_NUMBER: Int32 = 1
comptime E_TOO_SHORT_AFTER_IDD: Int32 = 2
comptime E_TOO_SHORT_NSN: Int32 = 3
comptime E_TOO_LONG: Int32 = 4

# test_number_length classification
comptime LEN_POSSIBLE: Int32 = 0
comptime LEN_TOO_SHORT: Int32 = 1
comptime LEN_TOO_LONG: Int32 = 2
comptime LEN_INVALID: Int32 = 3
comptime LEN_LOCAL_ONLY: Int32 = 4

# PNMOut status codes
comptime ST_OK: Int32 = 0
comptime ST_PARSE_ERROR: Int32 = 1
comptime ST_NON_ASCII: Int32 = 3

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime NSN_CAP = 24
comptime EXT_CAP = 24
comptime CAP_SLOTS = 12  # 2 * (max groups in the metadata (5) + 1)


# --------------------------------------------------------------------------
# Blob tables
# --------------------------------------------------------------------------


struct Meta(Copyable, Movable):
    var n_programs: Int64
    var prog_insts: I32Ptr  # [n_programs] instruction counts
    var prog_groups: I32Ptr  # [n_programs] capture group counts
    var prog_ops: I32Ptr  # [sum n_insts] op words
    var prog_xy: I32Ptr  # [2 * sum n_insts] x,y words
    var prog_off: I64Ptr  # [n_programs] start index into the inst arrays
    var n_strings: Int64
    var str_off: U32Ptr  # [n_strings] offsets into str_buf
    var str_len: U32Ptr  # [n_strings]
    var str_buf: U8Ptr
    var n_regions: Int64
    var reg_code: I32Ptr  # string idx
    var reg_cc: I32Ptr
    var reg_main: I32Ptr
    var reg_intl_prefix: I32Ptr
    var reg_np_str: I32Ptr  # national prefix string idx
    var reg_npfp: I32Ptr
    var reg_transform: I32Ptr
    var reg_leading: I32Ptr  # territory leadingDigits prog
    var reg_pref_extn: I32Ptr
    var reg_general: I32Ptr
    var reg_gmask: U32Ptr
    var reg_gmask_local: U32Ptr
    var reg_types: I32Ptr  # [10 * n_regions] type prog idx (-1 = none)
    var reg_tmask: U32Ptr  # [10 * n_regions]
    var reg_nfmt: I32Ptr  # national formats count
    var reg_nifmt: I32Ptr  # intl formats count
    # formats, both lists laid back to back: per format 4 x i32
    var reg_fmt_ptr: I64Ptr  # [n_regions] flat index into fmt_words
    var fmt_words: I32Ptr  # [4 * (nfmt + nifmt) summed]
    var max_insts: Int64
    var n_cc: Int64
    var cc_val: I32Ptr  # [n_cc] ascending
    var cc_start: I32Ptr  # [n_cc] start index into cc_region
    var cc_count: I32Ptr  # [n_cc]
    var cc_region: I32Ptr  # [sum counts] region idx, main first


    def __init__(
        out self,
        n_programs: Int64,
        prog_insts: I32Ptr,
        prog_groups: I32Ptr,
        prog_ops: I32Ptr,
        prog_xy: I32Ptr,
        prog_off: I64Ptr,
        n_strings: Int64,
        str_off: U32Ptr,
        str_len: U32Ptr,
        str_buf: U8Ptr,
        n_regions: Int64,
        reg_code: I32Ptr,
        reg_cc: I32Ptr,
        reg_main: I32Ptr,
        reg_intl_prefix: I32Ptr,
        reg_np_str: I32Ptr,
        reg_npfp: I32Ptr,
        reg_transform: I32Ptr,
        reg_leading: I32Ptr,
        reg_pref_extn: I32Ptr,
        reg_general: I32Ptr,
        reg_gmask: U32Ptr,
        reg_gmask_local: U32Ptr,
        reg_types: I32Ptr,
        reg_tmask: U32Ptr,
        reg_nfmt: I32Ptr,
        reg_nifmt: I32Ptr,
        reg_fmt_ptr: I64Ptr,
        fmt_words: I32Ptr,
        max_insts: Int64,
        n_cc: Int64,
        cc_val: I32Ptr,
        cc_start: I32Ptr,
        cc_count: I32Ptr,
        cc_region: I32Ptr,
    ):
        self.n_programs = n_programs
        self.prog_insts = prog_insts
        self.prog_groups = prog_groups
        self.prog_ops = prog_ops
        self.prog_xy = prog_xy
        self.prog_off = prog_off
        self.n_strings = n_strings
        self.str_off = str_off
        self.str_len = str_len
        self.str_buf = str_buf
        self.n_regions = n_regions
        self.reg_code = reg_code
        self.reg_cc = reg_cc
        self.reg_main = reg_main
        self.reg_intl_prefix = reg_intl_prefix
        self.reg_np_str = reg_np_str
        self.reg_npfp = reg_npfp
        self.reg_transform = reg_transform
        self.reg_leading = reg_leading
        self.reg_pref_extn = reg_pref_extn
        self.reg_general = reg_general
        self.reg_gmask = reg_gmask
        self.reg_gmask_local = reg_gmask_local
        self.reg_types = reg_types
        self.reg_tmask = reg_tmask
        self.reg_nfmt = reg_nfmt
        self.reg_nifmt = reg_nifmt
        self.reg_fmt_ptr = reg_fmt_ptr
        self.fmt_words = fmt_words
        self.max_insts = max_insts
        self.n_cc = n_cc
        self.cc_val = cc_val
        self.cc_start = cc_start
        self.cc_count = cc_count
        self.cc_region = cc_region

    def free(mut self):
        self.prog_insts.unsafe_free()
        self.prog_groups.unsafe_free()
        self.prog_ops.unsafe_free()
        self.prog_xy.unsafe_free()
        self.prog_off.unsafe_free()
        self.str_off.unsafe_free()
        self.str_len.unsafe_free()
        self.str_buf.unsafe_free()
        self.reg_code.unsafe_free()
        self.reg_cc.unsafe_free()
        self.reg_main.unsafe_free()
        self.reg_intl_prefix.unsafe_free()
        self.reg_np_str.unsafe_free()
        self.reg_npfp.unsafe_free()
        self.reg_transform.unsafe_free()
        self.reg_leading.unsafe_free()
        self.reg_pref_extn.unsafe_free()
        self.reg_general.unsafe_free()
        self.reg_gmask.unsafe_free()
        self.reg_gmask_local.unsafe_free()
        self.reg_types.unsafe_free()
        self.reg_tmask.unsafe_free()
        self.reg_nfmt.unsafe_free()
        self.reg_nifmt.unsafe_free()
        self.reg_fmt_ptr.unsafe_free()
        self.fmt_words.unsafe_free()
        self.cc_val.unsafe_free()
        self.cc_start.unsafe_free()
        self.cc_count.unsafe_free()
        self.cc_region.unsafe_free()


def _read_u32(blob: U8Ptr, off: Int64) -> UInt32:
    """Little-endian u32 at an arbitrary byte offset (the blob is packed, not
    aligned)."""
    var b0 = UInt32(blob[unsafe_offset=Int(off)])
    var b1 = UInt32(blob[unsafe_offset=Int(off + 1)])
    var b2 = UInt32(blob[unsafe_offset=Int(off + 2)])
    var b3 = UInt32(blob[unsafe_offset=Int(off + 3)])
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def _read_i32(blob: U8Ptr, off: Int64) -> Int32:
    return Int32(_read_u32(blob, off))


def _read_u16(blob: U8Ptr, off: Int64) -> UInt16:
    var b0 = UInt16(blob[unsafe_offset=Int(off)])
    var b1 = UInt16(blob[unsafe_offset=Int(off + 1)])
    return b0 | (b1 << 8)


def _parse_blob(blob: U8Ptr, blob_len: Int64) -> Handle:
    """Validate and copy the wrapper's metadata blob; NULL when malformed."""
    if blob_len < 28:
        return None
    if (
        _read_u32(blob, 0) != 0x314D4E50  # "PNM1"
        or _read_u32(blob, 4) != 1
    ):
        return None
    var n_programs = Int64(_read_u32(blob, 8))
    var n_regions = Int64(_read_u32(blob, 12))
    var n_strings = Int64(_read_u32(blob, 16))
    var cc_off = Int64(_read_u32(blob, 20))
    var n_cc = Int64(_read_u32(blob, 24))
    if (
        n_programs <= 0
        or n_programs > 100000
        or n_regions <= 0
        or n_regions > 10000
        or n_strings <= 0
        or n_strings > 100000
        or n_cc <= 0
        or n_cc > 10000
    ):
        return None
    var offs_table = Int64(28)
    var total_off_words = 4 * (n_programs + n_strings + n_regions)
    if offs_table + total_off_words > blob_len:
        return None
    var prog_off_pos = offs_table
    var str_off_pos = prog_off_pos + 4 * n_programs
    var reg_off_pos = str_off_pos + 4 * n_strings

    # Section bounds check (program section sizes vary; validate offsets are
    # inside the blob and monotone-ish by checking each read below).
    if cc_off < offs_table + total_off_words or cc_off + 8 * n_cc > blob_len:
        return None

    # --- copy programs
    var prog_insts = unsafe_alloc[Int32](Int(n_programs))
    var prog_groups = unsafe_alloc[Int32](Int(n_programs))
    var prog_off = unsafe_alloc[Int64](Int(n_programs))
    var total_insts = Int64(0)
    for i in range(Int(n_programs)):
        var po = Int64(_read_u32(blob, prog_off_pos + 4 * Int64(i)))
        if po + 8 > blob_len:
            prog_insts.unsafe_free()
            prog_groups.unsafe_free()
            prog_off.unsafe_free()
            return None
        var n_insts = Int64(_read_u32(blob, po))
        var n_groups = Int64(_read_u32(blob, po + 4))
        if n_insts <= 0 or n_insts > 100000 or n_groups < 0 or n_groups > 32:
            prog_insts.unsafe_free()
            prog_groups.unsafe_free()
            prog_off.unsafe_free()
            return None
        if po + 8 + 12 * n_insts > blob_len:
            prog_insts.unsafe_free()
            prog_groups.unsafe_free()
            prog_off.unsafe_free()
            return None
        prog_insts[unsafe_offset=i] = Int32(n_insts)
        prog_groups[unsafe_offset=i] = Int32(n_groups)
        prog_off[unsafe_offset=i] = total_insts
        total_insts += n_insts

    var prog_ops = unsafe_alloc[Int32](Int(total_insts))
    var prog_xy = unsafe_alloc[Int32](Int(2 * total_insts))
    for i in range(Int(n_programs)):
        var po = Int64(_read_u32(blob, prog_off_pos + 4 * Int64(i)))
        var base = prog_off[unsafe_offset=i]
        var n_insts = Int64(prog_insts[unsafe_offset=i])
        for j in range(Int(n_insts)):
            var op = _read_i32(blob, po + 8 + 12 * Int64(j))
            var x = _read_i32(blob, po + 8 + 12 * Int64(j) + 4)
            var y = _read_i32(blob, po + 8 + 12 * Int64(j) + 8)
            if op < 0 or op > 5:
                prog_insts.unsafe_free()
                prog_groups.unsafe_free()
                prog_off.unsafe_free()
                prog_ops.unsafe_free()
                prog_xy.unsafe_free()
                return None
            prog_ops[unsafe_offset=base + Int64(j)] = op
            prog_xy[unsafe_offset=2 * (base + Int64(j))] = x
            prog_xy[unsafe_offset=2 * (base + Int64(j)) + 1] = y

    # --- copy strings
    var str_off = unsafe_alloc[UInt32](Int(n_strings))
    var str_len = unsafe_alloc[UInt32](Int(n_strings))
    var total_str = Int64(0)
    for i in range(Int(n_strings)):
        var so = Int64(_read_u32(blob, str_off_pos + 4 * Int64(i)))
        if so + 4 > blob_len:
            str_off.unsafe_free()
            str_len.unsafe_free()
            return None
        var n = Int64(_read_u32(blob, so))
        if so + 4 + n > blob_len:
            str_off.unsafe_free()
            str_len.unsafe_free()
            return None
        str_off[unsafe_offset=i] = UInt32(total_str)
        str_len[unsafe_offset=i] = UInt32(n)
        total_str += n
    var str_buf = unsafe_alloc[UInt8](Int(total_str))
    for i in range(Int(n_strings)):
        var so = Int64(_read_u32(blob, str_off_pos + 4 * Int64(i)))
        var n = Int64(str_len[unsafe_offset=i])
        var dst = Int64(str_off[unsafe_offset=i])
        for j in range(Int(n)):
            str_buf[unsafe_offset=dst + Int64(j)] = blob[unsafe_offset=Int(so + 4 + Int64(j))]

    # --- copy regions
    var reg_code = unsafe_alloc[Int32](Int(n_regions))
    var reg_cc = unsafe_alloc[Int32](Int(n_regions))
    var reg_main = unsafe_alloc[Int32](Int(n_regions))
    var reg_intl = unsafe_alloc[Int32](Int(n_regions))
    var reg_np = unsafe_alloc[Int32](Int(n_regions))
    var reg_npfp = unsafe_alloc[Int32](Int(n_regions))
    var reg_tr = unsafe_alloc[Int32](Int(n_regions))
    var reg_lead = unsafe_alloc[Int32](Int(n_regions))
    var reg_pext = unsafe_alloc[Int32](Int(n_regions))
    var reg_gen = unsafe_alloc[Int32](Int(n_regions))
    var reg_gm = unsafe_alloc[UInt32](Int(n_regions))
    var reg_gml = unsafe_alloc[UInt32](Int(n_regions))
    var reg_types = unsafe_alloc[Int32](Int(10 * n_regions))
    var reg_tmask = unsafe_alloc[UInt32](Int(10 * n_regions))
    var reg_nfmt = unsafe_alloc[Int32](Int(n_regions))
    var reg_nifmt = unsafe_alloc[Int32](Int(n_regions))
    var reg_fmt_ptr = unsafe_alloc[Int64](Int(n_regions))

    # first pass: count formats
    var total_fmts = Int64(0)
    for i in range(Int(n_regions)):
        var ro = Int64(_read_u32(blob, reg_off_pos + 4 * Int64(i)))
        if ro + 46 + 120 > blob_len:
            return None
        var p = ro + 46 + 120
        for _ in range(2):
            if p + 4 > blob_len:
                return None
            var nf = Int64(_read_u32(blob, p))
            p += 4
            if nf < 0 or nf > 1000 or p + 16 * nf > blob_len:
                return None
            total_fmts += nf
            p += 16 * nf
    var fmt_words = unsafe_alloc[Int32](Int(4 * total_fmts))

    var fmt_flat = Int64(0)
    for i in range(Int(n_regions)):
        var ro = Int64(_read_u32(blob, reg_off_pos + 4 * Int64(i)))
        reg_code[unsafe_offset=i] = Int32(_read_u16(blob, ro))
        reg_cc[unsafe_offset=i] = Int32(_read_u16(blob, ro + 2))
        reg_main[unsafe_offset=i] = Int32(blob[unsafe_offset=Int(ro + 4)])
        reg_intl[unsafe_offset=i] = _read_i32(blob, ro + 6)
        reg_np[unsafe_offset=i] = _read_i32(blob, ro + 10)
        reg_npfp[unsafe_offset=i] = _read_i32(blob, ro + 14)
        reg_tr[unsafe_offset=i] = _read_i32(blob, ro + 18)
        # ro + 22: territory-level national prefix formatting rule (unused)
        reg_lead[unsafe_offset=i] = _read_i32(blob, ro + 26)
        reg_pext[unsafe_offset=i] = _read_i32(blob, ro + 30)
        reg_gen[unsafe_offset=i] = _read_i32(blob, ro + 34)
        reg_gm[unsafe_offset=i] = _read_u32(blob, ro + 38)
        reg_gml[unsafe_offset=i] = _read_u32(blob, ro + 42)
        var p = ro + 46
        for t in range(10):
            reg_types[unsafe_offset=10 * i + t] = _read_i32(blob, p)
            reg_tmask[unsafe_offset=10 * i + t] = _read_u32(blob, p + 4)
            p += 12
        reg_fmt_ptr[unsafe_offset=i] = fmt_flat
        var nf = Int64(_read_u32(blob, p))
        p += 4
        reg_nfmt[unsafe_offset=i] = Int32(nf)
        for f in range(Int(nf)):
            fmt_words[unsafe_offset=fmt_flat] = _read_i32(blob, p)
            fmt_words[unsafe_offset=fmt_flat + 1] = _read_i32(blob, p + 4)
            fmt_words[unsafe_offset=fmt_flat + 2] = _read_i32(blob, p + 8)
            fmt_words[unsafe_offset=fmt_flat + 3] = _read_i32(blob, p + 12)
            p += 16
            fmt_flat += 4
        var nif = Int64(_read_u32(blob, p))
        p += 4
        reg_nifmt[unsafe_offset=i] = Int32(nif)
        for f in range(Int(nif)):
            fmt_words[unsafe_offset=fmt_flat] = _read_i32(blob, p)
            fmt_words[unsafe_offset=fmt_flat + 1] = _read_i32(blob, p + 4)
            fmt_words[unsafe_offset=fmt_flat + 2] = _read_i32(blob, p + 8)
            fmt_words[unsafe_offset=fmt_flat + 3] = _read_i32(blob, p + 12)
            p += 16
            fmt_flat += 4

    # --- copy cc map (entries are variable length: u32 cc, u32 n, n x u32)
    var cc_val = unsafe_alloc[Int32](Int(n_cc))
    var cc_start = unsafe_alloc[Int32](Int(n_cc))
    var cc_count = unsafe_alloc[Int32](Int(n_cc))
    var total_regions = Int64(0)
    var p = cc_off
    for i in range(Int(n_cc)):
        if p + 8 > blob_len:
            return None
        cc_val[unsafe_offset=i] = _read_i32(blob, p)
        var n = _read_i32(blob, p + 4)
        if n <= 0 or n > 10000:
            return None
        cc_start[unsafe_offset=i] = Int32(total_regions)
        cc_count[unsafe_offset=i] = n
        total_regions += Int64(n)
        p += 8 + 4 * Int64(n)
    if p > blob_len:
        return None
    var cc_region = unsafe_alloc[Int32](Int(total_regions))
    p = cc_off
    for i in range(Int(n_cc)):
        var n = Int64(cc_count[unsafe_offset=i])
        var start = Int64(cc_start[unsafe_offset=i])
        for j in range(Int(n)):
            cc_region[unsafe_offset=start + Int64(j)] = _read_i32(blob, p + 8 + 4 * Int64(j))
        p += 8 + 4 * n

    var slot = unsafe_alloc[Meta](1)
    var maxi = Int64(1)
    for i in range(Int(n_programs)):
        var n = Int64(prog_insts[unsafe_offset=i])
        if n > maxi:
            maxi = n
    slot[] = Meta(
        n_programs,
        prog_insts,
        prog_groups,
        prog_ops,
        prog_xy,
        prog_off,
        n_strings,
        str_off,
        str_len,
        str_buf,
        n_regions,
        reg_code,
        reg_cc,
        reg_main,
        reg_intl,
        reg_np,
        reg_npfp,
        reg_tr,
        reg_lead,
        reg_pext,
        reg_gen,
        reg_gm,
        reg_gml,
        reg_types,
        reg_tmask,
        reg_nfmt,
        reg_nifmt,
        reg_fmt_ptr,
        fmt_words,
        maxi,
        n_cc,
        cc_val,
        cc_start,
        cc_count,
        cc_region,
    )
    return slot.unsafe_bitcast[UInt8]()


# --------------------------------------------------------------------------
# Pike VM (priority-ordered Thompson NFA simulation)
# --------------------------------------------------------------------------
#
# Threads are kept in preference order (SPLIT prefers x over y), which
# reproduces leftmost-biased backtracking: a successful MATCH cuts every
# lower-priority thread, and a later MATCH from a surviving higher-priority
# thread replaces the recorded result. All patterns operate on digit
# strings, so classes are 10-bit masks.


def _vm_run[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    prog_idx: Int32,
    s: Pointer[UInt8, o],
    slen: Int,
    mode: Int32,
    want_caps: Bool,
    caps: Pointer[Int32, MutUntrackedOrigin],
    clist_pc: I32Ptr,
    clist_caps: I32Ptr,
    nlist_pc: I32Ptr,
    nlist_caps: I32Ptr,
    seen: U8Ptr,
    stack_pc: I32Ptr,
    stack_caps: I32Ptr,
) -> Int:
    """Run program `prog_idx` over digit string s[0..slen).

    Returns the end position of the winning match, or -1 on failure. In
    FULL mode a match only counts at slen; in PREFIX mode the earliest
    highest-priority match wins. With want_caps, `caps` (CAP_SLOTS wide)
    receives the winning thread's capture slots.

    Thread model: a thread is (pc, caps[CAP_SLOTS]) in parallel flat
    arrays. The epsilon closure is an explicit stack; a popped slot is
    reused for the first child (caps stay in place) and copied for every
    further child, which preserves SPLIT priority order.
    """
    var base = meta[].prog_off[unsafe_offset=Int(prog_idx)]
    var n_insts = Int(meta[].prog_insts[unsafe_offset=Int(prog_idx)])
    var ops = meta[].prog_ops
    var xy = meta[].prog_xy

    # --- epsilon closure from pc 0 at pos 0 into clist
    for i in range(n_insts):
        seen[unsafe_offset=i] = 0
    var ncl = Int(0)
    var nstack = Int(1)
    stack_pc[unsafe_offset=0] = 0
    for k in range(CAP_SLOTS):
        stack_caps[unsafe_offset=k] = -1
    while nstack > 0:
        nstack -= 1
        var pc = stack_pc[unsafe_offset=nstack]
        if seen[unsafe_offset=Int(pc)] != 0:
            continue
        seen[unsafe_offset=Int(pc)] = 1
        var op = ops[unsafe_offset=base + Int64(pc)]
        if op == OP_JMP:
            stack_pc[unsafe_offset=nstack] = xy[unsafe_offset=2 * (base + Int64(pc))]
            nstack += 1
        elif op == OP_SPLIT:
            stack_pc[unsafe_offset=nstack] = xy[unsafe_offset=2 * (base + Int64(pc)) + 1]
            nstack += 1
            for k in range(CAP_SLOTS):
                stack_caps[unsafe_offset=nstack * CAP_SLOTS + k] = stack_caps[
                    unsafe_offset=(nstack - 1) * CAP_SLOTS + k
                ]
            stack_pc[unsafe_offset=nstack] = xy[unsafe_offset=2 * (base + Int64(pc))]
            nstack += 1
        elif op == OP_SAVE:
            var slot = xy[unsafe_offset=2 * (base + Int64(pc))]
            if want_caps and slot >= 0 and slot < CAP_SLOTS:
                stack_caps[unsafe_offset=nstack * CAP_SLOTS + Int(slot)] = 0
            stack_pc[unsafe_offset=nstack] = pc + 1
            nstack += 1
        elif op == OP_EOL:
            if slen == 0:
                stack_pc[unsafe_offset=nstack] = pc + 1
                nstack += 1
        else:  # CLASS or MATCH: keep thread in clist
            clist_pc[unsafe_offset=ncl] = pc
            for k in range(CAP_SLOTS):
                clist_caps[unsafe_offset=ncl * CAP_SLOTS + k] = stack_caps[
                    unsafe_offset=nstack * CAP_SLOTS + k
                ]
            ncl += 1

    var best_end = -1
    var have_best = False
    var pos = 0
    while True:
        for i in range(n_insts):
            seen[unsafe_offset=i] = 0
        var nnl = Int(0)
        var cut = False
        var ti = 0
        while ti < ncl and not cut:
            var pc = clist_pc[unsafe_offset=ti]
            var op = ops[unsafe_offset=base + Int64(pc)]
            if op == OP_CLS:
                if pos < slen:
                    var ch = s[unsafe_offset=pos]
                    if ch >= 48 and ch <= 57:
                        var mask = xy[unsafe_offset=2 * (base + Int64(pc))]
                        if (mask >> (Int32(ch) - 48)) & 1 != 0:
                            # epsilon closure of pc+1 at pos+1 into nlist
                            var nstack2 = Int(1)
                            stack_pc[unsafe_offset=0] = pc + 1
                            for k in range(CAP_SLOTS):
                                stack_caps[unsafe_offset=k] = clist_caps[
                                    unsafe_offset=ti * CAP_SLOTS + k
                                ]
                            while nstack2 > 0:
                                nstack2 -= 1
                                var pc2 = stack_pc[unsafe_offset=nstack2]
                                if seen[unsafe_offset=Int(pc2)] != 0:
                                    continue
                                seen[unsafe_offset=Int(pc2)] = 1
                                var op2 = ops[unsafe_offset=base + Int64(pc2)]
                                if op2 == OP_JMP:
                                    stack_pc[unsafe_offset=nstack2] = xy[
                                        unsafe_offset=2 * (base + Int64(pc2))
                                    ]
                                    nstack2 += 1
                                elif op2 == OP_SPLIT:
                                    stack_pc[unsafe_offset=nstack2] = xy[
                                        unsafe_offset=2 * (base + Int64(pc2)) + 1
                                    ]
                                    nstack2 += 1
                                    for k in range(CAP_SLOTS):
                                        stack_caps[unsafe_offset=nstack2 * CAP_SLOTS + k] = stack_caps[
                                            unsafe_offset=(nstack2 - 1) * CAP_SLOTS + k
                                        ]
                                    stack_pc[unsafe_offset=nstack2] = xy[
                                        unsafe_offset=2 * (base + Int64(pc2))
                                    ]
                                    nstack2 += 1
                                elif op2 == OP_SAVE:
                                    var slot2 = xy[unsafe_offset=2 * (base + Int64(pc2))]
                                    if want_caps and slot2 >= 0 and slot2 < CAP_SLOTS:
                                        stack_caps[
                                            unsafe_offset=nstack2 * CAP_SLOTS + Int(slot2)
                                        ] = Int32(pos + 1)
                                    stack_pc[unsafe_offset=nstack2] = pc2 + 1
                                    nstack2 += 1
                                elif op2 == OP_EOL:
                                    if pos + 1 == slen:
                                        stack_pc[unsafe_offset=nstack2] = pc2 + 1
                                        nstack2 += 1
                                else:
                                    nlist_pc[unsafe_offset=nnl] = pc2
                                    for k in range(CAP_SLOTS):
                                        nlist_caps[unsafe_offset=nnl * CAP_SLOTS + k] = stack_caps[
                                            unsafe_offset=nstack2 * CAP_SLOTS + k
                                        ]
                                    nnl += 1
            elif op == OP_MATCH:
                if mode == PREFIX or pos == slen:
                    best_end = pos
                    have_best = True
                    if want_caps:
                        for k in range(CAP_SLOTS):
                            caps[unsafe_offset=k] = clist_caps[
                                unsafe_offset=ti * CAP_SLOTS + k
                            ]
                    cut = True
            ti += 1
        if have_best and nnl == 0:
            break
        if pos >= slen:
            break
        # move nlist -> clist
        for i in range(nnl):
            clist_pc[unsafe_offset=i] = nlist_pc[unsafe_offset=i]
            for k in range(CAP_SLOTS):
                clist_caps[unsafe_offset=i * CAP_SLOTS + k] = nlist_caps[
                    unsafe_offset=i * CAP_SLOTS + k
                ]
        ncl = nnl
        pos += 1
    if have_best:
        return best_end
    return -1


# --------------------------------------------------------------------------
# Pipeline helpers (ASCII-only; see module docstring for the charset rule)
# --------------------------------------------------------------------------


def _is_sep(b: UInt8) -> Bool:
    """Viability separator class, ASCII members (case-insensitive 'x')."""
    return (
        b == 45  # '-'
        or b == 120  # 'x'
        or b == 88  # 'X'
        or b == 32  # ' '
        or b == 40  # '('
        or b == 41  # ')'
        or b == 46  # '.'
        or b == 91  # '['
        or b == 93  # ']'
        or b == 47  # '/'
        or b == 126  # '~'
        or b == 42  # '*'
    )


def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


def _is_alpha(b: UInt8) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)


def _alpha_digit(b: UInt8) -> UInt8:
    """ITU E.161 keypad letter mapping; 0 when not a letter."""
    var c = b
    if c >= 97:
        c -= 32
    if c >= 65 and c <= 67:
        return 50
    if c >= 68 and c <= 70:
        return 51
    if c >= 71 and c <= 73:
        return 52
    if c >= 74 and c <= 76:
        return 53
    if c >= 77 and c <= 79:
        return 54
    if c >= 80 and c <= 83:
        return 55
    if c >= 84 and c <= 86:
        return 56
    if c >= 87 and c <= 90:
        return 57
    return 0


def _extract_possible_number[o: Origin](text: Pointer[UInt8, o], tlen: Int) -> List[UInt8]:
    """Slice from the first '+' or digit, trim trailing non-word characters
    (except '#') and '_', and cut a second number started by '[\\\\/] *x'."""
    var start = -1
    for i in range(tlen):
        var b = text[unsafe_offset=i]
        if b == 43 or _is_digit(b):
            start = i
            break
    if start < 0:
        return List[UInt8]()
    var end = tlen
    while end > start:
        var b = text[unsafe_offset=end - 1]
        var word = _is_digit(b) or _is_alpha(b) or b == 35  # keep '#'
        if b == 95 or (not word):  # trim '_' and non-word
            end -= 1
        else:
            break
    var out = List[UInt8]()
    for i in range(start, end):
        # second-number cut: '\\' or '/', then spaces, then 'x'
        var b = text[unsafe_offset=i]
        if b == 92 or b == 47:
            var j = i + 1
            while j < end and text[unsafe_offset=j] == 32:
                j += 1
            if j < end and text[unsafe_offset=j] == 120:
                break
        out.append(b)
    return out^


def _match_lit[o: Origin](s: Pointer[UInt8, o], pos: Int, hlen: Int, lit: String) -> Int:
    """Case-insensitive ASCII match of literal `lit` at s[pos..hlen);
    returns the position right after the match, or -1."""
    var lb = lit.as_bytes()
    if pos + len(lb) > hlen:
        return -1
    for j in range(len(lb)):
        var b = s[unsafe_offset=pos + j]
        if b >= 65 and b <= 90:
            b += 32
        if b != lb[j]:
            return -1
    return pos + len(lb)


def _ext_digits_tail[o: Origin](s: Pointer[UInt8, o], k0: Int, hlen: Int, lo: Int, hi: Int) -> Int:
    """Optional '.'/':' then separator run then lo..hi digits reaching hlen.
    Returns the digit-run start, or -1."""
    var k = k0
    if k < hlen and (s[unsafe_offset=k] == 58 or s[unsafe_offset=k] == 46):
        k += 1
    while k < hlen and (
        s[unsafe_offset=k] == 32
        or s[unsafe_offset=k] == 9
        or s[unsafe_offset=k] == 44
        or s[unsafe_offset=k] == 45
    ):
        k += 1
    var d0 = k
    while k < hlen and _is_digit(s[unsafe_offset=k]):
        k += 1
    if k - d0 >= lo and k - d0 <= hi and k == hlen:
        return d0
    return -1


def _ext_match_at[o: Origin](s: Pointer[UInt8, o], slen: Int, start: Int) -> Int:
    """End-anchored extension-branch match at exactly `start`; returns the
    extension digit start, or -1. Mirrors the oracle's six-branch grammar
    (see _fallback.py); non-ASCII markers never occur (charset gate)."""
    var hlen = slen
    if hlen > 0 and s[unsafe_offset=hlen - 1] == 35:  # trailing '#'
        hlen -= 1

    # Branch A: ';ext=' + 1..20 digits (no separators, no trailing '#')
    if slen - start >= 5 and _match_lit(s, start, slen, ";ext=") == start + 5:
        var k = start + 5
        var d0 = k
        while k < slen and _is_digit(s[unsafe_offset=k]):
            k += 1
        if k - d0 >= 1 and k - d0 <= 20 and k == slen:
            return d0
        return -1

    # Branches B/C: pre-sep [ \t,]*
    var i = start
    while i < hlen and (
        s[unsafe_offset=i] == 32 or s[unsafe_offset=i] == 9 or s[unsafe_offset=i] == 44
    ):
        i += 1

    # Branch B: long markers, 1..20 digits (longest first). ASCII subset;
    # accented/fullwidth markers never occur behind the charset gate.
    for mi in range(8):
        var e: Int
        if mi == 0:
            e = _match_lit(s, i, hlen, "extension")
        elif mi == 1:
            e = _match_lit(s, i, hlen, "extensio")
        elif mi == 2:
            e = _match_lit(s, i, hlen, "xtensio")
        elif mi == 3:
            e = _match_lit(s, i, hlen, "extn")
        elif mi == 4:
            e = _match_lit(s, i, hlen, "xtn")
        elif mi == 5:
            e = _match_lit(s, i, hlen, "ext")
        elif mi == 6:
            e = _match_lit(s, i, hlen, "xt")
        else:
            e = _match_lit(s, i, hlen, "anexo")
        if e >= 0:
            var r = _ext_digits_tail(s, e, hlen, 1, 20)
            if r >= 0:
                return r
    # Branch C: short markers, 1..9 digits
    for mi in range(4):
        var e: Int
        if mi == 0:
            e = _match_lit(s, i, hlen, "int")
        elif mi == 1:
            e = _match_lit(s, i, hlen, "x")
        elif mi == 2:
            e = _match_lit(s, i, hlen, "#")
        else:
            e = _match_lit(s, i, hlen, "~")
        if e >= 0:
            var r = _ext_digits_tail(s, e, hlen, 1, 9)
            if r >= 0:
                return r

    # Branch D: [- ]+ 1..6 digits '#'  (trailing hash required)
    if slen > 0 and s[unsafe_offset=slen - 1] == 35:
        var j = start
        while j < slen and (s[unsafe_offset=j] == 45 or s[unsafe_offset=j] == 32):
            j += 1
        if j > start:
            var k = j
            while k < slen and _is_digit(s[unsafe_offset=k]):
                k += 1
            if k - j >= 1 and k - j <= 6 and k == slen - 1:
                return j
        return -1

    # Branches E/F: pre-sep [ \t]* (no comma)
    i = start
    while i < hlen and (s[unsafe_offset=i] == 32 or s[unsafe_offset=i] == 9):
        i += 1
    # Branch E: ',,' or ';' then 1..15 digits
    if i < hlen and s[unsafe_offset=i] == 59:  # ';'
        var r = _ext_digits_tail(s, i + 1, hlen, 1, 15)
        if r >= 0:
            return r
    elif i + 1 < hlen and s[unsafe_offset=i] == 44 and s[unsafe_offset=i + 1] == 44:
        var r = _ext_digits_tail(s, i + 2, hlen, 1, 15)
        if r >= 0:
            return r
    # Branch F: one or more ',' then 1..9 digits
    if i < hlen and s[unsafe_offset=i] == 44:
        var k = i
        while k < hlen and s[unsafe_offset=k] == 44:
            k += 1
        var r = _ext_digits_tail(s, k, hlen, 1, 9)
        if r >= 0:
            return r
    return -1


def _is_viable[o: Origin](s: Pointer[UInt8, o], slen: Int) -> Bool:
    """len >= 2 and either exactly two digits, or 3+ separator-delimited
    digit groups followed by separator/letter/digit characters and an
    optional end-anchored extension tail."""
    if slen < 2:
        return False
    if slen == 2 and _is_digit(s[unsafe_offset=0]) and _is_digit(s[unsafe_offset=1]):
        return True
    var i = 0
    while i < slen and s[unsafe_offset=i] == 43:  # '+'
        i += 1
    var groups = 0
    while groups < 3:
        while i < slen and _is_sep(s[unsafe_offset=i]):
            i += 1
        if i < slen and _is_digit(s[unsafe_offset=i]):
            groups += 1
            i += 1
        else:
            return False
    while i < slen:
        var b = s[unsafe_offset=i]
        if _is_sep(b) or _is_digit(b) or _is_alpha(b):
            if _ext_match_at(s, slen, i) >= 0:
                return True
            i += 1
        else:
            return _ext_match_at(s, slen, i) >= 0
    return True


def _strip_extension(s: List[UInt8]) -> Tuple[List[UInt8], List[UInt8]]:
    """Leftmost end-anchored extn match whose prefix stays viable.
    Returns (extension digits, remaining number)."""
    var ext = List[UInt8]()
    var rest = List[UInt8]()
    var n = len(s)
    for start in range(n):
        var r = _ext_match_at(s.unsafe_ptr(), n, start)
        if r >= 0 and _is_viable(s.unsafe_ptr(), start):
            var hlen = n
            if s[n - 1] == 35:
                hlen = n - 1
            for p in range(r, hlen):
                ext.append(s[p])
            for p in range(0, start):
                rest.append(s[p])
            return (ext^, rest^)
    for p in range(n):
        rest.append(s[p])
    return (ext^, rest^)


def _normalize[o: Origin](s: Pointer[UInt8, o], slen: Int) -> List[UInt8]:
    """3+ ASCII letters -> map every letter to its keypad digit; otherwise
    keep digits only."""
    var letters = 0
    for i in range(slen):
        if _is_alpha(s[unsafe_offset=i]):
            letters += 1
    var out = List[UInt8]()
    if letters >= 3:
        for i in range(slen):
            var b = s[unsafe_offset=i]
            if _is_digit(b):
                out.append(b)
            else:
                var d = _alpha_digit(b)
                if d != 0:
                    out.append(d)
        return out^
    for i in range(slen):
        var b = s[unsafe_offset=i]
        if _is_digit(b):
            out.append(b)
    return out^


# --------------------------------------------------------------------------
# Match wrappers with per-call scratch
# --------------------------------------------------------------------------


struct Scratch:
    var clist_pc: I32Ptr
    var nlist_pc: I32Ptr
    var stack_pc: I32Ptr
    var clist_caps: I32Ptr
    var nlist_caps: I32Ptr
    var stack_caps: I32Ptr
    var seen: U8Ptr
    var caps: I32Ptr
    var _maxi: Int
    var _block: I32Ptr

    def __init__(out self, maxi: Int):
        var i32s = 3 * maxi + 3 * maxi * CAP_SLOTS + CAP_SLOTS
        var block = unsafe_alloc[Int32](i32s)
        self._maxi = maxi
        self._block = block
        self.clist_pc = block
        self.nlist_pc = block + maxi
        self.stack_pc = block + 2 * maxi
        self.clist_caps = block + 3 * maxi
        self.nlist_caps = block + 3 * maxi + maxi * CAP_SLOTS
        self.stack_caps = block + 3 * maxi + 2 * maxi * CAP_SLOTS
        self.caps = block + 3 * maxi + 3 * maxi * CAP_SLOTS
        self.seen = unsafe_alloc[UInt8](maxi)

    def free(mut self):
        self._block.unsafe_free()
        self.seen.unsafe_free()


def _max_insts(meta: Pointer[Meta, MutUntrackedOrigin]) -> Int:
    var m = Int(1)
    for i in range(Int(meta[].n_programs)):
        var n = Int(meta[].prog_insts[unsafe_offset=i])
        if n > m:
            m = n
    return m


def _run[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    prog_idx: Int32,
    s: Pointer[UInt8, o],
    slen: Int,
    mode: Int32,
    want_caps: Bool,
) -> Int:
    return _vm_run(
        meta,
        prog_idx,
        s,
        slen,
        mode,
        want_caps,
        scr[].caps,
        scr[].clist_pc,
        scr[].clist_caps,
        scr[].nlist_pc,
        scr[].nlist_caps,
        scr[].seen,
        scr[].stack_pc,
        scr[].stack_caps,
    )


def _full[o: Origin](meta: Pointer[Meta, MutUntrackedOrigin], scr: Pointer[Scratch, MutUntrackedOrigin], prog_idx: Int32, s: Pointer[UInt8, o], slen: Int) -> Bool:
    if prog_idx < 0:
        return False
    return _run(meta, scr, prog_idx, s, slen, FULL, False) == slen


def _prefix[o: Origin](meta: Pointer[Meta, MutUntrackedOrigin], scr: Pointer[Scratch, MutUntrackedOrigin], prog_idx: Int32, s: Pointer[UInt8, o], slen: Int) -> Int:
    if prog_idx < 0:
        return -1
    return _run(meta, scr, prog_idx, s, slen, PREFIX, False)


def _full_caps[o: Origin](meta: Pointer[Meta, MutUntrackedOrigin], scr: Pointer[Scratch, MutUntrackedOrigin], prog_idx: Int32, s: Pointer[UInt8, o], slen: Int) -> Bool:
    if prog_idx < 0:
        return False
    return _run(meta, scr, prog_idx, s, slen, FULL, True) == slen


def _prefix_caps[o: Origin](meta: Pointer[Meta, MutUntrackedOrigin], scr: Pointer[Scratch, MutUntrackedOrigin], prog_idx: Int32, s: Pointer[UInt8, o], slen: Int) -> Int:
    if prog_idx < 0:
        return -1
    return _run(meta, scr, prog_idx, s, slen, PREFIX, True)


def _str_len(meta: Pointer[Meta, MutUntrackedOrigin], idx: Int32) -> Int:
    if idx < 0:
        return 0
    return Int(meta[].str_len[unsafe_offset=Int(idx)])


def _str_ptr(meta: Pointer[Meta, MutUntrackedOrigin], idx: Int32) -> Pointer[UInt8, MutUntrackedOrigin]:
    return meta[].str_buf + Int(meta[].str_off[unsafe_offset=Int(idx)])


def _append(var dst: List[UInt8], b: UInt8) -> List[UInt8]:
    dst.append(b)
    return dst^


def _append_str(var dst: List[UInt8], meta: Pointer[Meta, MutUntrackedOrigin], idx: Int32) -> List[UInt8]:
    if idx >= 0:
        var p = _str_ptr(meta, idx)
        var n = _str_len(meta, idx)
        for i in range(n):
            dst.append(p[unsafe_offset=i])
    return dst^


def _append_bytes(var dst: List[UInt8], src: List[UInt8]) -> List[UInt8]:
    for i in range(len(src)):
        dst.append(src[i])
    return dst^


def _append_int(var dst: List[UInt8], v: Int32) -> List[UInt8]:
    var digits = List[UInt8]()
    var x = Int(v)
    if x == 0:
        dst.append(48)
        return dst^
    while x > 0:
        digits.append(UInt8(48 + x % 10))
        x //= 10
    for i in range(len(digits) - 1, -1, -1):
        dst.append(digits[i])
    return dst^


def _starts_with(s: List[UInt8], prefix: List[UInt8]) -> Bool:
    if len(s) < len(prefix):
        return False
    for i in range(len(prefix)):
        if s[i] != prefix[i]:
            return False
    return True


def _replace_all(var hay: List[UInt8], needle: String, repl: List[UInt8]) -> List[UInt8]:
    """Replace every occurrence of `needle` in `hay` with `repl`."""
    var nb = needle.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i < len(hay):
        var hit = False
        if i + len(nb) <= len(hay):
            hit = True
            for j in range(len(nb)):
                if hay[i + j] != nb[j]:
                    hit = False
                    break
        if hit:
            out = _append_bytes(out^, repl)
            i += len(nb)
        else:
            out.append(hay[i])
            i += 1
    return out^


def _first_group_ref(template: List[UInt8]) -> Int:
    """Index of the digit in the first '$k' group reference, or -1."""
    var i = 0
    while i + 1 < len(template):
        if template[i] == 36 and template[i + 1] >= 49 and template[i + 1] <= 57:
            return i
        i += 1
    return -1


def _expand_template(template: List[UInt8], nsn: List[UInt8], caps: Pointer[Int32, MutUntrackedOrigin]) -> List[UInt8]:
    """Expand $1..$9 group references against capture slots."""
    var out = List[UInt8]()
    var i = 0
    while i < len(template):
        var b = template[i]
        if (
            b == 36
            and i + 1 < len(template)
            and template[i + 1] >= 49
            and template[i + 1] <= 57
        ):
            var g = Int(template[i + 1]) - 48
            if 2 * g + 1 < CAP_SLOTS:
                var a = Int(caps[unsafe_offset=2 * g])
                var c = Int(caps[unsafe_offset=2 * g + 1])
                if a >= 0 and c >= a:
                    for p in range(a, c):
                        out.append(nsn[p])
            i += 2
        else:
            out.append(b)
            i += 1
    return out^


# --------------------------------------------------------------------------
# Engine (mirrors _fallback.py; see its docstrings for the semantics)
# --------------------------------------------------------------------------


def _test_len(meta: Pointer[Meta, MutUntrackedOrigin], ri: Int32, n: Int) -> Int32:
    var mask = meta[].reg_gmask[unsafe_offset=Int(ri)]
    if mask == NO_LEN:
        return LEN_POSSIBLE
    var local = meta[].reg_gmask_local[unsafe_offset=Int(ri)]
    if local != NO_LEN and n < 32 and (local >> UInt32(n)) & 1 != 0:
        return LEN_LOCAL_ONLY
    if n > 31:
        return LEN_TOO_LONG
    var lo = -1
    var hi = -1
    for k in range(0, 19):
        if (mask >> UInt32(k)) & 1 != 0:
            if lo < 0:
                lo = k
            hi = k
    if lo < 0:
        return LEN_POSSIBLE
    if n < lo:
        return LEN_TOO_SHORT
    if n > hi:
        return LEN_TOO_LONG
    if (mask >> UInt32(n)) & 1 != 0:
        return LEN_POSSIBLE
    return LEN_INVALID


def _npfp_candidate(
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    ri: Int32,
    nsn: List[UInt8],
) -> List[UInt8]:
    var prog = meta[].reg_npfp[unsafe_offset=Int(ri)]
    if prog < 0 or len(nsn) == 0:
        return nsn.copy()
    var end = _prefix_caps(meta, scr, prog, nsn.unsafe_ptr(), len(nsn))
    if end < 0:
        return nsn.copy()
    var n_groups = Int(meta[].prog_groups[unsafe_offset=Int(prog)])
    var last_participated = False
    if n_groups >= 1 and 2 * n_groups < CAP_SLOTS:
        last_participated = scr[].caps[unsafe_offset=2 * n_groups] >= 0
    var candidate = List[UInt8]()
    var tr = meta[].reg_transform[unsafe_offset=Int(ri)]
    if tr >= 0 and last_participated:
        # expand the transform template against the captures, then the tail
        var tmpl = List[UInt8]()
        tmpl = _append_str(tmpl^, meta, tr)
        var expanded = _expand_template(tmpl, nsn, scr[].caps)
        candidate = _append_bytes(candidate^, expanded^)
        for p in range(end, len(nsn)):
            candidate.append(nsn[p])
    else:
        for p in range(end, len(nsn)):
            candidate.append(nsn[p])
    if len(candidate) == len(nsn):
        var same = True
        for i in range(len(nsn)):
            if candidate[i] != nsn[i]:
                same = False
                break
        if same:
            return nsn.copy()
    var gen = meta[].reg_general[unsafe_offset=Int(ri)]
    if _full(meta, scr, gen, nsn.unsafe_ptr(), len(nsn)) and not _full(
        meta, scr, gen, candidate.unsafe_ptr(), len(candidate)
    ):
        return nsn.copy()
    return candidate^


def _type_helper[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    ri: Int32,
    nsn: Pointer[UInt8, o],
    n: Int,
) -> Bool:
    var gmask = meta[].reg_gmask[unsafe_offset=Int(ri)]
    if gmask != NO_LEN and n < 32 and (gmask >> UInt32(n)) & 1 == 0:
        return False
    var gen = meta[].reg_general[unsafe_offset=Int(ri)]
    if not _full(meta, scr, gen, nsn, n):
        return False
    for t in range(10):
        var prog = meta[].reg_types[unsafe_offset=10 * Int(ri) + t]
        if prog < 0:
            continue
        var mask = meta[].reg_tmask[unsafe_offset=10 * Int(ri) + t]
        if mask != NO_LEN and n < 32 and (mask >> UInt32(n)) & 1 == 0:
            continue
        if _full(meta, scr, prog, nsn, n):
            return True
    return False


def _cc_index(meta: Pointer[Meta, MutUntrackedOrigin], cc: Int32) -> Int:
    var lo = 0
    var hi = Int(meta[].n_cc) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        var v = meta[].cc_val[unsafe_offset=mid]
        if v == cc:
            return mid
        if v < cc:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _main_region_idx(meta: Pointer[Meta, MutUntrackedOrigin], cci: Int) -> Int32:
    return meta[].cc_region[unsafe_offset=Int(meta[].cc_start[unsafe_offset=cci])]


def _region_for_number[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    cc: Int32,
    nsn: Pointer[UInt8, o],
    n: Int,
) -> Int32:
    var cci = _cc_index(meta, cc)
    if cci < 0:
        return -1
    var count = Int(meta[].cc_count[unsafe_offset=cci])
    var start = Int(meta[].cc_start[unsafe_offset=cci])
    if count == 1:
        return meta[].cc_region[unsafe_offset=start]
    for j in range(count):
        var ri = meta[].cc_region[unsafe_offset=start + j]
        var ld = meta[].reg_leading[unsafe_offset=Int(ri)]
        if ld >= 0:
            if _prefix(meta, scr, ld, nsn, n) >= 0:
                return ri
        elif _type_helper(meta, scr, ri, nsn, n):
            return ri
    return -1


def _is_possible(
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    cc: Int32,
    n: Int,
) -> Bool:
    var cci = _cc_index(meta, cc)
    if cci < 0:
        return False
    var r = _test_len(meta, _main_region_idx(meta, cci), n)
    return r == LEN_POSSIBLE or r == LEN_LOCAL_ONLY


def _is_valid[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    cc: Int32,
    nsn: Pointer[UInt8, o],
    n: Int,
) -> Bool:
    var ri = _region_for_number(meta, scr, cc, nsn, n)
    if ri < 0:
        return False
    return _type_helper(meta, scr, ri, nsn, n)


def _extract_cc(meta: Pointer[Meta, MutUntrackedOrigin], digits: List[UInt8]) -> Tuple[Int32, Int]:
    """Split a leading calling code (1-3 digits); (cc, rest_start) or (0,-1)."""
    var n = len(digits)
    if n == 0 or digits[0] == 48:
        return (0, -1)
    var value = 0
    for k in range(1, 4):
        if k > n:
            break
        value = value * 10 + (Int(digits[k - 1]) - 48)
        if _cc_index(meta, Int32(value)) >= 0:
            return (Int32(value), k)
    return (0, -1)


def _first_digit_zero(s: List[UInt8], start: Int) -> Bool:
    for i in range(start, len(s)):
        if _is_digit(s[i]):
            return s[i] == 48
    return False


def _maybe_extract_cc[o: Origin](
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    s: Pointer[UInt8, o],
    slen: Int,
    region_idx: Int32,
) -> Tuple[Int32, List[UInt8], Int32, Int32]:
    """Returns (cc, national, ccs, err_type). err_type -1 = ok."""
    var empty = List[UInt8]()
    if slen == 0:
        return (0, empty^, 20, -1)
    var full: List[UInt8]
    var ccs = Int32(20)
    if s[unsafe_offset=0] == 43:  # '+'
        full = _normalize(s + 1, slen - 1)
        ccs = 1
    else:
        full = _normalize(s, slen)
        if region_idx >= 0:
            var ip = meta[].reg_intl_prefix[unsafe_offset=Int(region_idx)]
            if ip >= 0:
                var end = _prefix(meta, scr, ip, full.unsafe_ptr(), len(full))
                if end > 0 and not _first_digit_zero(full, end):
                    var stripped = List[UInt8]()
                    for p in range(end, len(full)):
                        stripped.append(full[p])
                    full = stripped^
                    ccs = 5
    if ccs != 20:
        if len(full) <= 2:
            return (0, empty^, ccs, E_TOO_SHORT_AFTER_IDD)
        var got = _extract_cc(meta, full)
        if got[0] > 0:
            var national = List[UInt8]()
            for p in range(got[1], len(full)):
                national.append(full[p])
            return (got[0], national^, ccs, -1)
        return (0, empty^, ccs, E_INVALID_CC)
    # default-country path
    if region_idx >= 0:
        var cc = meta[].reg_cc[unsafe_offset=Int(region_idx)]
        var cc_digits = List[UInt8]()
        cc_digits = _append_int(cc_digits^, cc)
        if _starts_with(full, cc_digits):
            var rem = List[UInt8]()
            for p in range(len(cc_digits), len(full)):
                rem.append(full[p])
            var potential = _npfp_candidate(meta, scr, region_idx, rem)
            var gen = meta[].reg_general[unsafe_offset=Int(region_idx)]
            var full_ok = _full(meta, scr, gen, full.unsafe_ptr(), len(full))
            var pot_ok = _full(meta, scr, gen, potential.unsafe_ptr(), len(potential))
            var too_long = _test_len(meta, region_idx, len(full)) == LEN_TOO_LONG
            if (not full_ok and pot_ok) or too_long:
                return (cc, potential^, 10, -1)
    return (0, full^, 20, -1)


def _find_sub[o: Origin](s: Pointer[UInt8, o], slen: Int, lit: String) -> Int:
    var lb = lit.as_bytes()
    var n = len(lb)
    if n > slen:
        return -1
    for i in range(slen - n + 1):
        var hit = True
        for j in range(n):
            if s[unsafe_offset=i + j] != lb[j]:
                hit = False
                break
        if hit:
            return i
    return -1


def _parse_full(
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    text: Pointer[UInt8, MutUntrackedOrigin],
    tlen: Int,
    region_idx: Int32,
    dst: Pointer[Int32, MutUntrackedOrigin],
    out_nsn: Pointer[UInt8, MutUntrackedOrigin],
    out_ext: Pointer[UInt8, MutUntrackedOrigin],
) -> Int32:
    """Full parse pipeline; fills out[0..8] + out_nsn/out_ext buffers.
    Returns the PNMOut status (0 ok, 1 parse error, 3 non-ASCII)."""
    for i in range(tlen):
        if text[unsafe_offset=i] >= 0x80:
            return ST_NON_ASCII
    if tlen > MAX_INPUT:
        dst[unsafe_offset=1] = E_TOO_LONG
        dst[unsafe_offset=3] = -2  # "too long to parse" message variant
        return ST_PARSE_ERROR

    # build_national_number_for_parsing (RFC3966 phone-context / isdn)
    var s = List[UInt8]()
    var idx_pc = _find_sub(text, tlen, ";phone-context=")
    if idx_pc >= 0:
        var pc_start = idx_pc + 15
        var pc_end = tlen
        for i in range(pc_start, tlen):
            if text[unsafe_offset=i] == 59:  # ';'
                pc_end = i
                break
        var pc_len = pc_end - pc_start
        if pc_len == 0:
            dst[unsafe_offset=1] = E_NOT_A_NUMBER
            dst[unsafe_offset=3] = -3  # "phone-context invalid"
            return ST_PARSE_ERROR
        if text[unsafe_offset=pc_start] == 43:  # '+'
            for i in range(pc_start, pc_end):
                s.append(text[unsafe_offset=i])
        var idx_tel = _find_sub(text, tlen, "tel:")
        var nstart = idx_tel + 4 if idx_tel >= 0 else 0
        for i in range(nstart, idx_pc):
            s.append(text[unsafe_offset=i])
    else:
        s = _extract_possible_number(text, tlen)
    if len(s) > 0:
        var idx_isub = _find_sub(s.unsafe_ptr(), len(s), ";isub=")
        if idx_isub > 0:
            var cut = List[UInt8]()
            for i in range(idx_isub):
                cut.append(s[i])
            s = cut^

    if len(s) == 0 or not _is_viable(s.unsafe_ptr(), len(s)):
        dst[unsafe_offset=1] = E_NOT_A_NUMBER
        dst[unsafe_offset=3] = 0
        return ST_PARSE_ERROR
    if region_idx < 0 and s[0] != 43:
        dst[unsafe_offset=1] = E_INVALID_CC
        dst[unsafe_offset=3] = -1  # "missing or invalid default region"
        return ST_PARSE_ERROR

    var parts = _strip_extension(s)
    var ext = parts[0].copy()
    var body = parts[1].copy()

    var got = _maybe_extract_cc(meta, scr, body.unsafe_ptr(), len(body), region_idx)
    var cc = got[0]
    var national = got[1].copy()
    var ccs = got[2]
    var err = got[3]
    if err == E_INVALID_CC and len(body) > 0 and body[0] == 43:
        # Strip the plus sign(s) and try again without them.
        var p = 0
        while p < len(body) and body[p] == 43:
            p += 1
        var retry = _maybe_extract_cc(meta, scr, body.unsafe_ptr() + p, len(body) - p, region_idx)
        if retry[0] == 0 and retry[3] != -1:
            dst[unsafe_offset=1] = E_INVALID_CC
            dst[unsafe_offset=3] = 1  # "could not interpret after plus-sign"
            return ST_PARSE_ERROR
        if retry[0] == 0:
            dst[unsafe_offset=1] = E_INVALID_CC
            dst[unsafe_offset=3] = 1
            return ST_PARSE_ERROR
        cc = retry[0]
        national = retry[1].copy()
        ccs = 10
    elif err == E_INVALID_CC:
        dst[unsafe_offset=1] = E_INVALID_CC
        dst[unsafe_offset=3] = 5  # "country calling code not recognised"
        return ST_PARSE_ERROR
    elif err == E_TOO_SHORT_AFTER_IDD:
        dst[unsafe_offset=1] = E_TOO_SHORT_AFTER_IDD
        dst[unsafe_offset=3] = 0
        return ST_PARSE_ERROR

    # metadata for the national-prefix strip
    var meta_ri = region_idx
    if cc != 0:
        var cci = _cc_index(meta, cc)
        if cci >= 0:
            meta_ri = _main_region_idx(meta, cci)
    elif region_idx >= 0:
        cc = meta[].reg_cc[unsafe_offset=Int(region_idx)]

    if len(national) < 2:
        dst[unsafe_offset=1] = E_TOO_SHORT_NSN
        dst[unsafe_offset=3] = 0
        return ST_PARSE_ERROR
    if meta_ri >= 0:
        var candidate = _npfp_candidate(meta, scr, meta_ri, national)
        var tl = _test_len(meta, meta_ri, len(candidate))
        if tl != LEN_TOO_SHORT and tl != LEN_LOCAL_ONLY and tl != LEN_INVALID:
            national = candidate^
    if len(national) < 2:
        dst[unsafe_offset=1] = E_TOO_SHORT_NSN
        dst[unsafe_offset=3] = 0
        return ST_PARSE_ERROR
    if len(national) > MAX_NSN:
        dst[unsafe_offset=1] = E_TOO_LONG
        dst[unsafe_offset=3] = 0
        return ST_PARSE_ERROR

    dst[unsafe_offset=2] = cc
    dst[unsafe_offset=3] = ccs
    dst[unsafe_offset=4] = 0  # leading_zeros (computed wrapper-side)
    dst[unsafe_offset=5] = 1 if _is_valid(meta, scr, cc, national.unsafe_ptr(), len(national)) else 0
    dst[unsafe_offset=6] = 1 if _is_possible(meta, scr, cc, len(national)) else 0
    dst[unsafe_offset=7] = Int32(len(national))
    dst[unsafe_offset=8] = Int32(len(ext))
    for i in range(len(national)):
        out_nsn[unsafe_offset=i] = national[i]
    for i in range(len(ext)):
        out_ext[unsafe_offset=i] = ext[i]
    return ST_OK


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _format_nsn(
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    ri: Int32,
    nsn: List[UInt8],
    international: Bool,
) -> List[UInt8]:
    var n_fmts = Int(meta[].reg_nfmt[unsafe_offset=Int(ri)])
    var base = meta[].reg_fmt_ptr[unsafe_offset=Int(ri)]
    if international and meta[].reg_nifmt[unsafe_offset=Int(ri)] > 0:
        n_fmts = Int(meta[].reg_nifmt[unsafe_offset=Int(ri)])
        base += Int64(4 * Int(meta[].reg_nfmt[unsafe_offset=Int(ri)]))
    for f in range(n_fmts):
        var w = base + Int64(4 * f)
        var ld = meta[].fmt_words[unsafe_offset=w + 3]
        if ld >= 0 and _prefix(meta, scr, ld, nsn.unsafe_ptr(), len(nsn)) < 0:
            continue
        var pat = meta[].fmt_words[unsafe_offset=w]
        if not _full_caps(meta, scr, pat, nsn.unsafe_ptr(), len(nsn)):
            continue
        var tmpl = List[UInt8]()
        tmpl = _append_str(tmpl^, meta, meta[].fmt_words[unsafe_offset=w + 1])
        var rule = tmpl^
        var np_rule = meta[].fmt_words[unsafe_offset=w + 2]
        if not international and np_rule >= 0:
            # "$NP" -> national prefix; "$FG" -> the template's first
            # group reference; then the rule replaces that first reference.
            var np = List[UInt8]()
            np = _append_str(np^, meta, meta[].reg_np_str[unsafe_offset=Int(ri)])
            var np_rule_s = List[UInt8]()
            np_rule_s = _append_str(np_rule_s^, meta, np_rule)
            var processed = _replace_all(np_rule_s^, "$NP", np^)
            var first_ref = _first_group_ref(rule)
            var fg = List[UInt8]()
            if first_ref >= 0:
                fg.append(rule[first_ref])
                fg.append(rule[first_ref + 1])
            processed = _replace_all(processed^, "$FG", fg^)
            if first_ref >= 0:
                var replaced = List[UInt8]()
                for i in range(first_ref):
                    replaced.append(rule[i])
                replaced = _append_bytes(replaced^, processed^)
                for i in range(first_ref + 2, len(rule)):
                    replaced.append(rule[i])
                rule = replaced^
            else:
                rule = processed^
        return _expand_template(rule, nsn, scr[].caps)
    return nsn.copy()


def _format_number(
    meta: Pointer[Meta, MutUntrackedOrigin],
    scr: Pointer[Scratch, MutUntrackedOrigin],
    cc: Int32,
    nsn: List[UInt8],
    ext: List[UInt8],
    fmt: Int32,
) -> List[UInt8]:
    var out = List[UInt8]()
    if fmt == 0:  # E164
        out.append(43)
        out = _append_int(out^, cc)
        out = _append_bytes(out^, nsn)
        return out^
    var cci = _cc_index(meta, cc)
    if cci < 0:
        out = _append_bytes(out^, nsn)
        return out^
    var ri = _main_region_idx(meta, cci)
    if fmt == 1:  # INTERNATIONAL
        var formatted = _format_nsn(meta, scr, ri, nsn, True)
        out.append(43)
        out = _append_int(out^, cc)
        out.append(32)
        out = _append_bytes(out^, formatted^)
    else:  # NATIONAL
        var formatted = _format_nsn(meta, scr, ri, nsn, False)
        out = _append_bytes(out^, formatted^)
    if len(ext) > 0:
        var pref = meta[].reg_pref_extn[unsafe_offset=Int(ri)]
        if pref >= 0:
            out = _append_str(out^, meta, pref)
        else:
            for b in " ext. ".as_bytes():
                out.append(b)
        out = _append_bytes(out^, ext)
    return out^


# --------------------------------------------------------------------------
# Exported C ABI
# --------------------------------------------------------------------------


@export
def phonenumbersmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def phonenumbersmojo_metadata_create(blob: U8Ptr, blob_len: Int64) abi("C") -> Handle:
    if blob_len < 0:
        return None
    return _parse_blob(blob, blob_len)


@export
def phonenumbersmojo_metadata_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var p = handle.value().unsafe_bitcast[Meta]()
    p[].free()
    p.unsafe_free()


@export
def phonenumbersmojo_parse(
    handle: Handle,
    text: U8Ptr,
    text_len: Int64,
    region_idx: Int32,
    dst: I32Ptr,
) abi("C") -> Int32:
    if not handle or text_len < 0:
        return 2
    if text_len > 100000:
        return 2
    var meta = handle.value().unsafe_bitcast[Meta]()
    var scr = unsafe_alloc[Scratch](1)
    scr[] = Scratch(Int(meta[].max_insts))
    var out_nsn = dst.unsafe_bitcast[UInt8]() + 36
    var out_ext = dst.unsafe_bitcast[UInt8]() + 60
    var status = _parse_full(
        meta, scr, text, Int(text_len), region_idx, dst, out_nsn, out_ext
    )
    scr[].free()
    scr.unsafe_free()
    dst[unsafe_offset=0] = status
    return 0


@export
def phonenumbersmojo_validate(
    handle: Handle,
    cc: Int32,
    nsn: U8Ptr,
    nsn_len: Int64,
    out_flags: I32Ptr,
) abi("C") -> Int32:
    if not handle or nsn_len < 0 or nsn_len > 64:
        return 2
    var meta = handle.value().unsafe_bitcast[Meta]()
    var scr = unsafe_alloc[Scratch](1)
    scr[] = Scratch(Int(meta[].max_insts))
    var valid = _is_valid(meta, scr, cc, nsn, Int(nsn_len))
    var possible = _is_possible(meta, scr, cc, Int(nsn_len))
    out_flags[unsafe_offset=0] = (Int32(1) if valid else Int32(0)) | (Int32(2) if possible else Int32(0))
    scr[].free()
    scr.unsafe_free()
    return 0


@export
def phonenumbersmojo_format(
    handle: Handle,
    cc: Int32,
    nsn: U8Ptr,
    nsn_len: Int64,
    ext: U8Ptr,
    ext_len: Int64,
    fmt: Int32,
    dst: U8Ptr,
    out_cap: Int64,
) abi("C") -> Int32:
    if not handle or nsn_len < 0 or ext_len < 0:
        return -1
    var meta = handle.value().unsafe_bitcast[Meta]()
    var scr = unsafe_alloc[Scratch](1)
    scr[] = Scratch(Int(meta[].max_insts))
    var nsn_l = List[UInt8]()
    for i in range(Int(nsn_len)):
        nsn_l.append(nsn[unsafe_offset=i])
    var ext_l = List[UInt8]()
    for i in range(Int(ext_len)):
        ext_l.append(ext[unsafe_offset=i])
    var result = _format_number(meta, scr, cc, nsn_l, ext_l, fmt)
    scr[].free()
    scr.unsafe_free()
    if Int64(len(result)) > out_cap:
        return -2
    for i in range(len(result)):
        dst[unsafe_offset=i] = result[i]
    return Int32(len(result))


@export
def phonenumbersmojo_validate_batch(
    handle: Handle,
    packed: U8Ptr,
    offsets: I64Ptr,
    region_idx: I32Ptr,
    n: Int64,
    out_valid: U8Ptr,
    out_status: I32Ptr,
) abi("C") -> Int32:
    if not handle or n < 0:
        return 2
    var meta = handle.value().unsafe_bitcast[Meta]()
    var scr = unsafe_alloc[Scratch](1)
    scr[] = Scratch(Int(meta[].max_insts))
    var intbuf = unsafe_alloc[Int32](9)
    var nsnbuf = unsafe_alloc[UInt8](NSN_CAP)
    var extbuf = unsafe_alloc[UInt8](EXT_CAP)
    for row in range(Int(n)):
        var start = offsets[unsafe_offset=row]
        var end = offsets[unsafe_offset=row + 1]
        var status = _parse_full(
            meta,
            scr,
            packed + Int(start),
            Int(end - start),
            region_idx[unsafe_offset=row],
            intbuf,
            nsnbuf,
            extbuf,
        )
        if status == ST_OK:
            out_status[unsafe_offset=row] = 0
            out_valid[unsafe_offset=row] = UInt8(intbuf[unsafe_offset=5])
        elif status == ST_NON_ASCII:
            out_status[unsafe_offset=row] = 3
            out_valid[unsafe_offset=row] = 0
        else:
            out_status[unsafe_offset=row] = 1
            out_valid[unsafe_offset=row] = 0
    intbuf.unsafe_free()
    nsnbuf.unsafe_free()
    extbuf.unsafe_free()
    scr[].free()
    scr.unsafe_free()
    return 0
