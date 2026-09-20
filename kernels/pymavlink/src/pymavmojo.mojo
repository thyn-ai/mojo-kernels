"""Clean-room batch MAVLink v1/v2 stream decoder (table-driven).

Written fresh from the MAVLink serialization specification (frame layout,
CRC-16/MCRF4XX with per-message crc extra, zero-padded payload decoding,
robust resync-on-garbage scanning, and the MAVProxy/QGroundControl ".tlog"
8-byte big-endian microsecond timestamp framing) and pinned by black-box
differential tests against the published `pymavlink` package (the test
oracle). No pymavlink code is read, copied, or linked.

The kernel is a *generic* dialect-driven decoder: the message/field wire
layout tables (the bundled common.xml core dialect) are passed in through
the C ABI by the Python wrapper, which generates them from the protocol
specification (kernels/pymavlink/gen_tables_pymavlink.py ->
pymavlink_mojo/_dialect.py). This file contains only the engine:

  * A byte that is neither 0xFE (v1) nor 0xFD (v2) at a frame boundary is
    emitted as a one-byte BAD_DATA record ("Bad prefix"); scanning resumes
    at the next byte.
  * Frame length comes from the header length byte: v1 = mlen + 8,
    v2 = mlen + 12, plus a 13-byte signature block when the v2 incompat
    flag 0x01 is set. A v2 frame with any other incompat bit set becomes a
    whole-frame BAD_DATA record ("invalid incompat_flags ...").
  * A frame whose message id is not in the dialect becomes an UNKNOWN
    record carrying the raw frame bytes (no CRC check, exactly like
    pymavlink's MAVLink_unknown).
  * CRC-16/MCRF4XX over the frame (excluding magic, checksum and
    signature) plus the message's crc-extra byte; a mismatch becomes a
    whole-frame BAD_DATA record ("invalid MAVLink CRC ...").
  * Valid frames are decoded field-by-field (little-endian, wire offsets
    from the tables, zero padding when the payload is shorter than the
    full struct) into typed arenas in XML field order.

Output is a handle owning flat arenas: per-message fixed-stride records
(REC_STRIDE int64 slots) plus three spill arenas for decoded field values
(int64, float64, char bytes). The Python wrapper slices these into message
objects; the vendored pure-Python fallback produces the identical record
layout, so both backends cannot disagree.
"""

from std.memory import Pointer, bitcast, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime ERR_NONE: Int32 = 0
comptime ERR_BAD_ARGS: Int32 = 2

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I64PtrPtr = Pointer[I64Ptr, MutUntrackedOrigin]
comptime F64PtrPtr = Pointer[F64Ptr, MutUntrackedOrigin]
comptime U8PtrPtr = Pointer[U8Ptr, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime MAGIC_V1: UInt8 = 0xFE
comptime MAGIC_V2: UInt8 = 0xFD
comptime HLEN_V1: Int = 6
comptime HLEN_V2: Int = 10
comptime SIG_LEN: Int = 13
comptime IFLAG_SIGNED: Int = 0x01

# kind column values.
comptime KIND_MESSAGE: Int64 = 0
comptime KIND_BAD: Int64 = 1
comptime KIND_UNKNOWN: Int64 = 2

# reason codes (formatted into exact pymavlink reason strings by the wrapper).
comptime REASON_NONE: Int64 = 0
comptime REASON_BAD_PREFIX: Int64 = 1
comptime REASON_BAD_INCOMPAT: Int64 = 2
comptime REASON_BAD_CRC: Int64 = 3

# Field type codes (shared with the wrapper; see _dialect.py).
comptime TC_I8: UInt8 = 0
comptime TC_U8: UInt8 = 1
comptime TC_I16: UInt8 = 2
comptime TC_U16: UInt8 = 3
comptime TC_I32: UInt8 = 4
comptime TC_U32: UInt8 = 5
comptime TC_I64: UInt8 = 6
comptime TC_U64: UInt8 = 7
comptime TC_F32: UInt8 = 8
comptime TC_F64: UInt8 = 9
comptime TC_CHAR: UInt8 = 10

# Record layout (int64 slots per message).
comptime REC_STRIDE: Int = 26
comptime R_KIND: Int = 0
comptime R_WIRE_MSGID: Int = 1
comptime R_VERSION: Int = 2
comptime R_MLEN: Int = 3
comptime R_SEQ: Int = 4
comptime R_SYSID: Int = 5
comptime R_COMPID: Int = 6
comptime R_INCOMPAT: Int = 7
comptime R_COMPAT: Int = 8
comptime R_CRC: Int = 9
comptime R_SIGNED: Int = 10
comptime R_TS_USEC: Int = 11
comptime R_MSGBUF_OFF: Int = 12
comptime R_MSGBUF_LEN: Int = 13
comptime R_PAYLOAD_OFF: Int = 14
comptime R_PAYLOAD_LEN: Int = 15
comptime R_REASON: Int = 16
comptime R_REASON_A: Int = 17
comptime R_REASON_B: Int = 18
comptime R_REASON_C: Int = 19
comptime R_FI_OFF: Int = 20
comptime R_FI_N: Int = 21
comptime R_FF_OFF: Int = 22
comptime R_FF_N: Int = 23
comptime R_FU_OFF: Int = 24
comptime R_FU_N: Int = 25

# tlog scan window: 3 days in seconds (pymavlink's scan_timestamp bound).
comptime SCAN_BOUND_S: Float64 = 259200.0


struct Tables(Copyable):
    """Borrowed dialect tables (owned by the caller; read-only here)."""

    var n_msg: Int
    var msg_ids: U32Ptr  # sorted
    var msg_crc_extra: U8Ptr
    var msg_csize: U16Ptr
    var msg_nflds: U8Ptr
    var msg_foff: U32Ptr
    var fld_woff: U16Ptr
    var fld_type: U8Ptr
    var fld_alen: U16Ptr

    def __init__(
        out self,
        n_msg: Int,
        msg_ids: U32Ptr,
        msg_crc_extra: U8Ptr,
        msg_csize: U16Ptr,
        msg_nflds: U8Ptr,
        msg_foff: U32Ptr,
        fld_woff: U16Ptr,
        fld_type: U8Ptr,
        fld_alen: U16Ptr,
    ):
        self.n_msg = n_msg
        self.msg_ids = msg_ids
        self.msg_crc_extra = msg_crc_extra
        self.msg_csize = msg_csize
        self.msg_nflds = msg_nflds
        self.msg_foff = msg_foff
        self.fld_woff = fld_woff
        self.fld_type = fld_type
        self.fld_alen = fld_alen


struct ByteBuf(Copyable, Movable):
    """Growable byte arena. Contents are stable only until the next grow."""

    var ptr: U8Ptr
    var len: Int64
    var cap: Int64

    def __init__(out self):
        self.cap = 4096
        self.ptr = unsafe_alloc[UInt8](Int(self.cap))
        self.len = 0

    def ensure(mut self, extra: Int64):
        if self.len + extra <= self.cap:
            return
        var ncap = self.cap
        while ncap < self.len + extra:
            ncap = ncap * 2
        var np = unsafe_alloc[UInt8](Int(ncap))
        unsafe_memcpy(dest=np, src=self.ptr, count=Int(self.len))
        self.ptr.unsafe_free()
        self.ptr = np
        self.cap = ncap

    def append_byte(mut self, b: UInt8):
        self.ensure(1)
        self.ptr[unsafe_offset = Int(self.len)] = b
        self.len += 1

    def append(mut self, src: U8Ptr, n: Int64):
        if n <= 0:
            return
        self.ensure(n)
        unsafe_memcpy(dest=self.ptr.unsafe_offset(Int(self.len)), src=src, count=Int(n))
        self.len += n

    def free(mut self):
        self.ptr.unsafe_free()
        self.len = 0
        self.cap = 0


struct I64Vec(Copyable, Movable):
    """Growable int64 arena."""

    var ptr: I64Ptr
    var len: Int64
    var cap: Int64

    def __init__(out self):
        self.cap = 1024
        self.ptr = unsafe_alloc[Int64](Int(self.cap))
        self.len = 0

    def push(mut self, v: Int64):
        if self.len >= self.cap:
            var ncap = self.cap * 2
            var np = unsafe_alloc[Int64](Int(ncap))
            unsafe_memcpy(dest=np, src=self.ptr, count=Int(self.len))
            self.ptr.unsafe_free()
            self.ptr = np
            self.cap = ncap
        self.ptr[unsafe_offset = Int(self.len)] = v
        self.len += 1

    def free(mut self):
        self.ptr.unsafe_free()
        self.len = 0
        self.cap = 0


struct F64Vec(Copyable, Movable):
    """Growable float64 arena."""

    var ptr: F64Ptr
    var len: Int64
    var cap: Int64

    def __init__(out self):
        self.cap = 1024
        self.ptr = unsafe_alloc[Float64](Int(self.cap))
        self.len = 0

    def push(mut self, v: Float64):
        if self.len >= self.cap:
            var ncap = self.cap * 2
            var np = unsafe_alloc[Float64](Int(ncap))
            unsafe_memcpy(dest=np, src=self.ptr, count=Int(self.len))
            self.ptr.unsafe_free()
            self.ptr = np
            self.cap = ncap
        self.ptr[unsafe_offset = Int(self.len)] = v
        self.len += 1

    def free(mut self):
        self.ptr.unsafe_free()
        self.len = 0
        self.cap = 0


struct Results(Copyable, Movable):
    """All output arenas for one parse_buffer call."""

    var recs: I64Vec
    var fi: I64Vec
    var ff: F64Vec
    var fu: ByteBuf
    # tlog mode only: the logical byte stream (every byte that entered the
    # parse buffer, in order). Record offsets point into it.
    var stream: ByteBuf
    var n: Int64
    var n_errors: Int64
    var consumed: Int64

    def __init__(out self):
        self.recs = I64Vec()
        self.fi = I64Vec()
        self.ff = F64Vec()
        self.fu = ByteBuf()
        self.stream = ByteBuf()
        self.n = 0
        self.n_errors = 0
        self.consumed = 0

    def free(mut self):
        self.recs.free()
        self.fi.free()
        self.ff.free()
        self.fu.free()
        self.stream.free()


# --------------------------------------------------------------------------
# Low-level readers
# --------------------------------------------------------------------------


def _crc_step(crc: UInt32, b: UInt8) -> UInt32:
    """One CRC-16/MCRF4XX step."""
    var tmp = UInt32(b) ^ (crc & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF


def _pbyte(data: U8Ptr, poff: Int, plen: Int, idx: Int) -> UInt8:
    """Payload byte with MAVLink zero-padding semantics (0 past plen)."""
    if idx >= plen:
        return 0
    return data[unsafe_offset = poff + idx]


def _read_uint(data: U8Ptr, poff: Int, plen: Int, off: Int, size: Int) -> UInt64:
    """Little-endian unsigned read of `size` bytes at payload offset `off`."""
    var v = UInt64(0)
    for j in range(size):
        v |= UInt64(_pbyte(data, poff, plen, off + j)) << UInt64(8 * j)
    return v


def _read_be_u64(data: U8Ptr, off: Int) -> UInt64:
    """Big-endian u64 (tlog timestamp). Caller guarantees off+8 <= length."""
    var v = UInt64(0)
    for j in range(8):
        v = (v << 8) | UInt64(data[unsafe_offset = off + j])
    return v


def _find_msg(t: Tables, msgid: UInt32) -> Int:
    """Binary search in the sorted msg id table; -1 when not in the dialect."""
    var lo = 0
    var hi = t.n_msg - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        var v = Int(t.msg_ids[unsafe_offset=mid])
        if v == Int(msgid):
            return mid
        if v < Int(msgid):
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# --------------------------------------------------------------------------
# Record emission
# --------------------------------------------------------------------------


def _rec_begin(mut res: Results, kind: Int64):
    """Reserve one record; slots are filled by _rec_set."""
    for _ in range(REC_STRIDE):
        res.recs.push(0)
    res.recs.ptr[unsafe_offset = Int(res.recs.len) - REC_STRIDE + R_KIND] = kind


def _rec_set(mut res: Results, col: Int, value: Int64):
    res.recs.ptr[unsafe_offset = Int(res.recs.len) - REC_STRIDE + col] = value


def _decode_fields(
    t: Tables,
    data: U8Ptr,
    poff: Int,
    plen: Int,
    desc: Int,
    mut res: Results,
):
    """Decode one message payload into the field arenas (XML field order)."""
    var fbase = Int(t.msg_foff[unsafe_offset=desc])
    var nf = Int(t.msg_nflds[unsafe_offset=desc])
    for i in range(nf):
        var woff = Int(t.fld_woff[unsafe_offset = fbase + i])
        var tcode = t.fld_type[unsafe_offset = fbase + i]
        var alen = Int(t.fld_alen[unsafe_offset = fbase + i])
        if tcode == TC_CHAR:
            for j in range(alen):
                res.fu.append_byte(_pbyte(data, poff, plen, woff + j))
        elif tcode == TC_F32:
            var count = alen if alen > 0 else 1
            for j in range(count):
                var bits = UInt32(_read_uint(data, poff, plen, woff + 4 * j, 4))
                res.ff.push(Float64(bitcast[DType.float32](bits)))
        elif tcode == TC_F64:
            var count = alen if alen > 0 else 1
            for j in range(count):
                var bits = _read_uint(data, poff, plen, woff + 8 * j, 8)
                res.ff.push(bitcast[DType.float64](bits))
        else:
            var size = 1
            if tcode == TC_I16 or tcode == TC_U16:
                size = 2
            elif tcode == TC_I32 or tcode == TC_U32:
                size = 4
            elif tcode == TC_I64 or tcode == TC_U64:
                size = 8
            var count = alen if alen > 0 else 1
            for j in range(count):
                var v = _read_uint(data, poff, plen, woff + size * j, size)
                if tcode == TC_I8:
                    res.fi.push(Int64(v) - Int64(256 if v >= 128 else 0))
                elif tcode == TC_I16:
                    res.fi.push(Int64(v) - Int64(65536 if v >= 32768 else 0))
                elif tcode == TC_I32:
                    res.fi.push(Int64(v) - Int64(4294967296 if v >= 2147483648 else 0))
                elif tcode == TC_I64 or tcode == TC_U64:
                    # Raw bits; the wrapper applies signedness.
                    res.fi.push(bitcast[DType.int64](v))
                else:
                    res.fi.push(Int64(v))


def _emit_bad(
    mut res: Results,
    mb_off: Int,
    mb_len: Int,
    reason: Int64,
    ra: Int64,
    rb: Int64,
    rc: Int64,
    wire_msgid: Int64,
    ts_usec: Int64,
):
    _rec_begin(res, KIND_BAD)
    _rec_set(res, R_WIRE_MSGID, wire_msgid)
    _rec_set(res, R_VERSION, 0)
    _rec_set(res, R_CRC, -1)
    _rec_set(res, R_TS_USEC, ts_usec)
    _rec_set(res, R_MSGBUF_OFF, Int64(mb_off))
    _rec_set(res, R_MSGBUF_LEN, Int64(mb_len))
    _rec_set(res, R_PAYLOAD_OFF, 0)
    _rec_set(res, R_PAYLOAD_LEN, 0)
    _rec_set(res, R_REASON, reason)
    _rec_set(res, R_REASON_A, ra)
    _rec_set(res, R_REASON_B, rb)
    _rec_set(res, R_REASON_C, rc)
    _rec_set(res, R_FI_OFF, res.fi.len)
    _rec_set(res, R_FF_OFF, res.ff.len)
    _rec_set(res, R_FU_OFF, res.fu.len)
    res.n += 1
    res.n_errors += 1


def _emit_unknown(
    mut res: Results, mb_off: Int, mb_len: Int, msgid: Int64, ts_usec: Int64
):
    _rec_begin(res, KIND_UNKNOWN)
    _rec_set(res, R_WIRE_MSGID, msgid)
    _rec_set(res, R_VERSION, 0)
    _rec_set(res, R_CRC, -1)
    _rec_set(res, R_TS_USEC, ts_usec)
    _rec_set(res, R_MSGBUF_OFF, Int64(mb_off))
    _rec_set(res, R_MSGBUF_LEN, Int64(mb_len))
    _rec_set(res, R_REASON, REASON_NONE)
    _rec_set(res, R_FI_OFF, res.fi.len)
    _rec_set(res, R_FF_OFF, res.ff.len)
    _rec_set(res, R_FU_OFF, res.fu.len)
    res.n += 1


def _emit_message(
    t: Tables,
    data: U8Ptr,
    mb_off: Int,
    flen: Int,
    v2: Bool,
    hlen: Int,
    mlen: Int,
    incompat: Int,
    msgid: Int,
    desc: Int,
    crc_got: Int,
    signed: Bool,
    ts_usec: Int64,
    mut res: Results,
):
    var poff = mb_off + hlen
    var fi_off = res.fi.len
    var ff_off = res.ff.len
    var fu_off = res.fu.len
    _decode_fields(t, data, poff, mlen, desc, res)
    _rec_begin(res, KIND_MESSAGE)
    _rec_set(res, R_WIRE_MSGID, Int64(msgid))
    _rec_set(res, R_VERSION, Int64(2 if v2 else 1))
    _rec_set(res, R_MLEN, Int64(mlen))
    if v2:
        _rec_set(res, R_SEQ, Int64(data[unsafe_offset = mb_off + 4]))
        _rec_set(res, R_SYSID, Int64(data[unsafe_offset = mb_off + 5]))
        _rec_set(res, R_COMPID, Int64(data[unsafe_offset = mb_off + 6]))
        _rec_set(res, R_INCOMPAT, Int64(incompat))
        _rec_set(res, R_COMPAT, Int64(data[unsafe_offset = mb_off + 3]))
    else:
        _rec_set(res, R_SEQ, Int64(data[unsafe_offset = mb_off + 2]))
        _rec_set(res, R_SYSID, Int64(data[unsafe_offset = mb_off + 3]))
        _rec_set(res, R_COMPID, Int64(data[unsafe_offset = mb_off + 4]))
        _rec_set(res, R_INCOMPAT, 0)
        _rec_set(res, R_COMPAT, 0)
    _rec_set(res, R_CRC, Int64(crc_got))
    _rec_set(res, R_SIGNED, Int64(1 if signed else 0))
    _rec_set(res, R_TS_USEC, ts_usec)
    _rec_set(res, R_MSGBUF_OFF, Int64(mb_off))
    _rec_set(res, R_MSGBUF_LEN, Int64(flen))
    _rec_set(res, R_PAYLOAD_OFF, Int64(poff))
    _rec_set(res, R_PAYLOAD_LEN, Int64(mlen))
    _rec_set(res, R_REASON, REASON_NONE)
    _rec_set(res, R_FI_OFF, fi_off)
    _rec_set(res, R_FI_N, res.fi.len - fi_off)
    _rec_set(res, R_FF_OFF, ff_off)
    _rec_set(res, R_FF_N, res.ff.len - ff_off)
    _rec_set(res, R_FU_OFF, fu_off)
    _rec_set(res, R_FU_N, res.fu.len - fu_off)
    res.n += 1


# --------------------------------------------------------------------------
# The streaming state machine (pymavlink robust-parsing semantics)
# --------------------------------------------------------------------------


def _handle_frame(
    t: Tables,
    data: U8Ptr,
    mb_off: Int,
    flen: Int,
    v2: Bool,
    hlen: Int,
    mlen: Int,
    incompat: Int,
    sig: Int,
    ts_usec: Int64,
    mut res: Results,
):
    """One complete frame: version/flags checks, dialect lookup, CRC, decode."""
    if v2 and (incompat & 0xFE) != 0:
        # oracle: MAVError("invalid incompat_flags 0x%x 0x%x %u") -> BAD_DATA
        _emit_bad(
            res,
            mb_off,
            flen,
            REASON_BAD_INCOMPAT,
            Int64(incompat),
            Int64(MAGIC_V2),
            Int64(hlen + 2),
            -1,
            ts_usec,
        )
        return
    var msgid: Int
    if v2:
        msgid = (
            Int(data[unsafe_offset = mb_off + 7])
            | (Int(data[unsafe_offset = mb_off + 8]) << 8)
            | (Int(data[unsafe_offset = mb_off + 9]) << 16)
        )
    else:
        msgid = Int(data[unsafe_offset = mb_off + 5])
    var desc = _find_msg(t, UInt32(msgid))
    if desc < 0:
        # Not in the dialect: pymavlink returns MAVLink_unknown before any
        # CRC validation.
        _emit_unknown(res, mb_off, flen, Int64(msgid), ts_usec)
        return
    var crc_end = mb_off + flen - sig - 2
    var crc_got = Int(data[unsafe_offset = crc_end]) | (
        Int(data[unsafe_offset = crc_end + 1]) << 8
    )
    var crc = UInt32(0xFFFF)
    for i in range(mb_off + 1, crc_end):
        crc = _crc_step(crc, data[unsafe_offset = i])
    crc = _crc_step(crc, t.msg_crc_extra[unsafe_offset=desc])
    if Int(crc) != crc_got:
        _emit_bad(
            res,
            mb_off,
            flen,
            REASON_BAD_CRC,
            Int64(msgid),
            Int64(crc_got),
            Int64(crc),
            Int64(msgid),
            ts_usec,
        )
        return
    _emit_message(
        t,
        data,
        mb_off,
        flen,
        v2,
        hlen,
        mlen,
        incompat,
        msgid,
        desc,
        crc_got,
        sig == SIG_LEN,
        ts_usec,
        res,
    )


def _parse_one(
    t: Tables, data: U8Ptr, length: Int, mut pos: Int, ts_usec: Int64, mut res: Results
) -> Bool:
    """Emit at most one message starting at `pos`. False when the remaining
    bytes cannot complete a message (pos left at the incomplete frame)."""
    if pos >= length:
        return False
    var b = data[unsafe_offset = pos]
    if b != MAGIC_V1 and b != MAGIC_V2:
        _emit_bad(res, pos, 1, REASON_BAD_PREFIX, 0, 0, 0, -1, ts_usec)
        pos += 1
        return True
    var v2 = b == MAGIC_V2
    var hlen = HLEN_V2 if v2 else HLEN_V1
    if length - pos < 3:
        return False
    var mlen = Int(data[unsafe_offset = pos + 1])
    var incompat = Int(data[unsafe_offset = pos + 2]) if v2 else 0
    var sig = SIG_LEN if (v2 and (incompat & IFLAG_SIGNED) != 0) else 0
    var flen = mlen + hlen + 2 + sig
    if length - pos < flen:
        return False
    _handle_frame(t, data, pos, flen, v2, hlen, mlen, incompat, sig, ts_usec, res)
    pos += flen
    return True


def _parse_raw(t: Tables, data: U8Ptr, length: Int, mut res: Results) -> Int:
    """Raw MAVLink byte stream (serial capture, or tlog payload bytes)."""
    var pos = 0
    while _parse_one(t, data, length, pos, -1, res):
        pass
    return pos


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _parse_tlog(t: Tables, data: U8Ptr, length: Int, mut res: Results) -> Int:
    """".tlog" framing through pymavlink's mavlogfile recv machinery.

    pymavlink's reader interleaves two consumers of the file: pre_message
    reads each 8-byte big-endian timestamp straight from the file, while
    recv() appends file bytes to the parser's buffer. The parser's buffer
    can therefore lag the file position (after a MAVLink v2 frame the
    reader pulls 12 bytes, so a following short v1 frame leaves bytes
    buffered while the next timestamp is read past them) and can hold
    bytes from disjoint file ranges. This reproduces that machinery
    exactly: every byte recv() reads is appended to a logical stream (the
    arena record offsets point into), timestamps never enter it, and the
    scan_timestamp rescan consumes file bytes the same byte-wise way.

    Returns the "consumed" input offset: for a well-formed complete tlog
    this is `length`; otherwise the offset of the trailing unit (partial
    timestamp or partial frame) that produced no message.
    """
    var parse_pos = 0
    var file_pos = 0
    var expected = HLEN_V1 + 2  # pymavlink's initial expected_length
    var last_ts = 0.0
    var last_ts_valid = False
    var last_bad = False
    var cur_ts_u = Int64(0)  # timestamp for the next emitted message (µs)
    var ts_file_start: Int  # file offset of the latest timestamp read
    var ts_paired: Bool  # whether that timestamp produced a message
    # ring of the last 8 (stream offset, file offset) recv-append ranges,
    # for mapping an unparsed stream tail back to a file offset at the end
    var range_s = I64Vec()
    var range_f = I64Vec()
    for _ in range(8):
        range_s.push(0)
        range_f.push(0)
    var range_n = 0

    while True:
        # ---- pre_message: read the 8-byte big-endian timestamp from the FILE
        ts_file_start = file_pos
        ts_paired = False
        if length - file_pos < 8:
            # trailing partial timestamp: consumed from the file and dropped
            file_pos = length
        else:
            var ts_u = _read_be_u64(data, file_pos)
            file_pos += 8
            var tsec = Float64(ts_u) * 1e-6
            if last_bad and last_ts_valid and _abs(tsec - last_ts) > SCAN_BOUND_S:
                # scan_timestamp: slide the window one FILE byte at a time
                while _abs(tsec - last_ts) > SCAN_BOUND_S:
                    if file_pos >= length:
                        break
                    file_pos += 1
                    ts_u = _read_be_u64(data, file_pos - 8)
                    tsec = Float64(ts_u) * 1e-6
            cur_ts_u = Int64(bitcast[DType.int64](ts_u))
        # ---- recv loop: feed the parser until one message comes out (or EOF)
        var done = False
        while True:
            var avail = Int(res.stream.len) - parse_pos
            var n = expected - avail
            if n <= 0:
                n = 1
            var numnew = n
            if numnew > length - file_pos:
                numnew = length - file_pos
            if numnew > 0:
                range_s.ptr[unsafe_offset = range_n % 8] = res.stream.len
                range_f.ptr[unsafe_offset = range_n % 8] = Int64(file_pos)
                range_n += 1
                res.stream.append(data.unsafe_offset(file_pos), Int64(numnew))
                file_pos += numnew
            avail = Int(res.stream.len) - parse_pos
            var emitted = False
            if avail >= 1:
                var b = res.stream.ptr[unsafe_offset = parse_pos]
                if b != MAGIC_V1 and b != MAGIC_V2:
                    _emit_bad(res, parse_pos, 1, REASON_BAD_PREFIX, 0, 0, 0, -1, cur_ts_u)
                    parse_pos += 1
                    expected = HLEN_V1 + 2
                    emitted = True
                else:
                    var v2 = b == MAGIC_V2
                    var hlen = HLEN_V2 if v2 else HLEN_V1
                    if avail >= 3:
                        var mlen = Int(res.stream.ptr[unsafe_offset = parse_pos + 1])
                        var incompat = Int(res.stream.ptr[unsafe_offset = parse_pos + 2]) if v2 else 0
                        expected = mlen + hlen + 2 + (
                            SIG_LEN if (v2 and (incompat & IFLAG_SIGNED) != 0) else 0
                        )
                    if expected >= hlen + 2 and avail >= expected:
                        var flen = expected
                        var mlen2 = Int(res.stream.ptr[unsafe_offset = parse_pos + 1])
                        var incompat2 = Int(res.stream.ptr[unsafe_offset = parse_pos + 2]) if v2 else 0
                        var sig2 = SIG_LEN if (v2 and (incompat2 & IFLAG_SIGNED) != 0) else 0
                        _handle_frame(
                            t,
                            res.stream.ptr,
                            parse_pos,
                            flen,
                            v2,
                            hlen,
                            mlen2,
                            incompat2,
                            sig2,
                            cur_ts_u,
                            res,
                        )
                        parse_pos += flen
                        expected = hlen + 2
                        emitted = True
            if emitted:
                ts_paired = True
                break
            if numnew == 0:
                done = True
                break
        if done:
            if parse_pos < Int(res.stream.len):
                # unparsed stream tail: map back to its file offset, walking
                # the ring from the most recent append backwards (appends
                # arrive with increasing stream offsets, so the first match
                # in recency order is the range containing parse_pos).
                var k = 1
                while k <= 8 and k <= range_n:
                    var idx = (range_n - k) % 8
                    if Int(range_s.ptr[unsafe_offset = idx]) <= parse_pos:
                        return Int(range_f.ptr[unsafe_offset = idx]) + (
                            parse_pos - Int(range_s.ptr[unsafe_offset = idx])
                        )
                    k += 1
                return file_pos  # unreachable given the parse-buffer bound
            if not ts_paired:
                return ts_file_start
            return file_pos
        # ---- post_message bookkeeping: the timestamp attaches to the
        # message; only non-BAD_DATA messages advance the resync reference.
        var kind = res.recs.ptr[unsafe_offset = Int(res.recs.len) - REC_STRIDE + R_KIND]
        if kind == KIND_BAD:
            last_bad = True
        else:
            last_bad = False
            last_ts = Float64(bitcast[DType.uint64](cur_ts_u)) * 1e-6
            last_ts_valid = True


def _parse(t: Tables, data: U8Ptr, length: Int, tlog: Bool) -> Results:
    var res = Results()
    if tlog:
        res.consumed = Int64(_parse_tlog(t, data, length, res))
    else:
        res.consumed = Int64(_parse_raw(t, data, length, res))
    return res^


# --------------------------------------------------------------------------
# Exported C ABI
# --------------------------------------------------------------------------


@export
def pymavmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def pymavmojo_parse(
    data: U8Ptr,
    length: Int64,
    tlog_mode: Int32,
    n_msg: Int64,
    msg_ids: U32Ptr,
    msg_crc_extra: U8Ptr,
    msg_csize: U16Ptr,
    msg_nflds: U8Ptr,
    msg_foff: U32Ptr,
    fld_woff: U16Ptr,
    fld_type: U8Ptr,
    fld_alen: U16Ptr,
    out_err: I32Ptr,
) abi("C") -> Handle:
    """Parse a whole buffer. Returns a handle owning the result arenas."""
    if length < 0 or n_msg < 0:
        out_err[] = ERR_BAD_ARGS
        return None
    var t = Tables(
        Int(n_msg),
        msg_ids,
        msg_crc_extra,
        msg_csize,
        msg_nflds,
        msg_foff,
        fld_woff,
        fld_type,
        fld_alen,
    )
    var res = _parse(t, data, Int(length), tlog_mode != 0)
    out_err[] = ERR_NONE
    var boxed = unsafe_alloc[Results](1)
    boxed[] = res^
    return boxed.unsafe_bitcast[UInt8]()


@export
def pymavmojo_msg_count(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    var r = handle.value().unsafe_bitcast[Results]()
    return r[].n


@export
def pymavmojo_error_count(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    var r = handle.value().unsafe_bitcast[Results]()
    return r[].n_errors


@export
def pymavmojo_consumed(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    var r = handle.value().unsafe_bitcast[Results]()
    return r[].consumed


@export
def pymavmojo_arrays(
    handle: Handle,
    recs: I64PtrPtr,
    fi: I64PtrPtr,
    ff: F64PtrPtr,
    fu: U8PtrPtr,
    stream: U8PtrPtr,
    n_recs: I64Ptr,
    n_fi: I64Ptr,
    n_ff: I64Ptr,
    n_fu: I64Ptr,
    n_stream: I64Ptr,
) abi("C") -> Int32:
    """Expose the arenas; pointers stay valid until pymavmojo_free."""
    if not handle:
        return 1
    var r = handle.value().unsafe_bitcast[Results]()
    recs[] = r[].recs.ptr
    fi[] = r[].fi.ptr
    ff[] = r[].ff.ptr
    fu[] = r[].fu.ptr
    stream[] = r[].stream.ptr
    n_recs[] = r[].recs.len
    n_fi[] = r[].fi.len
    n_ff[] = r[].ff.len
    n_fu[] = r[].fu.len
    n_stream[] = r[].stream.len
    return 0


@export
def pymavmojo_free(handle: Handle) abi("C"):
    if not handle:
        return
    var r = handle.value().unsafe_bitcast[Results]()
    r[].free()
    r.unsafe_free()
