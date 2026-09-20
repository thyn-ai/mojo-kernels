"""Vendored pure-Python MAVLink batch decoder (the fallback backend).

Clean-room implementation of the same byte-level state machine as the Mojo
kernel (kernels/pymavlink/src/pymavmojo.mojo), driven by the same generated
dialect tables (:mod:`pymavlink_mojo._dialect`) and producing the same flat
arena layout (:mod:`pymavlink_mojo._layout`). It exists so the package is
silently correct on platforms without a native build (including Windows);
the differential test suite asserts both backends agree with the pymavlink
oracle message-for-message.

The layout is returned as plain Python lists (no numpy needed on this path).
"""

from __future__ import annotations

from pymavlink_mojo import _layout as L
from pymavlink_mojo._dialect import BY_ID


def _crc_step(crc: int, b: int) -> int:
    """One CRC-16/MCRF4XX step."""
    tmp = b ^ (crc & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF


def _pbyte(data: bytes, poff: int, plen: int, idx: int) -> int:
    """Payload byte with MAVLink zero-padding semantics (0 past plen)."""
    if idx >= plen:
        return 0
    return data[poff + idx]


def _read_uint(data: bytes, poff: int, plen: int, off: int, size: int) -> int:
    v = 0
    for j in range(size):
        v |= _pbyte(data, poff, plen, off + j) << (8 * j)
    return v


class _Arenas:
    __slots__ = ("recs", "fi", "ff", "fu", "stream", "n", "n_errors")

    def __init__(self) -> None:
        self.recs: list[int] = []
        self.fi: list[int] = []
        self.ff: list[float] = []
        self.fu = bytearray()
        # tlog mode only: the logical byte stream (every byte that entered
        # the parse buffer, in order). Record offsets point into it.
        self.stream = bytearray()
        self.n = 0
        self.n_errors = 0

    def rec_begin(self, kind: int) -> int:
        self.recs.extend([0] * L.REC_STRIDE)
        self.recs[-L.REC_STRIDE + L.R_KIND] = kind
        return len(self.recs) - L.REC_STRIDE

    def rec_set(self, base: int, col: int, value: int) -> None:
        self.recs[base + col] = value


def _decode_fields(data: bytes, poff: int, plen: int, desc: tuple, ar: _Arenas) -> None:
    """Decode one message payload into the field arenas (XML field order)."""
    import struct

    for _name, tcode, alen, woff in desc[4]:
        if tcode == L.TC_CHAR:
            for j in range(alen):
                ar.fu.append(_pbyte(data, poff, plen, woff + j))
        elif tcode == L.TC_F32:
            count = alen if alen > 0 else 1
            for j in range(count):
                bits = _read_uint(data, poff, plen, woff + 4 * j, 4)
                ar.ff.append(struct.unpack("<f", struct.pack("<I", bits))[0])
        elif tcode == L.TC_F64:
            count = alen if alen > 0 else 1
            for j in range(count):
                bits = _read_uint(data, poff, plen, woff + 8 * j, 8)
                ar.ff.append(struct.unpack("<d", struct.pack("<Q", bits))[0])
        else:
            size = 1
            if tcode in (L.TC_I16, L.TC_U16):
                size = 2
            elif tcode in (L.TC_I32, L.TC_U32):
                size = 4
            elif tcode in (L.TC_I64, L.TC_U64):
                size = 8
            count = alen if alen > 0 else 1
            for j in range(count):
                v = _read_uint(data, poff, plen, woff + size * j, size)
                if tcode == L.TC_I8:
                    ar.fi.append(v - 256 if v >= 128 else v)
                elif tcode == L.TC_I16:
                    ar.fi.append(v - 65536 if v >= 32768 else v)
                elif tcode == L.TC_I32:
                    ar.fi.append(v - 4294967296 if v >= 2147483648 else v)
                elif tcode == L.TC_I64:
                    ar.fi.append(v - (1 << 64) if v >= (1 << 63) else v)
                elif tcode == L.TC_U64:
                    # Raw bits in a signed-int64 slot (wrapper masks).
                    ar.fi.append(v - (1 << 64) if v >= (1 << 63) else v)
                else:
                    ar.fi.append(v)


def _emit_bad(
    ar: _Arenas,
    mb_off: int,
    mb_len: int,
    reason: int,
    ra: int,
    rb: int,
    rc: int,
    wire_msgid: int,
    ts_usec: int,
) -> None:
    base = ar.rec_begin(L.KIND_BAD)
    ar.rec_set(base, L.R_WIRE_MSGID, wire_msgid)
    ar.rec_set(base, L.R_VERSION, 0)
    ar.rec_set(base, L.R_CRC, -1)
    ar.rec_set(base, L.R_TS_USEC, ts_usec)
    ar.rec_set(base, L.R_MSGBUF_OFF, mb_off)
    ar.rec_set(base, L.R_MSGBUF_LEN, mb_len)
    ar.rec_set(base, L.R_PAYLOAD_OFF, 0)
    ar.rec_set(base, L.R_PAYLOAD_LEN, 0)
    ar.rec_set(base, L.R_REASON, reason)
    ar.rec_set(base, L.R_REASON_A, ra)
    ar.rec_set(base, L.R_REASON_B, rb)
    ar.rec_set(base, L.R_REASON_C, rc)
    ar.rec_set(base, L.R_FI_OFF, len(ar.fi))
    ar.rec_set(base, L.R_FF_OFF, len(ar.ff))
    ar.rec_set(base, L.R_FU_OFF, len(ar.fu))
    ar.n += 1
    ar.n_errors += 1


def _emit_unknown(ar: _Arenas, mb_off: int, mb_len: int, msgid: int, ts_usec: int) -> None:
    base = ar.rec_begin(L.KIND_UNKNOWN)
    ar.rec_set(base, L.R_WIRE_MSGID, msgid)
    ar.rec_set(base, L.R_VERSION, 0)
    ar.rec_set(base, L.R_CRC, -1)
    ar.rec_set(base, L.R_TS_USEC, ts_usec)
    ar.rec_set(base, L.R_MSGBUF_OFF, mb_off)
    ar.rec_set(base, L.R_MSGBUF_LEN, mb_len)
    ar.rec_set(base, L.R_REASON, L.REASON_NONE)
    ar.rec_set(base, L.R_FI_OFF, len(ar.fi))
    ar.rec_set(base, L.R_FF_OFF, len(ar.ff))
    ar.rec_set(base, L.R_FU_OFF, len(ar.fu))
    ar.n += 1


def _emit_message(
    data: bytes,
    mb_off: int,
    flen: int,
    v2: bool,
    hlen: int,
    mlen: int,
    incompat: int,
    msgid: int,
    desc: tuple,
    crc_got: int,
    signed: bool,
    ts_usec: int,
    ar: _Arenas,
) -> None:
    poff = mb_off + hlen
    fi_off, ff_off, fu_off = len(ar.fi), len(ar.ff), len(ar.fu)
    _decode_fields(data, poff, mlen, desc, ar)
    base = ar.rec_begin(L.KIND_MESSAGE)
    ar.rec_set(base, L.R_WIRE_MSGID, msgid)
    ar.rec_set(base, L.R_VERSION, 2 if v2 else 1)
    ar.rec_set(base, L.R_MLEN, mlen)
    if v2:
        ar.rec_set(base, L.R_SEQ, data[mb_off + 4])
        ar.rec_set(base, L.R_SYSID, data[mb_off + 5])
        ar.rec_set(base, L.R_COMPID, data[mb_off + 6])
        ar.rec_set(base, L.R_INCOMPAT, incompat)
        ar.rec_set(base, L.R_COMPAT, data[mb_off + 3])
    else:
        ar.rec_set(base, L.R_SEQ, data[mb_off + 2])
        ar.rec_set(base, L.R_SYSID, data[mb_off + 3])
        ar.rec_set(base, L.R_COMPID, data[mb_off + 4])
        ar.rec_set(base, L.R_INCOMPAT, 0)
        ar.rec_set(base, L.R_COMPAT, 0)
    ar.rec_set(base, L.R_CRC, crc_got)
    ar.rec_set(base, L.R_SIGNED, 1 if signed else 0)
    ar.rec_set(base, L.R_TS_USEC, ts_usec)
    ar.rec_set(base, L.R_MSGBUF_OFF, mb_off)
    ar.rec_set(base, L.R_MSGBUF_LEN, flen)
    ar.rec_set(base, L.R_PAYLOAD_OFF, poff)
    ar.rec_set(base, L.R_PAYLOAD_LEN, mlen)
    ar.rec_set(base, L.R_REASON, L.REASON_NONE)
    ar.rec_set(base, L.R_FI_OFF, fi_off)
    ar.rec_set(base, L.R_FI_N, len(ar.fi) - fi_off)
    ar.rec_set(base, L.R_FF_OFF, ff_off)
    ar.rec_set(base, L.R_FF_N, len(ar.ff) - ff_off)
    ar.rec_set(base, L.R_FU_OFF, fu_off)
    ar.rec_set(base, L.R_FU_N, len(ar.fu) - fu_off)
    ar.n += 1


def _handle_frame(
    data: bytes,
    mb_off: int,
    flen: int,
    v2: bool,
    hlen: int,
    mlen: int,
    incompat: int,
    sig: int,
    ts_usec: int,
    ar: _Arenas,
) -> None:
    """One complete frame: version/flags checks, dialect lookup, CRC, decode."""
    if v2 and (incompat & 0xFE) != 0:
        _emit_bad(
            ar, mb_off, flen, L.REASON_BAD_INCOMPAT, incompat, L.MAGIC_V2, hlen + 2,
            -1, ts_usec,
        )
        return
    if v2:
        msgid = (
            data[mb_off + 7]
            | (data[mb_off + 8] << 8)
            | (data[mb_off + 9] << 16)
        )
    else:
        msgid = data[mb_off + 5]
    desc = BY_ID.get(msgid)
    if desc is None:
        # Not in the dialect: pymavlink returns MAVLink_unknown before any
        # CRC validation.
        _emit_unknown(ar, mb_off, flen, msgid, ts_usec)
        return
    crc_end = mb_off + flen - sig - 2
    crc_got = data[crc_end] | (data[crc_end + 1] << 8)
    crc = 0xFFFF
    for i in range(mb_off + 1, crc_end):
        crc = _crc_step(crc, data[i])
    crc = _crc_step(crc, desc[2])
    if crc != crc_got:
        _emit_bad(
            ar, mb_off, flen, L.REASON_BAD_CRC, msgid, crc_got, crc, msgid, ts_usec
        )
        return
    _emit_message(
        data, mb_off, flen, v2, hlen, mlen, incompat, msgid, desc, crc_got,
        sig == L.SIG_LEN, ts_usec, ar,
    )


def _parse_one(data: bytes, length: int, pos: int, ts_usec: int, ar: _Arenas) -> int:
    """Emit at most one message starting at pos. Returns the new pos, or the
    unchanged pos when the remaining bytes cannot complete a message."""
    if pos >= length:
        return -1
    b = data[pos]
    if b != L.MAGIC_V1 and b != L.MAGIC_V2:
        _emit_bad(ar, pos, 1, L.REASON_BAD_PREFIX, 0, 0, 0, -1, ts_usec)
        return pos + 1
    v2 = b == L.MAGIC_V2
    hlen = L.HLEN_V2 if v2 else L.HLEN_V1
    if length - pos < 3:
        return -1
    mlen = data[pos + 1]
    incompat = data[pos + 2] if v2 else 0
    sig = L.SIG_LEN if (v2 and (incompat & L.IFLAG_SIGNED) != 0) else 0
    flen = mlen + hlen + 2 + sig
    if length - pos < flen:
        return -1
    _handle_frame(data, pos, flen, v2, hlen, mlen, incompat, sig, ts_usec, ar)
    return pos + flen


def _parse_raw(data: bytes, length: int, ar: _Arenas) -> int:
    """Raw MAVLink byte stream (serial capture, or tlog payload bytes)."""
    pos = 0
    while True:
        new_pos = _parse_one(data, length, pos, -1, ar)
        if new_pos < 0:
            return pos
        pos = new_pos


def _parse_tlog(data: bytes, length: int, ar: _Arenas) -> int:
    """".tlog" framing through pymavlink's mavlogfile recv machinery.

    pymavlink's reader interleaves two consumers of the file: pre_message
    reads each 8-byte big-endian timestamp straight from the file, while
    recv() appends file bytes to the parser's buffer. The parser's buffer
    can therefore lag the file position (after a MAVLink v2 frame the
    reader pulls 12 bytes, so a following short v1 frame leaves bytes
    buffered while the next timestamp is read past them) and can hold
    bytes from disjoint file ranges. This function reproduces that
    machinery exactly: every byte recv() reads is appended to a logical
    stream (``ar.stream``, what record offsets point into), timestamps
    never enter it, and the scan_timestamp rescan consumes file bytes the
    same byte-wise way.

    Returns the "consumed" input offset: for a well-formed complete tlog
    this is ``length``; otherwise it is the offset of the trailing unit
    (partial timestamp or partial frame) that produced no message.
    """
    stream = ar.stream
    parse_pos = 0
    file_pos = 0
    expected = L.HLEN_V1 + 2  # pymavlink's initial expected_length
    last_ts = 0.0
    last_ts_valid = False
    last_bad = False
    cur_ts_u = 0  # timestamp attached to the next emitted message (µs)
    ts_file_start = 0  # file offset of the latest timestamp read
    ts_paired = True  # whether that timestamp produced a message
    # ring of recent (stream offset, file offset) recv-append ranges, for
    # mapping an unparsed stream tail back to a file offset at the end
    ranges: list[tuple[int, int]] = []

    while True:
        # ---- pre_message: read the 8-byte big-endian timestamp from the FILE
        ts_file_start = file_pos
        ts_paired = False
        if length - file_pos < 8:
            # trailing partial timestamp: consumed from the file and dropped
            file_pos = length
        else:
            ts_u = int.from_bytes(data[file_pos : file_pos + 8], "big")
            file_pos += 8
            t = ts_u * 1e-6
            if last_bad and last_ts_valid and abs(t - last_ts) > L.SCAN_BOUND_S:
                # scan_timestamp: slide the window one FILE byte at a time
                # until the value lands within three days of the last good
                # timestamp (or the buffer runs out).
                while abs(t - last_ts) > L.SCAN_BOUND_S:
                    if file_pos >= length:
                        break
                    file_pos += 1
                    ts_u = int.from_bytes(data[file_pos - 8 : file_pos], "big")
                    t = ts_u * 1e-6
            # store the raw µs bits in a signed-int64 slot (wrapper masks)
            cur_ts_u = ts_u - (1 << 64) if ts_u >= (1 << 63) else ts_u
        # ---- recv loop: feed the parser until one message comes out (or EOF)
        done = False
        while True:
            avail = len(stream) - parse_pos
            n = expected - avail
            if n <= 0:
                n = 1
            numnew = min(n, length - file_pos)
            if numnew > 0:
                ranges.append((len(stream), file_pos))
                del ranges[:-8]
                stream += data[file_pos : file_pos + numnew]
                file_pos += numnew
            avail = len(stream) - parse_pos
            emitted = False
            if avail >= 1:
                b = stream[parse_pos]
                if b != L.MAGIC_V1 and b != L.MAGIC_V2:
                    _emit_bad(
                        ar, parse_pos, 1, L.REASON_BAD_PREFIX, 0, 0, 0, -1, cur_ts_u
                    )
                    parse_pos += 1
                    expected = L.HLEN_V1 + 2
                    emitted = True
                else:
                    v2 = b == L.MAGIC_V2
                    hlen = L.HLEN_V2 if v2 else L.HLEN_V1
                    if avail >= 3:
                        mlen = stream[parse_pos + 1]
                        incompat = stream[parse_pos + 2] if v2 else 0
                        expected = mlen + hlen + 2 + (
                            L.SIG_LEN if (v2 and (incompat & L.IFLAG_SIGNED)) else 0
                        )
                    if expected >= hlen + 2 and avail >= expected:
                        flen = expected
                        mlen = stream[parse_pos + 1]
                        incompat = stream[parse_pos + 2] if v2 else 0
                        sig = (
                            L.SIG_LEN if (v2 and (incompat & L.IFLAG_SIGNED)) else 0
                        )
                        _handle_frame(
                            stream, parse_pos, flen, v2, hlen, mlen, incompat, sig,
                            cur_ts_u, ar,
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
            if parse_pos < len(stream):
                # unparsed stream tail: map back to its file offset
                for s_start, f_start in reversed(ranges):
                    if s_start <= parse_pos:
                        return f_start + (parse_pos - s_start)
                return file_pos  # unreachable given the parse-buffer bound
            if not ts_paired:
                return ts_file_start
            return file_pos
        # ---- post_message bookkeeping: the timestamp attaches to the
        # message; only non-BAD_DATA messages advance the resync reference.
        kind = ar.recs[-L.REC_STRIDE + L.R_KIND]
        if kind == L.KIND_BAD:
            last_bad = True
        else:
            last_bad = False
            # cur_ts_u stores raw µs bits in a signed slot; the timestamp
            # itself is the unsigned value scaled (pymavlink's tusec*1e-6).
            last_ts = (cur_ts_u & ((1 << 64) - 1)) * 1e-6
            last_ts_valid = True


def parse(data: bytes, tlog_mode: bool):
    """Parse a whole buffer.

    Returns ``(recs, fi, ff, fu, src, consumed, n_errors)`` where the arenas
    are the shared record layout (see :mod:`pymavlink_mojo._layout`) and
    ``src`` is the byte string that record msgbuf/payload offsets slice:
    the input itself for raw streams, the logical parse stream for tlogs.
    """
    ar = _Arenas()
    if tlog_mode:
        consumed = _parse_tlog(data, len(data), ar)
        src = bytes(ar.stream)
    else:
        consumed = _parse_raw(data, len(data), ar)
        src = data
    return ar.recs, ar.fi, ar.ff, bytes(ar.fu), src, consumed, ar.n_errors
