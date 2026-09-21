"""Pure-Python mirror of the msgpackmojo kernel (the fallback byte engine).

Implements exactly the kernel's two byte transforms — instruction stream ->
MessagePack bytes, MessagePack bytes -> record stream — in portable Python,
from the same public MessagePack specification. The wrapper runs this engine
on platforms without the native kernel (including Windows) or when
``MSGPACK_MOJO_DISABLE_NATIVE=1`` is set. Because both engines consume and
produce the same two stream formats, backend choice cannot change any
observable result; the differential suite proves it.

``unpack_engine`` returns the same ``(status, error_pos, record_stream)``
triple as the native kernel so the wrapper's error mapping is shared.
"""

from __future__ import annotations

from math import copysign as _copysign
from math import inf as _INF
from struct import Struct

_S64 = Struct("<q").unpack_from
_U64 = Struct("<Q").unpack_from
_F64 = Struct("<d").unpack_from
_U32 = Struct("<I").unpack_from
_P64 = Struct("<Q").pack
_P32 = Struct("<I").pack
_PF64 = Struct("<d").pack
_PF32 = Struct(">f").pack  # wire format is big-endian

_OP_NIL = 0x00
_OP_FALSE = 0x01
_OP_TRUE = 0x02
_OP_S64 = 0x03
_OP_U64 = 0x04
_OP_F64 = 0x05
_OP_STR = 0x06
_OP_BIN = 0x07
_OP_ARRAY = 0x08
_OP_MAP = 0x09
_OP_EXT = 0x0A

STATUS_OK = 0
STATUS_FORMAT = 1
STATUS_TRUNCATED = 2
STATUS_EXTRA_DATA = 3

# Wire header encoders (big-endian lengths), mirroring the kernel's pack side.


def _enc_int(out: bytearray, bits: int, negative: bool) -> None:
    if negative:
        if bits >= 0xFFFFFFFFFFFFFFE0:  # -32..-1
            out.append(bits & 0xFF)
        elif bits >= 0xFFFFFFFFFFFFFF80:  # int8
            out += b"\xd0"
            out.append(bits & 0xFF)
        elif bits >= 0xFFFFFFFFFFFF8000:  # int16
            out += b"\xd1" + bits.to_bytes(8, "big")[-2:]
        elif bits >= 0xFFFFFFFF80000000:  # int32
            out += b"\xd2" + bits.to_bytes(8, "big")[-4:]
        else:
            out += b"\xd3" + bits.to_bytes(8, "big")
        return
    if bits < 0x80:
        out.append(bits)
    elif bits < 0x100:
        out += b"\xcc"
        out.append(bits)
    elif bits < 0x10000:
        out += b"\xcd" + bits.to_bytes(2, "big")
    elif bits < 0x100000000:
        out += b"\xce" + bits.to_bytes(4, "big")
    else:
        out += b"\xcf" + bits.to_bytes(8, "big")


def _enc_str_header(out: bytearray, n: int, allow_str8: bool) -> None:
    if n <= 31:
        out.append(0xA0 | n)
    elif allow_str8 and n <= 0xFF:
        out += b"\xd9"
        out.append(n)
    elif n <= 0xFFFF:
        out += b"\xda" + n.to_bytes(2, "big")
    else:
        out += b"\xdb" + n.to_bytes(4, "big")


def _enc_bin_header(out: bytearray, n: int) -> None:
    if n <= 0xFF:
        out += b"\xc4"
        out.append(n)
    elif n <= 0xFFFF:
        out += b"\xc5" + n.to_bytes(2, "big")
    else:
        out += b"\xc6" + n.to_bytes(4, "big")


def _enc_array_header(out: bytearray, n: int) -> None:
    if n <= 15:
        out.append(0x90 | n)
    elif n <= 0xFFFF:
        out += b"\xdc" + n.to_bytes(2, "big")
    else:
        out += b"\xdd" + n.to_bytes(4, "big")


def _enc_map_header(out: bytearray, n: int) -> None:
    if n <= 15:
        out.append(0x80 | n)
    elif n <= 0xFFFF:
        out += b"\xde" + n.to_bytes(2, "big")
    else:
        out += b"\xdf" + n.to_bytes(4, "big")


def pack_engine(stream: bytes, flags: int) -> bytes:
    """Instruction stream -> MessagePack bytes (mirror of the kernel)."""
    use_bin = bool(flags & 1)
    single = bool(flags & 2)
    out = bytearray()
    n = len(stream)
    ip = 0
    pending = 1
    while pending:
        if ip >= n:
            raise RuntimeError("truncated instruction stream")
        tag = stream[ip]
        ip += 1
        pending -= 1
        if tag == _OP_NIL:
            out.append(0xC0)
        elif tag == _OP_FALSE:
            out.append(0xC2)
        elif tag == _OP_TRUE:
            out.append(0xC3)
        elif tag == _OP_S64 or tag == _OP_U64:
            bits = _U64(stream, ip)[0]
            ip += 8
            _enc_int(out, bits, tag == _OP_S64 and bits >> 63 != 0)
        elif tag == _OP_F64:
            if single:
                v = _F64(stream, ip)[0]
                try:
                    out += b"\xca" + _PF32(v)
                except OverflowError:
                    # C-style double->float conversion saturates to +-inf
                    # (struct.pack raises instead); match the wire behavior.
                    out += b"\xca" + _PF32(_copysign(_INF, v))
            else:
                out += b"\xcb" + stream[ip : ip + 8][::-1]
            ip += 8
        elif tag == _OP_STR or tag == _OP_BIN:
            ln = _U32(stream, ip)[0]
            ip += 4
            if tag == _OP_STR:
                _enc_str_header(out, ln, use_bin)
            elif use_bin:
                _enc_bin_header(out, ln)
            else:
                _enc_str_header(out, ln, False)
            out += stream[ip : ip + ln]
            ip += ln
        elif tag == _OP_ARRAY or tag == _OP_MAP:
            count = _U32(stream, ip)[0]
            ip += 4
            if tag == _OP_ARRAY:
                _enc_array_header(out, count)
                pending += count
            else:
                _enc_map_header(out, count)
                pending += 2 * count
        elif tag == _OP_EXT:
            code = stream[ip]
            ln = _U32(stream, ip + 1)[0]
            ip += 5
            if ln == 1:
                out += b"\xd4"
            elif ln == 2:
                out += b"\xd5"
            elif ln == 4:
                out += b"\xd6"
            elif ln == 8:
                out += b"\xd7"
            elif ln == 16:
                out += b"\xd8"
            elif ln <= 0xFF:
                out += b"\xc7"
                out.append(ln)
            elif ln <= 0xFFFF:
                out += b"\xc8" + ln.to_bytes(2, "big")
            else:
                out += b"\xc9" + ln.to_bytes(4, "big")
            out.append(code)
            out += stream[ip : ip + ln]
            ip += ln
        else:
            raise RuntimeError(f"unknown instruction opcode 0x{tag:02x}")
    if ip != n:
        raise RuntimeError("trailing instruction bytes")
    return bytes(out)


def unpack_engine(data: bytes, flags: int) -> tuple[int, int, bytes]:
    """MessagePack bytes -> (status, error_pos, record stream).

    Single pass with a pending-element counter (same termination argument as
    the kernel: every iteration consumes at least one input byte, and a value
    is complete when the counter reaches zero).
    """
    raw = bool(flags & 1)
    rec = bytearray()
    n = len(data)
    ip = 0
    pending = 1
    str_tag = _OP_BIN if raw else _OP_STR
    while pending:
        if ip >= n:
            return STATUS_TRUNCATED, ip, b""
        at = ip
        b = data[ip]
        ip += 1
        pending -= 1
        if b <= 0x7F or b >= 0xE0:
            rec.append(_OP_S64)
            rec += _P64((b if b <= 0x7F else b - 256) & 0xFFFFFFFFFFFFFFFF)
        elif 0xA0 <= b <= 0xBF:
            ln = b & 0x1F
            if ip + ln > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(str_tag)
            rec += _P32(ln)
            rec += data[ip : ip + ln]
            ip += ln
        elif 0x90 <= b <= 0x9F:
            count = b & 0x0F
            rec.append(_OP_ARRAY)
            rec += _P32(count)
            pending += count
        elif 0x80 <= b <= 0x8F:
            count = b & 0x0F
            rec.append(_OP_MAP)
            rec += _P32(count)
            pending += 2 * count
        elif b == 0xC0:
            rec.append(_OP_NIL)
        elif b == 0xC2:
            rec.append(_OP_FALSE)
        elif b == 0xC3:
            rec.append(_OP_TRUE)
        elif b == 0xC1:
            return STATUS_FORMAT, at, b""
        elif 0xC4 <= b <= 0xC6:  # bin 8/16/32
            nlen = 1 << (b - 0xC4)
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            ln = int.from_bytes(data[ip : ip + nlen], "big")
            ip += nlen
            if ip + ln > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(_OP_BIN)
            rec += _P32(ln)
            rec += data[ip : ip + ln]
            ip += ln
        elif 0xC7 <= b <= 0xC9:  # ext 8/16/32
            nlen = 1 << (b - 0xC7)
            if ip + nlen + 1 > n:
                return STATUS_TRUNCATED, at, b""
            ln = int.from_bytes(data[ip : ip + nlen], "big")
            ip += nlen
            rec.append(_OP_EXT)
            rec.append(data[ip])
            ip += 1
            if ip + ln > n:
                return STATUS_TRUNCATED, at, b""
            rec += _P32(ln)
            rec += data[ip : ip + ln]
            ip += ln
        elif b == 0xCA or b == 0xCB:  # f32 / f64
            nlen = 4 if b == 0xCA else 8
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            v = (Struct(">f" if b == 0xCA else ">d").unpack_from(data, ip))[0]
            ip += nlen
            rec.append(_OP_F64)
            rec += _PF64(v)
        elif 0xCC <= b <= 0xCF:  # uint 8/16/32/64
            nlen = 1 << (b & 0x03)
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(_OP_U64)
            rec += _P64(int.from_bytes(data[ip : ip + nlen], "big"))
            ip += nlen
        elif 0xD0 <= b <= 0xD3:  # int 8/16/32/64
            nlen = 1 << (b & 0x03)
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(_OP_S64)
            rec += _P64(
                int.from_bytes(data[ip : ip + nlen], "big", signed=True)
                & 0xFFFFFFFFFFFFFFFF
            )
            ip += nlen
        elif 0xD4 <= b <= 0xD8:  # fixext 1/2/4/8/16
            ln = 1 << (b - 0xD4)
            if ip + 1 + ln > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(_OP_EXT)
            rec.append(data[ip])
            ip += 1
            rec += _P32(ln)
            rec += data[ip : ip + ln]
            ip += ln
        elif 0xD9 <= b <= 0xDB:  # str 8/16/32
            nlen = 1 << (b - 0xD9)
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            ln = int.from_bytes(data[ip : ip + nlen], "big")
            ip += nlen
            if ip + ln > n:
                return STATUS_TRUNCATED, at, b""
            rec.append(str_tag)
            rec += _P32(ln)
            rec += data[ip : ip + ln]
            ip += ln
        elif 0xDC <= b <= 0xDF:  # array 16/32, map 16/32
            nlen = 2 if b in (0xDC, 0xDE) else 4
            if ip + nlen > n:
                return STATUS_TRUNCATED, at, b""
            count = int.from_bytes(data[ip : ip + nlen], "big")
            ip += nlen
            rec.append(_OP_MAP if b >= 0xDE else _OP_ARRAY)
            rec += _P32(count)
            pending += 2 * count if b >= 0xDE else count
        else:
            return STATUS_FORMAT, at, b""
    if ip != n:
        return STATUS_EXTRA_DATA, ip, b""
    return STATUS_OK, -1, bytes(rec)
