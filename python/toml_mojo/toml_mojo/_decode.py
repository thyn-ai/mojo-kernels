"""Assembler for the tomlmojo record stream.

Turns the kernel's binary record stream (format documented in
``kernels/toml/src/tomlmojo.mojo``) into the same typed Python objects
``tomllib.loads`` produces: ``dict``/``list``/``str``/``int``/``float``/
``bool``/``datetime.date``/``datetime.time``/``datetime.datetime``.

Numbers arrive as their source literals and are converted with Python's own
``int()``/``float()`` — the exact conversions tomllib performs — so integer
and float values agree with the reference bit-for-bit (including bignum ints
outside int64 and every representable double).

Any inconsistency in the stream raises :class:`NativeUnavailable` (the
kernel's output is validated before it reaches user code) so the wrapper can
fall back to stdlib ``tomllib`` instead of returning corrupt data.
"""

from __future__ import annotations

import datetime as _dt
from struct import Struct

from toml_mojo._native import NativeUnavailable

_MAGIC = b"TMO1"

_U16 = Struct("<H")
_I16 = Struct("<h")
_U32 = Struct("<I")

_REC_TABLE = 1
_REC_AOT_ELEM = 2
_REC_VALUE = 3

_VT_STR = 1
_VT_INT_DEC = 2
_VT_INT_HEX = 3
_VT_INT_OCT = 4
_VT_INT_BIN = 5
_VT_FLOAT = 6
_VT_BOOL = 7
_VT_ARRAY = 8
_VT_INLINE = 9
_VT_DATE = 10
_VT_TIME = 11
_VT_DT_LOCAL = 12
_VT_DT_OFFSET = 13

_date = _dt.date
_time = _dt.time
_datetime = _dt.datetime
_utc = _dt.timezone.utc
_timedelta = _dt.timedelta
_timezone = _dt.timezone


def _corrupt(pos: int, why: str) -> NativeUnavailable:
    return NativeUnavailable(f"corrupt kernel stream at byte {pos}: {why}")


def _read_value(buf: bytes, pos: int):
    """Decode one tagged value; returns (value, next_pos)."""
    tag = buf[pos]
    pos += 1
    if tag == _VT_STR:
        (ln,) = _U32.unpack_from(buf, pos)
        pos += 4
        return buf[pos : pos + ln].decode("utf-8", "surrogatepass"), pos + ln
    if _VT_INT_DEC <= tag <= _VT_FLOAT:
        (ln,) = _U32.unpack_from(buf, pos)
        pos += 4
        lit = buf[pos : pos + ln]
        end = pos + ln
        if tag == _VT_INT_DEC:
            return int(lit), end
        if tag == _VT_INT_HEX:
            return int(lit, 16), end
        if tag == _VT_INT_OCT:
            return int(lit, 8), end
        if tag == _VT_INT_BIN:
            return int(lit, 2), end
        return float(lit), end
    if tag == _VT_BOOL:
        return buf[pos] != 0, pos + 1
    if tag == _VT_ARRAY:
        (cnt,) = _U32.unpack_from(buf, pos)
        pos += 4
        out = []
        for _ in range(cnt):
            v, pos = _read_value(buf, pos)
            out.append(v)
        return out, pos
    if tag == _VT_INLINE:
        (cnt,) = _U32.unpack_from(buf, pos)
        pos += 4
        out: dict = {}
        for _ in range(cnt):
            (nsegs,) = _U16.unpack_from(buf, pos)
            pos += 2
            if nsegs < 1:
                raise _corrupt(pos, "inline-table entry with empty key")
            segs = []
            for _ in range(nsegs):
                (klen,) = _U32.unpack_from(buf, pos)
                pos += 4
                segs.append(buf[pos : pos + klen].decode("utf-8", "surrogatepass"))
                pos += klen
            v, pos = _read_value(buf, pos)
            parent = out
            for name in segs[:-1]:
                nxt = parent.get(name)
                if nxt is None:
                    nxt = {}
                    parent[name] = nxt
                parent = nxt
            parent[segs[-1]] = v
        return out, pos
    if tag == _VT_DATE:
        (y,) = _U16.unpack_from(buf, pos)
        return _date(y, buf[pos + 2], buf[pos + 3]), pos + 4
    if tag == _VT_TIME:
        (us,) = _U32.unpack_from(buf, pos + 3)
        return _time(buf[pos], buf[pos + 1], buf[pos + 2], us), pos + 7
    if tag == _VT_DT_LOCAL or tag == _VT_DT_OFFSET:
        (y,) = _U16.unpack_from(buf, pos)
        (us,) = _U32.unpack_from(buf, pos + 7)
        end = pos + 11
        tz = None
        if tag == _VT_DT_OFFSET:
            (off,) = _I16.unpack_from(buf, end)
            end += 2
            tz = _utc if off == 0 else _timezone(_timedelta(minutes=off))
        return (
            _datetime(
                y, buf[pos + 2], buf[pos + 3], buf[pos + 4], buf[pos + 5], buf[pos + 6], us, tz
            )
        ), end
    raise _corrupt(pos - 1, f"unknown value tag {tag}")


def _navigate(root: dict, segs: list, pos: int):
    """Walk/create the container chain for a record's path prefix."""
    obj = root
    for name, aot_index in segs:
        if aot_index >= 0:
            try:
                obj = obj[name][aot_index]
            except (KeyError, IndexError, TypeError) as exc:
                raise _corrupt(pos, "array-of-tables descent misses its element") from exc
        else:
            nxt = obj.get(name)
            if nxt is None:
                nxt = {}
                obj[name] = nxt
            obj = nxt
    return obj


def _read_path(buf: bytes, pos: int) -> tuple[list, int]:
    (nsegs,) = _U16.unpack_from(buf, pos)
    pos += 2
    segs = []
    for _ in range(nsegs):
        flags = buf[pos]
        pos += 1
        aot_index = -1
        if flags & 1:
            (aot_index,) = _U32.unpack_from(buf, pos)
            pos += 4
        (klen,) = _U32.unpack_from(buf, pos)
        pos += 4
        segs.append((buf[pos : pos + klen].decode("utf-8", "surrogatepass"), aot_index))
        pos += klen
    return segs, pos


def assemble(buf: bytes) -> dict:
    """Assemble a kernel record stream into the parsed document."""
    if len(buf) < 4 or buf[:4] != _MAGIC:
        raise NativeUnavailable("kernel stream has a bad magic header")
    pos = 4
    n = len(buf)
    root: dict = {}
    try:
        while pos < n:
            tag = buf[pos]
            pos += 1
            segs, pos = _read_path(buf, pos)
            if not segs:
                raise _corrupt(pos, "record with empty path")
            if tag == _REC_TABLE:
                parent = _navigate(root, segs[:-1], pos)
                name, aot_index = segs[-1]
                if aot_index >= 0:
                    raise _corrupt(pos, "table record ends in an array descent")
                existing = parent.get(name)
                if existing is None:
                    parent[name] = {}
            elif tag == _REC_AOT_ELEM:
                (elem_index,) = _U32.unpack_from(buf, pos)
                pos += 4
                parent = _navigate(root, segs[:-1], pos)
                name, aot_index = segs[-1]
                if aot_index >= 0:
                    raise _corrupt(pos, "array-of-tables path ends in a descent")
                lst = parent.get(name)
                if lst is None:
                    lst = []
                    parent[name] = lst
                if len(lst) != elem_index:
                    raise _corrupt(pos, "array-of-tables element out of order")
                lst.append({})
            elif tag == _REC_VALUE:
                parent = _navigate(root, segs[:-1], pos)
                name, aot_index = segs[-1]
                if aot_index >= 0:
                    raise _corrupt(pos, "value record ends in an array descent")
                value, pos = _read_value(buf, pos)
                parent[name] = value
            else:
                raise _corrupt(pos - 1, f"unknown record tag {tag}")
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, NativeUnavailable):
            raise
        raise _corrupt(pos, str(exc)) from exc
    return root
