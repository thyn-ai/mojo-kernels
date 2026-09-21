"""Assembler for the msgpackmojo record stream.

Turns the kernel's record stream (layout documented in
``kernels/msgpack/src/msgpackmojo.mojo``) into the same typed Python objects
``msgpack.unpackb`` returns: ``None``/``bool``/``int``/``float``/``str``/
``bytes``/``list``/``dict``/``ExtType``/``Timestamp``.

Container assembly is iterative (an explicit stack of in-progress
containers), so nesting depth is bounded by memory, not by the Python call
stack. Option semantics (``use_list``, ``strict_map_key``, ``raw``,
``unicode_errors``, ``timestamp``, ``ext_hook``, ``max_*_len``) are applied
here, identically for both unpack backends.

Any structural inconsistency in the stream raises :class:`NativeUnavailable`
(the kernel's output is validated before it reaches user code), so the
wrapper can re-unpack with the pure-Python engine instead of returning
corrupt data.
"""

from __future__ import annotations

from struct import Struct

from msgpack_mojo._native import NativeUnavailable
from msgpack_mojo.ext import ExtType, Timestamp

_S64 = Struct("<q").unpack_from
_U64 = Struct("<Q").unpack_from
_F64 = Struct("<d").unpack_from
_U32 = Struct("<I").unpack_from

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

_NOKEY = object()  # map-frame sentinel: the next value is a key


def _corrupt(pos: int, why: str) -> NativeUnavailable:
    return NativeUnavailable(f"corrupt kernel stream at byte {pos}: {why}")


def assemble(
    rec: bytes,
    *,
    use_list: bool = True,
    strict_map_key: bool = True,
    timestamp: int = 0,
    unicode_errors: str | None = None,
    ext_hook=ExtType,
    max_str_len: int = -1,
    max_bin_len: int = -1,
    max_array_len: int = -1,
    max_map_len: int = -1,
    max_ext_len: int = -1,
):
    """Assemble one record stream into the unpacked Python value."""
    errors = unicode_errors or "strict"
    n = len(rec)
    i = 0
    # Stack of in-progress containers; each frame is
    # [container, remaining_slots, is_map, pending_key].
    stack: list = []
    while True:
        if i >= n:
            raise _corrupt(i, "truncated record")
        tag = rec[i]
        i += 1
        if tag == _OP_NIL:
            val = None
        elif tag == _OP_FALSE:
            val = False
        elif tag == _OP_TRUE:
            val = True
        elif tag == _OP_S64:
            if i + 8 > n:
                raise _corrupt(i, "truncated s64")
            val = _S64(rec, i)[0]
            i += 8
        elif tag == _OP_U64:
            if i + 8 > n:
                raise _corrupt(i, "truncated u64")
            val = _U64(rec, i)[0]
            i += 8
        elif tag == _OP_F64:
            if i + 8 > n:
                raise _corrupt(i, "truncated f64")
            val = _F64(rec, i)[0]
            i += 8
        elif tag == _OP_STR or tag == _OP_BIN:
            if i + 4 > n:
                raise _corrupt(i, "truncated bytes header")
            ln = _U32(rec, i)[0]
            i += 4
            if i + ln > n:
                raise _corrupt(i, "truncated bytes payload")
            if tag == _OP_STR:
                if max_str_len >= 0 and ln > max_str_len:
                    raise ValueError(f"{ln} exceeds max_str_len({max_str_len})")
                val = rec[i : i + ln].decode("utf-8", errors)
            else:
                if max_bin_len >= 0 and ln > max_bin_len:
                    raise ValueError(f"{ln} exceeds max_bin_len({max_bin_len})")
                val = rec[i : i + ln]
            i += ln
        elif tag == _OP_ARRAY or tag == _OP_MAP:
            if i + 4 > n:
                raise _corrupt(i, "truncated container header")
            count = _U32(rec, i)[0]
            i += 4
            if tag == _OP_ARRAY:
                if max_array_len >= 0 and count > max_array_len:
                    raise ValueError(f"{count} exceeds max_array_len({max_array_len})")
                if count == 0:
                    val = [] if use_list else ()
                else:
                    stack.append([[], count, False, None])
                    continue
            else:
                if max_map_len >= 0 and count > max_map_len:
                    raise ValueError(f"{count} exceeds max_map_len({max_map_len})")
                if count == 0:
                    val = {}
                else:
                    stack.append([{}, count, True, _NOKEY])
                    continue
        elif tag == _OP_EXT:
            if i + 5 > n:
                raise _corrupt(i, "truncated ext header")
            code = rec[i]
            if code > 127:
                code -= 256
            ln = _U32(rec, i + 1)[0]
            i += 5
            if i + ln > n:
                raise _corrupt(i, "truncated ext payload")
            if max_ext_len >= 0 and ln > max_ext_len:
                raise ValueError(f"{ln} exceeds max_ext_len({max_ext_len})")
            payload = rec[i : i + ln]
            i += ln
            if code == -1:
                val = _convert_timestamp(payload, timestamp)
            else:
                if code < 0:
                    raise ValueError("code must be 0~127")
                val = ext_hook(code, payload)
        else:
            raise _corrupt(i - 1, f"unknown record tag 0x{tag:02x}")
        # Deliver `val` to the innermost open container, closing frames as
        # they fill; returns only when the root value is complete.
        while True:
            if not stack:
                if i != n:
                    raise _corrupt(i, "trailing record bytes")
                return val
            fr = stack[-1]
            if fr[2]:  # map frame
                if fr[3] is _NOKEY:
                    if strict_map_key and type(val) not in (str, bytes):
                        raise ValueError(
                            f"{type(val).__name__} is not allowed for map key "
                            "when strict_map_key=True"
                        )
                    fr[3] = val
                    break
                fr[0][fr[3]] = val
                fr[3] = _NOKEY
                fr[1] -= 1
            else:
                fr[0].append(val)
                fr[1] -= 1
            if fr[1]:
                break
            stack.pop()
            if fr[2] or use_list:
                val = fr[0]
            else:
                val = tuple(fr[0])


def _convert_timestamp(payload: bytes, timestamp: int):
    """Timestamp extension (code -1) per the ``timestamp`` unpack option."""
    ts = Timestamp.from_bytes(payload)
    if timestamp == 0:
        return ts
    if timestamp == 1:
        return ts.to_unix()
    if timestamp == 2:
        return ts.to_unix_nano()
    if timestamp == 3:
        return ts.to_datetime()
    return ts
