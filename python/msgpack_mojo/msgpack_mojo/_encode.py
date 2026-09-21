"""Object-graph walker: Python values -> msgpackmojo instruction stream.

Shared by both pack backends (native Mojo engine and the vendored
pure-Python engine), so the two can never disagree about graph traversal,
type dispatch, or error behavior. The stream layout is documented in
``kernels/msgpack/src/msgpackmojo.mojo``; every value becomes one opcode
record, containers are prefix-counted (no close markers).

Dispatch mirrors the reference packer's observable behavior:

* exact types (``type(obj) is ...``) take the fast path; subclasses are
  packed as their base type unless ``strict_types=True``, in which case they
  are treated as unsupported (``default`` hook or ``TypeError``);
* ints are range-checked to [-(2**63), 2**64) and tagged signed/unsigned —
  out of range raises ``OverflowError``;
* ``str`` is UTF-8 encoded here (``unicode_errors`` applies); surrogates and
  header choice are the engines' job;
* ``msgpack_mojo.Timestamp`` (and ``datetime.datetime`` when
  ``datetime=True``) become the timestamp extension (code -1);
* anything else raises ``TypeError("can not serialize '<type>' object")``
  unless a ``default`` hook converts it.
"""

from __future__ import annotations

import datetime as _dt
from struct import Struct

from msgpack_mojo.ext import ExtType, Timestamp

_S64 = Struct("<q").pack
_U64 = Struct("<Q").pack
_F64 = Struct("<d").pack
_U32 = Struct("<I").pack

_OP_NIL = b"\x00"
_OP_FALSE = b"\x01"
_OP_TRUE = b"\x02"
_OP_S64 = b"\x03"
_OP_U64 = b"\x04"
_OP_F64 = b"\x05"
_OP_STR = b"\x06"
_OP_BIN = b"\x07"
_OP_ARRAY = b"\x08"
_OP_MAP = b"\x09"
_OP_EXT = b"\x0a"

_INT64_MIN = -(1 << 63)
_UINT64_MAX = (1 << 64) - 1


def encode_stream(
    obj,
    *,
    default=None,
    strict_types: bool = False,
    unicode_errors: str | None = None,
    use_datetime: bool = False,
) -> bytes:
    """Serialize the object graph into one instruction stream."""
    out: list[bytes] = []
    _walk(obj, out, out.append, default, strict_types, unicode_errors, use_datetime)
    return b"".join(out)


def _walk(obj, out, append, default, strict_types, unicode_errors, use_datetime) -> None:
    t = type(obj)
    if t is str:
        payload = obj.encode("utf-8", unicode_errors or "strict")
        append(_OP_STR + _U32(len(payload)))
        append(payload)
    elif t is int:
        if obj >= 0:
            if obj <= _UINT64_MAX:
                append(_OP_U64 + _U64(obj))
                return
            raise OverflowError("Integer value out of range")
        if obj >= _INT64_MIN:
            append(_OP_S64 + _S64(obj))
            return
        raise OverflowError("Integer value out of range")
    elif t is dict:
        append(_OP_MAP + _U32(len(obj)))
        for k, v in obj.items():
            _walk(k, out, append, default, strict_types, unicode_errors, use_datetime)
            _walk(v, out, append, default, strict_types, unicode_errors, use_datetime)
    elif t is list or t is tuple:
        append(_OP_ARRAY + _U32(len(obj)))
        for item in obj:
            _walk(item, out, append, default, strict_types, unicode_errors, use_datetime)
    elif t is float:
        append(_OP_F64 + _F64(obj))
    elif t is bytes or t is bytearray:
        payload = bytes(obj)
        append(_OP_BIN + _U32(len(payload)))
        append(payload)
    elif t is bool:
        append(_OP_TRUE if obj else _OP_FALSE)
    elif t is type(None):
        append(_OP_NIL)
    else:
        _walk_slow(obj, out, append, default, strict_types, unicode_errors, use_datetime)


def _walk_slow(obj, out, append, default, strict_types, unicode_errors, use_datetime) -> None:
    """Subclasses, extension values, datetimes, and unsupported types."""
    if isinstance(obj, ExtType):
        _walk_ext(obj[0], obj[1], append)
        return
    if isinstance(obj, Timestamp):
        _walk_ext(-1, obj.to_bytes(), append)
        return
    if use_datetime and isinstance(obj, _dt.datetime):
        if obj.tzinfo is None:
            raise ValueError(
                "can not serialize 'datetime.datetime' object where tzinfo=None"
            )
        _walk_ext(-1, Timestamp.from_datetime(obj).to_bytes(), append)
        return
    if not strict_types:
        # Subclass instances pack as their base type (reference behavior).
        if isinstance(obj, str):
            payload = obj.encode("utf-8", unicode_errors or "strict")
            append(_OP_STR + _U32(len(payload)))
            append(payload)
            return
        if isinstance(obj, (bytes, bytearray, memoryview)):
            payload = bytes(obj)
            append(_OP_BIN + _U32(len(payload)))
            append(payload)
            return
        if isinstance(obj, bool):  # before int: bool subclasses int
            append(_OP_TRUE if obj else _OP_FALSE)
            return
        if isinstance(obj, int):
            if 0 <= obj <= _UINT64_MAX:
                append(_OP_U64 + _U64(obj))
                return
            if _INT64_MIN <= obj < 0:
                append(_OP_S64 + _S64(obj))
                return
            raise OverflowError("Integer value out of range")
        if isinstance(obj, float):
            append(_OP_F64 + _F64(obj))
            return
        if isinstance(obj, (list, tuple)):
            append(_OP_ARRAY + _U32(len(obj)))
            for item in obj:
                _walk(item, out, append, default, strict_types, unicode_errors, use_datetime)
            return
        if isinstance(obj, dict):
            append(_OP_MAP + _U32(len(obj)))
            for k, v in obj.items():
                _walk(k, out, append, default, strict_types, unicode_errors, use_datetime)
                _walk(v, out, append, default, strict_types, unicode_errors, use_datetime)
            return
    if default is not None:
        _walk(
            default(obj), out, append, default, strict_types, unicode_errors, use_datetime
        )
        return
    raise TypeError(f"can not serialize '{type(obj).__name__}' object")


def _walk_ext(code: int, data, append) -> None:
    payload = bytes(data)
    append(_OP_EXT + (code & 0xFF).to_bytes(1, "little") + _U32(len(payload)))
    append(payload)
