"""MessagePack extension types, clean-room from the public spec.

``ExtType`` is the generic extension payload carrier; ``Timestamp`` is the
standard timestamp extension (type code -1) with the three wire forms from
the MessagePack timestamp extension specification:

* 4-byte payload:  uint32 seconds (nanoseconds implied 0)
* 8-byte payload:  uint64 ``nanoseconds << 34 | seconds`` (34-bit seconds)
* 12-byte payload: uint32 nanoseconds + int64 seconds

Behavior (validation messages, repr, epoch conversions) was probed black-box
against the published ``msgpack`` package; no reference sources were read.
"""

from __future__ import annotations

import datetime as _dt
from collections import namedtuple
from struct import Struct

_U32BE = Struct(">L")
_U64BE = Struct(">Q")
_TS12 = Struct(">Lq")

_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_NANO_MAX = 999_999_999


class ExtType(namedtuple("ExtType", "code data")):
    """A (code, payload) extension value; code must be in 0..127."""

    __slots__ = ()

    def __new__(cls, code: int, data: bytes):
        if not isinstance(code, int):
            raise TypeError("code must be int")
        if not 0 <= code <= 127:
            raise ValueError("code must be 0~127")
        return super().__new__(cls, code, data)


class Timestamp:
    """The timestamp extension value: whole seconds plus nanoseconds."""

    __slots__ = ("seconds", "nanoseconds")

    def __init__(self, seconds: int, nanoseconds: int = 0) -> None:
        if not 0 <= nanoseconds <= _NANO_MAX:
            raise ValueError("nanoseconds must be 0-999999999")
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    def __repr__(self) -> str:
        return f"Timestamp(seconds={self.seconds}, nanoseconds={self.nanoseconds})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds == other.seconds and self.nanoseconds == other.nanoseconds

    def __hash__(self) -> int:
        return hash((self.seconds, self.nanoseconds))

    def to_bytes(self) -> bytes:
        """Smallest timestamp wire payload (4 / 8 / 12 bytes, spec rules)."""
        if self.nanoseconds == 0 and 0 <= self.seconds < (1 << 32):
            return _U32BE.pack(self.seconds)
        if 0 <= self.seconds < (1 << 34):
            return _U64BE.pack((self.nanoseconds << 34) | self.seconds)
        return _TS12.pack(self.nanoseconds, self.seconds)

    @staticmethod
    def from_bytes(b: bytes) -> "Timestamp":
        if len(b) == 4:
            return Timestamp(seconds=_U32BE.unpack(b)[0])
        if len(b) == 8:
            v = _U64BE.unpack(b)[0]
            return Timestamp(seconds=v & 0x3FFFFFFFF, nanoseconds=v >> 34)
        if len(b) == 12:
            nsec, sec = _TS12.unpack(b)
            return Timestamp(seconds=sec, nanoseconds=nsec)
        raise ValueError("timestamp type requires 4, 8, or 12 bytes of data")

    def to_unix(self) -> float:
        return self.seconds + self.nanoseconds * 1e-9

    def to_unix_nano(self) -> int:
        return self.seconds * 1_000_000_000 + self.nanoseconds

    def to_datetime(self) -> _dt.datetime:
        # Nanoseconds truncate to whole microseconds (reference behavior).
        return _EPOCH + _dt.timedelta(
            seconds=self.seconds, microseconds=self.nanoseconds // 1000
        )

    @staticmethod
    def from_datetime(dt: _dt.datetime) -> "Timestamp":
        """Convert a datetime; naive datetimes are treated as UTC (matching
        the reference package's observable behavior)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        delta = dt - _EPOCH
        return Timestamp(
            seconds=delta.days * 86400 + delta.seconds,
            nanoseconds=delta.microseconds * 1000,
        )
