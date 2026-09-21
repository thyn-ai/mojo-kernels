"""Minimal tzinfo classes mirroring the behavior of dateutil.tz types.

`dateutil_mojo.parse` returns datetimes carrying these classes instead of
`dateutil.tz.tzutc` / `tzoffset` / `tzlocal`, so the package has no runtime
dependency on python-dateutil. They implement the same `utcoffset()` /
`tzname()` / `dst()` contracts, so results compare equal to oracle results
by instant, offset, and tzname.

Clean-room: written from the public datetime.tzinfo contract and black-box
observation of the oracle's returned offsets/names.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, tzinfo

ZERO = timedelta(0)
_HOUR = timedelta(hours=1)

# Message reproduced from black-box observation of the oracle's tzoffset().
_OFFSET_RANGE_MSG = (
    "offset must be a timedelta strictly between -timedelta(hours=24) "
    "and timedelta(hours=24)."
)


class tzutc(tzinfo):
    """UTC. Equivalent to ``dateutil.tz.tzutc()`` (and ``timezone.utc``)."""

    def utcoffset(self, dt: datetime | None) -> timedelta:  # noqa: ARG002
        return ZERO

    def dst(self, dt: datetime | None) -> timedelta:  # noqa: ARG002
        return ZERO

    def tzname(self, dt: datetime | None) -> str:  # noqa: ARG002
        return "UTC"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tzutc, tzoffset)):
            return other._offset() == ZERO
        return NotImplemented

    def __hash__(self) -> int:
        return hash("dateutil_mojo.tzutc")

    def __repr__(self) -> str:
        return "tzutc()"


class tzoffset(tzinfo):
    """Fixed UTC offset with an optional display name.

    Mirrors ``dateutil.tz.tzoffset``: offset 0 compares equal to ``tzutc``.
    Any integer offset is accepted (offsets beyond +/-24h are only rejected
    later by ``datetime.isoformat``/``timezone``, like the oracle).
    """

    __slots__ = ("_name", "_offset_td")

    def __init__(self, name: str | None, offset_seconds: int | float) -> None:
        if not isinstance(offset_seconds, (int, float)):
            raise TypeError(
                "Offset must be tzinfo subclass, tz string, or int offset."
            )
        self._name = name
        self._offset_td = timedelta(seconds=offset_seconds)

    def _offset(self) -> timedelta:
        return self._offset_td

    def utcoffset(self, dt: datetime | None) -> timedelta:  # noqa: ARG002
        return self._offset_td

    def dst(self, dt: datetime | None) -> timedelta:  # noqa: ARG002
        return ZERO

    def tzname(self, dt: datetime | None) -> str | None:  # noqa: ARG002
        return self._name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tzoffset):
            return self._offset_td == other._offset_td and self._name == other._name
        if isinstance(other, tzutc):
            return self._offset_td == ZERO
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._name, self._offset_td))

    def __repr__(self) -> str:
        return f"tzoffset({self._name!r}, {int(self._offset_td.total_seconds())})"


class tzlocal(tzinfo):
    """Local system timezone, mirroring ``dateutil.tz.tzlocal``.

    Offsets/names come from the C library via ``time.mktime`` /
    ``time.localtime``, so results track the machine timezone exactly like
    the oracle's ``tzlocal``.
    """

    def _isdst(self, dt: datetime) -> bool:
        # tm_isdst=-1 lets mktime guess; localtime then reports the C
        # library's decision for that wall time. For ambiguous wall times
        # (fall-back) the fold picks the side, like the oracle's tzlocal.
        base = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0, 0)
        try:
            if _time.daylight:
                std = _time.localtime(_time.mktime(base + (0,))).tm_isdst
                dst = _time.localtime(_time.mktime(base + (1,))).tm_isdst
                if std == 0 and dst == 1:
                    return not bool(dt.fold)
            stamp = _time.mktime(base + (-1,))
        except (OverflowError, OSError, ValueError):
            return False
        return _time.localtime(stamp).tm_isdst > 0

    def utcoffset(self, dt: datetime | None) -> timedelta:
        if dt is None:
            return -timedelta(seconds=_time.timezone)
        if _time.daylight and self._isdst(dt):
            return -timedelta(seconds=_time.altzone)
        return -timedelta(seconds=_time.timezone)

    def dst(self, dt: datetime | None) -> timedelta:
        if dt is None or not _time.daylight:
            return ZERO
        if self._isdst(dt):
            return timedelta(seconds=_time.timezone - _time.altzone)
        return ZERO

    def tzname(self, dt: datetime | None) -> str:
        if dt is not None and _time.daylight and self._isdst(dt):
            return _time.tzname[1]
        return _time.tzname[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, tzlocal)

    def __hash__(self) -> int:
        return hash("dateutil_mojo.tzlocal")

    def __repr__(self) -> str:
        return "tzlocal()"


def local_tznames() -> tuple[str, ...]:
    """The local standard/DST abbreviations (e.g. ('EST', 'EDT')).

    The oracle recognizes these as tz names and maps them to its tzlocal;
    this is the same runtime source, so behavior matches on any machine.
    """
    return tuple(_time.tzname)
