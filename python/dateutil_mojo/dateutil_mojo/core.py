"""Public API: ``parse`` (drop-in for ``dateutil.parser.parse``) and
``parse_column`` (batch parsing for ETL workloads).

Parsing runs on the native Mojo kernel for the hot formats (ISO 8601,
RFC 2822, US/EU numeric, named-month) when its shared library is available,
and transparently falls back to the pure-Python reference parser for
everything else and on every platform without the kernel. Both backends are
differentially tested against the python-dateutil oracle; results are
identical either way.
"""

from __future__ import annotations

from datetime import datetime

from dateutil_mojo import _native, _parser
from dateutil_mojo._native import NativeUnavailable
from dateutil_mojo._parser import ParserError, UnknownTimezoneWarning
from dateutil_mojo._tz import tzlocal, tzoffset, tzutc

__all__ = [
    "ParserError",
    "UnknownTimezoneWarning",
    "parse",
    "parse_column",
    "tzlocal",
    "tzoffset",
    "tzutc",
]

_KNOWN_KWARGS = {
    "dayfirst", "yearfirst", "ignoretz", "tzinfos", "default", "fuzzy",
    "fuzzy_with_tokens",
}


def _check_kwargs(kwargs: dict) -> None:
    unknown = set(kwargs) - _KNOWN_KWARGS
    if unknown:
        raise TypeError(
            f"Unknown keyword argument(s): {', '.join(sorted(unknown))}. "
            "dateutil_mojo.parse supports: " + ", ".join(sorted(_KNOWN_KWARGS))
        )


def _native_defaults(default: datetime) -> tuple[int, int, int, int, int, int, int]:
    return (
        default.year, default.month, default.day,
        default.hour, default.minute, default.second, default.microsecond,
    )


def _native_eligible(kwargs: dict, default: datetime | None) -> bool:
    """Whether the native fast path may handle this call.

    The kernel does not model fuzzy mode, tzinfos, or tz-aware defaults;
    those go straight to the Python reference (identical results).
    """
    if kwargs.get("fuzzy") or kwargs.get("fuzzy_with_tokens"):
        return False
    if kwargs.get("tzinfos") is not None:
        return False
    if default is not None and default.tzinfo is not None:
        return False
    return True


_TZUTC = None  # lazily created singleton
_TZOFFSET_CACHE: dict[int, object] = {}


def _tz_for(tzsec: int):
    """Shared tzinfo instances (most columns repeat one offset)."""
    global _TZUTC
    if tzsec == 0:
        if _TZUTC is None:
            _TZUTC = tzutc()
        return _TZUTC
    tz = _TZOFFSET_CACHE.get(tzsec)
    if tz is None:
        tz = tzoffset(None, tzsec)
        _TZOFFSET_CACHE[tzsec] = tz
    return tz


def _row_to_datetime(
    flat: list[int],
    base: int,
    ignoretz: bool,
    timestr: str,
) -> datetime:
    """Build a datetime from a native row (flat output + row offset).

    datetime-construction errors become ParserError with the input appended
    (like the oracle); tzoffset range errors escape as raw ValueError
    (also like the oracle).
    """
    try:
        dt = datetime(
            flat[base + 1], flat[base + 2], flat[base + 3],
            flat[base + 4], flat[base + 5], flat[base + 6], flat[base + 7],
        )
    except ValueError as exc:
        raise ParserError(f"{exc}: {timestr}") from exc
    tzsec = flat[base + 8]
    if ignoretz or tzsec == _native.NAIVE_SENTINEL:
        return dt
    return dt.replace(tzinfo=_tz_for(tzsec))


def parse(timestr, parserinfo=None, **kwargs):
    """Parse a string into a datetime, drop-in for ``dateutil.parser.parse``.

    Supported keyword arguments: ``dayfirst``, ``yearfirst``, ``ignoretz``,
    ``tzinfos``, ``default``, ``fuzzy``, ``fuzzy_with_tokens``. A custom
    ``parserinfo`` is not supported (raises ``NotImplementedError``).
    """
    if parserinfo is not None:
        raise NotImplementedError(
            "custom parserinfo is not supported by dateutil_mojo "
            "(see README: unsupported scope)"
        )
    _check_kwargs(kwargs)
    if not isinstance(timestr, str):
        raise TypeError(f"timestr must be str, got {type(timestr).__name__}")

    default = kwargs.get("default")
    ignoretz = bool(kwargs.get("ignoretz", False))

    if _native_eligible(kwargs, default):
        eff_default = default if default is not None else datetime.now()
        try:
            rows = _native.parse_batch_native(
                [timestr],
                dayfirst=bool(kwargs.get("dayfirst", False)),
                yearfirst=bool(kwargs.get("yearfirst", False)),
                defaults=_native_defaults(eff_default),
                pivot_base=datetime.now().year + 50,
            )
        except NativeUnavailable:
            rows = None
        if rows is not None and rows[0] == 0:
            return _row_to_datetime(rows, 0, ignoretz, timestr)
        # status != 0: fall through to the reference parser

    return _parser.parse(timestr, None, **kwargs)


def parse_column(strings, parserinfo=None, **kwargs) -> list[datetime]:
    """Parse a column of date strings (ETL batch API).

    Same semantics and keyword arguments as :func:`parse` per element; the
    batch runs through the native kernel in one FFI call when possible and
    falls back to the Python reference per element for the rest.
    """
    if parserinfo is not None:
        raise NotImplementedError(
            "custom parserinfo is not supported by dateutil_mojo "
            "(see README: unsupported scope)"
        )
    _check_kwargs(kwargs)
    strings = list(strings)
    for s in strings:
        if not isinstance(s, str):
            raise TypeError(f"all elements must be str, got {type(s).__name__}")

    default = kwargs.get("default")
    ignoretz = bool(kwargs.get("ignoretz", False))

    if _native_eligible(kwargs, default):
        eff_default = default if default is not None else datetime.now()
        try:
            rows = _native.parse_batch_native(
                strings,
                dayfirst=bool(kwargs.get("dayfirst", False)),
                yearfirst=bool(kwargs.get("yearfirst", False)),
                defaults=_native_defaults(eff_default),
                pivot_base=datetime.now().year + 50,
            )
        except NativeUnavailable:
            rows = None
        if rows is not None:
            out: list[datetime] = []
            w = 9
            for idx, s in enumerate(strings):
                base = idx * w
                if rows[base] == 0:
                    out.append(_row_to_datetime(rows, base, ignoretz, s))
                else:
                    out.append(_parser.parse(s, None, **kwargs))
            return out

    return [_parser.parse(s, None, **kwargs) for s in strings]
