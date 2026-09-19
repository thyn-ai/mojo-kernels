"""Public API: get_next / get_prev for 5-field cron expressions.

Results are datetime-equal to the `croniter` PyPI package (the oracle)
for the supported scope:

* fields: ``*``, ``*/n``, ``a-b`` (wrapping allowed), lists, steps ``a/n``
  and ``a-b/n``, month/weekday names (case-insensitive), ``?`` in the
  day-of-month / day-of-week fields, ``l`` (last day of month) in dom
  lists, and nth-weekday hash specs (``5#3``) in dow.
* dom/dow combine with OR by default (``day_or=True``) or AND, including
  the oracle's hash-spec semantics.
* naive datetimes in -> naive datetimes out; aware datetimes (zoneinfo)
  in -> aware datetimes out, with the oracle's DST behavior at
  spring-forward gaps and fall-back folds.
* ``start`` is exclusive at minute precision (seconds/microseconds are
  truncated), like the oracle's ``get_next``/``get_prev``.
* ``expand_from_start_time`` is accepted for API compatibility; with
  croniter 6.2.4 it has no observable effect on 5-field
  ``get_next``/``get_prev``, and this package matches that behavior.

The schedule-matching hot loop runs on the native Mojo kernel when its
shared library is available (macOS arm64 / Linux x86_64 wheels) and
transparently falls back to a pure-Python engine otherwise. Parsing,
validation, and all timezone/DST handling are shared by both backends,
so results are identical either way; the differential suite asserts
agreement with the oracle on both paths.
"""

from __future__ import annotations

import threading
from datetime import datetime

from . import _engine
from ._engine import FallbackEngine
from ._errors import (
    CroniterBadCronError,
    CroniterBadDateError,
    CroniterMojoError,
    CroniterNotAlphaError,
    CroniterUnsupportedSyntaxError,
)
from ._fields import Schedule, parse
from ._native import NativeEngine, NativeUnavailable, backend_info, native_available

__all__ = [
    "get_next",
    "get_prev",
    "native_available",
    "backend_info",
    "CroniterMojoError",
    "CroniterBadCronError",
    "CroniterNotAlphaError",
    "CroniterBadDateError",
    "CroniterUnsupportedSyntaxError",
]

_PARSE_CACHE_MAX = 1024
_cache_lock = threading.Lock()
_cache: dict[tuple[str, bool], "_Prepared"] = {}


class _Prepared:
    """Parsed schedule plus its seek engine (native when available)."""

    __slots__ = ("schedule", "engine", "backend")

    def __init__(self, schedule: Schedule) -> None:
        self.schedule = schedule
        try:
            self.engine = NativeEngine(schedule)
            self.backend = "native"
        except NativeUnavailable:
            self.engine = FallbackEngine(schedule)
            self.backend = "fallback"


def _prepare(expr: str, day_or: bool) -> _Prepared:
    key = (expr, day_or)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit
    prepared = _Prepared(parse(expr, day_or))
    with _cache_lock:
        if len(_cache) >= _PARSE_CACHE_MAX:
            # Deterministic simple eviction: drop the oldest inserted entry.
            _cache.pop(next(iter(_cache)))
        _cache[key] = prepared
    return prepared


def _validate_start(start) -> datetime:
    if not isinstance(start, datetime):
        raise TypeError(
            f"start must be a datetime.datetime, got {type(start).__name__}"
        )
    return start


def get_next(expr: str, start: datetime, *, day_or: bool = True, expand_from_start_time: bool = False) -> datetime:
    """Next datetime matching a 5-field cron expression, strictly after ``start``.

    Naive start -> naive result; aware (zoneinfo) start -> aware result in
    the same timezone. Raises CroniterBadCronError on malformed expressions
    and CroniterBadDateError when no match exists within 50 years.
    """
    del expand_from_start_time  # no observable effect on 5-field get_next (see module docstring)
    prepared = _prepare(expr, day_or)
    if prepared.schedule.gate_fails:
        raise CroniterBadDateError("failed to find next date")
    return _engine.find_next(prepared.engine, prepared.schedule, _validate_start(start))


def get_prev(expr: str, start: datetime, *, day_or: bool = True, expand_from_start_time: bool = False) -> datetime:
    """Last datetime matching a 5-field cron expression, strictly before ``start``.

    Same semantics and errors as :func:`get_next`, mirrored in time.
    """
    del expand_from_start_time  # no observable effect on 5-field get_prev (see module docstring)
    prepared = _prepare(expr, day_or)
    if prepared.schedule.gate_fails:
        raise CroniterBadDateError("failed to find next date")
    return _engine.find_prev(prepared.engine, prepared.schedule, _validate_start(start))
