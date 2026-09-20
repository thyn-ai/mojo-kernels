"""Pure-Python fallback engine + the shared datetime/DST driver.

Two layers:

1. :class:`FallbackEngine` — a pure-Python mirror of the Mojo kernel's
   wall-clock seek (same algorithm, same semantics, integer bitmasks).
   Used when the native library is unavailable.

2. :func:`find_next` / :func:`find_prev` — the timezone-aware driver shared
   by BOTH backends. It drives the engine over wall-clock candidates and
   resolves them against a ``zoneinfo`` timezone. The scan is anchored at
   the wall time of the start INSTANT (for an imaginary start — a wall
   time that never occurred — that is the real wall clock across the gap),
   and candidates are filtered by wall order against that anchor:

   - unambiguous wall time  -> one candidate instant
   - ambiguous wall time    -> two instants (fall-back folds); a whole
     contiguous ambiguous region is enumerated fold=0 pass then fold=1
     pass for next (reverse for prev). When the anchor sits inside the
     region, the anchor's own fold pass comes first (walls beyond the
     anchor), then the other pass (next) / only the anchor's pass (prev
     with fold=0) / both passes (prev with fold=1) — i.e. fold-major
     ordering, matching the oracle.
   - imaginary wall time    -> depends on the hour field (oracle-verified):
     * hour value set restricted (!= full 0..23): an imaginary schedule
       match maps to the gap-end instant (the first real instant after a
       spring-forward gap), even though it may not match the schedule
     * hour value set full ("*", "*/1", "0-23", ...): an imaginary match W
       maps to W+gap for next (W-gap for prev) when that shifted wall time
       itself matches the schedule, otherwise to the gap-end instant

   The search horizon is +/-50 years from the start year; exhausting it
   raises :class:`CroniterBadDateError`, like the oracle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ._errors import CroniterBadDateError
from ._fields import Schedule

_ONE_MINUTE = timedelta(minutes=1)
_ONE_HOUR = timedelta(hours=1)
_YEAR_SPAN = 50  # oracle search horizon: +/-50 years around the start year

_REAL, _AMBIGUOUS, _IMAGINARY = range(3)


def _instant(dt: datetime) -> datetime:
    """Absolute instant of an aware datetime as a naive UTC datetime.

    utcoffset() subtraction is exact integer math (no float timestamps),
    and it works for imaginary and ambiguous wall times alike.
    """
    return dt.replace(tzinfo=None) - dt.utcoffset()


def _wall_of_instant(start: datetime, tz) -> datetime:
    """Wall time in `tz` of the start's absolute instant, minute-truncated.

    Equals the truncated start wall for real/ambiguous starts; for an
    imaginary start it is the real wall clock of the same instant (across
    the gap). Computed with exact aware-datetime arithmetic (no floats).
    """
    wall = start.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)
    return wall.replace(second=0, microsecond=0)


def _is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def _dim(y: int, m: int) -> int:
    if m == 2:
        return 29 if _is_leap(y) else 28
    if m in (4, 6, 9, 11):
        return 30
    return 31


def _cron_dow(y: int, m: int, d: int) -> int:
    """Cron day-of-week (Sunday == 0); Howard Hinnant's days_from_civil."""
    yy = y - (1 if m <= 2 else 0)
    era = yy // 400  # Python // is floor division
    yoe = yy - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return (days + 4) % 7


def _day_match(s: Schedule, y: int, m: int, d: int) -> bool:
    dom_side = bool((s.doms >> d) & 1) or (s.dom_has_l and d == _dim(y, m))
    dow = _cron_dow(y, m, d)
    if s.dow_has_hash:
        nth = (d - 1) // 7 + 1
        dow_side = any(hd == dow and hn == nth for hd, hn in s.hash_specs)
    else:
        dow_side = bool((s.dows >> dow) & 1)
    if not s.dom_restricted and not s.dow_restricted:
        return True
    if not s.dow_restricted:
        return dom_side
    if not s.dom_restricted:
        return dow_side
    if s.day_or:
        # Hash + restricted dom + OR: hash rule alone decides (the parser
        # ran the static feasibility gate for this combination).
        if s.dow_has_hash:
            return dow_side
        return dom_side or dow_side
    return dom_side and dow_side


class FallbackEngine:
    """Pure-Python wall-clock seek over a parsed schedule. Mirror of the kernel."""

    def __init__(self, schedule: Schedule) -> None:
        self._s = schedule

    def seek_next(
        self, y: int, mo: int, d: int, h: int, mi: int, year_bound: int
    ) -> tuple[int, int, int, int, int] | None:
        s = self._s
        while y <= year_bound:
            if not (s.months >> mo) & 1:
                mo += 1
                if mo > 12:
                    mo = 1
                    y += 1
                d, h, mi = 1, 0, 0
                continue
            if d > _dim(y, mo):
                mo += 1
                if mo > 12:
                    mo = 1
                    y += 1
                d, h, mi = 1, 0, 0
                continue
            if not _day_match(s, y, mo, d):
                d += 1
                h, mi = 0, 0
                continue
            if h > 23:
                d += 1
                h, mi = 0, 0
                continue
            hh = -1
            for cand in range(h, 24):
                if (s.hours >> cand) & 1:
                    hh = cand
                    break
            if hh < 0:
                d += 1
                h, mi = 0, 0
                continue
            if hh > h:
                h = hh
                mi = 0
            mm = -1
            for cand in range(mi, 60):
                if (s.minutes >> cand) & 1:
                    mm = cand
                    break
            if mm < 0:
                h += 1
                mi = 0
                continue
            return (y, mo, d, h, mm)
        return None

    def seek_prev(
        self, y: int, mo: int, d: int, h: int, mi: int, year_bound: int
    ) -> tuple[int, int, int, int, int] | None:
        s = self._s
        while y >= year_bound:
            if mo < 1:
                mo = 12
                y -= 1
                d, h, mi = 31, 23, 59
                continue
            if not (s.months >> mo) & 1:
                mo -= 1
                d, h, mi = 31, 23, 59
                continue
            if d > _dim(y, mo):
                d = _dim(y, mo)
            if d < 1:
                mo -= 1
                d, h, mi = 31, 23, 59
                continue
            if not _day_match(s, y, mo, d):
                d -= 1
                h, mi = 23, 59
                continue
            if h < 0:
                d -= 1
                h, mi = 23, 59
                continue
            hh = -1
            for cand in range(h, -1, -1):
                if (s.hours >> cand) & 1:
                    hh = cand
                    break
            if hh < 0:
                d -= 1
                h, mi = 23, 59
                continue
            if hh < h:
                h = hh
                mi = 59
            mm = -1
            for cand in range(mi, -1, -1):
                if (s.minutes >> cand) & 1:
                    mm = cand
                    break
            if mm < 0:
                h -= 1
                mi = 59
                continue
            return (y, mo, d, h, mm)
        return None


# --- timezone classification -------------------------------------------------


def _classify(naive: datetime, tz) -> int:
    """Wall-time kind: _REAL (one instant), _AMBIGUOUS (two), _IMAGINARY (none)."""
    off0 = naive.replace(tzinfo=tz, fold=0).utcoffset()
    off1 = naive.replace(tzinfo=tz, fold=1).utcoffset()
    if off0 == off1:
        return _REAL
    return _AMBIGUOUS if off0 > off1 else _IMAGINARY


def _region(naive: datetime, tz, kind: int) -> tuple[datetime, datetime]:
    """Maximal contiguous run of same-kind wall minutes containing `naive`."""
    a = naive
    while True:
        try:
            prev = a - _ONE_MINUTE
        except OverflowError:
            break
        if _classify(prev, tz) != kind:
            break
        a = prev
    b = naive
    while True:
        try:
            nxt = b + _ONE_MINUTE
        except OverflowError:
            break
        if _classify(nxt, tz) != kind:
            break
        b = nxt
    return a, b  # inclusive


def _matches_in_region(engine, direction: int, a: datetime, b: datetime, year_bound: int) -> list[datetime]:
    """All schedule wall matches inside [a, b], ascending for next, descending for prev."""
    out: list[datetime] = []
    if direction > 0:
        cursor = a
        while cursor <= b:
            comps = engine.seek_next(cursor.year, cursor.month, cursor.day, cursor.hour, cursor.minute, year_bound)
            if comps is None:
                break
            w = datetime(*comps)
            if w > b:
                break
            out.append(w)
            cursor = w + _ONE_MINUTE
    else:
        cursor = b
        while cursor >= a:
            comps = engine.seek_prev(cursor.year, cursor.month, cursor.day, cursor.hour, cursor.minute, year_bound)
            if comps is None:
                break
            w = datetime(*comps)
            if w < a:
                break
            out.append(w)
            cursor = w - _ONE_MINUTE
    return out


def _handle_region_next(engine, schedule: Schedule, a: datetime, b: datetime, kind: int, tz, anchor: datetime, start_fold: int, start_utc: datetime, year_bound: int):
    """First candidate produced by non-real region [a, b] after `anchor`, or None.

    Mapped candidates (the gap end) must be strictly after the start
    instant; an instant exactly equal to the start's qualifies only when
    its wall time is past the anchor (oracle-verified: this is how a real
    match right at a gap end is included for an imaginary start).
    """

    def next_ok(wall: datetime, cand: datetime) -> bool:
        inst = _instant(cand)
        return inst > start_utc or (inst == start_utc and wall > anchor)

    if kind == _IMAGINARY:
        end_wall = b + _ONE_MINUTE  # first real wall after the gap (gap end)
        ge_cand = end_wall.replace(tzinfo=tz, fold=0)
        if schedule.hour_restricted:
            # Any schedule match in [a, end_wall] maps to the gap end.
            comps = engine.seek_next(a.year, a.month, a.day, a.hour, a.minute, year_bound)
            if comps is not None and datetime(*comps) <= end_wall and next_ok(end_wall, ge_cand):
                return ge_cand
            return None
        # Full hour field: an imaginary match W maps to the gap end when the
        # anchor wall precedes W-gap (oracle-verified boundary); otherwise W
        # is skipped and the scan continues to the next real match.
        gap = end_wall - a
        cursor = a
        while cursor <= b:
            comps = engine.seek_next(cursor.year, cursor.month, cursor.day, cursor.hour, cursor.minute, year_bound)
            if comps is None:
                break
            w = datetime(*comps)
            if w > b:
                break
            if anchor < w - gap and next_ok(end_wall, ge_cand):
                return ge_cand
            cursor = w + _ONE_MINUTE
        return None
    # Ambiguous region: fold passes in absolute order (f0 then f1).
    # With an f0 anchor the cross-fold (f1) pass covers ALL region walls
    # when the fold is a full hour (or the hour field is restricted);
    # sub-hour folds (Lord Howe) only yield the f1 walls before the anchor.
    matches = _matches_in_region(engine, +1, a, b, year_bound)
    gap = (b + _ONE_MINUTE) - a
    cross_all = schedule.hour_restricted or gap >= _ONE_HOUR
    if a <= anchor <= b:
        if start_fold == 0:
            passes = [(0, [m for m in matches if m > anchor])]
            if cross_all:
                passes.append((1, matches))
            else:
                # Sub-hour fold: f1 walls strictly before the anchor, plus
                # the region-start wall when the anchor sits exactly on it.
                passes.append((1, [m for m in matches if m < anchor or (anchor == a and m == a)]))
        else:
            passes = [(1, [m for m in matches if m > anchor])]
    else:
        passes = [(0, matches)]
        if cross_all:
            passes.append((1, matches))
    for fold, ms in passes:
        for m in ms:
            return m.replace(tzinfo=tz, fold=fold)
    return None


def _handle_region_prev(engine, schedule: Schedule, a: datetime, b: datetime, kind: int, tz, anchor: datetime, start_fold: int, allow_equal: bool, start_utc: datetime, year_bound: int):
    """Last candidate produced by non-real region [a, b] before `anchor`, or None."""

    def wall_ok(w: datetime) -> bool:
        return w < anchor or (allow_equal and w == anchor)

    def prev_ok(cand: datetime) -> bool:
        return _instant(cand) < start_utc

    if kind == _IMAGINARY:
        end_wall = b + _ONE_MINUTE
        ge_cand = end_wall.replace(tzinfo=tz, fold=0)
        if schedule.hour_restricted:
            comps = engine.seek_prev(end_wall.year, end_wall.month, end_wall.day, end_wall.hour, end_wall.minute, year_bound)
            if comps is not None and datetime(*comps) >= a and wall_ok(end_wall) and prev_ok(ge_cand):
                return ge_cand
            return None
        # Full hour field: an imaginary match W (descending) maps to the gap
        # end when the anchor wall follows W+gap (oracle-verified boundary);
        # otherwise W is skipped and the scan continues below.
        gap = end_wall - a
        cursor = b
        while cursor >= a:
            comps = engine.seek_prev(cursor.year, cursor.month, cursor.day, cursor.hour, cursor.minute, year_bound)
            if comps is None:
                break
            w = datetime(*comps)
            if w < a:
                break
            if anchor > w + gap and wall_ok(end_wall) and prev_ok(ge_cand):
                return ge_cand
            cursor = w - _ONE_MINUTE
        return None
    # Ambiguous region: fold passes in absolute order (f1 then f0 for prev).
    # With an f1 anchor the cross-fold (f0) pass covers ALL region walls
    # when the fold is a full hour (or the hour field is restricted);
    # sub-hour folds only yield the f0 walls after the anchor.
    matches = _matches_in_region(engine, -1, a, b, year_bound)
    gap = (b + _ONE_MINUTE) - a
    cross_all = schedule.hour_restricted or gap >= _ONE_HOUR
    if a <= anchor <= b:
        if start_fold == 1:
            passes = [(1, [m for m in matches if wall_ok(m)])]
            if cross_all:
                passes.append((0, matches))
            elif anchor == a:
                # Sub-hour fold: the f0 pass fires only when the anchor is
                # exactly the region start; then it covers all region walls.
                passes.append((0, matches))
        else:
            passes = [(0, [m for m in matches if wall_ok(m)])]
    else:
        passes = [(1, matches)]
        if cross_all:
            passes.append((0, matches))
    for fold, ms in passes:
        for m in ms:
            return m.replace(tzinfo=tz, fold=fold)
    return None


# --- shared drivers ------------------------------------------------------------


def find_next(engine, schedule: Schedule, start: datetime) -> datetime:
    """First schedule instant strictly after `start` (minute precision)."""
    try:
        return _find_next(engine, schedule, start)
    except OverflowError:
        raise CroniterBadDateError("failed to find next date") from None


def _find_next(engine, schedule: Schedule, start: datetime) -> datetime:
    tz = start.tzinfo
    year_bound = min(start.year + _YEAR_SPAN, 9999)
    tw = start.replace(second=0, microsecond=0)
    if tz is None:
        cur = tw + _ONE_MINUTE
        comps = engine.seek_next(cur.year, cur.month, cur.day, cur.hour, cur.minute, year_bound)
        if comps is None:
            raise CroniterBadDateError("failed to find next date")
        return datetime(*comps)

    # Anchor the wall-clock scan at the wall time OF THE START INSTANT.
    # For real/ambiguous starts this equals the truncated start wall; for an
    # imaginary start it is the real wall clock of the same instant, across
    # the gap — which is where the oracle begins iterating.
    anchor = _wall_of_instant(start, tz)
    start_fold = start.fold
    start_utc = _instant(start)
    if _classify(anchor, tz) != _REAL:
        # The anchor sits in a fold: qualifying candidates may hide at wall
        # minutes at or below the anchor, so begin the scan at the anchor
        # itself and let region handling decide.
        cur = anchor
    else:
        cur = anchor + _ONE_MINUTE
    while True:
        kind_cur = _classify(cur, tz)
        if kind_cur != _REAL:
            a, b = _region(cur, tz, kind_cur)
            result = _handle_region_next(engine, schedule, a, b, kind_cur, tz, anchor, start_fold, start_utc, year_bound)
            if result is not None:
                return result
            cur = b + _ONE_MINUTE
            continue
        comps = engine.seek_next(cur.year, cur.month, cur.day, cur.hour, cur.minute, year_bound)
        if comps is None:
            raise CroniterBadDateError("failed to find next date")
        w = datetime(*comps)
        kind = _classify(w, tz)
        if kind == _REAL:
            return w.replace(tzinfo=tz, fold=0)
        a, b = _region(w, tz, kind)
        result = _handle_region_next(engine, schedule, a, b, kind, tz, anchor, start_fold, start_utc, year_bound)
        if result is not None:
            return result
        cur = b + _ONE_MINUTE


def find_prev(engine, schedule: Schedule, start: datetime) -> datetime:
    """Last schedule instant strictly before `start` (minute precision)."""
    try:
        return _find_prev(engine, schedule, start)
    except OverflowError:
        raise CroniterBadDateError("failed to find next date") from None


def _find_prev(engine, schedule: Schedule, start: datetime) -> datetime:
    tz = start.tzinfo
    year_bound = max(start.year - _YEAR_SPAN, 1)
    tw = start.replace(second=0, microsecond=0)
    if tz is None:
        cur = tw if (start.second or start.microsecond) else tw - _ONE_MINUTE
        comps = engine.seek_prev(cur.year, cur.month, cur.day, cur.hour, cur.minute, year_bound)
        if comps is None:
            raise CroniterBadDateError("failed to find next date")
        return datetime(*comps)

    anchor = _wall_of_instant(start, tz)
    start_fold = start.fold
    start_utc = _instant(start)
    # With sub-minute seconds the truncated start wall minute still
    # qualifies for prev (its instant precedes the start's).
    allow_equal = bool(start.second or start.microsecond)
    if _classify(anchor, tz) != _REAL:
        # The anchor sits in a fold: qualifying candidates may hide at wall
        # minutes at or above the anchor (the other fold pass, or the gap
        # end), so begin the scan at the anchor itself.
        cur = anchor
    elif allow_equal:
        cur = anchor
    else:
        cur = anchor - _ONE_MINUTE
    while True:
        kind_cur = _classify(cur, tz)
        if kind_cur != _REAL:
            a, b = _region(cur, tz, kind_cur)
            result = _handle_region_prev(engine, schedule, a, b, kind_cur, tz, anchor, start_fold, allow_equal, start_utc, year_bound)
            if result is not None:
                return result
            cur = a - _ONE_MINUTE
            continue
        comps = engine.seek_prev(cur.year, cur.month, cur.day, cur.hour, cur.minute, year_bound)
        if comps is None:
            raise CroniterBadDateError("failed to find next date")
        w = datetime(*comps)
        kind = _classify(w, tz)
        if kind == _REAL:
            return w.replace(tzinfo=tz, fold=1)  # cosmetic fold, like the oracle's prev
        a, b = _region(w, tz, kind)
        result = _handle_region_prev(engine, schedule, a, b, kind, tz, anchor, start_fold, allow_equal, start_utc, year_bound)
        if result is not None:
            return result
        cur = a - _ONE_MINUTE
