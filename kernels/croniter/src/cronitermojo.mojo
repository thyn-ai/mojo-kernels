"""Clean-room 5-field cron schedule-matching kernel.

Written fresh from the classic Vixie-cron field semantics (minute / hour /
day-of-month / month / day-of-week with ranges, steps, lists, names and
nth-weekday-of-month hash specs) over proleptic-Gregorian calendar math
(Howard Hinnant's days-from-civil). No third-party Mojo code is used or
adapted, and no engine source was consulted: the behavior contract is the
published `croniter` PyPI package exercised as a black-box oracle by the
differential test suite.

The kernel works purely on naive wall-clock calendar components. Timezone
and DST handling lives in the Python control plane, which drives this
kernel one wall-clock candidate at a time.

Exported C ABI (v1):

    int32_t  cronitermojo_abi_version(void)
    void*    cronitermojo_create(minutes, hours, doms, dom_has_l, months,
                                 dows, dom_restricted, dow_restricted,
                                 day_or, dow_has_hash,
                                 hash_dow, hash_nth, n_hash)
    int32_t  cronitermojo_seek(handle, y, mo, d, h, mi, direction,
                               year_bound, out_ymdhm)
    void     cronitermojo_destroy(handle)

Field sets arrive as bitmasks (bit i set == value i allowed; dom bits 1..31,
month bits 1..12, dow bits 0..6 with Sunday == 0). `cronitermojo_seek` with
direction +1 returns the first matching wall minute >= the input components
(not scanning past `year_bound`); direction -1 returns the first match <=
the input (not scanning before `year_bound`). Results are written as five
int32s (year, month, day, hour, minute) into `out_ymdhm`. Returns 0 on a
match, 1 when no match exists within the year bound, 2 on invalid input.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# Direction values for `cronitermojo_seek`.
comptime DIR_NEXT: Int32 = 1
comptime DIR_PREV: Int32 = -1

# Seek return codes.
comptime RC_MATCH: Int32 = 0
comptime RC_NO_MATCH_WITHIN_BOUND: Int32 = 1
comptime RC_INVALID: Int32 = 2


struct CronSchedule(Copyable, Movable):
    """Owned, native copy of one parsed 5-field cron schedule."""

    var minutes: UInt64  # bits 0..59
    var hours: UInt32  # bits 0..23
    var doms: UInt32  # bits 1..31
    var months: UInt32  # bits 1..12
    var dows: UInt32  # bits 0..6 (Sunday == 0)
    var dom_has_l: Int32  # dom list contains the last-day-of-month item
    var dom_restricted: Int32
    var dow_restricted: Int32
    var day_or: Int32  # 1: restricted dom/dow combine with OR; 0: AND
    var dow_has_hash: Int32  # dow field uses nth-weekday specs (d#n)
    var n_hash: Int32
    var hash_dow: I32Ptr  # [n_hash] cron dow 0..6
    var hash_nth: I32Ptr  # [n_hash] occurrence 1..5

    def __init__(
        out self,
        minutes: UInt64,
        hours: UInt32,
        doms: UInt32,
        months: UInt32,
        dows: UInt32,
        dom_has_l: Int32,
        dom_restricted: Int32,
        dow_restricted: Int32,
        day_or: Int32,
        dow_has_hash: Int32,
        n_hash: Int32,
        hash_dow: I32Ptr,
        hash_nth: I32Ptr,
    ):
        self.minutes = minutes
        self.hours = hours
        self.doms = doms
        self.months = months
        self.dows = dows
        self.dom_has_l = dom_has_l
        self.dom_restricted = dom_restricted
        self.dow_restricted = dow_restricted
        self.day_or = day_or
        self.dow_has_hash = dow_has_hash
        self.n_hash = n_hash
        self.hash_dow = hash_dow
        self.hash_nth = hash_nth


def _is_leap(y: Int32) -> Bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def _dim(y: Int32, m: Int32) -> Int32:
    """Days in month (proleptic Gregorian)."""
    if m == 2:
        return 29 if _is_leap(y) else 28
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    return 31


def _cron_dow(y: Int32, m: Int32, d: Int32) -> Int32:
    """Cron day-of-week (Sunday == 0) via Howard Hinnant's days_from_civil.

    Epoch 1970-01-01 was a Thursday (cron dow 4).
    """
    var yy = Int(y)
    if m <= 2:
        yy -= 1
    var era = yy // 400
    if yy < 0 and yy % 400 != 0:
        era -= 1  # floor division for negative years
    var yoe = yy - era * 400  # [0, 399]
    var mp = (Int(m) + 9) % 12  # Mar=0 .. Feb=11
    var doy = (153 * mp + 2) // 5 + Int(d) - 1
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    var days = era * 146097 + doe - 719468
    var dow = (days + 4) % 7
    if dow < 0:
        dow += 7
    return Int32(dow)


def _day_match(s: CronSchedule, y: Int32, m: Int32, d: Int32) -> Bool:
    var dom_side = ((s.doms >> UInt32(d)) & 1) != 0
    if s.dom_has_l != 0 and d == _dim(y, m):
        dom_side = True
    var dow = _cron_dow(y, m, d)
    var dow_side = False
    if s.dow_has_hash != 0:
        var nth = (d - 1) // 7 + 1
        for i in range(Int(s.n_hash)):
            if (
                s.hash_dow[unsafe_offset=i] == dow
                and s.hash_nth[unsafe_offset=i] == nth
            ):
                dow_side = True
    else:
        dow_side = ((s.dows >> UInt32(dow)) & 1) != 0

    if s.dom_restricted == 0 and s.dow_restricted == 0:
        return True
    if s.dow_restricted == 0:
        return dom_side
    if s.dom_restricted == 0:
        return dow_side
    if s.day_or != 0:
        # With nth-weekday hash specs and OR semantics, the restricted dom
        # side does not participate in matching (the caller's parser runs
        # the static feasibility gate for that combination instead).
        if s.dow_has_hash != 0:
            return dow_side
        return dom_side or dow_side
    return dom_side and dow_side


def _month_allowed(s: CronSchedule, m: Int32) -> Bool:
    return ((s.months >> UInt32(m)) & 1) != 0


def _next_hour(s: CronSchedule, h: Int32) -> Int32:
    """Smallest allowed hour >= h, or -1."""
    var hh = h
    while hh < 24:
        if ((s.hours >> UInt32(hh)) & 1) != 0:
            return hh
        hh += 1
    return -1


def _prev_hour(s: CronSchedule, h: Int32) -> Int32:
    """Largest allowed hour <= h, or -1."""
    var hh = h
    while hh >= 0:
        if ((s.hours >> UInt32(hh)) & 1) != 0:
            return hh
        hh -= 1
    return -1


def _next_minute(s: CronSchedule, mi: Int32) -> Int32:
    """Smallest allowed minute >= mi, or -1."""
    var mm = mi
    while mm < 60:
        if ((s.minutes >> UInt64(mm)) & 1) != 0:
            return mm
        mm += 1
    return -1


def _prev_minute(s: CronSchedule, mi: Int32) -> Int32:
    """Largest allowed minute <= mi, or -1."""
    var mm = mi
    while mm >= 0:
        if ((s.minutes >> UInt64(mm)) & 1) != 0:
            return mm
        mm -= 1
    return -1


def _seek_next(s: CronSchedule, y0: Int32, mo0: Int32, d0: Int32, h0: Int32, mi0: Int32, year_bound: Int32, res: I32Ptr) -> Int32:
    var y = y0
    var mo = mo0
    var d = d0
    var h = h0
    var mi = mi0
    while y <= year_bound:
        if not _month_allowed(s, mo):
            # Jump to the next allowed month (resetting to its first day).
            mo += 1
            if mo > 12:
                mo = 1
                y += 1
            d = 1
            h = 0
            mi = 0
            continue
        if d > _dim(y, mo):
            mo += 1
            if mo > 12:
                mo = 1
                y += 1
            d = 1
            h = 0
            mi = 0
            continue
        if not _day_match(s, y, mo, d):
            d += 1
            h = 0
            mi = 0
            continue
        if h > 23:
            d += 1
            h = 0
            mi = 0
            continue
        var hh = _next_hour(s, h)
        if hh < 0:
            d += 1
            h = 0
            mi = 0
            continue
        if hh > h:
            h = hh
            mi = 0
        var mm = _next_minute(s, mi)
        if mm < 0:
            h += 1
            mi = 0
            continue
        res[unsafe_offset=0] = y
        res[unsafe_offset=1] = mo
        res[unsafe_offset=2] = d
        res[unsafe_offset=3] = h
        res[unsafe_offset=4] = mm
        return RC_MATCH
    return RC_NO_MATCH_WITHIN_BOUND


def _seek_prev(s: CronSchedule, y0: Int32, mo0: Int32, d0: Int32, h0: Int32, mi0: Int32, year_bound: Int32, res: I32Ptr) -> Int32:
    var y = y0
    var mo = mo0
    var d = d0
    var h = h0
    var mi = mi0
    while y >= year_bound:
        if mo < 1:
            mo = 12
            y -= 1
            d = 31
            h = 23
            mi = 59
            continue
        if not _month_allowed(s, mo):
            mo -= 1
            d = 31
            h = 23
            mi = 59
            continue
        if d > _dim(y, mo):
            d = _dim(y, mo)
        if d < 1:
            mo -= 1
            d = 31
            h = 23
            mi = 59
            continue
        if not _day_match(s, y, mo, d):
            d -= 1
            h = 23
            mi = 59
            continue
        if h < 0:
            d -= 1
            h = 23
            mi = 59
            continue
        var hh = _prev_hour(s, h)
        if hh < 0:
            d -= 1
            h = 23
            mi = 59
            continue
        if hh < h:
            h = hh
            mi = 59
        var mm = _prev_minute(s, mi)
        if mm < 0:
            h -= 1
            mi = 59
            continue
        res[unsafe_offset=0] = y
        res[unsafe_offset=1] = mo
        res[unsafe_offset=2] = d
        res[unsafe_offset=3] = h
        res[unsafe_offset=4] = mm
        return RC_MATCH
    return RC_NO_MATCH_WITHIN_BOUND


@export
def cronitermojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def cronitermojo_create(
    minutes: UInt64,
    hours: UInt32,
    doms: UInt32,
    dom_has_l: Int32,
    months: UInt32,
    dows: UInt32,
    dom_restricted: Int32,
    dow_restricted: Int32,
    day_or: Int32,
    dow_has_hash: Int32,
    hash_dow: I32Ptr,
    hash_nth: I32Ptr,
    n_hash: Int32,
) abi("C") -> Handle:
    """Copy the caller's schedule into a native handle; NULL on invalid input."""
    if minutes == 0 or hours == 0 or months == 0:
        return None
    if n_hash < 0 or n_hash > 64:
        return None
    if doms == 0 and dom_has_l == 0 and dom_restricted != 0:
        return None
    if dows == 0 and n_hash == 0 and dow_restricted != 0:
        return None

    var hd = hash_dow
    var hn = hash_nth
    if n_hash > 0:
        hd = unsafe_alloc[Int32](Int(n_hash))
        hn = unsafe_alloc[Int32](Int(n_hash))
        unsafe_memcpy(dest=hd, src=hash_dow, count=Int(n_hash))
        unsafe_memcpy(dest=hn, src=hash_nth, count=Int(n_hash))

    var sched = unsafe_alloc[CronSchedule](1)
    sched[] = CronSchedule(
        minutes,
        hours,
        doms,
        months,
        dows,
        dom_has_l,
        dom_restricted,
        dow_restricted,
        day_or,
        dow_has_hash,
        n_hash,
        hd,
        hn,
    )
    return sched.unsafe_bitcast[UInt8]()


@export
def cronitermojo_seek(
    handle: Handle,
    y: Int32,
    mo: Int32,
    d: Int32,
    h: Int32,
    mi: Int32,
    direction: Int32,
    year_bound: Int32,
    out_ymdhm: I32Ptr,
) abi("C") -> Int32:
    """Find the first matching wall minute at/after (or at/before) the input.

    The input components must be a valid, normalized calendar minute
    (the caller performs the initial +1/-1 minute step). Returns 0 on a
    match (components written to `out_ymdhm`), 1 when no match exists
    within `year_bound`, 2 on invalid input.
    """
    if not handle:
        return RC_INVALID
    if mo < 1 or mo > 12 or d < 1 or d > 31 or h < 0 or h > 23 or mi < 0 or mi > 59:
        return RC_INVALID
    if y < 1 or y > 9999:
        return RC_INVALID
    var s = handle.value().unsafe_bitcast[CronSchedule]()
    if direction == DIR_NEXT:
        return _seek_next(s[], y, mo, d, h, mi, year_bound, out_ymdhm)
    if direction == DIR_PREV:
        return _seek_prev(s[], y, mo, d, h, mi, year_bound, out_ymdhm)
    return RC_INVALID


@export
def cronitermojo_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var s = handle.value().unsafe_bitcast[CronSchedule]()
    if s[].n_hash > 0:
        s[].hash_dow.unsafe_free()
        s[].hash_nth.unsafe_free()
    s.unsafe_free()
