"""Differential tests: croniter_mojo must match the croniter oracle datetime-for-datetime.

Run twice by `scripts/test_all_croniter.sh`: once against the native Mojo
kernel and once with CRONITER_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle everywhere — results
are exact datetime comparisons (==), plus tzinfo/fold/utcoffset checks for
aware results.

The oracle is the published PyPI package, pinned to croniter==6.2.4 in
pixi.toml [pypi-dependencies].
croniter was used strictly as a black-box oracle: no oracle source was
read or adapted.

Scope under test (5-field crons): *, */n, a-b (incl. wrapping), lists,
steps a/n and a-b/n, names (jan/mon), ?, dom "l", dom/dow OR-vs-AND
(day_or), nth-weekday hash specs (incl. the static dom gate), start_date
truncation, expand_from_start_time (a no-op in oracle 6.2.4), and DST
boundaries with zoneinfo (spring-forward gaps + fall-back folds).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest
from croniter import croniter
from zoneinfo import ZoneInfo

import croniter_mojo
from croniter_mojo import CroniterBadDateError, CroniterMojoError

ORACLE_ERRORS = (Exception,)  # oracle raises croniter.Croniter* subclasses of Exception

NAIVE_STARTS = [
    datetime(2026, 9, 19, 14, 30, 45, 123456),  # Saturday, sub-minute noise
    datetime(2026, 9, 19, 14, 30, 0),  # on a minute boundary
    datetime(2024, 2, 28, 23, 59, 30),  # into a leap day
    datetime(2025, 2, 28, 23, 59, 30),  # non-leap February end
    datetime(2026, 1, 1, 0, 0, 0),  # year boundary
    datetime(2026, 12, 31, 23, 59, 0),
    datetime(1999, 12, 31, 23, 59, 0),  # pre-epoch-adjacent
    datetime(1970, 1, 1, 0, 0, 0),  # epoch
    datetime(2096, 3, 1, 0, 0, 0),  # 2100 is not a leap year
]

# Expressions spanning the supported scope; each tuple is (expr, also_try_day_or_false).
EXPR_CORPUS = [
    "* * * * *",
    "*/5 * * * *",
    "5/10 * * * *",
    "0/15 * * * *",
    "0 9 * * *",
    "30 14 10-20 * *",
    "0 9 1 * *",
    "0 9 15 * *",
    "0 9 * * mon",
    "0 9 * * mon,fri",
    "0 9 * * 0",
    "0 9 * * 7",
    "0 9 * * sun",
    "0 9 * jan *",
    "0 9 * JAN *",
    "0 9 ? * ?",
    "0 0 10-20/3 * *",
    "0 0 29 2 *",
    "0 0 29-3 * *",
    "0 0 1 1 sun",
    "0 0 13 * 5#1",  # static gate: raises
    "0 0 5 * 5#1",  # static gate passes; dom ignored for matching
    "0 0 * * 5#3",
    "0 0 * * 5#5",
    "0 0 * * mon#2",
    "0 0 * * 5#1,5#3",
    "0 0 * * mon#1,fri#2",
    "0 0 * * 0#2",
    "0 0 * * 7#2",
    "0 9-17/2 * * *",
    "50-10 * * * *",
    "50-10/5 * * * *",
    "0 22-2 * * *",
    "0 22-2/2 * * *",
    "0 0 * nov-feb *",
    "0 0 * * fri-mon",
    "0 0 * * 6-1",
    "0 0 * * 5-7",
    "0 0 * * mon-fri",
    "0 0 * * mon-fri/2",
    "0 0 * * sun-thu",
    "0 0 5/15 * *",
    "0 0 1 */3 *",
    "0 0 1 2/3 *",
    "0 9/6 * * *",
    "0 0 1 jan-mar *",
    "0 0 L * *",
    "0 0 l * *",
    "0 0 1,L * *",
    "0 0 15,l * *",
    "0 0 L * 5#1",  # gate: l does not intersect [1..7] -> raises
    "0 0 L * 5#4",  # gate: l intersects [22..28] via day 28 -> matches
    "0 0 28,L * 5#4",
    "0 0 10 * 5#1,5#3",  # gate fails -> raises
    "0 0 3 * 5#1,5#3",
    "0 0 17 * 5#1,5#3",
    "0 0 1-7 * 5#1",
    "0 0 */2 * 5#1",
    "0 0 5,20 * 5#1",
    "0 0 30 * 5#5",
    "0 0 5 * 5#1",
    "15,45 3,18 * * *",
    "0 0 31 * *",
    "0 0 31 4 *",  # April has no 31st
    "0 0 31 * mon",  # 31st OR Monday -> always matches
    "0 0 31 * sat",
]

TZS = [
    "America/New_York",  # 1h spring/fall
    "Europe/Berlin",  # transitions on different dates
    "Australia/Sydney",  # southern hemisphere
    "Australia/Lord_Howe",  # 30-minute DST shifts
    "Pacific/Auckland",  # UTC+12/+13
    "Asia/Tehran",  # +3:30 base offset (no recent DST)
    "America/Santiago",  # southern hemisphere, midnight-ish transitions
    "UTC",
]

# 2026 transition instants per tz (spring forward, fall back), for sweeps.
TRANSITION_WINDOWS = {
    "America/New_York": [(datetime(2026, 3, 8, 0, 0), datetime(2026, 3, 8, 5, 0)),
                         (datetime(2026, 10, 31, 22, 0), datetime(2026, 11, 1, 5, 0))],
    "Europe/Berlin": [(datetime(2026, 3, 29, 0, 0), datetime(2026, 3, 29, 5, 0)),
                      (datetime(2026, 10, 25, 0, 0), datetime(2026, 10, 25, 5, 0))],
    "Australia/Sydney": [(datetime(2026, 10, 4, 0, 0), datetime(2026, 10, 4, 5, 0)),
                         (datetime(2026, 4, 5, 0, 0), datetime(2026, 4, 5, 5, 0))],
    "Australia/Lord_Howe": [(datetime(2026, 10, 4, 0, 0), datetime(2026, 10, 4, 5, 0)),
                            (datetime(2026, 4, 5, 0, 0), datetime(2026, 4, 5, 5, 0))],
    "Pacific/Auckland": [(datetime(2026, 9, 27, 0, 0), datetime(2026, 9, 27, 5, 0)),
                         (datetime(2026, 4, 5, 0, 0), datetime(2026, 4, 5, 5, 0))],
    "Asia/Tehran": [(datetime(2026, 3, 21, 0, 0), datetime(2026, 3, 22, 5, 0))],
    "America/Santiago": [(datetime(2026, 9, 6, 20, 0), datetime(2026, 9, 7, 3, 0)),
                         (datetime(2026, 4, 4, 20, 0), datetime(2026, 4, 5, 3, 0))],
    "UTC": [(datetime(2026, 3, 8, 0, 0), datetime(2026, 3, 8, 5, 0))],
}

DST_EXPRS = [
    "* * * * *",
    "*/15 * * * *",
    "*/30 * * * *",
    "7/20 * * * *",
    "0 * * * *",
    "0,30 1,2 * * *",
    "45 1 * * *",
    "15 1 * * *",
    "0 2 * * *",
    "30 2 * * *",
    "0 3 * * *",
    "0 0 * * *",
]


def _oracle_chain(expr, start, direction, count, day_or):
    c = croniter(expr, start, day_or=day_or)
    out = []
    for _ in range(count):
        out.append(c.get_next(datetime) if direction > 0 else c.get_prev(datetime))
    return out


def _ours_chain(expr, start, direction, count, day_or):
    out = []
    cur = start
    for _ in range(count):
        cur = (
            croniter_mojo.get_next(expr, cur, day_or=day_or)
            if direction > 0
            else croniter_mojo.get_prev(expr, cur, day_or=day_or)
        )
        out.append(cur)
    return out


def _assert_aware_identical(oracle_dt, ours_dt, ctx):
    assert ours_dt == oracle_dt, f"{ctx}: {ours_dt!r} != oracle {oracle_dt!r}"
    if oracle_dt.tzinfo is not None:
        assert ours_dt.utcoffset() == oracle_dt.utcoffset(), ctx
        assert ours_dt.fold == oracle_dt.fold, f"{ctx}: fold {ours_dt.fold} != {oracle_dt.fold}"
        assert ours_dt.tzinfo is oracle_dt.tzinfo, ctx
    else:
        assert ours_dt.tzinfo is None, ctx


def _check_chain(expr, start, direction, count=5, day_or=True):
    ctx = f"expr={expr!r} start={start!r} dir={direction} day_or={day_or}"
    try:
        expected = _oracle_chain(expr, start, direction, count, day_or)
    except ORACLE_ERRORS as exc:
        with pytest.raises(CroniterMojoError):
            _ours_chain(expr, start, direction, count, day_or)
        return
    got = _ours_chain(expr, start, direction, count, day_or)
    assert len(got) == len(expected), ctx
    for i, (o, m) in enumerate(zip(expected, got)):
        _assert_aware_identical(o, m, f"{ctx} step={i}")


# ---------------------------------------------------------------- naive corpus


@pytest.mark.parametrize("expr", EXPR_CORPUS)
def test_naive_corpus_next(expr):
    for start in NAIVE_STARTS:
        _check_chain(expr, start, +1, count=4)
        _check_chain(expr, start, +1, count=4, day_or=False)


@pytest.mark.parametrize("expr", EXPR_CORPUS)
def test_naive_corpus_prev(expr):
    for start in NAIVE_STARTS:
        _check_chain(expr, start, -1, count=4)
        _check_chain(expr, start, -1, count=4, day_or=False)


def test_expected_backend():
    import os

    info = croniter_mojo.backend_info()
    if os.environ.get("CRONITER_MOJO_DISABLE_NATIVE") == "1":
        assert not info["native_available"]
    else:
        assert info["native_available"], f"native backend required for this run: {info}"


def test_expand_from_start_time_is_oracle_noop():
    starts = [
        datetime(2026, 9, 21, 9, 0, 0),
        datetime(2026, 9, 21, 9, 0, 30),
        datetime(2026, 9, 21, 9, 0, 0, 400000),
    ]
    exprs = ["0 9 * * mon", "* * * * *", "0 9 * * *", "0 * * * *"]
    for expr in exprs:
        for start in starts:
            for direction in (+1, -1):
                expand = _ours_chain_expand(expr, start, direction, True)
                plain = _ours_chain_expand(expr, start, direction, False)
                oracle_expand = _oracle_chain_expand(expr, start, direction, True)
                assert expand == plain == oracle_expand, (expr, start, direction)


def _ours_chain_expand(expr, start, direction, expand):
    if direction > 0:
        return croniter_mojo.get_next(expr, start, expand_from_start_time=expand)
    return croniter_mojo.get_prev(expr, start, expand_from_start_time=expand)


def _oracle_chain_expand(expr, start, direction, expand):
    c = croniter(expr, start, expand_from_start_time=expand)
    return c.get_next(datetime) if direction > 0 else c.get_prev(datetime)


# -------------------------------------------------------------------- errors


BAD_EXPRS = [
    "",
    "* * * *",
    "* * *",
    "60 * * * *",
    "* 24 * * *",
    "0 0 32 * *",
    "0 0 * 13 *",
    "0 0 * * 8",
    "abc * * * *",
    "*/0 * * * *",
    "0 0 * * 5#6",
    "0 0 * * 5#0",
    "0 0 L-3 * *",
    "0 0 * * 5L",
    "0 0 * * 1,5#3",  # mixing plain and hash dow
    "0 0 * * mon,5#3",
]

UNSUPPORTED_EXPRS = [
    "0 0 9 * * *",  # 6-field (seconds) — oracle accepts, out of our scope
    "0 0 9 * * * 2030",  # 7-field (year)
]

NEVER_EXPRS = [
    "0 0 31 2 *",
    "0 0 13 * 5#1",  # static gate
    "0 0 31 4 mon",  # day_or=False AND: April 31st that is a Monday
]


@pytest.mark.parametrize("expr", BAD_EXPRS)
def test_bad_expressions_raise_like_oracle(expr):
    start = datetime(2026, 9, 19, 14, 30)
    with pytest.raises(ORACLE_ERRORS):
        croniter(expr, start).get_next(datetime)
    with pytest.raises(CroniterMojoError):
        croniter_mojo.get_next(expr, start)
    with pytest.raises(CroniterMojoError):
        croniter_mojo.get_prev(expr, start)


@pytest.mark.parametrize("expr", UNSUPPORTED_EXPRS)
def test_out_of_scope_field_counts_raise(expr):
    start = datetime(2026, 9, 19, 14, 30)
    with pytest.raises(CroniterMojoError):
        croniter_mojo.get_next(expr, start)


@pytest.mark.parametrize("expr", NEVER_EXPRS)
def test_never_matching_raises_like_oracle(expr):
    start = datetime(2026, 9, 19, 14, 30)
    with pytest.raises(ORACLE_ERRORS):
        croniter(expr, start, day_or=False).get_next(datetime)
    with pytest.raises(CroniterBadDateError):
        croniter_mojo.get_next(expr, start, day_or=False)
    with pytest.raises(CroniterBadDateError):
        croniter_mojo.get_prev(expr, start, day_or=False)


def test_year_bound_edges():
    # No next minute exists after the last minute of 9999.
    with pytest.raises(ORACLE_ERRORS):
        croniter("* * * * *", datetime(9999, 12, 31, 23, 59)).get_next(datetime)
    with pytest.raises(CroniterBadDateError):
        croniter_mojo.get_next("* * * * *", datetime(9999, 12, 31, 23, 59))
    # get_prev near year 1: a matching date within 50 years is still found.
    o = croniter("0 0 29 2 *", datetime(10, 1, 1)).get_prev(datetime)
    m = croniter_mojo.get_prev("0 0 29 2 *", datetime(10, 1, 1))
    assert m == o == datetime(8, 2, 29)
    # ... and a genuinely unmatchable expression raises on both sides.
    with pytest.raises(ORACLE_ERRORS):
        croniter("0 0 31 2 *", datetime(10, 1, 1)).get_prev(datetime)
    with pytest.raises(CroniterBadDateError):
        croniter_mojo.get_prev("0 0 31 2 *", datetime(10, 1, 1))
    # 2100 is not a leap year: next Feb 29 after 2096 is 2104.
    o = croniter("0 0 29 2 *", datetime(2096, 3, 1)).get_next(datetime)
    m = croniter_mojo.get_next("0 0 29 2 *", datetime(2096, 3, 1))
    assert m == o == datetime(2104, 2, 29)


# ------------------------------------------------------------------ DST sweeps


@pytest.mark.parametrize("tz_name", TZS)
def test_dst_transition_sweeps(tz_name):
    tz = ZoneInfo(tz_name)
    for win_start, win_end in TRANSITION_WINDOWS[tz_name]:
        # Starts every 20 minutes inside the window, fold 0 and 1.
        starts = []
        cur = win_start
        while cur <= win_end:
            for fold in (0, 1):
                starts.append(cur.replace(tzinfo=tz, fold=fold))
            # keep wall-time stepping naive to walk through gaps/folds
            cur = cur.replace(tzinfo=None) + timedelta(minutes=20)
        for start in starts:
            for expr in DST_EXPRS:
                _check_chain(expr, start, +1, count=3)
                _check_chain(expr, start, -1, count=3)


def test_dst_daily_exprs_around_transitions():
    # Daily schedules across full transition weeks, both directions.
    for tz_name in ["America/New_York", "Australia/Lord_Howe", "Europe/Berlin"]:
        tz = ZoneInfo(tz_name)
        for win_start, _ in TRANSITION_WINDOWS[tz_name]:
            for expr in ["0 2 * * *", "30 2 * * *", "0 1 * * *", "30 1 * * *", "0 9 * * *"]:
                base = win_start.replace(tzinfo=tz) - timedelta(days=2)
                _check_chain(expr, base, +1, count=8)
                _check_chain(expr, base, -1, count=8)


# --------------------------------------------------------------- fuzz (seeded)


def _rand_field(rng, lo, hi, names=()):
    kind = rng.randrange(8)
    if kind == 0:
        return "*"
    if kind == 1:
        return f"*/{rng.randint(1, 12)}"
    if kind == 2:
        return str(rng.randint(lo, hi))
    if kind == 3:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        return f"{a}-{b}"
    if kind == 4:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        return f"{a}-{b}/{rng.randint(1, 12)}"
    if kind == 5:
        return f"{rng.randint(lo, hi)}/{rng.randint(1, 12)}"
    if kind == 6 and names:
        a, b = rng.choice(names), rng.choice(names)
        return f"{a}-{b}" if rng.random() < 0.5 else a
    return ",".join(str(rng.randint(lo, hi)) for _ in range(rng.randint(2, 3)))


def _rand_expr(rng):
    minute = _rand_field(rng, 0, 59)
    hour = _rand_field(rng, 0, 23)
    dom = _rand_field(rng, 1, 31)
    if rng.random() < 0.15:
        dom = "l" if rng.random() < 0.5 else f"{rng.randint(1, 31)},l"
    if rng.random() < 0.1:
        dom = "?"
    month = _rand_field(rng, 1, 12, names=("jan", "feb", "mar", "apr", "may", "jun",
                                           "jul", "aug", "sep", "oct", "nov", "dec"))
    dow_kind = rng.random()
    if dow_kind < 0.15:
        dow = "?"
    elif dow_kind < 0.35:
        specs = []
        for _ in range(rng.randint(1, 2)):
            d = rng.randint(0, 7)
            specs.append(f"{d}#{rng.randint(1, 5)}")
        dow = ",".join(specs)
    else:
        dow = _rand_field(rng, 0, 7, names=("sun", "mon", "tue", "wed", "thu", "fri", "sat"))
    return f"{minute} {hour} {dom} {month} {dow}"


def _rand_start(rng):
    year = rng.choice([2024, 2025, 2026, 2027, 2028, 2030])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return datetime(year, month, day, rng.randint(0, 23), rng.randint(0, 59),
                    rng.choice([0, 0, 0, 30, 59]), rng.choice([0, 0, 123456]))


def test_seeded_fuzz_naive():
    rng = random.Random(20260919)
    for _ in range(400):
        expr = _rand_expr(rng)
        start = _rand_start(rng)
        day_or = rng.random() < 0.75
        _check_chain(expr, start, +1, count=5, day_or=day_or)
        _check_chain(expr, start, -1, count=5, day_or=day_or)


def test_seeded_fuzz_aware():
    rng = random.Random(20260920)
    tz_names = ["America/New_York", "Europe/Berlin", "Australia/Sydney",
                "Australia/Lord_Howe", "Pacific/Auckland", "UTC"]
    for _ in range(150):
        expr = _rand_expr(rng)
        start = _rand_start(rng)
        tz = ZoneInfo(rng.choice(tz_names))
        aware = start.replace(tzinfo=tz, fold=rng.randrange(2))
        day_or = rng.random() < 0.75
        _check_chain(expr, aware, +1, count=5, day_or=day_or)
        _check_chain(expr, aware, -1, count=5, day_or=day_or)


def test_seeded_fuzz_aware_transition_months():
    # Starts concentrated in the transition months (Mar/Apr/Oct/Nov).
    rng = random.Random(20260921)
    tz_windows = [(tz, w) for tz, ws in TRANSITION_WINDOWS.items() for w in ws]
    for _ in range(200):
        tz_name, (win_start, _) = rng.choice(tz_windows)
        tz = ZoneInfo(tz_name)
        expr = _rand_expr(rng)
        delta_hours = rng.randint(-72, 72)
        base = win_start + timedelta(hours=delta_hours,
                                                            minutes=rng.randint(0, 59),
                                                            seconds=rng.choice([0, 30]))
        aware = base.replace(tzinfo=tz, fold=rng.randrange(2))
        day_or = rng.random() < 0.75
        _check_chain(expr, aware, +1, count=4, day_or=day_or)
        _check_chain(expr, aware, -1, count=4, day_or=day_or)
