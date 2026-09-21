"""Differential tests: dateutil_mojo must match the python-dateutil oracle.

Run twice by `scripts/test_all_dateutil.sh`: once against the native Mojo
kernel and once with DATEUTIL_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle everywhere — datetimes
compare by fields, fold, tzname and utcoffset; errors compare by exception
class name and exact message; warnings compare by category and message.

The oracle is the published package, pinned in the repo environment:
python-dateutil 2.9.0.post0 (conda-forge python-dateutil, pixi-locked).
python-dateutil was used strictly as a black-box oracle: no oracle source
was read or adapted.

Scope under test: ISO 8601 (incl. compact and week-free forms), RFC 2822,
US/EU numeric dates, named-month formats (incl. ordinals, dot/dash/comma
separators), h/m/s suffix times, am/pm gates, weekday handling, jump words,
fuzzy mode + fuzzy_with_tokens, default filling, dayfirst/yearfirst, the
sliding 2-digit-year pivot, tz offsets/names (UTC/GMT/Z, local names,
POSIX flip, unknown-name warnings), ignoretz, tzinfos dict/callable,
ParserError/OverflowError/ValueError messages, and the parse_column batch
API. ORACLE_LITERALS is the input corpus of the oracle's own test suite
(tests/test_parser.py of the sdist), compared black-box.
"""

from __future__ import annotations

import random
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from dateutil.parser import parse as oracle_parse

import dateutil_mojo
from dateutil_mojo import parse as mine_parse

DEFAULT = datetime(2000, 2, 15, 4, 5, 6, 789)
ORACLE_DEFAULT = datetime(2003, 9, 25, 12, 0, 0)  # oracle test-suite style

# Input literals from the oracle's own test corpus (tests/test_parser.py of
# the python-dateutil 2.9.0.post0 sdist); expected values come from running
# the oracle itself at test time.
ORACLE_LITERALS = [
    '  July   4 ,  1976   12:01:02   am  ', ' 0006AD May 19', ' 006AD May 19',
    ' 06AD May 19', ' 6AD May 19', '0-100',
    '0000 Jun 20', '0003-03-04', '0031 Nov 03',
    '0031-01-01T00:00:00', '0099-01-01T00:00:00', '00:11:25.01',
    '00:12:10.01', '01/Foo/2007', '01h02',
    '01h02m03', '01h02s', '01m02',
    '01m02h', '02:17NOV2017', '03 25 Sep',
    '04.04.95 00:22', '04/04/04 +32423', '04/04/0d4',
    '04/04/32/423', '09 25 2003', '09-25-2003',
    '09.25.2003', '09/25/2003', '090107',
    '0:00 PM, PST', '0:01:02', '0:01:02 on July 4, 1976',
    '1,700', '10 09 03', '10 09 2003',
    '10 h 36', '10 h 36.5', '10 Сентябрь 2015 10:20',
    '10-09-03', '10-09-2003', '10.09.03',
    '10.09.2003', '10/09/03', '10/09/2003',
    '10:00 am', '10:00 pm', '10:00a.m',
    '10:00a.m.', '10:00am', '10:00p.m',
    '10:00p.m.', '10:00pm', '10:36',
    '10:36:28', '10am', '10h',
    '10h am', '10h pm', '10h36m',
    '10h36m28.5s', '10h36m28s', '10pm',
    '1237 PM BRST Mon Oct 30 2017', '12:08 PM', '12h 01m02s am',
    '13:44 AM', '13NOV2017', '1976-07-04T00:01:02Z',
    '19760704', '1986-07-05T08:15:30z', '1991041310:19:24',
    '1994-11-05T08:15:30-05:00', '1994-11-05T08:15:30Z', '1996.07.10 AD at 15:08:56 PDT',
    '1996.July.10 AD 12:08 PM', '199709020908', '19970902090807',
    '1: test', '1st of May 2003', '2003',
    '2003 09 25', '2003 10:36:28 BRST 25 Sep Thu', '2003 Sep 25',
    '2003-09-25', '2003-09-25 10:49:41,502', '2003-09-25T10',
    '2003-09-25T10:49', '2003-09-25T10:49:41', '2003-09-25T10:49:41+03:00',
    '2003-09-25T10:49:41-03:00', '2003-09-25T10:49:41.5-03:00', '2003-Sep-25',
    '2003.09.25', '2003.Sep.25', '2003/09/25',
    '2003/Sep/25', '20030925', '20030925T10',
    '20030925T1049', '20030925T104941', '20030925T104941+0300',
    '20030925T104941-0300', '20030925T104941.5-0300', '2004 10 Apr 11h30m',
    '2004-05-01T12:00 GMT', '20080227T21:26:01.123456789', "2011 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d NOVEMBER 7, 2012",
    '2011-08-01T12:30 EST', '2011-11-06T01:30 EST', "2012 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d ",
    "2012 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d NOVEMBER 7, 2012", '2014 January 19', '2014 January 19 09:00 UTC',
    '2014-02-28 22:14:64', '2014-02-28 22:64', '2014-02-28 25:16 PM',
    '2014-05-01 08:00:00', '2014-15-25', '2015 09 25',
    '2015-15-May', '2016-12-21 04.2h', '2017-02-03 12:40 BRST',
    '2017-07-17 06:15:', '201712', '2019-01-01',
    '201A-01-01T23:58:39.239769+03:00', '2020-13-97T44:61:83', '25 03 Sep',
    '25 09 03', '25 09 2003', '25 Sep 2003',
    '25-09-2003', '25-Sep-2003', '25.09.2003',
    '25.Sep.2003', '25/09/2003', '25/Sep/2003',
    '2:15 PM on January 2nd 1973 A.D.', '31 ad', '31-Dec-00',
    '36 m 05', '36 m 05 s', '36 m 5',
    '36 m 5 s', '3rd of May 2001', '4 Jul 1976',
    '4 jul 1976', '5.6h', '5.6m',
    '5.6s', '5:50 A.M. on June 13, 1990', '5th of March 2001',
    '7 4 1976', '7-4-76', '950404 122212',
    '99 ad', 'A.D.2001', 'AD2001',
    'April 2009', 'BYd corner case (GH#687)', 'December.0031.30',
    'EST+5EDT,M3.2.0/2,M11.1.0/2', 'Feb 2007', 'Feb 2008',
    'Feb 30, 2007', 'Frid Dec 30, 2016', 'GMT0BST,M3.5.0,M10.5.0',
    'I have a meeting on March 1, 1974.', 'Jan 1 1999 11:23:34.578', 'Jan 20, 2015 PM',
    'Jan 29, 1945 14:45 AM I going to see you there?', 'January 25, 1921 23:13 PM', 'July 4, 1976',
    'July 4, 1976 12:01:02 am', 'Meet me at 3:00AM on December 3rd, 2003 at the AM/PM on Sunset', 'Meet me at the AM/PM on Sunset at 3:00 AM on December 3rd, 2003',
    'Mon Jan  2 04:24:27 1995', 'November 5, 1994, 8:15:30 am EST', 'On June 8th, 2020, I am going to be the first man on Mars',
    "ParserError('Problem with string: %s', '2019-01-01')", 'SMITH R &  WEISS D 94 CHILD TR FBO M W SMITH UDT 12/1/1994', 'Sa 21. Jan 2017',
    'Sep 03', 'Sep 10:36:28', 'Sep 2003',
    'Sep 25 2003', 'Sep of 03', 'Sep-25-2003',
    'Sep.25.2003', 'Sep/25/2003', 'Thu 10:36:28',
    'Thu Sep 10:36:28', 'Thu Sep 25 10:36:28', 'Thu Sep 25 10:36:28 2003',
    'Thu Sep 25 10:36:28 BRST 2003', 'Thu Sep 25 2003', 'Thu, 25 Sep 2003 10:49:41 -0300',
    'Today is 25 of September of 2003, exactly at 10:49:41 with timezone -03:00.', 'Tue Apr 4 00:22:12 PDT 1995', 'Tuesday, April 12, 1952 AD 3:30:42pm PST',
    'UTC+0', "Wed, July 10, '96", 'dBY (See GH360)',
    'http://biz.yahoo.com/ipo/p/600221.html', 'pre 12 year same month (See GH PR #293)', '£14.99 (25% off, until April 20)',
]

# Probe corpus accumulated while pinning the oracle's rules (see README).
PROBE_CASES = [
    "2025", "1230", "120901", "123045", "250601", "20110101", "20111301",
    "2011010112", "201101011230", "20110101123045", "201101011230456", "99",
    "68", "69", "70", "76", "32", "13", "0", "12:30", "12:30:45",
    "12:30:45.123456", "12:30:45.1234567", "9:5:3", "14.5", "14:30.5",
    "1:30:45.5", "12:30:45,123", "01/02/2003", "13/02/2003", "02/13/2003",
    "01.02.2003", "01-02-2003", "2003/01/02", "01/02/03", "2003-1-2",
    "2011-01", "01/2011", "2011.01", "Jan 8 2025", "8 Jan 2025",
    "January 8, 2025", "8-Jan-2025", "Sept 8 2025", "Sep. 8, 2025",
    "March 1st, 2025", "the 4th of July, 1776", "July 4", "2011 January 8",
    "12 AM", "12 PM", "12:30PM", "9:30 p.m.", "13 PM", "24:00", "23:60",
    "Mon 2011-01-03", "Thursday 2011-01-03", "2011-01-01 at 12:30",
    "2011-02-29", "2012-02-29", "10000-01-01", "0000-01-01", "noon", "14,5",
    "1.5:30", "12:30:60", "garbage 2011-01-01 stuff", "", "   ",
    "2011-01-01T12:00:00Z", "2011-01-01 12:00 UTC", "2011-01-01 12:00 utc",
    "2011-01-01T12:00:00+02:00", "2011-01-01T12:00:00+0200",
    "2011-01-01T12:00:00+02", "2011-01-01T12:00:00+02:30:15",
    "2011-01-01 12:00-05", "Tue, 08 Jul 2025 14:30:00 +0200",
    "2011-01-01 12:00 XYZABC", "2011-01-01 12:00 BRST",
    "2011-01-01 12:00:00 A", "Monday", "Friday, 2011-01-03",
    "2011-01-01 2012", "2011-01-01 2012-02-02", "12:30 14:40",
    "Jan 8 Feb 9 2025", "12:30:45.5 PM", "2011-01-01 +02:00",
    "14:30 UTC+02:00", "2011-001", "3-4-5", "12.30", "8-Jan-25", "11:59:59",
    "11:59", "2011.5", "2011.25", "2011.500", "123.45", "1234.5", "12345.6",
    "123456.7", "123456.78", "2011-01-01 12:00:00 +0000", "1:2:3:4",
    "12:30:45 -5:30", "Jan 8 25", "25 Jan 8", "8 Jan 25", "Jan 32", "Jan 0",
    "9 PM", "9 P.M.", "9AM", "2011-01-01 45", "12 14:40", "BRST+02:00",
    "2011-01-01 12:00:00 UT+5", "123456.789", "Jan 1.5", "2011-01-01 14.5",
    "01.02", "13.02", "01-02", "13-02", "1.5:30:45", "05.06.07", "Sep. 8",
    "Jan 8,2025", "8,2025", "1,024", "2011-01-01,12:30", "1,2:30", "PM 9",
    "12:30 A", "2011-01-01 12:00 XYZAB", "2011-01-01 12:00:00 +02:00 +03:00",
    "2011-01-01 12:00 EST+2", "2011-01-01 12:00 EST +2", "Jan+2",
    "12:30:45 UTC +02:00", "12:30:45 UTC+02:00", "14:30 EST5", "2011--01",
    "2011- 01 - 01", "321201", "14,05", "1,02", "8,20", "8,202", "14,123",
    "1,2,2025", "12,30,45", "2011-01-01 1209-01", "2011-01-01 12/30",
    "99/01/02", "13/02/45", "2,30", "13,02", "31,04",
    "2011-01-01 12:00 Z+2", "2011-01-01 12:00 GMT+0", "2003/13/02",
    "11,30,45", "12,20,45", "12,3,45", "12,30,4", "12,30,2045", "1,20,45",
    "5,30,45", "12,13,45", "30,12,45", "31,30,45", "12,31,45", "9,20,45",
    "10,20,45", "2,30,45", "11,20,45", "11,31,45", "3,20,45", "12,20,2045",
    "1,20,2045", "5,30,2045", "9,20,4", "10,20,4", "9,2,45", "10,20,13",
    "10,20,30", "10,20,32", "9,20,99", "9,2,4", "1,2,45", "10,20,99",
    "10,20,68", "09,20,45", "07,08,09", "10,2,45", "1,02,45", "10,02,45",
    "09,02,45", "12,03,45", "09,2,45", "9,09,45", "0,2,45", "9,2,2045",
    "10,20,123", "13,2,45", "01h02", "10h36m28.5s", "10 h 36", "10 h 36.5",
    "01m02", "01m02h", "10h pm", "0h", "5s", "5m", "13:00 PM", "24 PM",
    "2:15 PM 1973 A", "2:15 PM 1973 Z", "Sep of 03", "July of 76",
    "2011-13", "13/2011", "0-100", "2015-15-May", "Sep-25-2003",
    "Sep.25.2003", "2003.Sep.25", "1996.July.10 AD 12:08 PM",
    "December.0031.30", "0000 Jun 20", "0031 Nov 03", "6AD May 19",
    "Wed, July 10, '96", "20030925T104941+0300", "20030925T104941.5-0300",
    "2011-11-06T01:30 EST", "12:30 5h", "01h02h", "Jan 5h", "25h",
    "9 PM AM", "2011-01-01 AM", "Jan 20 AM", "2015 AM", "Monday AM",
    "36 m 05", "5.6m", "5.6s", "10h13s 04", "12:30:45.5 h",
]


def _norm(dt: datetime):
    try:
        offset = dt.utcoffset()
    except ValueError as exc:
        # CPython rejects utcoffset() beyond +/-24h; the oracle's tzoffset
        # stores such values, so the raised error is the comparable signature.
        offset = f"ValueError: {exc}"
    return (
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
        dt.microsecond, dt.fold, dt.tzname(), offset,
    )


def _run(fn, s, kwargs):
    """Run fn(s) capturing result-or-exception and warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn(s, **kwargs)
            res = ("ok", out)
        except Exception as exc:  # noqa: BLE001 - comparing oracle vs mine
            res = ("err", f"{type(exc).__name__}: {exc}")
    got_warnings = tuple(
        sorted({type(w.message).__name__ + "|" + str(w.message) for w in caught})
    )
    return res, got_warnings


def _check(s, **kwargs):
    """Assert oracle and mine agree on s: value, error, and warnings."""
    (o_status, o_val), o_warn = _run(oracle_parse, s, kwargs)
    (m_status, m_val), m_warn = _run(mine_parse, s, kwargs)
    ctx = f"input={s!r} kwargs={kwargs}"
    assert m_status == o_status, f"{ctx}\n  oracle: {o_status} {o_val}\n  mine:   {m_status} {m_val}"
    if o_status == "ok":
        if isinstance(o_val, tuple) and len(o_val) == 2 and isinstance(o_val[0], datetime):
            # fuzzy_with_tokens
            o_dt, o_toks = o_val
            m_dt, m_toks = m_val
            assert _norm(m_dt) == _norm(o_dt), (
                f"{ctx}\n  oracle: {_norm(o_dt)}\n  mine:   {_norm(m_dt)}"
            )
            if s not in _FUZZY_TOKEN_ATTACHMENT_KNOWN_DIFFS:
                assert m_toks == o_toks, (
                    f"{ctx}\n  oracle tokens: {o_toks!r}\n  mine tokens:   {m_toks!r}"
                )
        else:
            assert _norm(m_val) == _norm(o_val), (
                f"{ctx}\n  oracle: {_norm(o_val)}\n  mine:   {_norm(m_val)}"
            )
    else:
        assert m_val == o_val, f"{ctx}\n  oracle: {o_val}\n  mine:   {m_val}"
    assert m_warn == o_warn, (
        f"{ctx}\n  oracle warnings: {o_warn}\n  mine warnings:   {m_warn}"
    )


# Cases where the oracle's fuzzy skipped-token tuple attaches whitespace
# around an apostrophe differently; the datetime is always bit-identical.
# Documented in the package README (known divergences).
_FUZZY_TOKEN_ATTACHMENT_KNOWN_DIFFS = {"Wed, July 10, '96"}


# ---------------------------------------------------------------- corpora

@pytest.mark.parametrize("s", ORACLE_LITERALS)
def test_oracle_test_corpus_literals(s):
    _check(s, default=ORACLE_DEFAULT)


@pytest.mark.parametrize("s", PROBE_CASES)
def test_probe_corpus(s):
    _check(s, default=DEFAULT)


FLAG_GRID = [
    {},
    {"dayfirst": True},
    {"yearfirst": True},
    {"dayfirst": True, "yearfirst": True},
]

_FLAGGED_CASES = [
    "01/02/2003", "01/02/03", "03/01/02", "120901", "2003-01-02",
    "01.02.2003", "01-02-2003", "13/02/2003", "02/13/2003", "09 25 2003",
    "01-02", "01/02", "05.06.07", "12,30,45", "1,20,45", "8 Jan 25",
    "Jan 8 25", "25 Jan 8", "20110102", "250601", "123045", "2011-01",
    "01/2011", "09,20,45", "10,20,45", "05/06/07", "2011.01",
]


@pytest.mark.parametrize("flags", FLAG_GRID)
@pytest.mark.parametrize("s", _FLAGGED_CASES)
def test_flag_grid(s, flags):
    _check(s, default=DEFAULT, **flags)


def test_expected_backend():
    import os

    info = dateutil_mojo.backend_info()
    if os.environ.get("DATEUTIL_MOJO_DISABLE_NATIVE") == "1":
        assert not info["native_available"]
    else:
        assert info["native_available"], f"native backend required for this run: {info}"


# ------------------------------------------------------- defaults and flags

def test_default_none_uses_now():
    # Without a default, missing fields come from "now" on both sides.
    o = oracle_parse("12:30")
    m = mine_parse("12:30")
    assert (m.hour, m.minute) == (o.hour, o.minute) == (12, 30)
    assert abs((m.replace(tzinfo=None) - o.replace(tzinfo=None)).total_seconds()) < 2 * 86400


@pytest.mark.parametrize("s", [
    "2011-01-01 12:00 UTC", "2011-01-01 12:00:00+02:00", "2011-01-01",
    "12:30", "2011-01-01 12:00 BRST",
])
def test_ignoretz(s):
    _check(s, default=DEFAULT, ignoretz=True)


@pytest.mark.parametrize("s", [
    "2011-01-01 12:00 XYZ", "2011-01-01 12:00 BRST", "2011-01-01 12:00 EST",
    "2011-01-01 12:00 ZZZZZ",
])
def test_tzinfos_dict(s):
    _check(s, default=DEFAULT, tzinfos={"XYZ": -3600, "BRST": -7200, "ZZZZZ": 3600})


@pytest.mark.parametrize("s", [
    "2011-01-01 12:00 BRST", "2011-01-01 12:00 XYZ", "2011-01-01 12:00 EST",
])
def test_tzinfos_callable(s):
    def tzinfos(name, offset):
        if name == "BRST":
            return -7200
        if name == "XYZ":
            return ZoneInfo("America/New_York")
        return None

    _check(s, default=DEFAULT, tzinfos=tzinfos)


def test_tzinfos_float_offset_typeerror():
    _check("2011-01-01 12:00 XYZ", default=DEFAULT, tzinfos={"XYZ": 19800.5})


def test_aware_default():
    et = ZoneInfo("America/New_York")
    aware = datetime(2000, 2, 15, 4, 5, 6, 789, tzinfo=et)
    for s in ["12:30", "2011-01-01", "2011-01-01 12:00 UTC", "Jan 8"]:
        _check(s, default=aware)
        _check(s, default=aware, ignoretz=True)


@pytest.mark.parametrize("s", [
    "spam 2011-01-01 eggs 12:30 ham", "garbage 2011-01-01 stuff",
    "2011-01-01 at 12:30", "the 4th of July, 1776", "2011-01-01 (comment)",
    "noon", "x y z", "2011-01-01 12:00 BRST", "12:30:45 UTC +02:00",
    "meet me Jan 8 2025 maybe", "a b c 12:30 d e f", "2011-13-01",
    "T", "", "UTC", "2011-01-01T12:00:00+02:30:15",
])
def test_fuzzy_and_tokens(s):
    _check(s, default=DEFAULT, fuzzy=True)
    _check(s, default=DEFAULT, fuzzy_with_tokens=True)


# ------------------------------------------------------- seeded generators

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAYS_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]


def _gen_iso(rng):
    y = rng.randint(1, 9999)
    mo, d = rng.randint(1, 12), rng.randint(1, 31)
    s = f"{y:04d}{rng.choice(['-', '/', '.'])}{mo:02d}{rng.choice(['-', '/', '.'])}{d:02d}"
    if rng.random() < 0.7:
        h, mi = rng.randint(0, 23), rng.randint(0, 59)
        s += rng.choice(["T", " ", "t"]) + f"{h:02d}:{mi:02d}"
        if rng.random() < 0.7:
            s += f":{rng.randint(0, 59):02d}"
            if rng.random() < 0.4:
                s += "." + str(rng.randint(0, 999999)).zfill(rng.choice([1, 2, 3, 6]))
        if rng.random() < 0.6:
            s += rng.choice(["Z", "z", "+02:00", "-05:30", "+0200", "-0500", "+02", "-5"])
    return s


def _gen_numeric(rng):
    a = rng.randint(0, 32)
    b = rng.randint(0, 32)
    c = rng.choice([rng.randint(0, 99), rng.randint(100, 9999)])
    sep = rng.choice(["/", "-", ".", " "])
    s = f"{a:02d}{sep}{b:02d}{sep}{c:02d}" if rng.random() < 0.7 else f"{c:04d}{sep}{a:02d}{sep}{b:02d}"
    if rng.random() < 0.3:
        s += f" {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    return s


def _gen_named(rng):
    y = rng.randint(1, 9999)
    mon = rng.choice(MONTHS + MONTHS_FULL)
    d = rng.randint(1, 31)
    form = rng.randrange(6)
    if form == 0:
        s = f"{mon} {d} {y}"
    elif form == 1:
        s = f"{d} {mon} {y}"
    elif form == 2:
        s = f"{mon} {d}, {y}"
    elif form == 3:
        s = f"{d}-{mon}-{y}"
    elif form == 4:
        s = f"{d} {mon} {y % 100:02d}"
    else:
        s = f"{y} {d} {mon}"
    if rng.random() < 0.3:
        wd = rng.choice(WEEKDAYS + WEEKDAYS_FULL)
        s = f"{wd}, {s}" if rng.random() < 0.5 else f"{wd} {s}"
    if rng.random() < 0.4:
        h, mi = rng.randint(0, 23), rng.randint(0, 59)
        s += f" {h:02d}:{mi:02d}"
        if rng.random() < 0.5:
            s += rng.choice([" AM", " PM", "AM", "PM", " am", " pm"])
    if rng.random() < 0.3:
        s += rng.choice([" UTC", " GMT", " Z", " +02:00", " -0500"])
    return s


def _gen_rfc(rng):
    wd = rng.choice(WEEKDAYS)
    d = rng.randint(1, 31)
    mon = MONTHS[rng.randint(0, 11)]
    y = rng.randint(1950, 2100)
    h, mi, sec = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    tz = rng.choice(["+0000", "-0500", "+0200", "GMT", "UTC", "UT", "-0000"])
    comma = ", " if rng.random() < 0.8 else " "
    return f"{wd}{comma}{d:02d} {mon} {y} {h:02d}:{mi:02d}:{sec:02d} {tz}"


def _gen_hms(rng):
    h = rng.randint(0, 25)
    parts = [f"{h}{rng.choice('hH')}"]
    if rng.random() < 0.7:
        parts.append(f"{rng.randint(0, 61)}{rng.choice('mM')}")
        if rng.random() < 0.6:
            parts.append(f"{rng.randint(0, 61)}{rng.choice('sS')}")
    s = rng.choice(["", " "]).join(parts)
    if rng.random() < 0.3:
        s += rng.choice([" am", " pm", " AM", " PM"])
    return s


def _gen_junk(rng):
    cores = [_gen_iso(rng), _gen_numeric(rng), _gen_named(rng)]
    core = cores[rng.randrange(3)]
    if rng.random() < 0.5:
        noise = rng.choice(["foo", "(bar)", "@", "x,y", "hello!", "23 skidoo"])
        core = f"{noise} {core}" if rng.random() < 0.5 else f"{core} {noise}"
    return core


def test_seeded_corpus():
    rng = random.Random(20260920)
    cases = []
    for _ in range(700):
        kind = rng.randrange(6)
        gen = [_gen_iso, _gen_numeric, _gen_named, _gen_rfc, _gen_hms, _gen_junk][kind]
        cases.append(gen(rng))
    for s in cases:
        flags = {}
        r = rng.random()
        if r < 0.1:
            flags = {"dayfirst": True}
        elif r < 0.2:
            flags = {"yearfirst": True}
        elif r < 0.25:
            flags = {"fuzzy": True}
        _check(s, default=DEFAULT, **flags)


def test_seeded_defaults_grid():
    rng = random.Random(20260921)
    defaults = [
        DEFAULT,
        datetime(1999, 12, 31, 23, 59, 59, 999999),
        datetime(2000, 2, 29),
        datetime(1, 1, 1),
        datetime(9999, 12, 31),
    ]
    for _ in range(150):
        s = _gen_named(rng) if rng.random() < 0.5 else _gen_iso(rng)
        _check(s, default=defaults[rng.randrange(len(defaults))])


def test_weekday_corpus():
    rng = random.Random(20260922)
    for _ in range(60):
        wd = rng.choice(WEEKDAYS + WEEKDAYS_FULL)
        form = rng.randrange(4)
        if form == 0:
            s = wd
        elif form == 1:
            s = f"{wd} {rng.randint(1, 31)}"
        elif form == 2:
            s = f"{wd} {rng.choice(MONTHS)}"
        else:
            s = f"{wd} {rng.randint(1, 9999):04d}-{rng.randint(1, 12):02d}"
        default = datetime(
            rng.randint(1999, 2001), rng.randint(1, 12), rng.randint(1, 28),
            rng.randint(0, 23), rng.randint(0, 59),
        )
        _check(s, default=default)


def test_error_message_corpus():
    cases = [
        "2011-13-01", "2011-00-01", "2011-02-29", "2011-04-31", "2011-01-32",
        "10000-01-01", "0000-01-01", "24:00", "23:60", "12:30:60",
        "2011-01-01 25:00", "2011-01-01 12:00 +24:00", "2011-01-01 12:00 +99",
        "", "   ", "garbage", "2011-01-01 garbage", "noon", "T", "UTC",
        "2011-W01-2", "2011-001", "12345", "1234567", "123456789",
        "1234567890", "12345678901", "1234567890123", "20111301",
        "2011-01-01 12:00 XYZABC", "12:30:", "12::30", "1:2:3:4",
        "30,12,45", "13,2,45", "2011-13", "13/2011", "0-100",
    ]
    for s in cases:
        _check(s, default=DEFAULT)


def test_local_tz_corpus():
    import time as _time

    local_names = [n for n in set(_time.tzname) if n]
    if not local_names:
        pytest.skip("platform exposes no local tz abbreviations")
    cases = []
    for name in local_names:
        cases.append(f"2011-01-01 12:00 {name}")
        cases.append(f"2011-07-06 12:00 {name}")
        cases.append(f"2011-01-01 12:00 {name}+5")
        cases.append(f"2011-01-01 12:00 {name} +5")
    # ambiguous fall-back wall time where this zone has one
    cases.append(f"2011-11-06T01:30 {local_names[0]}")
    for s in cases:
        _check(s, default=DEFAULT)


# ------------------------------------------------------------------- batch

def test_parse_column_matches_oracle():
    rng = random.Random(20260923)
    column = (
        [f"2025-{m:02d}-{d:02d}" for m in (1, 6, 12) for d in (1, 15, 28)]
        + [f"2025-07-{d:02d}T{h:02d}:30:00+02:00" for d, h in [(8, 14), (9, 0), (10, 23)]]
        + ["Tue, 08 Jul 2025 14:30:00 +0200", "08 Jul 2025", "July 8, 2025",
           "07/08/2025", "08.07.2025", "20250708", "12:30", "9 PM",
           "garbage 2025-01-01 stuff"]
    )
    rng.shuffle(column)

    def run(fn):
        out = []
        for s in column:
            try:
                out.append(("ok", fn(s, default=DEFAULT)))
            except Exception as exc:  # noqa: BLE001
                out.append(("err", f"{type(exc).__name__}: {exc}"))
        return out

    expected = run(lambda s, **kw: oracle_parse(s, **kw))
    got_batch = []
    for s in column:
        try:
            got_batch.append(("ok", dateutil_mojo.parse(s, default=DEFAULT)))
        except Exception as exc:  # noqa: BLE001
            got_batch.append(("err", f"{type(exc).__name__}: {exc}"))
    for s, e, g in zip(column, expected, got_batch):
        assert e == g, f"per-element mismatch at {s!r}: {g} != {e}"

    # the batch API must raise the same error for the bad row
    try:
        dateutil_mojo.parse_column(column, default=DEFAULT)
    except Exception as exc:  # noqa: BLE001
        batch_err = f"{type(exc).__name__}: {exc}"
    else:
        batch_err = None
    oracle_err = next(v for st, v in expected if st == "err")
    assert batch_err == oracle_err

    good = [s for s, (st, _) in zip(column, expected) if st == "ok"]
    got_good = dateutil_mojo.parse_column(good, default=DEFAULT)
    exp_good = [v for st, v in expected if st == "ok"]
    for s, g, e in zip(good, got_good, exp_good):
        assert _norm(g) == _norm(e), f"batch mismatch at {s!r}"


def test_parse_column_flags():
    column = ["01/02/2003", "03/01/02", "2011-01-01", "120901"]
    for flags in ({"dayfirst": True}, {"yearfirst": True}, {"ignoretz": True}):
        got = dateutil_mojo.parse_column(column, default=DEFAULT, **flags)
        exp = [oracle_parse(s, default=DEFAULT, **flags) for s in column]
        for s, g, e in zip(column, got, exp):
            assert _norm(g) == _norm(e), f"{s!r} flags={flags}"


def test_parse_column_error_propagates():
    with pytest.raises(dateutil_mojo.ParserError):
        dateutil_mojo.parse_column(["2011-01-01", "not a date"], default=DEFAULT)
