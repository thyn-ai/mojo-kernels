"""Clean-room reimplementation of ``dateutil.parser.parse`` semantics.

Written purely from black-box observation of python-dateutil 2.9.0.post0
(the oracle): its documented behavior plus a large battery of probed
input/output pairs, including the oracle's own test-corpus inputs. No
oracle source was read or adapted.

Mirrors the oracle's observable contract: tokenization into number-ish
(``0-9 . : + - / ,``) and alpha runs; ISO 8601, RFC 2822, US/EU numeric and
named-month formats; h/m/s suffix times (``01h02m03s``); fraction
cascading; am/pm gates; weekdays; jump words; ordinal suffixes; ``of``
year marking; tz offsets/names incl. POSIX sign flip; ``dayfirst`` /
``yearfirst``; a sliding 2-digit-year pivot; ``default`` filling;
``ignoretz``; ``tzinfos``; ``fuzzy`` / ``fuzzy_with_tokens``; and the
oracle's exact exception classes and messages.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, tzinfo

from ._tz import local_tznames, tzlocal, tzoffset, tzutc

__all__ = ["ParserError", "UnknownTimezoneWarning", "parse"]

_NUMWORD = frozenset("0123456789.:+-/,")
_DIGITS = frozenset("0123456789")

# Recognized tz names, gated case-sensitively exactly as observed.
_UTC_KNOWN = frozenset({"UTC", "GMT", "UT"})  # 'Z'/'z' handled by len-1 rule
_UTC_ZERO = frozenset({"UTC", "GMT"})  # these (and Z/z) build tzutc; UT warns

_JUMP = frozenset({"and", "at", "of", "on", "ad", "t"})
_AMPM = {"am": 0, "a": 0, "pm": 1, "p": 1}
_HMS = {"h": 0, "m": 1, "s": 2}

_MONTHS: dict[str, int] = {}
for _i, _full in enumerate(
    [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    ]
):
    _MONTHS[_full] = _i + 1
    _MONTHS[_full[:3]] = _i + 1
_MONTHS["sept"] = 9

_WEEKDAYS: dict[str, int] = {}
for _i, _full in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
):
    _WEEKDAYS[_full] = _i
    _WEEKDAYS[_full[:3]] = _i

_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


class ParserError(ValueError):
    """Mirrors ``dateutil.parser.ParserError`` (a ValueError subclass)."""


class UnknownTimezoneWarning(RuntimeWarning):
    """Mirrors ``dateutil.parser.UnknownTimezoneWarning``."""


class _Tok:
    __slots__ = ("kind", "text", "start", "end")

    def __init__(self, kind: int, text: str, start: int, end: int) -> None:
        self.kind = kind  # 0 = numword run, 1 = alpha run, 2 = other, 3 = consumed
        self.text = text
        self.start = start
        self.end = end


def _tokenize(s: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in _NUMWORD:
            j = i + 1
            while j < n and s[j] in _NUMWORD:
                j += 1
            toks.append(_Tok(0, s[i:j], i, j))
            i = j
        elif c.isalpha():
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            toks.append(_Tok(1, s[i:j], i, j))
            i = j
        else:
            toks.append(_Tok(2, c, i, i + 1))
            i += 1
    return toks


def _year_pivot(y: int) -> int:
    """2-digit year pivot, sliding with the current year like the oracle:
    map into [now - 50, now + 50) (2026 -> 00..75 are 20xx, 76..99 19xx)."""
    if not 0 <= y <= 99:
        return y
    candidate = 2000 + y
    if candidate >= datetime.now().year + 50:
        candidate -= 100
    return candidate


def _split_float(text: str):
    """Split '123.45' / '123,45' into ('123', '45'); None if not that shape."""
    for sep in (".", ","):
        if sep in text:
            head, _, tail = text.partition(sep)
            if head and all(c in _DIGITS for c in head) and all(
                c in _DIGITS for c in tail
            ):
                return head, tail
            return None
    return None


def _parse_offset_body(body: str):
    """Parse unsigned offset body 'HH', 'H', 'HHMM', 'HH:MM', 'H:M' -> seconds."""
    if not body:
        return None
    if ":" in body:
        parts = body.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        if not all(c in _DIGITS for c in parts[0]) or not all(
            c in _DIGITS for c in parts[1]
        ):
            return None
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    if not all(c in _DIGITS for c in body):
        return None
    if len(body) <= 2:
        return int(body) * 3600
    if len(body) == 4:
        return int(body[:2]) * 3600 + int(body[2:]) * 60
    return None


class _State:
    __slots__ = (
        "timestr", "dayfirst", "yearfirst", "ymd", "month_name",
        "month_tokidx", "hour", "minute", "second", "microsecond",
        "weekday", "tzname", "tzname_known", "tzoffset", "date_seen",
        "date_complete", "day_explicit", "consumed", "forced",
        "of_next_year", "hms_next",
    )

    def __init__(self, timestr: str, dayfirst: bool, yearfirst: bool) -> None:
        self.timestr = timestr
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self.ymd: list[tuple[int, int, int]] = []  # (value, ndigits, token idx)
        self.forced: tuple[int | None, int | None, int | None] | None = None
        self.month_name: int | None = None
        self.month_tokidx: int | None = None
        self.hour: int | None = None
        self.minute: int | None = None
        self.second: int | None = None
        self.microsecond: int | None = None
        self.weekday: int | None = None
        self.tzname: str | None = None
        self.tzname_known = False
        self.tzoffset: int | None = None  # seconds east of UTC
        self.date_seen = False
        self.date_complete = False
        self.day_explicit = False
        self.consumed: list[tuple[int, int]] = []
        self.of_next_year = False
        self.hms_next = -1  # -1 none, 0 after h, 1 after m, 2 after s

    def mark(self, tok: _Tok) -> None:
        self.consumed.append((tok.start, tok.end))


class _Parser:
    def __init__(
        self,
        timestr: str,
        *,
        dayfirst: bool,
        yearfirst: bool,
        tzinfos,
        fuzzy: bool,
    ) -> None:
        self.st = _State(timestr, dayfirst, yearfirst)
        self.tzinfos = tzinfos
        self.fuzzy = fuzzy
        self.toks = _tokenize(timestr)
        self.skipped: list[tuple[int, int]] = []
        self.extra_kept = 0
        self.ordinal_spans: set[tuple[int, int]] = set()

    # ------------------------------------------------------------ utilities

    def _fail(self) -> None:
        raise ParserError(f"Unknown string format: {self.st.timestr}")

    def _unconsumable(self, i: int) -> None:
        if self.fuzzy:
            self.skipped.append((self.toks[i].start, self.toks[i].end))
        else:
            self._fail()

    def _hard(self) -> None:
        """Structural number-token failure: a hard error even in fuzzy
        mode (malformed numerics are not skippable in the oracle)."""
        raise ParserError(f"Unknown string format: {self.st.timestr}")

    def _complete_now(self) -> bool:
        st = self.st
        return st.date_complete or (
            st.month_name is not None and len(st.ymd) >= 2
        )

    def _is_ampm_tok(self, i: int) -> bool:
        toks = self.toks
        return (
            i < len(toks)
            and toks[i].kind == 1
            and toks[i].text.lower() in _AMPM
        )

    def _consume_ampm(self, i: int) -> int:
        """Consume an am/pm token at i (plus a '. m [ . ]' composite).

        Returns the ampm value (0=am, 1=pm).
        """
        st, toks = self.st, self.toks
        tok = toks[i]
        ampm = _AMPM[tok.text.lower()]
        st.mark(tok)
        toks[i].kind = 3
        if tok.text.lower() in ("a", "p") and i + 2 < len(toks):
            # "p.m." composite: only a lowercase 'm' is consumed; an
            # uppercase 'M' is an (unknown) tz name to the oracle.
            if toks[i + 1].text == "." and toks[i + 2].text == "m":
                for j in (i + 1, i + 2):
                    st.mark(toks[j])
                    toks[j].kind = 3
                if i + 3 < len(toks) and toks[i + 3].text == ".":
                    st.mark(toks[i + 3])
                    toks[i + 3].kind = 3
        return ampm

    def _apply_ampm(self, ampm: int) -> None:
        st = self.st
        if ampm == 0:
            if st.hour == 12:
                st.hour = 0
        elif st.hour is not None and st.hour < 12:
            st.hour += 12

    # ------------------------------------------------------ number handling

    def _handle_number(self, i: int) -> None:
        st = self.st
        tok = self.toks[i]
        text = tok.text

        # Lone separators between date parts ("2011 - 01", "Sep. 8").
        if text in ("-", ".", ","):
            if self.fuzzy:
                self.skipped.append((tok.start, tok.end))
            else:
                st.mark(tok)
            return
        while text.startswith("."):
            text = text[1:]
        comma_pos = -1
        if text.endswith(","):
            comma_pos = tok.end - 1
            text = text[:-1]
        if not text:
            st.mark(tok)
            return

        if "," in text:
            self._handle_comma_number(i, text)
        else:
            self._number_fragment(i, text)
        if comma_pos >= 0 and self.fuzzy:
            # A trailing comma after a complete date is a skipped token
            # ("2011-01-01, 12:30"); otherwise it is a kept separator.
            if st.date_complete:
                self.skipped.append((comma_pos, comma_pos + 1))
            else:
                self.extra_kept += 1

    def _handle_comma_number(self, i: int, text: str) -> None:
        st = self.st
        parts = text.split(",")
        if (
            len(parts) == 2
            and ":" in parts[0]
            and parts[1]
            and all(c in _DIGITS for c in parts[1])
            and all(c in "0123456789:." for c in parts[0])
        ):
            self._parse_time_token(i, parts[0], parts[1])
            return
        if "-" in text or "/" in text:
            self._hard()
            return
        if 2 <= len(parts) <= 3 and all(
            p and all(c in _DIGITS for c in p) for p in parts
        ):
            if len(parts) == 2:
                # A comma is a date separator only when it yields a valid
                # month ("1,02" -> Jan 2); a zero month is a hard error;
                # otherwise it is a decimal comma ("14,5" -> day 14).
                a, b = int(parts[0]), int(parts[1])
                na, nb = len(parts[0]), len(parts[1])
                if na == 4 or a > 31:
                    month = b
                elif nb == 4 or b > 31:
                    month = a
                else:
                    month = b if st.dayfirst else a
                if month == 0:
                    self._bad_month(i, month)
                    return
                if not 1 <= month <= 12:
                    self._handle_float(i, text.replace(",", ".", 1))
                    return
                self._date_group(i, [a, b], [na, nb])
                return
            self._comma_triple(i, parts)
            return
        # Otherwise each comma part is processed on its own ("1,2:30").
        st.mark(self.toks[i])
        for part in parts:
            self._number_fragment(i, part)

    def _bad_month(self, i: int, month: int) -> None:
        if self.fuzzy:
            self.skipped.append((self.toks[i].start, self.toks[i].end))
            return
        raise ParserError(
            f"bad month number {month}; must be 1-12: {self.st.timestr}"
        )

    def _comma_triple(self, i: int, parts: list[str]) -> None:
        """'MM,DD,YY'-style comma triples: the oracle's special rule.

        - 1-digit first part: month=p1, day=p2, and p3 is the year only
          when p2 was written with one digit ("1,2,45" -> 2045-01-02,
          "9,20,45" -> Sep 20, year from default).
        - 2-digit first part: month=p1 (invalid -> "bad month number");
          p2 is discarded; p3 is the day when <= 31, else the year
          ("10,20,13" -> Oct 13; "10,20,45" -> Oct, year 2045).
        """
        st = self.st
        p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
        n1, n2 = len(parts[0]), len(parts[1])
        y = m = d = None
        if n1 == 1:
            m, d = p1, p2
            if n2 == 1:
                y = _year_pivot(p3) if p3 <= 99 else p3
        else:
            if not 1 <= p1 <= 12:
                self._bad_month(i, p1)
                return
            m = p1
            if p3 <= 31:
                d = p3
            else:
                y = _year_pivot(p3) if p3 <= 99 else p3
        st.mark(self.toks[i])
        st.forced = (y, m, d)
        st.date_seen = True
        st.date_complete = True
        st.day_explicit = d is not None

    def _number_fragment(self, i: int, text: str) -> None:
        st = self.st
        if not text:
            st.mark(self.toks[i])
            return
        if text[0] in "+-":
            self._handle_signed(i, text)
            return
        if ":" in text:
            self._parse_time_token(i, text, None)
            return
        # After a complete date, a signed suffix reads as compact time with
        # a tz offset ("104941+0300", "104941.5-0300").
        if st.date_seen and st.hour is None:
            for j in range(1, len(text)):
                if text[j] in "+-" and (text[j] == "+" or "." in text[:j]):
                    left, right = text[:j], text[j:]
                    offset = _parse_offset_body(right[1:])
                    if offset is None:
                        self._unconsumable(i)
                        return
                    if "." in left:
                        self._handle_float(i, left)
                    else:
                        if not all(c in _DIGITS for c in left):
                            self._unconsumable(i)
                            return
                        self._digits_value(i, int(left), len(left))
                    if st.hour is None:
                        self._unconsumable(i)
                        return
                    st.tzoffset = offset if right[0] == "+" else -offset
                    return
        if "-" in text or "/" in text:
            has_dash = "-" in text
            raw = [p for p in text.replace("/", "-").split("-") if p != ""]
            if not raw or any(not all(c in _DIGITS for c in p) for p in raw):
                self._hard()
                return
            if st.date_seen:
                # Compact time with tz offsets ("2012-02-02" -> 20:12 -02:00).
                if not has_dash or st.hour is not None or len(raw[0]) not in (1, 2, 4, 6):
                    self._hard()
                    return
                self._compact_time_with_offsets(i, raw)
                return
            if len(raw) == 4:
                # "2011-01-01/02": date triple plus a time-from-digits part.
                if st.hour is not None:
                    self._hard()
                st.mark(self.toks[i])
                for v, ndv in zip(raw[:3], [len(x) for x in raw[:3]]):
                    st.ymd.append((int(v), len(v), i))
                st.date_seen = True
                st.date_complete = True
                st.day_explicit = True
                last = raw[3]
                if len(last) <= 2:
                    st.hour = int(last)
                elif len(last) == 4:
                    st.hour, st.minute = int(last[:2]), int(last[2:])
                elif len(last) == 6:
                    st.hour, st.minute, st.second = (
                        int(last[:2]), int(last[2:4]), int(last[4:])
                    )
                    st.microsecond = 0
                else:
                    self._hard()
                return
            if len(raw) > 4:
                self._hard()
                return
            self._date_group(i, [int(p) for p in raw], [len(p) for p in raw])
            return
        if "." in text:
            if text.endswith("."):
                text = text[:-1]
            if "." not in text:
                if not all(c in _DIGITS for c in text):
                    self._unconsumable(i)
                    return
                self._digits_value(i, int(text), len(text))
                return
            dot_parts = text.split(".")
            if len(dot_parts) == 3 and all(
                p and all(c in _DIGITS for c in p) for p in dot_parts
            ):
                self._date_group(
                    i, [int(p) for p in dot_parts], [len(p) for p in dot_parts]
                )
                return
            if len(dot_parts) >= 3:
                self._unconsumable(i)
                return
            if (
                len(dot_parts) == 2
                and st.month_name is not None
                and all(p and all(c in _DIGITS for c in p) for p in dot_parts)
                and (
                    int(dot_parts[0]) > 31
                    or int(dot_parts[1]) > 31
                    or len(dot_parts[0]) == 4
                    or len(dot_parts[1]) == 4
                )
            ):
                # "25.2003" with a named month: day + year entries.
                st.mark(self.toks[i])
                st.ymd.append((int(dot_parts[0]), len(dot_parts[0]), i))
                st.ymd.append((int(dot_parts[1]), len(dot_parts[1]), i))
                st.date_seen = True
                return
            self._handle_float(i, text)
            return
        if not all(c in _DIGITS for c in text):
            self._unconsumable(i)
            return
        self._digits_value(i, int(text), len(text))

    def _hms_suffix(self, i: int) -> int:
        """h/m/s unit if token i+1 is a single h/m/s letter, else -1."""
        toks = self.toks
        if i + 1 < len(toks) and toks[i + 1].kind == 1:
            return _HMS.get(toks[i + 1].text.lower(), -1)
        return -1

    def _consume_suffix(self, i: int) -> None:
        st, toks = self.st, self.toks
        st.mark(toks[i + 1])
        toks[i + 1].kind = 3

    def _assign_hms(self, slot: int, value: int, frac: str | None) -> None:
        st = self.st
        if slot == 0:
            st.hour = value
            if frac:
                st.minute = int(float("0." + frac) * 60)
        elif slot == 1:
            st.minute = value
            if frac:
                st.second = int(float("0." + frac) * 60)
        else:
            st.second = value
            st.microsecond = int((frac + "000000")[:6]) if frac else 0

    def _digits_value(self, i: int, value: int, nd: int) -> None:
        """A bare unsigned digit run (or a 1-part date group)."""
        st = self.st
        st.mark(self.toks[i])

        # 'of' year marking ("July of 76").
        if st.of_next_year:
            st.of_next_year = False
            st.ymd.append((_year_pivot(value) if nd <= 2 else value, 4, i))
            return

        # h/m/s suffix: "10h", "10.5h", "01h02m03s".
        slot = self._hms_suffix(i)
        if slot >= 0:
            self._consume_suffix(i)
            self._assign_hms(slot, value, None)
            st.hms_next = slot + 1
            return

        # Bare number continuing an h/m/s chain: "01h02" -> minute.
        if 0 <= st.hms_next <= 2:
            self._assign_hms(st.hms_next, value, None)
            st.hms_next += 1
            return
        if st.hms_next == 3:
            return  # after seconds: consumed and dropped ("10h13s 04")

        # am/pm lookahead: any bare number followed by am/pm is the hour
        # (overwriting any earlier hour: "2:15 PM 1973 A" -> hour 1973).
        if self._is_ampm_tok(i + 1):
            ampm = self._consume_ampm(i + 1)
            st.hour = value
            if value <= 12:
                self._apply_ampm(ampm)
            return

        if self._complete_now():
            if st.hour is None:
                if nd <= 2:
                    st.hour = value
                    return
                if nd == 4:
                    st.hour, st.minute = value // 100, value % 100
                    return
                if nd == 6:
                    st.hour, st.minute, st.second = (
                        value // 10000,
                        (value // 100) % 100,
                        value % 100,
                    )
                    st.microsecond = 0
                    return
            self._hard()
            return

        if nd == 8:
            y, rest = value // 10000, value % 10000
            st.ymd.extend(
                [(y, 4, i), (rest // 100, 2, i), (rest % 100, 2, i)]
            )
            st.date_seen = True
            st.date_complete = True
            st.day_explicit = True
            return
        if nd == 12 or nd == 14:
            if nd == 12:  # YYYYMMDDHHMM
                st.ymd.extend(
                    [
                        (value // 10**8, 4, i),
                        ((value // 10**6) % 100, 2, i),
                        ((value // 10**4) % 100, 2, i),
                    ]
                )
                st.hour = (value // 100) % 100
                st.minute = value % 100
            else:  # YYYYMMDDHHMMSS
                st.ymd.extend(
                    [
                        (value // 10**10, 4, i),
                        ((value // 10**8) % 100, 2, i),
                        ((value // 10**6) % 100, 2, i),
                    ]
                )
                st.hour = (value // 10**4) % 100
                st.minute = (value // 100) % 100
                st.second = value % 100
                # microseconds stay unset: the default fills them in
            st.date_seen = True
            st.date_complete = True
            st.day_explicit = True
            return
        if nd == 6:
            st.ymd.extend(
                [(value // 10000, 2, i), ((value // 100) % 100, 2, i), (value % 100, 2, i)]
            )
            st.date_seen = True
            st.date_complete = True
            st.day_explicit = True
            return
        if nd >= 3:
            # 3-5 digit and 7+ digit bare numbers act as years (validated at
            # datetime construction, like the oracle; >=11 digits overflow
            # there with a raw OverflowError).
            st.ymd.append((value, 4 if nd >= 4 else nd, i))
            if len(st.ymd) >= 3:
                st.date_seen = True
                st.date_complete = True
            return
        # 1-2 digits: day/month/year candidate.
        st.ymd.append((value, nd, i))
        if len(st.ymd) >= 3:
            st.date_seen = True
            st.date_complete = True

    def _handle_float(self, i: int, text: str) -> None:
        st = self.st
        pair = _split_float(text)
        if pair is None:
            self._hard()
            return
        intpart, fracpart = pair
        int_d, frac_d = len(intpart), len(fracpart)
        value = int(intpart)

        # h/m/s suffix with fraction: "10.5h" -> 10:30.
        slot = self._hms_suffix(i)
        if slot >= 0:
            st.mark(self.toks[i])
            self._consume_suffix(i)
            self._assign_hms(slot, value, fracpart)
            st.hms_next = slot + 1
            return

        # Bare float continuing an h/m/s chain ("10 h 36.5" -> 10:36:30).
        if 0 <= st.hms_next <= 2:
            st.mark(self.toks[i])
            self._assign_hms(st.hms_next, value, fracpart)
            st.hms_next += 1
            return
        if st.hms_next == 3:
            st.mark(self.toks[i])
            return

        if self._complete_now():
            # After a full date only 6-int-digit floats read as HHMMSS.f.
            if int_d == 6 and st.hour is None:
                st.mark(self.toks[i])
                st.hour = value // 10000
                st.minute = (value // 100) % 100
                st.second = value % 100
                st.microsecond = int((fracpart + "000000")[:6])
                return
            self._hard()
            return

        if int_d == 6:
            # HHMMSS.fraction (date fields come from the default).
            st.mark(self.toks[i])
            st.hour = value // 10000
            st.minute = (value // 100) % 100
            st.second = value % 100
            st.microsecond = int((fracpart + "000000")[:6])
            return
        if int_d >= 7:
            st.mark(self.toks[i])
            st.ymd.append((value, 4, i))
            return
        total = int_d + frac_d
        if total in (2, 3, 4, 6):
            st.mark(self.toks[i])
            st.ymd.append((value, 4 if value > 99 else int_d, i))
            if len(st.ymd) >= 3:
                st.date_seen = True
                st.date_complete = True
            return
        # total == 5 or >= 7 (non-HHMMSS): rejected.
        self._hard()

    def _handle_signed(self, i: int, text: str) -> None:
        """Pure signed offset token: '+02:00', '-0530', '+2'.

        With no time parsed yet a '-' acts as a date separator instead
        ("8-Jan-2025" -> ... "-2025" -> year 2025; "Sep-25-2003").
        """
        st = self.st
        sign = -1 if text[0] == "-" else 1
        body = text[1:]
        if st.hour is None:
            if body and all(c in _DIGITS for c in body):
                if text[0] == "+":
                    # '+' is offset-only; fuzzy recovers it as a number.
                    if not self.fuzzy:
                        self._hard()
                    self._digits_value(i, int(body), len(body))
                    return
                if self._complete_now() or st.month_name is not None:
                    self._digits_value(i, int(body), len(body))
                    return
                self._hard()
            if ":" in body:
                # "-05:30" is a time, not an offset, with no hour parsed yet;
                # the '+' form is recovered only in fuzzy mode.
                if text[0] == "-" or self.fuzzy:
                    self._parse_time_token(i, body, None)
                    return
            if text[0] == "-" and body and ("-" in body or "/" in body):
                sep = "-" if "-" in body else "/"
                raw = [p for p in body.split(sep) if p != ""]
                if raw and all(all(c in _DIGITS for c in p) for p in raw):
                    if len(raw) <= 3:
                        self._date_group(
                            i, [int(p) for p in raw], [len(p) for p in raw]
                        )
                        return
            self._hard()
        offset = _parse_offset_body(body)
        if offset is None:
            self._hard()
        st.mark(self.toks[i])
        offset *= sign
        if (
            st.tzname is not None
            and (st.tzname in _UTC_ZERO or st.tzname.lower() == "z")
            and st.tzoffset is None
        ):
            # "UTC +02:00" (spaced): the UTC-zone name wins, offset dropped.
            if self.fuzzy:
                self.extra_kept += 1
            return
        st.tzoffset = offset

    def _compact_time_with_offsets(self, i: int, raw: list[str]) -> None:
        """'2012-02-02' after a complete date -> 20:12 with -02:00 tz."""
        st = self.st
        first = raw[0]
        st.mark(self.toks[i])
        if len(first) <= 2:
            st.hour = int(first)
        elif len(first) == 4:
            st.hour, st.minute = int(first[:2]), int(first[2:])
        else:  # 6
            st.hour, st.minute, st.second = (
                int(first[:2]),
                int(first[2:4]),
                int(first[4:]),
            )
            st.microsecond = 0
        for part in raw[1:]:
            offset = _parse_offset_body(part)
            if offset is None:
                self._unconsumable(i)
                return
            st.tzoffset = -offset

    def _parse_time_token(self, i: int, text: str, comma_frac: str | None) -> None:
        st = self.st
        # Split off a tz offset suffix (sign at position > 0).
        time_part, off_part = text, None
        for j in range(1, len(text)):
            if text[j] in "+-":
                time_part, off_part = text[:j], text[j:]
                break
        groups = time_part.split(":")
        if len(groups) < 2 or len(groups) > 4 or any(g == "" for g in groups[:-1]):
            self._hard()
        if groups and groups[-1] == "":
            # "12:30:" — a trailing colon is recoverable in fuzzy mode.
            if not self.fuzzy:
                self._hard()
            self.skipped.append((self.toks[i].end - 1, self.toks[i].end))
            groups = groups[:-1]
        parsed: list[tuple[int, str | None]] = []
        tail_numbers: list[int] = []
        for g in groups:
            if g.count(".") == 2:
                # "12:30.5.6" — the dot-tail reads as month/day numbers.
                g0, t1, t2 = g.split(".")
                if not (
                    all(c in _DIGITS for c in g0)
                    and all(c in _DIGITS for c in t1)
                    and all(c in _DIGITS for c in t2)
                ):
                    self._hard()
                parsed.append((int(g0), None))
                tail_numbers.extend([int(t1), int(t2)])
                continue
            fp = _split_float(g)
            if fp is not None:
                parsed.append((int(fp[0]), fp[1]))
            elif all(c in _DIGITS for c in g) or (
                g.endswith(".") and all(c in _DIGITS for c in g[:-1])
            ):
                parsed.append((int(g.rstrip(".")), None))
            else:
                self._hard()
        st.mark(self.toks[i])
        st.hms_next = -1
        st.hour, _ = parsed[0]
        st.minute, minute_frac = parsed[1]
        if len(parsed) == 2:
            frac = comma_frac if comma_frac is not None else minute_frac
            if frac:
                # Fraction of a minute cascades to seconds only.
                st.second = int(float("0." + frac) * 60)
        else:
            sec, sec_frac = parsed[2]
            st.second = sec
            frac = comma_frac if comma_frac is not None else sec_frac
            st.microsecond = int((frac + "000000")[:6]) if frac else 0
        # A fourth colon part ("1:2:3:4") is a fuzzy-recovered day/year number.
        if len(parsed) == 4:
            if not self.fuzzy or self._complete_now():
                self._hard()
            self._digits_value(i, parsed[3][0], len(groups[3]))
        for tv in tail_numbers:
            self._digits_value(i, tv, len(str(tv)))
        if off_part:
            sign = -1 if off_part[0] == "-" else 1
            offset = _parse_offset_body(off_part[1:])
            if offset is None:
                # An offset-shaped suffix that fails to parse is a hard
                # error even in fuzzy mode ("12:00:00+02:30:15").
                raise ParserError(f"Unknown string format: {self.st.timestr}")
            st.tzoffset = sign * offset

    def _date_group(self, i: int, values: list[int], ndigits: list[int]) -> None:
        st = self.st
        if len(values) == 1:
            self._digits_value(i, values[0], ndigits[0])
            return
        if len(values) == 2:
            a, b = values
            na, nb = ndigits
            a_year = na == 4 or a > 31
            b_year = nb == 4 or b > 31
            # The non-year part of a 2-part date is at most 2 digits
            # ("2011-001" is rejected).
            if (a_year and nb > 2) or (b_year and na > 2):
                self._hard()
                return
            st.mark(self.toks[i])
            st.ymd.append((a, na, i))
            st.ymd.append((b, nb, i))
            st.date_seen = True
            if not a_year and not b_year:
                st.day_explicit = True
            return
        # 3 parts
        st.mark(self.toks[i])
        for v, nd in zip(values, ndigits):
            st.ymd.append((v, nd, i))
        st.date_seen = True
        st.date_complete = True
        st.day_explicit = True

    # ------------------------------------------------------- alpha handling

    def _handle_alpha(self, i: int) -> None:
        st, toks = self.st, self.toks
        tok = toks[i]
        w = tok.text
        lw = w.lower()
        st.of_next_year = False

        # Ordinal suffixes after a number ("1st", "21st").
        if lw in _ORDINAL_SUFFIXES and i > 0 and toks[i - 1].kind == 0:
            st.mark(tok)
            self.ordinal_spans.add((tok.start, tok.end))
            return

        if lw in _MONTHS and not st.date_complete and st.month_name is None:
            st.month_name = _MONTHS[lw]
            st.month_tokidx = i
            st.mark(tok)
            if i + 1 < len(toks) and toks[i + 1].text == ".":
                st.mark(toks[i + 1])
                toks[i + 1].kind = 3
            return

        if lw in _WEEKDAYS:
            st.weekday = _WEEKDAYS[lw]
            st.mark(tok)
            if i + 1 < len(toks) and toks[i + 1].text == ".":
                st.mark(toks[i + 1])
                toks[i + 1].kind = 3
            return

        if lw in _AMPM:
            # am/pm after an explicitly-set hour; hour > 12 is an error
            # ("13:00 PM"); bare "13 PM" is consumed by the number branch.
            if st.hour is None or st.hour > 12:
                self._unconsumable(i)
                return
            ampm = self._consume_ampm(i)
            self._apply_ampm(ampm)
            return

        if lw in _JUMP:
            if self.fuzzy:
                self.skipped.append((tok.start, tok.end))
            else:
                st.mark(tok)
            if lw == "of":
                st.of_next_year = True
            else:
                st.of_next_year = False
            return

        # Timezone names (gated on an already-parsed time).
        is_z = len(w) == 1 and lw == "z"
        if st.hour is not None and (
            w in _UTC_KNOWN or is_z or w in local_tznames()
        ):
            st.tzname = w
            st.tzname_known = True
            st.mark(tok)
            self._attached_offset_after_name(i)
            return

        if (
            st.hour is not None
            and self.tzinfos is not None
            and not callable(self.tzinfos)
            and w in self.tzinfos
        ):
            st.tzname = w
            st.tzname_known = True
            st.mark(tok)
            self._attached_offset_after_name(i)
            return

        # Unknown ALL-UPPERCASE name (<=5 chars): consumed as a tz name,
        # warned about (and dropped) at build unless tzinfos resolves it.
        # An attached signed offset flips POSIX-style, exactly as after a
        # recognized name: the oracle parses "EST+2" as tzoffset('EST', -7200).
        if (
            st.hour is not None
            and w.isalpha()
            and w.upper() == w
            and 1 <= len(w) <= 5
        ):
            st.tzname = w
            st.tzname_known = False
            st.mark(tok)
            if self.fuzzy:
                self.extra_kept += 1
            self._attached_offset_after_name(i)
            return

        self._unconsumable(i)

    def _attached_signed(self, i: int) -> bool:
        toks = self.toks
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            return (
                nxt.kind == 0
                and nxt.text[0] in "+-"
                and nxt.start == toks[i].end
            )
        return False

    def _attached_offset_after_name(self, i: int) -> None:
        """POSIX-style 'UTC+02:00': attached offset after a known tz name.

        The sign is flipped ('UTC+02:00' means 2 hours west of UTC).
        """
        st, toks = self.st, self.toks
        if self._attached_signed(i):
            nxt = toks[i + 1]
            sign = -1 if nxt.text[0] == "-" else 1
            offset = _parse_offset_body(nxt.text[1:])
            if offset is None:
                self._unconsumable(i + 1)
                return
            st.tzoffset = -sign * offset
            if st.tzname in _UTC_ZERO or (st.tzname or "").lower() == "z":
                # The oracle drops a UTC-zone name here: "UTC+2" is *not* UTC,
                # it is a numeric offset ("GMT+3" means "my time +3 is GMT").
                st.tzname = None
                st.tzname_known = False
            st.mark(nxt)
            nxt.kind = 3

    # ------------------------------------------------------------ main loop

    def run(self) -> None:
        for i, tok in enumerate(self.toks):
            if tok.kind == 3:
                continue
            if tok.kind == 0:
                self._handle_number(i)
            elif tok.kind == 1:
                self._handle_alpha(i)
            else:
                # Apostrophe year marker: "'96" -> 1996.
                if tok.text == "'" and i + 1 < len(self.toks):
                    nxt = self.toks[i + 1]
                    if (
                        nxt.kind == 0
                        and len(nxt.text) == 2
                        and all(c in _DIGITS for c in nxt.text)
                    ):
                        self.st.mark(tok)
                        self.st.ymd.append((_year_pivot(int(nxt.text)), 4, i + 1))
                        nxt.kind = 3
                        continue
                self._unconsumable(i)


def _resolve_ymd(st: _State):
    """Resolve collected numeric tokens + named month to (year, month, day)."""
    if st.forced is not None:
        return st.forced
    entries = st.ymd
    month = st.month_name
    year = day = None

    if month is not None:
        nums = list(entries)

        def named_year(v: int, nd: int) -> int:
            return _year_pivot(v) if nd <= 2 else v

        if len(nums) == 1:
            v, nd, _ = nums[0]
            if v > 31:
                year = v if v > 99 else _year_pivot(v)
            else:
                day = v
        elif len(nums) == 2:
            (v1, nd1, t1), (v2, nd2, t2) = nums
            y1, y2 = v1 > 31, v2 > 31
            if y1 and not y2:
                year, day = (v1 if v1 > 99 else _year_pivot(v1)), v2
            elif y2 and not y1:
                year, day = (v2 if v2 > 99 else _year_pivot(v2)), v1
            elif t1 == t2:
                # Same composite token ("2015-15-May", "December.0031.30"):
                # first is the year, second the day.
                year, day = named_year(v1, nd1), v2
            else:
                # Otherwise: with yearfirst a number before the month is the
                # year; else, if any number follows the month the first is
                # the day, and when both precede it the last one is
                # ("6 AD May 19" -> day 6; "8 25 Jan" -> day 25).
                mi = st.month_tokidx
                if st.yearfirst and t1 < mi:
                    year, day = named_year(v1, nd1), v2
                elif t1 > mi or t2 > mi:
                    day, year = v1, named_year(v2, nd2)
                else:
                    day, year = v2, named_year(v1, nd1)
        elif len(nums) > 2:
            raise ParserError(f"Unknown string format: {st.timestr}")
        return year, month, day

    if len(entries) == 1:
        v, nd, _ = entries[0]
        if nd >= 3 or v > 99:
            year = v
        elif v > 31:
            year = _year_pivot(v)
        else:
            day = v
        return year, None, day

    if len(entries) == 2:
        (v1, nd1, t1), (v2, nd2, t2) = entries
        same_tok = t1 == t2
        if (nd1 == 4 and same_tok) or v1 > 31:
            year, month = (v1 if (nd1 == 4 and same_tok) else _year_pivot(v1)), v2
            if not 1 <= month <= 12:
                raise ParserError(
                    f"bad month number {month}; must be 1-12: {st.timestr}"
                )
        elif (nd2 == 4 and same_tok) or v2 > 31:
            year, month = (v2 if (nd2 == 4 and same_tok) else _year_pivot(v2)), v1
            if not 1 <= month <= 12:
                raise ParserError(
                    f"bad month number {month}; must be 1-12: {st.timestr}"
                )
        elif st.dayfirst:
            day, month = v1, v2
        else:
            month, day = v1, v2
        return year, month, day

    if len(entries) == 3:
        (v1, nd1, t1), (v2, nd2, t2), (v3, nd3, t3) = entries
        same_tok = t1 == t2 == t3
        if (nd1 == 4 and same_tok) or v1 > 31:
            # Year first: dayfirst swaps the pair, but an invalid resulting
            # month falls back to the default order. Value-based years pivot;
            # only an explicit 4-digit year token stays as written.
            y = v1 if (nd1 == 4 and same_tok) else _year_pivot(v1)
            if st.dayfirst:
                m, d = v3, v2
                if m > 12 and d <= 12:
                    m, d = v2, v3
            else:
                m, d = v2, v3
        elif (nd3 == 4 and same_tok) or v3 > 31:
            # Year last: a >12 value forces the day slot.
            y = v3 if (nd3 == 4 and same_tok) else _year_pivot(v3)
            if v1 > 12:
                m, d = v2, v1
            elif v2 > 12:
                m, d = v1, v2
            elif st.dayfirst:
                m, d = v2, v1
            else:
                m, d = v1, v2
        elif st.yearfirst:
            y = _year_pivot(v1)
            if st.dayfirst:
                m, d = v3, v2
                if m > 12 and d <= 12:
                    m, d = v2, v3
            else:
                m, d = v2, v3
        elif v1 > 12:
            d, m, y = v1, v2, _year_pivot(v3)
        elif v2 > 12:
            m, d, y = v1, v2, _year_pivot(v3)
        elif st.dayfirst:
            d, m, y = v1, v2, _year_pivot(v3)
        else:
            m, d, y = v1, v2, _year_pivot(v3)
        return y, m, d

    if len(entries) > 3:
        raise ParserError(f"Unknown string format: {st.timestr}")

    return None, month, None


def _build_tzaware(st: _State, tzinfos):
    """Reproduce the oracle's tz construction from (tzname, tzoffset)."""
    tzname, tzoffset_s = st.tzname, st.tzoffset
    _FALL_THROUGH = object()

    if tzname is not None and tzinfos is not None:
        if callable(tzinfos):
            tzdata = tzinfos(tzname, tzoffset_s)
            if tzdata is None:
                return None  # a callable that declines: naive, no warning
        else:
            tzdata = tzinfos.get(tzname)
            if tzdata is None:
                tzdata = _FALL_THROUGH  # dict miss: normal chain below
        if tzdata is not None and tzdata is not _FALL_THROUGH:
            if isinstance(tzdata, tzinfo):
                return tzdata
            if isinstance(tzdata, bool):
                raise TypeError(
                    "Offset must be tzinfo subclass, tz string, or int offset."
                )
            if isinstance(tzdata, int):
                return tzoffset(tzname, tzdata)
            if isinstance(tzdata, str):
                from zoneinfo import ZoneInfo

                try:
                    return ZoneInfo(tzdata)
                except Exception as exc:
                    raise TypeError(
                        "Offset must be tzinfo subclass, tz string, or int offset."
                    ) from exc
            raise TypeError(
                "Offset must be tzinfo subclass, tz string, or int offset."
            )

    # A recognized local name ("EST") wins over any parsed offset ("EST+5").
    if tzname is not None and st.tzname_known and tzname in local_tznames():
        return tzlocal()

    is_z = tzname is not None and len(tzname) == 1 and tzname.lower() == "z"
    if tzoffset_s is not None:
        if tzoffset_s == 0:
            return tzutc()
        # The name rides along in the tzoffset unless it is a UTC-zero name
        # ("UT+5" -> tzoffset('UT', -18000), "EST +2" -> tzoffset('EST', 7200);
        # recognized or not makes no difference to the oracle here).
        keep = tzname if (tzname not in _UTC_ZERO and not is_z) else None
        return tzoffset(keep, tzoffset_s)

    if tzname is not None:
        if tzname in _UTC_ZERO or is_z:
            return tzutc()
        warnings.warn(
            f"tzname {tzname} identified but not understood.  Pass `tzinfos` "
            "argument in order to correctly return a timezone-aware datetime.  "
            "In a future version, this will raise an exception.",
            UnknownTimezoneWarning,
            stacklevel=4,
        )
    return None


def _local_ambiguous(dt: datetime) -> bool:
    """Whether the wall time exists in both local DST states (fall-back)."""
    import time as _time

    base = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0, 0)
    try:
        std = _time.localtime(_time.mktime(base + (0,))).tm_isdst
        dst = _time.localtime(_time.mktime(base + (1,))).tm_isdst
    except (OverflowError, OSError, ValueError):
        return False
    return std == 0 and dst == 1


def parse(
    timestr: str,
    parserinfo=None,
    *,
    dayfirst: bool | None = None,
    yearfirst: bool | None = None,
    ignoretz: bool = False,
    tzinfos=None,
    default: datetime | None = None,
    fuzzy: bool = False,
    fuzzy_with_tokens: bool = False,
):
    """Parse a datetime from a string, matching ``dateutil.parser.parse``.

    Only the keyword arguments above are supported; a custom ``parserinfo``
    raises ``NotImplementedError`` (documented unsupported scope).
    """
    if parserinfo is not None:
        raise NotImplementedError(
            "custom parserinfo is not supported by dateutil_mojo "
            "(see README: unsupported scope)"
        )
    if not isinstance(timestr, str):
        raise TypeError(f"timestr must be str, got {type(timestr).__name__}")

    if fuzzy_with_tokens:
        fuzzy = True

    parser = _Parser(
        timestr,
        dayfirst=bool(dayfirst),
        yearfirst=bool(yearfirst),
        tzinfos=tzinfos,
        fuzzy=fuzzy,
    )
    parser.run()
    st = parser.st

    year, month, day = _resolve_ymd(st)

    have_date = year is not None or month is not None or day is not None
    have_time = (
        st.hour is not None or st.minute is not None or st.second is not None
    )
    if not have_date and not have_time and st.weekday is None:
        raise ParserError(f"String does not contain a date: {timestr}")

    if default is None:
        default = datetime.now()

    repl = {
        "year": year if year is not None else default.year,
        "month": month if month is not None else default.month,
        "day": day if day is not None else default.day,
        "hour": st.hour if st.hour is not None else default.hour,
        "minute": st.minute if st.minute is not None else default.minute,
        "second": st.second if st.second is not None else default.second,
        "microsecond": (
            st.microsecond if st.microsecond is not None else default.microsecond
        ),
    }

    try:
        dt = datetime(**repl)
    except ValueError as exc:
        raise ParserError(f"{exc}: {timestr}") from exc

    if st.weekday is not None and day is None:
        dt += timedelta(days=(st.weekday - dt.weekday()) % 7)

    if not ignoretz:
        tz = _build_tzaware(st, tzinfos)
        if tz is not None:
            dt = dt.replace(tzinfo=tz)
            if isinstance(tz, tzlocal) and _local_ambiguous(dt):
                # Ambiguous local wall time: the fold is the side that
                # matches the parsed abbreviation ("... EST" -> standard).
                names = local_tznames()
                fold = 0 if (len(names) > 1 and st.tzname == names[1]) else 1
                dt = dt.replace(fold=fold)
        elif default.tzinfo is not None:
            dt = dt.replace(tzinfo=default.tzinfo)
    elif default.tzinfo is not None:
        # ignoretz drops only the string's timezone, never the default's.
        dt = dt.replace(tzinfo=default.tzinfo)

    if fuzzy_with_tokens:
        return dt, _skipped_tokens(
            timestr, parser.toks, parser.skipped, parser.extra_kept,
            parser.ordinal_spans,
        )
    return dt


def _skipped_tokens(
    timestr: str,
    toks: list[_Tok],
    skipped_spans: list[tuple[int, int]],
    extra_kept: int = 0,
    ordinal_spans: set[tuple[int, int]] | None = None,
) -> tuple[str, ...]:
    """Reconstruct the fuzzy skipped-token tuple like the oracle.

    Skipped tokens are the unconsumed tokens plus (in fuzzy mode) jump words
    and separator characters. Each maximal run of skipped tokens becomes one
    piece of original text; runs at token index 0-1 extend left to the end
    of the previous kept token, later runs start at the run's first token,
    and every piece ends at the next kept token. Whitespace gaps between
    kept tokens collapse into ' ' pieces (at most len(kept) - 2 of them).
    """
    skipped_set = set(skipped_spans)
    n = len(toks)
    statuses = [
        "skipped" if (t.start, t.end) in skipped_set else "kept" for t in toks
    ]
    kept_count = statuses.count("kept") + extra_kept
    ws_budget = max(0, kept_count - 2)

    pieces: list[tuple[int, str]] = []
    covered: list[tuple[int, int]] = []  # spans covered by skipped pieces
    i = 0
    while i < n:
        if statuses[i] == "kept":
            i += 1
            continue
        j = i
        while j + 1 < n and statuses[j + 1] == "skipped":
            j += 1
        next_start = toks[j + 1].start if j + 1 < n else len(timestr)
        ordinals = ordinal_spans or set()
        kept_before = sum(
            1
            for k in range(i)
            if statuses[k] == "kept"
            and (toks[k].start, toks[k].end) not in ordinals
        )
        prev_end = toks[i - 1].end if i >= 1 else 0
        is_end_run = j + 1 >= n
        if kept_before <= 1 or (is_end_run and kept_before <= 2):
            piece = timestr[prev_end:next_start]
            start_pos = prev_end
        else:
            piece = timestr[toks[i].start:next_start]
            start_pos = toks[i].start
        pieces.append((start_pos, piece))
        covered.append((start_pos, next_start))
        i = j + 1

    # whitespace gaps between kept tokens, not covered by a skipped piece
    ws: list[tuple[int, str]] = []
    if ws_budget:
        prev_end = 0
        for k in range(n):
            if statuses[k] != "kept":
                continue
            gap = (prev_end, toks[k].start)
            if gap[0] < gap[1] and not any(
                c0 <= gap[0] and gap[1] <= c1 for c0, c1 in covered
            ):
                ws.append((gap[0], " "))
            prev_end = toks[k].end

    out = [text for _, text in sorted(pieces + ws[:ws_budget])]
    return tuple(piece for piece in out if piece)
