"""5-field cron expression parser, producing compact bitmasks for the engines.

Clean-room implementation of the classic cron field grammar, with the
behavioral details matched against the black-box `croniter` oracle
(differential suite: tests/test_croniter_differential.py):

    field      := item ("," item)*
    item       := "*" | "?" | <value> | <value> "-" <value>   (any may carry "/step")
    step form  := <base> "/" <n>     n >= 1; bare "a/n" means "a-max/n"
    value      := integer | 3-letter month name (month field)
                          | 3-letter weekday name (dow field)
                          | "l" (day-of-month field: last day of month)
    dow        := 0..6 with Sunday == 0; 7 is an alias for 0
    hash       := <dow value> "#" <1..5>   (nth weekday of month, dow field only;
                          a dow field is either all-hash or all-plain, never mixed)

Ranges wrap around when the low end exceeds the high end ("50-10" in the
minute field is 50..59,0..10; "nov-feb" is Nov..Feb). Steps continue across
the wrap point. "?" is an alias for "*" in the dom/dow fields.

A field whose text is exactly "*" or "?" is "unrestricted"; anything else is
"restricted". When both dom and dow are restricted they combine with OR by
default (``day_or=True``) or AND (``day_or=False``) — except when the dow
field uses hash specs and ``day_or=True``: then matching uses the hash rule
alone, and a restricted dom acts only as a static feasibility gate (the
dom value set must intersect the nth-week windows [7n-6, 7n] of the hash
specs, otherwise no date can ever match).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._errors import (
    CroniterBadCronError,
    CroniterNotAlphaError,
    CroniterUnsupportedSyntaxError,
)

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DOW_NAMES = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
}

# (min, max) per field position: minute, hour, dom, month, dow.
# dow keeps max 7 here so "7" parses; 7 is normalized to 0 afterwards.
_FIELD_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_MINUTE, _HOUR, _DOM, _MONTH, _DOW = range(5)

# Static-gate window for a hash spec d#n: the nth weekday always falls on
# days 7*(n-1)+1 .. 7*n of the month.
_MONTH_LAST_DAYS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}  # Feb counts leap

_MAX_HASH_SPECS = 64  # mirrors the kernel's create-time bound


@dataclass(frozen=True)
class Schedule:
    """A parsed 5-field cron schedule as engine-ready bitmasks."""

    minutes: int  # bits 0..59
    hours: int  # bits 0..23
    doms: int  # bits 1..31
    dom_has_l: bool
    months: int  # bits 1..12
    dows: int  # bits 0..6 (Sunday == 0)
    dom_restricted: bool
    dow_restricted: bool
    hour_restricted: bool  # hour value set != full 0..23 (semantic, not textual)
    day_or: bool
    dow_has_hash: bool
    hash_specs: tuple[tuple[int, int], ...]  # (cron dow 0..6, nth 1..5)
    gate_fails: bool  # static infeasibility: hash + restricted dom + day_or


def _field_value(token: str, field: int, expr: str) -> int:
    """Parse one value token (integer or field-appropriate name) to an int."""
    if field == _MONTH and token in MONTH_NAMES:
        return MONTH_NAMES[token]
    if field == _DOW and token in DOW_NAMES:
        return DOW_NAMES[token]
    try:
        return int(token)
    except ValueError:
        raise CroniterNotAlphaError(f"[{expr}] is not acceptable") from None


def _check_range(value: int, field: int, expr: str) -> int:
    lo, hi = _FIELD_BOUNDS[field]
    if value < lo or value > hi:
        raise CroniterBadCronError(f"[{expr}] is not acceptable, out of range")
    if field == _DOW and value == 7:
        return 0  # Sunday alias
    return value


def _expand(lo: int, hi: int, step: int, field: int) -> set[int]:
    """Values of lo-hi/step (oracle-verified expansion rules).

    - Degenerate ranges collapse: "a-a" is the full field range ("*"), and
      "a-a/n" is "*/n".
    - Wrapping ranges (lo > hi, e.g. 50-10/5, nov-feb): the first segment
      is range(lo, max+1, step); the wrapped segment starts at
      (last + step) - (max - min) when the step overshoots the field
      maximum, or exactly at min when last + step == max + 1 — this
      reproduces the oracle's off-by-one wrap arithmetic exactly.

    (7 was already normalized to 0 by the caller, so the dow domain is 0..6.)
    """
    fmin, fmax = _FIELD_BOUNDS[field]
    if field == _DOW:
        fmin, fmax = 0, 6
    if lo == hi:
        lo, hi = fmin, fmax
    if lo <= hi:
        return set(range(lo, hi + 1, step))
    first = list(range(lo, fmax + 1, step))
    values = set(first)
    if first:
        nxt = first[-1] + step
        start2 = fmin if nxt == fmax + 1 else nxt - (fmax - fmin)
        if start2 <= hi:
            values.update(range(start2, hi + 1, step))
    return values


def _parse_plain_field(text: str, field: int, expr: str) -> tuple[set[int], bool]:
    """Parse one non-hash field; returns (values, has_last_day_item)."""
    lo_bound, hi_bound = _FIELD_BOUNDS[field]
    values: set[int] = set()
    has_l = False
    for item in text.split(","):
        if not item:
            raise CroniterBadCronError(f"[{expr}] is not acceptable")
        base, slash, step_text = item.partition("/")
        if slash:
            try:
                step = int(step_text)
            except ValueError:
                raise CroniterNotAlphaError(f"[{expr}] is not acceptable") from None
            if step <= 0:
                raise CroniterBadCronError(
                    f"[{expr}] step '{step_text}' in field {field} is not acceptable"
                )
        else:
            step = 1

        if base in ("*", "?"):
            if base == "?" and field not in (_DOM, _DOW):
                raise CroniterNotAlphaError(f"[{expr}] is not acceptable")
            lo, hi = lo_bound, hi_bound
            if field == _DOW:
                hi = 6
            values.update(_expand(lo, hi, step, field))
            continue

        if base == "l" and field == _DOM and not slash:
            has_l = True
            continue
        if field == _DOM and "l" in base:
            raise CroniterBadCronError(
                f"[{expr}] bands '{base}' in field {field} are not acceptable"
            )

        if "-" in base:
            lo_text, _, hi_text = base.partition("-")
            lo = _check_range(_field_value(lo_text, field, expr), field, expr)
            hi = _check_range(_field_value(hi_text, field, expr), field, expr)
            values.update(_expand(lo, hi, step, field))
        else:
            lo = _check_range(_field_value(base, field, expr), field, expr)
            if slash:
                hi = hi_bound if field != _DOW else 6
                values.update(_expand(lo, hi, step, field))
            else:
                values.add(lo)
    return values, has_l


def _parse_dow_field(text: str, expr: str) -> tuple[set[int], tuple[tuple[int, int], ...]]:
    """Parse the day-of-week field: either plain values or hash specs."""
    items = text.split(",")
    hash_specs: list[tuple[int, int]] = []
    plain_items: list[str] = []
    for item in items:
        if "#" in item:
            dow_text, _, nth_text = item.partition("#")
            dow = _check_range(_field_value(dow_text, _DOW, expr), _DOW, expr)
            try:
                nth = int(nth_text)
            except ValueError:
                raise CroniterNotAlphaError(f"[{expr}] is not acceptable") from None
            if nth < 1 or nth > 5:
                raise CroniterBadCronError(
                    f"[{expr}] is not acceptable. Invalid day_of_week value: '{nth_text}'"
                )
            hash_specs.append((dow, nth))
        else:
            plain_items.append(item)
    if hash_specs and plain_items:
        raise CroniterUnsupportedSyntaxError(
            "day-of-week field does not support mixing literal values and "
            f"nth day of week syntax.  Cron: '{expr}'"
        )
    if len(hash_specs) > _MAX_HASH_SPECS:
        raise CroniterUnsupportedSyntaxError(
            f"[{expr}] too many nth-day-of-week specs (max {_MAX_HASH_SPECS})"
        )
    if hash_specs:
        return set(), tuple(hash_specs)
    values, _ = _parse_plain_field(text, _DOW, expr)
    return values, ()


def _hash_gate_fails(
    doms: set[int], dom_has_l: bool, hash_specs: tuple[tuple[int, int], ...], allowed_months: set[int]
) -> bool:
    """Static feasibility gate for hash-dow + restricted dom under day_or.

    The nth weekday of a month always lands on days 7n-6..7n; if the
    restricted dom set cannot intersect those windows, no date can match.
    A dom "l" item contributes the last days of the ALLOWED months only
    (February contributes 28 and 29: leap and non-leap Februaries both
    exist within any 50-year horizon).
    """
    windows: set[int] = set()
    for _, nth in hash_specs:
        windows.update(range(7 * (nth - 1) + 1, 7 * nth + 1))
    candidates = set(doms)
    if dom_has_l:
        for m in allowed_months:
            candidates.add(_MONTH_LAST_DAYS[m])
            if m == 2:
                candidates.add(28)  # non-leap Februaries also exist in any horizon
    return candidates.isdisjoint(windows)


def parse(expr: str, day_or: bool = True) -> Schedule:
    """Parse a 5-field cron expression into a :class:`Schedule`.

    Raises CroniterBadCronError (or a subclass) on malformed input and
    CroniterUnsupportedSyntaxError for out-of-scope constructs (6/7-field
    expressions, mixed plain/hash dow lists).
    """
    if not isinstance(expr, str):
        raise CroniterBadCronError(
            f"cron expression must be a string, got {type(expr).__name__}"
        )
    fields = expr.lower().split()
    if len(fields) != 5:
        if len(fields) in (6, 7):
            raise CroniterUnsupportedSyntaxError(
                f"[{expr}] croniter_mojo supports 5-field expressions "
                "(minute hour day-of-month month day-of-week); "
                "seconds/year fields are out of scope"
            )
        raise CroniterBadCronError(
            "Exactly 5, 6 or 7 columns has to be specified for iterator expression."
        )

    raw_min, raw_hour, raw_dom, raw_month, raw_dow = fields
    minutes, _ = _parse_plain_field(raw_min, _MINUTE, expr)
    hours, _ = _parse_plain_field(raw_hour, _HOUR, expr)
    doms, dom_has_l = _parse_plain_field(raw_dom, _DOM, expr)
    months, _ = _parse_plain_field(raw_month, _MONTH, expr)
    dows, hash_specs = _parse_dow_field(raw_dow, expr)

    dom_restricted = raw_dom not in ("*", "?")
    dow_restricted = raw_dow not in ("*", "?")
    # Oracle quirk (Vixie-cron DOM_STAR heritage): a dom field written as
    # "*/n" decides alone when the dow expansion covers the whole week —
    # e.g. "0 0 */2 * fri-thu" fires on odd days only, while
    # "0 0 5 * fri-thu" uses plain OR (dom is not in "*/" form).
    if (
        day_or
        and dow_restricted
        and dom_restricted
        and not hash_specs
        and dows == set(range(7))
        and raw_dom.startswith("*/")
    ):
        dow_restricted = False
    dow_has_hash = bool(hash_specs)

    # Static infeasibility gates, both reported as CroniterBadDateError on
    # the first seek (as the oracle does):
    #  1. hash-dow + restricted dom + day_or: the dom set must intersect the
    #     nth-weekday windows.
    #  2. restricted dom with no "l": at least one dom value must exist as a
    #     day of an allowed month (e.g. day 31 in a September-only schedule
    #     can never match, and the oracle raises rather than letting the dow
    #     side match under OR).
    hash_gate_fails = (
        dow_has_hash
        and dom_restricted
        and day_or
        and _hash_gate_fails(doms, dom_has_l, hash_specs, months)
    )
    dom_infeasible = False
    if dom_restricted and not dom_has_l:
        possible_days: set[int] = set()
        for m in months:
            possible_days.update(range(1, _MONTH_LAST_DAYS[m] + 1))
            if m == 2:
                possible_days.add(28)
        dom_infeasible = doms.isdisjoint(possible_days)
    gate_fails = hash_gate_fails or dom_infeasible

    def mask(values: set[int]) -> int:
        m = 0
        for v in values:
            m |= 1 << v
        return m

    return Schedule(
        minutes=mask(minutes),
        hours=mask(hours),
        doms=mask(doms),
        dom_has_l=dom_has_l,
        months=mask(months),
        dows=mask(dows),
        dom_restricted=dom_restricted,
        dow_restricted=dow_restricted,
        hour_restricted=hours != set(range(24)),
        day_or=day_or,
        dow_has_hash=dow_has_hash,
        hash_specs=hash_specs,
        gate_fails=gate_fails,
    )
