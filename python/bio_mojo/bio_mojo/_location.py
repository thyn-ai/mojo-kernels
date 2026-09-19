"""GenBank feature-location parser and normalizer (shared by both backends).

Reproduces the oracle's (``Bio.SeqIO`` / biopython 1.88) ``str(location)``
rendering for the supported subset, pinned by black-box probes:

    123..456                          -> [122:456](+)
    <123..>456                        -> [<122:>456](+)
    123                               -> [122:123](+)
    50^51            (adjacent)       -> [50:50](+)
    complement(123..456)              -> [122:456](-)
    complement(complement(5..10))     -> [4:10](-)    (strand stays -1)
    join(5..10,20..30)                -> join{[4:10](+), [19:30](+)}
    join(5..10)                       -> [4:10](+)    (single part unwraps)
    order(5..10,20..30)               -> order{[4:10](+), [19:30](+)}
    complement(join(5..10,20..30))    -> join{[19:30](-), [4:10](-)}
    complement(order(5..10,20..30))   -> order{[19:30](-), [4:10](-)}
    J00194.1:100..200                 -> J00194.1[99:200](+)

Anything else yields ``None`` — the same value the oracle assigns to
``feature.location`` for ``one-of(...)``, non-adjacent ``a^b``, reversed
ranges (``10..5``), fuzzy between positions, nested compound operators,
``complement(join(complement(...),...))``, and unparseable strings.
"""

from __future__ import annotations

from bio_mojo.records import CompoundLocation, SimpleLocation

_OPERATORS = ("join", "order")


def _encloses(s: str, open_idx: int) -> bool:
    """True if the '(' at open_idx is closed by the FINAL char of s."""
    if not s.endswith(")"):
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


def _split_top_commas(s: str) -> list[str] | None:
    """Split at top-level commas; None on unbalanced parentheses."""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    if depth != 0:
        return None
    parts.append(s[start:])
    return parts


def _parse_fuzz_pos(p: str) -> tuple[str, int] | None:
    """'[<|>]digits' -> (fuzz, value); None on anything else."""
    if not p:
        return None
    fuzz = ""
    if p[0] in "<>":
        fuzz = p[0]
        p = p[1:]
    if not p or not p.isdigit():
        return None
    return fuzz, int(p)


def _parse_simple(s: str, strand: int) -> SimpleLocation | None:
    accession: str | None = None
    if ":" in s:
        acc, _, rest = s.partition(":")
        # Remote accession (e.g. J00194.1); must contain a letter so plain
        # '123:456' is not mistaken for one.
        if (
            not acc
            or not all(c.isalnum() or c in "._" for c in acc)
            or not any(c.isalpha() for c in acc)
            or not rest
        ):
            return None
        accession = acc
        s = rest

    if ".." in s:
        left, sep, right = s.partition("..")
        if not sep or ".." in right:
            return None
        lp = _parse_fuzz_pos(left)
        rp = _parse_fuzz_pos(right)
        if lp is None or rp is None:
            return None
        lf, a = lp
        rf, b = rp
        if a < 1 or b < 1 or a > b:
            return None
        return SimpleLocation(a - 1, b, strand, lf, rf, accession)
    if "^" in s:
        left, sep, right = s.partition("^")
        if not sep or left.startswith(("<", ">")) or right.startswith(("<", ">")):
            return None
        lp = _parse_fuzz_pos(left)
        rp = _parse_fuzz_pos(right)
        if lp is None or rp is None:
            return None
        _, a = lp
        _, b = rp
        # Only the adjacent bond a^(a+1) parses; anything else is None (as
        # the oracle leaves e.g. 5^9 unparsed).
        if a < 1 or b != a + 1:
            return None
        return SimpleLocation(a, a, strand, "", "", accession)
    pp = _parse_fuzz_pos(s)
    if pp is None:
        return None
    fuzz, n = pp
    if n < 1:
        return None
    return SimpleLocation(n - 1, n, strand, fuzz, "", accession)


def _parse_part(s: str) -> SimpleLocation | None:
    """One compound member: a simple location or complement(simple)."""
    if s.startswith("complement("):
        if not _encloses(s, 10):
            return None
        inner = s[11:-1]
        if inner.startswith(("join(", "order(", "complement(")):
            return None  # nested operators inside a compound member
        return _parse_simple(inner, -1)
    if s.startswith(("join(", "order(")):
        return None  # nested compound -> oracle leaves unparsed
    return _parse_simple(s, 1)


def _parse_compound(op: str, inner: str) -> SimpleLocation | CompoundLocation | None:
    parts = _split_top_commas(inner)
    if parts is None or not parts or any(p == "" for p in parts):
        return None
    simple_parts: list[SimpleLocation] = []
    for p in parts:
        sp = _parse_part(p)
        if sp is None:
            return None
        simple_parts.append(sp)
    if len(simple_parts) == 1:
        return simple_parts[0]  # single-part compound unwraps
    return CompoundLocation(op, tuple(simple_parts))


def _complement_compound(op: str, inner: str) -> SimpleLocation | CompoundLocation | None:
    """complement(join/order(...)): parts reversed, each strand flipped.

    Members must be plain simple locations (the oracle leaves
    complement(join(complement(...), ...)) unparsed).
    """
    parts = _split_top_commas(inner)
    if parts is None or not parts or any(p == "" for p in parts):
        return None
    out: list[SimpleLocation] = []
    for p in parts:
        if p.startswith(("join(", "order(", "complement(")):
            return None
        sp = _parse_simple(p, -1)
        if sp is None:
            return None
        out.append(sp)
    out.reverse()
    if len(out) == 1:
        return out[0]
    return CompoundLocation(op, tuple(out))


def parse_location(s: str) -> SimpleLocation | CompoundLocation | None:
    """Normalize a raw GenBank location string; None when unparsed.

    Whitespace must already be removed by the caller. Never raises.
    """
    if not s:
        return None
    if s.startswith("complement(") and _encloses(s, 10):
        inner = s[11:-1]
        if inner.startswith("complement(") and _encloses(inner, 10):
            # Double complement: the oracle keeps strand -1 (it does not
            # toggle back), on the innermost simple location.
            inner2 = inner[11:-1]
            if inner2.startswith(("join(", "order(", "complement(")):
                return None
            return _parse_simple(inner2, -1)
        for op in _OPERATORS:
            if inner.startswith(op + "(") and _encloses(inner, len(op)):
                return _complement_compound(op, inner[len(op) + 1 : -1])
        if inner.startswith(("join(", "order(", "complement(")):
            return None
        return _parse_simple(inner, -1)
    for op in _OPERATORS:
        if s.startswith(op + "(") and _encloses(s, len(op)):
            return _parse_compound(op, s[len(op) + 1 : -1])
    if s.startswith(("join(", "order(")):
        return None  # unbalanced operator syntax
    return _parse_simple(s, 1)
