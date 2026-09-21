"""Pure-Python engine over the compiled phonenumbers-mojo tables.

This module is both the pure-Python fallback backend and the executable
specification for the Mojo kernel: the kernel implements the same pipeline
over the same blob, and the differential suite asserts identical results on
both backends against the PyPI `phonenumbers` oracle.

The pipeline mirrors the documented libphonenumber parse/validate/format
semantics, reverse-engineered and pinned behaviorally against the oracle
(phonenumbers 9.0.39):

  1. extract_possible_number: slice from the first '+'/'＋'/digit, trim a
     trailing run of non-word characters (except '#') and '_', and cut a
     second number started by '[\\/] *x'.
  2. viability: at least 2 chars and either exactly 2 digits or 3+ digit
     groups separated by the separator set, followed by separator/letter/
     digit characters and an optional extension tail.
  3. extension strip: six end-anchored marker branches (see _match_extn),
     leftmost match wins, applied only when the prefix stays viable.
  4. normalize: ASCII letters map to keypad digits, ASCII digits and a
     leading '+'/'＋' are kept, everything else is dropped.
  5. country calling code extraction: '+' prefix, then the default region's
     international prefix (IDD), then the default region's own code when the
     number starts with it, else the default region's code as context.
  6. national prefix strip: nationalPrefixForParsing prefix match, with the
     transform rule; the strip is kept unless the original matches the
     general pattern and the stripped candidate does not.
  7. length checks: NSN must be 2..17 digits.

Validity: the number must full-match the selected region's general pattern
and at least one type pattern (with that type's possible lengths, when the
type records them). Possibility: the NSN length must be one of the general
possible lengths. Region selection among regions sharing a calling code:
first region whose leadingDigits prefix-matches wins; otherwise the first
region whose patterns match the number; otherwise the main region.

Formatting: E164 = '+cc nsn'; INTERNATIONAL = '+cc ' + intl-template
application (national prefix rules never apply); NATIONAL = template
application with the format's national prefix rule. When no format matches,
the raw NSN is used. Extensions are appended as the region's
preferredExtnPrefix (or ' ext. ') plus the digits, for NATIONAL and
INTERNATIONAL only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Blob opcodes (must match kernels/phonenumbers/gen_tables.py).
OP_CLS, OP_SPLIT, OP_JMP, OP_SAVE, OP_MATCH, OP_EOL = range(6)

NO_LEN_CONSTRAINT = 0xFFFFFFFF
MAX_NSN_LEN = 17
MIN_NSN_LEN = 2

# NumberParseException error types (oracle parity).
INVALID_COUNTRY_CODE = 0
NOT_A_NUMBER = 1
TOO_SHORT_AFTER_IDD = 2
TOO_SHORT_NSN = 3
TOO_LONG = 4

# PhoneNumberFormat values (oracle parity).
FMT_E164 = 0
FMT_INTERNATIONAL = 1
FMT_NATIONAL = 2

# Capture-slot capacity in the Pike VM. Format patterns bind at most 9
# template references; 16 groups (34 slots) covers every shipped pattern.
MAX_GROUPS = 16


class ParseError(Exception):
    """Internal carrier for parse failures: (error_type, message)."""

    def __init__(self, error_type: int, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


# ---------------------------------------------------------------- blob tables


@dataclass
class Program:
    insts: list[tuple[int, int, int]]
    n_groups: int


@dataclass
class Region:
    idx: int
    code: str
    cc: int
    main: bool
    intl_prefix_prog: int
    national_prefix: str | None
    npfp_prog: int
    np_transform: str | None
    leading_digits_prog: int
    preferred_extn_prefix: str | None
    general_prog: int
    general_mask: int
    general_mask_local: int
    # (prog, lenmask) per type, fixed order (see gen_tables.TYPE_TAGS)
    types: list[tuple[int, int]] = field(default_factory=list)
    # (pattern_prog, format_template, np_rule, leading_digits_prog)
    formats: list[tuple[int, str, str | None, int]] = field(default_factory=list)
    intl_formats: list[tuple[int, str, str | None, int]] = field(default_factory=list)


class Tables:
    """Parsed view of the compiled blob; shared by all engine calls."""

    def __init__(self, blob: bytes) -> None:
        if blob[:4] != b"PNM1":
            raise ValueError("bad blob magic")
        (_version, n_programs, n_regions, n_strings, cc_off, _cc_count) = struct.unpack_from(
            "<IIIIII", blob, 4
        )
        pos = 28
        prog_offsets = struct.unpack_from(f"<{n_programs}I", blob, pos)
        pos += 4 * n_programs
        str_offsets = struct.unpack_from(f"<{n_strings}I", blob, pos)
        pos += 4 * n_strings
        reg_offsets = struct.unpack_from(f"<{n_regions}I", blob, pos)

        self.programs: list[Program] = []
        for off in prog_offsets:
            n_insts, n_groups = struct.unpack_from("<II", blob, off)
            insts = []
            p = off + 8
            for _ in range(n_insts):
                insts.append(struct.unpack_from("<iii", blob, p))
                p += 12
            self.programs.append(Program(insts, n_groups))

        self.strings: list[str] = []
        for off in str_offsets:
            (n,) = struct.unpack_from("<I", blob, off)
            self.strings.append(blob[off + 4 : off + 4 + n].decode("utf-8"))

        def s(i: int) -> str | None:
            return None if i < 0 else self.strings[i]

        self.regions: list[Region] = []
        for idx, off in enumerate(reg_offsets):
            code_idx, cc, main, _res = struct.unpack_from("<HHBB", blob, off)
            (
                intl_prefix_prog,
                national_prefix_str,
                npfp_prog,
                np_transform_str,
                _np_rule_str,
                leading_digits_prog,
                pref_extn_str,
                general_prog,
            ) = struct.unpack_from("<8i", blob, off + 6)
            general_mask, general_mask_local = struct.unpack_from("<II", blob, off + 38)
            p = off + 46
            types = []
            for _ in range(10):
                prog, mask, _mask_local = struct.unpack_from("<iII", blob, p)
                types.append((prog, mask))
                p += 12
            fmt_lists = []
            for _ in range(2):
                (n_fmt,) = struct.unpack_from("<I", blob, p)
                p += 4
                fmts = []
                for _ in range(n_fmt):
                    pattern_prog, format_str, np_rule, ld_prog = struct.unpack_from("<4i", blob, p)
                    p += 16
                    fmts.append((pattern_prog, s(format_str), s(np_rule), ld_prog))
                fmt_lists.append(fmts)
            self.regions.append(
                Region(
                    idx=idx,
                    code=self.strings[code_idx],
                    cc=cc,
                    main=bool(main),
                    intl_prefix_prog=intl_prefix_prog,
                    national_prefix=s(national_prefix_str),
                    npfp_prog=npfp_prog,
                    np_transform=s(np_transform_str),
                    leading_digits_prog=leading_digits_prog,
                    preferred_extn_prefix=s(pref_extn_str),
                    general_prog=general_prog,
                    general_mask=general_mask,
                    general_mask_local=general_mask_local,
                    types=types,
                    formats=fmt_lists[0],
                    intl_formats=fmt_lists[1],
                )
            )

        self.cc_map: dict[int, list[int]] = {}
        p = cc_off
        for _ in range(_cc_count):
            cc, n = struct.unpack_from("<II", blob, p)
            p += 8
            self.cc_map[cc] = list(struct.unpack_from(f"<{n}I", blob, p))
            p += 4 * n
        self.region_by_code: dict[str, int] = {r.code: r.idx for r in self.regions}

    @classmethod
    def from_default_blob(cls) -> "Tables":
        from phonenumbers_mojo._data import load_blob

        return cls(load_blob())


# ------------------------------------------------------------------ Pike VM
#
# Priority-ordered Thompson NFA simulation. Threads are kept in preference
# order (SPLIT prefers x over y), which reproduces leftmost-biased
# backtracking semantics: a successful MATCH cuts every lower-priority
# thread, and a later MATCH from a surviving (higher-priority) thread
# replaces the recorded result.

FULL = 0  # whole string must be consumed (re.fullmatch semantics)
PREFIX = 1  # anchored at position 0, may end anywhere (re.match semantics)


def _add_thread(
    prog: Program,
    pc: int,
    pos: int,
    caps: list[int],
    out: list[tuple[int, list[int]]],
    seen: set[int],
    want_caps: bool,
) -> None:
    """Epsilon closure: follow SPLIT/JMP/SAVE/EOL before the next consuming inst."""
    stack = [(pc, caps)]
    while stack:
        pc, caps = stack.pop()
        if pc in seen:
            continue
        seen.add(pc)
        op, x, y = prog.insts[pc]
        if op == OP_JMP:
            stack.append((x, caps))
        elif op == OP_SPLIT:
            # y is lower priority: push it first so x pops first.
            stack.append((y, caps))
            stack.append((x, caps))
        elif op == OP_SAVE:
            if want_caps:
                caps = caps.copy()
                caps[x] = pos
            stack.append((pc + 1, caps))
        elif op == OP_EOL:
            if pos == _EOL_POS[0]:
                stack.append((pc + 1, caps))
        else:  # CLASS or MATCH
            out.append((pc, caps))


# _add_thread needs the current input length for EOL; threaded through a
# module cell to keep the closure iterative and allocation-free per call site.
_EOL_POS = [0]


def run_program(
    tables: Tables, prog_idx: int, s: str, mode: int, want_caps: bool = False
) -> tuple[bool, int, list[int] | None]:
    """Run program `prog_idx` against digit string `s`.

    Returns (matched, end_pos, caps|None). In FULL mode end_pos == len(s) on
    success. In PREFIX mode end_pos is where the winning thread matched.
    """
    prog = tables.programs[prog_idx]
    n_slots = 2 * (prog.n_groups + 1)
    _EOL_POS[0] = len(s)
    clist: list[tuple[int, list[int]]] = []
    seen: set[int] = set()
    _add_thread(prog, 0, 0, [-1] * n_slots, clist, seen, want_caps)
    best: tuple[int, list[int]] | None = None
    pos = 0
    while True:
        nlist: list[tuple[int, list[int]]] = []
        nseen: set[int] = set()
        cut = False
        for pc, caps in clist:
            if cut:
                break
            op, x, _y = prog.insts[pc]
            if op == OP_CLS:
                if pos < len(s):
                    ch = s[pos]
                    if "0" <= ch <= "9" and (x >> (ord(ch) - 48)) & 1:
                        _add_thread(prog, pc + 1, pos + 1, caps, nlist, nseen, want_caps)
            elif op == OP_MATCH:
                if mode == PREFIX or pos == len(s):
                    best = (pos, caps)
                    cut = True  # remaining threads are lower priority
                # FULL mode with pos < len: this thread dies.
        if best is not None and not nlist:
            break
        if pos >= len(s):
            break
        clist = nlist
        pos += 1
    if best is None:
        return (False, 0, None)
    return (True, best[0], best[1] if want_caps else None)


def full_match(tables: Tables, prog_idx: int, s: str, want_caps: bool = False):
    if prog_idx < 0:
        return (False, 0, None)
    return run_program(tables, prog_idx, s, FULL, want_caps)


def prefix_match(tables: Tables, prog_idx: int, s: str, want_caps: bool = False):
    if prog_idx < 0:
        return (False, 0, None)
    return run_program(tables, prog_idx, s, PREFIX, want_caps)


# ------------------------------------------------------------- input handling

# Separator set accepted between digit groups by the viability rule (the
# libphonenumber "phone punctuation" set, observed from the oracle).
_SEPARATORS = frozenset(
    "-x‐-―−ー－-／ \xa0\xad\u200b\u2060\u3000()（）［］.[]/~⁓∼～*"
)

_ALPHA_TO_DIGIT = {}
for _letters, _digit in (
    ("ABC", "2"), ("DEF", "3"), ("GHI", "4"), ("JKL", "5"), ("MNO", "6"),
    ("PQRS", "7"), ("TUV", "8"), ("WXYZ", "9"),
):
    for _ch in _letters:
        _ALPHA_TO_DIGIT[_ch] = _digit
        _ALPHA_TO_DIGIT[_ch.lower()] = _digit


def _extract_possible_number(text: str) -> str:
    start = -1
    for i, ch in enumerate(text):
        if ch == "+" or ch == "＋" or ch.isdigit():
            start = i
            break
    if start < 0:
        return ""
    s = text[start:]
    # Trim trailing run of '_' or non-word chars except '#'.
    end = len(s)
    while end > 0:
        ch = s[end - 1]
        if ch == "_" or (ch != "#" and not ch.isalnum()):
            end -= 1
        else:
            break
    s = s[:end]
    # Cut a second number: '[\\/] *x'
    for i, ch in enumerate(s):
        if ch in "\\/":
            j = i + 1
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "x":
                return s[:i]
    return s


# Extension marker branches, mirroring the observed oracle grammar
# (case-insensitive, end-anchored, leftmost start wins):
#   A: ';ext=' + 1..20 digits
#   B: [ \xa0\t,]* (e?xt(?:ensi(ó|ó)?)?n? | ｅ?ｘｔｎ? | доб | anexo) [:.．]? [ \xa0\t,-]* 1..20 digits #?
#   C: [ \xa0\t,]* (x|ｘ|#|＃|~|～|int|ｉｎｔ) [:.．]? [ \xa0\t,-]* 1..9 digits #?
#   D: [- ]+ 1..6 digits '#'
#   E: [ \xa0\t]* (,,|;) [:.．]? [ \xa0\t,-]* 1..15 digits #?
#   F: [ \xa0\t]* ,+ [:.．]? [ \xa0\t,-]* 1..9 digits #?
_EXT_MARKERS_LONG = (
    # e?xt(?:ensi(?:o|o\u0301|\u00f3))?n?  (longest first so backtracking
    # prefers them; "extensi"/"extensin" alone are NOT markers)
    "extensión", "extensión", "extensió", "extensió",
    "xtensión", "xtensión", "xtensió", "xtensió",
    "extension", "extensio", "xtensio", "extn", "xtn", "ext", "xt",
    "ｅｘｔｎ", "ｅｘｔ", "anexo", "доб",
)
_EXT_MARKERS_SHORT = ("int", "ｉｎｔ", "x", "ｘ", "#", "＃", "~", "～")
_EXT_SEP_BC = frozenset(" \xa0\t,")
_EXT_SEP_EF = frozenset(" \xa0\t")
_EXT_POST_SEP = frozenset(" \xa0\t,-")
_EXT_DOT = frozenset(":.．")


def _match_extn_at(s: str, start: int) -> tuple[int, int] | None:
    """Try to match one extn branch starting exactly at `start`, anchored at
    the end of `s`. Returns the (start, end) span of the extension digits,
    or None."""
    body = s[start:]
    lower = body.lower()
    hash_trimmed = body[:-1] if body.endswith("#") else body
    lower_ht = hash_trimmed.lower()

    def digits_after(i: int, lo: int, hi: int) -> tuple[int, int] | None:
        """Optional dot/separator run then lo..hi digits reaching the end.
        `i` is a body-relative index right after the marker."""
        if i < len(hash_trimmed) and hash_trimmed[i] in _EXT_DOT:
            i += 1
        while i < len(hash_trimmed) and hash_trimmed[i] in _EXT_POST_SEP:
            i += 1
        j = i
        while j < len(hash_trimmed) and "0" <= hash_trimmed[j] <= "9":
            j += 1
        if lo <= j - i <= hi and j == len(hash_trimmed):
            return (start + i, start + j)
        return None

    # Branch A: ';ext=' + 1..20 digits (no separators, no trailing '#')
    if lower.startswith(";ext="):
        j = 5
        k = j
        while k < len(body) and "0" <= body[k] <= "9":
            k += 1
        if 1 <= k - j <= 20 and k == len(body):
            return (start + j, start + k)
        return None

    def pre_sep(seps: frozenset) -> int:
        i = 0
        while i < len(body) and body[i] in seps:
            i += 1
        return i

    # Branch B: long markers, 1..20 digits
    i = pre_sep(_EXT_SEP_BC)
    for m in _EXT_MARKERS_LONG:
        if lower_ht.startswith(m, i):
            r = digits_after(i + len(m), 1, 20)
            if r is not None:
                return r
    # Branch C: short markers, 1..9 digits
    for m in _EXT_MARKERS_SHORT:
        if lower_ht.startswith(m, i):
            r = digits_after(i + len(m), 1, 9)
            if r is not None:
                return r
    # Branch D: [- ]+ 1..6 digits '#'  (trailing hash required)
    if body.endswith("#"):
        j = 0
        while j < len(body) and body[j] in "- ":
            j += 1
        if j > 0:
            k = j
            while k < len(body) and "0" <= body[k] <= "9":
                k += 1
            if 1 <= k - j <= 6 and k == len(body) - 1:
                return (start + j, start + k)
        return None
    # Branch E: ',,' or ';' then 1..15 digits
    i = pre_sep(_EXT_SEP_EF)
    if lower.startswith(",,", i) or lower.startswith(";", i):
        r = digits_after(i + (2 if lower.startswith(",,", i) else 1), 1, 15)
        if r is not None:
            return r
    # Branch F: one or more ',' then 1..9 digits
    if lower.startswith(",", i):
        k = i
        while k < len(hash_trimmed) and hash_trimmed[k] == ",":
            k += 1
        r = digits_after(k, 1, 9)
        if r is not None:
            return r
    return None


def strip_extension(s: str) -> tuple[str, str]:
    """Leftmost end-anchored extn match whose prefix stays viable.

    Returns (extension_digits_or_empty, remaining_number)."""
    for start in range(len(s)):
        r = _match_extn_at(s, start)
        if r is not None and is_viable(s[:start]):
            return s[r[0] : r[1]], s[:start]
    return "", s


def is_viable(s: str) -> bool:
    """Mirror of the oracle viability rule: len >= 2 and either exactly two
    digits, or 3+ separator-delimited digit groups followed by separator/
    letter/digit characters and an optional end-anchored extension tail."""
    if len(s) < MIN_NSN_LEN:
        return False
    if len(s) == 2 and s.isascii() and s.isdigit():
        return True
    i = 0
    n = len(s)
    while i < n and s[i] in "+＋":
        i += 1
    groups = 0
    while groups < 3:
        while i < n and _is_separator(s[i]):
            i += 1
        if i < n and s[i].isdigit():
            groups += 1
            i += 1
        else:
            return False
    # Tail: separators, ASCII letters, digits; an end-anchored extension may
    # start at any tail position (the reference regex backtracks to find it).
    while i < n:
        ch = s[i]
        if _is_separator(ch) or ch.isdigit() or (ch.isascii() and ch.isalpha()):
            if _match_extn_at(s, i) is not None:
                return True
            i += 1
        else:
            return _match_extn_at(s, i) is not None
    return True


def _is_separator(ch: str) -> bool:
    """Viability separators; case-insensitive ('X' counts as 'x')."""
    return ch in _SEPARATORS or (ch.isascii() and ch.lower() in _SEPARATORS)


def _decimal_value(ch: str) -> int:
    """Decimal value of a Unicode digit; ASCII fast path."""
    o = ord(ch)
    if 48 <= o <= 57:
        return o - 48
    if 0xFF10 <= o <= 0xFF19:  # full-width
        return o - 0xFF10
    if 0x0660 <= o <= 0x0669:  # Arabic-Indic
        return o - 0x0660
    if 0x06F0 <= o <= 0x06F9:  # Extended Arabic-Indic
        return o - 0x06F0
    import unicodedata

    return unicodedata.decimal(ch)


def _normalize(text: str) -> str:
    """The oracle's _normalize: with 3+ ASCII letters, map every letter to
    its keypad digit; otherwise keep (decimal) digits only."""
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if letters >= 3:
        out = []
        for ch in text:
            if ch.isdigit():
                out.append(str(_decimal_value(ch)))
            else:
                d = _ALPHA_TO_DIGIT.get(ch)
                if d is not None:
                    out.append(d)
        return "".join(out)
    return "".join(str(_decimal_value(ch)) for ch in text if ch.isdigit())


# ------------------------------------------------------------- engine proper


@dataclass
class Parsed:
    cc: int
    nsn: str  # national significant number, leading zeros preserved
    ext: str | None
    ccs: int  # country code source (internal; the oracle reports 0)


LEN_IS_POSSIBLE = 0
LEN_TOO_SHORT = 1
LEN_TOO_LONG = 2
LEN_INVALID_LENGTH = 3
LEN_LOCAL_ONLY = 4

_RFC3966_PHONE_CONTEXT = ";phone-context="
_RFC3966_ISDN_SUBADDRESS = ";isub="
_RFC3966_PREFIX = "tel:"
_MAX_INPUT_LENGTH = 250


def _test_number_length(region: Region, nsn: str) -> int:
    """Possible-length classification against the region's general desc."""
    mask = region.general_mask
    if mask == NO_LEN_CONSTRAINT:
        return LEN_IS_POSSIBLE
    n = len(nsn)
    local = region.general_mask_local
    if local != NO_LEN_CONSTRAINT and (local >> n) & 1:
        return LEN_LOCAL_ONLY
    lengths = [k for k in range(MAX_NSN_LEN + 2) if (mask >> k) & 1]
    if not lengths:
        return LEN_IS_POSSIBLE
    if n < lengths[0]:
        return LEN_TOO_SHORT
    if n > lengths[-1]:
        return LEN_TOO_LONG
    if (mask >> n) & 1:
        return LEN_IS_POSSIBLE
    return LEN_INVALID_LENGTH


def _expand_template(template: str, s: str, caps: list[int] | None) -> str:
    """Expand $1..$9 group references against capture slots."""
    out = []
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == "$" and i + 1 < len(template) and template[i + 1].isdigit():
            g = int(template[i + 1])
            if caps is not None and 2 * g + 1 < len(caps):
                a, b = caps[2 * g], caps[2 * g + 1]
                if a >= 0 and b >= a:
                    out.append(s[a:b])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _npfp_candidate(tables: Tables, region: Region, nsn: str) -> str:
    """Inner national-prefix rule (oracle _maybe_strip_national_prefix...):

    Prefix-match nationalPrefixForParsing with captures. When the pattern's
    LAST capture group participated and a transform rule exists, the
    candidate is the transform expansion of the matched span plus the tail;
    otherwise the matched prefix is dropped verbatim. The original is kept
    when it full-matches the general pattern and the candidate does not.
    """
    if region.npfp_prog < 0 or not nsn:
        return nsn
    matched, end, caps = prefix_match(tables, region.npfp_prog, nsn, want_caps=True)
    if not matched:
        return nsn
    prog = tables.programs[region.npfp_prog]
    last_group_participated = (
        caps is not None and prog.n_groups >= 1 and caps[2 * prog.n_groups] >= 0
    )
    if region.np_transform is not None and last_group_participated:
        candidate = _expand_template(region.np_transform, nsn, caps) + nsn[end:]
    else:
        candidate = nsn[end:]
    if candidate == nsn:
        return nsn
    if full_match(tables, region.general_prog, nsn)[0] and not full_match(
        tables, region.general_prog, candidate
    )[0]:
        return nsn
    return candidate


def _extract_cc(tables: Tables, digits: str) -> tuple[int, str] | None:
    """Split a leading country calling code (1-3 digits) from `digits`.
    Codes never begin with '0'."""
    if not digits or digits[0] == "0":
        return None
    for k in (1, 2, 3):
        if k <= len(digits):
            cc = int(digits[:k])
            if cc in tables.cc_map:
                return cc, digits[k:]
    return None


def _type_helper_known(tables: Tables, region: Region, nsn: str) -> bool:
    """The oracle's number-type check: general desc (lengths + pattern) must
    match, and at least one type desc (lengths + pattern) must match."""
    if region.general_mask != NO_LEN_CONSTRAINT and not (
        region.general_mask >> len(nsn)
    ) & 1:
        return False
    if not full_match(tables, region.general_prog, nsn)[0]:
        return False
    ln = len(nsn)
    for prog, mask in region.types:
        if prog < 0:
            continue
        if mask != NO_LEN_CONSTRAINT and not (mask >> ln) & 1:
            continue
        if full_match(tables, prog, nsn)[0]:
            return True
    return False


def _region_for_number(tables: Tables, cc: int, nsn: str) -> Region | None:
    """Region resolution among regions sharing a calling code: first whose
    leadingDigits prefix-matches, else first whose type check passes, else
    None (matches the oracle; there is no main-region fallback here)."""
    idxs = tables.cc_map.get(cc)
    if not idxs:
        return None
    if len(idxs) == 1:
        return tables.regions[idxs[0]]
    for i in idxs:
        r = tables.regions[i]
        if r.leading_digits_prog >= 0:
            if prefix_match(tables, r.leading_digits_prog, nsn)[0]:
                return r
        elif _type_helper_known(tables, r, nsn):
            return r
    return None


def is_possible(tables: Tables, cc: int, nsn: str) -> bool:
    """Possible = length admitted by the MAIN region's general national or
    local-only lengths (the oracle deliberately uses the main region)."""
    idxs = tables.cc_map.get(cc)
    if not idxs:
        return False
    return _test_number_length(tables.regions[idxs[0]], nsn) in (
        LEN_IS_POSSIBLE,
        LEN_LOCAL_ONLY,
    )


def is_valid(tables: Tables, cc: int, nsn: str) -> bool:
    region = _region_for_number(tables, cc, nsn)
    if region is None:
        return False
    return _type_helper_known(tables, region, nsn)


def is_supported_region(tables: Tables, code: str) -> bool:
    """Regions accepted as a default region (case-sensitive; "001" excluded)."""
    idx = tables.region_by_code.get(code)
    return idx is not None and tables.regions[idx].code != "001"


def _build_national_number_for_parsing(text: str) -> str:
    """RFC3966 phone-context/isdn handling, else extract_possible_number."""
    idx_pc = text.find(_RFC3966_PHONE_CONTEXT)
    if idx_pc >= 0:
        pc_start = idx_pc + len(_RFC3966_PHONE_CONTEXT)
        pc_end = text.find(";", pc_start)
        phone_context = text[pc_start:] if pc_end < 0 else text[pc_start:pc_end]
        if not phone_context:
            raise ParseError(NOT_A_NUMBER, "The phone-context value is invalid")
        national = phone_context if phone_context[0] == "+" else ""
        idx_tel = text.find(_RFC3966_PREFIX)
        national_start = idx_tel + len(_RFC3966_PREFIX) if idx_tel >= 0 else 0
        national += text[national_start:idx_pc]
    else:
        national = _extract_possible_number(text)
    idx_isub = national.find(_RFC3966_ISDN_SUBADDRESS)
    if idx_isub > 0:
        national = national[:idx_isub]
    return national


def _first_digit_is_zero(s: str) -> bool:
    for ch in s:
        if ch.isdigit():
            return _decimal_value(ch) == 0
    return False


def _maybe_extract_country_code(
    tables: Tables, s: str, region: Region | None
) -> tuple[int, str, int]:
    """Returns (country_code, national_number, country_code_source).

    Mirrors the oracle: '+' handling, IDD strip (skipped when the first digit
    after the prefix is 0), and the default-country code strip gated by
    general-pattern viability / too-long.
    """
    if not s:
        return (0, "", 20)
    ccs = 20  # FROM_DEFAULT_COUNTRY
    if s[0] in "+＋":
        full = _normalize(s.lstrip("+＋"))
        ccs = 1  # FROM_NUMBER_WITH_PLUS_SIGN
    else:
        full = _normalize(s)
        if region is not None and region.intl_prefix_prog >= 0:
            matched, end, _ = prefix_match(tables, region.intl_prefix_prog, full)
            if matched and end > 0 and not _first_digit_is_zero(full[end:]):
                full = full[end:]
                ccs = 5  # FROM_NUMBER_WITH_IDD
    if ccs != 20:
        if len(full) <= MIN_NSN_LEN:
            raise ParseError(
                TOO_SHORT_AFTER_IDD,
                "Phone number had an IDD, but after this was not long enough "
                "to be a viable phone number.",
            )
        got = _extract_cc(tables, full)
        if got is not None:
            return got[0], got[1], ccs
        raise ParseError(
            INVALID_COUNTRY_CODE, "Country calling code supplied was not recognised."
        )
    # Default-country path: strip the region's own calling code when the
    # remainder is a better candidate than the whole.
    if region is not None:
        cc_str = str(region.cc)
        if full.startswith(cc_str):
            potential = _npfp_candidate(tables, region, full[len(cc_str):])
            full_ok = full_match(tables, region.general_prog, full)[0]
            potential_ok = full_match(tables, region.general_prog, potential)[0]
            too_long = _test_number_length(region, full) == LEN_TOO_LONG
            if (not full_ok and potential_ok) or too_long:
                return region.cc, potential, 10
    return 0, full, 20


def parse(tables: Tables, text: str, region_code: str | None) -> Parsed:
    if text is None:
        raise ParseError(NOT_A_NUMBER, "The phone number supplied was None.")
    if not isinstance(text, str):
        raise TypeError("number must be a string")
    if len(text) > _MAX_INPUT_LENGTH:
        raise ParseError(TOO_LONG, "The string supplied was too long to parse.")

    s = _build_national_number_for_parsing(text)
    if not s or not is_viable(s):
        raise ParseError(NOT_A_NUMBER, "The string supplied did not seem to be a phone number.")
    region_ok = region_code is not None and is_supported_region(tables, region_code)
    if not region_ok and not s.startswith(("+", "\uff0b")):
        raise ParseError(INVALID_COUNTRY_CODE, "Missing or invalid default region.")
    ext, s = strip_extension(s)

    region: Region | None = None
    if region_code is not None:
        idx = tables.region_by_code.get(region_code)
        if idx is not None and tables.regions[idx].code != "001":
            region = tables.regions[idx]

    try:
        cc, national, ccs = _maybe_extract_country_code(tables, s, region)
    except ParseError as exc:
        if exc.error_type == INVALID_COUNTRY_CODE and s.startswith(("+", "\uff0b")):
            # Strip the plus sign(s) and try again without them.
            cc2, national2, _ = _maybe_extract_country_code(tables, s.lstrip("+\uff0b"), region)
            if cc2 == 0:
                raise ParseError(
                    INVALID_COUNTRY_CODE, "Could not interpret numbers after plus-sign."
                ) from None
            cc, national, ccs = cc2, national2, 10
        else:
            raise

    # Metadata for the national-prefix strip: the default region when no
    # calling code was extracted, else the main region for the code (when it
    # differs from the supplied region).
    metadata = region
    if cc != 0:
        main_idxs = tables.cc_map.get(cc)
        main_region = tables.regions[main_idxs[0]] if main_idxs else None
        if main_region is not None and main_region.code != region_code:
            metadata = main_region
    elif region is not None:
        cc = region.cc

    if len(national) < MIN_NSN_LEN:
        raise ParseError(TOO_SHORT_NSN, "The string supplied is too short to be a phone number.")
    if metadata is not None:
        candidate = _npfp_candidate(tables, metadata, national)
        if _test_number_length(metadata, candidate) not in (
            LEN_TOO_SHORT,
            LEN_LOCAL_ONLY,
            LEN_INVALID_LENGTH,
        ):
            national = candidate
    if len(national) < MIN_NSN_LEN:
        raise ParseError(TOO_SHORT_NSN, "The string supplied is too short to be a phone number.")
    if len(national) > MAX_NSN_LEN:
        raise ParseError(TOO_LONG, "The string supplied is too long to be a phone number.")
    return Parsed(cc=cc, nsn=national, ext=ext or None, ccs=ccs)


# --------------------------------------------------------------- formatting


def format_number(tables: Tables, cc: int, nsn: str, ext: str | None, fmt: int) -> str:
    idxs = tables.cc_map.get(cc)
    if not idxs:
        # Unknown calling code: the oracle returns the raw NSN for
        # NATIONAL/INTERNATIONAL and "+cc nsn" for E164.
        return f"+{cc}{nsn}" if fmt == FMT_E164 else nsn
    # Formatting metadata is shared per calling code and lives on the MAIN
    # region (e.g. US for all of NANPA) — the oracle formats via
    # region_code_for_country_code, not the number's own region.
    region = tables.regions[idxs[0]]
    if fmt == FMT_E164:
        return f"+{cc}{nsn}"
    if fmt == FMT_INTERNATIONAL:
        formatted = _format_nsn(tables, region, nsn, international=True)
        body = f"+{cc} {formatted}"
    else:  # FMT_NATIONAL
        body = _format_nsn(tables, region, nsn, international=False)
    if ext:
        prefix = region.preferred_extn_prefix or " ext. "
        body = body + prefix + ext
    return body


def _format_nsn(tables: Tables, region: Region, nsn: str, international: bool) -> str:
    fmts = region.intl_formats if (international and region.intl_formats) else region.formats
    for pattern_prog, template, np_rule, ld_prog in fmts:
        if ld_prog >= 0 and not prefix_match(tables, ld_prog, nsn)[0]:
            continue
        matched, _end, caps = full_match(tables, pattern_prog, nsn, want_caps=True)
        if not matched:
            continue
        rule = template
        if not international and np_rule:
            np = region.national_prefix or ""
            # "$NP" is the national prefix; "$FG" expands to whatever the
            # template's FIRST group reference captures (e.g. for AR mobile
            # "$2 15-$3-$4" with rule "$NP$FG" -> "0" + group2).
            first_ref = _first_group_ref(template)
            fg = f"${first_ref}" if first_ref else ""
            rule_processed = np_rule.replace("$NP", np).replace("$FG", fg)
            rule = _replace_first_group_ref(template, rule_processed)
        return _expand_template(rule, nsn, caps)
    return nsn


def _first_group_ref(template: str) -> int | None:
    i = 0
    while i + 1 < len(template):
        if template[i] == "$" and template[i + 1].isdigit():
            return int(template[i + 1])
        i += 1
    return None


def _replace_first_group_ref(template: str, replacement: str) -> str:
    i = 0
    while i + 1 < len(template):
        if template[i] == "$" and template[i + 1].isdigit():
            return template[:i] + replacement + template[i + 2:]
        i += 1
    return template
