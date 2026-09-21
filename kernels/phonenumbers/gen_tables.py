#!/usr/bin/env python3
"""Compile libphonenumber PhoneNumberMetadata.xml into the phonenumbers-mojo blob.

Clean-room: this generator consumes only the metadata XML (Apache-2.0 *data*
from the libphonenumber project, vendored under ./data/ with its license
notice) and emits a compact binary table that both the Mojo kernel and the
pure-Python fallback parse at load time. No libphonenumber code is read,
adapted, or embedded; the matching engine below is an independent
implementation of a Thompson NFA over a digit alphabet.

Regex subset supported (verified against the 9.0.39 metadata): digit
literals, ``\\d``, character classes with ranges/negation, ``(...)`` and
``(?:...)`` groups, alternation, ``? * + {n} {n,} {n,m}`` quantifiers, and
the ``$`` end assertion (used by nationalPrefixForParsing). Every pattern
operates on digit strings: parse normalizes input to digits before any
pattern runs, so classes compile to 10-bit digit masks.

Blob layout (all integers little-endian u32 unless noted)::

    header:  magic "PNM1", u32 version, u32 n_programs, u32 n_regions,
             u32 n_strings, u32 cc_map_off, u32 cc_map_count
    program offsets:  u32[n_programs]
    string offsets:   u32[n_strings]  (each string: u32 byte_len + utf8)
    region offsets:   u32[n_regions]
    cc_map @ cc_map_off: per entry u32 cc, u32 n_regions, u32 region_idx[]
    program:  u32 n_insts, u32 n_groups, then n_insts * (i32 op, i32 x, i32 y)
              ops: 0=CLASS(x=10-bit mask) 1=SPLIT(x,y) 2=JMP(x)
                   3=SAVE(x=slot) 4=MATCH 5=EOL
    region record:
      u16 code_str_idx, u16 country_code, u8 main_for_code, u8 reserved
      i32 intl_prefix_prog, i32 national_prefix_str,
      i32 np_for_parsing_prog, i32 np_transform_str,
      i32 np_formatting_rule_str (territory-level; unused today),
      i32 leading_digits_prog, i32 preferred_extn_prefix_str,
      i32 general_prog, u32 general_lenmask, u32 general_lenmask_local,
      10 x (i32 type_prog, u32 lenmask, u32 lenmask_local),
      u32 n_formats, then per format:
        i32 pattern_prog, i32 format_str, i32 np_rule_str, i32 ld_prog
      u32 n_intl_formats, same record shape

Length masks: bit k set => a national significant number of length k is
possible. 0xFFFFFFFF means "no constraint recorded".

Usage: python3 gen_tables.py  (writes ../../python/phonenumbers_mojo/phonenumbers_mojo/_data.py)
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
XML_PATH = HERE / "data" / "PhoneNumberMetadata.xml"
OUT_PY = HERE.parent.parent / "python" / "phonenumbers_mojo" / "phonenumbers_mojo" / "_data.py"

BLOB_VERSION = 1
NO_LEN_CONSTRAINT = 0xFFFFFFFF
MAX_NSN_LEN = 17  # ITU-T E.164 significant digits cap, mirrored by the oracle

# Type descs in a fixed order; validity = general match AND any type match.
TYPE_TAGS = [
    "fixedLine", "mobile", "pager", "uan", "tollFree",
    "premiumRate", "sharedCost", "personalNumber", "voip", "voicemail",
]

# ---------------------------------------------------------------- regex parse


class RegexError(ValueError):
    pass


def _strip_ws(pattern: str) -> str:
    """The metadata XML pretty-prints patterns; all whitespace is insignificant."""
    return "".join(pattern.split())


def parse_regex(pattern: str):
    """Parse the metadata regex subset into an AST.

    AST nodes (tuples):
      ('cls', mask)          10-bit digit mask
      ('eol',)
      ('grp', cap|None, node)
      ('alt', [nodes])
      ('cat', [nodes])
      ('rep', node, min, max|None)
    """
    src = _strip_ws(pattern)
    pos = 0
    group_counter = [0]  # mutable cell; group 0 is the whole match

    def peek() -> str:
        return src[pos] if pos < len(src) else ""

    def expect(ch: str) -> None:
        nonlocal pos
        if peek() != ch:
            raise RegexError(f"expected {ch!r} at {pos} in {src!r}")
        pos += 1

    def parse_class() -> tuple[str, int]:
        nonlocal pos
        expect("[")
        negate = False
        if peek() == "^":
            negate = True
            pos += 1
        mask = 0

        def one_char() -> int:
            nonlocal pos
            ch = src[pos]
            if ch == "\\":
                pos += 1
                esc = src[pos]
                pos += 1
                if esc == "d":
                    return -1  # sentinel: all digits
                if esc in "dwsSDWbB":
                    raise RegexError(f"unsupported escape \\{esc} in class in {src!r}")
                esc = {"n": "\n", "t": "\t", "r": "\r", "f": "\f"}.get(esc, esc)
                return ord(esc)
            pos += 1
            return ord(ch)

        while peek() != "]":
            if pos >= len(src):
                raise RegexError(f"unterminated class in {src!r}")
            lo = one_char()
            if lo == -1:
                mask |= 0x3FF
                continue
            if peek() == "-" and pos + 1 < len(src) and src[pos + 1] != "]":
                pos += 1  # consume '-'
                hi = one_char()
                if hi == -1:
                    raise RegexError(f"bad range endpoint in {src!r}")
                for cp in range(lo, hi + 1):
                    _mask_digit(mask_holder := [mask], cp)
                    mask = mask_holder[0]
            else:
                _mask_digit(mask_holder := [mask], lo)
                mask = mask_holder[0]
        expect("]")
        if negate:
            mask ^= 0x3FF
        return ("cls", mask)

    def parse_atom():
        nonlocal pos
        ch = peek()
        if ch == "(":
            pos += 1
            cap = None
            if src.startswith("?:", pos):
                pos += 2
            else:
                group_counter[0] += 1
                cap = group_counter[0]
            node = parse_alt()
            expect(")")
            return ("grp", cap, node)
        if ch == "[":
            return parse_class()
        if ch == "$":
            pos += 1
            return ("eol",)
        if ch == "\\":
            pos += 1
            esc = src[pos]
            pos += 1
            if esc == "d":
                return ("cls", 0x3FF)
            if esc in "wsSDWbB":
                raise RegexError(f"unsupported escape \\{esc} in {src!r}")
            esc = {"n": "\n", "t": "\t", "r": "\r", "f": "\f"}.get(esc, esc)
            if not esc.isdigit() or not esc.isascii():
                raise RegexError(f"non-digit literal {esc!r} in {src!r}")
            return ("cls", 1 << int(esc))
        if ch and ch not in ")|?*+{":
            pos += 1
            if not ch.isdigit() or not ch.isascii():
                raise RegexError(f"non-digit literal {ch!r} in {src!r}")
            return ("cls", 1 << int(ch))
        raise RegexError(f"unexpected {ch!r} at {pos} in {src!r}")

    def parse_repeat():
        nonlocal pos
        atom = parse_atom()
        while True:
            ch = peek()
            if ch == "?":
                pos += 1
                atom = ("rep", atom, 0, 1)
            elif ch == "*":
                pos += 1
                atom = ("rep", atom, 0, None)
            elif ch == "+":
                pos += 1
                atom = ("rep", atom, 1, None)
            elif ch == "{":
                end = src.index("}", pos)
                body = src[pos + 1:end]
                if "," in body:
                    lo_s, hi_s = body.split(",", 1)
                    lo = int(lo_s)
                    hi = int(hi_s) if hi_s else None
                else:
                    lo = hi = int(body)
                if hi is not None and hi < lo:
                    raise RegexError(f"bad quantifier {{{body}}} in {src!r}")
                atom = ("rep", atom, lo, hi)
                pos = end + 1
            else:
                return atom

    def parse_cat():
        nodes = []
        while pos < len(src) and peek() not in ")":
            if peek() == "|":
                break
            nodes.append(parse_repeat())
        if not nodes:
            return ("cat", [])
        if len(nodes) == 1:
            return nodes[0]
        return ("cat", nodes)

    def parse_alt():
        nonlocal pos
        branches = [parse_cat()]
        while peek() == "|":
            pos += 1
            branches.append(parse_cat())
        if len(branches) == 1:
            return branches[0]
        return ("alt", branches)

    ast = parse_alt()
    if pos != len(src):
        raise RegexError(f"trailing garbage at {pos} in {src!r}")
    return ast, group_counter[0]


def _mask_digit(holder: list[int], cp: int) -> None:
    ch = chr(cp)
    if not ch.isdigit() or not ch.isascii():
        raise RegexError(f"non-digit class member {ch!r}")
    holder[0] |= 1 << int(ch)


# ------------------------------------------------------------- NFA compiler

OP_CLS, OP_SPLIT, OP_JMP, OP_SAVE, OP_MATCH, OP_EOL = range(6)


def compile_program(ast, n_groups: int) -> list[tuple[int, int, int]]:
    """Thompson construction with backtracking-priority SPLIT ordering."""
    insts: list[list[int]] = []

    def emit(op: int, x: int = 0, y: int = 0) -> int:
        insts.append([op, x, y])
        return len(insts) - 1

    def comp(node) -> None:
        kind = node[0]
        if kind == "cls":
            emit(OP_CLS, node[1])
        elif kind == "eol":
            emit(OP_EOL)
        elif kind == "grp":
            _, cap, sub = node
            if cap is not None:
                emit(OP_SAVE, 2 * cap)
            comp(sub)
            if cap is not None:
                emit(OP_SAVE, 2 * cap + 1)
        elif kind == "cat":
            for sub in node[1]:
                comp(sub)
        elif kind == "alt":
            branches = node[1]
            jmp_fixups = []
            for i, branch in enumerate(branches):
                if i + 1 < len(branches):
                    split_at = emit(OP_SPLIT, 0, 0)
                    insts[split_at][1] = len(insts)  # x: this branch
                    comp(branch)
                    jmp_fixups.append(emit(OP_JMP, 0))
                    insts[split_at][2] = len(insts)  # y: next branch
                else:
                    comp(branch)
            end = len(insts)
            for at in jmp_fixups:
                insts[at][1] = end
        elif kind == "rep":
            _, sub, lo, hi = node[0], node[1], node[2], node[3]
            for _ in range(lo):
                comp(sub)
            if hi is None:
                # star/plus tail: greedy loop
                split_at = emit(OP_SPLIT, 0, 0)
                insts[split_at][1] = len(insts)
                comp(sub)
                emit(OP_JMP, split_at)
                insts[split_at][2] = len(insts)
            else:
                for _ in range(hi - lo):
                    split_at = emit(OP_SPLIT, 0, 0)
                    insts[split_at][1] = len(insts)
                    comp(sub)
                    insts[split_at][2] = len(insts)
        else:  # pragma: no cover
            raise AssertionError(kind)

    # Whole match is capture group 0.
    emit(OP_SAVE, 0)
    comp(ast)
    emit(OP_SAVE, 1)
    emit(OP_MATCH)
    return [tuple(i) for i in insts]


# ------------------------------------------------------------ blob assembly


class StringTable:
    def __init__(self) -> None:
        self._idx: dict[str, int] = {}
        self.strings: list[str] = []

    def intern(self, s: str | None) -> int:
        if s is None:
            return -1
        if s not in self._idx:
            self._idx[s] = len(self.strings)
            self.strings.append(s)
        return self._idx[s]


class ProgramTable:
    def __init__(self) -> None:
        self._idx: dict[tuple, int] = {}
        self.programs: list[tuple[list[tuple[int, int, int]], int]] = []

    def compile(self, pattern: str | None) -> int:
        """Compile a metadata pattern; returns program index or -1 for None."""
        if pattern is None:
            return -1
        ast, n_groups = parse_regex(pattern)
        insts = compile_program(ast, n_groups)
        key = tuple(insts)
        if key not in self._idx:
            self._idx[key] = len(self.programs)
            self.programs.append((insts, n_groups))
        return self._idx[key]


def _possible_len_mask(el: ElementTree.Element | None, attr: str) -> int:
    """possibleLengths mask for `national`/`localOnly`; entries may be ranges
    like ``[5-12]`` mixed with plain lengths. NO_LEN_CONSTRAINT when absent."""
    if el is None:
        return NO_LEN_CONSTRAINT
    pl = el.find("possibleLengths")
    if pl is None:
        return NO_LEN_CONSTRAINT
    spec = pl.get(attr)
    if not spec:
        return NO_LEN_CONSTRAINT
    mask = 0
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("[") and tok.endswith("]"):
            lo_s, hi_s = tok[1:-1].split("-", 1)
            for n in range(int(lo_s), int(hi_s) + 1):
                mask |= 1 << n
        else:
            mask |= 1 << int(tok)
    return mask


def _len_mask(el: ElementTree.Element | None) -> int:
    return _possible_len_mask(el, "national")


def _len_mask_local(el: ElementTree.Element | None) -> int:
    return _possible_len_mask(el, "localOnly")


def build_regions(root, progs: ProgramTable, strs: StringTable):
    regions = []
    for terr in root.iter("territory"):
        tid = terr.get("id")
        cc = int(terr.get("countryCode"))
        main = 1 if terr.get("mainCountryForCode") == "true" else 0

        gd = terr.find("generalDesc")
        gd_pat_el = gd.find("nationalNumberPattern") if gd is not None else None
        gd_pat = gd_pat_el.text if gd_pat_el is not None else None
        general_prog = progs.compile(gd_pat)
        general_mask = _len_mask(gd)
        general_mask_local = _len_mask_local(gd)

        # When the XML omits nationalPrefixForParsing, the reference metadata
        # build sets it equal to nationalPrefix (verified against the 9.0.39
        # cooked data, e.g. DE "0" -> npfp "0").
        npfp_text = terr.get("nationalPrefixForParsing")
        if npfp_text is None:
            npfp_text = terr.get("nationalPrefix")

        type_records = []
        for tag in TYPE_TAGS:
            el = terr.find(tag)
            pat_el = el.find("nationalNumberPattern") if el is not None else None
            prog = progs.compile(pat_el.text if pat_el is not None else None)
            type_records.append((prog, _len_mask(el), _len_mask_local(el)))

        if general_mask == NO_LEN_CONSTRAINT:
            union = 0
            for _, m, _ in type_records:
                if m != NO_LEN_CONSTRAINT:
                    union |= m
            general_mask = union if union else NO_LEN_CONSTRAINT
        if general_mask_local == NO_LEN_CONSTRAINT:
            union = 0
            for _, _, m in type_records:
                if m != NO_LEN_CONSTRAINT:
                    union |= m
            general_mask_local = union if union else NO_LEN_CONSTRAINT
        # The reference metadata build guarantees national and local-only
        # lengths never overlap: local-only entries that collide with a
        # national length are dropped at build time (verified across all 254
        # regions in 9.0.39 — this is why e.g. GB 7-digit candidates count as
        # IS_POSSIBLE while US 7-digit candidates count as LOCAL_ONLY).
        if general_mask != NO_LEN_CONSTRAINT and general_mask_local != NO_LEN_CONSTRAINT:
            general_mask_local &= ~general_mask
            if general_mask_local == 0:
                general_mask_local = NO_LEN_CONSTRAINT

        def compile_formats(intl: bool) -> list[tuple[int, int, int, int]]:
            out = []
            avail = terr.find("availableFormats")
            if avail is None:
                return out
            for nf in avail.findall("numberFormat"):
                intl_el = nf.find("intlFormat")
                fmt_el = nf.find("format")
                if intl:
                    if intl_el is not None and (intl_el.text or "").strip() == "NA":
                        continue
                    template = (
                        intl_el.text if intl_el is not None and intl_el.text else fmt_el.text
                    )
                else:
                    template = fmt_el.text if fmt_el is not None else None
                if template is None:
                    raise RegexError(f"numberFormat without <format> in {tid}")
                pattern_prog = progs.compile(nf.get("pattern"))
                format_str = strs.intern(template)
                np_rule = strs.intern(nf.get("nationalPrefixFormattingRule"))
                lds = nf.findall("leadingDigits")
                # The reference engine consults only the LAST leadingDigits
                # pattern for eligibility (verified against the 9.0.39 data).
                ld_prog = progs.compile(lds[-1].text) if lds else -1
                out.append((pattern_prog, format_str, np_rule, ld_prog))
            return out

        regions.append(
            {
                "id_str": strs.intern(tid),
                "id": tid,
                "cc": cc,
                "main": main,
                "intl_prefix_prog": progs.compile(terr.get("internationalPrefix")),
                "national_prefix_str": strs.intern(terr.get("nationalPrefix")),
                "npfp_prog": progs.compile(npfp_text),
                "np_transform_str": strs.intern(terr.get("nationalPrefixTransformRule")),
                "np_rule_str": strs.intern(terr.get("nationalPrefixFormattingRule")),
                "leading_digits_prog": progs.compile(terr.get("leadingDigits")),
                "pref_extn_str": strs.intern(terr.get("preferredExtnPrefix")),
                "general_prog": general_prog,
                "general_mask": general_mask,
                "general_mask_local": general_mask_local,
                "types": type_records,
                "formats": compile_formats(False),
                "intl_formats": compile_formats(True),
            }
        )
    return regions


def serialize(progs: ProgramTable, strs: StringTable, regions: list[dict]) -> bytes:
    # cc -> region idxs, main region first then XML document order
    cc_map: dict[int, list[int]] = {}
    for i, r in enumerate(regions):
        cc_map.setdefault(r["cc"], []).append(i)
    for cc, idxs in cc_map.items():
        idxs.sort(key=lambda i: (0 if regions[i]["main"] else 1, i))

    out = bytearray()
    n_programs, n_regions, n_strings = len(progs.programs), len(regions), len(strs.strings)
    header = struct.pack(
        "<4sIIIIII", b"PNM1", BLOB_VERSION, n_programs, n_regions, n_strings, 0, 0
    )
    out += header
    cc_map_off_pos = 20  # offset of cc_map_off field within header
    cc_map_count_pos = 24

    prog_offsets_pos = len(out)
    out += b"\x00" * (4 * n_programs)
    str_offsets_pos = len(out)
    out += b"\x00" * (4 * n_strings)
    reg_offsets_pos = len(out)
    out += b"\x00" * (4 * n_regions)

    prog_offsets = []
    for insts, n_groups in progs.programs:
        prog_offsets.append(len(out))
        out += struct.pack("<II", len(insts), n_groups)
        for op, x, y in insts:
            out += struct.pack("<iii", op, x, y)

    str_offsets = []
    for s in strs.strings:
        str_offsets.append(len(out))
        b = s.encode("utf-8")
        out += struct.pack("<I", len(b))
        out += b

    reg_offsets = []
    for r in regions:
        reg_offsets.append(len(out))
        out += struct.pack(
            "<HHBB",
            r["id_str"],
            r["cc"],
            r["main"],
            0,
        )
        out += struct.pack(
            "<8i",
            r["intl_prefix_prog"],
            r["national_prefix_str"],
            r["npfp_prog"],
            r["np_transform_str"],
            r["np_rule_str"],
            r["leading_digits_prog"],
            r["pref_extn_str"],
            r["general_prog"],
        )
        out += struct.pack("<II", r["general_mask"], r["general_mask_local"])
        for prog, mask, mask_local in r["types"]:
            out += struct.pack("<iII", prog, mask, mask_local)
        for fmt_list in (r["formats"], r["intl_formats"]):
            out += struct.pack("<I", len(fmt_list))
            for pattern_prog, format_str, np_rule, ld_prog in fmt_list:
                out += struct.pack("<4i", pattern_prog, format_str, np_rule, ld_prog)

    cc_map_off = len(out)
    for cc in sorted(cc_map):
        idxs = cc_map[cc]
        out += struct.pack("<II", cc, len(idxs))
        for i in idxs:
            out += struct.pack("<I", i)

    # patch offset tables + header cc_map fields
    for i, off in enumerate(prog_offsets):
        struct.pack_into("<I", out, prog_offsets_pos + 4 * i, off)
    for i, off in enumerate(str_offsets):
        struct.pack_into("<I", out, str_offsets_pos + 4 * i, off)
    for i, off in enumerate(reg_offsets):
        struct.pack_into("<I", out, reg_offsets_pos + 4 * i, off)
    struct.pack_into("<I", out, cc_map_off_pos, cc_map_off)
    struct.pack_into("<I", out, cc_map_count_pos, len(cc_map))
    return bytes(out)


def main() -> None:
    root = ElementTree.parse(XML_PATH).getroot()
    progs, strs = ProgramTable(), StringTable()
    regions = build_regions(root, progs, strs)
    blob = serialize(progs, strs, regions)
    payload = base64.b85encode(zlib.compress(blob, 9)).decode("ascii")
    lines = [payload[i : i + 100] for i in range(0, len(payload), 100)]
    n_cc = len({r['cc'] for r in regions})
    OUT_PY.parent.mkdir(parents=True, exist_ok=True)
    OUT_PY.write_text(
        '"""Generated by kernels/phonenumbers/gen_tables.py — do not edit.\n\n'
        "The blob below compiles the libphonenumber 9.0.39 phone-numbering\n"
        "metadata (Apache-2.0 data, (c) The Libphonenumber Authors; see\n"
        "kernels/phonenumbers/data/NOTICE) into the table format documented in\n"
        "gen_tables.py: digit-mask NFA programs, possible-length masks and\n"
        "format templates for every region. Both the Mojo kernel and the\n"
        "pure-Python fallback in this package parse this exact blob.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import zlib\n\n"
        "BLOB_B85 = (\n"
        + "".join(f'    b"{line}"\n' for line in lines)
        + ")\n\n"
        "BLOB_VERSION = 1\n\n\n"
        "def load_blob() -> bytes:\n"
        "    return zlib.decompress(base64.b85decode(BLOB_B85))\n",
        encoding="utf-8",
    )
    print(
        f"regions={len(regions)} ccs={n_cc} programs={len(progs.programs)} "
        f"strings={len(strs.strings)} blob={len(blob)}B "
        f"packed={len(payload)}B -> {OUT_PY}"
    )


if __name__ == "__main__":
    sys.exit(main())
