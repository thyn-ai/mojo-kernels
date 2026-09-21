"""Clean-room Markdown -> HTML engine reproducing ``mistune.markdown()`` (3.3.4).

Written fresh from the published CommonMark 0.31.2 spec plus black-box
observation of the published ``mistune`` package (probe inputs -> observed
outputs; no mistune source was read or adapted). The default
``mistune.markdown()`` configuration is *not* plain CommonMark; this module
mirrors its observable deviations:

- no raw HTML: ``<...>`` in text is escaped, there are no HTML blocks
- no entity decoding in text (``&copy;`` -> ``&amp;copy;``); entities *are*
  decoded inside link destinations and titles (semicolon required, HTML5
  named table, numeric references follow the HTML5 replacement rules)
- fenced code keeps its trailing newline; indented code does not
- ``<li>`` is never followed by a newline; every other block-level element
  ends its rendered output with ``\\n``
- link/image/autolink URLs are scheme-checked against an allowlist
  (http, https, ftp, ftps, irc, ircs, mailto, tel); anything else renders
  as ``#harmful-link``
- quote continuation without ``>`` absorbs plain text and ATX headings but
  not fences/lists/thematic breaks; setext underlines are not recognised on
  quote-lazy lines but are on item-lazy lines

Both the native kernel and this fallback implement the same rules, so the
two backends agree on every input.
"""

from __future__ import annotations

import re
from html.entities import html5 as _HTML5_ENTITIES

# --------------------------------------------------------------------------
# character classes (shared, in spirit, with the Mojo kernel: identical
# classification tables keep the two backends byte-identical)
# --------------------------------------------------------------------------

# CommonMark "Unicode whitespace": Zs + tab/LF/FF/CR + U+2028/U+2029.
_WS_CODEPOINTS = frozenset(
    " \t\n\x0b\x0c\r"
    "             \u2028\u2029  　"
)


def _is_ws(ch: str) -> bool:
    return ch in _WS_CODEPOINTS


# Unicode general category P* (punctuation), as (lo, hi) inclusive ranges.
_PUNCT_RANGES = (
    (33, 47), (58, 64), (91, 96), (123, 126),
    (161, 169), (171, 172), (174, 177), (180, 180),
    (182, 184), (187, 187), (191, 191), (215, 215),
    (247, 247), (706, 709), (722, 735), (741, 747),
    (749, 749), (751, 767), (885, 885), (894, 894),
    (900, 901), (903, 903), (1014, 1014), (1154, 1154),
    (1370, 1375), (1417, 1418), (1421, 1423), (1470, 1470),
    (1472, 1472), (1475, 1475), (1478, 1478), (1523, 1524),
    (1542, 1551), (1563, 1563), (1565, 1567), (1642, 1645),
    (1748, 1748), (1758, 1758), (1769, 1769), (1789, 1790),
    (1792, 1805), (2038, 2041), (2046, 2047), (2096, 2110),
    (2142, 2142), (2184, 2184), (2404, 2405), (2416, 2416),
    (2546, 2547), (2554, 2555), (2557, 2557), (2678, 2678),
    (2800, 2801), (2928, 2928), (3059, 3066), (3191, 3191),
    (3199, 3199), (3204, 3204), (3407, 3407), (3449, 3449),
    (3572, 3572), (3647, 3647), (3663, 3663), (3674, 3675),
    (3841, 3863), (3866, 3871), (3892, 3892), (3894, 3894),
    (3896, 3896), (3898, 3901), (3973, 3973), (4030, 4037),
    (4039, 4044), (4046, 4058), (4170, 4175), (4254, 4255),
    (4347, 4347), (4960, 4968), (5008, 5017), (5120, 5120),
    (5741, 5742), (5787, 5788), (5867, 5869), (5941, 5942),
    (6100, 6102), (6104, 6107), (6144, 6154), (6464, 6464),
    (6468, 6469), (6622, 6655), (6686, 6687), (6816, 6822),
    (6824, 6829), (7002, 7018), (7028, 7038), (7164, 7167),
    (7227, 7231), (7294, 7295), (7360, 7367), (7379, 7379),
    (8125, 8125), (8127, 8129), (8141, 8143), (8157, 8159),
    (8173, 8175), (8189, 8190), (8208, 8231), (8240, 8286),
    (8314, 8318), (8330, 8334), (8352, 8384), (8448, 8449),
    (8451, 8454), (8456, 8457), (8468, 8468), (8470, 8472),
    (8478, 8483), (8485, 8485), (8487, 8487), (8489, 8489),
    (8494, 8494), (8506, 8507), (8512, 8516), (8522, 8525),
    (8527, 8527), (8586, 8587), (8592, 9254), (9280, 9290),
    (9372, 9449), (9472, 10101), (10132, 11123), (11126, 11157),
    (11159, 11263), (11493, 11498), (11513, 11516), (11518, 11519),
    (11632, 11632), (11776, 11822), (11824, 11869), (11904, 11929),
    (11931, 12019), (12032, 12245), (12272, 12283), (12289, 12292),
    (12296, 12320), (12336, 12336), (12342, 12343), (12349, 12351),
    (12443, 12444), (12448, 12448), (12539, 12539), (12688, 12689),
    (12694, 12703), (12736, 12771), (12800, 12830), (12842, 12871),
    (12880, 12880), (12896, 12927), (12938, 12976), (12992, 13311),
    (19904, 19967), (42128, 42182), (42238, 42239), (42509, 42511),
    (42611, 42611), (42622, 42622), (42738, 42743), (42752, 42774),
    (42784, 42785), (42889, 42890), (43048, 43051), (43062, 43065),
    (43124, 43127), (43214, 43215), (43256, 43258), (43260, 43260),
    (43310, 43311), (43359, 43359), (43457, 43469), (43486, 43487),
    (43612, 43615), (43639, 43641), (43742, 43743), (43760, 43761),
    (43867, 43867), (43882, 43883), (44011, 44011), (64297, 64297),
    (64434, 64450), (64830, 64847), (64975, 64975), (65020, 65023),
    (65040, 65049), (65072, 65106), (65108, 65126), (65128, 65131),
    (65281, 65295), (65306, 65312), (65339, 65344), (65371, 65381),
    (65504, 65510), (65512, 65518), (65532, 65533), (65792, 65794),
    (65847, 65855), (65913, 65929), (65932, 65934), (65936, 65948),
    (65952, 65952), (66000, 66044), (66463, 66463), (66512, 66512),
    (66927, 66927), (67671, 67671), (67703, 67704), (67871, 67871),
    (67903, 67903), (68176, 68184), (68223, 68223), (68296, 68296),
    (68336, 68342), (68409, 68415), (68505, 68508), (69293, 69293),
    (69461, 69465), (69510, 69513), (69703, 69709), (69819, 69820),
    (69822, 69825), (69952, 69955), (70004, 70005), (70085, 70088),
    (70093, 70093), (70107, 70107), (70109, 70111), (70200, 70205),
    (70313, 70313), (70731, 70735), (70746, 70747), (70749, 70749),
    (70854, 70854), (71105, 71127), (71233, 71235), (71264, 71276),
    (71353, 71353), (71484, 71487), (71739, 71739), (72004, 72006),
    (72162, 72162), (72255, 72262), (72346, 72348), (72350, 72354),
    (72448, 72457), (72769, 72773), (72816, 72817), (73463, 73464),
    (73539, 73551), (73685, 73713), (73727, 73727), (74864, 74868),
    (77809, 77810), (92782, 92783), (92917, 92917), (92983, 92991),
    (92996, 92997), (93847, 93850), (94178, 94178), (113820, 113820),
    (113823, 113823), (118608, 118723), (118784, 119029), (119040, 119078),
    (119081, 119140), (119146, 119148), (119171, 119172), (119180, 119209),
    (119214, 119274), (119296, 119361), (119365, 119365), (119552, 119638),
    (120513, 120513), (120539, 120539), (120571, 120571), (120597, 120597),
    (120629, 120629), (120655, 120655), (120687, 120687), (120713, 120713),
    (120745, 120745), (120771, 120771), (120832, 121343), (121399, 121402),
    (121453, 121460), (121462, 121475), (121477, 121483), (123215, 123215),
    (123647, 123647), (125278, 125279), (126124, 126124), (126128, 126128),
    (126254, 126254), (126704, 126705), (126976, 127019), (127024, 127123),
    (127136, 127150), (127153, 127167), (127169, 127183), (127185, 127221),
    (127245, 127405), (127462, 127490), (127504, 127547), (127552, 127560),
    (127568, 127569), (127584, 127589), (127744, 128727), (128732, 128748),
    (128752, 128764), (128768, 128886), (128891, 128985), (128992, 129003),
    (129008, 129008), (129024, 129035), (129040, 129095), (129104, 129113),
    (129120, 129159), (129168, 129197), (129200, 129201), (129280, 129619),
    (129632, 129645), (129648, 129660), (129664, 129672), (129680, 129725),
    (129727, 129733), (129742, 129755), (129760, 129768), (129776, 129784),
    (129792, 129938), (129940, 129994),
)

# ASCII punctuation: backslash-escapable per CommonMark.
_ASCII_PUNCT = frozenset('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if cp < 128:
        return ch in _ASCII_PUNCT
    lo, hi = 0, len(_PUNCT_RANGES) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b = _PUNCT_RANGES[mid]
        if cp < a:
            hi = mid - 1
        elif cp > b:
            lo = mid + 1
        else:
            return True
    return False


# --------------------------------------------------------------------------
# HTML escaping (mistune escapes & < > " everywhere, including code spans)
# --------------------------------------------------------------------------

_ESCAPE_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --------------------------------------------------------------------------
# entity decoding (link destinations and titles only)
# --------------------------------------------------------------------------

# HTML5 numeric character reference replacements for C1 controls.
_NUMERIC_C1 = {
    0x80: 0x20AC, 0x82: 0x201A, 0x83: 0x0192, 0x84: 0x201E, 0x85: 0x2026,
    0x86: 0x2020, 0x87: 0x2021, 0x88: 0x02C6, 0x89: 0x2030, 0x8A: 0x0160,
    0x8B: 0x2039, 0x8C: 0x0152, 0x8E: 0x017D, 0x91: 0x2018, 0x92: 0x2019,
    0x93: 0x201C, 0x94: 0x201D, 0x95: 0x2022, 0x96: 0x2013, 0x97: 0x2014,
    0x98: 0x02DC, 0x99: 0x2122, 0x9A: 0x0161, 0x9B: 0x203A, 0x9C: 0x0153,
    0x9E: 0x017E, 0x9F: 0x0178,
}

_ENTITY_RE = re.compile(r"&(#(?:[xX][0-9a-fA-F]+|[0-9]+)|[a-zA-Z][a-zA-Z0-9]*);")


def _numeric_charref(body: str) -> str:
    if body[1] in "xX":
        cp = int(body[2:], 16)
    else:
        cp = int(body[1:], 10)
    if cp == 0 or cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
        return "�"
    if cp in _NUMERIC_C1:
        return chr(_NUMERIC_C1[cp])
    # HTML5 drops noncharacters entirely (matches html.unescape / mistune).
    if 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE:
        return ""
    return chr(cp)


def _decode_entities(s: str) -> str:
    """Decode semicolon-terminated named/numeric references; others literal."""
    def repl(m: re.Match) -> str:
        body = m.group(1)
        if body.startswith("#"):
            return _numeric_charref(body)
        value = _HTML5_ENTITIES.get(body + ";")
        return value if value is not None else m.group(0)

    return _ENTITY_RE.sub(repl, s)


# --------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------

_SAFE_SCHEMES = frozenset(
    ("http", "https", "ftp", "ftps", "irc", "ircs", "mailto", "tel")
)

# Bytes that never get percent-encoded in destinations.
_URL_SAFE = frozenset(
    b"!#$%&()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    b"_abcdefghijklmnopqrstuvwxyz~"
)

def _check_url(url: str) -> str:
    """Allowlist check: disallowed schemes render as ``#harmful-link``."""
    scheme = None
    for idx, ch in enumerate(url):
        if ch == ":":
            scheme = url[:idx]
            break
        if ch in "/?#":
            break
    if scheme is not None and scheme.lower() not in _SAFE_SCHEMES:
        return "#harmful-link"
    return url


def _escape_url(raw: str) -> str:
    """mistune's destination rendering: entity-decode, percent-encode, HTML-escape."""
    decoded = _decode_entities(raw)
    out = []
    for b in decoded.encode("utf-8"):
        if b in _URL_SAFE:
            out.append(chr(b))
        else:
            out.append(f"%{b:02X}")
    return _escape_html("".join(out))


def _escape_title(raw: str) -> str:
    return _escape_html(_decode_entities(raw))


# --------------------------------------------------------------------------
# block tree
# --------------------------------------------------------------------------

B_DOC, B_QUOTE, B_LIST, B_ITEM, B_PARA, B_HEAD, B_HR, B_CODE_IND, B_CODE_F, B_HTML = range(10)


class _Blk:
    __slots__ = (
        "t", "children", "lines", "level", "info", "fence_ch", "fence_len",
        "fence_indent", "ordered", "start", "bullet", "delim", "content_indent",
        "blank_after", "tight", "html_end", "html_tag",
    )

    def __init__(self, t: int) -> None:
        self.t = t
        self.children: list[_Blk] = []
        self.lines: list[str] = []
        self.level = 0
        self.info = ""
        self.fence_ch = ""
        self.fence_len = 0
        self.fence_indent = 0
        self.ordered = False
        self.start = 1
        self.bullet = ""
        self.delim = ""
        self.content_indent = 0
        self.blank_after = False
        self.tight = True
        self.html_end = ""
        self.html_tag = ""


class _Cur:
    """Line cursor with CommonMark partial-tab handling.

    Consuming N columns of indentation may split a tab: the unconsumed part
    of the tab's expansion stays as literal spaces in the line.
    """

    __slots__ = ("line", "pos", "col")

    def __init__(self, line: str) -> None:
        self.line = line
        self.pos = 0
        self.col = 0

    def indent(self) -> int:
        """Columns of whitespace at the cursor, without consuming (tabs expanded)."""
        n = 0
        i = self.pos
        col = self.col
        line = self.line
        while i < len(line):
            ch = line[i]
            if ch == " ":
                col += 1
                n += 1
            elif ch == "\t":
                w = 4 - (col % 4)
                col += w
                n += w
            else:
                break
            i += 1
        return n

    def skip_columns(self, n: int) -> None:
        """Consume exactly n columns of indentation (splits tabs)."""
        while n > 0 and self.pos < len(self.line):
            ch = self.line[self.pos]
            if ch == " ":
                self.pos += 1
                self.col += 1
                n -= 1
            elif ch == "\t":
                w = 4 - (self.col % 4)
                if w <= n:
                    self.pos += 1
                    self.col += w
                    n -= w
                else:
                    # partial tab: the rest of the expansion stays as spaces
                    self.line = self.line[: self.pos] + " " * (w - n) + self.line[self.pos + 1 :]
                    self.col += n
                    n = 0
            else:
                break

    def skip_spaces(self, n: int) -> int:
        """Consume up to n columns of indentation; returns columns consumed."""
        got = 0
        while got < n and self.pos < len(self.line):
            ch = self.line[self.pos]
            if ch == " ":
                self.pos += 1
                self.col += 1
                got += 1
            elif ch == "\t":
                w = 4 - (self.col % 4)
                if w <= n - got:
                    self.pos += 1
                    self.col += w
                    got += w
                else:
                    self.line = self.line[: self.pos] + " " * (w - (n - got)) + self.line[self.pos + 1 :]
                    self.col += n - got
                    got = n
            else:
                break
        return got

    def rest(self) -> str:
        return self.line[self.pos :]

    def blank_rest(self) -> bool:
        return self.rest().strip(" \t") == ""


# --- block-start recognizers (operate on the cursor rest, indent already
# --- verified <= 3 by the caller unless noted) ---

_ATX_RE = re.compile(r"^(#{1,6})(?:[ \t]+|$)(.*)$")
_SETEXT_RE = re.compile(r"^(=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
_BULLET_RE = re.compile(r"^([-*+])([ \t]|$)(.*)$", re.S)
_ORDERED_RE = re.compile(r"^([0-9]{1,9})([.)])([ \t]|$)(.*)$", re.S)


def _is_thematic(s: str) -> bool:
    s = s.strip(" \t")
    if len(s) < 3:
        return False
    stripped = s.replace(" ", "").replace("\t", "")
    if len(stripped) < 3:
        return False
    return all(c == stripped[0] for c in stripped) and stripped[0] in "*-_"


def _parse_atx(rest: str) -> tuple[int, str] | None:
    m = _ATX_RE.match(rest)
    if not m:
        return None
    level = len(m.group(1))
    content = m.group(2).strip(" \t")
    if content and all(c == "#" for c in content):
        content = ""  # the whole remainder is the closing sequence
    else:
        m2 = re.match(r"^(.*?)[ \t]+#+[ \t]*$", content, re.S)
        if m2:
            content = m2.group(1).rstrip(" \t")
    return level, content


def _match_list_marker(rest: str):
    """Return (ordered, start, bullet, delim, marker_width, after) or None."""
    m = _BULLET_RE.match(rest)
    if m:
        return (False, 0, m.group(1), "", 1, m.group(3))
    m = _ORDERED_RE.match(rest)
    if m:
        return (True, int(m.group(1)), "", m.group(2), len(m.group(1)) + 1, m.group(4))
    return None



# --------------------------------------------------------------------------
# HTML blocks (CommonMark types 1-7); mistune renders them escaped in <p>
# --------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    "address article aside base basefont blockquote body caption center col "
    "colgroup dd details dialog dir div dl dt fieldset figcaption figure "
    "footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe "
    "legend li link main menu menuitem nav noframes ol optgroup option p "
    "param search section summary table tbody td tfoot th thead title tr "
    "track ul".split()
)

_TYPE1_RE = __import__("re").compile(r"(?i)^<(script|pre|style|textarea)(?=[ \t>]|$)")
_TAGNAME_RE = __import__("re").compile(r"(?i)^</?([A-Za-z][A-Za-z0-9-]*)(?=[ \t/>]|$)")
_ATTR = r"[A-Za-z_:][A-Za-z0-9_.:-]*"
_ATTR_VAL = r"(?:[^ \t\"'=<>`]+|'[^']*'|\"[^\"]*\")"
_TYPE7_RE = __import__("re").compile(
    r"^(?:<[A-Za-z][A-Za-z0-9-]*(?:[ \t]+"
    + _ATTR
    + r"(?:[ \t]*=[ \t]*"
    + _ATTR_VAL
    + r")?)*[ \t]*/?>|</[A-Za-z][A-Za-z0-9-]*[ \t]*>)[ \t]*$"
)


def _html_block_start(rest: str, tip_is_para: bool):
    """Detect an HTML block start; returns (end_kind, end_tag) or None.

    end_kind: "tag" (type 1), a terminator string ("-->", "?>", ">", "]]>"),
    or "blank" (types 6/7: end at a blank line).
    """
    m = _TYPE1_RE.match(rest)
    if m:
        return "tag", m.group(1).lower()
    if rest.startswith("<!--"):
        return "-->", ""
    if rest.startswith("<?"):
        return "?>", ""
    if re.match(r"^<![A-Za-z]", rest):
        return ">", ""
    if rest.startswith("<![CDATA["):
        return "]]>", ""
    m = _TAGNAME_RE.match(rest)
    if m and m.group(1).lower() in _BLOCK_TAGS:
        return "blank", ""
    if not tip_is_para:
        m7 = _TYPE7_RE.match(rest)
        if m7 and m7.group(0).strip() != "":
            low = re.match(r"^</?([A-Za-z][A-Za-z0-9-]*)", rest)
            if low and low.group(1).lower() not in ("script", "pre", "style", "textarea"):
                return "blank", ""
    return None


# --------------------------------------------------------------------------
# block parser
# --------------------------------------------------------------------------

class _Parser:
    def __init__(self) -> None:
        self.root = _Blk(B_DOC)
        self.open: list[_Blk] = [self.root]
        self.refs: dict = {}

    # -- block tree helpers ------------------------------------------------

    def tip(self) -> _Blk:
        return self.open[-1]

    def add_block(self, b: _Blk) -> _Blk:
        parent = self.tip()
        parent.children.append(b)
        for blk in self.open:
            blk.blank_after = False
        return b

    def pop_to(self, index: int) -> None:
        """Close open blocks from index onward."""
        while len(self.open) > index:
            self._close_block(self.open.pop())

    def _close_block(self, b: _Blk) -> None:
        if b.t == B_PARA:
            b.lines = _extract_ref_defs(b.lines, self.refs)
            if not b.lines:
                # a paragraph that held only reference definitions vanishes
                for parent in reversed(self.open):
                    if parent.children and parent.children[-1] is b:
                        parent.children.pop()
                        parent.blank_after = True
                        break

    # -- main loop ----------------------------------------------------------

    def parse(self, text: str) -> _Blk:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for line in lines:
            self.process_line(line)
        self.pop_to(1)
        self._finalize(self.root)
        return self.root

    def _finalize(self, blk: _Blk) -> None:
        for child in blk.children:
            self._finalize(child)
        if blk.t == B_LIST:
            blk.tight = self._is_tight(blk)

    @staticmethod
    def _is_tight(lst: _Blk) -> bool:
        items = lst.children
        for idx, item in enumerate(items):
            if item.blank_after and idx < len(items) - 1:
                return False
            for j, child in enumerate(item.children):
                if child.blank_after and j < len(item.children) - 1:
                    return False
        return True

    # -- one line ------------------------------------------------------------

    def process_line(self, line: str) -> None:
        cur = _Cur(line)
        raw_blank = line.strip(" \t") == ""
        open_blocks = self.open
        absorbs: list[int] = []  # container types that voted to absorb the line
        close_from: int | None = None
        i = 1

        # phase 1: container walk
        while i < len(open_blocks):
            b = open_blocks[i]
            if b.t == B_QUOTE:
                if raw_blank:
                    close_from = i
                    break
                if self._match_quote(cur):
                    i += 1
                    continue
                if b.blank_after or self._lazy_kind(cur, in_item=False) == "close":
                    close_from = i
                    break
                absorbs.append((i, B_QUOTE))
                i += 1
                continue
            if b.t == B_ITEM:
                if raw_blank:
                    if not b.children:
                        close_from = i
                        break
                    i += 1
                    continue
                if cur.indent() >= b.content_indent:
                    cur.skip_columns(b.content_indent)
                    i += 1
                    continue
                if b.blank_after or self._lazy_kind(cur, in_item=True) == "close":
                    close_from = i
                    break
                absorbs.append((i, B_ITEM))
                i += 1
                continue
            if b.t == B_CODE_F:
                if any(t == B_QUOTE for _, t in absorbs):
                    close_from = min(i for i, t in absorbs if t == B_QUOTE)
                    break
                self._fence_line(b, cur)
                return
            if b.t == B_HTML:
                if any(t == B_QUOTE for _, t in absorbs):
                    close_from = min(i for i, t in absorbs if t == B_QUOTE)
                    break
                if self._html_line(b, cur):
                    return
                break
            if b.t == B_PARA:
                if cur.blank_rest():
                    self._close_block(self.open.pop())
                    b.blank_after = True
                    self._note_blank()
                    return
                if absorbs:
                    self._lazy_paragraph(b, cur, absorbs[-1][1])
                    return
                break  # paragraph reached: phase 2 may interrupt it
            if b.t == B_CODE_IND:
                if any(t == B_QUOTE for _, t in absorbs):
                    close_from = min(i for i, t in absorbs if t == B_QUOTE)
                    break
                if cur.blank_rest():
                    c2 = _Cur(cur.rest())
                    c2.skip_spaces(4)
                    b.lines.append(c2.rest())
                    return
                if cur.indent() >= 4:
                    cur.skip_columns(4)
                    b.lines.append(cur.rest())
                    return
                self.open.pop()
                break
            # B_LIST / B_DOC: transparent
            i += 1

        if close_from is not None:
            self.pop_to(close_from)
            absorbs = [(k, t) for k, t in absorbs if k < close_from]

        if raw_blank or cur.blank_rest():
            self._note_blank()
            return

        self._open_new_blocks(cur, absorbs[-1][1] if absorbs else None)

    def _note_blank(self) -> None:
        if len(self.open) <= 1:
            return
        open_blocks = self.open
        deepest = open_blocks[-1]
        deepest.blank_after = True
        if deepest.children:
            deepest.children[-1].blank_after = True
        # propagate up the container chain, but not past a block quote
        for b in reversed(open_blocks[1:]):
            b.blank_after = True
            if b.t == B_QUOTE:
                break

    # -- quote matching -------------------------------------------------------

    def _match_quote(self, cur: _Cur) -> bool:
        save = (cur.pos, cur.col, cur.line)
        cur.skip_spaces(3)
        if not cur.rest().startswith(">"):
            cur.pos, cur.col, cur.line = save
            return False
        cur.pos += 1
        cur.col += 1
        self._skip_one_space(cur)
        return True

    @staticmethod
    def _skip_one_space(cur: _Cur) -> None:
        """Consume one optional space-width after '>' or a list marker."""
        if cur.pos < len(cur.line) and cur.line[cur.pos] == " ":
            cur.pos += 1
            cur.col += 1
        elif cur.pos < len(cur.line) and cur.line[cur.pos] == "\t":
            w = 4 - (cur.col % 4)
            if w == 1:
                cur.pos += 1
                cur.col += 1
            else:
                cur.line = cur.line[: cur.pos] + " " * (w - 1) + cur.line[cur.pos + 1 :]
                cur.col += 1

    # -- lazy continuation ----------------------------------------------------

    @staticmethod
    def _lazy_kind(cur: _Cur, in_item: bool) -> str:
        """Does a failing container absorb this line or close for it?"""
        probe = _Cur(cur.rest())
        probe.skip_spaces(3)
        stripped = probe.rest()
        fm = _FENCE_RE.match(stripped)
        if fm and not (fm.group(1)[0] == "`" and "`" in fm.group(2)):
            return "close"
        if _is_thematic(stripped):
            return "close"
        if _match_list_marker(stripped):
            return "close"
        if in_item:
            if _ATX_RE.match(stripped):
                return "close"
            if stripped.startswith(">"):
                return "close"
        return "absorb"

    def _lazy_paragraph(self, para: _Blk, cur: _Cur, innermost: int) -> None:
        """The walk reached an open paragraph through absorbed containers."""
        lead = _Cur(cur.rest())
        lead.skip_spaces(3)
        srest = lead.rest()
        if innermost == B_ITEM:
            m = _SETEXT_RE.match(srest)
            if m:
                self._convert_setext(para, m.group(1))
                return
        if innermost == B_QUOTE:
            atx = _parse_atx(srest)
            if atx is not None:
                self.open.pop()  # close the paragraph
                h = self.add_block(_Blk(B_HEAD))
                h.level, h.lines = atx[0], [atx[1]]
                return
        if cur.indent() < 4 and self._open_html(cur, tip_is_para=True):
            return
        para.lines.append(cur.rest().lstrip(" \t"))

    # -- html blocks ---------------------------------------------------------

    def _html_line(self, blk: _Blk, cur: _Cur) -> bool:
        """Feed a line to an open HTML block; returns False if it just closed."""
        rest = cur.rest()
        kind = blk.html_end
        if kind == "blank":
            if rest.strip(" \t") == "":
                self.open.pop()
                return False
        elif kind == "tag":
            blk.lines.append(rest)
            if re.search(r"(?i)</" + re.escape(blk.html_tag), rest):
                self.open.pop()
            return True
        else:
            blk.lines.append(rest)
            if kind in rest:
                self.open.pop()
            return True
        blk.lines.append(rest)
        return True

    def _open_html(self, cur: _Cur, tip_is_para: bool) -> bool:
        start = _html_block_start(cur.rest(), tip_is_para)
        if start is None:
            return False
        kind, tag = start
        self._close_list_for_leaf()
        self._interrupt_paragraph()
        b = self.add_block(_Blk(B_HTML))
        b.html_end, b.html_tag = kind, tag
        b.lines.append(cur.rest())
        # same-line end conditions for terminator kinds
        if kind not in ("blank", "tag") and kind in b.lines[0]:
            pass  # block closes immediately (do not push)
        elif kind == "tag" and re.search(r"(?i)</" + re.escape(tag), b.lines[0]):
            pass
        else:
            self.open.append(b)
        return True

    # -- fenced code ---------------------------------------------------------

    def _fence_line(self, blk: _Blk, cur: _Cur) -> None:
        rest = cur.rest()
        probe = _Cur(rest)
        probe.skip_spaces(3)
        m = re.match(r"^(`{3,}|~{3,})[ \t]*$", probe.rest())
        if m and m.group(1)[0] == blk.fence_ch and len(m.group(1)) >= blk.fence_len:
            self.open.pop()
            return
        c2 = _Cur(rest)
        c2.skip_spaces(blk.fence_indent)
        blk.lines.append(c2.rest())

    # -- phase 2: new block starts -------------------------------------------

    def _open_new_blocks(self, cur: _Cur, lazy_innermost) -> None:
        while True:
            if cur.blank_rest():
                return
            tip = self.tip()
            if lazy_innermost is not None:
                lead = _Cur(cur.rest())
                lead.skip_spaces(3)
                srest = lead.rest()
                indented4 = cur.indent() >= 4
                if lazy_innermost == B_QUOTE and not indented4:
                    atx = _parse_atx(srest)
                    if atx is not None:
                        self._interrupt_paragraph()
                        h = self.add_block(_Blk(B_HEAD))
                        h.level, h.lines = atx[0], [atx[1]]
                        return
                if lazy_innermost == B_ITEM and tip.t == B_PARA and not indented4:
                    m = _SETEXT_RE.match(srest)
                    if m:
                        self._convert_setext(tip, m.group(1))
                        return
                if not indented4 and self._open_html(cur, tip.t == B_PARA):
                    return
                self._add_paragraph_line(cur)
                return

            indent = cur.indent()
            if indent <= 3:
                save = (cur.pos, cur.col, cur.line)
                cur.skip_columns(indent)
                rest = cur.rest()
                if rest.startswith(">"):
                    self._close_list_for_leaf()
                    self._interrupt_paragraph()
                    q = self.add_block(_Blk(B_QUOTE))
                    self.open.append(q)
                    cur.pos += 1
                    cur.col += 1
                    self._skip_one_space(cur)
                    continue
                atx = _parse_atx(rest)
                if atx is not None:
                    self._close_list_for_leaf()
                    self._interrupt_paragraph()
                    h = self.add_block(_Blk(B_HEAD))
                    h.level, h.lines = atx[0], [atx[1]]
                    return
                fm = _FENCE_RE.match(rest)
                if fm and not (fm.group(1)[0] == "`" and "`" in fm.group(2)):
                    self._close_list_for_leaf()
                    self._interrupt_paragraph()
                    b = self.add_block(_Blk(B_CODE_F))
                    b.fence_ch = fm.group(1)[0]
                    b.fence_len = len(fm.group(1))
                    b.fence_indent = indent
                    b.info = fm.group(2).strip(" \t")
                    self.open.append(b)
                    return
                if self._open_html(cur, tip.t == B_PARA):
                    return
                if tip.t == B_PARA:
                    m = _SETEXT_RE.match(rest)
                    if m:
                        self._convert_setext(tip, m.group(1))
                        return
                if _is_thematic(rest):
                    self._close_list_for_leaf()
                    self._interrupt_paragraph()
                    self.add_block(_Blk(B_HR))
                    return
                lm = _match_list_marker(rest)
                if lm is not None:
                    ordered, start, _bullet, _delim, _width, after = lm
                    if tip.t == B_PARA and (
                        after.strip(" \t") == "" or (ordered and start != 1)
                    ):
                        cur.pos, cur.col, cur.line = save
                        self._add_paragraph_line(cur)
                        return
                    if self._start_list_item(cur, indent, lm):
                        continue
                    return
                cur.pos, cur.col, cur.line = save
            if indent >= 4:
                if tip.t == B_PARA:
                    self._add_paragraph_line(cur)
                    return
                self._close_list_for_leaf()
                b = self.add_block(_Blk(B_CODE_IND))
                self.open.append(b)
                cur.skip_columns(4)
                b.lines.append(cur.rest())
                return
            self._add_paragraph_line(cur)
            return

    def _close_list_for_leaf(self) -> None:
        """A non-item block cannot be a direct child of a list; close it."""
        if self.tip().t == B_LIST:
            self.open.pop()

    def _interrupt_paragraph(self) -> None:
        if self.open[-1].t == B_PARA:
            self._close_block(self.open.pop())

    def _convert_setext(self, para: _Blk, marker: str) -> None:
        self._close_block(para)  # extract any reference definitions first
        if not para.lines:
            # the paragraph held only definitions: the underline is plain text
            if self.open and self.open[-1] is para:
                self.open.pop()
            self._add_paragraph_line(_Cur(marker))
            return
        if self.open and self.open[-1] is para:
            self.open.pop()
        parent = self.tip()
        assert parent.children and parent.children[-1] is para
        parent.children.pop()
        h = self.add_block(_Blk(B_HEAD))
        h.level = 1 if marker.startswith("=") else 2
        h.lines = [ln.rstrip(" \t") for ln in para.lines]
        h.blank_after = para.blank_after

    def _add_paragraph_line(self, cur: _Cur) -> None:
        tip = self.tip()
        if tip.t == B_LIST:
            self.open.pop()
            tip = self.tip()
        if tip.t != B_PARA:
            tip = self.add_block(_Blk(B_PARA))
            self.open.append(tip)
        tip.lines.append(cur.rest().lstrip(" \t"))

    def _start_list_item(self, cur: _Cur, marker_indent: int, lm) -> bool:
        """Open (or continue) a list + item; cursor at the marker. Returns True."""
        ordered, start, bullet, delim, width, _after = lm
        self._interrupt_paragraph()
        tip = self.tip()
        lst = None
        if tip.t == B_LIST:
            same = tip.ordered == ordered and (
                (ordered and tip.delim == delim) or (not ordered and tip.bullet == bullet)
            )
            if same:
                lst = tip
            else:
                self.open.pop()  # different list type: close it first
        if lst is None:
            lst = self.add_block(_Blk(B_LIST))
            lst.ordered, lst.start, lst.bullet, lst.delim = ordered, start, bullet, delim
            self.open.append(lst)
        item = self.add_block(_Blk(B_ITEM))
        self.open.append(item)
        cur.pos += width
        cur.col += width
        if cur.blank_rest():
            item.content_indent = marker_indent + width + 1
            return True
        peek = _Cur(cur.line)
        peek.pos, peek.col = cur.pos, cur.col
        spaces = peek.skip_spaces(5)
        if 1 <= spaces <= 4:
            cur.skip_columns(spaces)
            item.content_indent = marker_indent + width + spaces
        else:
            cur.skip_columns(1)
            item.content_indent = marker_indent + width + 1
        return True


# --------------------------------------------------------------------------
# link reference definitions
# --------------------------------------------------------------------------

_LABEL_MAX = 999


def _normalize_label(s: str) -> str:
    return re.sub(r"[ \t\n]+", " ", s.strip(" \t\n")).casefold()


def _unescape(s: str) -> str:
    """Resolve backslash escapes (ASCII punctuation only)."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in _ASCII_PUNCT:
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _scan_link_label(text: str, pos: int) -> tuple[str, int] | None:
    """Scan [label] starting at pos; returns (raw_label, end_after_] )."""
    if pos >= len(text) or text[pos] != "[":
        return None
    i = pos + 1
    start = i
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ASCII_PUNCT:
            i += 2
            continue
        if ch == "[":
            return None  # labels may not contain unescaped brackets
        if ch == "]":
            label = text[start:i]
            if len(label) > _LABEL_MAX:
                return None
            return label, i + 1
        i += 1
    return None


def _scan_link_dest(text: str, pos: int) -> tuple[str, int] | None:
    if pos >= len(text):
        return None
    if text[pos] == "<":
        i = pos + 1
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ASCII_PUNCT:
                i += 2
                continue
            if ch == "\n":
                return None
            if ch == ">":
                return _unescape(text[pos + 1 : i]), i + 1
            if ch == "<":
                return None
            i += 1
        return None
    # bare destination: no spaces/controls, balanced parens
    i = pos
    depth = 0
    start = pos
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ASCII_PUNCT:
            i += 2
            continue
        if ch in " \t\n" or ord(ch) < 0x20:
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        i += 1
    if depth != 0 or i == start:
        return None
    return _unescape(text[start:i]), i


def _scan_link_title(text: str, pos: int) -> tuple[str, int] | None:
    if pos >= len(text):
        return None
    op = text[pos]
    if op == '"':
        cl = '"'
    elif op == "'":
        cl = "'"
    elif op == "(":
        cl = ")"
    else:
        return None
    i = pos + 1
    start = i
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ASCII_PUNCT:
            i += 2
            continue
        if op == "(" and ch == "(":
            return None
        if ch == cl:
            return _unescape(text[start:i]), i + 1
        i += 1
    return None


def _skip_spaces_nl(text: str, pos: int, max_nl: int = 1) -> int:
    """Skip spaces/tabs and up to max_nl newlines."""
    nl = 0
    while pos < len(text):
        ch = text[pos]
        if ch in " \t":
            pos += 1
        elif ch == "\n" and nl < max_nl:
            nl += 1
            pos += 1
        else:
            break
    return pos


def _parse_ref_def(text: str, pos: int):
    """Try to parse one link reference definition at pos.

    Returns (label, dest, title, end_pos) or None. end_pos is after the
    definition's last line (including its line ending handling).
    """
    lab = _scan_link_label(text, pos)
    if lab is None:
        return None
    label_raw, p = lab
    if p >= len(text) or text[p] != ":":
        return None
    label = _normalize_label(label_raw)
    if not label:
        return None
    p += 1
    p = _skip_spaces_nl(text, p)
    d = _scan_link_dest(text, p)
    if d is None:
        return None
    dest, p = d
    # title: needs separation (space/tab or up to one newline)
    sep_end = _skip_spaces_nl(text, p)
    had_sep = sep_end > p
    title = ""
    if had_sep:
        t = _scan_link_title(text, sep_end)
        if t is not None:
            tval, tend = t
            # after the title only spaces to EOL
            rest = text[tend:]
            eol = rest.find("\n")
            tail = rest if eol < 0 else rest[:eol]
            if tail.strip(" \t") == "":
                title = tval
                p = tend
                # consume to EOL
                p = _skip_spaces_nl(text, p, max_nl=0)
                return label, dest, title, p
        # no valid title: the rest of the line must be blank
        rest = text[p:]
        eol = rest.find("\n")
        tail = rest if eol < 0 else rest[:eol]
        if tail.strip(" \t") != "":
            return None
        return label, dest, "", p
    # no separation at all: only valid if at EOL/EOF
    rest = text[p:]
    eol = rest.find("\n")
    tail = rest if eol < 0 else rest[:eol]
    if tail.strip(" \t") != "":
        return None
    return label, dest, "", p


def _extract_ref_defs(lines: list[str], refs: dict) -> list[str]:
    """Strip leading link reference definitions from paragraph lines."""
    text = "\n".join(lines)
    pos = 0
    while True:
        # up to 3 leading spaces allowed before the label
        m = re.match(r"[ ]{0,3}(?=\[)", text[pos:])
        if not m:
            break
        r = _parse_ref_def(text, pos + m.end())
        if r is None:
            break
        label, dest, title, end = r
        if label not in refs:
            refs[label] = (dest, title)
        pos = end
        # skip the line ending
        if pos < len(text) and text[pos] == "\n":
            pos += 1
        if pos >= len(text):
            break
    rest = text[pos:]
    if rest == "":
        return []
    return rest.split("\n")


# --------------------------------------------------------------------------
# inline parsing
# --------------------------------------------------------------------------

N_TEXT, N_EM, N_STRONG, N_CODE, N_LINK, N_IMG, N_BR, N_SOFT = range(8)


class _Node:
    __slots__ = ("t", "text", "children", "dest", "title")

    def __init__(self, t: int, text: str = "") -> None:
        self.t = t
        self.text = text
        self.children: list[_Node] = []
        self.dest = ""
        self.title = ""


_HTML_TOKEN_RE = re.compile(
    r"^<!--(?!>|->)(?:(?!--)[\s\S])*?(?<!-)-->"
    r"|^<\?[\s\S]*?\?>"
    r"|^<![A-Za-z][^>]*>"
    r"|^<!\[CDATA\[[\s\S]*?\]\]>"
    r"|^</?[A-Za-z][A-Za-z0-9-]*"
    r"(?:\s+[A-Za-z_:][A-Za-z0-9_.:-]*"
    r"(?:\s*=\s*(?:[^ \t\"'=<>`]+|'[^']*'|\"[^\"]*\"))?)*"
    r"\s*/?>"
)

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]"
    r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)
_URI_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]{1,31}):[^ <>\x00-\x20]*$")


class _Inline:
    def __init__(self, refs: dict) -> None:
        self.refs = refs
        self.nodes: list[_Node] = []
        self.delims: list[dict] = []
        self.brackets: list[dict] = []

    # -- main scan -----------------------------------------------------------

    def parse(self, text: str) -> list[_Node]:
        self.nodes = []
        self.delims = []
        self.brackets = []
        text = text.rstrip(" \t")
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "\\":
                if i + 1 < n and text[i + 1] == "\n":
                    self._trim_trailing_spaces(hard=True)
                    self.nodes.append(_Node(N_BR))
                    i += 2
                    continue
                if i + 1 < n and text[i + 1] in _ASCII_PUNCT:
                    self.nodes.append(_Node(N_TEXT, text[i + 1]))
                    i += 2
                    continue
                self.nodes.append(_Node(N_TEXT, "\\"))
                i += 1
                continue
            if ch == "`":
                j = i
                while j < n and text[j] == "`":
                    j += 1
                run = j - i
                # find matching run
                k = j
                found = -1
                while k < n:
                    if text[k] == "`":
                        m = k
                        while m < n and text[m] == "`":
                            m += 1
                        if m - k == run:
                            found = m
                            break
                        k = m
                    else:
                        k += 1
                if found < 0:
                    self.nodes.append(_Node(N_TEXT, "`" * run))
                    i = j
                    continue
                content = text[j : found - run]
                node = _Node(N_CODE, self._code_text(content))
                self.nodes.append(node)
                i = found
                continue
            if ch == "<":
                m = re.match(r"^<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^ <>\x00-\x20]*)>", text[i:])
                if m:
                    uri = m.group(1)
                    node = _Node(N_LINK)
                    node.dest = uri
                    node.title = ""
                    node.children.append(_Node(N_TEXT, uri))
                    self.nodes.append(node)
                    i += m.end()
                    continue
                m2 = re.match(r"^<([^ <>@\x00-\x20]+@[^ <>@\x00-\x20]+)>", text[i:])
                if m2 and _EMAIL_RE.match(m2.group(1)):
                    addr = m2.group(1)
                    node = _Node(N_LINK)
                    node.dest = "mailto:" + addr
                    node.title = ""
                    node.children.append(_Node(N_TEXT, addr))
                    node.text = "email"
                    self.nodes.append(node)
                    i += m2.end()
                    continue
                m3 = _HTML_TOKEN_RE.match(text[i:])
                if m3:
                    self.nodes.append(_Node(N_TEXT, m3.group(0)))
                    i += m3.end()
                    continue
                self.nodes.append(_Node(N_TEXT, "<"))
                i += 1
                continue
            if ch in "*_":
                j = i
                while j < n and text[j] == ch:
                    j += 1
                run = text[i:j]
                before = text[i - 1] if i > 0 else "\n"
                after = text[j] if j < n else "\n"
                can_open, can_close = self._flanking(ch, before, after)
                node = _Node(N_TEXT, run)
                self.nodes.append(node)
                self.delims.append(
                    {
                        "node": node,
                        "ch": ch,
                        "count": run.__len__(),
                        "orig": run.__len__(),
                        "can_open": can_open,
                        "can_close": can_close,
                        "active": True,
                    }
                )
                i = j
                continue
            if ch == "[" or (ch == "!" and i + 1 < n and text[i + 1] == "["):
                is_img = ch == "!"
                node = _Node(N_TEXT, "![" if is_img else "[")
                self.nodes.append(node)
                self.brackets.append(
                    {
                        "node": node,
                        "img": is_img,
                        "active": True,
                        "mark": len(self.delims),
                        "spos": i + (2 if is_img else 1),
                    }
                )
                i += 2 if is_img else 1
                continue
            if ch == "]":
                consumed = self._close_bracket(text, i)
                if consumed > i:
                    i = consumed
                    continue
                self.nodes.append(_Node(N_TEXT, "]"))
                i += 1
                continue
            if ch == "\n":
                hard = self._trim_trailing_spaces(hard=False)
                if hard:
                    self.nodes.append(_Node(N_BR))
                else:
                    self.nodes.append(_Node(N_SOFT, "\n"))
                i += 1
                continue
            # plain text run: up to the next special char
            j = i
            while j < n and text[j] not in "\\`<*_[!\n]":
                j += 1
            if j == i:  # '!' not followed by '['
                j = i + 1
            self.nodes.append(_Node(N_TEXT, text[i:j]))
            i = j
        self._merge_text()
        self._process_emphasis(self.delims)
        self._merge_text()
        return self.nodes

    def _merge_text(self) -> None:
        delim_nodes = {id(d["node"]) for d in self.delims}
        out: list[_Node] = []
        for node in self.nodes:
            if (
                out
                and node.t == N_TEXT
                and out[-1].t == N_TEXT
                and id(node) not in delim_nodes
                and id(out[-1]) not in delim_nodes
            ):
                out[-1].text += node.text
            else:
                out.append(node)
        self.nodes = out

    def _trim_trailing_spaces(self, hard: bool) -> bool:
        """Handle spaces before a line ending. Returns True for a hard break."""
        if not self.nodes:
            return False
        last = self.nodes[-1]
        if last.t != N_TEXT:
            return False
        stripped = last.text.rstrip(" ")
        nspaces = len(last.text) - len(stripped)
        if hard or nspaces:
            last.text = stripped
        if hard:
            return True
        return nspaces >= 2

    @staticmethod
    def _code_text(content: str) -> str:
        c = content.replace("\n", " ")
        if (
            len(c) >= 2
            and c[0] == " "
            and c[-1] == " "
            and c.strip(" ") != ""
        ):
            c = c[1:-1]
        return c

    @staticmethod
    def _flanking(ch: str, before: str, after: str) -> tuple[bool, bool]:
        before_ws = _is_ws(before)
        after_ws = _is_ws(after)
        before_p = _is_punct(before)
        after_p = _is_punct(after)
        left = (not after_ws) and (not after_p or (before_ws or before_p))
        right = (not before_ws) and (not before_p or (after_ws or after_p))
        if ch == "*":
            return left, right
        # underscore: intraword rules
        can_open = left and (not right or before_p)
        can_close = right and (not left or after_p)
        return can_open, can_close

    # -- links / images -------------------------------------------------------

    def _close_bracket(self, text: str, i: int) -> int:
        """A ']' at position i: try to form a link/image. Returns new pos or 0."""
        if not self.brackets:
            return 0
        bi = len(self.brackets) - 1
        br = self.brackets[bi]
        if not br["active"]:
            # an inactive opener still pairs with the ']', but forms nothing
            self.brackets.pop()
            return 0
        br = self.brackets[bi]
        opener = br["node"]
        # locate opener node index
        oi = self.nodes.index(opener)
        label_text = text[br["spos"] : i]  # raw source between [ and ]
        after = i + 1
        dest = title = None
        end = after
        n = len(text)
        # inline link: [text](dest title)
        if after < n and text[after] == "(":
            p = _skip_spaces_nl(text, after + 1)
            d = _scan_link_dest(text, p)
            if d is None and p < n and text[p] == ")":
                dest, title, end = "", "", p + 1
            elif d is not None:
                dest_v, p2 = d
                p3 = _skip_spaces_nl(text, p2)
                title_v = ""
                t = _scan_link_title(text, p3)
                if t is not None:
                    title_v, p3 = t
                    p3 = _skip_spaces_nl(text, p3)
                if p3 < n and text[p3] == ")":
                    dest, title, end = dest_v, title_v, p3 + 1
        if dest is None:
            # full/collapsed reference
            if after < n and text[after] == "[":
                lab = _scan_link_label(text, after)
                if lab is not None:
                    raw, lab_end = lab
                    key = _normalize_label(raw) if raw.strip() else _normalize_label(label_text)
                    hit = self.refs.get(key)
                    if hit is not None:
                        dest, title = hit
                        end = lab_end
            if dest is None:
                # shortcut reference
                key = _normalize_label(label_text)
                hit = self.refs.get(key)
                if hit is not None and not (after < n and text[after] == "["):
                    dest, title = hit
                    end = after
        if dest is None:
            # not a link: drop the opener bracket record
            self.brackets.pop(bi)
            return 0
        # build the node
        node = _Node(N_IMG if br["img"] else N_LINK)
        node.dest = dest
        node.title = title
        node.children = self.nodes[oi + 1 :]
        del self.nodes[oi:]
        self.nodes.append(node)
        # process emphasis within the link text in isolation (delimiters
        # inside a link can never match delimiters outside it)
        mark = br["mark"]
        sub = self.delims[mark:]
        if sub:
            self._process_emphasis(sub)
        del self.delims[mark:]
        self.brackets.pop(bi)
        # deactivate earlier link openers (no nested links)
        if not br["img"]:
            for b in self.brackets:
                if not b["img"]:
                    b["active"] = False
        return end

    def _plain(self, opener: _Node) -> str:
        try:
            oi = self.nodes.index(opener)
        except ValueError:
            return ""
        return "".join(self._node_text(x) for x in self.nodes[oi + 1 :])

    def _node_text(self, node: _Node) -> str:
        if node.t in (N_TEXT, N_CODE):
            return node.text
        if node.t in (N_SOFT, N_BR):
            return "\n"
        return "".join(self._node_text(c) for c in node.children)

    # -- emphasis --------------------------------------------------------------

    def _process_emphasis(self, delims: list) -> None:
        # Mirrors the reference emphasis algorithm: closers are visited in
        # document order; openers_bottom is keyed by (char, closer can_open,
        # closer length mod 3) so incompatible earlier failures are skipped.
        delims = [d for d in delims if d["count"] > 0]
        openers_bottom: dict = {}
        ci = 0
        while ci < len(delims):
            c = delims[ci]
            if not (c["can_close"] and c["count"] > 0):
                ci += 1
                continue
            ch = c["ch"]
            key = (ch, c["can_open"], c["orig"] % 3)
            bottom = openers_bottom.get(key, -1)
            found = -1
            oi = ci - 1
            while oi > bottom:
                o = delims[oi]
                if o["ch"] == ch and o["can_open"] and o["count"] > 0:
                    odd = (
                        (c["can_open"] or o["can_close"])
                        and (o["orig"] + c["orig"]) % 3 == 0
                        and (o["orig"] % 3 != 0 or c["orig"] % 3 != 0)
                    )
                    if not odd:
                        found = oi
                        break
                oi -= 1
            if found < 0:
                openers_bottom[key] = ci - 1
                ci += 1
                continue
            o = delims[found]
            use = 2 if (o["count"] >= 2 and c["count"] >= 2) else 1
            self._wrap_emphasis(o, c, use)
            o["count"] -= use
            c["count"] -= use
            # drop delimiters between opener and closer
            del delims[found + 1 : ci]
            if o["count"] == 0:
                if o["node"].text == "":
                    self._remove_node(o["node"])
                del delims[found]
                ci = found
            else:
                ci = found + 1
            if c["count"] == 0:
                if c["node"].text == "":
                    self._remove_node(c["node"])
                del delims[ci]
                # do not advance: the next closer is now at ci

    def _remove_node(self, node: _Node) -> None:
        if node in self.nodes:
            self.nodes.remove(node)
        else:
            self._remove_from(self.nodes, node)

    def _remove_from(self, nodes: list[_Node], node: _Node) -> bool:
        for x in nodes:
            if x.children:
                if node in x.children:
                    x.children.remove(node)
                    return True
                if self._remove_from(x.children, node):
                    return True
        return False

    def _wrap_emphasis(self, o: dict, c: dict, use: int) -> None:
        onode, cnode = o["node"], c["node"]
        onode.text = onode.text[:-use] if len(onode.text) >= use else ""
        cnode.text = cnode.text[use:]
        parent_list, oi, ci = self._find_span(onode, cnode)
        if parent_list is None:
            return
        node = _Node(N_STRONG if use == 2 else N_EM)
        node.children = parent_list[oi + 1 : ci]
        parent_list[oi + 1 : ci] = [node]

    def _find_span(self, a: _Node, b: _Node):
        """Find the sibling list containing both a and b (a before b)."""
        return self._find_span_in(self.nodes, a, b)

    def _find_span_in(self, nodes: list[_Node], a: _Node, b: _Node):
        if a in nodes and b in nodes:
            return nodes, nodes.index(a), nodes.index(b)
        for x in nodes:
            if x.children:
                r = self._find_span_in(x.children, a, b)
                if r[0] is not None:
                    return r
        return None, -1, -1


# --------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _render_inline(nodes: list[_Node]) -> str:
    out = []
    for node in nodes:
        t = node.t
        if t == N_TEXT:
            out.append(_escape_html(node.text))
        elif t == N_CODE:
            out.append(f"<code>{_escape_html(node.text)}</code>")
        elif t == N_EM:
            out.append(f"<em>{_render_inline(node.children)}</em>")
        elif t == N_STRONG:
            out.append(f"<strong>{_render_inline(node.children)}</strong>")
        elif t == N_LINK:
            href = _escape_url(_check_url_render(node.dest))
            title = f' title="{_escape_title(node.title)}"' if node.title else ""
            out.append(f'<a href="{href}"{title}>{_render_inline(node.children)}</a>')
        elif t == N_IMG:
            src = _escape_url(_check_url_render(node.dest))
            title = f' title="{_escape_title(node.title)}"' if node.title else ""
            alt = _escape_html(_alt_text(node.children))
            out.append(f'<img src="{src}" alt="{alt}"{title} />')
        elif t == N_BR:
            out.append("<br />\n")
        elif t == N_SOFT:
            out.append("\n")
    return "".join(out)


def _check_url_render(dest: str) -> str:
    """Safety check happens on the entity-decoded form, before URL escaping."""
    decoded = _decode_entities(dest)
    scheme = None
    for idx, ch in enumerate(decoded):
        if ch == ":":
            scheme = decoded[:idx]
            break
        if ch in "/?#":
            break
    if scheme is not None and scheme.lower() not in _SAFE_SCHEMES:
        return "#harmful-link"
    return dest


def _alt_text(nodes: list[_Node]) -> str:
    out = []
    for node in nodes:
        t = node.t
        if t in (N_TEXT, N_CODE):
            out.append(node.text)
        elif t in (N_SOFT, N_BR):
            out.append("\n")
        else:
            out.append(_alt_text(node.children))
    return "".join(out)


def _render_blocks(blocks: list[_Blk], refs: dict, out: list[str]) -> None:
    for b in blocks:
        t = b.t
        if t == B_PARA:
            text = "\n".join(b.lines)
            nodes = _Inline(refs).parse(text)
            out.append(f"<p>{_render_inline(nodes)}</p>\n")
        elif t == B_HEAD:
            text = "\n".join(b.lines)
            nodes = _Inline(refs).parse(text)
            out.append(f"<h{b.level}>{_render_inline(nodes)}</h{b.level}>\n")
        elif t == B_HR:
            out.append("<hr />\n")
        elif t == B_CODE_IND:
            lines = list(b.lines)
            while lines and lines[-1] == "":
                lines.pop()
            out.append(f"<pre><code>{_escape_html(chr(10).join(lines))}</code></pre>\n")
        elif t == B_CODE_F:
            content = "".join(ln + "\n" for ln in b.lines)
            cls = ""
            info = b.info.split()
            if info:
                lang = _escape_html(_decode_entities(_unescape(info[0])))
                cls = f' class="language-{lang}"'
            out.append(f"<pre><code{cls}>{_escape_html(content)}</code></pre>\n")
        elif t == B_HTML:
            lines = list(b.lines)
            while lines and lines[-1] == "":
                lines.pop()
            out.append(f"<p>{_escape_html(chr(10).join(lines))}</p>\n")
        elif t == B_QUOTE:
            out.append("<blockquote>\n")
            _render_blocks(b.children, refs, out)
            out.append("</blockquote>\n")
        elif t == B_LIST:
            tag = "ol" if b.ordered else "ul"
            attr = ""
            if b.ordered and b.start != 1:
                attr = f' start="{b.start}"'
            out.append(f"<{tag}{attr}>\n")
            for item in b.children:
                _render_item(item, refs, out, b.tight)
            out.append(f"</{tag}>\n")


def _render_item(item: _Blk, refs: dict, out: list[str], tight: bool) -> None:
    out.append("<li>")
    if tight:
        for b in item.children:
            if b.t == B_PARA:
                text = "\n".join(b.lines)
                nodes = _Inline(refs).parse(text)
                out.append(_render_inline(nodes))
            else:
                _render_blocks([b], refs, out)
    else:
        _render_blocks(item.children, refs, out)
    out.append("</li>\n")


# --------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def markdown(text: str) -> str:
    """Render Markdown to HTML, matching mistune.markdown() byte-for-byte.

    Call contract mirrors mistune: None/"" render as ""; other non-string
    input fails the same way mistune fails (the first string operation is
    ``text.replace`` in both implementations).
    """
    if text is None or text == "":
        return ""
    parser = _Parser()
    root = parser.parse(text)
    out: list[str] = []
    _render_blocks(root.children, parser.refs, out)
    return "".join(out)
