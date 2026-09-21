"""Clean-room Markdown -> HTML kernel reproducing mistune.markdown() (3.3.4).

Written fresh from the published CommonMark 0.31.2 spec plus black-box
observed behavior of the published `mistune` package (probe inputs ->
observed outputs; no mistune source was read or adapted). Ported from the
package's pure-Python engine (python/mistune_mojo/mistune_mojo/_reference.py),
which documents the observed mistune semantics this kernel mirrors:
no raw HTML and no entity decoding in text, entities decoded in link
destinations/titles, fenced code keeps its trailing newline while indented
code does not, URL scheme allowlist with `#harmful-link` fallback, and
mistune's container lazy-continuation rules.

Exported C ABI (single-shot: render a whole document per call):

    int32_t  mistunemojo_abi_version(void)
    void*    mistunemojo_render(const uint8_t* data, int64_t length)
    int32_t  mistunemojo_result_status(void* handle)  -> 0 ok / 2 unsupported
    int64_t  mistunemojo_result_size(void* handle)    -> html byte length
    uint8_t* mistunemojo_result_data(void* handle)    -> html bytes
    void     mistunemojo_result_destroy(void* handle)

Status 2 ("unsupported construct") is returned when the document needs a
feature outside the kernel's scope: a named HTML entity outside the built-in
table in a link destination/title, or a non-ASCII link-reference label
(Unicode casefolding). The Python wrapper then re-renders with its
byte-identical pure-Python engine, so end users see no difference.
"""

from std.collections import Dict
from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# Block types.
comptime B_DOC: Int = 0
comptime B_QUOTE: Int = 1
comptime B_LIST: Int = 2
comptime B_ITEM: Int = 3
comptime B_PARA: Int = 4
comptime B_HEAD: Int = 5
comptime B_HR: Int = 6
comptime B_CODE_IND: Int = 7
comptime B_CODE_F: Int = 8
comptime B_HTML: Int = 9

# Inline node types.
comptime N_TEXT: Int = 0
comptime N_EM: Int = 1
comptime N_STRONG: Int = 2
comptime N_CODE: Int = 3
comptime N_LINK: Int = 4
comptime N_IMG: Int = 5
comptime N_BR: Int = 6
comptime N_SOFT: Int = 7


# --------------------------------------------------------------------------
# small string helpers (byte-oriented; all delimiter scans are ASCII-safe)
# --------------------------------------------------------------------------


def b_at(s: String, i: Int) -> Int:
    """Byte at position i, or -1 out of range."""
    if i < 0 or i >= s.byte_length():
        return -1
    return Int(s.as_bytes()[i])


struct ByteWriter(Copyable, Movable):
    """Accumulates raw bytes; converts to String once (always valid UTF-8)."""

    var data: List[UInt8]

    def __init__(out self):
        self.data = List[UInt8]()

    def push(mut self, b: Int):
        self.data.append(UInt8(b))

    def write(mut self, s: String):
        var n = s.byte_length()
        if n == 0:
            return
        var old = len(self.data)
        self.data.resize(old + n, 0)
        unsafe_memcpy(dest=self.data.unsafe_ptr() + old, src=s.unsafe_ptr(), count=n)

    def write_span(mut self, s: String, start: Int, end: Int):
        var n = end - start
        if n <= 0:
            return
        var old = len(self.data)
        self.data.resize(old + n, 0)
        unsafe_memcpy(dest=self.data.unsafe_ptr() + old, src=s.unsafe_ptr() + start, count=n)

    def finish(self) -> String:
        var copy = self.data.copy()
        return String(unsafe_from_utf8=copy^)


def pushb(mut s: String, b: Int):
    """Append one ASCII byte to a string (only safe for ASCII: < 0x80)."""
    var one = List[UInt8]()
    one.append(UInt8(b))
    s += String(unsafe_from_utf8=one^)



def str_split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    var parts = s.split(sep)
    for i in range(len(parts)):
        out.append(String(parts[i]))
    return out^


def slice(s: String, start: Int) -> String:
    var n = s.byte_length()
    if start >= n:
        return String("")
    return String(s[byte=start:n])


def slice(s: String, start: Int, end: Int) -> String:
    var n = s.byte_length()
    var e = end
    if e > n:
        e = n
    if start >= e:
        return String("")
    return String(s[byte=start:e])


def starts_with(s: String, prefix: String) -> Bool:
    return s.byte_length() >= prefix.byte_length() and String(
        s[byte=0 : prefix.byte_length()]
    ) == prefix


def lstrip_of(s: String, chars: String) -> String:
    var i = 0
    var n = s.byte_length()
    while i < n and _in_chars(b_at(s, i), chars):
        i += 1
    return slice(s, i)


def rstrip_of(s: String, chars: String) -> String:
    var n = s.byte_length()
    while n > 0 and _in_chars(b_at(s, n - 1), chars):
        n -= 1
    return slice(s, 0, n)


def strip_of(s: String, chars: String) -> String:
    return lstrip_of(rstrip_of(s, chars), chars)


def _in_chars(b: Int, chars: String) -> Bool:
    if b < 0:
        return False
    for i in range(chars.byte_length()):
        if Int(chars.as_bytes()[i]) == b:
            return True
    return False


def is_blank(s: String) -> Bool:
    for i in range(s.byte_length()):
        var b = Int(s.as_bytes()[i])
        if b != 32 and b != 9:
            return False
    return True


# --------------------------------------------------------------------------
# character classes (same tables as the Python engine: identical
# classification keeps the two backends byte-identical)
# --------------------------------------------------------------------------


def is_ws_cp(cp: Int) -> Bool:
    # CommonMark "Unicode whitespace": Zs + tab/LF/FF/CR + U+2028/U+2029.
    if cp == 32 or cp == 9 or cp == 10 or cp == 11 or cp == 12 or cp == 13:
        return True
    if cp == 0xA0 or cp == 0x1680 or cp == 0x2028 or cp == 0x2029:
        return True
    if cp == 0x202F or cp == 0x205F or cp == 0x3000:
        return True
    return 0x2000 <= cp <= 0x200A


comptime PUNCT_ASCII: String = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def is_punct_cp(cp: Int, ranges: List[Int]) -> Bool:
    # Unicode general categories P* and S* (CommonMark "punctuation").
    if cp < 128:
        return _in_chars(cp, PUNCT_ASCII)
    var lo = 0
    var hi = (len(ranges) // 2) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        var a = ranges[mid * 2]
        var b = ranges[mid * 2 + 1]
        if cp < a:
            hi = mid - 1
        elif cp > b:
            lo = mid + 1
        else:
            return True
    return False


comptime PUNCT_RANGES_STR: String = "33,47,58,64,91,96,123,126,161,169,171,172,174,177,180,180,182,184,187,187,191,191,215,215,247,247,706,709,722,735,741,747,749,749,751,767,885,885,894,894,900,901,903,903,1014,1014,1154,1154,1370,1375,1417,1418,1421,1423,1470,1470,1472,1472,1475,1475,1478,1478,1523,1524,1542,1551,1563,1563,1565,1567,1642,1645,1748,1748,1758,1758,1769,1769,1789,1790,1792,1805,2038,2041,2046,2047,2096,2110,2142,2142,2184,2184,2404,2405,2416,2416,2546,2547,2554,2555,2557,2557,2678,2678,2800,2801,2928,2928,3059,3066,3191,3191,3199,3199,3204,3204,3407,3407,3449,3449,3572,3572,3647,3647,3663,3663,3674,3675,3841,3863,3866,3871,3892,3892,3894,3894,3896,3896,3898,3901,3973,3973,4030,4037,4039,4044,4046,4058,4170,4175,4254,4255,4347,4347,4960,4968,5008,5017,5120,5120,5741,5742,5787,5788,5867,5869,5941,5942,6100,6102,6104,6107,6144,6154,6464,6464,6468,6469,6622,6655,6686,6687,6816,6822,6824,6829,7002,7018,7028,7038,7164,7167,7227,7231,7294,7295,7360,7367,7379,7379,8125,8125,8127,8129,8141,8143,8157,8159,8173,8175,8189,8190,8208,8231,8240,8286,8314,8318,8330,8334,8352,8384,8448,8449,8451,8454,8456,8457,8468,8468,8470,8472,8478,8483,8485,8485,8487,8487,8489,8489,8494,8494,8506,8507,8512,8516,8522,8525,8527,8527,8586,8587,8592,9254,9280,9290,9372,9449,9472,10101,10132,11123,11126,11157,11159,11263,11493,11498,11513,11516,11518,11519,11632,11632,11776,11822,11824,11869,11904,11929,11931,12019,12032,12245,12272,12283,12289,12292,12296,12320,12336,12336,12342,12343,12349,12351,12443,12444,12448,12448,12539,12539,12688,12689,12694,12703,12736,12771,12800,12830,12842,12871,12880,12880,12896,12927,12938,12976,12992,13311,19904,19967,42128,42182,42238,42239,42509,42511,42611,42611,42622,42622,42738,42743,42752,42774,42784,42785,42889,42890,43048,43051,43062,43065,43124,43127,43214,43215,43256,43258,43260,43260,43310,43311,43359,43359,43457,43469,43486,43487,43612,43615,43639,43641,43742,43743,43760,43761,43867,43867,43882,43883,44011,44011,64297,64297,64434,64450,64830,64847,64975,64975,65020,65023,65040,65049,65072,65106,65108,65126,65128,65131,65281,65295,65306,65312,65339,65344,65371,65381,65504,65510,65512,65518,65532,65533,65792,65794,65847,65855,65913,65929,65932,65934,65936,65948,65952,65952,66000,66044,66463,66463,66512,66512,66927,66927,67671,67671,67703,67704,67871,67871,67903,67903,68176,68184,68223,68223,68296,68296,68336,68342,68409,68415,68505,68508,69293,69293,69461,69465,69510,69513,69703,69709,69819,69820,69822,69825,69952,69955,70004,70005,70085,70088,70093,70093,70107,70107,70109,70111,70200,70205,70313,70313,70731,70735,70746,70747,70749,70749,70854,70854,71105,71127,71233,71235,71264,71276,71353,71353,71484,71487,71739,71739,72004,72006,72162,72162,72255,72262,72346,72348,72350,72354,72448,72457,72769,72773,72816,72817,73463,73464,73539,73551,73685,73713,73727,73727,74864,74868,77809,77810,92782,92783,92917,92917,92983,92991,92996,92997,93847,93850,94178,94178,113820,113820,113823,113823,118608,118723,118784,119029,119040,119078,119081,119140,119146,119148,119171,119172,119180,119209,119214,119274,119296,119361,119365,119365,119552,119638,120513,120513,120539,120539,120571,120571,120597,120597,120629,120629,120655,120655,120687,120687,120713,120713,120745,120745,120771,120771,120832,121343,121399,121402,121453,121460,121462,121475,121477,121483,123215,123215,123647,123647,125278,125279,126124,126124,126128,126128,126254,126254,126704,126705,126976,127019,127024,127123,127136,127150,127153,127167,127169,127183,127185,127221,127245,127405,127462,127490,127504,127547,127552,127560,127568,127569,127584,127589,127744,128727,128732,128748,128752,128764,128768,128886,128891,128985,128992,129003,129008,129008,129024,129035,129040,129095,129104,129113,129120,129159,129168,129197,129200,129201,129280,129619,129632,129645,129648,129660,129664,129672,129680,129725,129727,129733,129742,129755,129760,129768,129776,129784,129792,129938,129940,129994"


def load_punct_ranges() -> List[Int]:
    var out = List[Int]()
    var parts = PUNCT_RANGES_STR.split(",")
    for i in range(len(parts)):
        try:
            out.append(Int(String(parts[i])))
        except:
            pass
    return out^



def prev_cp(s: String, i: Int) -> Int:
    """Codepoint ending at byte position i (UTF-8 decode backwards)."""
    if i <= 0:
        return 10  # beginning of line counts as whitespace
    var j = i - 1
    var b = b_at(s, j)
    while j > 0 and b >= 0x80 and b < 0xC0:
        j -= 1
        b = b_at(s, j)
    if b < 0x80:
        return b
    var n = i - j
    if n == 2:
        return ((b & 0x1F) << 6) | (b_at(s, j + 1) & 0x3F)
    if n == 3:
        return ((b & 0x0F) << 12) | ((b_at(s, j + 1) & 0x3F) << 6) | (b_at(s, j + 2) & 0x3F)
    if n == 4:
        return (
            ((b & 0x07) << 18)
            | ((b_at(s, j + 1) & 0x3F) << 12)
            | ((b_at(s, j + 2) & 0x3F) << 6)
            | (b_at(s, j + 3) & 0x3F)
        )
    return b


def next_cp(s: String, i: Int) -> Int:
    """Codepoint starting at byte position i (UTF-8 decode forwards)."""
    if i >= s.byte_length():
        return 10  # end of line counts as whitespace
    var b = b_at(s, i)
    if b < 0x80:
        return b
    if b >= 0xF0:
        return (
            ((b & 0x07) << 18)
            | ((b_at(s, i + 1) & 0x3F) << 12)
            | ((b_at(s, i + 2) & 0x3F) << 6)
            | (b_at(s, i + 3) & 0x3F)
        )
    if b >= 0xE0:
        return ((b & 0x0F) << 12) | ((b_at(s, i + 1) & 0x3F) << 6) | (b_at(s, i + 2) & 0x3F)
    return ((b & 0x1F) << 6) | (b_at(s, i + 1) & 0x3F)


def cp_to_utf8(cp: Int) -> String:
    var out = ByteWriter()
    if cp < 0x80:
        out.push(cp)
    elif cp < 0x800:
        out.push(0xC0 | (cp >> 6))
        out.push(0x80 | (cp & 0x3F))
    elif cp < 0x10000:
        out.push(0xE0 | (cp >> 12))
        out.push(0x80 | ((cp >> 6) & 0x3F))
        out.push(0x80 | (cp & 0x3F))
    else:
        out.push(0xF0 | (cp >> 18))
        out.push(0x80 | ((cp >> 12) & 0x3F))
        out.push(0x80 | ((cp >> 6) & 0x3F))
        out.push(0x80 | (cp & 0x3F))
    return out.finish()


# --------------------------------------------------------------------------
# HTML escaping (mistune escapes & < > " everywhere, including code spans)
# --------------------------------------------------------------------------


def escape_html_into(mut out: ByteWriter, s: String):
    var n = s.byte_length()
    var i = 0
    var start = 0
    while i < n:
        var b = Int(s.as_bytes()[i])
        if b == 38 or b == 60 or b == 62 or b == 34:
            out.write_span(s, start, i)
            if b == 38:
                out.write("&amp;")
            elif b == 60:
                out.write("&lt;")
            elif b == 62:
                out.write("&gt;")
            else:
                out.write("&quot;")
            i += 1
            start = i
        else:
            i += 1
    out.write_span(s, start, n)


def escape_html(s: String) -> String:
    var out = ByteWriter()
    for i in range(s.byte_length()):
        var b = Int(s.as_bytes()[i])
        if b == 38:
            out.write("&amp;")
        elif b == 60:
            out.write("&lt;")
        elif b == 62:
            out.write("&gt;")
        elif b == 34:
            out.write("&quot;")
        else:
            out.push(b)
    return out.finish()


# --------------------------------------------------------------------------
# entity decoding (link destinations and titles only)
# --------------------------------------------------------------------------


def numeric_charref(body: String) -> String:
    """body is like '#169' or '#xA9'; HTML5 numeric reference rules."""
    var cp = 0
    var ok = True
    if starts_with(body, "#x") or starts_with(body, "#X"):
        var i = 2
        while i < body.byte_length():
            var b = Int(body.as_bytes()[i])
            var d = -1
            if 48 <= b <= 57:
                d = b - 48
            elif 65 <= b <= 70:
                d = b - 55
            elif 97 <= b <= 102:
                d = b - 87
            else:
                ok = False
                break
            cp = cp * 16 + d
            i += 1
    else:
        var i = 1
        while i < body.byte_length():
            var b = Int(body.as_bytes()[i])
            if not (48 <= b <= 57):
                ok = False
                break
            cp = cp * 10 + (b - 48)
            i += 1
    if not ok:
        return String("")
    if cp == 0 or cp > 0x10FFFF or (0xD800 <= cp <= 0xDFFF):
        return cp_to_utf8(0xFFFD)
    # HTML5 C1-control remapping
    var c1 = c1_replacement(cp)
    if c1 >= 0:
        return cp_to_utf8(c1)
    # noncharacters are dropped entirely (matches html.unescape / mistune)
    if 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE:
        return String("")
    return cp_to_utf8(cp)


def c1_replacement(cp: Int) -> Int:
    if cp == 0x80:
        return 0x20AC
    if cp == 0x82:
        return 0x201A
    if cp == 0x83:
        return 0x0192
    if cp == 0x84:
        return 0x201E
    if cp == 0x85:
        return 0x2026
    if cp == 0x86:
        return 0x2020
    if cp == 0x87:
        return 0x2021
    if cp == 0x88:
        return 0x02C6
    if cp == 0x89:
        return 0x2030
    if cp == 0x8A:
        return 0x0160
    if cp == 0x8B:
        return 0x2039
    if cp == 0x8C:
        return 0x0152
    if cp == 0x8E:
        return 0x017D
    if cp == 0x91:
        return 0x2018
    if cp == 0x92:
        return 0x2019
    if cp == 0x93:
        return 0x201C
    if cp == 0x94:
        return 0x201D
    if cp == 0x95:
        return 0x2022
    if cp == 0x96:
        return 0x2013
    if cp == 0x97:
        return 0x2014
    if cp == 0x98:
        return 0x02DC
    if cp == 0x99:
        return 0x2122
    if cp == 0x9A:
        return 0x0161
    if cp == 0x9B:
        return 0x203A
    if cp == 0x9C:
        return 0x0153
    if cp == 0x9E:
        return 0x017E
    if cp == 0x9F:
        return 0x0178
    return -1


def named_entity(name: String) raises -> String:
    """Small built-in HTML5 named-entity table.

    A miss raises Error("unsupported") so the render falls back to the
    pure-Python engine, which carries the full table.
    """
    if name == "amp":
        return "&"
    if name == "lt":
        return "<"
    if name == "gt":
        return ">"
    if name == "quot":
        return '"'
    if name == "apos":
        return "'"
    if name == "nbsp":
        return cp_to_utf8(0xA0)
    if name == "copy":
        return cp_to_utf8(0xA9)
    if name == "reg":
        return cp_to_utf8(0xAE)
    if name == "deg":
        return cp_to_utf8(0xB0)
    if name == "para":
        return cp_to_utf8(0xB6)
    if name == "sect":
        return cp_to_utf8(0xA7)
    if name == "middot":
        return cp_to_utf8(0xB7)
    if name == "plusmn":
        return cp_to_utf8(0xB1)
    if name == "times":
        return cp_to_utf8(0xD7)
    if name == "divide":
        return cp_to_utf8(0xF7)
    if name == "AElig":
        return cp_to_utf8(0xC6)
    if name == "aelig":
        return cp_to_utf8(0xE6)
    if name == "Dcaron":
        return cp_to_utf8(0x10E)
    if name == "dcaron":
        return cp_to_utf8(0x10F)
    if name == "Ouml":
        return cp_to_utf8(0xD6)
    if name == "ouml":
        return cp_to_utf8(0xF6)
    if name == "Auml":
        return cp_to_utf8(0xC4)
    if name == "auml":
        return cp_to_utf8(0xE4)
    if name == "Uuml":
        return cp_to_utf8(0xDC)
    if name == "uuml":
        return cp_to_utf8(0xFC)
    if name == "szlig":
        return cp_to_utf8(0xDF)
    if name == "euro":
        return cp_to_utf8(0x20AC)
    if name == "pound":
        return cp_to_utf8(0xA3)
    if name == "yen":
        return cp_to_utf8(0xA5)
    if name == "cent":
        return cp_to_utf8(0xA2)
    if name == "trade":
        return cp_to_utf8(0x2122)
    if name == "hellip":
        return cp_to_utf8(0x2026)
    if name == "mdash":
        return cp_to_utf8(0x2014)
    if name == "ndash":
        return cp_to_utf8(0x2013)
    if name == "lsquo":
        return cp_to_utf8(0x2018)
    if name == "rsquo":
        return cp_to_utf8(0x2019)
    if name == "ldquo":
        return cp_to_utf8(0x201C)
    if name == "rdquo":
        return cp_to_utf8(0x201D)
    if name == "laquo":
        return cp_to_utf8(0xAB)
    if name == "raquo":
        return cp_to_utf8(0xBB)
    if name == "bull":
        return cp_to_utf8(0x2022)
    if name == "dagger":
        return cp_to_utf8(0x2020)
    if name == "Dagger":
        return cp_to_utf8(0x2021)
    if name == "permil":
        return cp_to_utf8(0x2030)
    if name == "HilbertSpace":
        return cp_to_utf8(0x210B)
    if name == "DifferentialD":
        return cp_to_utf8(0x2146)
    if name == "ClockwiseContourIntegral":
        return cp_to_utf8(0x2232)
    if name == "CounterClockwiseContourIntegral":
        return cp_to_utf8(0x2233)
    if name == "ngE":
        return cp_to_utf8(0x2267) + cp_to_utf8(0x338)
    if name == "nGt":
        return cp_to_utf8(0x226B) + cp_to_utf8(0x338)
    if name == "nLt":
        return cp_to_utf8(0x226A) + cp_to_utf8(0x338)
    if name == "there4":
        return cp_to_utf8(0x2234)
    if name == "varepsilon":
        return cp_to_utf8(0x3F5)
    raise Error("unsupported")


def decode_entities(s: String) raises -> String:
    """Decode semicolon-terminated named/numeric references; others literal."""
    var out = ByteWriter()
    var i = 0
    var n = s.byte_length()
    while i < n:
        var b = Int(s.as_bytes()[i])
        if b != 38:  # &
            out.push(b)
            i += 1
            continue
        # scan &#...; or &name;
        var j = i + 1
        if j < n and Int(s.as_bytes()[j]) == 35:  # '#'
            j += 1
            var is_x = False
            if j < n and (Int(s.as_bytes()[j]) == 120 or Int(s.as_bytes()[j]) == 88):
                is_x = True
                j += 1
            var dstart = j
            while j < n and j - dstart < 12:
                var c = Int(s.as_bytes()[j])
                var ok = (48 <= c <= 57) or (is_x and ((65 <= c <= 70) or (97 <= c <= 102)))
                if not ok:
                    break
                j += 1
            if j > dstart and j < n and Int(s.as_bytes()[j]) == 59:
                out.write(numeric_charref(slice(s, i + 1, j)))
                i = j + 1
            else:
                out.push(b)
                i += 1
            continue
        # named: letter then alnum
        if j < n:
            var c0 = Int(s.as_bytes()[j])
            if (65 <= c0 <= 90) or (97 <= c0 <= 122):
                j += 1
                while j < n and j - i < 34:
                    var c = Int(s.as_bytes()[j])
                    if not ((65 <= c <= 90) or (97 <= c <= 122) or (48 <= c <= 57)):
                        break
                    j += 1
                if j < n and Int(s.as_bytes()[j]) == 59:
                    out.write(named_entity(slice(s, i + 1, j)))
                    i = j + 1
                    continue
        out.push(b)
        i += 1
    return out.finish()


# --------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------


def is_safe_scheme(scheme: String) -> Bool:
    var low = scheme.lower()
    return (
        low == "http"
        or low == "https"
        or low == "ftp"
        or low == "ftps"
        or low == "irc"
        or low == "ircs"
        or low == "mailto"
        or low == "tel"
    )


def check_url(url: String) -> Bool:
    """True if the URL passes the scheme allowlist."""
    var i = 0
    var n = url.byte_length()
    while i < n:
        var b = Int(url.as_bytes()[i])
        if b == 58:  # ':'
            return is_safe_scheme(slice(url, 0, i))
        if b == 47 or b == 63 or b == 35:  # / ? #
            return True
        i += 1
    return True


def url_byte_safe(b: Int) -> Bool:
    if 48 <= b <= 57 or (65 <= b <= 90) or (97 <= b <= 122):
        return True
    return _in_chars(b, "!#$%&()*+,-./:;=?@_~")


comptime HEX: String = "0123456789ABCDEF"


def escape_url(raw: String) raises -> String:
    """mistune's destination rendering: entity-decode, %-encode, HTML-escape."""
    var decoded = decode_entities(raw)
    var out = ByteWriter()
    for i in range(decoded.byte_length()):
        var b = Int(decoded.as_bytes()[i])
        if url_byte_safe(b):
            out.push(b)
        else:
            out.write("%")
            out.push(Int(HEX.as_bytes()[b >> 4]))
            out.push(Int(HEX.as_bytes()[b & 15]))
    return escape_html(out.finish())


def escape_title(raw: String) raises -> String:
    return escape_html(decode_entities(raw))


def url_is_harmful(raw: String) raises -> Bool:
    """Scheme check happens on the entity-decoded form."""
    return not check_url(decode_entities(raw))


# --------------------------------------------------------------------------
# block tree
# --------------------------------------------------------------------------


struct Blk(Copyable, Movable):
    var t: Int
    var children: List[Int]
    var lines: List[String]
    var level: Int
    var info: String
    var fence_ch: Int
    var fence_len: Int
    var fence_indent: Int
    var ordered: Bool
    var start: Int
    var bullet: Int
    var delim: Int
    var content_indent: Int
    var blank_after: Bool
    var tight: Bool
    var html_end: Int  # 0 none, 1 tag, 2 "-->", 3 "?>", 4 ">", 5 "]]>", 6 blank
    var html_tag: String

    def __init__(out self, t: Int):
        self.t = t
        self.children = List[Int]()
        self.lines = List[String]()
        self.level = 0
        self.info = String("")
        self.fence_ch = 0
        self.fence_len = 0
        self.fence_indent = 0
        self.ordered = False
        self.start = 1
        self.bullet = 0
        self.delim = 0
        self.content_indent = 0
        self.blank_after = False
        self.tight = True
        self.html_end = 0
        self.html_tag = String("")


struct Cur(Copyable, Movable):
    """Line cursor with CommonMark partial-tab handling."""

    var line: String
    var pos: Int
    var col: Int

    def __init__(out self, line: String):
        self.line = line
        self.pos = 0
        self.col = 0

    def indent(self) -> Int:
        """Columns of whitespace at the cursor, without consuming."""
        var n = 0
        var i = self.pos
        var col = self.col
        while i < self.line.byte_length():
            var b = Int(self.line.as_bytes()[i])
            if b == 32:
                col += 1
                n += 1
            elif b == 9:
                var w = 4 - (col % 4)
                col += w
                n += w
            else:
                break
            i += 1
        return n

    def skip_columns(mut self, n0: Int):
        """Consume exactly n columns of indentation (splits tabs)."""
        var n = n0
        while n > 0 and self.pos < self.line.byte_length():
            var b = Int(self.line.as_bytes()[self.pos])
            if b == 32:
                self.pos += 1
                self.col += 1
                n -= 1
            elif b == 9:
                var w = 4 - (self.col % 4)
                if w <= n:
                    self.pos += 1
                    self.col += w
                    n -= w
                else:
                    self.line = (
                        slice(self.line, 0, self.pos)
                        + String(" ") * (w - n)
                        + slice(self.line, self.pos + 1)
                    )
                    self.col += n
                    n = 0
            else:
                break

    def skip_spaces(mut self, n0: Int) -> Int:
        """Consume up to n columns of indentation; returns columns consumed."""
        var n = n0
        var got = 0
        while got < n and self.pos < self.line.byte_length():
            var b = Int(self.line.as_bytes()[self.pos])
            if b == 32:
                self.pos += 1
                self.col += 1
                got += 1
            elif b == 9:
                var w = 4 - (self.col % 4)
                if w <= n - got:
                    self.pos += 1
                    self.col += w
                    got += w
                else:
                    self.line = (
                        slice(self.line, 0, self.pos)
                        + String(" ") * (w - (n - got))
                        + slice(self.line, self.pos + 1)
                    )
                    self.col += n - got
                    got = n
            else:
                break
        return got

    def rest(self) -> String:
        return slice(self.line, self.pos)

    def blank_rest(self) -> Bool:
        return is_blank(self.rest())


# --- block-start recognizers (indent already verified <= 3) ---


def parse_atx(rest: String) -> Int:
    """Returns (level << 8) | content_start... caller re-slices; -1 if no match.

    Encoded: level * 1000000 + content_start; content is stripped of
    closing hashes by atx_content().
    """
    return atx_level(rest)


def atx_level(rest: String) -> Int:
    """ATX heading level (1-6) if the line starts an ATX heading, else -1."""
    var i = 0
    var n = rest.byte_length()
    while i < n and Int(rest.as_bytes()[i]) == 35 and i < 6:
        i += 1
    if i == 0:
        return -1
    if i < n:
        var b = Int(rest.as_bytes()[i])
        if b != 32 and b != 9:
            return -1
    return i


def atx_content(rest: String, level: Int) -> String:
    var content = strip_of(slice(rest, level), " \t")
    # closing sequence: only hashes, or space + hash-run at the end
    var n = content.byte_length()
    var all_hash = n > 0
    for i in range(n):
        if Int(content.as_bytes()[i]) != 35:
            all_hash = False
            break
    if all_hash:
        return String("")
    # find trailing hash run preceded by a space
    var j = n
    while j > 0 and (Int(content.as_bytes()[j - 1]) == 32 or Int(content.as_bytes()[j - 1]) == 9):
        j -= 1
    var k = j
    while k > 0 and Int(content.as_bytes()[k - 1]) == 35:
        k -= 1
    if k < j and k > 0 and (Int(content.as_bytes()[k - 1]) == 32 or Int(content.as_bytes()[k - 1]) == 9):
        content = rstrip_of(slice(content, 0, k - 1), " \t")
    return content


def setext_level(rest: String) -> Int:
    """1 for an all-= line, 2 for all--, else -1."""
    var s = rstrip_of(rest, " \t")
    var n = s.byte_length()
    if n == 0:
        return -1
    var b = Int(s.as_bytes()[0])
    if b != 61 and b != 45:
        return -1
    for i in range(1, n):
        if Int(s.as_bytes()[i]) != b:
            return -1
    return 1 if b == 61 else 2


def is_thematic(rest: String) -> Bool:
    var s = strip_of(rest, " \t")
    var n = s.byte_length()
    if n < 3:
        return False
    var ch = -1
    var count = 0
    for i in range(n):
        var b = Int(s.as_bytes()[i])
        if b == 32 or b == 9:
            continue
        if ch < 0:
            if b != 42 and b != 45 and b != 95:
                return False
            ch = b
        if b != ch:
            return False
        count += 1
    return count >= 3


def fence_start(rest: String) -> Int:
    """Fence marker: returns char code (` or ~) * 100 + run length, else -1."""
    var n = rest.byte_length()
    if n < 3:
        return -1
    var ch = Int(rest.as_bytes()[0])
    if ch != 96 and ch != 126:
        return -1
    var i = 1
    while i < n and Int(rest.as_bytes()[i]) == ch:
        i += 1
    if i < 3:
        return -1
    if ch == 96:
        # backtick fences: info may not contain a backtick
        var info = slice(rest, i)
        if info.find("`") >= 0:
            return -1
    return ch * 100 + i


def fence_info(rest: String, run_len: Int) -> String:
    return strip_of(slice(rest, run_len), " \t")


def match_list_marker(rest: String) -> Int:
    """Returns an encoded marker: ordered?1:0, start, width... via out params.

    Encoding: ordered * 1_000_000_000 + start * 1_000 + width * 10 + ok(1).
    Bullet: start = bullet char code; width = 1.
    Returns 0 (falsy) when no marker matches.
    """
    var n = rest.byte_length()
    if n == 0:
        return 0
    var b0 = Int(rest.as_bytes()[0])
    if b0 == 45 or b0 == 43 or b0 == 42:  # - + *
        if n == 1:
            return 1 + (1 << 1) + (b0 << 5)
        var b1 = Int(rest.as_bytes()[1])
        if b1 == 32 or b1 == 9:
            return 1 + (1 << 1) + (b0 << 5)
        return 0
    if 48 <= b0 <= 57:
        var i = 0
        var num = 0
        while i < n and i < 9 and 48 <= Int(rest.as_bytes()[i]) <= 57:
            num = num * 10 + (Int(rest.as_bytes()[i]) - 48)
            i += 1
        if i == 0 or i > 9:
            return 0
        if i < n:
            var d = Int(rest.as_bytes()[i])
            if d == 46 or d == 41:  # . or )
                if i + 1 == n:
                    return 1 + ((i + 1) << 1) + (num << 5) + (1 << 35)
                var b1 = Int(rest.as_bytes()[i + 1])
                if b1 == 32 or b1 == 9:
                    return 1 + ((i + 1) << 1) + (num << 5) + (1 << 35)
    return 0


def marker_after(rest: String, width: Int) -> String:
    return slice(rest, width)


# --------------------------------------------------------------------------
# HTML blocks (CommonMark types 1-7); mistune renders them escaped in <p>
# --------------------------------------------------------------------------

comptime BLOCK_TAGS_STR: String = "address article aside base basefont blockquote body caption center col colgroup dd details dialog dir div dl dt fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu menuitem nav noframes ol optgroup option p param search section summary table tbody td tfoot th thead title tr track ul"



def is_block_tag(tag: String) -> Bool:
    var low = tag.lower()
    var tags = BLOCK_TAGS_STR.split(" ")
    for i in range(len(tags)):
        if String(tags[i]) == low:
            return True
    return False


def scan_tagname(rest: String, start: Int) -> Int:
    """Length of the tag name at rest[start:], 0 if none."""
    var i = start
    var n = rest.byte_length()
    if i >= n:
        return 0
    var b0 = Int(rest.as_bytes()[i])
    if not ((65 <= b0 <= 90) or (97 <= b0 <= 122)):
        return 0
    i += 1
    while i < n:
        var b = Int(rest.as_bytes()[i])
        if (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b == 45:
            i += 1
        else:
            break
    return i - start


def html_block_start(rest: String, tip_is_para: Bool) -> Int:
    """Detect an HTML block start.

    Returns html_end kind: 1 tag(+tag code in html_tag via global not needed;
    caller re-derives), 2 "-->", 3 "?>", 4 ">", 5 "]]>", 6 "blank", else 0.
    """
    var n = rest.byte_length()
    if n < 2 or Int(rest.as_bytes()[0]) != 60:  # <
        return 0
    # type 1: <script|<pre|<style|<textarea
    for tag in List[String](["script", "pre", "style", "textarea"]):
        if starts_with_ci(rest, "<" + tag):
            var after = b_at(rest, 1 + tag.byte_length())
            if after < 0 or after == 32 or after == 9 or after == 62:
                return 1
    # type 2: comment
    if starts_with(rest, "<!--"):
        return 2
    # type 3: processing instruction
    if starts_with(rest, "<?"):
        return 3
    # type 4: declaration / type 5: CDATA
    if b_at(rest, 1) == 33:  # !
        if starts_with(rest, "<![CDATA["):
            return 5
        var c2 = b_at(rest, 2)
        if (65 <= c2 <= 90) or (97 <= c2 <= 122):
            return 4
    # type 6: block tag name
    var i = 1
    if b_at(rest, i) == 47:  # /
        i += 1
    var tn = scan_tagname(rest, i)
    if tn > 0:
        var after = b_at(rest, i + tn)
        if after < 0 or after == 32 or after == 9 or after == 62 or (after == 47 and b_at(rest, i + tn + 1) == 62):
            var tag = slice(rest, i, i + tn)
            if is_block_tag(tag):
                return 6
    # type 7: complete open/closing tag alone on the line
    if not tip_is_para and type7_line(rest):
        return 6
    return 0


def starts_with_ci(s: String, prefix: String) -> Bool:
    if s.byte_length() < prefix.byte_length():
        return False
    return slice(s, 0, prefix.byte_length()).lower() == prefix.lower()


def type7_line(rest: String) -> Bool:
    """The whole line (minus trailing ws) is a single open/closing tag."""
    var n = rest.byte_length()
    var i = 1
    var closing = False
    if b_at(rest, i) == 47:
        closing = True
        i += 1
    var tn = scan_tagname(rest, i)
    if tn == 0:
        return False
    var tag = slice(rest, i, i + tn).lower()
    if tag == "script" or tag == "pre" or tag == "style" or tag == "textarea":
        return False
    i += tn
    if closing:
        # optional spaces then '>'
        while i < n and (Int(rest.as_bytes()[i]) == 32 or Int(rest.as_bytes()[i]) == 9):
            i += 1
        if i < n and Int(rest.as_bytes()[i]) == 62:
            i += 1
            return is_blank(slice(rest, i))
        return False
    # attributes
    while True:
        var j = i
        while j < n and (Int(rest.as_bytes()[j]) == 32 or Int(rest.as_bytes()[j]) == 9):
            j += 1
        if j < n and Int(rest.as_bytes()[j]) == 62:
            return is_blank(slice(rest, j + 1))
        if j < n and Int(rest.as_bytes()[j]) == 47 and b_at(rest, j + 1) == 62:
            return is_blank(slice(rest, j + 2))
        if j == i:
            return False  # no whitespace before attribute
        i = j
        # attribute name
        var a = i
        if a >= n:
            return False
        var c0 = Int(rest.as_bytes()[a])
        if not ((65 <= c0 <= 90) or (97 <= c0 <= 122) or c0 == 95 or c0 == 58):
            return False
        a += 1
        while a < n:
            var c = Int(rest.as_bytes()[a])
            if (65 <= c <= 90) or (97 <= c <= 122) or (48 <= c <= 57) or c == 95 or c == 46 or c == 58 or c == 45:
                a += 1
            else:
                break
        i = a
        # optional value
        var k = i
        while k < n and (Int(rest.as_bytes()[k]) == 32 or Int(rest.as_bytes()[k]) == 9):
            k += 1
        if k < n and Int(rest.as_bytes()[k]) == 61:
            k += 1
            while k < n and (Int(rest.as_bytes()[k]) == 32 or Int(rest.as_bytes()[k]) == 9):
                k += 1
            # value: unquoted / single / double
            if k >= n:
                return False
            var q = Int(rest.as_bytes()[k])
            if q == 39 or q == 34:
                k += 1
                while k < n and Int(rest.as_bytes()[k]) != q:
                    k += 1
                if k >= n:
                    return False
                k += 1
            else:
                var u = k
                while u < n:
                    var c = Int(rest.as_bytes()[u])
                    if c <= 32 or c == 34 or c == 39 or c == 61 or c == 60 or c == 62 or c == 96:
                        break
                    u += 1
                if u == k:
                    return False
                k = u
            i = k
    return False


def html_end_hit(rest: String, kind: Int, tag: String) -> Bool:
    if kind == 1:
        return rest.lower().find("</" + tag.lower()) >= 0
    if kind == 2:
        return rest.find("-->") >= 0
    if kind == 3:
        return rest.find("?>") >= 0
    if kind == 4:
        return rest.find(">") >= 0
    if kind == 5:
        return rest.find("]]>") >= 0
    return False


# --------------------------------------------------------------------------
# link reference definitions
# --------------------------------------------------------------------------


struct RefDef(Copyable, Movable):
    var dest: String
    var title: String

    def __init__(out self, dest: String, title: String):
        self.dest = dest
        self.title = title


def normalize_label(s: String) raises -> String:
    """Collapse internal whitespace runs and ASCII-lowercase.

    Non-ASCII bytes raise Error("unsupported"): Unicode casefolding is
    outside the kernel's scope (the fallback engine handles it).
    """
    var out = ByteWriter()
    var in_ws = False
    for i in range(s.byte_length()):
        var b = Int(s.as_bytes()[i])
        if b >= 0x80:
            raise Error("unsupported")
        if b == 32 or b == 9 or b == 10:
            in_ws = True
        else:
            if in_ws and len(out.data) > 0:
                out.push(32)
            in_ws = False
            if 65 <= b <= 90:
                out.push(b + 32)
            else:
                out.push(b)
    return out.finish()


def unescape(s: String) -> String:
    """Resolve backslash escapes (ASCII punctuation only)."""
    var out = ByteWriter()
    var i = 0
    var n = s.byte_length()
    while i < n:
        var b = Int(s.as_bytes()[i])
        if b == 92 and i + 1 < n and _in_chars(Int(s.as_bytes()[i + 1]), PUNCT_ASCII):
            out.push(Int(s.as_bytes()[i + 1]))
            i += 2
        else:
            out.push(b)
            i += 1
    return out.finish()


def scan_link_label(text: String, pos: Int) -> Int:
    """End position (after ']') of a link label starting at pos, else -1."""
    var n = text.byte_length()
    if pos >= n or Int(text.as_bytes()[pos]) != 91:  # [
        return -1
    var i = pos + 1
    while i < n:
        var b = Int(text.as_bytes()[i])
        if b == 92 and i + 1 < n and _in_chars(Int(text.as_bytes()[i + 1]), PUNCT_ASCII):
            i += 2
            continue
        if b == 91:
            return -1  # labels may not contain unescaped brackets
        if b == 93:
            if i - (pos + 1) > 999:
                return -1
            return i + 1
        i += 1
    return -1


def scan_link_dest(text: String, pos: Int) -> Int:
    """End position after a link destination at pos, else -1."""
    var n = text.byte_length()
    if pos >= n:
        return -1
    if Int(text.as_bytes()[pos]) == 60:  # <
        var i = pos + 1
        while i < n:
            var b = Int(text.as_bytes()[i])
            if b == 92 and i + 1 < n and _in_chars(Int(text.as_bytes()[i + 1]), PUNCT_ASCII):
                i += 2
                continue
            if b == 10:
                return -1
            if b == 62:
                return i + 1
            if b == 60:
                return -1
            i += 1
        return -1
    var i = pos
    var depth = 0
    while i < n:
        var b = Int(text.as_bytes()[i])
        if b == 92 and i + 1 < n and _in_chars(Int(text.as_bytes()[i + 1]), PUNCT_ASCII):
            i += 2
            continue
        if b == 32 or b == 9 or b == 10 or b < 0x20:
            break
        if b == 40:
            depth += 1
        elif b == 41:
            if depth == 0:
                break
            depth -= 1
        i += 1
    if depth != 0 or i == pos:
        return -1
    return i


def scan_link_title(text: String, pos: Int) -> Int:
    """End position after a link title at pos, else -1."""
    var n = text.byte_length()
    if pos >= n:
        return -1
    var op = Int(text.as_bytes()[pos])
    var cl: Int
    if op == 34:
        cl = 34
    elif op == 39:
        cl = 39
    elif op == 40:
        cl = 41
    else:
        return -1
    var i = pos + 1
    while i < n:
        var b = Int(text.as_bytes()[i])
        if b == 92 and i + 1 < n and _in_chars(Int(text.as_bytes()[i + 1]), PUNCT_ASCII):
            i += 2
            continue
        if op == 40 and b == 40:
            return -1
        if b == cl:
            return i + 1
        i += 1
    return -1


def skip_spaces_nl(text: String, pos: Int, max_nl: Int) -> Int:
    var p = pos
    var nl = 0
    var n = text.byte_length()
    while p < n:
        var b = Int(text.as_bytes()[p])
        if b == 32 or b == 9:
            p += 1
        elif b == 10 and nl < max_nl:
            nl += 1
            p += 1
        else:
            break
    return p


def eol_tail_blank(text: String, pos: Int) -> Bool:
    """Only spaces/tabs from pos to the end of the line."""
    var n = text.byte_length()
    var p = pos
    while p < n:
        var b = Int(text.as_bytes()[p])
        if b == 10:
            return True
        if b != 32 and b != 9:
            return False
        p += 1
    return True


def parse_ref_def(text: String, pos: Int, mut refs: Dict[String, RefDef]) raises -> Int:
    """Try one reference definition at pos; returns end pos, or -1."""
    var n = text.byte_length()
    var lab_end = scan_link_label(text, pos)
    if lab_end < 0:
        return -1
    var raw_label = slice(text, pos + 1, lab_end - 1)
    if lab_end >= n or Int(text.as_bytes()[lab_end]) != 58:  # ':'
        return -1
    var label = normalize_label(raw_label)
    if label.byte_length() == 0:
        return -1
    var p = skip_spaces_nl(text, lab_end + 1, 1)
    var dest_end = scan_link_dest(text, p)
    if dest_end < 0:
        return -1
    var dest: String
    if Int(text.as_bytes()[p]) == 60:
        dest = unescape(slice(text, p + 1, dest_end - 1))
    else:
        dest = unescape(slice(text, p, dest_end))
    p = dest_end
    var sep_end = skip_spaces_nl(text, p, 1)
    var title = String("")
    var ok = False
    if sep_end > p:
        var t_end = scan_link_title(text, sep_end)
        if t_end >= 0 and eol_tail_blank(text, t_end):
            var op = Int(text.as_bytes()[sep_end])
            title = unescape(slice(text, sep_end + 1, t_end - 1))
            p = skip_spaces_nl(text, t_end, 0)
            ok = True
        if not ok and eol_tail_blank(text, p):
            ok = True  # titleless definition ending the line
        if not ok:
            return -1
    else:
        if not eol_tail_blank(text, p):
            return -1
    if label not in refs:
        refs[label] = RefDef(dest, title)
    return p


def extract_ref_defs(lines: List[String], mut refs: Dict[String, RefDef]) raises -> List[String]:
    """Strip leading link reference definitions from paragraph lines."""
    var text = "\n".join(lines)
    var pos = 0
    var n = text.byte_length()
    while pos < n:
        # up to 3 leading spaces before the label
        var sp = 0
        while sp < 3 and pos + sp < n and Int(text.as_bytes()[pos + sp]) == 32:
            sp += 1
        if pos + sp >= n or Int(text.as_bytes()[pos + sp]) != 91:
            break
        var end = parse_ref_def(text, pos + sp, refs)
        if end < 0:
            break
        pos = end
        if pos < n and Int(text.as_bytes()[pos]) == 10:
            pos += 1
    var rest = slice(text, pos)
    if rest.byte_length() == 0:
        return List[String]()
    return str_split(rest, "\n")


# --------------------------------------------------------------------------
# block parser
# --------------------------------------------------------------------------


struct Parser:
    var blocks: List[Blk]
    var open: List[Int]
    var refs: Dict[String, RefDef]

    def __init__(out self):
        self.blocks = List[Blk]()
        self.blocks.append(Blk(B_DOC))
        self.open = List[Int]()
        self.open.append(0)
        self.refs = Dict[String, RefDef]()

    def tip(self) -> Int:
        return self.open[len(self.open) - 1]

    def add_block(mut self, t: Int) -> Int:
        var parent = self.tip()
        self.blocks.append(Blk(t))
        var idx = len(self.blocks) - 1
        self.blocks[parent].children.append(idx)
        for k in range(len(self.open)):
            self.blocks[self.open[k]].blank_after = False
        return idx

    def close_block(mut self, b: Int) raises:
        if self.blocks[b].t == B_PARA:
            self.blocks[b].lines = extract_ref_defs(self.blocks[b].lines, self.refs)
            if len(self.blocks[b].lines) == 0:
                # a paragraph that held only reference definitions vanishes
                for k in range(len(self.open) - 1, -1, -1):
                    var parent = self.open[k]
                    var nch = len(self.blocks[parent].children)
                    if nch > 0 and self.blocks[parent].children[nch - 1] == b:
                        self.blocks[parent].children.pop()
                        self.blocks[parent].blank_after = True
                        break

    def pop_to(mut self, index: Int) raises:
        while len(self.open) > index:
            var b = self.open.pop()
            self.close_block(b)

    def interrupt_paragraph(mut self) raises:
        if self.blocks[self.tip()].t == B_PARA:
            var b = self.open.pop()
            self.close_block(b)

    def finalize(mut self, idx: Int):
        for k in range(len(self.blocks[idx].children)):
            self.finalize(self.blocks[idx].children[k])
        if self.blocks[idx].t == B_LIST:
            self.blocks[idx].tight = self.is_tight(idx)

    def is_tight(self, lst: Int) -> Bool:
        var items = self.blocks[lst].children.copy()
        for idx in range(len(items)):
            var item = items[idx]
            if self.blocks[item].blank_after and idx < len(items) - 1:
                return False
            var children = self.blocks[item].children.copy()
            for j in range(len(children)):
                if self.blocks[children[j]].blank_after and j < len(children) - 1:
                    return False
        return True

    # -- main loop ---------------------------------------------------------

    def parse(mut self, text0: String) raises -> Int:
        var text = text0.replace("\r\n", "\n").replace("\r", "\n")
        var lines = str_split(text, "\n")
        var nlines = len(lines)
        if nlines > 0 and lines[nlines - 1].byte_length() == 0:
            var kept_lines = List[String]()
            for k in range(nlines - 1):
                kept_lines.append(lines[k])
            lines = kept_lines.copy()
        for i in range(len(lines)):
            self.process_line(lines[i])
        self.pop_to(1)
        self.finalize(0)
        return 0

    def process_line(mut self, line: String) raises:
        var cur = Cur(line)
        var raw_blank = is_blank(line)
        var absorbs = List[Int]()  # (index << 8) | block type
        var close_from = -1
        var i = 1

        # phase 1: container walk
        while i < len(self.open):
            var b = self.open[i]
            var bt = self.blocks[b].t
            if bt == B_QUOTE:
                if raw_blank:
                    close_from = i
                    break
                if self.match_quote(cur):
                    i += 1
                    continue
                if self.blocks[b].blank_after or self.lazy_kind(cur, False) == 0:
                    close_from = i
                    break
                absorbs.append((i << 8) | B_QUOTE)
                i += 1
                continue
            if bt == B_ITEM:
                if raw_blank:
                    if len(self.blocks[b].children) == 0:
                        close_from = i
                        break
                    i += 1
                    continue
                if cur.indent() >= self.blocks[b].content_indent:
                    cur.skip_columns(self.blocks[b].content_indent)
                    i += 1
                    continue
                if self.blocks[b].blank_after or self.lazy_kind(cur, True) == 0:
                    close_from = i
                    break
                absorbs.append((i << 8) | B_ITEM)
                i += 1
                continue
            if bt == B_CODE_F:
                if self.absorbed_by_quote(absorbs):
                    close_from = self.first_quote(absorbs)
                    break
                self.fence_line(b, cur)
                return
            if bt == B_PARA:
                if cur.blank_rest():
                    var bb = self.open.pop()
                    self.close_block(bb)
                    self.blocks[bb].blank_after = True
                    self.note_blank()
                    return
                if len(absorbs) > 0:
                    self.lazy_paragraph(b, cur, absorbs[len(absorbs) - 1] & 0xFF)
                    return
                break  # paragraph reached: phase 2 may interrupt it
            if bt == B_CODE_IND:
                if self.absorbed_by_quote(absorbs):
                    close_from = self.first_quote(absorbs)
                    break
                if cur.blank_rest():
                    var c2 = Cur(cur.rest())
                    c2.skip_spaces(4)
                    self.blocks[b].lines.append(c2.rest())
                    return
                if cur.indent() >= 4:
                    cur.skip_columns(4)
                    self.blocks[b].lines.append(cur.rest())
                    return
                _ = self.open.pop()
                break
            if bt == B_HTML:
                if self.absorbed_by_quote(absorbs):
                    close_from = self.first_quote(absorbs)
                    break
                if self.html_line(b, cur):
                    return
                break
            # B_LIST / B_DOC: transparent
            i += 1

        if close_from >= 0:
            self.pop_to(close_from)
            var kept = List[Int]()
            for k in range(len(absorbs)):
                if (absorbs[k] >> 8) < close_from:
                    kept.append(absorbs[k])
            absorbs = kept.copy()

        if raw_blank or cur.blank_rest():
            self.note_blank()
            return

        var lazy_innermost = -1
        if len(absorbs) > 0:
            lazy_innermost = absorbs[len(absorbs) - 1] & 0xFF
        self.open_new_blocks(cur, lazy_innermost)

    def absorbed_by_quote(self, absorbs: List[Int]) -> Bool:
        for k in range(len(absorbs)):
            if (absorbs[k] & 0xFF) == B_QUOTE:
                return True
        return False

    def first_quote(self, absorbs: List[Int]) -> Int:
        var m = 1 << 30
        for k in range(len(absorbs)):
            if (absorbs[k] & 0xFF) == B_QUOTE:
                var idx = absorbs[k] >> 8
                if idx < m:
                    m = idx
        return m

    def note_blank(mut self):
        if len(self.open) <= 1:
            return
        var deepest = self.tip()
        self.blocks[deepest].blank_after = True
        var nch = len(self.blocks[deepest].children)
        if nch > 0:
            self.blocks[self.blocks[deepest].children[nch - 1]].blank_after = True
        # propagate up the container chain, but not past a block quote
        for k in range(len(self.open) - 1, 0, -1):
            var b = self.open[k]
            self.blocks[b].blank_after = True
            if self.blocks[b].t == B_QUOTE:
                break

    def match_quote(mut self, mut cur: Cur) -> Bool:
        var save = cur.copy()
        cur.skip_spaces(3)
        if b_at(cur.rest(), 0) != 62:
            cur = save.copy()
            return False
        cur.pos += 1
        cur.col += 1
        self.skip_one_space(cur)
        return True

    def skip_one_space(self, mut cur: Cur):
        var line = cur.line
        if cur.pos < line.byte_length():
            var b = Int(line.as_bytes()[cur.pos])
            if b == 32:
                cur.pos += 1
                cur.col += 1
            elif b == 9:
                var w = 4 - (cur.col % 4)
                if w == 1:
                    cur.pos += 1
                    cur.col += 1
                else:
                    cur.line = (
                        slice(line, 0, cur.pos)
                        + String(" ") * (w - 1)
                        + slice(line, cur.pos + 1)
                    )
                    cur.col += 1

    def lazy_kind(self, cur: Cur, in_item: Bool) -> Int:
        """0 = the container closes; 1 = it absorbs the line."""
        var probe = Cur(cur.rest())
        probe.skip_spaces(3)
        var stripped = probe.rest()
        if fence_start(stripped) >= 0:
            return 0
        if is_thematic(stripped):
            return 0
        if match_list_marker(stripped) != 0:
            return 0
        if in_item:
            if atx_level(stripped) >= 0:
                return 0
            if b_at(stripped, 0) == 62:
                return 0
        return 1

    def lazy_paragraph(mut self, para: Int, cur: Cur, innermost: Int) raises:
        var lead = Cur(cur.rest())
        lead.skip_spaces(3)
        var srest = lead.rest()
        if innermost == B_ITEM:
            var sl = setext_level(srest)
            if sl > 0:
                self.convert_setext(para, rstrip_of(srest, " \t"))
                return
        if innermost == B_QUOTE:
            var lv = atx_level(srest)
            if lv > 0:
                _ = self.open.pop()  # close the paragraph
                var h = self.add_block(B_HEAD)
                self.blocks[h].level = lv
                self.blocks[h].lines.append(atx_content(srest, lv))
                return
        if cur.indent() < 4 and self.open_html(cur, True):
            return
        self.blocks[para].lines.append(lstrip_of(cur.rest(), " \t"))

    def fence_line(mut self, blk: Int, cur: Cur):
        var rest = cur.rest()
        var probe = Cur(rest)
        probe.skip_spaces(3)
        var pr = probe.rest()
        var n = pr.byte_length()
        var ch = self.blocks[blk].fence_ch
        var i = 0
        while i < n and Int(pr.as_bytes()[i]) == ch:
            i += 1
        if i >= self.blocks[blk].fence_len and is_blank(slice(pr, i)):
            _ = self.open.pop()
            return
        var c2 = Cur(rest)
        c2.skip_spaces(self.blocks[blk].fence_indent)
        self.blocks[blk].lines.append(c2.rest())

    def html_line(mut self, blk: Int, cur: Cur) -> Bool:
        """Feed a line to an open HTML block; returns False if it just closed."""
        var rest = cur.rest()
        var kind = self.blocks[blk].html_end
        if kind == 6:  # "blank"
            if is_blank(rest):
                _ = self.open.pop()
                return False
            self.blocks[blk].lines.append(rest)
            return True
        self.blocks[blk].lines.append(rest)
        if html_end_hit(rest, kind, self.blocks[blk].html_tag):
            _ = self.open.pop()
        return True

    def open_html(mut self, cur: Cur, tip_is_para: Bool) raises -> Bool:
        var start = html_block_start(cur.rest(), tip_is_para)
        if start == 0:
            return False
        self.close_list_for_leaf()
        self.interrupt_paragraph()
        var b = self.add_block(B_HTML)
        self.blocks[b].html_end = start
        if start == 1:
            var rest = cur.rest()
            var low = rest.lower()
            for tag in List[String](["script", "pre", "style", "textarea"]):
                if low.find("<" + tag) == 0:
                    self.blocks[b].html_tag = tag
                    break
        self.blocks[b].lines.append(cur.rest())
        # same-line end conditions
        var first = self.blocks[b].lines[0]
        if html_end_hit(first, start, self.blocks[b].html_tag):
            pass  # block closes immediately (do not push)
        else:
            self.open.append(b)
        return True

    def close_list_for_leaf(mut self):
        if self.blocks[self.tip()].t == B_LIST:
            _ = self.open.pop()

    def convert_setext(mut self, para: Int, marker: String) raises:
        var level = 1 if b_at(marker, 0) == 61 else 2
        self.close_block(para)  # extract any reference definitions first
        if len(self.blocks[para].lines) == 0:
            if len(self.open) > 0 and self.open[len(self.open) - 1] == para:
                _ = self.open.pop()
            var c = Cur(marker)
            self.add_paragraph_line(c)
            return
        if len(self.open) > 0 and self.open[len(self.open) - 1] == para:
            _ = self.open.pop()
        var parent = self.tip()
        var nch = len(self.blocks[parent].children)
        _ = self.blocks[parent].children.pop()
        var h = self.add_block(B_HEAD)
        self.blocks[h].level = level
        for k in range(len(self.blocks[para].lines)):
            self.blocks[h].lines.append(rstrip_of(self.blocks[para].lines[k], " \t"))
        self.blocks[h].blank_after = self.blocks[para].blank_after

    def add_paragraph_line(mut self, cur: Cur) raises:
        var tip = self.tip()
        if self.blocks[tip].t == B_LIST:
            _ = self.open.pop()
            tip = self.tip()
        if self.blocks[tip].t != B_PARA:
            tip = self.add_block(B_PARA)
            self.open.append(tip)
        self.blocks[tip].lines.append(lstrip_of(cur.rest(), " \t"))

    def open_new_blocks(mut self, cur0: Cur, lazy_innermost: Int) raises:
        var cur = cur0.copy()
        while True:
            if cur.blank_rest():
                return
            var tip = self.tip()
            var tip_t = self.blocks[tip].t
            if lazy_innermost >= 0:
                var lead = Cur(cur.rest())
                lead.skip_spaces(3)
                var srest = lead.rest()
                var indented4 = cur.indent() >= 4
                if lazy_innermost == B_QUOTE and not indented4:
                    var lv = atx_level(srest)
                    if lv > 0:
                        self.interrupt_paragraph()
                        var h = self.add_block(B_HEAD)
                        self.blocks[h].level = lv
                        self.blocks[h].lines.append(atx_content(srest, lv))
                        return
                if lazy_innermost == B_ITEM and tip_t == B_PARA and not indented4:
                    var sl = setext_level(srest)
                    if sl > 0:
                        self.convert_setext(tip, rstrip_of(srest, " \t"))
                        return
                if not indented4 and self.open_html(cur, tip_t == B_PARA):
                    return
                self.add_paragraph_line(cur)
                return

            var indent = cur.indent()
            if indent <= 3:
                var save = cur.copy()
                cur.skip_columns(indent)
                var rest = cur.rest()
                if b_at(rest, 0) == 62:  # '>'
                    self.close_list_for_leaf()
                    self.interrupt_paragraph()
                    var q = self.add_block(B_QUOTE)
                    self.open.append(q)
                    cur.pos += 1
                    cur.col += 1
                    self.skip_one_space(cur)
                    continue
                var lv = atx_level(rest)
                if lv > 0:
                    self.close_list_for_leaf()
                    self.interrupt_paragraph()
                    var h = self.add_block(B_HEAD)
                    self.blocks[h].level = lv
                    self.blocks[h].lines.append(atx_content(rest, lv))
                    return
                var fs = fence_start(rest)
                if fs >= 0:
                    self.close_list_for_leaf()
                    self.interrupt_paragraph()
                    var fb = self.add_block(B_CODE_F)
                    self.blocks[fb].fence_ch = fs // 100
                    self.blocks[fb].fence_len = fs % 100
                    self.blocks[fb].fence_indent = indent
                    self.blocks[fb].info = fence_info(rest, fs % 100)
                    self.open.append(fb)
                    return
                if self.open_html(cur, tip_t == B_PARA):
                    return
                if tip_t == B_PARA:
                    var sl = setext_level(rest)
                    if sl > 0:
                        self.convert_setext(tip, rstrip_of(rest, " \t"))
                        return
                if is_thematic(rest):
                    self.close_list_for_leaf()
                    self.interrupt_paragraph()
                    _ = self.add_block(B_HR)
                    return
                var lm = match_list_marker(rest)
                if lm != 0:
                    var ordered = (lm >> 35) != 0
                    var start = (lm >> 5) & 0x3FFFFFFF
                    var width = (lm >> 1) & 15
                    var after = marker_after(rest, width)
                    if tip_t == B_PARA and (
                        is_blank(after) or (ordered and start != 1)
                    ):
                        cur = save.copy()
                        self.add_paragraph_line(cur)
                        return
                    if self.start_list_item(cur, indent, lm):
                        continue
                    return
                cur = save.copy()
            if indent >= 4:
                if tip_t == B_PARA:
                    self.add_paragraph_line(cur)
                    return
                self.close_list_for_leaf()
                var cb = self.add_block(B_CODE_IND)
                self.open.append(cb)
                cur.skip_columns(4)
                self.blocks[cb].lines.append(cur.rest())
                return
            self.add_paragraph_line(cur)
            return

    def start_list_item(mut self, mut cur: Cur, marker_indent: Int, lm: Int) raises -> Bool:
        var ordered = (lm >> 35) != 0
        var start = (lm >> 5) & 0x3FFFFFFF
        var width = (lm >> 1) & 15
        var bullet = 0
        var delim = 0
        if ordered:
            delim = b_at(cur.rest(), width - 1)
        else:
            bullet = b_at(cur.rest(), 0)
        self.interrupt_paragraph()
        var tip = self.tip()
        var lst = -1
        if self.blocks[tip].t == B_LIST:
            var same = self.blocks[tip].ordered == ordered and (
                (ordered and self.blocks[tip].delim == delim)
                or ((not ordered) and self.blocks[tip].bullet == bullet)
            )
            if same:
                lst = tip
            else:
                _ = self.open.pop()  # different list type: close it first
        if lst < 0:
            lst = self.add_block(B_LIST)
            self.blocks[lst].ordered = ordered
            self.blocks[lst].start = start
            self.blocks[lst].bullet = bullet
            self.blocks[lst].delim = delim
            self.open.append(lst)
        var item = self.add_block(B_ITEM)
        self.open.append(item)
        cur.pos += width
        cur.col += width
        if cur.blank_rest():
            self.blocks[item].content_indent = marker_indent + width + 1
            return True
        var peek = Cur(cur.line)
        peek.pos = cur.pos
        peek.col = cur.col
        var spaces = peek.skip_spaces(5)
        if 1 <= spaces <= 4:
            cur.skip_columns(spaces)
            self.blocks[item].content_indent = marker_indent + width + spaces
        else:
            cur.skip_columns(1)
            self.blocks[item].content_indent = marker_indent + width + 1
        return True


# --------------------------------------------------------------------------
# inline parsing
# --------------------------------------------------------------------------


struct Node(Copyable, Movable):
    var t: Int
    var text: String
    var children: List[Int]
    var parent: Int  # -1 = root list
    var dest: String
    var title: String

    def __init__(out self, t: Int, text: String):
        self.t = t
        self.text = text
        self.children = List[Int]()
        self.parent = -1
        self.dest = String("")
        self.title = String("")


struct Delim(Copyable, Movable):
    var node: Int
    var ch: Int
    var count: Int
    var orig: Int
    var can_open: Bool
    var can_close: Bool

    def __init__(out self, node: Int, ch: Int, count: Int, can_open: Bool, can_close: Bool):
        self.node = node
        self.ch = ch
        self.count = count
        self.orig = count
        self.can_open = can_open
        self.can_close = can_close


struct Bracket(Copyable, Movable):
    var node: Int
    var img: Bool
    var active: Bool
    var mark: Int
    var spos: Int

    def __init__(out self, node: Int, img: Bool, mark: Int, spos: Int):
        self.node = node
        self.img = img
        self.active = True
        self.mark = mark
        self.spos = spos


struct Inline:
    var nodes: List[Node]
    var roots: List[Int]
    var delims: List[Delim]
    var brackets: List[Bracket]
    var refs: Dict[String, RefDef]
    var punct: List[Int]
    var punct_ready: Bool
    var delim_nodes: Dict[Int, Bool]

    def __init__(out self):
        self.nodes = List[Node]()
        self.roots = List[Int]()
        self.delims = List[Delim]()
        self.brackets = List[Bracket]()
        self.refs = Dict[String, RefDef]()
        self.punct = List[Int]()
        self.punct_ready = False
        self.delim_nodes = Dict[Int, Bool]()

    def is_punct(mut self, cp: Int) -> Bool:
        # ASCII takes the fast path; the Unicode table is built on first use.
        if cp < 128:
            return _in_chars(cp, PUNCT_ASCII)
        if not self.punct_ready:
            self.punct = load_punct_ranges()
            self.punct_ready = True
        return is_punct_cp(cp, self.punct)

    def add_node(mut self, t: Int, text: String) -> Int:
        self.nodes.append(Node(t, text))
        var idx = len(self.nodes) - 1
        self.roots.append(idx)
        return idx

    def siblings_of(mut self, idx: Int) raises -> List[Int]:
        var p = self.nodes[idx].parent
        if p < 0:
            return self.roots.copy()
        return self.nodes[p].children.copy()

    def remove_node(mut self, idx: Int) raises:
        var p = self.nodes[idx].parent
        if p < 0:
            var k = self.roots.index(idx)
            self.roots.pop(k)
        else:
            var k = self.nodes[p].children.index(idx)
            self.nodes[p].children.pop(k)

    def parse(mut self, text0: String, refs: Dict[String, RefDef]) raises -> List[Int]:
        var text = rstrip_of(text0, " \t")
        var i = 0
        var n = text.byte_length()
        while i < n:
            var ch = Int(text.as_bytes()[i])
            if ch == 92:  # backslash
                if i + 1 < n and Int(text.as_bytes()[i + 1]) == 10:
                    self.trim_trailing_spaces(True)
                    _ = self.add_node(N_BR, "")
                    i += 2
                    continue
                if i + 1 < n and _in_chars(Int(text.as_bytes()[i + 1]), PUNCT_ASCII):
                    _ = self.add_node(N_TEXT, slice(text, i + 1, i + 2))
                    i += 2
                    continue
                _ = self.add_node(N_TEXT, "\\")
                i += 1
                continue
            if ch == 96:  # backtick
                var j = i
                while j < n and Int(text.as_bytes()[j]) == 96:
                    j += 1
                var run = j - i
                var k = j
                var found = -1
                while k < n:
                    if Int(text.as_bytes()[k]) == 96:
                        var m = k
                        while m < n and Int(text.as_bytes()[m]) == 96:
                            m += 1
                        if m - k == run:
                            found = m
                            break
                        k = m
                    else:
                        k += 1
                if found < 0:
                    _ = self.add_node(N_TEXT, String("`") * run)
                    i = j
                    continue
                _ = self.add_node(N_CODE, self.code_text(slice(text, j, found - run)))
                i = found
                continue
            if ch == 60:  # '<'
                var adv = self.try_autolink(text, i)
                if adv > 0:
                    i = adv
                    continue
                var hlen = html_token_len(text, i)
                if hlen > 0:
                    _ = self.add_node(N_TEXT, slice(text, i, i + hlen))
                    i += hlen
                    continue
                _ = self.add_node(N_TEXT, "<")
                i += 1
                continue
            if ch == 42 or ch == 95:  # * or _
                var j = i
                while j < n and Int(text.as_bytes()[j]) == ch:
                    j += 1
                var run = j - i
                var before = prev_cp(text, i)
                var after = next_cp(text, j)
                var fl = self.flanking(ch, before, after)
                var node = self.add_node(N_TEXT, slice(text, i, j))
                self.delims.append(Delim(node, ch, run, fl[0], fl[1]))
                self.delim_nodes[node] = True
                i = j
                continue
            if ch == 91 or (ch == 33 and i + 1 < n and Int(text.as_bytes()[i + 1]) == 91):
                var is_img = ch == 33
                var node = self.add_node(N_TEXT, "![" if is_img else "[")
                self.brackets.append(Bracket(node, is_img, len(self.delims), i + (2 if is_img else 1)))
                i += 2 if is_img else 1
                continue
            if ch == 93:  # ]
                var consumed = self.close_bracket(text, i, refs)
                if consumed > i:
                    i = consumed
                    continue
                _ = self.add_node(N_TEXT, "]")
                i += 1
                continue
            if ch == 10:  # newline
                var hard = self.trim_trailing_spaces(False)
                if hard:
                    _ = self.add_node(N_BR, "")
                else:
                    _ = self.add_node(N_SOFT, "\n")
                i += 1
                continue
            # plain text run: up to the next special char
            var j = i
            while j < n:
                var c = Int(text.as_bytes()[j])
                if c == 92 or c == 96 or c == 60 or c == 42 or c == 95 or c == 91 or c == 33 or c == 10 or c == 93:
                    break
                j += 1
            if j == i:  # '!' not followed by '['
                j = i + 1
            _ = self.add_node(N_TEXT, slice(text, i, j))
            i = j
        self.merge_text()
        var d = self.delims.copy()
        self.process_emphasis(d)
        self.delims = d.copy()
        self.merge_text()
        return self.roots.copy()

    def merge_text(mut self) raises:
        var out = List[Int]()
        for k in range(len(self.roots)):
            var idx = self.roots[k]
            if (
                len(out) > 0
                and self.nodes[idx].t == N_TEXT
                and self.nodes[out[len(out) - 1]].t == N_TEXT
                and not self.is_delim_node(idx)
                and not self.is_delim_node(out[len(out) - 1])
            ):
                var t_add = self.nodes[idx].text
                self.nodes[out[len(out) - 1]].text += t_add
            else:
                out.append(idx)
        self.roots = out.copy()

    def is_delim_node(self, idx: Int) -> Bool:
        return idx in self.delim_nodes

    def trim_trailing_spaces(mut self, hard: Bool) raises -> Bool:
        if len(self.roots) == 0:
            return False
        var last = self.roots[len(self.roots) - 1]
        if self.nodes[last].t != N_TEXT:
            return False
        var stripped = rstrip_of(self.nodes[last].text, " ")
        var nspaces = self.nodes[last].text.byte_length() - stripped.byte_length()
        if hard or nspaces > 0:
            self.nodes[last].text = stripped
        if hard:
            return True
        return nspaces >= 2

    def code_text(self, content: String) raises -> String:
        var c = content.replace("\n", " ")
        if (
            c.byte_length() >= 2
            and Int(c.as_bytes()[0]) == 32
            and Int(c.as_bytes()[c.byte_length() - 1]) == 32
            and not is_blank(c)
        ):
            c = slice(c, 1, c.byte_length() - 1)
        return c

    def try_autolink(mut self, text: String, i: Int) raises -> Int:
        """'<': URI or email autolink. Returns the position after '>', or 0."""
        var n = text.byte_length()
        var j = i + 1
        while j < n and Int(text.as_bytes()[j]) != 62 and Int(text.as_bytes()[j]) != 60 and Int(text.as_bytes()[j]) > 32:
            j += 1
        if j >= n or Int(text.as_bytes()[j]) != 62:
            return 0
        var body = slice(text, i + 1, j)
        # URI autolink: scheme (letter, then alnum/+/-/. up to 32) + ':' + rest
        var sp = scan_scheme(body)
        if sp > 0 and body.byte_length() >= sp + 1:
            var node = self.add_node(N_LINK, "")
            self.nodes[node].dest = body
            var child = self.add_node(N_TEXT, body)
            self.nodes[child].parent = node
            self.nodes[node].children.append(child)
            _ = self.roots.pop()  # child moved under the link
            return j + 1
        # email autolink
        if is_email(body):
            var node = self.add_node(N_LINK, "")
            self.nodes[node].dest = "mailto:" + body
            self.nodes[node].text = "email"
            var child = self.add_node(N_TEXT, body)
            self.nodes[child].parent = node
            self.nodes[node].children.append(child)
            _ = self.roots.pop()
            return j + 1
        return 0

    def flanking(mut self, ch: Int, before: Int, after: Int) -> List[Bool]:
        var before_ws = is_ws_cp(before)
        var after_ws = is_ws_cp(after)
        var before_p = self.is_punct(before)
        var after_p = self.is_punct(after)
        var left = (not after_ws) and (not after_p or (before_ws or before_p))
        var right = (not before_ws) and (not before_p or (after_ws or after_p))
        if ch == 42:  # '*'
            return List[Bool]([left, right])
        # underscore: intraword rules
        var can_open = left and (not right or before_p)
        var can_close = right and (not left or after_p)
        return List[Bool]([can_open, can_close])

    def close_bracket(mut self, text: String, i: Int, refs: Dict[String, RefDef]) raises -> Int:
        """A ']' at position i: try to form a link/image. Returns new pos or 0."""
        if len(self.brackets) == 0:
            return 0
        var bi = len(self.brackets) - 1
        var br = self.brackets[bi].copy()
        if not br.active:
            self.brackets.pop()
            return 0
        var opener = br.node
        var oi = self.roots.index(opener)
        var label_text = slice(text, br.spos, i)
        var after = i + 1
        var n = text.byte_length()
        var dest = String("")
        var title = String("")
        var end = after
        var have = False
        # inline link: [text](dest title)
        if after < n and Int(text.as_bytes()[after]) == 40:  # (
            var p = skip_spaces_nl(text, after + 1, 1)
            var d_end = scan_link_dest(text, p)
            if d_end < 0 and p < n and Int(text.as_bytes()[p]) == 41:
                dest = ""
                end = p + 1
                have = True
            elif d_end >= 0:
                var dv: String
                if Int(text.as_bytes()[p]) == 60:
                    dv = unescape(slice(text, p + 1, d_end - 1))
                else:
                    dv = unescape(slice(text, p, d_end))
                var p3 = skip_spaces_nl(text, d_end, 1)
                var tv = String("")
                var t_end = scan_link_title(text, p3)
                if t_end >= 0:
                    tv = unescape(slice(text, p3 + 1, t_end - 1))
                    p3 = skip_spaces_nl(text, t_end, 1)
                if p3 < n and Int(text.as_bytes()[p3]) == 41:
                    dest = dv
                    title = tv
                    end = p3 + 1
                    have = True
        if not have:
            # full/collapsed reference
            if after < n and Int(text.as_bytes()[after]) == 91:  # [
                var lab_end = scan_link_label(text, after)
                if lab_end >= 0:
                    var raw = slice(text, after + 1, lab_end - 1)
                    var key = String("")
                    if not is_blank(raw):
                        key = normalize_label(raw)
                    else:
                        key = normalize_label(label_text)
                    if key in refs:
                        var rd = refs[key].copy()
                        dest = rd.dest
                        title = rd.title
                        end = lab_end
                        have = True
            if not have and not (after < n and Int(text.as_bytes()[after]) == 91):
                # shortcut reference
                var key = normalize_label(label_text)
                if key in refs:
                    var rd = refs[key].copy()
                    dest = rd.dest
                    title = rd.title
                    end = after
                    have = True
        if not have:
            self.brackets.pop()
            return 0
        # move everything after the opener into the new node
        var kids = List[Int]()
        for k in range(oi + 1, len(self.roots)):
            kids.append(self.roots[k])
        # drop the '[' marker and everything after it
        var kept_roots = List[Int]()
        for k in range(oi):
            kept_roots.append(self.roots[k])
        self.roots = kept_roots.copy()
        # build the node (appended as a fresh root)
        var node = self.add_node(N_IMG if br.img else N_LINK, "")
        self.nodes[node].dest = dest
        self.nodes[node].title = title
        for k in range(len(kids)):
            self.nodes[kids[k]].parent = node
            self.nodes[node].children.append(kids[k])
        # process emphasis within the link text in isolation
        var mark = br.mark
        var sub = List[Delim]()
        for k in range(mark, len(self.delims)):
            sub.append(self.delims[k].copy())
        if len(sub) > 0:
            self.process_emphasis(sub)
        var kept_delims = List[Delim]()
        for k in range(mark):
            kept_delims.append(self.delims[k].copy())
        self.delims = kept_delims.copy()
        self.brackets.pop()
        # deactivate earlier link openers (no nested links)
        if not br.img:
            for k in range(len(self.brackets)):
                if not self.brackets[k].img:
                    self.brackets[k].active = False
        return end

    def process_emphasis(mut self, mut delims: List[Delim]) raises:
        var openers_bottom = Dict[Int, Int]()
        var ci = 0
        while ci < len(delims):
            var c = delims[ci].copy()
            if not (c.can_close and c.count > 0):
                ci += 1
                continue
            var key = delim_key(c)
            var bottom = openers_bottom.get(key, -1)
            var found = -1
            var oi = ci - 1
            while oi > bottom:
                var o = delims[oi].copy()
                if o.ch == c.ch and o.can_open and o.count > 0:
                    var odd = (
                        (c.can_open or o.can_close)
                        and (o.orig + c.orig) % 3 == 0
                        and (o.orig % 3 != 0 or c.orig % 3 != 0)
                    )
                    if not odd:
                        found = oi
                        break
                oi -= 1
            if found < 0:
                openers_bottom[key] = ci - 1
                ci += 1
                continue
            var o = delims[found].copy()
            var use = 2 if (o.count >= 2 and c.count >= 2) else 1
            self.wrap_emphasis(o, c, use)
            o.count -= use
            c.count -= use
            delims[found] = o.copy()
            delims[ci] = c.copy()
            # drop delimiters between opener and closer
            var nd = List[Delim]()
            for k in range(found + 1):
                nd.append(delims[k].copy())
            for k in range(ci, len(delims)):
                nd.append(delims[k].copy())
            delims = nd.copy()
            if o.count == 0:
                if self.nodes[o.node].text.byte_length() == 0:
                    self.remove_node(o.node)
                delims.pop(found)
                ci = found
            else:
                ci = found + 1
            if ci < len(delims):
                c = delims[ci].copy()
                if c.count == 0:
                    if self.nodes[c.node].text.byte_length() == 0:
                        self.remove_node(c.node)
                    delims.pop(ci)
        # callers own their lists; nothing to write back

    def wrap_emphasis(mut self, o: Delim, c: Delim, use: Int) raises:
        var onode = o.node
        var cnode = c.node
        var ot = self.nodes[onode].text
        self.nodes[onode].text = slice(ot, 0, max(0, ot.byte_length() - use))
        self.nodes[cnode].text = slice(self.nodes[cnode].text, use)
        # both nodes must be in the same sibling list
        if self.nodes[onode].parent != self.nodes[cnode].parent:
            return
        var sibs = self.siblings_of(onode)
        var oi = sibs.index(onode)
        var ci = sibs.index(cnode)
        # create the wrapper node directly in the arena (not in roots)
        self.nodes.append(Node(N_STRONG if use == 2 else N_EM, ""))
        var node = len(self.nodes) - 1
        self.nodes[node].parent = self.nodes[onode].parent
        var kids = List[Int]()
        for k in range(oi + 1, ci):
            kids.append(sibs[k])
        # splice: [.. opener, node, closer ..]
        for k in range(len(kids)):
            self.nodes[kids[k]].parent = node
            self.nodes[node].children.append(kids[k])
        var spliced = List[Int]()
        for k in range(oi + 1):
            spliced.append(sibs[k])
        spliced.append(node)
        for k in range(ci, len(sibs)):
            spliced.append(sibs[k])
        if self.nodes[onode].parent < 0:
            self.roots = spliced.copy()
        else:
            var p = self.nodes[onode].parent
            self.nodes[p].children = spliced.copy()

    def add_child_node(mut self, t: Int, text: String, parent: Int) -> Int:
        self.nodes.append(Node(t, text))
        var idx = len(self.nodes) - 1
        self.nodes[idx].parent = parent
        return idx


def delim_key(d: Delim) -> Int:
    return (d.ch << 4) | ((1 if d.can_open else 0) << 2) | (d.orig % 3)


# (flanking is an Inline method: the Unicode table is built lazily)


def scan_scheme(body: String) -> Int:
    """Length of a valid URI scheme at body start (2-32 chars), else 0."""
    var n = body.byte_length()
    if n < 3:
        return 0
    var b0 = Int(body.as_bytes()[0])
    if not ((65 <= b0 <= 90) or (97 <= b0 <= 122)):
        return 0
    var i = 1
    while i < n and i < 32:
        var b = Int(body.as_bytes()[i])
        if (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b == 43 or b == 45 or b == 46:
            i += 1
        else:
            break
    if i >= 2 and i < n and Int(body.as_bytes()[i]) == 58:
        return i
    return 0


def is_email(body: String) -> Bool:
    var n = body.byte_length()
    var at = body.find("@")
    if at <= 0 or at >= n - 1:
        return False
    # local part
    for k in range(at):
        var b = Int(body.as_bytes()[k])
        if not (
            (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57)
            or _in_chars(b, ".!#$%&'*+/=?^_`{|}~-")
        ):
            return False
    # domain: labels of alnum/dash (<=63, no leading/trailing dash)
    var dom = slice(body, at + 1)
    var labels = str_split(dom, ".")
    for k in range(len(labels)):
        var lab = labels[k]
        if lab.byte_length() == 0 or lab.byte_length() > 63:
            return False
        var b0 = Int(lab.as_bytes()[0])
        var b1 = Int(lab.as_bytes()[lab.byte_length() - 1])
        if b0 == 45 or b1 == 45:
            return False
        for j in range(lab.byte_length()):
            var b = Int(lab.as_bytes()[j])
            if not ((65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b == 45):
                return False
    return True


def html_token_len(text: String, i: Int) -> Int:
    """Length of an inline raw-HTML token at i (starts with '<'), else 0."""
    var n = text.byte_length()
    if i + 1 >= n:
        return 0
    var c1 = Int(text.as_bytes()[i + 1])
    if c1 == 33:  # !
        if starts_with(slice(text, i), "<!--"):
            # comment: not > or -> after <!--, no -- inside, not - before -->
            if b_at(text, i + 4) == 62:
                return 0
            if b_at(text, i + 4) == 45 and b_at(text, i + 5) == 62:
                return 0
            var j = i + 4
            while j + 2 < n + 1:
                if b_at(text, j) == 45 and b_at(text, j + 1) == 45 and b_at(text, j + 2) == 62:
                    if b_at(text, j - 1) == 45:
                        return 0
                    return j + 3
                if b_at(text, j) == 45 and b_at(text, j + 1) == 45:
                    return 0  # -- inside comment
                j += 1
            return 0
        if starts_with(slice(text, i), "<![CDATA["):
            var e = text.find("]]>", i + 9)
            return e + 3 - i if e >= 0 else 0
        var c2 = b_at(text, i + 2)
        if (65 <= c2 <= 90) or (97 <= c2 <= 122):
            var e = text.find(">", i + 2)
            return e + 1 - i if e >= 0 else 0
        return 0
    if c1 == 63:  # ?
        var e = text.find("?>", i + 2)
        return e + 2 - i if e >= 0 else 0
    # open/closing tag
    var j = i + 1
    if c1 == 47:
        j = i + 2
    var tn = scan_tagname(text, j)
    if tn == 0:
        return 0
    j += tn
    # attributes
    while True:
        var k = j
        while k < n and (Int(text.as_bytes()[k]) == 32 or Int(text.as_bytes()[k]) == 9 or Int(text.as_bytes()[k]) == 10):
            k += 1
        if k < n and Int(text.as_bytes()[k]) == 62:
            return k + 1 - i
        if k < n and Int(text.as_bytes()[k]) == 47 and b_at(text, k + 1) == 62:
            return k + 2 - i
        if k == j:
            return 0
        j = k
        # attribute name
        var c0 = b_at(text, j)
        if not ((65 <= c0 <= 90) or (97 <= c0 <= 122) or c0 == 95 or c0 == 58):
            return 0
        j += 1
        while j < n:
            var c = Int(text.as_bytes()[j])
            if (65 <= c <= 90) or (97 <= c <= 122) or (48 <= c <= 57) or c == 95 or c == 46 or c == 58 or c == 45:
                j += 1
            else:
                break
        # optional value
        var k2 = j
        while k2 < n and (Int(text.as_bytes()[k2]) == 32 or Int(text.as_bytes()[k2]) == 9 or Int(text.as_bytes()[k2]) == 10):
            k2 += 1
        if k2 < n and Int(text.as_bytes()[k2]) == 61:
            k2 += 1
            while k2 < n and (Int(text.as_bytes()[k2]) == 32 or Int(text.as_bytes()[k2]) == 9 or Int(text.as_bytes()[k2]) == 10):
                k2 += 1
            if k2 >= n:
                return 0
            var q = Int(text.as_bytes()[k2])
            if q == 39 or q == 34:
                k2 += 1
                while k2 < n and Int(text.as_bytes()[k2]) != q:
                    k2 += 1
                if k2 >= n:
                    return 0
                k2 += 1
            else:
                var u = k2
                while u < n:
                    var c = Int(text.as_bytes()[u])
                    if c <= 32 or c == 34 or c == 39 or c == 61 or c == 60 or c == 62 or c == 96:
                        break
                    u += 1
                if u == k2:
                    return 0
                k2 = u
            j = k2
    return 0


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


struct Renderer:
    var refs: Dict[String, RefDef]

    def __init__(out self, refs: Dict[String, RefDef]):
        self.refs = refs.copy()

    def render_inline(mut self, nodes: List[Node], roots: List[Int], mut out: ByteWriter) raises:
        for k in range(len(roots)):
            var ni = roots[k]
            var t = nodes[ni].t
            if t == N_TEXT:
                escape_html_into(out, nodes[ni].text)
            elif t == N_CODE:
                out.write("<code>")
                escape_html_into(out, nodes[ni].text)
                out.write("</code>")
            elif t == N_EM:
                out.write("<em>")
                self.render_inline(nodes, nodes[ni].children, out)
                out.write("</em>")
            elif t == N_STRONG:
                out.write("<strong>")
                self.render_inline(nodes, nodes[ni].children, out)
                out.write("</strong>")
            elif t == N_LINK:
                if url_is_harmful(nodes[ni].dest):
                    out.write('<a href="#harmful-link">')
                else:
                    out.write('<a href="')
                    out.write(escape_url(nodes[ni].dest))
                    out.write('"')
                    if nodes[ni].title.byte_length() > 0:
                        out.write(' title="')
                        out.write(escape_title(nodes[ni].title))
                        out.write('"')
                    out.write(">")
                self.render_inline(nodes, nodes[ni].children, out)
                out.write("</a>")
            elif t == N_IMG:
                if url_is_harmful(nodes[ni].dest):
                    out.write('<img src="#harmful-link"')
                else:
                    out.write('<img src="')
                    out.write(escape_url(nodes[ni].dest))
                    out.write('"')
                out.write(' alt="')
                escape_html_into(out, self.alt_text(nodes, nodes[ni].children))
                out.write('"')
                if nodes[ni].title.byte_length() > 0:
                    out.write(' title="')
                    out.write(escape_title(nodes[ni].title))
                    out.write('"')
                out.write(" />")
            elif t == N_BR:
                out.write("<br />\n")
            elif t == N_SOFT:
                out.write("\n")

    def alt_text(self, nodes: List[Node], kids: List[Int]) raises -> String:
        var out = String()
        for k in range(len(kids)):
            var ni = kids[k]
            if nodes[ni].t == N_TEXT or nodes[ni].t == N_CODE:
                out += nodes[ni].text
            elif nodes[ni].t == N_SOFT or nodes[ni].t == N_BR:
                out += "\n"
            else:
                out += self.alt_text(nodes, nodes[ni].children)
        return out

    def inline_html_into(mut self, text: String, mut out: ByteWriter) raises:
        var parser = Inline()
        var roots = parser.parse(text, self.refs)
        self.render_inline(parser.nodes, roots, out)

    def render_blocks(mut self, blocks: List[Blk], idxs: List[Int], mut out: ByteWriter) raises:
        for k in range(len(idxs)):
            var b = blocks[idxs[k]].copy()
            var t = b.t
            if t == B_PARA:
                var text = "\n".join(b.lines)
                out.write("<p>")
                self.inline_html_into(text, out)
                out.write("</p>\n")
            elif t == B_HEAD:
                var text = "\n".join(b.lines)
                out.write("<h" + String(b.level) + ">")
                self.inline_html_into(text, out)
                out.write("</h" + String(b.level) + ">\n")
            elif t == B_HR:
                out.write("<hr />\n")
            elif t == B_CODE_IND:
                var lines = List[String]()
                for j in range(len(b.lines)):
                    lines.append(b.lines[j])
                while len(lines) > 0 and lines[len(lines) - 1].byte_length() == 0:
                    lines.pop()
                out.write("<pre><code>")
                escape_html_into(out, "\n".join(lines))
                out.write("</code></pre>\n")
            elif t == B_CODE_F:
                var content = String()
                for j in range(len(b.lines)):
                    content += b.lines[j] + "\n"
                var cls = String("")
                var info = str_split(b.info, " ")
                var word = String("")
                for j in range(len(info)):
                    if info[j].byte_length() > 0:
                        word = info[j]
                        break
                if word.byte_length() > 0:
                    cls = ' class="language-' + escape_html(decode_entities(unescape(word))) + '"'
                out.write("<pre><code" + cls + ">")
                escape_html_into(out, content)
                out.write("</code></pre>\n")
            elif t == B_HTML:
                var lines = List[String]()
                for j in range(len(b.lines)):
                    lines.append(b.lines[j])
                while len(lines) > 0 and lines[len(lines) - 1].byte_length() == 0:
                    lines.pop()
                out.write("<p>")
                escape_html_into(out, "\n".join(lines))
                out.write("</p>\n")
            elif t == B_QUOTE:
                out.write("<blockquote>\n")
                self.render_blocks(blocks, b.children, out)
                out.write("</blockquote>\n")
            elif t == B_LIST:
                var tag = "ol" if b.ordered else "ul"
                var attr = String("")
                if b.ordered and b.start != 1:
                    attr = ' start="' + String(b.start) + '"'
                out.write("<" + tag + attr + ">\n")
                for j in range(len(b.children)):
                    self.render_item(blocks, b.children[j], out, b.tight)
                out.write("</" + tag + ">\n")

    def render_item(mut self, blocks: List[Blk], item: Int, mut out: ByteWriter, tight: Bool) raises:
        out.write("<li>")
        var b = blocks[item].copy()
        if tight:
            for j in range(len(b.children)):
                var child = blocks[b.children[j]].copy()
                if child.t == B_PARA:
                    var text = "\n".join(child.lines)
                    self.inline_html_into(text, out)
                else:
                    var single = List[Int]([b.children[j]])
                    self.render_blocks(blocks, single, out)
        else:
            self.render_blocks(blocks, b.children, out)
        out.write("</li>\n")


# --------------------------------------------------------------------------
# entry point + C ABI
# --------------------------------------------------------------------------


struct RenderResult(Copyable, Movable):
    var data: U8Ptr
    var size: Int64
    var status: Int32

    def __init__(out self, data: U8Ptr, size: Int64, status: Int32):
        self.data = data
        self.size = size
        self.status = status


def render_markdown(text: String) raises -> String:
    if text.byte_length() == 0:
        return String("")
    var parser = Parser()
    _ = parser.parse(text)
    var renderer = Renderer(parser.refs)
    var out = ByteWriter()
    renderer.render_blocks(parser.blocks, parser.blocks[0].children, out)
    return out.finish()


@export
def mistunemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def mistunemojo_render(data: U8Ptr, length: Int64) abi("C") -> Handle:
    var res = unsafe_alloc[RenderResult](1)
    var n = Int(length)
    if n < 0:
        n = 0
    var raw = List[UInt8]()
    for i in range(n):
        raw.append(data[i])
    var text = String(unsafe_from_utf8=raw^)
    try:
        var html = render_markdown(text)
        var size = Int64(html.byte_length())
        var buf = unsafe_alloc[UInt8](Int(size))
        unsafe_memcpy(dest=buf, src=html.unsafe_ptr(), count=Int(size))
        res[] = RenderResult(buf, size, 0)
    except:
        var dummy = unsafe_alloc[UInt8](1)
        res[] = RenderResult(dummy, 0, 2)
    return res.unsafe_bitcast[UInt8]()


@export
def mistunemojo_result_status(handle: Handle) abi("C") -> Int32:
    if not handle:
        return 2
    return handle.value().unsafe_bitcast[RenderResult]()[].status


@export
def mistunemojo_result_size(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    return handle.value().unsafe_bitcast[RenderResult]()[].size


@export
def mistunemojo_result_data(handle: Handle) abi("C") -> Handle:
    if not handle:
        return None
    return handle.value().unsafe_bitcast[RenderResult]()[].data


@export
def mistunemojo_result_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var res = handle.value().unsafe_bitcast[RenderResult]()
    res[].data.unsafe_free()
    res.unsafe_free()
