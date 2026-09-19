"""VADER sentiment scoring kernel, replicating the published rule set.

The rules are re-implemented from the algorithm as documented in Hutto, C.J. &
Gilbert, E.E. (2014), "VADER: A Parsimonious Rule-based Model for Sentiment
Analysis of Social Media Text" (ICWSM-14), matching the observable behavior of
the reference Python implementation (PyPI vaderSentiment 3.3.2) rule for rule:
punctuation emphasis, ALL-CAPS emphasis, booster/dampener words, the negation
window, "but" contrast, "no"/"least"/"never so-this" special cases, degree
modifiers, and emoji/emoticon handling. No third-party Mojo code is used.

Exported C ABI (one analyzer handle is built once from the lexicon tables,
then each call scores one UTF-8 text):

    int32_t  vadermojo_abi_version(void)
    void*    vadermojo_analyzer_create(words_blob, word_offsets, values,
                                       n_words, emoji_keys, desc_blob,
                                       desc_offsets, n_emojis)
    int32_t  vadermojo_polarity(handle, text_bytes, text_len, out4)
    void     vadermojo_analyzer_destroy(handle)

`vadermojo_polarity` writes four raw (unrounded) float64 scores in reference
order — neg, neu, pos, compound — into `out4`. Decimal rounding to the public
3/4-digit precision is applied by the Python wrapper so that the rounding
mode is byte-identical to the reference's `round()`.

All text is processed as Unicode code points (UInt32) decoded from the UTF-8
input; all arithmetic is IEEE-754 float64 in the same operation order as the
reference implementation, so raw scores agree bit-for-bit.
"""

from std.collections import Array, Dict
from std.math import sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Empirically derived constants of the VADER rule set.
comptime B_INCR: Float64 = 0.293
comptime B_DECR: Float64 = -0.293
comptime C_INCR: Float64 = 0.733
comptime N_SCALAR: Float64 = -0.74

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime SPACE: UInt32 = 0x20
# Element counts of the generated range tables (comptime-asserted below).
comptime U_N: Int = 1302
comptime NU_N: Int = 314
comptime BANG: UInt32 = 0x21  # '!'
comptime QMARK: UInt32 = 0x3F  # '?'


struct Token(Copyable, Movable):
    """One whitespace-delimited word: code points after punctuation stripping,
    their ASCII-lowercased form, and that form's UTF-8 bytes as a dict key."""

    var cps: List[UInt32]
    var lower_cps: List[UInt32]
    var lower_str: String

    def __init__(
        out self,
        var cps: List[UInt32],
        var lower_cps: List[UInt32],
        var lower_str: String,
    ):
        self.cps = cps^
        self.lower_cps = lower_cps^
        self.lower_str = lower_str^


struct Analyzer(Copyable, Movable):
    """Owned lexicon tables plus the static VADER rule tables."""

    var lexicon: Dict[String, Float64]
    var emojis: Dict[UInt32, List[UInt32]]
    var boosters: Dict[String, Float64]
    var negate: Dict[String, Bool]
    var specials: Dict[String, Float64]

    def __init__(out self):
        self.lexicon = Dict[String, Float64]()
        self.emojis = Dict[UInt32, List[UInt32]]()
        self.boosters = _make_booster_dict()
        self.negate = _make_negate_set()
        self.specials = _make_special_cases()


# ---------------------------------------------------------------------------
# Static rule tables (values of the VADER rule set).
# ---------------------------------------------------------------------------


def _make_booster_dict() -> Dict[String, Float64]:
    """Booster/dampener 'degree adverb' table (B_INCR / B_DECR valued)."""
    var d = Dict[String, Float64]()
    d["absolutely"] = B_INCR
    d["amazingly"] = B_INCR
    d["awfully"] = B_INCR
    d["completely"] = B_INCR
    d["considerable"] = B_INCR
    d["considerably"] = B_INCR
    d["decidedly"] = B_INCR
    d["deeply"] = B_INCR
    d["effing"] = B_INCR
    d["enormous"] = B_INCR
    d["enormously"] = B_INCR
    d["entirely"] = B_INCR
    d["especially"] = B_INCR
    d["exceptional"] = B_INCR
    d["exceptionally"] = B_INCR
    d["extreme"] = B_INCR
    d["extremely"] = B_INCR
    d["fabulously"] = B_INCR
    d["flipping"] = B_INCR
    d["flippin"] = B_INCR
    d["frackin"] = B_INCR
    d["fracking"] = B_INCR
    d["fricking"] = B_INCR
    d["frickin"] = B_INCR
    d["frigging"] = B_INCR
    d["friggin"] = B_INCR
    d["fully"] = B_INCR
    d["fuckin"] = B_INCR
    d["fucking"] = B_INCR
    d["fuggin"] = B_INCR
    d["fugging"] = B_INCR
    d["greatly"] = B_INCR
    d["hella"] = B_INCR
    d["highly"] = B_INCR
    d["hugely"] = B_INCR
    d["incredible"] = B_INCR
    d["incredibly"] = B_INCR
    d["intensely"] = B_INCR
    d["major"] = B_INCR
    d["majorly"] = B_INCR
    d["more"] = B_INCR
    d["most"] = B_INCR
    d["particularly"] = B_INCR
    d["purely"] = B_INCR
    d["quite"] = B_INCR
    d["really"] = B_INCR
    d["remarkably"] = B_INCR
    d["so"] = B_INCR
    d["substantially"] = B_INCR
    d["thoroughly"] = B_INCR
    d["total"] = B_INCR
    d["totally"] = B_INCR
    d["tremendous"] = B_INCR
    d["tremendously"] = B_INCR
    d["uber"] = B_INCR
    d["unbelievably"] = B_INCR
    d["unusually"] = B_INCR
    d["utter"] = B_INCR
    d["utterly"] = B_INCR
    d["very"] = B_INCR
    d["almost"] = B_DECR
    d["barely"] = B_DECR
    d["hardly"] = B_DECR
    d["just enough"] = B_DECR
    d["kind of"] = B_DECR
    d["kinda"] = B_DECR
    d["kindof"] = B_DECR
    d["kind-of"] = B_DECR
    d["less"] = B_DECR
    d["little"] = B_DECR
    d["marginal"] = B_DECR
    d["marginally"] = B_DECR
    d["occasional"] = B_DECR
    d["occasionally"] = B_DECR
    d["partly"] = B_DECR
    d["scarce"] = B_DECR
    d["scarcely"] = B_DECR
    d["slight"] = B_DECR
    d["slightly"] = B_DECR
    d["somewhat"] = B_DECR
    d["sort of"] = B_DECR
    d["sorta"] = B_DECR
    d["sortof"] = B_DECR
    d["sort-of"] = B_DECR
    return d^


def _make_negate_set() -> Dict[String, Bool]:
    """Negation words of the VADER rule set."""
    var d = Dict[String, Bool]()
    d["aint"] = True
    d["arent"] = True
    d["cannot"] = True
    d["cant"] = True
    d["couldnt"] = True
    d["darent"] = True
    d["didnt"] = True
    d["doesnt"] = True
    d["ain't"] = True
    d["aren't"] = True
    d["can't"] = True
    d["couldn't"] = True
    d["daren't"] = True
    d["didn't"] = True
    d["doesn't"] = True
    d["dont"] = True
    d["hadnt"] = True
    d["hasnt"] = True
    d["havent"] = True
    d["isnt"] = True
    d["mightnt"] = True
    d["mustnt"] = True
    d["neither"] = True
    d["don't"] = True
    d["hadn't"] = True
    d["hasn't"] = True
    d["haven't"] = True
    d["isn't"] = True
    d["mightn't"] = True
    d["mustn't"] = True
    d["neednt"] = True
    d["needn't"] = True
    d["never"] = True
    d["none"] = True
    d["nope"] = True
    d["nor"] = True
    d["not"] = True
    d["nothing"] = True
    d["nowhere"] = True
    d["oughtnt"] = True
    d["shant"] = True
    d["shouldnt"] = True
    d["uhuh"] = True
    d["wasnt"] = True
    d["werent"] = True
    d["oughtn't"] = True
    d["shan't"] = True
    d["shouldn't"] = True
    d["uh-uh"] = True
    d["wasn't"] = True
    d["weren't"] = True
    d["without"] = True
    d["wont"] = True
    d["wouldnt"] = True
    d["won't"] = True
    d["wouldn't"] = True
    d["rarely"] = True
    d["seldom"] = True
    d["despite"] = True
    return d^


def _make_special_cases() -> Dict[String, Float64]:
    """Special-case idioms/phrases that contain lexicon words."""
    var d = Dict[String, Float64]()
    d["the shit"] = 3.0
    d["the bomb"] = 3.0
    d["bad ass"] = 1.5
    d["badass"] = 1.5
    d["bus stop"] = 0.0
    d["yeah right"] = -2.0
    d["kiss of death"] = -1.5
    d["to die for"] = 3.0
    d["beating heart"] = 3.5
    return d^


# ---------------------------------------------------------------------------
# Code-point helpers (Python str semantics, ASCII case mapping).
# ---------------------------------------------------------------------------


def _is_py_space(cp: UInt32) -> Bool:
    """Unicode whitespace exactly as Python's str.split()/str.strip() see it."""
    return (
        (cp >= 0x09 and cp <= 0x0D)
        or (cp >= 0x1C and cp <= 0x20)
        or cp == 0x85
        or cp == 0xA0
        or cp == 0x1680
        or (cp >= 0x2000 and cp <= 0x200A)
        or cp == 0x2028
        or cp == 0x2029
        or cp == 0x202F
        or cp == 0x205F
        or cp == 0x3000
    )


def _is_ascii_punct(cp: UInt32) -> Bool:
    """Membership in Python's string.punctuation (32 ASCII characters)."""
    return (
        (cp >= 0x21 and cp <= 0x2F)
        or (cp >= 0x3A and cp <= 0x40)
        or (cp >= 0x5B and cp <= 0x60)
        or (cp >= 0x7B and cp <= 0x7E)
    )


def _is_upper(
    upper: Array[UInt32, U_N], not_upper: Array[UInt32, NU_N], cps: List[UInt32]
) -> Bool:
    """Python str.isupper(): at least one cased character and every cased
    character uppercase. Casing uses the generated Unicode tables (matching
    the reference interpreter's per-character isupper/islower/istitle)."""
    var has_upper = False
    for cp in cps:
        var is_u = _search(upper, cp)
        if not is_u and _search(not_upper, cp):
            return False
        if is_u:
            has_upper = True
    return has_upper


def _search[table_len: Int](table: Array[UInt32, table_len], cp: UInt32) -> Bool:
    """Binary search for `cp` in flattened inclusive [lo, hi] ranges."""
    var lo = 0
    var hi = table_len // 2  # one-past-last range index
    while lo < hi:
        var mid = (lo + hi) // 2
        if cp < table[mid * 2]:
            hi = mid
        elif cp > table[mid * 2 + 1]:
            lo = mid + 1
        else:
            return True
    return False


def _lower(cps: List[UInt32]) -> List[UInt32]:
    """str.lower(), specialized to what VADER's lookups can observe.

    A-Z map to a-z. U+212A (KELVIN SIGN) maps to 'k' — the only non-ASCII code
    point whose lowercase is pure ASCII, and therefore the only one that can
    change a lookup against the (ASCII) lexicon/booster/negation tables. Every
    other non-ASCII code point passes through unchanged: its Python-lowercased
    form is also non-ASCII, so it can never equal an ASCII key, a rule literal
    ('no', 'never', 'so', 'but', ...), or the "n't" negation substring; the
    lexicon's two non-ASCII keys (':-Þ', ':Þ') are unreachable in the
    reference (its per-token .lower() can never produce them) and are filtered
    out of the native tables by the wrapper.
    """
    var out = List[UInt32](capacity=len(cps))
    for cp in cps:
        if cp >= 0x41 and cp <= 0x5A:
            out.append(cp + 0x20)
        elif cp == 0x212A:
            out.append(UInt32(0x6B))
        else:
            out.append(cp)
    return out^


def _contains_nt(lower_cps: List[UInt32]) -> Bool:
    """Substring check for "n't" (the reference's negated() include_nt path)."""
    var n = len(lower_cps)
    var i = 0
    while i + 3 <= n:
        if (
            lower_cps[i] == 0x6E  # 'n'
            and lower_cps[i + 1] == 0x27  # '\''
            and lower_cps[i + 2] == 0x74  # 't'
        ):
            return True
        i += 1
    return False


def _decode_utf8(ptr: U8Ptr, n: Int) -> List[UInt32]:
    """Decode UTF-8 bytes to code points. Input always comes from Python's
    str.encode('utf-8', 'surrogatepass'), so it is well-formed; lone surrogate
    code points pass through as-is (they match nothing, as in the reference).
    Malformed bytes, which cannot occur on that path, decode as U+FFFD."""
    var out = List[UInt32](capacity=n)
    var i = 0
    while i < n:
        var b0 = UInt32(ptr[unsafe_offset=i])
        if b0 < 0x80:
            out.append(b0)
            i += 1
        elif b0 >= 0xC0 and b0 < 0xE0 and i + 1 < n:
            var b1 = UInt32(ptr[unsafe_offset=i + 1])
            if b1 >= 0x80 and b1 < 0xC0:
                out.append(((b0 & 0x1F) << 6) | (b1 & 0x3F))
                i += 2
            else:
                out.append(0xFFFD)
                i += 1
        elif b0 >= 0xE0 and b0 < 0xF0 and i + 2 < n:
            var b1 = UInt32(ptr[unsafe_offset=i + 1])
            var b2 = UInt32(ptr[unsafe_offset=i + 2])
            if (b1 >= 0x80 and b1 < 0xC0) and (b2 >= 0x80 and b2 < 0xC0):
                out.append(((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F))
                i += 3
            else:
                out.append(0xFFFD)
                i += 1
        elif b0 >= 0xF0 and i + 3 < n:
            var b1 = UInt32(ptr[unsafe_offset=i + 1])
            var b2 = UInt32(ptr[unsafe_offset=i + 2])
            var b3 = UInt32(ptr[unsafe_offset=i + 3])
            if (
                (b1 >= 0x80 and b1 < 0xC0)
                and (b2 >= 0x80 and b2 < 0xC0)
                and (b3 >= 0x80 and b3 < 0xC0)
            ):
                out.append(
                    ((b0 & 0x07) << 18)
                    | ((b1 & 0x3F) << 12)
                    | ((b2 & 0x3F) << 6)
                    | (b3 & 0x3F)
                )
                i += 4
            else:
                out.append(0xFFFD)
                i += 1
        else:
            out.append(0xFFFD)
            i += 1
    return out^


def _encode_utf8(cps: List[UInt32]) -> List[UInt8]:
    """Encode code points to UTF-8 bytes (surrogates as 3-byte forms, matching
    Python's 'surrogatepass' so keys round-trip byte-identically)."""
    var out = List[UInt8](capacity=len(cps))
    for cp in cps:
        if cp < 0x80:
            out.append(UInt8(cp))
        elif cp < 0x800:
            out.append(UInt8(0xC0 | (cp >> 6)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            out.append(UInt8(0xE0 | (cp >> 12)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
        else:
            out.append(UInt8(0xF0 | (cp >> 18)))
            out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            out.append(UInt8(0x80 | (cp & 0x3F)))
    return out^


def _key_of(cps: List[UInt32]) -> String:
    """Dict key for a code-point sequence (its UTF-8 bytes as a String)."""
    var bytes = _encode_utf8(cps)
    return String(unsafe_from_utf8=Span(bytes))


# ---------------------------------------------------------------------------
# Rule helpers (same decomposition as the reference implementation).
# ---------------------------------------------------------------------------


def _negated_word(a: Analyzer, tok: Token) -> Bool:
    """negated([word]) on the lowered word: NEGATE membership or "n't"."""
    if tok.lower_str in a.negate:
        return True
    return _contains_nt(tok.lower_cps)


def _scalar_inc_dec(
    a: Analyzer,
    upper: Array[UInt32, U_N],
    not_upper: Array[UInt32, NU_N],
    tok: Token,
    valence: Float64,
    is_cap_diff: Bool,
) -> Float64:
    """Booster/dampener scalar for one preceding word or emoticon."""
    var scalar = Float64(0.0)
    var booster = a.boosters.get(tok.lower_str)
    if booster:
        scalar = booster.value()
        if valence < 0:
            scalar *= -1
        # check if the booster/dampener word is in ALLCAPS (while others aren't)
        if _is_upper(upper, not_upper, tok.cps) and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


def _negation_check(
    a: Analyzer, var valence: Float64, toks: List[Token], start_i: Int, i: Int
) -> Float64:
    """Negation window of 1-3 words before the lexicon item, including the
    reference's 'never so/this' amplification and 'without doubt' exemption,
    with the reference's operator-precedence behavior in the start_i == 2 arm."""
    if start_i == 0:
        if _negated_word(a, toks[i - 1]):
            valence = valence * N_SCALAR
    elif start_i == 1:
        if toks[i - 2].lower_str == "never" and (
            toks[i - 1].lower_str == "so" or toks[i - 1].lower_str == "this"
        ):
            valence = valence * 1.25
        elif toks[i - 2].lower_str == "without" and toks[
            i - 1
        ].lower_str == "doubt":
            valence = valence
        elif _negated_word(a, toks[i - 2]):
            valence = valence * N_SCALAR
    elif start_i == 2:
        if (
            toks[i - 3].lower_str == "never"
            and (toks[i - 2].lower_str == "so" or toks[i - 2].lower_str == "this")
        ) or (toks[i - 1].lower_str == "so" or toks[i - 1].lower_str == "this"):
            valence = valence * 1.25
        elif toks[i - 3].lower_str == "without" and (
            toks[i - 2].lower_str == "doubt" or toks[i - 1].lower_str == "doubt"
        ):
            valence = valence
        elif _negated_word(a, toks[i - 3]):
            valence = valence * N_SCALAR
    return valence


def _join2(x: String, y: String) -> String:
    return x + String(" ") + y


def _join3(x: String, y: String, z: String) -> String:
    return x + String(" ") + y + String(" ") + z


def _special_idioms_check(
    a: Analyzer, var valence: Float64, toks: List[Token], i: Int
) -> Float64:
    """Special-case idioms around position i (requires i >= 3), then the
    booster bi-grams ('sort of', 'kind of', ...) ending at i - 1."""
    var onezero = _join2(toks[i - 1].lower_str, toks[i].lower_str)
    var twoonezero = _join3(toks[i - 2].lower_str, toks[i - 1].lower_str, toks[i].lower_str)
    var twoone = _join2(toks[i - 2].lower_str, toks[i - 1].lower_str)
    var threetwoone = _join3(toks[i - 3].lower_str, toks[i - 2].lower_str, toks[i - 1].lower_str)
    var threetwo = _join2(toks[i - 3].lower_str, toks[i - 2].lower_str)

    var sequences = List[String]()
    sequences.append(onezero.copy())
    sequences.append(twoonezero.copy())
    sequences.append(twoone.copy())
    sequences.append(threetwoone.copy())
    sequences.append(threetwo.copy())
    for seq in sequences:
        var special = a.specials.get(seq)
        if special:
            valence = special.value()
            break

    var n = len(toks)
    if n - 1 > i:
        var zeroone = _join2(toks[i].lower_str, toks[i + 1].lower_str)
        var special = a.specials.get(zeroone)
        if special:
            valence = special.value()
    if n - 1 > i + 1:
        var zeroonetwo = _join3(toks[i].lower_str, toks[i + 1].lower_str, toks[i + 2].lower_str)
        var special = a.specials.get(zeroonetwo)
        if special:
            valence = special.value()

    var n_grams = List[String]()
    n_grams.append(threetwoone^)
    n_grams.append(threetwo^)
    n_grams.append(twoone^)
    for n_gram in n_grams:
        var booster = a.boosters.get(n_gram)
        if booster:
            valence = valence + booster.value()
    return valence


def _least_check(
    a: Analyzer, var valence: Float64, toks: List[Token], i: Int
) -> Float64:
    """Negation via 'least' (except 'at least' / 'very least')."""
    if (
        i > 1
        and toks[i - 1].lower_str not in a.lexicon
        and toks[i - 1].lower_str == "least"
    ):
        if toks[i - 2].lower_str != "at" and toks[i - 2].lower_str != "very":
            valence = valence * N_SCALAR
    elif (
        i > 0
        and toks[i - 1].lower_str not in a.lexicon
        and toks[i - 1].lower_str == "least"
    ):
        valence = valence * N_SCALAR
    return valence


def _sentiment_valence(
    a: Analyzer,
    upper: Array[UInt32, U_N],
    not_upper: Array[UInt32, NU_N],
    toks: List[Token],
    i: Int,
    is_cap_diff: Bool,
) -> Float64:
    """Valence of the token at i with all rule adjustments applied."""
    var valence = Float64(0.0)
    ref item = toks[i].lower_str
    var lex = a.lexicon.get(item)
    if lex:
        valence = lex.value()
        var n = len(toks)
        # "no" as negation of an adjacent lexicon item vs a stand-alone item
        if item == "no" and i != n - 1 and toks[i + 1].lower_str in a.lexicon:
            valence = 0.0
        if (
            (i > 0 and toks[i - 1].lower_str == "no")
            or (i > 1 and toks[i - 2].lower_str == "no")
            or (
                i > 2
                and toks[i - 3].lower_str == "no"
                and (toks[i - 1].lower_str == "or" or toks[i - 1].lower_str == "nor")
            )
        ):
            valence = lex.value() * N_SCALAR

        # ALL-CAPS emphasis (while other words aren't all caps)
        if _is_upper(upper, not_upper, toks[i].cps) and is_cap_diff:
            if valence > 0:
                valence += C_INCR
            else:
                valence -= C_INCR

        for start_i in range(3):
            # dampen the scalar modifier of preceding words and emoticons
            # (excluding the ones that immediately precede the item) based on
            # their distance from the current item.
            if i > start_i and toks[i - (start_i + 1)].lower_str not in a.lexicon:
                var s = _scalar_inc_dec(
                    a, upper, not_upper, toks[i - (start_i + 1)], valence, is_cap_diff
                )
                if start_i == 1 and s != 0:
                    s = s * 0.95
                if start_i == 2 and s != 0:
                    s = s * 0.9
                valence = valence + s
                valence = _negation_check(a, valence, toks, start_i, i)
                if start_i == 2:
                    valence = _special_idioms_check(a, valence, toks, i)

        valence = _least_check(a, valence, toks, i)
    return valence


def _allcap_differential(
    upper: Array[UInt32, U_N], not_upper: Array[UInt32, NU_N], toks: List[Token]
) -> Bool:
    """True if some but not all words are ALL CAPS."""
    var allcap_words = 0
    for tok in toks:
        if _is_upper(upper, not_upper, tok.cps):
            allcap_words += 1
    var cap_differential = len(toks) - allcap_words
    return cap_differential > 0 and cap_differential < len(toks)


def _but_check(toks: List[Token], mut sentiments: List[Float64]):
    """Contrastive conjunction 'but': sentiment before the first 'but' is
    halved, after it is boosted 1.5x. Replicates the reference's index-based
    mutation loop exactly, including its behavior on duplicate values."""
    var bi = -1
    for idx in range(len(toks)):
        if toks[idx].lower_str == "but":
            bi = idx
            break
    if bi < 0:
        return
    var k = 0
    var m = len(sentiments)
    while k < m:
        var sentiment = sentiments[k]
        # first index whose element == sentiment (float equality; -0.0 == 0.0)
        var si = -1
        for j in range(m):
            if sentiments[j] == sentiment:
                si = j
                break
        if si < bi:
            sentiments[si] = sentiment * 0.5
        elif si > bi:
            sentiments[si] = sentiment * 1.5
        k += 1


def _normalize(score: Float64) -> Float64:
    """Normalize the score between -1 and 1 (alpha = 15)."""
    var norm_score = score / sqrt((score * score) + 15.0)
    if norm_score < -1.0:
        return -1.0
    if norm_score > 1.0:
        return 1.0
    return norm_score


def _strip_punc_if_word(cps: List[UInt32]) -> List[UInt32]:
    """Strip leading/trailing ASCII punctuation; if two or fewer code points
    remain it was likely an emoticon, so keep the original token."""
    var s = 0
    var e = len(cps)
    while s < e and _is_ascii_punct(cps[s]):
        s += 1
    while e > s and _is_ascii_punct(cps[e - 1]):
        e -= 1
    var out_list = List[UInt32](capacity=len(cps))
    if e - s <= 2:
        for cp in cps:
            out_list.append(cp)
    else:
        for idx in range(s, e):
            out_list.append(cps[idx])
    return out_list^


def _split_and_clean(sub: List[UInt32]) -> List[Token]:
    """Python str.split() on Unicode whitespace, then per-token punctuation
    stripping, ASCII lowering, and dict-key materialization."""
    var toks = List[Token]()
    var n = len(sub)
    var i = 0
    while i < n:
        while i < n and _is_py_space(sub[i]):
            i += 1
        if i >= n:
            break
        var start = i
        while i < n and not _is_py_space(sub[i]):
            i += 1
        var word = List[UInt32](capacity=i - start)
        for idx in range(start, i):
            word.append(sub[idx])
        var cps = _strip_punc_if_word(word)
        var lower_cps = _lower(cps)
        var key = _key_of(lower_cps)
        toks.append(Token(cps^, lower_cps^, key^))
    return toks^


def _emoji_substitute(a: Analyzer, text_cps: List[UInt32]) -> List[UInt32]:
    """Replace each emoji code point with its textual description (single
    space inserted when the previous character was not a space), then strip
    whitespace at both ends."""
    var sub = List[UInt32](capacity=len(text_cps))
    var prev_space = True
    for cp in text_cps:
        var emoji = a.emojis.get(cp)
        if emoji:
            if not prev_space:
                sub.append(SPACE)
            for d in emoji.value():
                sub.append(d)
            prev_space = False
        else:
            sub.append(cp)
            prev_space = cp == SPACE
    var start = 0
    var end = len(sub)
    while start < end and _is_py_space(sub[start]):
        start += 1
    while end > start and _is_py_space(sub[end - 1]):
        end -= 1
    var out_list = List[UInt32](capacity=end - start)
    for idx in range(start, end):
        out_list.append(sub[idx])
    return out_list^


def _score_valence(sentiments: List[Float64], sub: List[UInt32], out4: F64Ptr):
    """Combine per-word valences into raw neg/neu/pos/compound scores."""
    if len(sentiments) == 0:
        out4[unsafe_offset=0] = 0.0
        out4[unsafe_offset=1] = 0.0
        out4[unsafe_offset=2] = 0.0
        out4[unsafe_offset=3] = 0.0
        return

    var sum_s = Float64(0.0)
    for s in sentiments:
        sum_s += s

    # emphasis from exclamation points (up to 4) and question marks (2 or 3+),
    # counted over the whole (emoji-substituted, stripped) text
    var ep_count = 0
    var qm_count = 0
    for cp in sub:
        if cp == BANG:
            ep_count += 1
        elif cp == QMARK:
            qm_count += 1
    if ep_count > 4:
        ep_count = 4
    var ep_amplifier = Float64(ep_count) * 0.292
    var qm_amplifier = Float64(0.0)
    if qm_count > 1:
        if qm_count <= 3:
            qm_amplifier = Float64(qm_count) * 0.18
        else:
            qm_amplifier = 0.96
    var punct_emph_amplifier = ep_amplifier + qm_amplifier

    if sum_s > 0:
        sum_s += punct_emph_amplifier
    elif sum_s < 0:
        sum_s -= punct_emph_amplifier

    var compound = _normalize(sum_s)

    # discriminate between positive, negative and neutral sentiment scores
    var pos_sum = Float64(0.0)
    var neg_sum = Float64(0.0)
    var neu_count = 0
    for s in sentiments:
        if s > 0:
            pos_sum += s + 1
        if s < 0:
            neg_sum += s - 1
        if s == 0:
            neu_count += 1

    if pos_sum > abs(neg_sum):
        pos_sum += punct_emph_amplifier
    elif pos_sum < abs(neg_sum):
        neg_sum -= punct_emph_amplifier

    var total = pos_sum + abs(neg_sum) + Float64(neu_count)
    var pos = abs(pos_sum / total)
    var neg = abs(neg_sum / total)
    var neu = abs(Float64(neu_count) / total)

    out4[unsafe_offset=0] = neg
    out4[unsafe_offset=1] = neu
    out4[unsafe_offset=2] = pos
    out4[unsafe_offset=3] = compound


def _polarity(a: Analyzer, text_cps: List[UInt32], out4: F64Ptr):
    """The full VADER pipeline for one text: emoji substitution, tokenize,
    per-word valence, 'but' contrast, final scoring."""
    comptime assert len(_UPPER_RANGES) == U_N
    comptime assert len(_NOT_UPPER_RANGES) == NU_N
    var upper = materialize[_UPPER_RANGES]()
    var not_upper = materialize[_NOT_UPPER_RANGES]()
    var sub = _emoji_substitute(a, text_cps)
    var toks = _split_and_clean(sub)
    var n = len(toks)
    var is_cap_diff = _allcap_differential(upper, not_upper, toks)

    var sentiments = List[Float64](capacity=n)
    for i in range(n):
        var valence = Float64(0.0)
        # booster words and the 'kind of' dampener contribute no valence
        if toks[i].lower_str in a.boosters:
            sentiments.append(valence)
            continue
        if (
            i < n - 1
            and toks[i].lower_str == "kind"
            and toks[i + 1].lower_str == "of"
        ):
            sentiments.append(valence)
            continue
        sentiments.append(_sentiment_valence(a, upper, not_upper, toks, i, is_cap_diff))

    _but_check(toks, sentiments)
    _score_valence(sentiments, sub, out4)


# ---------------------------------------------------------------------------
# Exported C ABI.
# ---------------------------------------------------------------------------


@export
def vadermojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def vadermojo_analyzer_create(
    words_blob: U8Ptr,
    word_offsets: I64Ptr,
    values: F64Ptr,
    n_words: Int64,
    emoji_keys: U32Ptr,
    desc_blob: U8Ptr,
    desc_offsets: I64Ptr,
    n_emojis: Int64,
) abi("C") -> Handle:
    """Build a native analyzer from the lexicon tables; NULL on invalid input.

    `words_blob`/`word_offsets` hold the concatenated UTF-8 bytes of the
    lexicon words (word i spans offsets[i]..offsets[i+1]) and `values` its
    float64 valences. `emoji_keys` holds one Unicode code point per emoji and
    `desc_blob`/`desc_offsets` the concatenated UTF-8 description bytes.
    Everything is copied; the caller may free its buffers after the call.
    """
    if n_words < 0 or n_emojis < 0:
        return None

    var a = unsafe_alloc[Analyzer](1)
    a[] = Analyzer()
    for w in range(Int(n_words)):
        var s = Int(word_offsets[unsafe_offset=w])
        var e = Int(word_offsets[unsafe_offset=w + 1])
        if e < s:
            a.unsafe_free()
            return None
        var key = String(
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=words_blob.unsafe_offset(s), length=e - s
            )
        )
        a[].lexicon[key^] = values[unsafe_offset=w]
    for k in range(Int(n_emojis)):
        var s = Int(desc_offsets[unsafe_offset=k])
        var e = Int(desc_offsets[unsafe_offset=k + 1])
        if e < s:
            a.unsafe_free()
            return None
        a[].emojis[emoji_keys[unsafe_offset=k]] = _decode_utf8(
            desc_blob.unsafe_offset(s), e - s
        )
    return a.unsafe_bitcast[UInt8]()


@export
def vadermojo_polarity(
    handle: Handle,
    text: U8Ptr,
    text_len: Int64,
    out4: F64Ptr,
) abi("C") -> Int32:
    """Score one UTF-8 text. `out` must hold four float64 slots, filled with
    raw (unrounded) neg, neu, pos, compound. Returns 0 on success, 1 on a NULL
    handle, 2 on a negative text length."""
    if not handle:
        return 1
    if text_len < 0:
        return 2
    var a = handle.value().unsafe_bitcast[Analyzer]()
    var text_cps = _decode_utf8(text, Int(text_len))
    _polarity(a[], text_cps, out4)
    return 0


@export
def vadermojo_analyzer_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var a = handle.value().unsafe_bitcast[Analyzer]()
    a[] = Analyzer()  # reassign: the old lexicon tables are deinitialized
    a.unsafe_free()


# ---------------------------------------------------------------------------
# Unicode case tables (generated)
# ---------------------------------------------------------------------------


# GENERATED from CPython 3.12 unicodedata 15.0.0 per-code-point
# str.isupper() / str.islower() / str.istitle() ground truth; do not edit by hand.
# Flattened inclusive [lo, hi] code-point ranges, binary-searched.

comptime _UPPER_RANGES = [
    UInt32(65), UInt32(90), UInt32(192), UInt32(214), UInt32(216), UInt32(222), UInt32(256), UInt32(256), UInt32(258), UInt32(258), UInt32(260), UInt32(260), UInt32(262), UInt32(262),
    UInt32(264), UInt32(264), UInt32(266), UInt32(266), UInt32(268), UInt32(268), UInt32(270), UInt32(270), UInt32(272), UInt32(272), UInt32(274), UInt32(274), UInt32(276), UInt32(276),
    UInt32(278), UInt32(278), UInt32(280), UInt32(280), UInt32(282), UInt32(282), UInt32(284), UInt32(284), UInt32(286), UInt32(286), UInt32(288), UInt32(288), UInt32(290), UInt32(290),
    UInt32(292), UInt32(292), UInt32(294), UInt32(294), UInt32(296), UInt32(296), UInt32(298), UInt32(298), UInt32(300), UInt32(300), UInt32(302), UInt32(302), UInt32(304), UInt32(304),
    UInt32(306), UInt32(306), UInt32(308), UInt32(308), UInt32(310), UInt32(310), UInt32(313), UInt32(313), UInt32(315), UInt32(315), UInt32(317), UInt32(317), UInt32(319), UInt32(319),
    UInt32(321), UInt32(321), UInt32(323), UInt32(323), UInt32(325), UInt32(325), UInt32(327), UInt32(327), UInt32(330), UInt32(330), UInt32(332), UInt32(332), UInt32(334), UInt32(334),
    UInt32(336), UInt32(336), UInt32(338), UInt32(338), UInt32(340), UInt32(340), UInt32(342), UInt32(342), UInt32(344), UInt32(344), UInt32(346), UInt32(346), UInt32(348), UInt32(348),
    UInt32(350), UInt32(350), UInt32(352), UInt32(352), UInt32(354), UInt32(354), UInt32(356), UInt32(356), UInt32(358), UInt32(358), UInt32(360), UInt32(360), UInt32(362), UInt32(362),
    UInt32(364), UInt32(364), UInt32(366), UInt32(366), UInt32(368), UInt32(368), UInt32(370), UInt32(370), UInt32(372), UInt32(372), UInt32(374), UInt32(374), UInt32(376), UInt32(377),
    UInt32(379), UInt32(379), UInt32(381), UInt32(381), UInt32(385), UInt32(386), UInt32(388), UInt32(388), UInt32(390), UInt32(391), UInt32(393), UInt32(395), UInt32(398), UInt32(401),
    UInt32(403), UInt32(404), UInt32(406), UInt32(408), UInt32(412), UInt32(413), UInt32(415), UInt32(416), UInt32(418), UInt32(418), UInt32(420), UInt32(420), UInt32(422), UInt32(423),
    UInt32(425), UInt32(425), UInt32(428), UInt32(428), UInt32(430), UInt32(431), UInt32(433), UInt32(435), UInt32(437), UInt32(437), UInt32(439), UInt32(440), UInt32(444), UInt32(444),
    UInt32(452), UInt32(452), UInt32(455), UInt32(455), UInt32(458), UInt32(458), UInt32(461), UInt32(461), UInt32(463), UInt32(463), UInt32(465), UInt32(465), UInt32(467), UInt32(467),
    UInt32(469), UInt32(469), UInt32(471), UInt32(471), UInt32(473), UInt32(473), UInt32(475), UInt32(475), UInt32(478), UInt32(478), UInt32(480), UInt32(480), UInt32(482), UInt32(482),
    UInt32(484), UInt32(484), UInt32(486), UInt32(486), UInt32(488), UInt32(488), UInt32(490), UInt32(490), UInt32(492), UInt32(492), UInt32(494), UInt32(494), UInt32(497), UInt32(497),
    UInt32(500), UInt32(500), UInt32(502), UInt32(504), UInt32(506), UInt32(506), UInt32(508), UInt32(508), UInt32(510), UInt32(510), UInt32(512), UInt32(512), UInt32(514), UInt32(514),
    UInt32(516), UInt32(516), UInt32(518), UInt32(518), UInt32(520), UInt32(520), UInt32(522), UInt32(522), UInt32(524), UInt32(524), UInt32(526), UInt32(526), UInt32(528), UInt32(528),
    UInt32(530), UInt32(530), UInt32(532), UInt32(532), UInt32(534), UInt32(534), UInt32(536), UInt32(536), UInt32(538), UInt32(538), UInt32(540), UInt32(540), UInt32(542), UInt32(542),
    UInt32(544), UInt32(544), UInt32(546), UInt32(546), UInt32(548), UInt32(548), UInt32(550), UInt32(550), UInt32(552), UInt32(552), UInt32(554), UInt32(554), UInt32(556), UInt32(556),
    UInt32(558), UInt32(558), UInt32(560), UInt32(560), UInt32(562), UInt32(562), UInt32(570), UInt32(571), UInt32(573), UInt32(574), UInt32(577), UInt32(577), UInt32(579), UInt32(582),
    UInt32(584), UInt32(584), UInt32(586), UInt32(586), UInt32(588), UInt32(588), UInt32(590), UInt32(590), UInt32(880), UInt32(880), UInt32(882), UInt32(882), UInt32(886), UInt32(886),
    UInt32(895), UInt32(895), UInt32(902), UInt32(902), UInt32(904), UInt32(906), UInt32(908), UInt32(908), UInt32(910), UInt32(911), UInt32(913), UInt32(929), UInt32(931), UInt32(939),
    UInt32(975), UInt32(975), UInt32(978), UInt32(980), UInt32(984), UInt32(984), UInt32(986), UInt32(986), UInt32(988), UInt32(988), UInt32(990), UInt32(990), UInt32(992), UInt32(992),
    UInt32(994), UInt32(994), UInt32(996), UInt32(996), UInt32(998), UInt32(998), UInt32(1000), UInt32(1000), UInt32(1002), UInt32(1002), UInt32(1004), UInt32(1004), UInt32(1006), UInt32(1006),
    UInt32(1012), UInt32(1012), UInt32(1015), UInt32(1015), UInt32(1017), UInt32(1018), UInt32(1021), UInt32(1071), UInt32(1120), UInt32(1120), UInt32(1122), UInt32(1122), UInt32(1124), UInt32(1124),
    UInt32(1126), UInt32(1126), UInt32(1128), UInt32(1128), UInt32(1130), UInt32(1130), UInt32(1132), UInt32(1132), UInt32(1134), UInt32(1134), UInt32(1136), UInt32(1136), UInt32(1138), UInt32(1138),
    UInt32(1140), UInt32(1140), UInt32(1142), UInt32(1142), UInt32(1144), UInt32(1144), UInt32(1146), UInt32(1146), UInt32(1148), UInt32(1148), UInt32(1150), UInt32(1150), UInt32(1152), UInt32(1152),
    UInt32(1162), UInt32(1162), UInt32(1164), UInt32(1164), UInt32(1166), UInt32(1166), UInt32(1168), UInt32(1168), UInt32(1170), UInt32(1170), UInt32(1172), UInt32(1172), UInt32(1174), UInt32(1174),
    UInt32(1176), UInt32(1176), UInt32(1178), UInt32(1178), UInt32(1180), UInt32(1180), UInt32(1182), UInt32(1182), UInt32(1184), UInt32(1184), UInt32(1186), UInt32(1186), UInt32(1188), UInt32(1188),
    UInt32(1190), UInt32(1190), UInt32(1192), UInt32(1192), UInt32(1194), UInt32(1194), UInt32(1196), UInt32(1196), UInt32(1198), UInt32(1198), UInt32(1200), UInt32(1200), UInt32(1202), UInt32(1202),
    UInt32(1204), UInt32(1204), UInt32(1206), UInt32(1206), UInt32(1208), UInt32(1208), UInt32(1210), UInt32(1210), UInt32(1212), UInt32(1212), UInt32(1214), UInt32(1214), UInt32(1216), UInt32(1217),
    UInt32(1219), UInt32(1219), UInt32(1221), UInt32(1221), UInt32(1223), UInt32(1223), UInt32(1225), UInt32(1225), UInt32(1227), UInt32(1227), UInt32(1229), UInt32(1229), UInt32(1232), UInt32(1232),
    UInt32(1234), UInt32(1234), UInt32(1236), UInt32(1236), UInt32(1238), UInt32(1238), UInt32(1240), UInt32(1240), UInt32(1242), UInt32(1242), UInt32(1244), UInt32(1244), UInt32(1246), UInt32(1246),
    UInt32(1248), UInt32(1248), UInt32(1250), UInt32(1250), UInt32(1252), UInt32(1252), UInt32(1254), UInt32(1254), UInt32(1256), UInt32(1256), UInt32(1258), UInt32(1258), UInt32(1260), UInt32(1260),
    UInt32(1262), UInt32(1262), UInt32(1264), UInt32(1264), UInt32(1266), UInt32(1266), UInt32(1268), UInt32(1268), UInt32(1270), UInt32(1270), UInt32(1272), UInt32(1272), UInt32(1274), UInt32(1274),
    UInt32(1276), UInt32(1276), UInt32(1278), UInt32(1278), UInt32(1280), UInt32(1280), UInt32(1282), UInt32(1282), UInt32(1284), UInt32(1284), UInt32(1286), UInt32(1286), UInt32(1288), UInt32(1288),
    UInt32(1290), UInt32(1290), UInt32(1292), UInt32(1292), UInt32(1294), UInt32(1294), UInt32(1296), UInt32(1296), UInt32(1298), UInt32(1298), UInt32(1300), UInt32(1300), UInt32(1302), UInt32(1302),
    UInt32(1304), UInt32(1304), UInt32(1306), UInt32(1306), UInt32(1308), UInt32(1308), UInt32(1310), UInt32(1310), UInt32(1312), UInt32(1312), UInt32(1314), UInt32(1314), UInt32(1316), UInt32(1316),
    UInt32(1318), UInt32(1318), UInt32(1320), UInt32(1320), UInt32(1322), UInt32(1322), UInt32(1324), UInt32(1324), UInt32(1326), UInt32(1326), UInt32(1329), UInt32(1366), UInt32(4256), UInt32(4293),
    UInt32(4295), UInt32(4295), UInt32(4301), UInt32(4301), UInt32(5024), UInt32(5109), UInt32(7312), UInt32(7354), UInt32(7357), UInt32(7359), UInt32(7680), UInt32(7680), UInt32(7682), UInt32(7682),
    UInt32(7684), UInt32(7684), UInt32(7686), UInt32(7686), UInt32(7688), UInt32(7688), UInt32(7690), UInt32(7690), UInt32(7692), UInt32(7692), UInt32(7694), UInt32(7694), UInt32(7696), UInt32(7696),
    UInt32(7698), UInt32(7698), UInt32(7700), UInt32(7700), UInt32(7702), UInt32(7702), UInt32(7704), UInt32(7704), UInt32(7706), UInt32(7706), UInt32(7708), UInt32(7708), UInt32(7710), UInt32(7710),
    UInt32(7712), UInt32(7712), UInt32(7714), UInt32(7714), UInt32(7716), UInt32(7716), UInt32(7718), UInt32(7718), UInt32(7720), UInt32(7720), UInt32(7722), UInt32(7722), UInt32(7724), UInt32(7724),
    UInt32(7726), UInt32(7726), UInt32(7728), UInt32(7728), UInt32(7730), UInt32(7730), UInt32(7732), UInt32(7732), UInt32(7734), UInt32(7734), UInt32(7736), UInt32(7736), UInt32(7738), UInt32(7738),
    UInt32(7740), UInt32(7740), UInt32(7742), UInt32(7742), UInt32(7744), UInt32(7744), UInt32(7746), UInt32(7746), UInt32(7748), UInt32(7748), UInt32(7750), UInt32(7750), UInt32(7752), UInt32(7752),
    UInt32(7754), UInt32(7754), UInt32(7756), UInt32(7756), UInt32(7758), UInt32(7758), UInt32(7760), UInt32(7760), UInt32(7762), UInt32(7762), UInt32(7764), UInt32(7764), UInt32(7766), UInt32(7766),
    UInt32(7768), UInt32(7768), UInt32(7770), UInt32(7770), UInt32(7772), UInt32(7772), UInt32(7774), UInt32(7774), UInt32(7776), UInt32(7776), UInt32(7778), UInt32(7778), UInt32(7780), UInt32(7780),
    UInt32(7782), UInt32(7782), UInt32(7784), UInt32(7784), UInt32(7786), UInt32(7786), UInt32(7788), UInt32(7788), UInt32(7790), UInt32(7790), UInt32(7792), UInt32(7792), UInt32(7794), UInt32(7794),
    UInt32(7796), UInt32(7796), UInt32(7798), UInt32(7798), UInt32(7800), UInt32(7800), UInt32(7802), UInt32(7802), UInt32(7804), UInt32(7804), UInt32(7806), UInt32(7806), UInt32(7808), UInt32(7808),
    UInt32(7810), UInt32(7810), UInt32(7812), UInt32(7812), UInt32(7814), UInt32(7814), UInt32(7816), UInt32(7816), UInt32(7818), UInt32(7818), UInt32(7820), UInt32(7820), UInt32(7822), UInt32(7822),
    UInt32(7824), UInt32(7824), UInt32(7826), UInt32(7826), UInt32(7828), UInt32(7828), UInt32(7838), UInt32(7838), UInt32(7840), UInt32(7840), UInt32(7842), UInt32(7842), UInt32(7844), UInt32(7844),
    UInt32(7846), UInt32(7846), UInt32(7848), UInt32(7848), UInt32(7850), UInt32(7850), UInt32(7852), UInt32(7852), UInt32(7854), UInt32(7854), UInt32(7856), UInt32(7856), UInt32(7858), UInt32(7858),
    UInt32(7860), UInt32(7860), UInt32(7862), UInt32(7862), UInt32(7864), UInt32(7864), UInt32(7866), UInt32(7866), UInt32(7868), UInt32(7868), UInt32(7870), UInt32(7870), UInt32(7872), UInt32(7872),
    UInt32(7874), UInt32(7874), UInt32(7876), UInt32(7876), UInt32(7878), UInt32(7878), UInt32(7880), UInt32(7880), UInt32(7882), UInt32(7882), UInt32(7884), UInt32(7884), UInt32(7886), UInt32(7886),
    UInt32(7888), UInt32(7888), UInt32(7890), UInt32(7890), UInt32(7892), UInt32(7892), UInt32(7894), UInt32(7894), UInt32(7896), UInt32(7896), UInt32(7898), UInt32(7898), UInt32(7900), UInt32(7900),
    UInt32(7902), UInt32(7902), UInt32(7904), UInt32(7904), UInt32(7906), UInt32(7906), UInt32(7908), UInt32(7908), UInt32(7910), UInt32(7910), UInt32(7912), UInt32(7912), UInt32(7914), UInt32(7914),
    UInt32(7916), UInt32(7916), UInt32(7918), UInt32(7918), UInt32(7920), UInt32(7920), UInt32(7922), UInt32(7922), UInt32(7924), UInt32(7924), UInt32(7926), UInt32(7926), UInt32(7928), UInt32(7928),
    UInt32(7930), UInt32(7930), UInt32(7932), UInt32(7932), UInt32(7934), UInt32(7934), UInt32(7944), UInt32(7951), UInt32(7960), UInt32(7965), UInt32(7976), UInt32(7983), UInt32(7992), UInt32(7999),
    UInt32(8008), UInt32(8013), UInt32(8025), UInt32(8025), UInt32(8027), UInt32(8027), UInt32(8029), UInt32(8029), UInt32(8031), UInt32(8031), UInt32(8040), UInt32(8047), UInt32(8120), UInt32(8123),
    UInt32(8136), UInt32(8139), UInt32(8152), UInt32(8155), UInt32(8168), UInt32(8172), UInt32(8184), UInt32(8187), UInt32(8450), UInt32(8450), UInt32(8455), UInt32(8455), UInt32(8459), UInt32(8461),
    UInt32(8464), UInt32(8466), UInt32(8469), UInt32(8469), UInt32(8473), UInt32(8477), UInt32(8484), UInt32(8484), UInt32(8486), UInt32(8486), UInt32(8488), UInt32(8488), UInt32(8490), UInt32(8493),
    UInt32(8496), UInt32(8499), UInt32(8510), UInt32(8511), UInt32(8517), UInt32(8517), UInt32(8544), UInt32(8559), UInt32(8579), UInt32(8579), UInt32(9398), UInt32(9423), UInt32(11264), UInt32(11311),
    UInt32(11360), UInt32(11360), UInt32(11362), UInt32(11364), UInt32(11367), UInt32(11367), UInt32(11369), UInt32(11369), UInt32(11371), UInt32(11371), UInt32(11373), UInt32(11376), UInt32(11378), UInt32(11378),
    UInt32(11381), UInt32(11381), UInt32(11390), UInt32(11392), UInt32(11394), UInt32(11394), UInt32(11396), UInt32(11396), UInt32(11398), UInt32(11398), UInt32(11400), UInt32(11400), UInt32(11402), UInt32(11402),
    UInt32(11404), UInt32(11404), UInt32(11406), UInt32(11406), UInt32(11408), UInt32(11408), UInt32(11410), UInt32(11410), UInt32(11412), UInt32(11412), UInt32(11414), UInt32(11414), UInt32(11416), UInt32(11416),
    UInt32(11418), UInt32(11418), UInt32(11420), UInt32(11420), UInt32(11422), UInt32(11422), UInt32(11424), UInt32(11424), UInt32(11426), UInt32(11426), UInt32(11428), UInt32(11428), UInt32(11430), UInt32(11430),
    UInt32(11432), UInt32(11432), UInt32(11434), UInt32(11434), UInt32(11436), UInt32(11436), UInt32(11438), UInt32(11438), UInt32(11440), UInt32(11440), UInt32(11442), UInt32(11442), UInt32(11444), UInt32(11444),
    UInt32(11446), UInt32(11446), UInt32(11448), UInt32(11448), UInt32(11450), UInt32(11450), UInt32(11452), UInt32(11452), UInt32(11454), UInt32(11454), UInt32(11456), UInt32(11456), UInt32(11458), UInt32(11458),
    UInt32(11460), UInt32(11460), UInt32(11462), UInt32(11462), UInt32(11464), UInt32(11464), UInt32(11466), UInt32(11466), UInt32(11468), UInt32(11468), UInt32(11470), UInt32(11470), UInt32(11472), UInt32(11472),
    UInt32(11474), UInt32(11474), UInt32(11476), UInt32(11476), UInt32(11478), UInt32(11478), UInt32(11480), UInt32(11480), UInt32(11482), UInt32(11482), UInt32(11484), UInt32(11484), UInt32(11486), UInt32(11486),
    UInt32(11488), UInt32(11488), UInt32(11490), UInt32(11490), UInt32(11499), UInt32(11499), UInt32(11501), UInt32(11501), UInt32(11506), UInt32(11506), UInt32(42560), UInt32(42560), UInt32(42562), UInt32(42562),
    UInt32(42564), UInt32(42564), UInt32(42566), UInt32(42566), UInt32(42568), UInt32(42568), UInt32(42570), UInt32(42570), UInt32(42572), UInt32(42572), UInt32(42574), UInt32(42574), UInt32(42576), UInt32(42576),
    UInt32(42578), UInt32(42578), UInt32(42580), UInt32(42580), UInt32(42582), UInt32(42582), UInt32(42584), UInt32(42584), UInt32(42586), UInt32(42586), UInt32(42588), UInt32(42588), UInt32(42590), UInt32(42590),
    UInt32(42592), UInt32(42592), UInt32(42594), UInt32(42594), UInt32(42596), UInt32(42596), UInt32(42598), UInt32(42598), UInt32(42600), UInt32(42600), UInt32(42602), UInt32(42602), UInt32(42604), UInt32(42604),
    UInt32(42624), UInt32(42624), UInt32(42626), UInt32(42626), UInt32(42628), UInt32(42628), UInt32(42630), UInt32(42630), UInt32(42632), UInt32(42632), UInt32(42634), UInt32(42634), UInt32(42636), UInt32(42636),
    UInt32(42638), UInt32(42638), UInt32(42640), UInt32(42640), UInt32(42642), UInt32(42642), UInt32(42644), UInt32(42644), UInt32(42646), UInt32(42646), UInt32(42648), UInt32(42648), UInt32(42650), UInt32(42650),
    UInt32(42786), UInt32(42786), UInt32(42788), UInt32(42788), UInt32(42790), UInt32(42790), UInt32(42792), UInt32(42792), UInt32(42794), UInt32(42794), UInt32(42796), UInt32(42796), UInt32(42798), UInt32(42798),
    UInt32(42802), UInt32(42802), UInt32(42804), UInt32(42804), UInt32(42806), UInt32(42806), UInt32(42808), UInt32(42808), UInt32(42810), UInt32(42810), UInt32(42812), UInt32(42812), UInt32(42814), UInt32(42814),
    UInt32(42816), UInt32(42816), UInt32(42818), UInt32(42818), UInt32(42820), UInt32(42820), UInt32(42822), UInt32(42822), UInt32(42824), UInt32(42824), UInt32(42826), UInt32(42826), UInt32(42828), UInt32(42828),
    UInt32(42830), UInt32(42830), UInt32(42832), UInt32(42832), UInt32(42834), UInt32(42834), UInt32(42836), UInt32(42836), UInt32(42838), UInt32(42838), UInt32(42840), UInt32(42840), UInt32(42842), UInt32(42842),
    UInt32(42844), UInt32(42844), UInt32(42846), UInt32(42846), UInt32(42848), UInt32(42848), UInt32(42850), UInt32(42850), UInt32(42852), UInt32(42852), UInt32(42854), UInt32(42854), UInt32(42856), UInt32(42856),
    UInt32(42858), UInt32(42858), UInt32(42860), UInt32(42860), UInt32(42862), UInt32(42862), UInt32(42873), UInt32(42873), UInt32(42875), UInt32(42875), UInt32(42877), UInt32(42878), UInt32(42880), UInt32(42880),
    UInt32(42882), UInt32(42882), UInt32(42884), UInt32(42884), UInt32(42886), UInt32(42886), UInt32(42891), UInt32(42891), UInt32(42893), UInt32(42893), UInt32(42896), UInt32(42896), UInt32(42898), UInt32(42898),
    UInt32(42902), UInt32(42902), UInt32(42904), UInt32(42904), UInt32(42906), UInt32(42906), UInt32(42908), UInt32(42908), UInt32(42910), UInt32(42910), UInt32(42912), UInt32(42912), UInt32(42914), UInt32(42914),
    UInt32(42916), UInt32(42916), UInt32(42918), UInt32(42918), UInt32(42920), UInt32(42920), UInt32(42922), UInt32(42926), UInt32(42928), UInt32(42932), UInt32(42934), UInt32(42934), UInt32(42936), UInt32(42936),
    UInt32(42938), UInt32(42938), UInt32(42940), UInt32(42940), UInt32(42942), UInt32(42942), UInt32(42944), UInt32(42944), UInt32(42946), UInt32(42946), UInt32(42948), UInt32(42951), UInt32(42953), UInt32(42953),
    UInt32(42960), UInt32(42960), UInt32(42966), UInt32(42966), UInt32(42968), UInt32(42968), UInt32(42997), UInt32(42997), UInt32(65313), UInt32(65338), UInt32(66560), UInt32(66599), UInt32(66736), UInt32(66771),
    UInt32(66928), UInt32(66938), UInt32(66940), UInt32(66954), UInt32(66956), UInt32(66962), UInt32(66964), UInt32(66965), UInt32(68736), UInt32(68786), UInt32(71840), UInt32(71871), UInt32(93760), UInt32(93791),
    UInt32(119808), UInt32(119833), UInt32(119860), UInt32(119885), UInt32(119912), UInt32(119937), UInt32(119964), UInt32(119964), UInt32(119966), UInt32(119967), UInt32(119970), UInt32(119970), UInt32(119973), UInt32(119974),
    UInt32(119977), UInt32(119980), UInt32(119982), UInt32(119989), UInt32(120016), UInt32(120041), UInt32(120068), UInt32(120069), UInt32(120071), UInt32(120074), UInt32(120077), UInt32(120084), UInt32(120086), UInt32(120092),
    UInt32(120120), UInt32(120121), UInt32(120123), UInt32(120126), UInt32(120128), UInt32(120132), UInt32(120134), UInt32(120134), UInt32(120138), UInt32(120144), UInt32(120172), UInt32(120197), UInt32(120224), UInt32(120249),
    UInt32(120276), UInt32(120301), UInt32(120328), UInt32(120353), UInt32(120380), UInt32(120405), UInt32(120432), UInt32(120457), UInt32(120488), UInt32(120512), UInt32(120546), UInt32(120570), UInt32(120604), UInt32(120628),
    UInt32(120662), UInt32(120686), UInt32(120720), UInt32(120744), UInt32(120778), UInt32(120778), UInt32(125184), UInt32(125217), UInt32(127280), UInt32(127305), UInt32(127312), UInt32(127337), UInt32(127344), UInt32(127369),
]

comptime _NOT_UPPER_RANGES = [
    UInt32(65), UInt32(90), UInt32(97), UInt32(122), UInt32(170), UInt32(170), UInt32(181), UInt32(181), UInt32(186), UInt32(186), UInt32(192), UInt32(214), UInt32(216), UInt32(246),
    UInt32(248), UInt32(442), UInt32(444), UInt32(447), UInt32(452), UInt32(659), UInt32(661), UInt32(696), UInt32(704), UInt32(705), UInt32(736), UInt32(740), UInt32(837), UInt32(837),
    UInt32(880), UInt32(883), UInt32(886), UInt32(887), UInt32(890), UInt32(893), UInt32(895), UInt32(895), UInt32(902), UInt32(902), UInt32(904), UInt32(906), UInt32(908), UInt32(908),
    UInt32(910), UInt32(929), UInt32(931), UInt32(1013), UInt32(1015), UInt32(1153), UInt32(1162), UInt32(1327), UInt32(1329), UInt32(1366), UInt32(1376), UInt32(1416), UInt32(4256), UInt32(4293),
    UInt32(4295), UInt32(4295), UInt32(4301), UInt32(4301), UInt32(4304), UInt32(4346), UInt32(4348), UInt32(4351), UInt32(5024), UInt32(5109), UInt32(5112), UInt32(5117), UInt32(7296), UInt32(7304),
    UInt32(7312), UInt32(7354), UInt32(7357), UInt32(7359), UInt32(7424), UInt32(7615), UInt32(7680), UInt32(7957), UInt32(7960), UInt32(7965), UInt32(7968), UInt32(8005), UInt32(8008), UInt32(8013),
    UInt32(8016), UInt32(8023), UInt32(8025), UInt32(8025), UInt32(8027), UInt32(8027), UInt32(8029), UInt32(8029), UInt32(8031), UInt32(8061), UInt32(8064), UInt32(8116), UInt32(8118), UInt32(8124),
    UInt32(8126), UInt32(8126), UInt32(8130), UInt32(8132), UInt32(8134), UInt32(8140), UInt32(8144), UInt32(8147), UInt32(8150), UInt32(8155), UInt32(8160), UInt32(8172), UInt32(8178), UInt32(8180),
    UInt32(8182), UInt32(8188), UInt32(8305), UInt32(8305), UInt32(8319), UInt32(8319), UInt32(8336), UInt32(8348), UInt32(8450), UInt32(8450), UInt32(8455), UInt32(8455), UInt32(8458), UInt32(8467),
    UInt32(8469), UInt32(8469), UInt32(8473), UInt32(8477), UInt32(8484), UInt32(8484), UInt32(8486), UInt32(8486), UInt32(8488), UInt32(8488), UInt32(8490), UInt32(8493), UInt32(8495), UInt32(8500),
    UInt32(8505), UInt32(8505), UInt32(8508), UInt32(8511), UInt32(8517), UInt32(8521), UInt32(8526), UInt32(8526), UInt32(8544), UInt32(8575), UInt32(8579), UInt32(8580), UInt32(9398), UInt32(9449),
    UInt32(11264), UInt32(11492), UInt32(11499), UInt32(11502), UInt32(11506), UInt32(11507), UInt32(11520), UInt32(11557), UInt32(11559), UInt32(11559), UInt32(11565), UInt32(11565), UInt32(42560), UInt32(42605),
    UInt32(42624), UInt32(42653), UInt32(42786), UInt32(42887), UInt32(42891), UInt32(42894), UInt32(42896), UInt32(42954), UInt32(42960), UInt32(42961), UInt32(42963), UInt32(42963), UInt32(42965), UInt32(42969),
    UInt32(42994), UInt32(42998), UInt32(43000), UInt32(43002), UInt32(43824), UInt32(43866), UInt32(43868), UInt32(43881), UInt32(43888), UInt32(43967), UInt32(64256), UInt32(64262), UInt32(64275), UInt32(64279),
    UInt32(65313), UInt32(65338), UInt32(65345), UInt32(65370), UInt32(66560), UInt32(66639), UInt32(66736), UInt32(66771), UInt32(66776), UInt32(66811), UInt32(66928), UInt32(66938), UInt32(66940), UInt32(66954),
    UInt32(66956), UInt32(66962), UInt32(66964), UInt32(66965), UInt32(66967), UInt32(66977), UInt32(66979), UInt32(66993), UInt32(66995), UInt32(67001), UInt32(67003), UInt32(67004), UInt32(67456), UInt32(67456),
    UInt32(67459), UInt32(67461), UInt32(67463), UInt32(67504), UInt32(67506), UInt32(67514), UInt32(68736), UInt32(68786), UInt32(68800), UInt32(68850), UInt32(71840), UInt32(71903), UInt32(93760), UInt32(93823),
    UInt32(119808), UInt32(119892), UInt32(119894), UInt32(119964), UInt32(119966), UInt32(119967), UInt32(119970), UInt32(119970), UInt32(119973), UInt32(119974), UInt32(119977), UInt32(119980), UInt32(119982), UInt32(119993),
    UInt32(119995), UInt32(119995), UInt32(119997), UInt32(120003), UInt32(120005), UInt32(120069), UInt32(120071), UInt32(120074), UInt32(120077), UInt32(120084), UInt32(120086), UInt32(120092), UInt32(120094), UInt32(120121),
    UInt32(120123), UInt32(120126), UInt32(120128), UInt32(120132), UInt32(120134), UInt32(120134), UInt32(120138), UInt32(120144), UInt32(120146), UInt32(120485), UInt32(120488), UInt32(120512), UInt32(120514), UInt32(120538),
    UInt32(120540), UInt32(120570), UInt32(120572), UInt32(120596), UInt32(120598), UInt32(120628), UInt32(120630), UInt32(120654), UInt32(120656), UInt32(120686), UInt32(120688), UInt32(120712), UInt32(120714), UInt32(120744),
    UInt32(120746), UInt32(120770), UInt32(120772), UInt32(120779), UInt32(122624), UInt32(122633), UInt32(122635), UInt32(122654), UInt32(122661), UInt32(122666), UInt32(122928), UInt32(122989), UInt32(125184), UInt32(125251),
    UInt32(127280), UInt32(127305), UInt32(127312), UInt32(127337), UInt32(127344), UInt32(127369),
]
