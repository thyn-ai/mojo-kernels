"""Clean-room language-detection scoring kernel (n-gram naive Bayes).

Written fresh from the published algorithm description of Nakatani Shuyo's
language-detection library (as packaged on PyPI as `langdetect`): text is
preprocessed (URL/e-mail masking, Vietnamese normalization, space collapsing,
non-Latin cleaning), 1..3-grams are extracted with script-based
normalization, and per-language probabilities are estimated by the library's
seeded sampling procedure (7 trials of multiplicative updates with smoothing
alpha = 0.5 + gauss()*0.05, base frequency 10000, convergence at 0.99999).

No third-party Mojo code is used or adapted; the language profiles and data
tables are not compiled in either — they are loaded at runtime by the Python
wrapper from the installed `langdetect` package and handed over as a binary
blob (see python/langdetect_mojo/langdetect_mojo/_data.py for the format).

Determinism: the sampler reproduces CPython's `random.Random(seed)` bit-for-
bit (MT19937 with init_by_array seeding, res53 `random()`, getrandbits-based
`_randbelow`, and the Box-Muller `gauss` with its second-value cache), so for
a given seed the emitted probability vector is bit-identical to the seeded
reference on platforms where libm cos/sin/log agree with CPython's (observed
bit-identical on macOS and Linux, since both call the same system libm).

Exported C ABI (v1):

    int32_t  langdetectmojo_abi_version(void)
    void*    langdetectmojo_profiles_create(const uint8_t* blob, int64_t blob_len)
    int32_t  langdetectmojo_detect(void* handle, const uint8_t* text,
                                   int64_t text_len, uint64_t seed,
                                   double* out_probs, int64_t out_cap,
                                   uint32_t flags)
    void     langdetectmojo_profiles_destroy(void* handle)

langdetectmojo_detect returns 0 on success (out_probs filled with n_langs
values), 1 when the text has no usable features (the reference raises its
'No features in text.' error), 2 on invalid arguments. All arithmetic is
IEEE-754 float64 in the reference's operation order. `flags` bit 0 selects
the summation algorithm used when normalizing probabilities: 0 = CPython
3.12+ Neumaier-compensated sum(), 1 = the pre-3.12 left-to-right sum(); the
wrapper sets it from the interpreter version so results match the reference
bit-for-bit on either.
"""

from std.bit import count_leading_zeros
from std.ffi import external_call
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Algorithm constants (reference Detector defaults).
comptime N_TRIAL: Int64 = 7
comptime ALPHA_DEFAULT: Float64 = 0.5
comptime ALPHA_WIDTH: Float64 = 0.05
comptime ITERATION_LIMIT: Int64 = 1000
comptime CONV_THRESHOLD: Float64 = 0.99999
comptime BASE_FREQ: Float64 = 10000.0
comptime MAX_TEXT_LENGTH: Int64 = 10000
comptime SP: UInt32 = 0x20  # ' '

# detect() flags
comptime FLAG_NAIVE_SUM: UInt32 = 1  # pre-3.12 CPython sum() semantics

# Blob header layout (all little-endian u64 words).
comptime BLOB_MAGIC: UInt64 = 0x4C444D4F4A4F3031  # "LDMOJO01"
comptime BLOB_VERSION: UInt64 = 1
comptime HDR_WORDS: Int64 = 20
# word indices
comptime H_N_LANGS: Int64 = 2
comptime H_N_KEYS: Int64 = 3
comptime H_N_ENTRIES: Int64 = 4
comptime H_OFF_LANGS: Int64 = 5
comptime H_OFF_KEYS: Int64 = 6
comptime H_OFF_KIDOFF: Int64 = 7
comptime H_OFF_ENT: Int64 = 8
comptime H_OFF_CJK: Int64 = 9
comptime H_N_CJK: Int64 = 10
comptime H_OFF_LATIN1: Int64 = 11
comptime H_N_LATIN1: Int64 = 12
comptime H_OFF_VI_ALPHA: Int64 = 13
comptime H_N_VI_ALPHA: Int64 = 14
comptime H_OFF_VI_DMARK: Int64 = 15
comptime H_N_VI_DMARK: Int64 = 16
comptime H_OFF_VI_ROWS: Int64 = 17
comptime H_OFF_UPPER: Int64 = 18
comptime H_N_UPPER: Int64 = 19

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


# --------------------------------------------------------------------------
# CPython-compatible MT19937 (random.Random for integer seeds)
# --------------------------------------------------------------------------


struct CRandom:
    """Bit-exact re-implementation of CPython's random.Random(seed) for
    non-negative integer seeds < 2**64 (init_by_array with the little-endian
    u32 key of the seed), covering random(), getrandbits/_randbelow and
    gauss(0.0, 1.0) with the gauss_next cache.
    """

    var mt: List[UInt32]
    var idx: Int
    var gauss_next: Float64
    var has_gauss: Bool

    def __init__(out self, seed: UInt64):
        self.mt = List[UInt32](capacity=624)
        for _ in range(624):
            self.mt.append(0)
        self.idx = 624
        self.gauss_next = 0.0
        self.has_gauss = False
        var key = List[UInt32]()
        var v = seed
        while True:
            key.append(UInt32(v & 0xFFFFFFFF))
            v >>= 32
            if v == 0:
                break
        self._init_by_array(key)

    def _init_genrand(mut self, s: UInt32):
        self.mt[0] = s
        for i in range(1, 624):
            var prev = UInt64(self.mt[i - 1])
            var x = (1812433253 * (prev ^ (prev >> 30)) + UInt64(i)) & 0xFFFFFFFF
            self.mt[i] = UInt32(x)

    def _init_by_array(mut self, key: List[UInt32]):
        self._init_genrand(19650218)
        var i = 1
        var j = 0
        var k = 624
        if len(key) > k:
            k = len(key)
        while k > 0:
            var prev = UInt64(self.mt[i - 1])
            var mixed = (prev ^ (prev >> 30)) * 1664525
            var x = (
                UInt64(self.mt[i]) ^ (mixed & 0xFFFFFFFF)
            ) + UInt64(key[j]) + UInt64(j)
            self.mt[i] = UInt32(x & 0xFFFFFFFF)
            i += 1
            j += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
            if j >= len(key):
                j = 0
            k -= 1
        for _ in range(623):
            var prev = UInt64(self.mt[i - 1])
            var mixed = (prev ^ (prev >> 30)) * 1566083941
            var x = (UInt64(self.mt[i]) ^ (mixed & 0xFFFFFFFF)) - UInt64(i)
            self.mt[i] = UInt32(x & 0xFFFFFFFF)
            i += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
        self.mt[0] = 0x80000000

    def _twist(mut self):
        for kk in range(624):
            var y = (self.mt[kk] & 0x80000000) | (
                self.mt[(kk + 1) % 624] & 0x7FFFFFFF
            )
            var nxt = self.mt[(kk + 397) % 624] ^ (y >> 1)
            if (y & 1) != 0:
                nxt ^= 0x9908B0DF
            self.mt[kk] = nxt
        self.idx = 0

    def next_u32(mut self) -> UInt32:
        if self.idx >= 624:
            self._twist()
        var y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def getrandbits(mut self, k: Int) -> UInt64:
        # CPython fills little-endian u32 words and shifts the top word right.
        if k <= 32:
            return UInt64(self.next_u32() >> UInt32(32 - k))
        var lo = UInt64(self.next_u32())
        var hi = UInt64(self.next_u32() >> UInt32(64 - k))
        return lo | (hi << 32)

    def random(mut self) -> Float64:
        var a = self.next_u32() >> 5
        var b = self.next_u32() >> 6
        var hi = Float64(a) * 67108864.0
        return (hi + Float64(b)) * (1.0 / 9007199254740992.0)

    def randbelow(mut self, n: UInt64) -> UInt64:
        # CPython Random._randbelow: rejection sampling on getrandbits.
        var k = 1
        if n > 0:
            k = 64 - Int(count_leading_zeros(n))
        var r = self.getrandbits(k)
        while r >= n:
            r = self.getrandbits(k)
        return r

    def gauss(mut self) -> Float64:
        # CPython random.gauss(0.0, 1.0): Box-Muller with the cached second
        # deviate; cos/sin/log/sqrt resolve to the same system libm CPython
        # calls, keeping the stream bit-identical in practice.
        if self.has_gauss:
            self.has_gauss = False
            return self.gauss_next
        var x2pi = self.random() * 6.283185307179586
        var g2rad = external_call["sqrt", Float64](
            -2.0 * external_call["log", Float64](1.0 - self.random())
        )
        var z = external_call["cos", Float64](x2pi) * g2rad
        self.gauss_next = external_call["sin", Float64](x2pi) * g2rad
        self.has_gauss = True
        return z


# --------------------------------------------------------------------------
# Native profile store (parsed from the wrapper's blob; all data copied)
# --------------------------------------------------------------------------


struct Profiles(Copyable, Movable):
    var n_langs: Int64
    var n_keys: Int64
    var n_entries: Int64
    var keys: U64Ptr  # [n_keys] packed n-grams, strictly ascending
    var kid_off: U64Ptr  # [n_keys + 1] CSR entry offsets
    var ent_lang: U32Ptr  # [n_entries]
    var ent_prob: F64Ptr  # [n_entries]
    var n_cjk: Int64
    var cjk_from: U32Ptr  # [n_cjk] sorted
    var cjk_to: U32Ptr  # [n_cjk]
    var n_latin1: Int64
    var latin1: U32Ptr  # [n_latin1] excluded Latin-1 supplement codepoints
    var n_via: Int64
    var vi_alpha: U32Ptr  # [n_via]
    var n_vid: Int64
    var vi_dmark: U32Ptr  # [n_vid]
    var vi_rows: U32Ptr  # [5 * n_via] normalized Vietnamese codepoints
    var n_upper: Int64
    var upper_lo: U32Ptr  # [n_upper] sorted range starts
    var upper_hi: U32Ptr  # [n_upper] range ends (inclusive)

    def __init__(
        out self,
        n_langs: Int64,
        n_keys: Int64,
        n_entries: Int64,
        keys: U64Ptr,
        kid_off: U64Ptr,
        ent_lang: U32Ptr,
        ent_prob: F64Ptr,
        n_cjk: Int64,
        cjk_from: U32Ptr,
        cjk_to: U32Ptr,
        n_latin1: Int64,
        latin1: U32Ptr,
        n_via: Int64,
        vi_alpha: U32Ptr,
        n_vid: Int64,
        vi_dmark: U32Ptr,
        vi_rows: U32Ptr,
        n_upper: Int64,
        upper_lo: U32Ptr,
        upper_hi: U32Ptr,
    ):
        self.n_langs = n_langs
        self.n_keys = n_keys
        self.n_entries = n_entries
        self.keys = keys
        self.kid_off = kid_off
        self.ent_lang = ent_lang
        self.ent_prob = ent_prob
        self.n_cjk = n_cjk
        self.cjk_from = cjk_from
        self.cjk_to = cjk_to
        self.n_latin1 = n_latin1
        self.latin1 = latin1
        self.n_via = n_via
        self.vi_alpha = vi_alpha
        self.n_vid = n_vid
        self.vi_dmark = vi_dmark
        self.vi_rows = vi_rows
        self.n_upper = n_upper
        self.upper_lo = upper_lo
        self.upper_hi = upper_hi

    def free(mut self):
        self.keys.unsafe_free()
        self.kid_off.unsafe_free()
        self.ent_lang.unsafe_free()
        self.ent_prob.unsafe_free()
        self.cjk_from.unsafe_free()
        self.cjk_to.unsafe_free()
        self.latin1.unsafe_free()
        self.vi_alpha.unsafe_free()
        self.vi_dmark.unsafe_free()
        self.vi_rows.unsafe_free()
        self.upper_lo.unsafe_free()
        self.upper_hi.unsafe_free()


def _find_key(p: Profiles, key: UInt64) -> Int:
    """Binary search the packed-key table; return the key id or -1."""
    var lo = Int(0)
    var hi = Int(p.n_keys) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        var k = p.keys[unsafe_offset=mid]
        if key < k:
            hi = mid - 1
        elif key > k:
            lo = mid + 1
        else:
            return mid
    return -1


def _is_upper(p: Profiles, cp: UInt32) -> Bool:
    var lo = Int(0)
    var hi = Int(p.n_upper) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if cp < p.upper_lo[unsafe_offset=mid]:
            hi = mid - 1
        elif cp > p.upper_hi[unsafe_offset=mid]:
            lo = mid + 1
        else:
            return True
    return False


def _cjk_map(p: Profiles, cp: UInt32) -> UInt32:
    var lo = Int(0)
    var hi = Int(p.n_cjk) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        var f = p.cjk_from[unsafe_offset=mid]
        if cp < f:
            hi = mid - 1
        elif cp > f:
            lo = mid + 1
        else:
            return p.cjk_to[unsafe_offset=mid]
    return cp


def _in_table(tab: U32Ptr, n: Int64, cp: UInt32) -> Int:
    """Linear membership scan; returns the index or -1 (tables are tiny)."""
    for i in range(Int(n)):
        if tab[unsafe_offset=i] == cp:
            return i
    return -1


def _normalize_cp(p: Profiles, cp: UInt32) -> UInt32:
    """Per-script normalization (reference NGram.normalize). Blocks not
    listed here pass through unchanged, exactly like the reference."""
    if cp <= 0x7F:  # Basic Latin: keep letters, blank everything else
        if cp < 0x41 or (cp > 0x5A and cp < 0x61) or cp > 0x7A:
            return SP
        return cp
    if cp <= 0xFF:  # Latin-1 Supplement: blank the excluded set
        if _in_table(p.latin1, p.n_latin1, cp) >= 0:
            return SP
        return cp
    if cp >= 0x180 and cp <= 0x24F:  # Latin Extended-B: Romanian comma-below fix
        if cp == 0x219:
            return 0x15F
        if cp == 0x21B:
            return 0x163
        return cp
    if cp >= 0x2000 and cp <= 0x206F:  # General Punctuation
        return SP
    if cp >= 0x600 and cp <= 0x6FF:  # Arabic: Farsi yeh => Arabic yeh
        if cp == 0x6CC:
            return 0x64A
        return cp
    if cp >= 0x1E00 and cp <= 0x1EFF:  # Latin Extended Additional (Vietnamese)
        if cp >= 0x1EA0:
            return 0x1EC3
        return cp
    if cp >= 0x3040 and cp <= 0x309F:  # Hiragana
        return 0x3042
    if cp >= 0x30A0 and cp <= 0x30FF:  # Katakana
        return 0x30A2
    if (cp >= 0x3100 and cp <= 0x312F) or (cp >= 0x31A0 and cp <= 0x31BF):  # Bopomofo
        return 0x3105
    if cp >= 0x4E00 and cp <= 0x9FFF:  # CJK Unified Ideographs
        return _cjk_map(p, cp)
    if cp >= 0xAC00 and cp <= 0xD7AF:  # Hangul Syllables
        return 0xAC00
    return cp


# --------------------------------------------------------------------------
# Byte-level URL / e-mail masking (the reference regexes only match ASCII,
# so byte-level matching on UTF-8 is exactly equivalent to running them on
# codepoints: multi-byte sequences never contain ASCII bytes)
# --------------------------------------------------------------------------


def _is_url_ch(b: UInt8) -> Bool:
    # [-_.?&~;+=/#0-9A-Za-z]
    if (b >= 0x30 and b <= 0x39) or (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
        return True
    return (
        b == 0x2D  # -
        or b == 0x5F  # _
        or b == 0x2E  # .
        or b == 0x3F  # ?
        or b == 0x26  # &
        or b == 0x7E  # ~
        or b == 0x3B  # ;
        or b == 0x2B  # +
        or b == 0x3D  # =
        or b == 0x2F  # /
        or b == 0x23  # #
    )


def _is_mail_a(b: UInt8) -> Bool:
    # [-_.0-9A-Za-z]
    if (b >= 0x30 and b <= 0x39) or (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
        return True
    return b == 0x2D or b == 0x5F or b == 0x2E


def _is_mail_b(b: UInt8) -> Bool:
    # [-_0-9A-Za-z]
    if (b >= 0x30 and b <= 0x39) or (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
        return True
    return b == 0x2D or b == 0x5F


def _run_url(text: U8Ptr, n: Int64, start: Int64, cap: Int64) -> Int64:
    var i = start
    while i < n and (i - start) < cap and _is_url_ch(text[unsafe_offset=Int(i)]):
        i += 1
    return i - start


def _sub_urls(text: U8Ptr, n: Int64) -> List[UInt8]:
    """re.sub(r'https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}', ' ', text)."""
    var out = List[UInt8]()
    var last = Int64(0)
    var i = Int64(0)
    while i < n:
        var body = Int64(-1)
        if text[unsafe_offset=Int(i)] == 0x68 and i + 7 < n:  # 'h', shortest "http://x" is 8
            if (
                text[unsafe_offset=Int(i + 1)] == 0x74  # t
                and text[unsafe_offset=Int(i + 2)] == 0x74  # t
                and text[unsafe_offset=Int(i + 3)] == 0x70  # p
            ):
                var k = i + 4
                if text[unsafe_offset=Int(k)] == 0x73:  # greedy 's?'
                    if (
                        k + 3 < n
                        and text[unsafe_offset=Int(k + 1)] == 0x3A  # :
                        and text[unsafe_offset=Int(k + 2)] == 0x2F  # /
                        and text[unsafe_offset=Int(k + 3)] == 0x2F  # /
                    ):
                        body = k + 4
                if body < 0:  # 's?' backtracked to empty
                    if (
                        text[unsafe_offset=Int(k)] == 0x3A
                        and text[unsafe_offset=Int(k + 1)] == 0x2F
                        and text[unsafe_offset=Int(k + 2)] == 0x2F
                    ):
                        body = k + 3
        if body >= 0:
            var run = _run_url(text, n, body, 2076)
            if run >= 1:
                for j in range(Int(last), Int(i)):
                    out.append(text[unsafe_offset=j])
                out.append(UInt8(0x20))
                i = body + run
                last = i
                continue
        i += 1
    for j in range(Int(last), Int(n)):
        out.append(text[unsafe_offset=j])
    return out^


def _sub_mails(var text: List[UInt8]) -> List[UInt8]:
    """re.sub(r'[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}', ' ', text).

    Replicates Python-re leftmost matching with greedy quantifiers and
    backtracking: for each start, the largest possible A-run is tried first,
    then the largest B-run (giving up one char to C when C would starve).
    Only the match span matters (the replacement is a single space).
    """
    var out = List[UInt8]()
    var n = Int64(len(text))
    var last = Int64(0)
    var i = Int64(0)
    while i < n:
        var end = Int64(-1)
        if _is_mail_a(text[Int(i)]):
            var a_len = _run_mail_a(text, i, 64)
            while a_len >= 1 and end < 0:
                var at = i + a_len
                if at < n and text[Int(at)] == 0x40:  # @
                    var j = at + 1
                    var b_len = _run_mail_b(text, j, 255)
                    if b_len >= 1:
                        var c_len = _run_mail_a(text, j + b_len, 255)
                        if c_len >= 1:
                            end = j + b_len + c_len
                        elif b_len >= 2:
                            b_len -= 1
                            c_len = _run_mail_a(text, j + b_len, 255)
                            end = j + b_len + c_len
                a_len -= 1
        if end >= 0:
            for j in range(Int(last), Int(i)):
                out.append(text[j])
            out.append(UInt8(0x20))
            i = end
            last = i
        else:
            i += 1
    for j in range(Int(last), Int(n)):
        out.append(text[j])
    return out^


def _run_mail_a(text: List[UInt8], start: Int64, cap: Int64) -> Int64:
    var i = start
    var n = Int64(len(text))
    while i < n and (i - start) < cap and _is_mail_a(text[Int(i)]):
        i += 1
    return i - start


def _run_mail_b(text: List[UInt8], start: Int64, cap: Int64) -> Int64:
    var i = start
    var n = Int64(len(text))
    while i < n and (i - start) < cap and _is_mail_b(text[Int(i)]):
        i += 1
    return i - start


# --------------------------------------------------------------------------
# Text preprocessing (codepoint domain)
# --------------------------------------------------------------------------


def _decode_utf8(var buf: List[UInt8]) -> List[UInt32]:
    """UTF-8 decoder that also accepts surrogatepass-encoded lone surrogates
    (WTF-8), mirroring Python str semantics for any text the wrapper can
    send. Truncated sequences decode the consumed bytes as-is; the wrapper
    only ever sends well-formed output of str.encode('utf-8','surrogatepass').
    """
    var out = List[UInt32]()
    var i = 0
    var n = len(buf)
    while i < n:
        var b0 = UInt32(buf[i])
        if b0 < 0x80:
            out.append(b0)
            i += 1
        elif b0 < 0xC0:
            out.append(b0)  # stray continuation byte: pass through
            i += 1
        elif b0 < 0xE0:
            var cp = b0 & 0x1F
            if i + 1 < n:
                cp = (cp << 6) | (UInt32(buf[i + 1]) & 0x3F)
                i += 2
            else:
                i += 1
            out.append(cp)
        elif b0 < 0xF0:
            var cp = b0 & 0x0F
            var take = 1
            if i + 2 < n:
                cp = (cp << 6) | (UInt32(buf[i + 1]) & 0x3F)
                cp = (cp << 6) | (UInt32(buf[i + 2]) & 0x3F)
                take = 3
            out.append(cp)
            i += take
        else:
            var cp = b0 & 0x07
            var take = 1
            if i + 3 < n:
                cp = (cp << 6) | (UInt32(buf[i + 1]) & 0x3F)
                cp = (cp << 6) | (UInt32(buf[i + 2]) & 0x3F)
                cp = (cp << 6) | (UInt32(buf[i + 3]) & 0x3F)
                take = 4
            out.append(cp)
            i += take
    return out^


def _normalize_vi(p: Profiles, var cps: List[UInt32]) -> List[UInt32]:
    """Map (alphabet, combining diacritic) pairs to precomposed codepoints,
    exactly the reference's ALPHABET_WITH_DMARK regex substitution."""
    var out = List[UInt32]()
    var i = 0
    var n = len(cps)
    while i < n:
        var a = _in_table(p.vi_alpha, p.n_via, cps[i])
        if a >= 0 and i + 1 < n:
            var d = _in_table(p.vi_dmark, p.n_vid, cps[i + 1])
            if d >= 0:
                out.append(p.vi_rows[unsafe_offset=d * Int(p.n_via) + a])
                i += 2
                continue
        out.append(cps[i])
        i += 1
    return out^


def _collapse_and_clean(p: Profiles, var cps: List[UInt32]) -> List[UInt32]:
    """Space-collapse with the 10000-codepoint cap, then the reference's
    cleaning_text: drop Latin-range chars when they are outnumbered 2:1 by
    non-Latin chars (U+0300+ outside Latin Extended Additional)."""
    var out = List[UInt32]()
    var m = len(cps)
    if m > Int(MAX_TEXT_LENGTH):
        m = Int(MAX_TEXT_LENGTH)
    var pre = UInt32(0)  # 0 != ' ', so the first char is always kept
    for i in range(m):
        var ch = cps[i]
        if ch != SP or pre != SP:
            out.append(ch)
        pre = ch
    var latin = 0
    var non_latin = 0
    for i in range(len(out)):
        var ch = out[i]
        if ch >= 0x41 and ch <= 0x7A:
            latin += 1
        elif ch >= 0x300 and not (ch >= 0x1E00 and ch <= 0x1EFF):
            non_latin += 1
    if latin * 2 < non_latin:
        var cleaned = List[UInt32]()
        for i in range(len(out)):
            var ch = out[i]
            if not (ch >= 0x41 and ch <= 0x7A):
                cleaned.append(ch)
        return cleaned^
    return out^


# --------------------------------------------------------------------------
# N-gram extraction (reference NGram state machine + Detector._extract_ngrams)
# --------------------------------------------------------------------------


def _extract_ngrams(p: Profiles, cps: List[UInt32]) -> List[UInt32]:
    """Sliding 3-codepoint window with the reference's capitalword rule;
    emits (as profile key ids) the 1..3-grams that exist in the profiles."""
    var ids = List[UInt32]()
    var g = List[UInt32](capacity=4)
    g.append(SP)
    var capital = False
    for i in range(len(cps)):
        var ch = _normalize_cp(p, cps[i])
        var last = g[len(g) - 1]
        var skip_append = False
        if last == SP:
            g.clear()
            g.append(SP)
            capital = False
            if ch == SP:
                skip_append = True
        elif len(g) >= 3:
            g[0] = g[1]
            g[1] = g[2]
            _ = g.pop()
        if not skip_append:
            g.append(ch)
            if _is_upper(p, ch):
                if _is_upper(p, last):
                    capital = True
            else:
                capital = False
        if capital:
            continue
        # for n in 1..3: emit grams[-n:] when it is a known profile key
        for n in range(1, 4):
            if len(g) < n:
                break
            if n == 1 and g[len(g) - 1] == SP:
                continue  # w == ' ' is never emitted
            # pack left-aligned: first codepoint of the n-gram in bits 0..20,
            # exactly how the wrapper packs the profile keys
            var key = UInt64(0)
            for k in range(n):
                key |= UInt64(g[len(g) - n + k]) << UInt64(21 * k)
            var kid = _find_key(p, key)
            if kid >= 0:
                ids.append(UInt32(kid))
    return ids^


# --------------------------------------------------------------------------
# Seeded sampling detector (reference Detector._detect_block)
# --------------------------------------------------------------------------


def _detect_block(
    p: Profiles, var ids: List[UInt32], seed: UInt64, outp: F64Ptr, flags: UInt32
):
    var n_langs = Int(p.n_langs)
    var rng = CRandom(seed)
    var langprob = List[Float64](capacity=n_langs)
    for _ in range(n_langs):
        langprob.append(0.0)
    var prob = List[Float64](capacity=n_langs)
    var addv = List[Float64](capacity=n_langs)
    for _ in range(n_langs):
        prob.append(0.0)
        addv.append(0.0)
    var init_p = 1.0 / Float64(p.n_langs)
    for _t in range(Int(N_TRIAL)):
        for j in range(n_langs):
            prob[j] = init_p
        var alpha = ALPHA_DEFAULT + rng.gauss() * ALPHA_WIDTH
        var i = Int64(0)
        while True:
            var kid = Int(ids[Int(rng.randbelow(UInt64(len(ids))))])
            var weight = alpha / BASE_FREQ
            for j in range(n_langs):
                addv[j] = weight
            var e0 = Int(p.kid_off[unsafe_offset=kid])
            var e1 = Int(p.kid_off[unsafe_offset=kid + 1])
            for e in range(e0, e1):
                addv[Int(p.ent_lang[unsafe_offset=e])] = (
                    weight + p.ent_prob[unsafe_offset=e]
                )
            for j in range(n_langs):
                prob[j] = prob[j] * addv[j]
            if i % 5 == 0:
                # CPython 3.12+ sum() uses Neumaier (compensated) summation,
                # older interpreters a plain left-to-right sum; replicate the
                # interpreter's flavor so the normalized vector is identical.
                var sump = Float64(0.0)
                if (flags & FLAG_NAIVE_SUM) != 0:
                    for j in range(n_langs):
                        sump += prob[j]
                else:
                    var hi = Float64(0.0)
                    var lo = Float64(0.0)
                    for j in range(n_langs):
                        var x = prob[j]
                        var t = hi + x
                        if abs(hi) >= abs(x):
                            lo += (hi - t) + x
                        else:
                            lo += (x - t) + hi
                        hi = t
                    sump = hi + lo
                var maxp = Float64(0.0)
                for j in range(n_langs):
                    var pj = prob[j] / sump
                    if maxp < pj:
                        maxp = pj
                    prob[j] = pj
                if maxp > CONV_THRESHOLD or i >= ITERATION_LIMIT:
                    break
            i += 1
        for j in range(n_langs):
            langprob[j] += prob[j] / Float64(N_TRIAL)
    for j in range(n_langs):
        outp[unsafe_offset=j] = langprob[j]


# --------------------------------------------------------------------------
# Exported C ABI
# --------------------------------------------------------------------------


@export
def langdetectmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def _read_u64(blob: U8Ptr, off: Int64) -> UInt64:
    # `off` is always a multiple of 8 by blob construction
    return blob.unsafe_bitcast[UInt64]()[unsafe_offset=Int(off // 8)]


def _read_u32(blob: U8Ptr, off: Int64) -> UInt32:
    # `off` is always a multiple of 4 by blob construction
    return blob.unsafe_bitcast[UInt32]()[unsafe_offset=Int(off // 4)]


def _read_f64(blob: U8Ptr, off: Int64) -> Float64:
    return blob.unsafe_bitcast[Float64]()[unsafe_offset=Int(off // 8)]


def _copy_u32_table(blob: U8Ptr, off: Int64, n: Int64) -> U32Ptr:
    var dst = unsafe_alloc[UInt32](Int(n))
    for i in range(Int(n)):
        dst[unsafe_offset=i] = _read_u32(blob, off + Int64(i) * 4)
    return dst


def _copy_u64_table(blob: U8Ptr, off: Int64, n: Int64) -> U64Ptr:
    var dst = unsafe_alloc[UInt64](Int(n))
    for i in range(Int(n)):
        dst[unsafe_offset=i] = _read_u64(blob, off + Int64(i) * 8)
    return dst


def _parse_blob(blob: U8Ptr, blob_len: Int64) -> Handle:
    """Validate and copy the wrapper's profile blob; NULL when malformed.

    All validation happens against the caller's blob first; only after every
    check passes does the kernel allocate and copy, so a rejected blob never
    leaves partial state behind.
    """
    if blob_len < HDR_WORDS * 8:
        return None
    if _read_u64(blob, 0) != BLOB_MAGIC or _read_u64(blob, 8) != BLOB_VERSION:
        return None
    var n_langs = _read_u64(blob, H_N_LANGS * 8)
    var n_keys = _read_u64(blob, H_N_KEYS * 8)
    var n_entries = _read_u64(blob, H_N_ENTRIES * 8)
    if n_langs <= 0 or n_langs > 4096 or n_keys < 0 or n_entries < 0:
        return None
    var n_cjk = _read_u64(blob, H_N_CJK * 8)
    var n_latin1 = _read_u64(blob, H_N_LATIN1 * 8)
    var n_via = _read_u64(blob, H_N_VI_ALPHA * 8)
    var n_vid = _read_u64(blob, H_N_VI_DMARK * 8)
    var n_upper = _read_u64(blob, H_N_UPPER * 8)
    if (
        n_cjk < 0
        or n_cjk > 1 << 22
        or n_latin1 < 0
        or n_latin1 > 1 << 12
        or n_via <= 0
        or n_via > 1 << 12
        or n_vid <= 0
        or n_vid > 1 << 12
        or n_upper <= 0
        or n_upper > 1 << 16
    ):
        return None

    # Section (offset, byte length) table; bound-check before any dereference.
    var off_langs = Int64(_read_u64(blob, H_OFF_LANGS * 8))
    var off_keys = Int64(_read_u64(blob, H_OFF_KEYS * 8))
    var off_kidoff = Int64(_read_u64(blob, H_OFF_KIDOFF * 8))
    var off_ent = Int64(_read_u64(blob, H_OFF_ENT * 8))
    var off_cjk = Int64(_read_u64(blob, H_OFF_CJK * 8))
    var off_latin1 = Int64(_read_u64(blob, H_OFF_LATIN1 * 8))
    var off_via = Int64(_read_u64(blob, H_OFF_VI_ALPHA * 8))
    var off_vid = Int64(_read_u64(blob, H_OFF_VI_DMARK * 8))
    var off_vrows = Int64(_read_u64(blob, H_OFF_VI_ROWS * 8))
    var off_upper = Int64(_read_u64(blob, H_OFF_UPPER * 8))
    var offs = List[Int64]()
    var lens = List[Int64]()
    offs.append(off_langs)
    lens.append(Int64(0))  # informational section; only its offset is checked
    offs.append(off_keys)
    lens.append(Int64(n_keys) * 8)
    offs.append(off_kidoff)
    lens.append((Int64(n_keys) + 1) * 8)
    offs.append(off_ent)
    lens.append(Int64(n_entries) * 16)
    offs.append(off_cjk)
    lens.append(Int64(n_cjk) * 8)
    offs.append(off_latin1)
    lens.append(Int64(n_latin1) * 4)
    offs.append(off_via)
    lens.append(Int64(n_via) * 4)
    offs.append(off_vid)
    lens.append(Int64(n_vid) * 4)
    offs.append(off_vrows)
    lens.append(Int64(n_via) * 5 * 4)
    offs.append(off_upper)
    lens.append(Int64(n_upper) * 8)
    for s in range(len(offs)):
        if offs[s] < HDR_WORDS * 8 or offs[s] % 8 != 0:
            return None
        if offs[s] + lens[s] > blob_len:
            return None

    # Content validation straight from the caller's blob (no allocation yet).
    if _read_u64(blob, off_kidoff) != 0 or _read_u64(
        blob, off_kidoff + Int64(n_keys) * 8
    ) != UInt64(n_entries):
        return None
    for i in range(1, Int(n_keys)):
        if _read_u64(blob, off_keys + Int64(i - 1) * 8) >= _read_u64(
            blob, off_keys + Int64(i) * 8
        ):
            return None
    for e in range(Int(n_entries)):
        if _read_u32(blob, off_ent + Int64(e) * 16) >= UInt32(n_langs):
            return None
    for i in range(1, Int(n_cjk)):
        if _read_u32(blob, off_cjk + Int64(i - 1) * 8) >= _read_u32(
            blob, off_cjk + Int64(i) * 8
        ):
            return None
    for i in range(Int(n_upper)):
        var lo = _read_u32(blob, off_upper + Int64(i) * 8)
        var hi = _read_u32(blob, off_upper + Int64(i) * 8 + 4)
        if lo > hi:
            return None
        if i > 0 and lo <= _read_u32(blob, off_upper + Int64(i - 1) * 8 + 4):
            return None

    # Everything checks out: copy into kernel-owned storage.
    var keys = _copy_u64_table(blob, off_keys, Int64(n_keys))
    var kid_off = _copy_u64_table(blob, off_kidoff, Int64(n_keys) + 1)
    var ent_lang = unsafe_alloc[UInt32](Int(n_entries))
    var ent_prob = unsafe_alloc[Float64](Int(n_entries))
    for e in range(Int(n_entries)):
        ent_lang[unsafe_offset=e] = _read_u32(blob, off_ent + Int64(e) * 16)
        ent_prob[unsafe_offset=e] = _read_f64(blob, off_ent + Int64(e) * 16 + 8)
    var cjk_from = unsafe_alloc[UInt32](Int(n_cjk))
    var cjk_to = unsafe_alloc[UInt32](Int(n_cjk))
    for i in range(Int(n_cjk)):
        cjk_from[unsafe_offset=i] = _read_u32(blob, off_cjk + Int64(i) * 8)
        cjk_to[unsafe_offset=i] = _read_u32(blob, off_cjk + Int64(i) * 8 + 4)
    var upper_lo = unsafe_alloc[UInt32](Int(n_upper))
    var upper_hi = unsafe_alloc[UInt32](Int(n_upper))
    for i in range(Int(n_upper)):
        upper_lo[unsafe_offset=i] = _read_u32(blob, off_upper + Int64(i) * 8)
        upper_hi[unsafe_offset=i] = _read_u32(blob, off_upper + Int64(i) * 8 + 4)

    var slot = unsafe_alloc[Profiles](1)
    slot[] = Profiles(
        Int64(n_langs),
        Int64(n_keys),
        Int64(n_entries),
        keys,
        kid_off,
        ent_lang,
        ent_prob,
        Int64(n_cjk),
        cjk_from,
        cjk_to,
        Int64(n_latin1),
        _copy_u32_table(blob, off_latin1, Int64(n_latin1)),
        Int64(n_via),
        _copy_u32_table(blob, off_via, Int64(n_via)),
        Int64(n_vid),
        _copy_u32_table(blob, off_vid, Int64(n_vid)),
        _copy_u32_table(blob, off_vrows, Int64(n_via) * 5),
        Int64(n_upper),
        upper_lo,
        upper_hi,
    )
    return slot.unsafe_bitcast[UInt8]()


@export
def langdetectmojo_profiles_create(blob: U8Ptr, blob_len: Int64) abi("C") -> Handle:
    if blob_len < 0:
        return None
    return _parse_blob(blob, blob_len)


@export
def langdetectmojo_detect(
    handle: Handle,
    text: U8Ptr,
    text_len: Int64,
    seed: UInt64,
    out_probs: F64Ptr,
    out_cap: Int64,
    flags: UInt32,
) abi("C") -> Int32:
    if not handle:
        return 2
    var p = handle.value().unsafe_bitcast[Profiles]()
    if out_cap < p[].n_langs or text_len < 0:
        return 2

    # 1-2. URL then e-mail masking (byte level, ASCII-only patterns)
    var no_urls = _sub_urls(text, text_len)
    var no_mails = _sub_mails(no_urls^)
    # 3-5. decode, Vietnamese normalization, collapse + clean
    var cps = _decode_utf8(no_mails^)
    cps = _normalize_vi(p[], cps^)
    cps = _collapse_and_clean(p[], cps^)
    # 6. n-gram extraction against the profile key set
    var ids = _extract_ngrams(p[], cps^)
    if len(ids) == 0:
        return 1
    # 7. seeded sampling over the profile vectors
    _detect_block(p[], ids^, seed, out_probs, flags)
    return 0


@export
def langdetectmojo_profiles_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var p = handle.value().unsafe_bitcast[Profiles]()
    p[].free()
    p.unsafe_free()
