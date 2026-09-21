# dateutilmojo: Mojo kernel for the dateutil-mojo package.
#
# Fast-path parser for the hot date formats (ISO 8601, RFC 2822, US/EU
# numeric, named-month) with semantics verified against the python-dateutil
# 2.9.0.post0 oracle. Anything outside the implemented shape returns status
# 1 ("unhandled") and the Python wrapper re-parses with the reference
# implementation, so behavior is always oracle-exact.
#
# Clean-room: written from black-box observation of the oracle.
#
# Stable C ABI (v1):
#   int32_t dateutilmojo_abi_version(void)
#   int32_t dateutilmojo_parse_batch(const uint8_t* data,
#                                    const int64_t* offsets,  // n+1
#                                    int64_t n, uint32_t flags,
#                                    const int32_t* defaults, // 7 fields + pivot base
#                                    int32_t* out,            // n*9
#                                    int64_t out_len)
#
# Per output row: [status, year, month, day, hour, minute, second,
#                  microsecond, tzoffset_seconds]. status 0 = parsed
# (INT32_MIN tzoffset = naive), status 1 = unhandled (fall back to Python).

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]

comptime ABI_VERSION: Int32 = 1
comptime UNSET: Int32 = -2147483648  # INT32_MIN sentinel
comptime MAX_TOKENS: Int = 64
comptime I32MAX: Int64 = 2147483647


# ------------------------------------------------------------------ helpers

def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


def _is_numword(b: UInt8) -> Bool:
    if b >= 48 and b <= 57:  # 0-9
        return True
    return b == 46 or b == 58 or b == 43 or b == 45 or b == 47 or b == 44  # .:+-/,


def _is_alpha(b: UInt8) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)


def _is_space(b: UInt8) -> Bool:
    return b == 32 or (b >= 9 and b <= 13)


def _lower(b: UInt8) -> UInt8:
    if b >= 65 and b <= 90:
        return b + 32
    return b


def _word_eq(s: U8Ptr, start: Int, end: Int, word: String, ci: Bool) -> Bool:
    var wb = word.as_bytes()
    if end - start != len(wb):
        return False
    for i in range(len(wb)):
        var b = s[start + i]
        if ci:
            b = _lower(b)
        if Int(b) != Int(wb[i]):
            return False
    return True


def _parse_uint(s: U8Ptr, start: Int, end: Int) -> Int64:
    var v: Int64 = 0
    for i in range(start, end):
        v = v * 10 + Int64(s[i] - 48)
    return v


def _all_digits(s: U8Ptr, start: Int, end: Int) -> Bool:
    if end <= start:
        return False
    for i in range(start, end):
        if not _is_digit(s[i]):
            return False
    return True


def _pivot(y: Int32, base: Int32) -> Int32:
    if y < 0 or y > 99:
        return y
    var c = y + 2000
    if c >= base:
        c -= 100
    return c


# Offset body "HH", "H", "HHMM", "HH:MM", "H:M" -> seconds; -1 = invalid.
def _offset_body(s: U8Ptr, start: Int, end: Int) -> Int32:
    var n = end - start
    if n <= 0:
        return -1
    var colon = -1
    for i in range(start, end):
        if s[i] == 58:
            if colon >= 0:
                return -1
            colon = i
    if colon >= 0:
        if colon == start or colon + 1 >= end:
            return -1
        if not _all_digits(s, start, colon) or not _all_digits(s, colon + 1, end):
            return -1
        var h = _parse_uint(s, start, colon)
        var m = _parse_uint(s, colon + 1, end)
        if h > I32MAX or m > I32MAX:
            return -1
        return Int32(h * 3600 + m * 60)
    if not _all_digits(s, start, end):
        return -1
    if n <= 2:
        return Int32(_parse_uint(s, start, end) * 3600)
    if n == 4:
        return Int32(
            _parse_uint(s, start, start + 2) * 3600
            + _parse_uint(s, start + 2, end) * 60
        )
    return -1


# ------------------------------------------------------------------- parser

struct Parser:
    var data: U8Ptr
    var n: Int
    var dayfirst: Bool
    var yearfirst: Bool
    var pivot_base: Int32
    var tkind: List[Int32]
    var tstart: List[Int]
    var tend: List[Int]
    var yv: List[Int32]
    var ynd: List[Int32]
    var ytok: List[Int32]
    var month: Int32
    var month_tok: Int32
    var hour: Int32
    var minute: Int32
    var second: Int32
    var micro: Int32
    var weekday: Int32
    var tz_utczero: Bool
    var tzoffset: Int32
    var date_seen: Bool
    var day_explicit: Bool
    var parts: List[Int]
    var g: List[Int]
    var vals: List[Int64]
    var fracs: List[Int]

    def __init__(out self, data: U8Ptr, n: Int, flags: UInt32, pivot_base: Int32):
        self.data = data
        self.n = n
        self.dayfirst = (flags & 1) != 0
        self.yearfirst = (flags & 2) != 0
        self.pivot_base = pivot_base
        self.tkind = List[Int32](capacity=32)
        self.tstart = List[Int](capacity=32)
        self.tend = List[Int](capacity=32)
        self.yv = List[Int32](capacity=8)
        self.ynd = List[Int32](capacity=8)
        self.ytok = List[Int32](capacity=8)
        self.month = 0
        self.month_tok = -1
        self.hour = -1
        self.minute = -1
        self.second = -1
        self.micro = -1
        self.weekday = -1
        self.tz_utczero = False
        self.tzoffset = UNSET
        self.date_seen = False
        self.day_explicit = False
        self.parts = List[Int](capacity=8)
        self.g = List[Int](capacity=8)
        self.vals = List[Int64](capacity=4)
        self.fracs = List[Int](capacity=8)

    def reset(mut self, data: U8Ptr, n: Int):
        """Reinitialize this parser for the next row (keeps list capacity)."""
        self.data = data
        self.n = n
        self.tkind.clear()
        self.tstart.clear()
        self.tend.clear()
        self.yv.clear()
        self.ynd.clear()
        self.ytok.clear()
        self.month = 0
        self.month_tok = -1
        self.hour = -1
        self.minute = -1
        self.second = -1
        self.micro = -1
        self.weekday = -1
        self.tz_utczero = False
        self.tzoffset = UNSET
        self.date_seen = False
        self.day_explicit = False
        self.parts.clear()
        self.g.clear()
        self.vals.clear()
        self.fracs.clear()

    # --------------------------------------------------------- tokenization

    def tokenize(mut self) -> Bool:
        var i = 0
        while i < self.n:
            var b = self.data[i]
            if b >= 128:
                return False  # non-ASCII: defer to Python
            if _is_space(b):
                i += 1
                continue
            if _is_numword(b):
                var j = i + 1
                while j < self.n and _is_numword(self.data[j]):
                    j += 1
                self._push(0, i, j)
                i = j
            elif _is_alpha(b):
                var j = i + 1
                while j < self.n and _is_alpha(self.data[j]):
                    j += 1
                self._push(1, i, j)
                i = j
            else:
                return False  # punctuation: defer to Python
            if len(self.tkind) > MAX_TOKENS:
                return False
        return True

    def _push(mut self, kind: Int32, start: Int, end: Int):
        self.tkind.append(kind)
        self.tstart.append(start)
        self.tend.append(end)

    def _is_word(self, i: Int, word: String, ci: Bool) -> Bool:
        return _word_eq(self.data, self.tstart[i], self.tend[i], word, ci)

    def _ampm_follows(self, i: Int) -> Bool:
        if i + 1 >= len(self.tkind) or self.tkind[i + 1] != 1:
            return False
        return (
            self._is_word(i + 1, "am", True)
            or self._is_word(i + 1, "pm", True)
            or self._is_word(i + 1, "a", True)
            or self._is_word(i + 1, "p", True)
        )

    def _attached_signed(self, i: Int) -> Bool:
        if i + 1 >= len(self.tkind):
            return False
        if self.tkind[i + 1] != 0:
            return False
        if self.tstart[i + 1] != self.tend[i]:
            return False
        var b = self.data[self.tstart[i + 1]]
        return b == 43 or b == 45

    # ------------------------------------------------------- number tokens

    def handle_number(mut self, i: Int) -> Bool:
        var start = self.tstart[i]
        var end = self.tend[i]
        var n = end - start
        var b0 = self.data[start]

        if n == 1 and (b0 == 45 or b0 == 46 or b0 == 44):
            return True  # lone separator
        if self.data[end - 1] == 44 and n > 1:
            end -= 1
            n -= 1

        var has_comma = False
        var has_colon = False
        var has_dash = False
        var has_slash = False
        var has_dot = False
        for k in range(start, end):
            var b = self.data[k]
            if b == 44:
                has_comma = True
            elif b == 58:
                has_colon = True
            elif b == 45:
                has_dash = True
            elif b == 47:
                has_slash = True
            elif b == 46:
                has_dot = True

        if has_comma:
            return self._handle_comma(i, start, end)
        if b0 == 43 or b0 == 45:
            return self._handle_signed(i, start, end)
        if has_colon:
            return self._parse_time(i, start, end, -1, -1)
        if has_dash or has_slash:
            return self._handle_dashed(i, start, end, has_dash)
        if has_dot:
            return self._handle_dotted(i, start, end)
        if not _all_digits(self.data, start, end):
            return False
        return self._digits(i, start, end)

    def _handle_comma(mut self, i: Int, start: Int, end: Int) -> Bool:
        # Accept only "H:MM[:SS[.f]],fff" (decimal comma after a time).
        var comma = -1
        var ncomma = 0
        for k in range(start, end):
            if self.data[k] == 44:
                ncomma += 1
                comma = k
        if ncomma != 1:
            return False
        var colon_seen = False
        for k in range(start, comma):
            var b = self.data[k]
            if b == 58:
                colon_seen = True
            elif not (_is_digit(b) or b == 46):
                return False
        if not colon_seen:
            return False
        if not _all_digits(self.data, comma + 1, end):
            return False
        return self._parse_time(i, start, comma, comma + 1, end)

    def _handle_signed(mut self, i: Int, start: Int, end: Int) -> Bool:
        var b0 = self.data[start]
        if self.hour < 0:
            # '-' acts as a date separator ("8-Jan-2025"); '+' is offset-only.
            if b0 == 45 and _all_digits(self.data, start + 1, end):
                return self._digits(i, start + 1, end)
            return False
        var off = _offset_body(self.data, start + 1, end)
        if off < 0:
            return False
        if b0 == 45:
            off = -off
        if self.tz_utczero and self.tzoffset == UNSET:
            return True  # spaced offset after UTC/GMT/Z: dropped
        self.tzoffset = off
        return True

    def _handle_dashed(mut self, i: Int, start: Int, end: Int, has_dash: Bool) -> Bool:
        var sep: UInt8 = 45
        if not has_dash:
            sep = 47
        self.parts.clear()
        var prev = start
        for k in range(start, end + 1):
            if k == end or self.data[k] == sep:
                if k > prev:
                    self.parts.append(prev)
                    self.parts.append(k)
                prev = k + 1
        var nparts = len(self.parts) // 2
        if nparts == 0:
            return False
        for p in range(nparts):
            var plen = self.parts[2 * p + 1] - self.parts[2 * p]
            if plen > 18 or not _all_digits(self.data, self.parts[2 * p], self.parts[2 * p + 1]):
                return False
        if self.date_seen:
            # Compact time with tz offsets ("2012-02-02" -> 20:12 -02:00).
            if sep == 47 or self.hour >= 0:
                return False
            var flen = self.parts[1] - self.parts[0]
            if flen != 1 and flen != 2 and flen != 4 and flen != 6:
                return False
            if flen <= 2:
                self.hour = Int32(_parse_uint(self.data, self.parts[0], self.parts[1]))
            elif flen == 4:
                self.hour = Int32(_parse_uint(self.data, self.parts[0], self.parts[0] + 2))
                self.minute = Int32(_parse_uint(self.data, self.parts[0] + 2, self.parts[1]))
            else:
                self.hour = Int32(_parse_uint(self.data, self.parts[0], self.parts[0] + 2))
                self.minute = Int32(_parse_uint(self.data, self.parts[0] + 2, self.parts[0] + 4))
                self.second = Int32(_parse_uint(self.data, self.parts[0] + 4, self.parts[1]))
                self.micro = 0
            for p in range(1, nparts):
                var off = _offset_body(self.data, self.parts[2 * p], self.parts[2 * p + 1])
                if off < 0:
                    return False
                self.tzoffset = -off
            return True
        if nparts > 3:
            return False
        if nparts == 1:
            return self._digits(i, self.parts[0], self.parts[1])
        if nparts == 2:
            var a = _parse_uint(self.data, self.parts[0], self.parts[1])
            var b = _parse_uint(self.data, self.parts[2], self.parts[3])
            if a > I32MAX or b > I32MAX:
                return False
            var na = self.parts[1] - self.parts[0]
            var nb = self.parts[3] - self.parts[2]
            var a_year = na == 4 or a > 31
            var b_year = nb == 4 or b > 31
            if (a_year and nb > 2) or (b_year and na > 2):
                return False
            # Year-month pairs with an invalid month are a special error in
            # the Python reference ("bad month number"); defer to it.
            if (a_year and (b < 1 or b > 12)) or (b_year and (a < 1 or a > 12)):
                return False
            self._ymd(Int32(a), Int32(na), i)
            self._ymd(Int32(b), Int32(nb), i)
            self.date_seen = True
            if not a_year and not b_year:
                self.day_explicit = True
            return True
        for p in range(3):
            var v = _parse_uint(self.data, self.parts[2 * p], self.parts[2 * p + 1])
            if v > I32MAX:
                return False
            self._ymd(
                Int32(v),
                Int32(self.parts[2 * p + 1] - self.parts[2 * p]),
                i,
            )
        self.date_seen = True
        self.day_explicit = True
        return True

    def _handle_dotted(mut self, i: Int, start: Int, end: Int) -> Bool:
        var e = end
        if self.data[e - 1] == 46:
            e -= 1
            if e <= start:
                return True
            if not _all_digits(self.data, start, e):
                return False
            return self._digits(i, start, e)
        self.parts.clear()
        var prev = start
        for k in range(start, e + 1):
            if k == e or self.data[k] == 46:
                self.parts.append(prev)
                self.parts.append(k)
                prev = k + 1
        if len(self.parts) != 6:
            return False  # 2-part or 4+-part dot tokens defer to Python
        for p in range(3):
            var plen = self.parts[2 * p + 1] - self.parts[2 * p]
            if plen > 18 or not _all_digits(self.data, self.parts[2 * p], self.parts[2 * p + 1]):
                return False
        for p in range(3):
            var v = _parse_uint(self.data, self.parts[2 * p], self.parts[2 * p + 1])
            if v > I32MAX:
                return False
            self._ymd(Int32(v), Int32(self.parts[2 * p + 1] - self.parts[2 * p]), i)
        self.date_seen = True
        self.day_explicit = True
        return True

    def _ymd(mut self, v: Int32, nd: Int32, tok: Int):
        self.yv.append(v)
        self.ynd.append(nd)
        self.ytok.append(Int32(tok))

    def _digits(mut self, i: Int, start: Int, end: Int) -> Bool:
        var nd = end - start
        if nd > 18:
            return False  # huge numbers: Python reproduces the exact error
        var value = _parse_uint(self.data, start, end)

        # am/pm lookahead: "9 PM" -> hour 9.
        if nd <= 2 and value <= 24 and self.hour < 0 and self._ampm_follows(i):
            self.hour = Int32(value)
            return True

        if self.date_seen or (self.month != 0 and len(self.yv) >= 2):
            if self.hour < 0:
                if nd <= 2:
                    self.hour = Int32(value)
                    return True
                if nd == 4:
                    self.hour = Int32(value // 100)
                    self.minute = Int32(value % 100)
                    return True
                if nd == 6:
                    self.hour = Int32(value // 10000)
                    self.minute = Int32((value // 100) % 100)
                    self.second = Int32(value % 100)
                    self.micro = 0
                    return True
            return False

        if nd == 8:
            self._ymd(Int32(value // 10000), 4, i)
            self._ymd(Int32((value // 100) % 100), 2, i)
            self._ymd(Int32(value % 100), 2, i)
            self.date_seen = True
            self.day_explicit = True
            return True
        if nd == 12 or nd == 14:
            if nd == 12:
                self._ymd(Int32(value // 100000000), 4, i)
                self._ymd(Int32((value // 1000000) % 100), 2, i)
                self._ymd(Int32((value // 10000) % 100), 2, i)
                self.hour = Int32((value // 100) % 100)
                self.minute = Int32(value % 100)
            else:
                self._ymd(Int32(value // 10000000000), 4, i)
                self._ymd(Int32((value // 100000000) % 100), 2, i)
                self._ymd(Int32((value // 1000000) % 100), 2, i)
                self.hour = Int32((value // 10000) % 100)
                self.minute = Int32((value // 100) % 100)
                self.second = Int32(value % 100)
            self.date_seen = True
            self.day_explicit = True
            return True
        if nd == 6:
            self._ymd(Int32(value // 10000), 2, i)
            self._ymd(Int32((value // 100) % 100), 2, i)
            self._ymd(Int32(value % 100), 2, i)
            self.date_seen = True
            self.day_explicit = True
            return True
        if nd >= 3:
            if value > I32MAX:
                return False  # Python reproduces the exact OverflowError
            var nd_eff: Int32 = 4
            if nd < 4:
                nd_eff = Int32(nd)
            self._ymd(Int32(value), nd_eff, i)
            if len(self.yv) >= 3:
                self.date_seen = True
            return True
        self._ymd(Int32(value), Int32(nd), i)
        if len(self.yv) >= 3:
            self.date_seen = True
        return True

    # ------------------------------------------------------------ time token

    def _parse_time(
        mut self, i: Int, start: Int, end: Int, cf_start: Int, cf_end: Int
    ) -> Bool:
        var time_end = end
        var off_start = -1
        for k in range(start + 1, end):
            var b = self.data[k]
            if b == 43 or b == 45:
                time_end = k
                off_start = k
                break
        self.g.clear()
        var prev = start
        for k in range(start, time_end + 1):
            if k == time_end or self.data[k] == 58:
                self.g.append(prev)
                self.g.append(k)
                prev = k + 1
        var ng = len(self.g) // 2
        if ng < 2 or ng > 3:
            return False
        self.vals.clear()
        self.fracs.clear()
        # fracs: (start, end) per group; (-1, -1) if none
        for p in range(ng):
            var gs = self.g[2 * p]
            var ge = self.g[2 * p + 1]
            if ge <= gs:
                return False
            var dot = -1
            for k in range(gs, ge):
                if self.data[k] == 46:
                    if dot >= 0:
                        return False
                    dot = k
            if dot < 0:
                if not _all_digits(self.data, gs, ge):
                    return False
                self.vals.append(_parse_uint(self.data, gs, ge))
                self.fracs.append(-1)
                self.fracs.append(-1)
            else:
                if not _all_digits(self.data, gs, dot):
                    return False
                if dot + 1 < ge and not _all_digits(self.data, dot + 1, ge):
                    return False
                self.vals.append(_parse_uint(self.data, gs, dot))
                if dot + 1 < ge:
                    self.fracs.append(dot + 1)
                    self.fracs.append(ge)
                else:
                    self.fracs.append(-1)
                    self.fracs.append(-1)
        if self.vals[0] > I32MAX or self.vals[1] > I32MAX or (ng == 3 and self.vals[2] > I32MAX):
            return False
        self.hour = Int32(self.vals[0])
        self.minute = Int32(self.vals[1])
        if ng == 2:
            var f0 = -1
            var f1 = -1
            if cf_start >= 0:
                f0 = cf_start
                f1 = cf_end
            elif self.fracs[2] >= 0:
                f0 = self.fracs[2]
                f1 = self.fracs[3]
            if f0 >= 0:
                var frac = Float64(_parse_uint(self.data, f0, f1))
                var scale = Float64(1.0)
                for _ in range(f1 - f0):
                    scale *= 10.0
                self.second = Int32(Int64(frac / scale * 60.0))
        else:
            self.second = Int32(self.vals[2])
            var f0 = -1
            var f1 = -1
            if cf_start >= 0:
                f0 = cf_start
                f1 = cf_end
            elif self.fracs[4] >= 0:
                f0 = self.fracs[4]
                f1 = self.fracs[5]
            if f0 >= 0:
                self.micro = self._frac_to_micro(f0, f1)
            else:
                self.micro = 0
        if off_start >= 0:
            var off = _offset_body(self.data, off_start + 1, end)
            if off < 0:
                return False
            if self.data[off_start] == 45:
                off = -off
            self.tzoffset = off
        return True

    def _frac_to_micro(self, f0: Int, f1: Int) -> Int32:
        var v: Int64 = 0
        var digits = 0
        for k in range(f0, f1):
            if digits < 6:
                v = v * 10 + Int64(self.data[k] - 48)
                digits += 1
        while digits < 6:
            v *= 10
            digits += 1
        return Int32(v)

    # --------------------------------------------------------- alpha tokens

    def handle_alpha(mut self, i: Int) -> Bool:
        var s = self.tstart[i]
        var n = self.tend[i] - s

        # Ordinal suffix after a number.
        if n == 2 and i > 0 and self.tkind[i - 1] == 0:
            if (
                self._is_word(i, "st", True)
                or self._is_word(i, "nd", True)
                or self._is_word(i, "rd", True)
                or self._is_word(i, "th", True)
            ):
                return True

        var m = self._month_of(i)
        if m > 0:
            if self.date_seen or self.month != 0:
                return False
            self.month = m
            self.month_tok = Int32(i)
            self._eat_dot(i)
            return True

        var wd = self._weekday_of(i)
        if wd >= 0:
            self.weekday = wd
            self._eat_dot(i)
            return True

        if self.hour >= 0 and self.hour <= 12:
            var ap = -1
            if self._is_word(i, "am", True) or self._is_word(i, "a", True):
                ap = 0
            elif self._is_word(i, "pm", True) or self._is_word(i, "p", True):
                ap = 1
            if ap >= 0:
                if n == 1 and i + 2 < len(self.tkind):
                    if (
                        self.tkind[i + 1] == 0
                        and self.tend[i + 1] - self.tstart[i + 1] == 1
                        and self.data[self.tstart[i + 1]] == 46
                        and self.tkind[i + 2] == 1
                        and self._is_word(i + 2, "m", False)
                    ):
                        self.tkind[i + 1] = 3
                        self.tkind[i + 2] = 3
                        if (
                            i + 3 < len(self.tkind)
                            and self.tkind[i + 3] == 0
                            and self.tend[i + 3] - self.tstart[i + 3] == 1
                            and self.data[self.tstart[i + 3]] == 46
                        ):
                            self.tkind[i + 3] = 3
                if ap == 0:
                    if self.hour == 12:
                        self.hour = 0
                elif self.hour < 12:
                    self.hour += 12
                return True
        elif self._is_word(i, "am", True) or self._is_word(i, "pm", True) or (
            self._is_word(i, "a", True) or self._is_word(i, "p", True)
        ):
            return False  # am/pm with hour > 12: Python reproduces the error

        if self._is_word(i, "of", True):
            return False  # 'of' year marking lives in the Python reference
        if (
            self._is_word(i, "and", True)
            or self._is_word(i, "at", True)
            or self._is_word(i, "on", True)
            or self._is_word(i, "ad", True)
            or self._is_word(i, "t", True)
        ):
            return True  # jump word

        if self.hour >= 0:
            var is_utc = self._is_word(i, "UTC", False) or self._is_word(i, "GMT", False)
            var is_z = n == 1 and _lower(self.data[s]) == 122
            if is_utc or is_z:
                self.tz_utczero = True
                if self._attached_signed(i):
                    var ns = self.tstart[i + 1]
                    var ne = self.tend[i + 1]
                    var off = _offset_body(self.data, ns + 1, ne)
                    if off < 0:
                        return False
                    if self.data[ns] == 43:
                        off = -off  # POSIX flip
                    self.tzoffset = off
                    self.tkind[i + 1] = 3
                return True
        return False

    def _eat_dot(mut self, i: Int):
        if i + 1 < len(self.tkind):
            if (
                self.tkind[i + 1] == 0
                and self.tend[i + 1] - self.tstart[i + 1] == 1
                and self.data[self.tstart[i + 1]] == 46
            ):
                self.tkind[i + 1] = 3

    def _month_of(self, i: Int) -> Int32:
        if self._is_word(i, "jan", True) or self._is_word(i, "january", True):
            return 1
        if self._is_word(i, "feb", True) or self._is_word(i, "february", True):
            return 2
        if self._is_word(i, "mar", True) or self._is_word(i, "march", True):
            return 3
        if self._is_word(i, "apr", True) or self._is_word(i, "april", True):
            return 4
        if self._is_word(i, "may", True):
            return 5
        if self._is_word(i, "jun", True) or self._is_word(i, "june", True):
            return 6
        if self._is_word(i, "jul", True) or self._is_word(i, "july", True):
            return 7
        if self._is_word(i, "aug", True) or self._is_word(i, "august", True):
            return 8
        if (
            self._is_word(i, "sep", True)
            or self._is_word(i, "sept", True)
            or self._is_word(i, "september", True)
        ):
            return 9
        if self._is_word(i, "oct", True) or self._is_word(i, "october", True):
            return 10
        if self._is_word(i, "nov", True) or self._is_word(i, "november", True):
            return 11
        if self._is_word(i, "dec", True) or self._is_word(i, "december", True):
            return 12
        return 0

    def _weekday_of(self, i: Int) -> Int32:
        if self._is_word(i, "mon", True) or self._is_word(i, "monday", True):
            return 0
        if self._is_word(i, "tue", True) or self._is_word(i, "tuesday", True):
            return 1
        if self._is_word(i, "wed", True) or self._is_word(i, "wednesday", True):
            return 2
        if self._is_word(i, "thu", True) or self._is_word(i, "thursday", True):
            return 3
        if self._is_word(i, "fri", True) or self._is_word(i, "friday", True):
            return 4
        if self._is_word(i, "sat", True) or self._is_word(i, "saturday", True):
            return 5
        if self._is_word(i, "sun", True) or self._is_word(i, "sunday", True):
            return 6
        return -1

    # ------------------------------------------------------------ resolution

    def fields(self, defaults: I32Ptr, dst: I32Ptr):
        var y: Int32 = -1
        var mo: Int32 = -1
        var d: Int32 = -1
        var nymd = len(self.yv)
        if self.month != 0:
            mo = self.month
            if nymd == 1:
                var v = self.yv[0]
                var nd = self.ynd[0]
                if v > 99:
                    y = v
                elif v > 31:
                    y = _pivot(v, self.pivot_base)
                else:
                    d = v
            elif nymd == 2:
                var v1 = self.yv[0]
                var v2 = self.yv[1]
                var nd1 = self.ynd[0]
                var nd2 = self.ynd[1]
                var y1 = v1 > 31
                var y2 = v2 > 31
                if y1 and not y2:
                    d = v2
                    y = v1 if nd1 == 4 else _pivot(v1, self.pivot_base)
                elif y2 and not y1:
                    d = v1
                    y = v2 if nd2 == 4 else _pivot(v2, self.pivot_base)
                elif self.ytok[0] == self.ytok[1]:
                    # Same composite token ("2015-15-May"): year, then day.
                    y = v1 if nd1 == 4 else _pivot(v1, self.pivot_base)
                    d = v2
                elif self.yearfirst and self.ytok[0] < self.month_tok:
                    y = v1 if nd1 == 4 else _pivot(v1, self.pivot_base)
                    d = v2
                else:
                    # Any number after the month -> first is the day; both
                    # before it -> the last is ("8 25 Jan" -> day 25).
                    var mi = self.month_tok
                    if self.ytok[0] > mi or self.ytok[1] > mi:
                        d = v1
                        y = v2 if nd2 == 4 else _pivot(v2, self.pivot_base)
                    else:
                        d = v2
                        y = v1 if nd1 == 4 else _pivot(v1, self.pivot_base)
        elif nymd == 1:
            var v = self.yv[0]
            var nd = self.ynd[0]
            if nd >= 3 or v > 99:
                y = v
            elif v > 31:
                y = _pivot(v, self.pivot_base)
            else:
                d = v
        elif nymd == 2:
            var v1 = self.yv[0]
            var v2 = self.yv[1]
            var nd1 = self.ynd[0]
            var nd2 = self.ynd[1]
            var same2 = self.ytok[0] == self.ytok[1]
            if (nd1 == 4 and same2) or v1 > 31:
                y = v1 if (nd1 == 4 and same2) else _pivot(v1, self.pivot_base)
                mo = v2
            elif (nd2 == 4 and same2) or v2 > 31:
                y = v2 if (nd2 == 4 and same2) else _pivot(v2, self.pivot_base)
                mo = v1
            elif self.dayfirst:
                d = v1
                mo = v2
            else:
                mo = v1
                d = v2
        elif nymd == 3:
            var v1 = self.yv[0]
            var v2 = self.yv[1]
            var v3 = self.yv[2]
            var nd1 = self.ynd[0]
            var nd3 = self.ynd[2]
            var same_tok = self.ytok[0] == self.ytok[1] and self.ytok[1] == self.ytok[2]
            if (nd1 == 4 and same_tok) or v1 > 31:
                y = v1 if (nd1 == 4 and same_tok) else _pivot(v1, self.pivot_base)
                if self.dayfirst:
                    mo = v3
                    d = v2
                    if mo > 12 and d <= 12:
                        mo = v2
                        d = v3
                else:
                    mo = v2
                    d = v3
            elif (nd3 == 4 and same_tok) or v3 > 31:
                y = v3 if (nd3 == 4 and same_tok) else _pivot(v3, self.pivot_base)
                if v1 > 12:
                    mo = v2
                    d = v1
                elif v2 > 12:
                    mo = v1
                    d = v2
                elif self.dayfirst:
                    mo = v2
                    d = v1
                else:
                    mo = v1
                    d = v2
            elif self.yearfirst:
                y = _pivot(v1, self.pivot_base)
                if self.dayfirst:
                    mo = v3
                    d = v2
                else:
                    mo = v2
                    d = v3
            elif v1 > 12:
                d = v1
                mo = v2
                y = _pivot(v3, self.pivot_base)
            elif v2 > 12:
                mo = v1
                d = v2
                y = _pivot(v3, self.pivot_base)
            elif self.dayfirst:
                d = v1
                mo = v2
                y = _pivot(v3, self.pivot_base)
            else:
                mo = v1
                d = v2
                y = _pivot(v3, self.pivot_base)

        dst[1] = y if y >= 0 else defaults[0]
        dst[2] = mo if mo >= 0 else defaults[1]
        dst[3] = d if d >= 0 else defaults[2]
        dst[4] = self.hour if self.hour >= 0 else defaults[3]
        dst[5] = self.minute if self.minute >= 0 else defaults[4]
        dst[6] = self.second if self.second >= 0 else defaults[5]
        dst[7] = self.micro if self.micro >= 0 else defaults[6]
        var tz = self.tzoffset
        if tz == UNSET and self.tz_utczero:
            tz = 0
        dst[8] = tz


# -------------------------------------------------------------------- entry

def _parse_one(
    mut p: Parser,
    data: U8Ptr,
    n: Int,
    flags: UInt32,
    defaults: I32Ptr,
    dst: I32Ptr,
):
    dst[0] = 1
    p.reset(data, n)
    if not p.tokenize():
        return
    var i = 0
    while i < len(p.tkind):
        if p.tkind[i] == 0:
            if not p.handle_number(i):
                return
        elif p.tkind[i] == 1:
            if not p.handle_alpha(i):
                return
        i += 1
    if p.weekday >= 0 and not p.day_explicit:
        return  # weekday arithmetic lives in the Python reference
    if len(p.yv) == 0 and p.month == 0 and p.hour < 0 and p.weekday < 0:
        return  # "String does not contain a date" (Python raises it exactly)
    p.fields(defaults, dst)
    dst[0] = 0


@export
def dateutilmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def dateutilmojo_parse_batch(
    data: U8Ptr,
    offsets: I64Ptr,
    n: Int64,
    flags: UInt32,
    defaults: I32Ptr,
    dst: I32Ptr,
    dst_len: Int64,
) abi("C") -> Int32:
    if n < 0 or dst_len < n * 9:
        return 2
    var p = Parser(data, 0, flags, defaults[7])
    for i in range(Int(n)):
        var start = offsets[i]
        var end = offsets[i + 1]
        if start < 0 or end < start:
            return 2
        _parse_one(
            p,
            data + Int(start),
            Int(end - start),
            flags,
            defaults,
            dst + i * 9,
        )
    return 0
