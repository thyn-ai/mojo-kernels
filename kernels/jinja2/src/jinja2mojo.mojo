"""Clean-room jinja2 template lexer (structural scan only).

Written fresh from the jinja2 template-language documentation plus black-box
behavioural probes of the reference package (PyPI jinja2 3.1.6): the oracle
was fed inputs and its token stream observed; no oracle source was read or
adapted.

The kernel performs the per-character scan that dominates the reference
lexer's runtime: it splits template bytes into data / expression tokens,
tracks logical line numbers (`\\r\\n`, `\\r` and `\\n` each count once),
matches strings/numbers/names/operators, and enforces bracket balance.
Value semantics (string unescaping, integer/float conversion, whitespace
trimming, newline normalisation) are computed by the Python wrapper, which
also re-validates names; anything outside the probed subset (raw blocks,
unterminated constructs, balance errors, unknown characters) makes the
kernel return STATUS_FALLBACK so the wrapper reruns the reference lexer.

Exported C ABI (v1) — one whole template per call, all data through
caller-owned buffers:

    int32_t jinja2mojo_abi_version(void)
    int32_t jinja2mojo_scan(const uint8_t* src, int64_t src_len,
                            uint8_t* out_tokens, int64_t out_cap,
                            int64_t* out_count)

`out_tokens` is an array of 24-byte little-endian records:

    offset  size  field
    0       1     kind (1=data 2=name 3=integer 4=float 5=string 6=operator
                        7=block_begin 8=block_end 9=variable_begin
                        10=variable_end 11=comment)
    1       1     aux: begin/end sign (0 none, 43 '+', 45 '-'); string quote
    2       1     aux2: comment end sign (0 none, 43 '+', 45 '-')
    3       1     reserved (0)
    4       4     lineno (1-based, logical newlines)
    8       4     byte offset of the value slice in `src`
    12      4     byte length of the value slice
    16      8     reserved (0)

Comment records describe the consumed comment (their signs drive the
wrapper's whitespace trimming); they are dropped before the token stream
is built.

Statuses: 0 = tokens written, 1 = fall back to the reference lexer,
2 = `out_cap` too small (`out_count` holds the record count needed; retry).
"""

from std.memory import Pointer
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]

comptime STATUS_OK: Int32 = 0
comptime STATUS_FALLBACK: Int32 = 1
comptime STATUS_OVERFLOW: Int32 = 2

# Token kinds (byte 0 of a record).
comptime K_DATA: UInt8 = 1
comptime K_NAME: UInt8 = 2
comptime K_INTEGER: UInt8 = 3
comptime K_FLOAT: UInt8 = 4
comptime K_STRING: UInt8 = 5
comptime K_OPERATOR: UInt8 = 6
comptime K_BLOCK_BEGIN: UInt8 = 7
comptime K_BLOCK_END: UInt8 = 8
comptime K_VARIABLE_BEGIN: UInt8 = 9
comptime K_VARIABLE_END: UInt8 = 10
comptime K_COMMENT: UInt8 = 11

comptime RECORD_SIZE: Int = 24

# Byte constants.
comptime CH_LF: UInt8 = 0x0A
comptime CH_CR: UInt8 = 0x0D
comptime CH_TAB: UInt8 = 0x09
comptime CH_FF: UInt8 = 0x0C
comptime CH_VT: UInt8 = 0x0B
comptime CH_SPACE: UInt8 = 0x20
comptime CH_LBRACE: UInt8 = 0x7B  # {
comptime CH_RBRACE: UInt8 = 0x7D  # }
comptime CH_LBRACKET: UInt8 = 0x5B  # [
comptime CH_RBRACKET: UInt8 = 0x5D  # ]
comptime CH_LPAREN: UInt8 = 0x28  # (
comptime CH_RPAREN: UInt8 = 0x29  # )
comptime CH_PERCENT: UInt8 = 0x25  # %
comptime CH_HASH: UInt8 = 0x23  # #
comptime CH_MINUS: UInt8 = 0x2D  # -
comptime CH_PLUS: UInt8 = 0x2B  # +
comptime CH_SQUOTE: UInt8 = 0x27  # '
comptime CH_DQUOTE: UInt8 = 0x22  # "
comptime CH_BACKSLASH: UInt8 = 0x5C  # \
comptime CH_UNDERSCORE: UInt8 = 0x5F  # _
comptime CH_DOT: UInt8 = 0x2E  # .
comptime CH_ZERO: UInt8 = 0x30  # 0
comptime CH_NINE: UInt8 = 0x39  # 9


def _is_digit(b: UInt8) -> Bool:
    return b >= CH_ZERO and b <= CH_NINE


def _is_hex(b: UInt8) -> Bool:
    return (
        (b >= CH_ZERO and b <= CH_NINE)
        or (b >= 0x41 and b <= 0x46)  # A-F
        or (b >= 0x61 and b <= 0x66)  # a-f
    )


def _is_base_digit(b: UInt8, base: Int32) -> Bool:
    if base == 2:
        return b == 0x30 or b == 0x31
    if base == 8:
        return b >= 0x30 and b <= 0x37
    return _is_hex(b)


def _is_name_start(b: UInt8) -> Bool:
    # ASCII letters and underscore, plus any non-ASCII byte (the wrapper
    # re-validates the decoded name against the reference's word semantics).
    return (
        (b >= 0x41 and b <= 0x5A)
        or (b >= 0x61 and b <= 0x7A)
        or b == CH_UNDERSCORE
        or b >= 0x80
    )


def _is_name_char(b: UInt8) -> Bool:
    return _is_name_start(b) or _is_digit(b)


def _is_expr_ws(b: UInt8) -> Bool:
    return (
        b == CH_SPACE
        or b == CH_TAB
        or b == CH_LF
        or b == CH_CR
        or b == CH_FF
        or b == CH_VT
    )


struct Scanner:
    """Per-call scanner state over caller-owned source bytes.

    `count` always tracks the full record count, even past `cap`: a call
    that overflows still finishes the scan so the caller can retry with
    exactly `count` records of room.
    """

    var src: U8Ptr
    var n: Int
    var pos: Int
    var lineno: UInt32
    var outp: U8Ptr
    var cap: Int
    var count: Int
    var balance: List[UInt8]
    var failed: Bool

    def __init__(out self, src: U8Ptr, n: Int, outp: U8Ptr, cap: Int):
        self.src = src
        self.n = n
        self.pos = 0
        self.lineno = 1
        self.outp = outp
        self.cap = cap
        self.count = 0
        self.balance = List[UInt8]()
        self.failed = False

    def byte(self, i: Int) -> UInt8:
        return self.src[unsafe_offset=i]

    def emit(mut self, kind: UInt8, aux: UInt8, aux2: UInt8, lineno: UInt32, off: Int, length: Int):
        """Append one 24-byte little-endian record (skipped past capacity)."""
        if self.count >= self.cap:
            self.count += 1
            return
        var base = self.count * RECORD_SIZE
        self.outp[unsafe_offset=base] = kind
        self.outp[unsafe_offset=base + 1] = aux
        self.outp[unsafe_offset=base + 2] = aux2
        self.outp[unsafe_offset=base + 3] = 0
        var v = lineno
        for k in range(4):
            self.outp[unsafe_offset=base + 4 + k] = UInt8(v & 0xFF)
            v >>= 8
        var o = UInt32(off)
        for k in range(4):
            self.outp[unsafe_offset=base + 8 + k] = UInt8(o & 0xFF)
            o >>= 8
        var l = UInt32(length)
        for k in range(4):
            self.outp[unsafe_offset=base + 12 + k] = UInt8(l & 0xFF)
            l >>= 8
        for k in range(16, 24):
            self.outp[unsafe_offset=base + k] = 0
        self.count += 1

    def count_newline(mut self):
        """Advance over one logical newline at self.pos (\\r\\n, \\r or \\n)."""
        if self.byte(self.pos) == CH_CR:
            self.pos += 1
            if self.pos < self.n and self.byte(self.pos) == CH_LF:
                self.pos += 1
        else:
            self.pos += 1
        self.lineno += 1

    def scan_data(mut self):
        """DATA mode: consume literal text up to the next tag open or EOF."""
        var start = self.pos
        var start_lineno = self.lineno
        while self.pos < self.n:
            var b = self.byte(self.pos)
            if b == CH_LBRACE and self.pos + 1 < self.n:
                var n2 = self.byte(self.pos + 1)
                if n2 == CH_LBRACE or n2 == CH_PERCENT or n2 == CH_HASH:
                    break
            if b == CH_LF or b == CH_CR:
                self.count_newline()
            else:
                self.pos += 1
        if self.pos > start:
            self.emit(K_DATA, 0, 0, start_lineno, start, self.pos - start)
        if self.pos >= self.n:
            return
        var n2 = self.byte(self.pos + 1)
        if n2 == CH_LBRACE:
            self.scan_tag(K_VARIABLE_BEGIN, K_VARIABLE_END, CH_RBRACE, False)
        elif n2 == CH_PERCENT:
            self.scan_tag(K_BLOCK_BEGIN, K_BLOCK_END, CH_PERCENT, True)
        else:
            self.scan_comment()

    def scan_tag(mut self, begin_kind: UInt8, end_kind: UInt8, end_ch: UInt8, is_block: Bool):
        """Consume a `{{`/`{%` open and its whole expression through the close."""
        var open_lineno = self.lineno
        var sign: UInt8 = 0
        var j = self.pos + 2
        if j < self.n and (self.byte(j) == CH_MINUS or self.byte(j) == CH_PLUS):
            sign = self.byte(j)
            j += 1
        self.emit(begin_kind, sign, 0, open_lineno, self.pos, j - self.pos)
        self.pos = j
        var first = True
        while self.pos < self.n and not self.failed:
            var b = self.byte(self.pos)
            if b == CH_LF or b == CH_CR:
                self.count_newline()
                continue
            if _is_expr_ws(b):
                self.pos += 1
                continue
            # End markers are markers only at bracket depth zero.
            if len(self.balance) == 0:
                if b == end_ch and self.pos + 1 < self.n and self.byte(self.pos + 1) == CH_RBRACE:
                    self.emit(end_kind, 0, 0, self.lineno, self.pos, 2)
                    self.pos += 2
                    return
                # Signed end markers. The '+' form exists for blocks only:
                # probed from the oracle, '{{ x +}}' lexes as add + '}}'.
                if b == CH_MINUS or (b == CH_PLUS and is_block):
                    if (
                        self.pos + 2 < self.n
                        and self.byte(self.pos + 1) == end_ch
                        and self.byte(self.pos + 2) == CH_RBRACE
                    ):
                        self.emit(end_kind, b, 0, self.lineno, self.pos, 3)
                        self.pos += 3
                        return
                if b == CH_RBRACE:
                    self.failed = True  # bare '}' at depth 0: reference errors
                    return
            if b == CH_SQUOTE or b == CH_DQUOTE:
                self.scan_string(b)
                first = False
                continue
            if _is_digit(b):
                self.scan_number()
                first = False
                continue
            if _is_name_start(b):
                self.scan_name(is_block and first)
                first = False
                continue
            self.scan_operator()
            first = False
        if not self.failed and len(self.balance) > 0:
            self.failed = True  # unbalanced bracket at EOF: reference errors

    def scan_string(mut self, quote: UInt8):
        var start = self.pos
        var tok_lineno = self.lineno
        var i = self.pos + 1
        while i < self.n:
            var b = self.byte(i)
            if b == CH_BACKSLASH:
                if i + 1 >= self.n:
                    self.failed = True  # dangling backslash: reference errors
                    return
                var nb = self.byte(i + 1)
                if nb == CH_LF:
                    self.lineno += 1
                elif nb == CH_CR:
                    self.lineno += 1
                    if i + 2 < self.n and self.byte(i + 2) == CH_LF:
                        i += 1
                i += 2
                continue
            if b == CH_LF:
                self.lineno += 1
            elif b == CH_CR:
                self.lineno += 1
                if i + 1 < self.n and self.byte(i + 1) == CH_LF:
                    i += 1
            elif b == quote:
                # Value slice is the raw content between the quotes.
                self.emit(K_STRING, quote, 0, tok_lineno, start + 1, i - start - 1)
                self.pos = i + 1
                return
            i += 1
        self.failed = True  # unterminated string: reference errors

    def scan_number(mut self):
        var start = self.pos
        var tok_lineno = self.lineno
        if self.byte(start) == CH_ZERO and start + 1 < self.n:
            var p = self.byte(start + 1)
            if p == 0x78 or p == 0x58:  # x X
                self.scan_based(start, tok_lineno, 16)
                return
            if p == 0x62 or p == 0x42:  # b B
                self.scan_based(start, tok_lineno, 2)
                return
            if p == 0x6F or p == 0x4F:  # o O
                self.scan_based(start, tok_lineno, 8)
                return
        var j = self.scan_digits(start)
        var is_float = False
        if (
            j < self.n
            and self.byte(j) == CH_DOT
            and j + 1 < self.n
            and _is_digit(self.byte(j + 1))
        ):
            is_float = True
            j = self.scan_digits(j + 1)
        if j < self.n and (self.byte(j) == 0x65 or self.byte(j) == 0x45):  # e E
            var k = j + 1
            if k < self.n and (self.byte(k) == CH_PLUS or self.byte(k) == CH_MINUS):
                k += 1
            if k < self.n and _is_digit(self.byte(k)):
                is_float = True
                j = self.scan_digits(k)
        if is_float:
            # Float mantissas allow leading zeros (probed: '00.5', '01e2').
            self.emit(K_FLOAT, 0, 0, tok_lineno, start, j - start)
            self.pos = j
            return
        if self.byte(start) == CH_ZERO:
            # Decimal integers follow Python-literal rules (probed): a zero
            # run ('0', '00', '0_0', '000_000') is one token, but a nonzero
            # digit after the run starts a new token ('042' -> 0, 42).
            j = start + 1
            while j < self.n:
                if self.byte(j) == CH_ZERO:
                    j += 1
                elif (
                    self.byte(j) == CH_UNDERSCORE
                    and j + 1 < self.n
                    and self.byte(j + 1) == CH_ZERO
                ):
                    j += 2
                else:
                    break
            self.emit(K_INTEGER, 0, 0, tok_lineno, start, j - start)
            self.pos = j
            return
        self.emit(K_INTEGER, 0, 0, tok_lineno, start, j - start)
        self.pos = j

    def scan_digits(mut self, j0: Int) -> Int:
        """Scan digit (digit | '_' digit)* starting at j0; return the end index."""
        var j = j0
        while j < self.n:
            var b = self.byte(j)
            if _is_digit(b):
                j += 1
            elif b == CH_UNDERSCORE and j + 1 < self.n and _is_digit(self.byte(j + 1)):
                j += 2
            else:
                break
        return j

    def scan_based(mut self, start: Int, tok_lineno: UInt32, base: Int32):
        """0-prefixed integer; degrades to a decimal '0' when no digit follows."""
        var j = start + 2
        if j < self.n and self.byte(j) == CH_UNDERSCORE:
            j += 1
        if j < self.n and _is_base_digit(self.byte(j), base):
            j += 1
            while j < self.n:
                var b = self.byte(j)
                if _is_base_digit(b, base):
                    j += 1
                elif b == CH_UNDERSCORE and j + 1 < self.n and _is_base_digit(self.byte(j + 1), base):
                    j += 2
                else:
                    break
            self.emit(K_INTEGER, 0, 0, tok_lineno, start, j - start)
            self.pos = j
            return
        # No valid digit after the prefix: just the leading '0' is a number.
        self.emit(K_INTEGER, 0, 0, tok_lineno, start, 1)
        self.pos = start + 1

    def scan_name(mut self, raw_check: Bool):
        var start = self.pos
        var tok_lineno = self.lineno
        var j = self.pos + 1
        while j < self.n and _is_name_char(self.byte(j)):
            j += 1
        if raw_check and j - start == 3:
            if (
                self.byte(start) == 0x72  # r
                and self.byte(start + 1) == 0x61  # a
                and self.byte(start + 2) == 0x77  # w
            ):
                # `{% raw %}` switches the reference lexer into raw-text mode,
                # which is outside this kernel's subset: fall back.
                self.failed = True
                return
        self.emit(K_NAME, 0, 0, tok_lineno, start, j - start)
        self.pos = j

    def scan_operator(mut self):
        var b = self.byte(self.pos)
        var tok_lineno = self.lineno
        if self.pos + 1 < self.n:
            var n2 = self.byte(self.pos + 1)
            var two = (
                (b == 0x2A and n2 == 0x2A)  # **
                or (b == 0x2F and n2 == 0x2F)  # //
                or (b == 0x3D and n2 == 0x3D)  # ==
                or (b == 0x21 and n2 == 0x3D)  # !=
                or (b == 0x3E and n2 == 0x3D)  # >=
                or (b == 0x3C and n2 == 0x3D)  # <=
            )
            if two:
                self.emit(K_OPERATOR, 0, 0, tok_lineno, self.pos, 2)
                self.pos += 2
                return
        if b == CH_LPAREN or b == CH_LBRACKET or b == CH_LBRACE:
            self.balance.append(b)
            self.emit(K_OPERATOR, 0, 0, tok_lineno, self.pos, 1)
            self.pos += 1
            return
        if b == CH_RPAREN or b == CH_RBRACKET or b == CH_RBRACE:
            if len(self.balance) == 0:
                self.failed = True  # unmatched closer: reference errors
                return
            var top = self.balance[len(self.balance) - 1]
            var ok = (
                (b == CH_RPAREN and top == CH_LPAREN)
                or (b == CH_RBRACKET and top == CH_LBRACKET)
                or (b == CH_RBRACE and top == CH_LBRACE)
            )
            if not ok:
                self.failed = True  # mismatched closer: reference errors
                return
            _ = self.balance.pop()
            self.emit(K_OPERATOR, 0, 0, tok_lineno, self.pos, 1)
            self.pos += 1
            return
        var single = (
            b == CH_PLUS
            or b == CH_MINUS
            or b == 0x2A  # *
            or b == 0x2F  # /
            or b == CH_PERCENT
            or b == 0x3D  # =
            or b == 0x7E  # ~
            or b == 0x3A  # :
            or b == CH_DOT
            or b == 0x2C  # ,
            or b == 0x3B  # ;
            or b == 0x7C  # |
            or b == 0x3C  # <
            or b == 0x3E  # >
        )
        if single:
            self.emit(K_OPERATOR, 0, 0, tok_lineno, self.pos, 1)
            self.pos += 1
            return
        self.failed = True  # unknown character: reference errors

    def scan_comment(mut self):
        """Consume `{# ... #}` (signs included) and record it for trimming."""
        var open_lineno = self.lineno
        var sign_b: UInt8 = 0
        var j = self.pos + 2
        if j < self.n and (self.byte(j) == CH_MINUS or self.byte(j) == CH_PLUS):
            sign_b = self.byte(j)
            j += 1
        var start = self.pos
        self.pos = j
        var i = j
        while i < self.n:
            var b = self.byte(i)
            if b == CH_HASH and i + 1 < self.n and self.byte(i + 1) == CH_RBRACE:
                var sign_e: UInt8 = 0
                if i > j and (self.byte(i - 1) == CH_MINUS or self.byte(i - 1) == CH_PLUS):
                    sign_e = self.byte(i - 1)
                self.emit(K_COMMENT, sign_b, sign_e, open_lineno, start, i + 2 - start)
                self.pos = i + 2
                return
            if b == CH_LF:
                self.lineno += 1
            elif b == CH_CR:
                self.lineno += 1
                if i + 1 < self.n and self.byte(i + 1) == CH_LF:
                    i += 1
            i += 1
        self.failed = True  # missing end of comment: reference errors


@export
def jinja2mojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def jinja2mojo_scan(
    src: U8Ptr,
    src_len: Int64,
    out_tokens: U8Ptr,
    out_cap: Int64,
    out_count: I64Ptr,
) abi("C") -> Int32:
    """Scan one template into 24-byte token records.

    Returns STATUS_OK with `out_count` records written, STATUS_FALLBACK for
    anything outside the kernel's probed subset (the wrapper then reruns the
    reference lexer, which also reproduces every lex-time error), or
    STATUS_OVERFLOW when `out_cap` is too small (`out_count` then holds the
    record count the caller must provide room for on retry).
    """
    out_count[unsafe_offset=0] = 0
    if src_len < 0 or out_cap < 0:
        return STATUS_FALLBACK
    if src_len == 0:
        return STATUS_OK
    var n = Int(src_len)
    var s = Scanner(src, n, out_tokens, Int(out_cap))
    while s.pos < n and not s.failed:
        s.scan_data()
    if s.failed:
        out_count[unsafe_offset=0] = 0
        return STATUS_FALLBACK
    out_count[unsafe_offset=0] = Int64(s.count)
    if s.count > s.cap:
        return STATUS_OVERFLOW
    return STATUS_OK
