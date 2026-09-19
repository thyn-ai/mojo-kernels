"""Clean-room JMESPath evaluation kernel.

Written fresh from the JMESPath language specification (jmespath.org) plus
black-box behavioural probes of the reference package (PyPI jmespath 1.0.1).
No third-party code is used or adapted.

Exported C ABI (v1) — one whole search per call, all data through caller-owned
or kernel-owned buffers:

    int32_t jmespathmojo_abi_version(void)
    int32_t jmespathmojo_search(const char* expr, int64_t expr_len,
                                const char* data, int64_t data_len,
                                char** out_buf, int64_t* out_len)
    void    jmespathmojo_free(char* buf, int64_t len)

`data` is the document serialized with CPython marshal format v4 (the
Python wrapper passes `marshal.dumps(data, 4)` output — a binary walk that
preserves exact Python types). The result is a freshly allocated buffer
holding the result serialized as JSON (floats formatted with
Python-repr-compatible shortest round-trip digits, so `json.loads`
reproduces bit-identical values).

Status codes: 0 = result ready. Any non-zero status tells the wrapper the
kernel cannot answer this call with reference-identical semantics (invalid
expression, evaluation/type error, an in-spec construct outside the kernel's
implemented subset, or data outside the kernel's JSON subset); the wrapper
then recomputes with its vendored pure-Python implementation, which mirrors
the reference exactly (including raising the same error classes).

Semantics are evaluated over an index-based node arena: nodes reference
children by arena index, so field access, projections and sort_by share
subtrees instead of copying them. Float accumulation (sum/avg) is sequential
in document order — the same operation order as the reference — so results
are bit-identical, never reassociated.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

comptime ABI_VERSION: Int32 = 1

# Call statuses (non-zero => wrapper recomputes with the vendored fallback).
comptime STATUS_OK: Int32 = 0
comptime STATUS_EXPR: Int32 = 1  # expression rejected by the kernel parser
comptime STATUS_EVAL: Int32 = 2  # evaluation-time type/arity/domain error
comptime STATUS_UNSUPPORTED: Int32 = 3  # valid construct outside kernel subset
comptime STATUS_DATA: Int32 = 4  # data outside the kernel's JSON subset

# Node tags.
comptime T_NULL: Int32 = 0
comptime T_FALSE: Int32 = 1
comptime T_TRUE: Int32 = 2
comptime T_INT: Int32 = 3
comptime T_FLOAT: Int32 = 4
comptime T_STR: Int32 = 5
comptime T_ARR: Int32 = 6
comptime T_OBJ: Int32 = 7
comptime T_TUPLE: Int32 = 8  # Python tuple: NOT an array for JMESPath

# Shared well-known nodes (created first in every arena).
comptime NULL_IDX: Int32 = 0
comptime FALSE_IDX: Int32 = 1
comptime TRUE_IDX: Int32 = 2

# Int64 bounds as literals (Mojo has no Int64.MAX constant in scope here).
comptime I64_MAX: Int64 = 9223372036854775807
comptime I64_MIN: Int64 = -9223372036854775807 - 1

# 2^53: above this magnitude int64->float64 conversion loses exactness; the
# kernel punts mixed int/float comparisons and exact int avg division past it
# (the wrapper's fallback computes those calls exactly, like the reference).
comptime EXACT_INT_LIMIT: Float64 = 9007199254740992.0

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U8PtrPtr = Pointer[U8Ptr, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


struct Node(Copyable, Movable):
    """One JSON value in the arena. Arrays/objects reference children by
    arena index (kids), so subtrees are shared, never deep-copied."""

    var tag: Int32
    var i: Int64  # T_INT payload
    var f: Float64  # T_FLOAT payload
    var s: String  # T_STR payload (UTF-8)
    var kids: List[Int32]  # T_ARR: element indices; T_OBJ: value indices
    var keys: List[String]  # T_OBJ: key names, parallel to kids

    def __init__(
        out self,
        tag: Int32,
        i: Int64 = 0,
        f: Float64 = 0.0,
        s: String = String(),
        var kids: List[Int32] = List[Int32](),
        var keys: List[String] = List[String](),
    ):
        self.tag = tag
        self.i = i
        self.f = f
        self.s = s
        self.kids = kids^
        self.keys = keys^


def is_number_tag(tag: Int32) -> Bool:
    return tag == T_INT or tag == T_FLOAT


def new_arena() -> List[Node]:
    var arena = List[Node]()
    arena.append(Node(T_NULL))
    arena.append(Node(T_FALSE))
    arena.append(Node(T_TRUE))
    return arena^


def add_int(mut arena: List[Node], value: Int64) -> Int32:
    arena.append(Node(T_INT, i=value))
    return Int32(len(arena) - 1)


def add_float(mut arena: List[Node], value: Float64) -> Int32:
    arena.append(Node(T_FLOAT, f=value))
    return Int32(len(arena) - 1)


def add_str(mut arena: List[Node], value: String) -> Int32:
    arena.append(Node(T_STR, s=value))
    return Int32(len(arena) - 1)


def add_arr(mut arena: List[Node], var kids: List[Int32]) -> Int32:
    arena.append(Node(T_ARR, kids=kids^))
    return Int32(len(arena) - 1)


def add_obj(
    mut arena: List[Node], var keys: List[String], var kids: List[Int32]
) -> Int32:
    arena.append(Node(T_OBJ, kids=kids^, keys=keys^))
    return Int32(len(arena) - 1)


def bool_idx(value: Bool) -> Int32:
    return TRUE_IDX if value else FALSE_IDX

struct NumSum:
    """CPython 3.12 builtin sum() semantics, bit-for-bit: ints accumulate as
    an exact integer (arbitrary precision upstream; int64 here with an
    overflow punt); the first float switches to float summation with the
    generic int->float conversion; float items use Neumaier compensation
    while later int items are added naively (matching bltinmodule.c)."""

    var seen_float: Bool
    var iacc: Int64
    var f: Float64
    var c: Float64
    var err: Int32

    def __init__(out self):
        self.seen_float = False
        self.iacc = 0
        self.f = 0.0
        self.c = 0.0
        self.err = STATUS_OK

    def add(mut self, tag: Int32, i: Int64, x: Float64):
        if tag == T_INT:
            if not self.seen_float:
                if i > 0 and self.iacc > I64_MAX - i:
                    self.err = STATUS_UNSUPPORTED
                    return
                if i < 0 and self.iacc < I64_MIN - i:
                    self.err = STATUS_UNSUPPORTED
                    return
                self.iacc += i
            else:
                # ints added in the float phase get no compensation (CPython)
                self.f += Float64(i)
        elif tag == T_FLOAT:
            if not self.seen_float:
                self.seen_float = True
                # generic PyNumber_Add transition: float(int_acc) + x
                self.f = Float64(self.iacc) + x
            else:
                var t = self.f + x
                var af = self.f
                if af < 0.0:
                    af = -af
                var ax = x
                if ax < 0.0:
                    ax = -ax
                if af >= ax:
                    self.c += (self.f - t) + x
                else:
                    self.c += (x - t) + self.f
                self.f = t
        else:
            self.err = STATUS_EVAL

    def final(self) -> Float64:
        # CPython adds the compensation only when it is truthy and finite.
        if self.c != 0.0 and self.c == self.c and (
            self.c <= 1.7976931348623157e308 and self.c >= -1.7976931348623157e308
        ):
            return self.f + self.c
        return self.f


def make_nan() -> Float64:
    var p = unsafe_alloc[Float64](1)
    var u = p.unsafe_bitcast[UInt64]()
    u[unsafe_offset=0] = 0x7FF8000000000000
    var v = p[unsafe_offset=0]
    p.unsafe_free()
    return v


def make_inf(negative: Bool) -> Float64:
    var p = unsafe_alloc[Float64](1)
    var u = p.unsafe_bitcast[UInt64]()
    u[unsafe_offset=0] = 0xFFF0000000000000 if negative else 0x7FF0000000000000
    var v = p[unsafe_offset=0]
    p.unsafe_free()
    return v


# ---------------------------------------------------------------------------
# JSON parser (Python json.loads semantics: NaN/Infinity/-Infinity accepted;
# duplicate object keys keep first position with the last value).
# ---------------------------------------------------------------------------


struct JsonParser[origin: Origin]:
    """Recursive-descent JSON parser over a borrowed byte buffer.

    On any deviation from the subset the kernel guarantees (invalid JSON,
    integers past int64, invalid UTF-8), `failed` is set and the caller
    reports STATUS_DATA (or STATUS_EXPR for backtick literals), engaging the
    wrapper's fallback — which parses with Python's json and never diverges.
    """

    var buf: Pointer[UInt8, Self.origin]
    var length: Int
    var pos: Int
    var failed: Bool
    var unsupported: Bool  # valid JSON beyond the kernel subset (punt)

    def __init__(out self, buf: Pointer[UInt8, Self.origin], length: Int):
        self.buf = buf
        self.length = length
        self.pos = 0
        self.failed = False
        self.unsupported = False

    def at_end(self) -> Bool:
        return self.pos >= self.length

    def peek(self) -> UInt8:
        if self.pos >= self.length:
            return 0
        return self.buf[unsafe_offset=self.pos]

    def skip_ws(mut self):
        while not self.at_end():
            var c = self.peek()
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                return

    def expect(mut self, c: UInt8) -> Bool:
        if self.peek() == c:
            self.pos += 1
            return True
        self.failed = True
        return False

    def parse_document(mut self, mut arena: List[Node]) -> Int32:
        """Parse one JSON value; requires clean EOF. Returns the root node
        index, or NULL_IDX with self.failed set."""
        self.skip_ws()
        var root = self.parse_value(arena)
        if self.failed:
            return NULL_IDX
        self.skip_ws()
        if not self.at_end():
            self.failed = True
            return NULL_IDX
        return root

    def parse_value(mut self, mut arena: List[Node]) -> Int32:
        if self.at_end():
            self.failed = True
            return NULL_IDX
        var c = self.peek()
        if c == 0x7B:  # {
            return self.parse_object(arena)
        if c == 0x5B:  # [
            return self.parse_array(arena)
        if c == 0x22:  # "
            var s = self.parse_string()
            if self.failed:
                return NULL_IDX
            return add_str(arena, s)
        if c == 0x74:  # true
            return self.parse_lit_word(arena, "true", TRUE_IDX)
        if c == 0x66:  # false
            return self.parse_lit_word(arena, "false", FALSE_IDX)
        if c == 0x6E:  # null
            return self.parse_lit_word(arena, "null", NULL_IDX)
        if c == 0x4E:  # NaN (Python json extension)
            var nan_idx = add_float(arena, make_nan())
            return self.parse_lit_word(arena, "NaN", nan_idx)
        if c == 0x49:  # Infinity (Python json extension)
            var inf_idx = add_float(arena, make_inf(False))
            return self.parse_lit_word(arena, "Infinity", inf_idx)
        if c == 0x2D:  # - : number or -Infinity
            if self.pos + 1 < self.length and self.buf[unsafe_offset=self.pos + 1] == 0x49:
                self.pos += 1
                var ninf_idx = add_float(arena, make_inf(True))
                return self.parse_lit_word(arena, "Infinity", ninf_idx)
            return self.parse_number(arena)
        if (c >= 0x30 and c <= 0x39):
            return self.parse_number(arena)
        self.failed = True
        return NULL_IDX

    def parse_lit_word(
        mut self, mut arena: List[Node], word: String, node: Int32
    ) -> Int32:
        var wb = word.as_bytes()
        for k in range(len(wb)):
            if self.at_end() or self.peek() != wb[k]:
                self.failed = True
                return NULL_IDX
            self.pos += 1
        return node

    def parse_object(mut self, mut arena: List[Node]) -> Int32:
        if not self.expect(0x7B):  # {
            return NULL_IDX
        var keys = List[String]()
        var kids = List[Int32]()
        self.skip_ws()
        if self.peek() == 0x7D:  # }
            self.pos += 1
            return add_obj(arena, keys^, kids^)
        while True:
            self.skip_ws()
            var key = self.parse_string()
            if self.failed:
                return NULL_IDX
            self.skip_ws()
            if not self.expect(0x3A):  # :
                return NULL_IDX
            self.skip_ws()
            var val = self.parse_value(arena)
            if self.failed:
                return NULL_IDX
            # Duplicate keys: last value wins, first position kept (Python).
            var found = -1
            for k in range(len(keys)):
                if keys[k] == key:
                    found = k
                    break
            if found >= 0:
                kids[found] = val
            else:
                keys.append(key)
                kids.append(val)
            self.skip_ws()
            var c = self.peek()
            if c == 0x2C:  # ,
                self.pos += 1
                continue
            if c == 0x7D:  # }
                self.pos += 1
                return add_obj(arena, keys^, kids^)
            self.failed = True
            return NULL_IDX

    def parse_array(mut self, mut arena: List[Node]) -> Int32:
        if not self.expect(0x5B):  # [
            return NULL_IDX
        var kids = List[Int32]()
        self.skip_ws()
        if self.peek() == 0x5D:  # ]
            self.pos += 1
            return add_arr(arena, kids^)
        while True:
            self.skip_ws()
            var val = self.parse_value(arena)
            if self.failed:
                return NULL_IDX
            kids.append(val)
            self.skip_ws()
            var c = self.peek()
            if c == 0x2C:  # ,
                self.pos += 1
                continue
            if c == 0x5D:  # ]
                self.pos += 1
                return add_arr(arena, kids^)
            self.failed = True
            return NULL_IDX

    def parse_hex4(mut self) -> UInt32:
        var v = UInt32(0)
        for _ in range(4):
            if self.at_end():
                self.failed = True
                return 0
            var c = self.peek()
            var d = UInt32(0)
            if c >= 0x30 and c <= 0x39:
                d = UInt32(c - 0x30)
            elif c >= 0x61 and c <= 0x66:
                d = UInt32(c - 0x61 + 10)
            elif c >= 0x41 and c <= 0x46:
                d = UInt32(c - 0x41 + 10)
            else:
                self.failed = True
                return 0
            v = v * 16 + d
            self.pos += 1
        return v

    def parse_string_body(mut self, mut out: String):
        """Scan one JSON string body (after the opening quote) into `out`,
        decoding escapes to UTF-8. Sets failed on invalid input."""
        while True:
            if self.at_end():
                self.failed = True
                return
            var c = self.peek()
            if c == 0x22:  # closing "
                self.pos += 1
                return
            if c == 0x5C:  # backslash escape
                self.pos += 1
                if self.at_end():
                    self.failed = True
                    return
                var e = self.peek()
                self.pos += 1
                if e == 0x22:
                    out.append(Codepoint(0x22))
                elif e == 0x5C:
                    out.append(Codepoint(0x5C))
                elif e == 0x2F:
                    out.append(Codepoint(0x2F))
                elif e == 0x62:
                    out.append(Codepoint(0x08))
                elif e == 0x66:
                    out.append(Codepoint(0x0C))
                elif e == 0x6E:
                    out.append(Codepoint(0x0A))
                elif e == 0x72:
                    out.append(Codepoint(0x0D))
                elif e == 0x74:
                    out.append(Codepoint(0x09))
                elif e == 0x75:  # \uXXXX with surrogate-pair support
                    var cp = self.parse_hex4()
                    if self.failed:
                        return
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        # high surrogate: require \uXXXX low surrogate
                        if (
                            self.pos + 1 < self.length
                            and self.buf[unsafe_offset=self.pos] == 0x5C
                            and self.buf[unsafe_offset=self.pos + 1] == 0x75
                        ):
                            self.pos += 2
                            var lo = self.parse_hex4()
                            if self.failed:
                                return
                            if lo < 0xDC00 or lo > 0xDFFF:
                                # Python json keeps lone surrogates; we cannot
                                # encode them in UTF-8 -> punt to the fallback.
                                self.failed = True
                                self.unsupported = True
                                return
                            cp = 0x10000 + ((cp - 0xD800) * 0x400) + (lo - 0xDC00)
                        else:
                            self.failed = True  # lone high surrogate -> punt
                            self.unsupported = True
                            return
                    elif cp >= 0xDC00 and cp <= 0xDFFF:
                        self.failed = True  # lone low surrogate -> punt
                        self.unsupported = True
                        return
                    out.append(Codepoint(unsafe_unchecked_codepoint=cp))
                else:
                    self.failed = True
                    return
            elif c < 0x20:
                self.failed = True  # raw control chars are invalid JSON
                return
            elif c < 0x80:
                out.append(Codepoint(c))
                self.pos += 1
            else:
                # UTF-8 multibyte sequence: decode and re-encode (validates).
                var cp = UInt32(0)
                var nbytes = 0
                if c >= 0xC2 and c <= 0xDF:
                    cp = UInt32(c - 0xC0)
                    nbytes = 2
                elif c >= 0xE0 and c <= 0xEF:
                    cp = UInt32(c - 0xE0)
                    nbytes = 3
                elif c >= 0xF0 and c <= 0xF4:
                    cp = UInt32(c - 0xF0)
                    nbytes = 4
                else:
                    self.failed = True
                    return
                self.pos += 1
                for _ in range(nbytes - 1):
                    if self.at_end():
                        self.failed = True
                        return
                    var cb = self.peek()
                    if cb < 0x80 or cb > 0xBF:
                        self.failed = True
                        return
                    cp = cp * 0x40 + UInt32(cb - 0x80)
                    self.pos += 1
                out.append(Codepoint(unsafe_unchecked_codepoint=cp))

    def parse_string(mut self) -> String:
        var out = String()
        if not self.expect(0x22):
            self.failed = True
            return out
        self.parse_string_body(out)
        return out

    def parse_number(mut self, mut arena: List[Node]) -> Int32:
        var start = self.pos
        var is_float = False
        if self.peek() == 0x2D:  # -
            self.pos += 1
        # integer part
        if self.at_end():
            self.failed = True
            return NULL_IDX
        var c0 = self.peek()
        if c0 == 0x30:  # leading 0 must be alone
            self.pos += 1
        elif c0 >= 0x31 and c0 <= 0x39:
            while not self.at_end() and self.peek() >= 0x30 and self.peek() <= 0x39:
                self.pos += 1
        else:
            self.failed = True
            return NULL_IDX
        # fraction
        if not self.at_end() and self.peek() == 0x2E:  # .
            is_float = True
            self.pos += 1
            if self.at_end() or self.peek() < 0x30 or self.peek() > 0x39:
                self.failed = True
                return NULL_IDX
            while not self.at_end() and self.peek() >= 0x30 and self.peek() <= 0x39:
                self.pos += 1
        # exponent
        if not self.at_end() and (self.peek() == 0x65 or self.peek() == 0x45):  # e E
            is_float = True
            self.pos += 1
            if not self.at_end() and (self.peek() == 0x2B or self.peek() == 0x2D):
                self.pos += 1
            if self.at_end() or self.peek() < 0x30 or self.peek() > 0x39:
                self.failed = True
                return NULL_IDX
            while not self.at_end() and self.peek() >= 0x30 and self.peek() <= 0x39:
                self.pos += 1
        if not is_float:
            # int64 with overflow detection; overflow -> punt (Python would
            # produce an arbitrary-precision int).
            var i = start
            var neg = False
            if self.buf[unsafe_offset=i] == 0x2D:
                neg = True
                i += 1
            var acc = Int64(0)
            while i < self.pos:
                var d = Int64(self.buf[unsafe_offset=i] - 0x30)
                if acc > (I64_MAX - d) / 10:
                    self.failed = True
                    self.unsupported = True  # Python bigint: punt
                    return NULL_IDX
                acc = acc * 10 + d
                i += 1
            if neg:
                acc = -acc
            return add_int(arena, acc)
        # float: correctly rounded parse through Mojo's float reader (bit-
        # identical to Python's json.loads for finite values; out-of-range
        # magnitudes become +-inf / +-0.0 exactly like Python).
        var s = String()
        for k in range(start, self.pos):
            s.append(Codepoint(self.buf[unsafe_offset=k]))
        var value = Float64(0.0)
        try:
            value = Float64(s)
        except:
            # Unreachable for grammar-valid numbers (Mojo maps out-of-range
            # exponents to +-inf / +-0.0 like Python); punt defensively.
            self.failed = True
            return NULL_IDX
        return add_float(arena, value)


# ---------------------------------------------------------------------------
# Expression lexer
# ---------------------------------------------------------------------------

comptime TK_EOF: Int32 = 0
comptime TK_DOT: Int32 = 1
comptime TK_LBRACKET: Int32 = 2
comptime TK_RBRACKET: Int32 = 3
comptime TK_LPAREN: Int32 = 4
comptime TK_RPAREN: Int32 = 5
comptime TK_LBRACE: Int32 = 6
comptime TK_RBRACE: Int32 = 7
comptime TK_COMMA: Int32 = 8
comptime TK_COLON: Int32 = 9
comptime TK_STAR: Int32 = 10
comptime TK_PIPE: Int32 = 11
comptime TK_OR: Int32 = 12
comptime TK_AND: Int32 = 13
comptime TK_NOT: Int32 = 14
comptime TK_EQ: Int32 = 15
comptime TK_NE: Int32 = 16
comptime TK_LT: Int32 = 17
comptime TK_LE: Int32 = 18
comptime TK_GT: Int32 = 19
comptime TK_GE: Int32 = 20
comptime TK_CURRENT: Int32 = 21
comptime TK_AMP: Int32 = 22
comptime TK_QUESTION: Int32 = 23
comptime TK_IDENT: Int32 = 24
comptime TK_QIDENT: Int32 = 25
comptime TK_RAWSTR: Int32 = 26
comptime TK_LITERAL: Int32 = 27
comptime TK_NUMBER: Int32 = 28


struct Token(Copyable, Movable):
    var kind: Int32
    var s: String  # IDENT/QIDENT/RAWSTR payload
    var i: Int64  # NUMBER payload
    var lit: Int32  # LITERAL: arena node of the parsed JSON value

    def __init__(
        out self,
        kind: Int32,
        s: String = String(),
        i: Int64 = 0,
        lit: Int32 = NULL_IDX,
    ):
        self.kind = kind
        self.s = s
        self.i = i
        self.lit = lit


# Expression AST node kinds.
comptime E_IDENTITY: Int32 = 0
comptime E_CURRENT: Int32 = 1
comptime E_FIELD: Int32 = 2  # s=name
comptime E_DOT: Int32 = 3  # a=left, b=right
comptime E_INDEX: Int32 = 4  # a=source, v0=index
comptime E_PROJECT: Int32 = 5  # a=source, b=rhs, sub=proj kind (+slice below)
comptime E_FILTER: Int32 = 6  # a=source, b=pred, c=rhs
comptime E_PIPE: Int32 = 7  # a=left, b=right
comptime E_OR: Int32 = 8  # a, b
comptime E_AND: Int32 = 9  # a, b
comptime E_NOT: Int32 = 10  # a
comptime E_CMP: Int32 = 11  # sub=op, a, b
comptime E_LITERAL: Int32 = 12  # lit=arena node
comptime E_FUNC: Int32 = 13  # s=name, args=list of expr indices
comptime E_EXPREF: Int32 = 14  # a=expr

# Projection kinds (E_PROJECT sub).
comptime P_LISTWILD: Int32 = 0  # [*]
comptime P_OBJWILD: Int32 = 1  # .*
comptime P_FLATTEN: Int32 = 2  # []
comptime P_SLICE: Int32 = 3  # [a:b:c] (v0/v1/v2 + f0/f1/f2 presence flags)

# Comparison ops (E_CMP sub).
comptime CMP_EQ: Int32 = 0
comptime CMP_NE: Int32 = 1
comptime CMP_LT: Int32 = 2
comptime CMP_LE: Int32 = 3
comptime CMP_GT: Int32 = 4
comptime CMP_GE: Int32 = 5

# Binding powers (higher binds tighter). Verified against the reference:
# pipe < or < and < comparisons < flatten < not/projection-rhs < bracket < dot.
# Flatten (`[]`) is unusual: `a == b[]` groups as `a == (b[])` but `!a[]`
# groups as `(!a)[]`, and a projection's right side never swallows a flatten
# (`matrix[][]` flattens the whole result again).
comptime BP_PIPE: Int32 = 5
comptime BP_OR: Int32 = 10
comptime BP_AND: Int32 = 15
comptime BP_CMP: Int32 = 20
comptime BP_FLATTEN: Int32 = 30
comptime BP_NOT: Int32 = 40  # unary ! grabs paths/wildcards, not flatten/ops
comptime BP_PROJ: Int32 = 40  # projection rhs: ./[ chains, stops at flatten
comptime BP_BRACKET: Int32 = 55
comptime BP_DOT: Int32 = 60


struct Expr(Copyable, Movable):
    var kind: Int32
    var sub: Int32
    var a: Int32
    var b: Int32
    var c: Int32
    var s: String
    var v0: Int64
    var v1: Int64
    var v2: Int64
    var f0: Bool
    var f1: Bool
    var f2: Bool
    var lit: Int32
    var args: List[Int32]

    def __init__(
        out self,
        kind: Int32,
        sub: Int32 = 0,
        a: Int32 = 0,
        b: Int32 = 0,
        c: Int32 = 0,
        s: String = String(),
        v0: Int64 = 0,
        v1: Int64 = 0,
        v2: Int64 = 0,
        f0: Bool = False,
        f1: Bool = False,
        f2: Bool = False,
        lit: Int32 = NULL_IDX,
        var args: List[Int32] = List[Int32](),
    ):
        self.kind = kind
        self.sub = sub
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2
        self.lit = lit
        self.args = args^


struct Engine:
    """One whole search call: expression tokens + AST, the node arena the
    document and all intermediate results live in, and the error status."""

    var arena: List[Node]
    var exprs: List[Expr]
    var toks: List[Token]
    var tp: Int  # token cursor (parser)
    var src: List[UInt8]  # expression bytes
    var sp: Int  # source cursor (lexer)
    var err: Int32

    def __init__(out self, var src: List[UInt8]):
        self.arena = new_arena()
        self.exprs = List[Expr]()
        self.toks = List[Token]()
        self.tp = 0
        self.src = src^
        self.sp = 0
        self.err = STATUS_OK

    # -- lexer -------------------------------------------------------------

    def lex_peek(self, ahead: Int = 0) -> UInt8:
        var p = self.sp + ahead
        if p >= len(self.src):
            return 0
        return self.src[p]

    def lex_fail(mut self):
        self.err = STATUS_EXPR

    def lex(mut self):
        """Tokenize self.src into self.toks; self.err set on any invalid
        token (the wrapper then re-parses with the vendored implementation,
        which raises the exact reference error class)."""
        while self.sp < len(self.src):
            var c = self.src[self.sp]
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.sp += 1
                continue
            self.sp += 1
            if c == 0x2E:  # .
                self.toks.append(Token(TK_DOT))
            elif c == 0x2C:  # ,
                self.toks.append(Token(TK_COMMA))
            elif c == 0x5B:  # [
                self.toks.append(Token(TK_LBRACKET))
            elif c == 0x5D:  # ]
                self.toks.append(Token(TK_RBRACKET))
            elif c == 0x28:  # (
                self.toks.append(Token(TK_LPAREN))
            elif c == 0x29:  # )
                self.toks.append(Token(TK_RPAREN))
            elif c == 0x7B:  # {
                self.toks.append(Token(TK_LBRACE))
            elif c == 0x7D:  # }
                self.toks.append(Token(TK_RBRACE))
            elif c == 0x3A:  # :
                self.toks.append(Token(TK_COLON))
            elif c == 0x2A:  # *
                self.toks.append(Token(TK_STAR))
            elif c == 0x40:  # @
                self.toks.append(Token(TK_CURRENT))
            elif c == 0x3F:  # ?
                self.toks.append(Token(TK_QUESTION))
            elif c == 0x7C:  # | or ||
                if self.lex_peek() == 0x7C:
                    self.sp += 1
                    self.toks.append(Token(TK_OR))
                else:
                    self.toks.append(Token(TK_PIPE))
            elif c == 0x26:  # & or &&
                if self.lex_peek() == 0x26:
                    self.sp += 1
                    self.toks.append(Token(TK_AND))
                else:
                    self.toks.append(Token(TK_AMP))
            elif c == 0x21:  # ! or !=
                if self.lex_peek() == 0x3D:
                    self.sp += 1
                    self.toks.append(Token(TK_NE))
                else:
                    self.toks.append(Token(TK_NOT))
            elif c == 0x3D:  # ==
                if self.lex_peek() == 0x3D:
                    self.sp += 1
                    self.toks.append(Token(TK_EQ))
                else:
                    self.lex_fail()
                    return
            elif c == 0x3C:  # < or <=
                if self.lex_peek() == 0x3D:
                    self.sp += 1
                    self.toks.append(Token(TK_LE))
                else:
                    self.toks.append(Token(TK_LT))
            elif c == 0x3E:  # > or >=
                if self.lex_peek() == 0x3D:
                    self.sp += 1
                    self.toks.append(Token(TK_GE))
                else:
                    self.toks.append(Token(TK_GT))
            elif c == 0x60:  # ` literal
                self.lex_literal()
                if self.err != STATUS_OK:
                    return
            elif c == 0x22:  # " quoted identifier
                var name = self.lex_quoted()
                if self.err != STATUS_OK:
                    return
                self.toks.append(Token(TK_QIDENT, s=name))
            elif c == 0x27:  # ' raw string
                var raw = self.lex_raw_string()
                if self.err != STATUS_OK:
                    return
                self.toks.append(Token(TK_RAWSTR, s=raw))
            elif (c >= 0x30 and c <= 0x39) or (c == 0x2D and self.lex_peek() >= 0x30 and self.lex_peek() <= 0x39):
                var num = self.lex_number()
                self.toks.append(Token(TK_NUMBER, i=num))
            elif (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F:
                var start = self.sp - 1
                while self.sp < len(self.src):
                    var d = self.src[self.sp]
                    if (d >= 0x41 and d <= 0x5A) or (d >= 0x61 and d <= 0x7A) or (d >= 0x30 and d <= 0x39) or d == 0x5F:
                        self.sp += 1
                    else:
                        break
                var name = String()
                for k in range(start, self.sp):
                    name.append(Codepoint(self.src[k]))
                self.toks.append(Token(TK_IDENT, s=name))
            else:
                self.lex_fail()
                return
        self.toks.append(Token(TK_EOF))

    def lex_number(mut self) -> Int64:
        # entry: self.sp is just past the first character ('-' or a digit),
        # so the number starts at self.sp - 1.
        var i = self.sp - 1
        var neg = self.src[i] == 0x2D
        if neg:
            i += 1
        var acc = Int64(0)
        var overflow = False
        while i < len(self.src) and self.src[i] >= 0x30 and self.src[i] <= 0x39:
            var d = Int64(self.src[i] - 0x30)
            if acc > (I64_MAX - d) / 10:
                overflow = True
            if not overflow:
                acc = acc * 10 + d
            i += 1
        self.sp = i
        if overflow:
            return I64_MIN if neg else I64_MAX  # out of range either way
        return -acc if neg else acc

    def lex_quoted(mut self) -> String:
        """JSON string semantics for "quoted identifiers"."""
        var out = String()
        while True:
            if self.sp >= len(self.src):
                self.lex_fail()
                return out
            var c = self.src[self.sp]
            self.sp += 1
            if c == 0x22:  # closing "
                return out
            if c == 0x5C:  # escape
                if self.sp >= len(self.src):
                    self.lex_fail()
                    return out
                var e = self.src[self.sp]
                self.sp += 1
                if e == 0x22 or e == 0x5C or e == 0x2F:
                    out.append(Codepoint(e))
                elif e == 0x62:
                    out.append(Codepoint(0x08))
                elif e == 0x66:
                    out.append(Codepoint(0x0C))
                elif e == 0x6E:
                    out.append(Codepoint(0x0A))
                elif e == 0x72:
                    out.append(Codepoint(0x0D))
                elif e == 0x74:
                    out.append(Codepoint(0x09))
                elif e == 0x75:
                    var cp = self.lex_hex4()
                    if self.err != STATUS_OK:
                        return out
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        if self.sp + 1 < len(self.src) and self.src[self.sp] == 0x5C and self.src[self.sp + 1] == 0x75:
                            self.sp += 2
                            var lo = self.lex_hex4()
                            if self.err != STATUS_OK:
                                return out
                            if lo < 0xDC00 or lo > 0xDFFF:
                                self.lex_fail()  # lone surrogate: punt
                                return out
                            cp = 0x10000 + ((cp - 0xD800) * 0x400) + (lo - 0xDC00)
                        else:
                            self.lex_fail()
                            return out
                    elif cp >= 0xDC00 and cp <= 0xDFFF:
                        self.lex_fail()
                        return out
                    out.append(Codepoint(unsafe_unchecked_codepoint=cp))
                else:
                    self.lex_fail()
                    return out
            elif c < 0x80:
                out.append(Codepoint(c))
            else:
                # UTF-8 passthrough (expression strings are valid UTF-8).
                var cp = UInt32(0)
                var nbytes = 0
                if c >= 0xC2 and c <= 0xDF:
                    cp = UInt32(c - 0xC0)
                    nbytes = 2
                elif c >= 0xE0 and c <= 0xEF:
                    cp = UInt32(c - 0xE0)
                    nbytes = 3
                elif c >= 0xF0 and c <= 0xF4:
                    cp = UInt32(c - 0xF0)
                    nbytes = 4
                else:
                    self.lex_fail()
                    return out
                for _ in range(nbytes - 1):
                    if self.sp >= len(self.src):
                        self.lex_fail()
                        return out
                    var cb = self.src[self.sp]
                    if cb < 0x80 or cb > 0xBF:
                        self.lex_fail()
                        return out
                    cp = cp * 0x40 + UInt32(cb - 0x80)
                    self.sp += 1
                out.append(Codepoint(unsafe_unchecked_codepoint=cp))
        return out

    def lex_hex4(mut self) -> UInt32:
        var v = UInt32(0)
        for _ in range(4):
            if self.sp >= len(self.src):
                self.lex_fail()
                return 0
            var c = self.src[self.sp]
            var d = UInt32(0)
            if c >= 0x30 and c <= 0x39:
                d = UInt32(c - 0x30)
            elif c >= 0x61 and c <= 0x66:
                d = UInt32(c - 0x61 + 10)
            elif c >= 0x41 and c <= 0x46:
                d = UInt32(c - 0x41 + 10)
            else:
                self.lex_fail()
                return 0
            v = v * 16 + d
            self.sp += 1
        return v

    def lex_raw_string(mut self) -> String:
        """Raw strings: \\X -> X when X is a quote, else both chars kept."""
        var out = String()
        while True:
            if self.sp >= len(self.src):
                self.lex_fail()
                return out
            var c = self.src[self.sp]
            if c == 0x27:  # closing '
                self.sp += 1
                return out
            if c == 0x5C:  # backslash
                if self.sp + 1 >= len(self.src):
                    self.lex_fail()
                    return out
                var n = self.src[self.sp + 1]
                if n == 0x27:
                    out.append(Codepoint(0x27))
                    self.sp += 2
                    continue
                # \\ stays two chars in raw strings (per the reference).
                out.append(Codepoint(0x5C))
                if n < 0x80:
                    out.append(Codepoint(n))
                    self.sp += 2
                    continue
                self.sp += 1  # at n; read_utf8 advances past the sequence
                var ecp = self.read_utf8()
                if self.err != STATUS_OK:
                    return out
                out.append(Codepoint(unsafe_unchecked_codepoint=ecp))
                continue
            if c < 0x80:
                out.append(Codepoint(c))
                self.sp += 1
            else:
                var cp = self.read_utf8()
                if self.err != STATUS_OK:
                    return out
                out.append(Codepoint(unsafe_unchecked_codepoint=cp))
        return out

    def read_utf8(mut self) -> UInt32:
        """Decode one UTF-8 sequence at self.sp (advancing past it)."""
        var c = self.src[self.sp]
        var cp = UInt32(0)
        var nbytes = 0
        if c >= 0xC2 and c <= 0xDF:
            cp = UInt32(c - 0xC0)
            nbytes = 2
        elif c >= 0xE0 and c <= 0xEF:
            cp = UInt32(c - 0xE0)
            nbytes = 3
        elif c >= 0xF0 and c <= 0xF4:
            cp = UInt32(c - 0xF0)
            nbytes = 4
        else:
            self.lex_fail()
            return 0
        self.sp += 1
        for _ in range(nbytes - 1):
            if self.sp >= len(self.src):
                self.lex_fail()
                return 0
            var cb = self.src[self.sp]
            if cb < 0x80 or cb > 0xBF:
                self.lex_fail()
                return 0
            cp = cp * 0x40 + UInt32(cb - 0x80)
            self.sp += 1
        return cp

    def lex_literal(mut self):
        """Backtick JSON literal: parse content as JSON; if that fails, as a
        JSON string body (the reference's quote-wrapped fallback); if that
        fails too, reject (the fallback then raises LexerError)."""
        var content = List[UInt8]()
        while True:
            if self.sp >= len(self.src):
                self.lex_fail()
                return
            var c = self.src[self.sp]
            if c == 0x60:  # closing `
                self.sp += 1
                break
            if c == 0x5C:
                if self.sp + 1 >= len(self.src):
                    self.lex_fail()
                    return
                var n = self.src[self.sp + 1]
                if n == 0x60:
                    content.append(0x60)
                else:
                    content.append(0x5C)
                    content.append(n)
                self.sp += 2
                continue
            content.append(c)
            self.sp += 1
        # Python str.lstrip() semantics for the ASCII whitespace subset.
        var start = 0
        while start < len(content) and (
            content[start] == 0x20
            or content[start] == 0x09
            or content[start] == 0x0A
            or content[start] == 0x0D
        ):
            start += 1
        # Python lstrip also strips \x0b \x0c and unicode spaces; rather
        # than mirror that exactly, punt those exotic literals.
        if start < len(content) and (content[start] == 0x0B or content[start] == 0x0C):
            self.err = STATUS_UNSUPPORTED
            return
        if start < len(content) and content[start] >= 0x80:
            var tmp = List[UInt8]()
            tmp.append(content[start])
            for k in range(start + 1, len(content)):
                tmp.append(content[k])
            var first_cp = UInt32(0)
            var b0 = tmp[0]
            if b0 >= 0xC2 and b0 <= 0xDF and len(tmp) >= 2:
                first_cp = UInt32(b0 - 0xC0) * 0x40 + UInt32(tmp[1] - 0x80)
            elif b0 >= 0xE0 and b0 <= 0xEF and len(tmp) >= 3:
                first_cp = (UInt32(b0 - 0xE0) * 0x40 + UInt32(tmp[1] - 0x80)) * 0x40 + UInt32(tmp[2] - 0x80)
            elif b0 >= 0xF0 and len(tmp) >= 4:
                first_cp = ((UInt32(b0 - 0xF0) * 0x40 + UInt32(tmp[1] - 0x80)) * 0x40 + UInt32(tmp[2] - 0x80)) * 0x40 + UInt32(tmp[3] - 0x80)
            if (
                first_cp == 0x85
                or first_cp == 0xA0
                or first_cp == 0x1680
                or (first_cp >= 0x2000 and first_cp <= 0x200A)
                or first_cp == 0x2028
                or first_cp == 0x2029
                or first_cp == 0x202F
                or first_cp == 0x205F
                or first_cp == 0x3000
            ):
                self.err = STATUS_UNSUPPORTED
                return
        var body = List[UInt8]()
        for k in range(start, len(content)):
            body.append(content[k])
        # Attempt 1: strict JSON.
        var p = JsonParser(body.unsafe_ptr(), len(body))
        var lit = p.parse_document(self.arena)
        if p.unsupported:
            # valid JSON the kernel cannot represent (e.g. bigint literals)
            self.err = STATUS_UNSUPPORTED
            return
        if p.failed:
            self.err = STATUS_OK  # not fatal: try the string-body form
            # Attempt 2: JSON string body (reference wraps in quotes).
            var wrapped = List[UInt8]()
            wrapped.append(0x22)
            for k in range(len(body)):
                wrapped.append(body[k])
            wrapped.append(0x22)
            var p2 = JsonParser(wrapped.unsafe_ptr(), len(wrapped))
            var s = p2.parse_string()
            if p2.failed or not p2.at_end():
                self.lex_fail()
                return
            lit = add_str(self.arena, s)
        self.toks.append(Token(TK_LITERAL, lit=lit))

    # -- parser ------------------------------------------------------------

    def push_expr(mut self, var e: Expr) -> Int32:
        self.exprs.append(e^)
        return Int32(len(self.exprs) - 1)

    def cur_tok(self) -> Int32:
        return self.toks[self.tp].kind

    def advance_tok(mut self):
        if self.tp < len(self.toks) - 1:
            self.tp += 1

    def parse(mut self) -> Int32:
        """Lex and parse the expression; returns the root expr index."""
        self.lex()
        if self.err != STATUS_OK:
            return -1
        if self.cur_tok() == TK_EOF:
            self.err = STATUS_EXPR  # empty expression
            return -1
        var root = self.parse_expr(0)
        if self.err != STATUS_OK:
            return -1
        if self.cur_tok() != TK_EOF:
            self.err = STATUS_EXPR  # trailing tokens
            return -1
        return root

    def led_bp(self, tk: Int32) -> Int32:
        if tk == TK_DOT:
            return BP_DOT
        if tk == TK_LBRACKET:
            # `[]` (flatten) has its own low binding power; all other
            # brackets bind tightly.
            if self.tp + 1 < len(self.toks) and self.toks[self.tp + 1].kind == TK_RBRACKET:
                return BP_FLATTEN
            return BP_BRACKET
        if tk >= TK_EQ and tk <= TK_GE:
            return BP_CMP
        if tk == TK_AND:
            return BP_AND
        if tk == TK_OR:
            return BP_OR
        if tk == TK_PIPE:
            return BP_PIPE
        return 0

    def parse_expr(mut self, min_bp: Int32) -> Int32:
        var left = self.parse_nud()
        if self.err != STATUS_OK:
            return -1
        return self.parse_leds(left, min_bp)

    def parse_leds(mut self, left: Int32, min_bp: Int32) -> Int32:
        var cur = left
        while True:
            var tk = self.cur_tok()
            var bp = self.led_bp(tk)
            if bp == 0 or bp < min_bp:
                return cur
            cur = self.parse_led(cur, tk)
            if self.err != STATUS_OK:
                return -1
        return cur

    def parse_nud(mut self) -> Int32:
        var tk = self.cur_tok()
        if tk == TK_IDENT:
            var name = self.toks[self.tp].s
            self.advance_tok()
            if self.cur_tok() == TK_LPAREN:
                return self.parse_function(name)
            return self.push_expr(Expr(E_FIELD, s=name))
        if tk == TK_QIDENT:
            var name = self.toks[self.tp].s
            self.advance_tok()
            return self.push_expr(Expr(E_FIELD, s=name))
        if tk == TK_RAWSTR:
            var s = self.toks[self.tp].s
            self.advance_tok()
            var idx = add_str(self.arena, s)
            return self.push_expr(Expr(E_LITERAL, lit=idx))
        if tk == TK_LITERAL:
            var lit = self.toks[self.tp].lit
            self.advance_tok()
            return self.push_expr(Expr(E_LITERAL, lit=lit))
        if tk == TK_CURRENT:
            self.advance_tok()
            return self.push_expr(Expr(E_CURRENT))
        if tk == TK_STAR:
            self.advance_tok()
            var src = self.push_expr(Expr(E_CURRENT))
            var rhs = self.parse_continuation()
            return self.push_expr(Expr(E_PROJECT, sub=P_OBJWILD, a=src, b=rhs))
        if tk == TK_LBRACKET:
            var src = self.push_expr(Expr(E_CURRENT))
            return self.parse_bracket(src)
        if tk == TK_LPAREN:
            self.advance_tok()
            var inner = self.parse_expr(0)
            if self.err != STATUS_OK:
                return -1
            if self.cur_tok() != TK_RPAREN:
                self.err = STATUS_EXPR
                return -1
            self.advance_tok()
            return inner
        if tk == TK_NOT:
            self.advance_tok()
            var operand = self.parse_expr(BP_NOT)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_NOT, a=operand))
        if tk == TK_AMP:
            self.advance_tok()
            var operand = self.parse_expr(0)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_EXPREF, a=operand))
        if tk == TK_LBRACE:
            # Multiselect hash: in-spec, outside the kernel subset.
            self.err = STATUS_UNSUPPORTED
            return -1
        # Everything else is invalid in primary position.
        self.err = STATUS_EXPR
        return -1

    def parse_led(mut self, left: Int32, tk: Int32) -> Int32:
        if tk == TK_DOT:
            return self.parse_dot(left)
        if tk == TK_LBRACKET:
            return self.parse_bracket(left)
        if tk == TK_PIPE:
            self.advance_tok()
            var rhs = self.parse_expr(BP_PIPE + 1)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_PIPE, a=left, b=rhs))
        if tk == TK_OR:
            self.advance_tok()
            var rhs = self.parse_expr(BP_OR + 1)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_OR, a=left, b=rhs))
        if tk == TK_AND:
            self.advance_tok()
            var rhs = self.parse_expr(BP_AND + 1)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_AND, a=left, b=rhs))
        if tk >= TK_EQ and tk <= TK_GE:
            var op = Int32(0)
            if tk == TK_NE:
                op = CMP_NE
            elif tk == TK_LT:
                op = CMP_LT
            elif tk == TK_LE:
                op = CMP_LE
            elif tk == TK_GT:
                op = CMP_GT
            elif tk == TK_GE:
                op = CMP_GE
            self.advance_tok()
            var rhs = self.parse_expr(BP_CMP + 1)
            if self.err != STATUS_OK:
                return -1
            return self.push_expr(Expr(E_CMP, sub=op, a=left, b=rhs))
        self.err = STATUS_EXPR
        return -1

    def parse_dot(mut self, left: Int32) -> Int32:
        self.advance_tok()  # consume '.'
        var tk = self.cur_tok()
        if tk == TK_IDENT:
            var name = self.toks[self.tp].s
            self.advance_tok()
            if self.cur_tok() == TK_LPAREN:
                var f = self.parse_function(name)
                if self.err != STATUS_OK:
                    return -1
                return self.push_expr(Expr(E_DOT, a=left, b=f))
            var field = self.push_expr(Expr(E_FIELD, s=name))
            return self.push_expr(Expr(E_DOT, a=left, b=field))
        if tk == TK_QIDENT:
            var name = self.toks[self.tp].s
            self.advance_tok()
            var field = self.push_expr(Expr(E_FIELD, s=name))
            return self.push_expr(Expr(E_DOT, a=left, b=field))
        if tk == TK_STAR:
            self.advance_tok()
            var rhs = self.parse_continuation()
            return self.push_expr(Expr(E_PROJECT, sub=P_OBJWILD, a=left, b=rhs))
        if tk == TK_LBRACKET or tk == TK_LBRACE:
            # Multiselect list/hash: in-spec, outside the kernel subset.
            self.err = STATUS_UNSUPPORTED
            return -1
        self.err = STATUS_EXPR
        return -1

    def parse_bracket(mut self, source: Int32) -> Int32:
        self.advance_tok()  # consume '['
        var tk = self.cur_tok()
        if tk == TK_STAR:
            self.advance_tok()
            if self.cur_tok() != TK_RBRACKET:
                self.err = STATUS_EXPR
                return -1
            self.advance_tok()
            var rhs = self.parse_continuation()
            return self.push_expr(Expr(E_PROJECT, sub=P_LISTWILD, a=source, b=rhs))
        if tk == TK_RBRACKET:
            self.advance_tok()
            var rhs = self.parse_continuation()
            return self.push_expr(Expr(E_PROJECT, sub=P_FLATTEN, a=source, b=rhs))
        if tk == TK_QUESTION:
            self.advance_tok()
            var pred = self.parse_expr(0)
            if self.err != STATUS_OK:
                return -1
            if self.cur_tok() != TK_RBRACKET:
                self.err = STATUS_EXPR
                return -1
            self.advance_tok()
            var rhs = self.parse_continuation()
            return self.push_expr(Expr(E_FILTER, a=source, b=pred, c=rhs))
        if tk == TK_NUMBER or tk == TK_COLON:
            var v0 = Int64(0)
            var v1 = Int64(0)
            var v2 = Int64(0)
            var f0 = False
            var f1 = False
            var f2 = False
            if tk == TK_NUMBER:
                v0 = self.toks[self.tp].i
                f0 = True
                self.advance_tok()
            if self.cur_tok() == TK_RBRACKET and f0:
                self.advance_tok()
                return self.push_expr(Expr(E_INDEX, a=source, v0=v0))
            if self.cur_tok() != TK_COLON:
                self.err = STATUS_EXPR
                return -1
            self.advance_tok()
            if self.cur_tok() == TK_NUMBER:
                v1 = self.toks[self.tp].i
                f1 = True
                self.advance_tok()
            if self.cur_tok() == TK_COLON:
                self.advance_tok()
                if self.cur_tok() == TK_NUMBER:
                    v2 = self.toks[self.tp].i
                    f2 = True
                    self.advance_tok()
            if self.cur_tok() != TK_RBRACKET:
                self.err = STATUS_EXPR
                return -1
            self.advance_tok()
            var rhs = self.parse_continuation()
            return self.push_expr(
                Expr(
                    E_PROJECT,
                    sub=P_SLICE,
                    a=source,
                    b=rhs,
                    v0=v0,
                    v1=v1,
                    v2=v2,
                    f0=f0,
                    f1=f1,
                    f2=f2,
                )
            )
        self.err = STATUS_EXPR
        return -1

    def parse_continuation(mut self) -> Int32:
        """The right-hand side of a projection: a ./[ chain evaluated per
        element, stopping before pipes and boolean/comparison operators."""
        var tk = self.cur_tok()
        if tk == TK_DOT or tk == TK_LBRACKET:
            var src = self.push_expr(Expr(E_CURRENT))
            return self.parse_leds(src, BP_PROJ)
        return self.push_expr(Expr(E_CURRENT))

    def parse_function(mut self, name: String) -> Int32:
        self.advance_tok()  # consume '('
        var args = List[Int32]()
        if self.cur_tok() != TK_RPAREN:
            while True:
                var arg = self.parse_expr(0)
                if self.err != STATUS_OK:
                    return -1
                args.append(arg)
                if self.cur_tok() == TK_COMMA:
                    self.advance_tok()
                    # a trailing comma before ')' is accepted by the reference
                    if self.cur_tok() == TK_RPAREN:
                        break
                    continue
                break
        if self.cur_tok() != TK_RPAREN:
            self.err = STATUS_EXPR
            return -1
        self.advance_tok()
        return self.push_expr(Expr(E_FUNC, s=name, args=args^))

    # -- interpreter ---------------------------------------------------------

    def truthy(self, idx: Int32) -> Bool:
        ref node = self.arena[Int(idx)]
        if node.tag == T_NULL or node.tag == T_FALSE:
            return False
        if node.tag == T_STR:
            return node.s.byte_length() > 0
        if node.tag == T_ARR or node.tag == T_OBJ:
            return len(node.kids) > 0
        return True  # numbers (including 0/0.0), true, tuples are truthy

    def eval(mut self, e: Int32, cur: Int32) -> Int32:
        if self.err != STATUS_OK:
            return NULL_IDX
        var kind = self.exprs[Int(e)].kind
        if kind == E_CURRENT:
            return cur
        if kind == E_LITERAL:
            return self.exprs[Int(e)].lit
        if kind == E_FIELD:
            ref node = self.arena[Int(cur)]
            if node.tag != T_OBJ:
                return NULL_IDX
            ref name = self.exprs[Int(e)].s
            for k in range(len(node.keys)):
                if node.keys[k] == name:
                    return node.kids[k]
            return NULL_IDX
        if kind == E_DOT:
            var a = self.exprs[Int(e)].a
            var b = self.exprs[Int(e)].b
            var mid = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            return self.eval(b, mid)
        if kind == E_INDEX:
            var a = self.exprs[Int(e)].a
            var v0 = self.exprs[Int(e)].v0
            var src = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            ref node = self.arena[Int(src)]
            if node.tag != T_ARR:
                return NULL_IDX
            var n = Int64(len(node.kids))
            var i = v0
            if i < 0:
                i += n
            if i < 0 or i >= n:
                return NULL_IDX
            return node.kids[Int(i)]
        if kind == E_PROJECT:
            return self.eval_project(e, cur)
        if kind == E_FILTER:
            return self.eval_filter(e, cur)
        if kind == E_PIPE:
            var a = self.exprs[Int(e)].a
            var b = self.exprs[Int(e)].b
            var mid = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            return self.eval(b, mid)
        if kind == E_OR:
            var a = self.exprs[Int(e)].a
            var b = self.exprs[Int(e)].b
            var l = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            if self.truthy(l):
                return l
            return self.eval(b, cur)
        if kind == E_AND:
            var a = self.exprs[Int(e)].a
            var b = self.exprs[Int(e)].b
            var l = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            if not self.truthy(l):
                return l
            return self.eval(b, cur)
        if kind == E_NOT:
            var a = self.exprs[Int(e)].a
            var v = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            return bool_idx(not self.truthy(v))
        if kind == E_CMP:
            var sub = self.exprs[Int(e)].sub
            var a = self.exprs[Int(e)].a
            var b = self.exprs[Int(e)].b
            var x = self.eval(a, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            var y = self.eval(b, cur)
            if self.err != STATUS_OK:
                return NULL_IDX
            return self.eval_compare(sub, x, y)
        if kind == E_FUNC:
            return self.eval_function(e, cur)
        # E_EXPREF evaluated in value position (e.g. a top-level `&a`): the
        # reference returns a live object; outside the kernel subset.
        self.err = STATUS_UNSUPPORTED
        return NULL_IDX

    def eval_project(mut self, e: Int32, cur: Int32) -> Int32:
        var sub = self.exprs[Int(e)].sub
        var a = self.exprs[Int(e)].a
        var rhs = self.exprs[Int(e)].b
        var src = self.eval(a, cur)
        if self.err != STATUS_OK:
            return NULL_IDX
        ref node = self.arena[Int(src)]
        var items = List[Int32]()
        if sub == P_OBJWILD:
            if node.tag != T_OBJ:
                return NULL_IDX
            for k in range(len(node.kids)):
                items.append(node.kids[k])
        else:
            if node.tag != T_ARR:
                return NULL_IDX
            if sub == P_LISTWILD:
                for k in range(len(node.kids)):
                    items.append(node.kids[k])
            elif sub == P_FLATTEN:
                for k in range(len(node.kids)):
                    var kid = node.kids[k]
                    if self.arena[Int(kid)].tag == T_ARR:
                        for j in range(len(self.arena[Int(kid)].kids)):
                            items.append(self.arena[Int(kid)].kids[j])
                    else:
                        items.append(kid)
            elif sub == P_SLICE:
                var v0 = self.exprs[Int(e)].v0
                var v1 = self.exprs[Int(e)].v1
                var v2 = self.exprs[Int(e)].v2
                var f0 = self.exprs[Int(e)].f0
                var f1 = self.exprs[Int(e)].f1
                var f2 = self.exprs[Int(e)].f2
                var n = Int64(len(node.kids))
                var step = v2 if f2 else Int64(1)
                if step == 0:
                    # the reference raises ValueError("slice step cannot be
                    # zero"); the fallback reproduces it exactly.
                    self.err = STATUS_EVAL
                    return NULL_IDX
                var lo = Int64(0)
                var hi = n
                if step < 0:
                    lo = Int64(-1)
                    hi = n - 1
                var start = Int64(0)
                if not f0:
                    start = lo if step > 0 else hi
                else:
                    start = v0
                    if start < 0:
                        start += n
                        if start < lo:
                            start = lo
                    elif start > hi:
                        start = hi
                var stop = Int64(0)
                if not f1:
                    stop = n if step > 0 else Int64(-1)
                else:
                    stop = v1
                    if stop < 0:
                        stop += n
                        if stop < lo:
                            stop = lo
                    elif stop > hi:
                        stop = hi
                var i = start
                if step > 0:
                    while i < stop:
                        items.append(node.kids[Int(i)])
                        i += step
                else:
                    while i > stop:
                        items.append(node.kids[Int(i)])
                        i += step
            else:
                self.err = STATUS_EXPR
                return NULL_IDX
        # Project rhs over items, dropping null results (spec semantics,
        # including the identity-rhs form: `a[*]` drops null elements).
        var rhs_kind = self.exprs[Int(rhs)].kind
        if rhs_kind == E_CURRENT:
            var kept = List[Int32]()
            for k in range(len(items)):
                if self.arena[Int(items[k])].tag != T_NULL:
                    kept.append(items[k])
            return add_arr(self.arena, kept^)
        var out = List[Int32]()
        for k in range(len(items)):
            var r = self.eval(rhs, items[k])
            if self.err != STATUS_OK:
                return NULL_IDX
            if self.arena[Int(r)].tag != T_NULL:
                out.append(r)
        return add_arr(self.arena, out^)

    def eval_filter(mut self, e: Int32, cur: Int32) -> Int32:
        var a = self.exprs[Int(e)].a
        var pred = self.exprs[Int(e)].b
        var rhs = self.exprs[Int(e)].c
        var src = self.eval(a, cur)
        if self.err != STATUS_OK:
            return NULL_IDX
        if self.arena[Int(src)].tag != T_ARR:
            return NULL_IDX
        var rhs_kind = self.exprs[Int(rhs)].kind
        # copy the element list: evaluating the predicate may grow the arena
        var kids = List[Int32]()
        var nk = len(self.arena[Int(src)].kids)
        for k in range(nk):
            kids.append(self.arena[Int(src)].kids[k])
        var out = List[Int32]()
        for k in range(nk):
            var kid = kids[k]
            var p = self.eval(pred, kid)
            if self.err != STATUS_OK:
                return NULL_IDX
            if self.truthy(p):
                var r = kid
                if rhs_kind != E_CURRENT:
                    r = self.eval(rhs, kid)
                    if self.err != STATUS_OK:
                        return NULL_IDX
                if self.arena[Int(r)].tag != T_NULL:
                    out.append(r)
        return add_arr(self.arena, out^)

    def cmp_numbers(mut self, x: Int32, y: Int32) -> Int32:
        """-1/0/1 for number nodes; 2 when unordered (NaN involved).
        Punts (STATUS_UNSUPPORTED) on int/float mixes past 2^53."""
        ref nx = self.arena[Int(x)]
        ref ny = self.arena[Int(y)]
        var tx = nx.tag
        var ty = ny.tag
        if tx == T_INT and ty == T_INT:
            var a = nx.i
            var b = ny.i
            return Int32(-1) if a < b else (Int32(1) if a > b else Int32(0))
        var fx = nx.f if tx == T_FLOAT else Float64(nx.i)
        var fy = ny.f if ty == T_FLOAT else Float64(ny.i)
        if fx != fx or fy != fy:
            return 2
        if (tx == T_INT and (nx.i >= 9007199254740992 or nx.i <= -9007199254740992)) or (
            ty == T_INT and (ny.i >= 9007199254740992 or ny.i <= -9007199254740992)
        ):
            # mixed int/float compare past exact float64 range: punt
            self.err = STATUS_UNSUPPORTED
            return 0
        return Int32(-1) if fx < fy else (Int32(1) if fx > fy else Int32(0))

    def deep_equal(mut self, x: Int32, y: Int32) -> Bool:
        var tx = self.arena[Int(x)].tag
        var ty = self.arena[Int(y)].tag
        if is_number_tag(tx) and is_number_tag(ty):
            var c = self.cmp_numbers(x, y)
            if self.err != STATUS_OK:
                return False
            return c == 0
        if tx != ty:
            return False
        if tx == T_STR:
            return self.arena[Int(x)].s == self.arena[Int(y)].s
        if tx == T_ARR or tx == T_TUPLE:
            var nx_len = len(self.arena[Int(x)].kids)
            if nx_len != len(self.arena[Int(y)].kids):
                return False
            for k in range(nx_len):
                var ex = self.arena[Int(x)].kids[k]
                var ey = self.arena[Int(y)].kids[k]
                if not self.deep_equal(ex, ey):
                    return False
            return True
        if tx == T_OBJ:
            var nk = len(self.arena[Int(x)].keys)
            if nk != len(self.arena[Int(y)].keys):
                return False
            for k in range(nk):
                var key = self.arena[Int(x)].keys[k]
                var found = -1
                for j in range(len(self.arena[Int(y)].keys)):
                    if self.arena[Int(y)].keys[j] == key:
                        found = j
                        break
                if found < 0:
                    return False
                var vx = self.arena[Int(x)].kids[k]
                var vy = self.arena[Int(y)].kids[found]
                if not self.deep_equal(vx, vy):
                    return False
            return True
        return tx == ty  # null / false / true singletons

    def eval_compare(mut self, op: Int32, x: Int32, y: Int32) -> Int32:
        var tx = self.arena[Int(x)].tag
        var ty = self.arena[Int(y)].tag
        if op == CMP_EQ or op == CMP_NE:
            var r = self.deep_equal(x, y)
            if self.err != STATUS_OK:
                return NULL_IDX
            return bool_idx(r if op == CMP_EQ else not r)
        # ordering
        if is_number_tag(tx) and is_number_tag(ty):
            var c = self.cmp_numbers(x, y)
            if self.err != STATUS_OK:
                return NULL_IDX
            if c == 2:
                return FALSE_IDX  # NaN: every ordering comparison is False
            if op == CMP_LT:
                return bool_idx(c < 0)
            if op == CMP_LE:
                return bool_idx(c <= 0)
            if op == CMP_GT:
                return bool_idx(c > 0)
            return bool_idx(c >= 0)
        if tx == T_STR and ty == T_STR:
            ref sx = self.arena[Int(x)].s
            ref sy = self.arena[Int(y)].s
            if op == CMP_LT:
                return bool_idx(sx < sy)
            if op == CMP_LE:
                return bool_idx(sx < sy or sx == sy)
            if op == CMP_GT:
                return bool_idx(sy < sx)
            return bool_idx(sy < sx or sx == sy)
        if (tx == T_STR and is_number_tag(ty)) or (is_number_tag(tx) and ty == T_STR):
            # the reference propagates Python's TypeError here; the fallback
            # reproduces it exactly.
            self.err = STATUS_EVAL
            return NULL_IDX
        return NULL_IDX  # ordering on bool/null/array/object -> null

    # -- functions -----------------------------------------------------------

    def check_arity(mut self, nargs: Int, want: Int) -> Bool:
        if nargs != want:
            self.err = STATUS_EVAL
            return False
        return True

    def eval_function(mut self, e: Int32, cur: Int32) -> Int32:
        var name = self.exprs[Int(e)].s
        var nargs = len(self.exprs[Int(e)].args)
        # Evaluate value args; expref args stay as expression references.
        var values = List[Int32]()
        var is_expref = List[Bool]()
        for k in range(nargs):
            var ae = self.exprs[Int(e)].args[k]
            if self.exprs[Int(ae)].kind == E_EXPREF:
                values.append(ae)
                is_expref.append(True)
            else:
                var v = self.eval(ae, cur)
                if self.err != STATUS_OK:
                    return NULL_IDX
                values.append(v)
                is_expref.append(False)

        if name == "abs":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_abs(values, is_expref)
        if name == "avg":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_avg(values, is_expref)
        if name == "ceil":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_ceil_floor(values, is_expref, True)
        if name == "floor":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_ceil_floor(values, is_expref, False)
        if name == "contains":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_contains(values, is_expref)
        if name == "ends_with":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_starts_ends(values, is_expref, False)
        if name == "starts_with":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_starts_ends(values, is_expref, True)
        if name == "join":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_join(values, is_expref)
        if name == "keys":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_keys_values(values, is_expref, True)
        if name == "values":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_keys_values(values, is_expref, False)
        if name == "length":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_length(values, is_expref)
        if name == "map":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_map(values, is_expref)
        if name == "max":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_max_min(values, is_expref, True)
        if name == "min":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_max_min(values, is_expref, False)
        if name == "max_by":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_max_min_by(values, is_expref, True)
        if name == "min_by":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_max_min_by(values, is_expref, False)
        if name == "merge":
            if nargs < 1:
                self.err = STATUS_EVAL
                return NULL_IDX
            return self.fn_merge(values, is_expref)
        if name == "not_null":
            if nargs < 1:
                self.err = STATUS_EVAL
                return NULL_IDX
            for k in range(nargs):
                if is_expref[k]:
                    # expref is not null; reference returns the expression
                    # object, which the kernel cannot represent -> punt.
                    self.err = STATUS_UNSUPPORTED
                    return NULL_IDX
                if self.arena[Int(values[k])].tag != T_NULL:
                    return values[k]
            return NULL_IDX
        if name == "reverse":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_reverse(values, is_expref)
        if name == "sort":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_sort(values, is_expref)
        if name == "sort_by":
            if not self.check_arity(nargs, 2):
                return NULL_IDX
            return self.fn_sort_by(values, is_expref)
        if name == "sum":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_sum(values, is_expref)
        if name == "to_array":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            if is_expref[0]:
                self.err = STATUS_UNSUPPORTED
                return NULL_IDX
            var v = values[0]
            if self.arena[Int(v)].tag == T_ARR:
                return v
            var kids = List[Int32]()
            kids.append(v)
            return add_arr(self.arena, kids^)
        if name == "to_number":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            return self.fn_to_number(values, is_expref)
        if name == "to_string":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            if is_expref[0]:
                # reference stringifies a live object (non-deterministic).
                self.err = STATUS_UNSUPPORTED
                return NULL_IDX
            var v = values[0]
            if self.arena[Int(v)].tag == T_STR:
                return v
            var out = List[UInt8]()
            if not serialize_node(self.arena, v, out, True, True):
                self.err = STATUS_UNSUPPORTED
                return NULL_IDX
            var s = String()
            for k in range(len(out)):
                s.append(Codepoint(out[k]))
            return add_str(self.arena, s)
        if name == "type":
            if not self.check_arity(nargs, 1):
                return NULL_IDX
            if is_expref[0]:
                return NULL_IDX  # reference returns None for type(&expr)
            var tag = self.arena[Int(values[0])].tag
            if tag == T_TUPLE:
                return NULL_IDX  # tuples are not a JMESPath type (reference: null)
            var label = String("null")
            if tag == T_FALSE or tag == T_TRUE:
                label = String("boolean")
            elif tag == T_INT or tag == T_FLOAT:
                label = String("number")
            elif tag == T_STR:
                label = String("string")
            elif tag == T_ARR:
                label = String("array")
            elif tag == T_OBJ:
                label = String("object")
            return add_str(self.arena, label)
        # Unknown function: the fallback raises UnknownFunctionError.
        self.err = STATUS_EVAL
        return NULL_IDX

    def reject_expref(mut self, is_expref: List[Bool]) -> Bool:
        for k in range(len(is_expref)):
            if is_expref[k]:
                self.err = STATUS_EVAL
                return True
        return False

    def fn_abs(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if node.tag == T_INT:
            if node.i == I64_MIN:
                self.err = STATUS_UNSUPPORTED  # Python bigint result
                return NULL_IDX
            return add_int(self.arena, -node.i if node.i < 0 else node.i)
        if node.tag == T_FLOAT:
            var f = node.f
            return add_float(self.arena, -f if f < 0.0 else (0.0 if f == 0.0 else f))
        self.err = STATUS_EVAL
        return NULL_IDX

    def fn_ceil_floor(
        mut self, values: List[Int32], is_expref: List[Bool], is_ceil: Bool
    ) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if node.tag == T_INT:
            return v  # ceil/floor of an int is the same int
        if node.tag != T_FLOAT:
            self.err = STATUS_EVAL
            return NULL_IDX
        var f = node.f
        if f != f or f > 9.223372036854776e18 or f < -9.223372036854776e18:
            # NaN (reference raises ValueError) / inf or huge (OverflowError
            # or bigint): the fallback reproduces the exact behaviour.
            self.err = STATUS_UNSUPPORTED
            return NULL_IDX
        var t = Float64(Int64(f))  # truncate toward zero
        var r = t
        if is_ceil:
            if f > t:
                r = t + 1.0
        else:
            if f < t:
                r = t - 1.0
        return add_int(self.arena, Int64(r))

    def fn_contains(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var hay = values[0]
        var needle = values[1]
        ref node = self.arena[Int(hay)]
        if node.tag == T_ARR:
            var nh = len(node.kids)
            for k in range(nh):
                var el = self.arena[Int(hay)].kids[k]
                if self.deep_equal(el, needle):
                    if self.err != STATUS_OK:
                        return NULL_IDX
                    return TRUE_IDX
            return FALSE_IDX
        if node.tag == T_STR:
            if self.arena[Int(needle)].tag != T_STR:
                self.err = STATUS_EVAL  # reference raises TypeError
                return NULL_IDX
            ref hs = self.arena[Int(hay)].s
            ref ns = self.arena[Int(needle)].s
            var hb = hs.as_bytes()
            var nb = ns.as_bytes()
            var hn = len(hb)
            var nn = len(nb)
            if nn == 0:
                return TRUE_IDX
            if nn > hn:
                return FALSE_IDX
            for start in range(hn - nn + 1):
                var ok = True
                for j in range(nn):
                    if hb[start + j] != nb[j]:
                        ok = False
                        break
                if ok:
                    return TRUE_IDX
            return FALSE_IDX
        self.err = STATUS_EVAL
        return NULL_IDX

    def fn_starts_ends(
        mut self, values: List[Int32], is_expref: List[Bool], is_start: Bool
    ) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        if self.arena[Int(values[0])].tag != T_STR or self.arena[Int(values[1])].tag != T_STR:
            self.err = STATUS_EVAL
            return NULL_IDX
        ref s = self.arena[Int(values[0])].s
        ref p = self.arena[Int(values[1])].s
        var sb = s.as_bytes()
        var pb = p.as_bytes()
        var sn = len(sb)
        var pn = len(pb)
        if pn > sn:
            return FALSE_IDX
        var off = 0 if is_start else sn - pn
        for j in range(pn):
            if sb[off + j] != pb[j]:
                return FALSE_IDX
        return TRUE_IDX

    def fn_join(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        if self.arena[Int(values[0])].tag != T_STR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var arr = values[1]
        ref node = self.arena[Int(arr)]
        if node.tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        for k in range(len(node.kids)):
            if self.arena[Int(node.kids[k])].tag != T_STR:
                self.err = STATUS_EVAL
                return NULL_IDX
        var out = String()
        ref sep = self.arena[Int(values[0])].s
        var sep_cps = List[UInt32]()
        for cp in sep.codepoints():
            sep_cps.append(UInt32(cp))
        for k in range(len(node.kids)):
            if k > 0:
                for j in range(len(sep_cps)):
                    out.append(Codepoint(unsafe_unchecked_codepoint=sep_cps[j]))
            ref es = self.arena[Int(node.kids[k])].s
            for cp in es.codepoints():
                out.append(cp)
        return add_str(self.arena, out)

    def fn_keys_values(
        mut self, values: List[Int32], is_expref: List[Bool], want_keys: Bool
    ) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if node.tag != T_OBJ:
            self.err = STATUS_EVAL
            return NULL_IDX
        var out = List[Int32]()
        if want_keys:
            var nk = len(node.keys)
            for k in range(nk):
                var key = self.arena[Int(v)].keys[k]
                out.append(add_str(self.arena, key^))
        else:
            var nv = len(node.kids)
            for k in range(nv):
                out.append(self.arena[Int(v)].kids[k])
        return add_arr(self.arena, out^)

    def fn_length(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if node.tag == T_STR:
            var n = Int64(0)
            for cp in node.s.codepoints():
                n += 1
            return add_int(self.arena, n)
        if node.tag == T_ARR or node.tag == T_OBJ:
            return add_int(self.arena, Int64(len(node.kids)))
        self.err = STATUS_EVAL
        return NULL_IDX

    def fn_map(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if not is_expref[0] or is_expref[1]:
            self.err = STATUS_EVAL
            return NULL_IDX
        var arr = values[1]
        if self.arena[Int(arr)].tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var body = self.exprs[Int(values[0])].a
        var out = List[Int32]()
        var nk = len(self.arena[Int(arr)].kids)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            var r = self.eval(body, kid)
            if self.err != STATUS_OK:
                return NULL_IDX
            out.append(r)  # map keeps nulls (unlike a projection)
        return add_arr(self.arena, out^)

    def validate_number_or_string_array(
        mut self, arr: Int32
    ) -> Int32:
        """0 = uniform numbers, 1 = uniform strings, -1 = not an array,
        -2 = mixed/other (STATUS_EVAL). Empty arrays count as numbers."""
        ref node = self.arena[Int(arr)]
        if node.tag != T_ARR:
            return -1
        var cls = Int32(-1)
        for k in range(len(node.kids)):
            var tag = self.arena[Int(node.kids[k])].tag
            var this = Int32(0) if is_number_tag(tag) else (Int32(1) if tag == T_STR else Int32(2))
            if this == 2:
                self.err = STATUS_EVAL
                return -2
            if cls == -1:
                cls = this
            elif cls != this:
                self.err = STATUS_EVAL
                return -2
        return cls if cls != -1 else Int32(0)

    def fn_max_min(
        mut self, values: List[Int32], is_expref: List[Bool], want_max: Bool
    ) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var arr = values[0]
        var cls = self.validate_number_or_string_array(arr)
        if cls < 0:
            self.err = STATUS_EVAL
            return NULL_IDX
        var nk = len(self.arena[Int(arr)].kids)
        if nk == 0:
            return NULL_IDX
        var best = self.arena[Int(arr)].kids[0]
        for k in range(1, nk):
            var cand = self.arena[Int(arr)].kids[k]
            var better = False
            if cls == 0:
                var c = self.cmp_numbers(cand, best)
                if self.err != STATUS_OK:
                    return NULL_IDX
                better = c > 0 if want_max else (c < 0 if c != 2 else False)
            else:
                ref cs = self.arena[Int(cand)].s
                ref bs = self.arena[Int(best)].s
                better = bs < cs if want_max else cs < bs
            if better:
                best = cand
        return best

    def fn_max_min_by(
        mut self, values: List[Int32], is_expref: List[Bool], want_max: Bool
    ) -> Int32:
        if is_expref[0] or not is_expref[1]:
            self.err = STATUS_EVAL
            return NULL_IDX
        var arr = values[0]
        if self.arena[Int(arr)].tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var nk = len(self.arena[Int(arr)].kids)
        if nk == 0:
            return NULL_IDX
        var body = self.exprs[Int(values[1])].a
        var best_elem = Int32(0)
        var best_key = Int32(0)
        var cls = Int32(-1)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            var key = self.eval(body, kid)
            if self.err != STATUS_OK:
                return NULL_IDX
            var tag = self.arena[Int(key)].tag
            var this = Int32(0) if is_number_tag(tag) else (Int32(1) if tag == T_STR else Int32(2))
            if this == 2:
                self.err = STATUS_EVAL  # e.g. null keys are rejected
                return NULL_IDX
            if cls == -1:
                cls = this
            elif cls != this:
                self.err = STATUS_EVAL
                return NULL_IDX
            if k == 0:
                best_elem = kid
                best_key = key
                continue
            var better = False
            if cls == 0:
                var c = self.cmp_numbers(key, best_key)
                if self.err != STATUS_OK:
                    return NULL_IDX
                better = c > 0 if want_max else (c < 0 if c != 2 else False)
            else:
                ref cs = self.arena[Int(key)].s
                ref bs = self.arena[Int(best_key)].s
                better = bs < cs if want_max else cs < bs
            if better:
                best_elem = kid
                best_key = key
        return best_elem

    def fn_merge(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var keys = List[String]()
        var kids = List[Int32]()
        for a in range(len(values)):
            ref node = self.arena[Int(values[a])]
            if node.tag != T_OBJ:
                self.err = STATUS_EVAL
                return NULL_IDX
            for k in range(len(node.keys)):
                var found = -1
                for j in range(len(keys)):
                    if keys[j] == node.keys[k]:
                        found = j
                        break
                if found >= 0:
                    kids[found] = node.kids[k]  # overwrite keeps position
                else:
                    keys.append(node.keys[k])
                    kids.append(node.kids[k])
        return add_obj(self.arena, keys^, kids^)

    def fn_reverse(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if node.tag == T_ARR:
            var out = List[Int32]()
            var n = len(node.kids)
            for k in range(n):
                out.append(node.kids[n - 1 - k])
            return add_arr(self.arena, out^)
        if node.tag == T_STR:
            var cps = List[UInt32]()
            for cp in node.s.codepoints():
                cps.append(UInt32(cp))
            var out = String()
            var n = len(cps)
            for k in range(n):
                out.append(Codepoint(unsafe_unchecked_codepoint=cps[n - 1 - k]))
            return add_str(self.arena, out)
        self.err = STATUS_EVAL
        return NULL_IDX

    def fn_sort(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var arr = values[0]
        var cls = self.validate_number_or_string_array(arr)
        if cls < 0:
            self.err = STATUS_EVAL
            return NULL_IDX
        var kids = List[Int32]()
        var order = List[Int32]()
        var nk = len(self.arena[Int(arr)].kids)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            # NaN makes the reference's sorted() order unreproducible: punt.
            if self.arena[Int(kid)].tag == T_FLOAT and self.arena[Int(kid)].f != self.arena[Int(kid)].f:
                self.err = STATUS_UNSUPPORTED
                return NULL_IDX
            kids.append(kid)
            order.append(Int32(k))
        self.merge_sort(order, kids, cls)
        if self.err != STATUS_OK:
            return NULL_IDX
        var out = List[Int32]()
        for k in range(len(order)):
            out.append(kids[Int(order[k])])
        return add_arr(self.arena, out^)

    def fn_sort_by(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if is_expref[0] or not is_expref[1]:
            self.err = STATUS_EVAL
            return NULL_IDX
        var arr = values[0]
        if self.arena[Int(arr)].tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var body = self.exprs[Int(values[1])].a
        var kids = List[Int32]()
        var keys = List[Int32]()
        var cls = Int32(-1)
        var nk = len(self.arena[Int(arr)].kids)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            var key = self.eval(body, kid)
            if self.err != STATUS_OK:
                return NULL_IDX
            var tag = self.arena[Int(key)].tag
            if tag == T_FLOAT and self.arena[Int(key)].f != self.arena[Int(key)].f:
                self.err = STATUS_UNSUPPORTED  # NaN key
                return NULL_IDX
            var this = Int32(0) if is_number_tag(tag) else (Int32(1) if tag == T_STR else Int32(2))
            if this == 2:
                self.err = STATUS_EVAL  # null keys are rejected
                return NULL_IDX
            if cls == -1:
                cls = this
            elif cls != this:
                self.err = STATUS_EVAL
                return NULL_IDX
            kids.append(kid)
            keys.append(key)
        var order = List[Int32]()
        for k in range(len(keys)):
            order.append(Int32(k))
        self.merge_sort(order, keys, cls if cls != -1 else Int32(0))
        if self.err != STATUS_OK:
            return NULL_IDX
        var out = List[Int32]()
        for k in range(len(order)):
            out.append(kids[Int(order[k])])
        return add_arr(self.arena, out^)

    def merge_sort(mut self, mut order: List[Int32], keys: List[Int32], cls: Int32):
        """Stable merge sort of `order` by compare(keys[a], keys[b])."""
        var n = len(order)
        if n <= 1:
            return
        var aux = List[Int32]()
        for k in range(n):
            aux.append(order[k])
        self.merge_sort_range(order, aux, keys, cls, 0, n)

    def merge_sort_range(
        mut self,
        mut order: List[Int32],
        mut aux: List[Int32],
        keys: List[Int32],
        cls: Int32,
        lo: Int,
        hi: Int,
    ):
        if hi - lo <= 1:
            return
        var mid = lo + (hi - lo) // 2
        self.merge_sort_range(aux, order, keys, cls, lo, mid)
        self.merge_sort_range(aux, order, keys, cls, mid, hi)
        # merge aux[lo:mid] and aux[mid:hi] into order[lo:hi]
        var i = lo
        var j = mid
        var k = lo
        while i < mid and j < hi:
            # aux entries are positions; compare their key nodes.
            var take_left = True
            if cls == 0:
                var c = self.cmp_numbers(keys[Int(aux[i])], keys[Int(aux[j])])
                if self.err != STATUS_OK:
                    return
                take_left = c <= 0 or c == 2
            else:
                ref ls = self.arena[Int(keys[Int(aux[i])])].s
                ref rs = self.arena[Int(keys[Int(aux[j])])].s
                take_left = not (rs < ls)
            if take_left:
                order[k] = aux[i]
                i += 1
            else:
                order[k] = aux[j]
                j += 1
            k += 1
        while i < mid:
            order[k] = aux[i]
            i += 1
            k += 1
        while j < hi:
            order[k] = aux[j]
            j += 1
            k += 1

    def fn_sum(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var arr = values[0]
        if self.arena[Int(arr)].tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var acc = NumSum()
        var nk = len(self.arena[Int(arr)].kids)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            ref kn = self.arena[Int(kid)]
            acc.add(kn.tag, kn.i, kn.f)
            if acc.err != STATUS_OK:
                self.err = acc.err
                return NULL_IDX
        if acc.seen_float:
            return add_float(self.arena, acc.final())
        return add_int(self.arena, acc.iacc)

    def fn_avg(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var arr = values[0]
        if self.arena[Int(arr)].tag != T_ARR:
            self.err = STATUS_EVAL
            return NULL_IDX
        var acc = NumSum()
        var nk = len(self.arena[Int(arr)].kids)
        for k in range(nk):
            var kid = self.arena[Int(arr)].kids[k]
            ref kn = self.arena[Int(kid)]
            acc.add(kn.tag, kn.i, kn.f)
            if acc.err != STATUS_OK:
                self.err = acc.err
                return NULL_IDX
        var n = Int64(nk)
        if n == 0:
            return NULL_IDX  # avg of an empty array is null
        if acc.seen_float:
            return add_float(self.arena, acc.final() / Float64(n))
        if acc.iacc >= 9007199254740992 or acc.iacc <= -9007199254740992:
            # exact int/int division past float64 precision: punt
            self.err = STATUS_UNSUPPORTED
            return NULL_IDX
        return add_float(self.arena, Float64(acc.iacc) / Float64(n))

    def fn_to_number(mut self, values: List[Int32], is_expref: List[Bool]) -> Int32:
        if self.reject_expref(is_expref):
            return NULL_IDX
        var v = values[0]
        ref node = self.arena[Int(v)]
        if is_number_tag(node.tag):
            return v
        if node.tag != T_STR:
            return NULL_IDX
        var s_copy = node.s
        return self.parse_python_number(s_copy)

    def parse_python_number(mut self, s: String) -> Int32:
        """Python int()/float() attempt semantics (used by to_number):
        try int grammar, then float grammar, else null. Non-ASCII input and
        int64-overflowing integers punt to the fallback (Python handles
        unicode digits and arbitrary precision)."""
        var b = s.as_bytes()
        var lo = 0
        var hi = len(b)
        # strip ASCII whitespace (Python str.strip() set, ASCII subset)
        while lo < hi and (b[lo] == 0x20 or b[lo] == 0x09 or b[lo] == 0x0A or b[lo] == 0x0D or b[lo] == 0x0B or b[lo] == 0x0C):
            lo += 1
        while hi > lo and (b[hi - 1] == 0x20 or b[hi - 1] == 0x09 or b[hi - 1] == 0x0A or b[hi - 1] == 0x0D or b[hi - 1] == 0x0B or b[hi - 1] == 0x0C):
            hi -= 1
        if lo >= hi:
            return NULL_IDX
        for k in range(lo, hi):
            if b[k] >= 0x80:
                self.err = STATUS_UNSUPPORTED  # unicode digits/whitespace
                return NULL_IDX
        var start = lo
        var neg = False
        if b[lo] == 0x2B or b[lo] == 0x2D:  # + -
            neg = b[lo] == 0x2D
            lo += 1
        # inf / infinity / nan (case-insensitive), sign allowed
        var rest = hi - lo
        if self.word_matches(b, lo, hi, "inf") or self.word_matches(b, lo, hi, "infinity"):
            return add_float(self.arena, make_inf(neg))
        if self.word_matches(b, lo, hi, "nan"):
            return add_float(self.arena, make_nan())
        _ = rest
        # int grammar: digit (underscore? digit)*
        var int_ok = lo < hi
        var prev_digit = False
        var digit_count = 0
        var k = lo
        while k < hi:
            var c = b[k]
            if c >= 0x30 and c <= 0x39:
                prev_digit = True
                digit_count += 1
            elif c == 0x5F:  # _
                if not prev_digit:
                    int_ok = False
                    break
                prev_digit = False
            else:
                int_ok = False
                break
            k += 1
        if int_ok and not prev_digit:
            int_ok = False  # trailing underscore
        if int_ok and digit_count > 0:
            var acc = Int64(0)
            for i in range(lo, hi):
                var c = b[i]
                if c == 0x5F:
                    continue
                var d = Int64(c - 0x30)
                if acc > (I64_MAX - d) / 10:
                    self.err = STATUS_UNSUPPORTED  # Python bigint
                    return NULL_IDX
                acc = acc * 10 + d
            return add_int(self.arena, -acc if neg else acc)
        # float grammar (Python float() syntax: underscores between digits)
        var fs = String()
        if neg:
            fs.append(Codepoint(0x2D))
        var state = 0  # 0=int part, 1=frac, 2=exp
        var digits_int = 0
        var digits_frac = 0
        var digits_exp = 0
        var float_ok = True
        var j = lo
        while j < hi:
            var c = b[j]
            var pc = UInt8(0)
            if j > lo:
                pc = b[j - 1]
            var pc_is_digit = pc >= 0x30 and pc <= 0x39
            if c >= 0x30 and c <= 0x39:
                fs.append(Codepoint(c))
                if state == 0:
                    digits_int += 1
                elif state == 1:
                    digits_frac += 1
                else:
                    digits_exp += 1
            elif c == 0x5F:  # _ only between digits
                if not pc_is_digit:
                    float_ok = False
                    break
                if j + 1 >= hi:
                    float_ok = False
                    break
                var nc = b[j + 1]
                if nc < 0x30 or nc > 0x39:
                    float_ok = False
                    break
                # valid underscore: dropped from the cleaned string
            elif c == 0x2E and state == 0 and pc != 0x5F:  # .
                fs.append(Codepoint(c))
                state = 1
            elif (c == 0x65 or c == 0x45) and state != 2 and pc != 0x5F:  # e E
                fs.append(Codepoint(0x65))
                state = 2
            elif (c == 0x2B or c == 0x2D) and state == 2 and digits_exp == 0 and (pc == 0x65 or pc == 0x45):
                fs.append(Codepoint(c))  # exponent sign
            else:
                float_ok = False
                break
            j += 1
        if not float_ok or (digits_int == 0 and digits_frac == 0):
            return NULL_IDX
        if state == 2 and digits_exp == 0:
            return NULL_IDX  # 'e' with no exponent digits
        var value = Float64(0.0)
        try:
            value = Float64(fs)
        except:
            self.err = STATUS_UNSUPPORTED
            return NULL_IDX
        return add_float(self.arena, value)

    def word_matches(self, b: Span[UInt8, _], lo: Int, hi: Int, word: String) -> Bool:
        var wb = word.as_bytes()
        if hi - lo != len(wb):
            return False
        for k in range(len(wb)):
            var c = b[lo + k]
            var w = wb[k]
            # case-insensitive on ASCII letters
            if c >= 0x41 and c <= 0x5A:
                c += 0x20
            if w >= 0x41 and w <= 0x5A:
                w += 0x20
            if c != w:
                return False
        return True


# ---------------------------------------------------------------------------
# Result serializer (JSON; floats in Python-repr style so json.loads
# reproduces bit-identical values; ascii_mode escapes non-ASCII for
# to_string(), which mirrors json.dumps(ensure_ascii=True)).
# ---------------------------------------------------------------------------


def append_str_bytes(s: String, mut out: List[UInt8]):
    var b = s.as_bytes()
    for k in range(len(b)):
        out.append(b[k])


def append_hex4(cp: UInt32, mut out: List[UInt8]):
    var digits = String("0123456789abcdef")
    out.append(0x5C)  # \
    out.append(0x75)  # u
    out.append(digits.as_bytes()[Int((cp >> 12) & 0xF)])
    out.append(digits.as_bytes()[Int((cp >> 8) & 0xF)])
    out.append(digits.as_bytes()[Int((cp >> 4) & 0xF)])
    out.append(digits.as_bytes()[Int(cp & 0xF)])


def escape_string(s: String, mut out: List[UInt8], ascii_mode: Bool):
    out.append(0x22)  # "
    if not ascii_mode:
        var b = s.as_bytes()
        for k in range(len(b)):
            var c = b[k]
            if c == 0x22:
                out.append(0x5C)
                out.append(0x22)
            elif c == 0x5C:
                out.append(0x5C)
                out.append(0x5C)
            elif c == 0x08:
                out.append(0x5C)
                out.append(0x62)
            elif c == 0x09:
                out.append(0x5C)
                out.append(0x74)
            elif c == 0x0A:
                out.append(0x5C)
                out.append(0x6E)
            elif c == 0x0C:
                out.append(0x5C)
                out.append(0x66)
            elif c == 0x0D:
                out.append(0x5C)
                out.append(0x72)
            elif c < 0x20:
                append_hex4(UInt32(c), out)
            else:
                out.append(c)  # includes raw UTF-8 continuation bytes
    else:
        for cp in s.codepoints():
            var v = UInt32(cp)
            if v == 0x22:
                out.append(0x5C)
                out.append(0x22)
            elif v == 0x5C:
                out.append(0x5C)
                out.append(0x5C)
            elif v == 0x08:
                out.append(0x5C)
                out.append(0x62)
            elif v == 0x09:
                out.append(0x5C)
                out.append(0x74)
            elif v == 0x0A:
                out.append(0x5C)
                out.append(0x6E)
            elif v == 0x0C:
                out.append(0x5C)
                out.append(0x66)
            elif v == 0x0D:
                out.append(0x5C)
                out.append(0x72)
            elif v < 0x20:
                append_hex4(v, out)
            elif v < 0x80:
                out.append(UInt8(v))
            elif v <= 0xFFFF:
                append_hex4(v, out)
            else:
                # astral plane: UTF-16 surrogate pair (json.dumps style)
                var u = v - 0x10000
                append_hex4(0xD800 + (u >> 10), out)
                append_hex4(0xDC00 + (u & 0x3FF), out)
    out.append(0x22)  # "


def fmt_float(f: Float64) -> String:
    """Python-repr-compatible float formatting (Mojo's shortest round-trip
    layout matches CPython repr on finite values; non-finite get json's
    Python-extension spellings)."""
    if f != f:
        return String("NaN")
    if f > 1.7976931348623157e308:
        return String("Infinity")
    if f < -1.7976931348623157e308:
        return String("-Infinity")
    return String(f)


def serialize_node(
    arena: List[Node],
    idx: Int32,
    mut out: List[UInt8],
    ascii_mode: Bool,
    allow_tuple: Bool = False,
) -> Bool:
    """False when the value cannot be represented in the result JSON
    (Python tuples reach the top level only through the fallback)."""
    ref node = arena[Int(idx)]
    if node.tag == T_TUPLE:
        if not allow_tuple:
            return False
        # to_string() renders tuples as arrays, like json.dumps
        out.append(0x5B)  # [
        for k in range(len(node.kids)):
            if k > 0:
                out.append(0x2C)  # ,
            if not serialize_node(arena, node.kids[k], out, ascii_mode, allow_tuple):
                return False
        out.append(0x5D)  # ]
        return True
    if node.tag == T_NULL:
        append_str_bytes(String("null"), out)
        return True
    if node.tag == T_FALSE:
        append_str_bytes(String("false"), out)
        return True
    if node.tag == T_TRUE:
        append_str_bytes(String("true"), out)
        return True
    if node.tag == T_INT:
        append_str_bytes(String(node.i), out)
        return True
    if node.tag == T_FLOAT:
        append_str_bytes(fmt_float(node.f), out)
        return True
    if node.tag == T_STR:
        escape_string(node.s, out, ascii_mode)
        return True
    if node.tag == T_ARR:
        out.append(0x5B)  # [
        for k in range(len(node.kids)):
            if k > 0:
                out.append(0x2C)  # ,
            if not serialize_node(arena, node.kids[k], out, ascii_mode, allow_tuple):
                return False
        out.append(0x5D)  # ]
        return True
    # object
    out.append(0x7B)  # {
    for k in range(len(node.kids)):
        if k > 0:
            out.append(0x2C)  # ,
        escape_string(node.keys[k], out, ascii_mode)
        out.append(0x3A)  # :
        if not serialize_node(arena, node.kids[k], out, ascii_mode, allow_tuple):
            return False
    out.append(0x7D)  # }
    return True


# ---------------------------------------------------------------------------
# Exported C ABI
# ---------------------------------------------------------------------------


@export
def jmespathmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def jmespathmojo_search(
    expr_ptr: U8Ptr,
    expr_len: Int64,
    data_ptr: U8Ptr,
    data_len: Int64,
    out_buf: U8PtrPtr,
    out_len: I64Ptr,
) abi("C") -> Int32:
    if expr_len < 0 or data_len < 0:
        return STATUS_DATA
    var src = List[UInt8]()
    for k in range(Int(expr_len)):
        src.append(expr_ptr[unsafe_offset=k])
    var engine = Engine(src^)
    var root_expr = engine.parse()
    if engine.err != STATUS_OK:
        return engine.err
    # heuristic arena pre-allocation (nodes ~= payload/12, capped)
    var reserve = Int(data_len) // 12
    if reserve > 2_000_000:
        reserve = 2_000_000
    engine.arena.reserve(reserve + 64)
    var payload = List[UInt8]()
    payload.resize(Int(data_len), 0)
    unsafe_memcpy(dest=payload.unsafe_ptr(), src=data_ptr, count=Int(data_len))
    var parser = MarshalParser(payload^)
    var doc = parser.parse_document(engine.arena)
    if parser.failed:
        return STATUS_DATA
    var result = engine.eval(root_expr, doc)
    if engine.err != STATUS_OK:
        return engine.err
    var out = List[UInt8]()
    if not serialize_node(engine.arena, result, out, False):
        # tuples (or other non-JSON types) in the result: the vendored
        # fallback reproduces them exactly.
        return STATUS_UNSUPPORTED
    var buf = unsafe_alloc[UInt8](len(out))
    unsafe_memcpy(dest=buf, src=out.unsafe_ptr(), count=len(out))
    out_buf[unsafe_offset=0] = buf
    out_len[unsafe_offset=0] = Int64(len(out))
    return STATUS_OK


@export
def jmespathmojo_free(buf: U8Ptr, len: Int64) abi("C"):
    buf.unsafe_free()


# ---------------------------------------------------------------------------
# Marshal parser (CPython marshal format v4, the wrapper's wire format).
# Binary TLV: little-endian ints, raw 8-byte floats, raw UTF-8 strings —
# no text parsing anywhere, and exact Python types (tuples stay tuples,
# so the kernel mirrors the reference's list-vs-tuple distinction; anything
# outside the subset below punts to the vendored fallback).
# ---------------------------------------------------------------------------

comptime M_NULL: UInt8 = 0x30  # '0' (dict terminator only)
comptime M_NONE: UInt8 = 0x4E  # 'N'
comptime M_TRUE: UInt8 = 0x54  # 'T'
comptime M_FALSE: UInt8 = 0x46  # 'F'
comptime M_INT: UInt8 = 0x69  # 'i' (int32 LE)
comptime M_INT64: UInt8 = 0x49  # 'I' (int64 LE; legacy, accepted)
comptime M_LONG: UInt8 = 0x6C  # 'l' (arbitrary precision, base 2^15 digits)
comptime M_FLOAT: UInt8 = 0x66  # 'f' (text; not written by CPython 3)
comptime M_BINFLOAT: UInt8 = 0x67  # 'g' (8 bytes LE)
comptime M_STRING: UInt8 = 0x73  # 's' (bytes -> punt)
comptime M_UNICODE: UInt8 = 0x75  # 'u' (int32 len + UTF-8)
comptime M_ASCII: UInt8 = 0x61  # 'a'
comptime M_ASCII_INTERNED: UInt8 = 0x41  # 'A'
comptime M_SHORT_ASCII: UInt8 = 0x7A  # 'z'
comptime M_SHORT_ASCII_INTERNED: UInt8 = 0x5A  # 'Z'
comptime M_LIST: UInt8 = 0x5B  # '['
comptime M_TUPLE: UInt8 = 0x28  # '('
comptime M_SMALL_TUPLE: UInt8 = 0x29  # ')'
comptime M_DICT: UInt8 = 0x7B  # '{'
comptime M_SET: UInt8 = 0x3C  # '<'
comptime M_FROZENSET: UInt8 = 0x3E  # '>'
comptime M_REF: UInt8 = 0x72  # 'r'
comptime M_FLAG: UInt8 = 0x80


struct MarshalParser:
    """Parser for the wrapper's wire format. Owns the payload; strings are
    bulk-copied out of it (Python str is always valid UTF-8, so no
    re-validation is needed on this path)."""

    var data: List[UInt8]
    var pos: Int
    var failed: Bool
    var unsupported: Bool
    var refs: List[Int32]  # arena node indices of FLAG_REF objects, in order

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.pos = 0
        self.failed = False
        self.unsupported = False
        self.refs = List[Int32]()

    def at_end(self) -> Bool:
        return self.pos >= len(self.data)

    def read_u8(mut self) -> UInt8:
        if self.pos >= len(self.data):
            self.failed = True
            return 0
        var b = self.data[self.pos]
        self.pos += 1
        return b

    # Marshal integers/floats are little-endian (all supported platforms are
    # little-endian), so they are read with direct bitcasts — no byte loops.
    def read_i32(mut self) -> Int32:
        if self.pos + 4 > len(self.data):
            self.failed = True
            return 0
        var v = (self.data.unsafe_ptr() + self.pos).unsafe_bitcast[Int32]()[
            unsafe_offset=0
        ]
        self.pos += 4
        return v

    def read_i64(mut self) -> Int64:
        if self.pos + 8 > len(self.data):
            self.failed = True
            return 0
        var v = (self.data.unsafe_ptr() + self.pos).unsafe_bitcast[Int64]()[
            unsafe_offset=0
        ]
        self.pos += 8
        return v

    def read_float(mut self) -> Float64:
        if self.pos + 8 > len(self.data):
            self.failed = True
            return 0.0
        var v = (self.data.unsafe_ptr() + self.pos).unsafe_bitcast[Float64]()[
            unsafe_offset=0
        ]
        self.pos += 8
        return v

    def read_string(mut self, nbytes: Int) -> String:
        """Bulk-copy n payload bytes into a String (valid UTF-8 by
        construction: the payload comes from marshal.dumps of a Python
        document)."""
        var end = self.pos + nbytes
        if end > len(self.data):
            self.failed = True
            return String()
        var s = String(StringSpan(unsafe_from_utf8=Span(self.data)[self.pos:end]))
        self.pos = end
        return s

    def parse_document(mut self, mut arena: List[Node]) -> Int32:
        var root = self.read_obj(arena)
        if self.failed:
            return NULL_IDX
        if not self.at_end():
            self.failed = True
            return NULL_IDX
        return root

    def read_obj(mut self, mut arena: List[Node]) -> Int32:
        var code = self.read_u8()
        if self.failed:
            return NULL_IDX
        var flagged = (code & M_FLAG) != 0
        var t = code & ~M_FLAG
        # containers reserve their refs slot before reading children
        var slot = -1
        if flagged and (t == M_LIST or t == M_TUPLE or t == M_SMALL_TUPLE or t == M_DICT):
            slot = len(self.refs)
            self.refs.append(NULL_IDX)
        var node = NULL_IDX
        if t == M_NONE or t == M_NULL:
            node = NULL_IDX
        elif t == M_TRUE:
            node = TRUE_IDX
        elif t == M_FALSE:
            node = FALSE_IDX
        elif t == M_INT:
            node = add_int(arena, Int64(self.read_i32()))
        elif t == M_INT64:
            node = add_int(arena, self.read_i64())
        elif t == M_LONG:
            var ndigits = self.read_i32()
            var neg = ndigits < 0
            if neg:
                ndigits = -ndigits
            if ndigits > 4:
                # > 2^60: beyond int64 range guaranteed -> punt
                self.unsupported = True
                self.failed = True
                return NULL_IDX
            var acc = Int64(0)
            var shift = 0
            var overflow = False
            for _ in range(ndigits):
                # CPython writes base-2^15 digits as uint16 LE each
                if self.pos + 2 > len(self.data):
                    self.failed = True
                    return NULL_IDX
                var digit = Int64(self.data[self.pos]) | (
                    Int64(self.data[self.pos + 1]) << 8
                )
                self.pos += 2
                if digit >= 32768:
                    self.failed = True
                    return NULL_IDX
                acc += digit << Int64(shift)
                shift += 15
                if acc < 0:
                    overflow = True
            if self.failed:
                return NULL_IDX
            if overflow:
                self.unsupported = True
                self.failed = True
                return NULL_IDX
            node = add_int(arena, -acc if neg else acc)
        elif t == M_BINFLOAT:
            node = add_float(arena, self.read_float())
        elif t == M_FLOAT:
            # text form (not written by CPython 3); one-byte length
            var n = Int(self.read_u8())
            if self.failed or self.pos + n > len(self.data):
                self.failed = True
                return NULL_IDX
            var fs = String()
            for _ in range(n):
                fs.append(Codepoint(self.data[self.pos]))
                self.pos += 1
            var fv = Float64(0.0)
            try:
                fv = Float64(fs)
            except:
                self.failed = True
                return NULL_IDX
            node = add_float(arena, fv)
        elif t == M_UNICODE or t == M_ASCII or t == M_ASCII_INTERNED:
            var n = Int(self.read_i32())
            if self.failed or n < 0:
                self.failed = True
                return NULL_IDX
            node = add_str(arena, self.read_string(n))
        elif t == M_SHORT_ASCII or t == M_SHORT_ASCII_INTERNED:
            var n = Int(self.read_u8())
            if self.failed:
                return NULL_IDX
            node = add_str(arena, self.read_string(n))
        elif t == M_LIST:
            var n = Int(self.read_i32())
            if self.failed or n < 0:
                self.failed = True
                return NULL_IDX
            var kids = List[Int32](capacity=n)
            for _ in range(n):
                kids.append(self.read_obj(arena))
                if self.failed:
                    return NULL_IDX
            node = add_arr(arena, kids^)
        elif t == M_TUPLE or t == M_SMALL_TUPLE:
            var n = 0
            if t == M_SMALL_TUPLE:
                n = Int(self.read_u8())
            else:
                n = Int(self.read_i32())
            if self.failed or n < 0:
                self.failed = True
                return NULL_IDX
            var kids = List[Int32](capacity=n)
            for _ in range(n):
                kids.append(self.read_obj(arena))
                if self.failed:
                    return NULL_IDX
            arena.append(Node(T_TUPLE, kids=kids^))
            node = Int32(len(arena) - 1)
        elif t == M_DICT:
            node = self.read_dict(arena)
            if self.failed:
                return NULL_IDX
        elif t == M_REF:
            var ri = Int(self.read_i32())
            if self.failed or ri < 0 or ri >= len(self.refs):
                self.failed = True
                return NULL_IDX
            node = self.refs[ri]
            var rtag = arena[Int(node)].tag
            if rtag == T_ARR or rtag == T_OBJ or rtag == T_TUPLE:
                # container refs create cycles: outside the kernel subset
                self.unsupported = True
                self.failed = True
                return NULL_IDX
        else:
            # bytes, sets, code objects, and anything unknown: punt
            self.unsupported = True
            self.failed = True
            return NULL_IDX
        if flagged:
            if slot >= 0:
                self.refs[slot] = node
            else:
                self.refs.append(node)
        return node

    def read_dict(mut self, mut arena: List[Node]) -> Int32:
        """Dict entries: keys are read directly as strings (no arena node
        unless the key is ref-registered), values as full objects."""
        var keys = List[String]()
        var kids = List[Int32]()
        while True:
            if self.pos >= len(self.data):
                self.failed = True
                return NULL_IDX
            if self.data[self.pos] == M_NULL:
                self.pos += 1
                break
            var code = self.data[self.pos]
            var t = code & ~M_FLAG
            var key = String()
            if t == M_REF:
                self.pos += 1
                var ri = Int(self.read_i32())
                if self.failed or ri < 0 or ri >= len(self.refs):
                    self.failed = True
                    return NULL_IDX
                var knode = self.refs[ri]
                if arena[Int(knode)].tag != T_STR:
                    self.unsupported = True
                    self.failed = True
                    return NULL_IDX
                key = arena[Int(knode)].s
            elif t == M_UNICODE or t == M_ASCII or t == M_ASCII_INTERNED or t == M_SHORT_ASCII or t == M_SHORT_ASCII_INTERNED:
                self.pos += 1
                var n = 0
                if t == M_SHORT_ASCII or t == M_SHORT_ASCII_INTERNED:
                    n = Int(self.read_u8())
                else:
                    n = Int(self.read_i32())
                if self.failed or n < 0:
                    self.failed = True
                    return NULL_IDX
                key = self.read_string(n)
                if (code & M_FLAG) != 0:
                    # ref-registered key (interned): keep an arena node so
                    # later M_REF keys/values can resolve to it.
                    self.refs.append(add_str(arena, key))
            else:
                # non-string dict keys: outside the kernel subset
                self.unsupported = True
                self.failed = True
                return NULL_IDX
            var v = self.read_obj(arena)
            if self.failed:
                return NULL_IDX
            keys.append(key)
            kids.append(v)
        return add_obj(arena, keys^, kids^)

