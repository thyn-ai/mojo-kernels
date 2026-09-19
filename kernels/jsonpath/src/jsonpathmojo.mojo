"""Clean-room JSONPath query engine (h2non/jsonpath-ng semantics, ext dialect).

Written fresh from black-box-observed behavior of the published PyPI package
jsonpath_ng 1.7.0 (`jsonpath_ng.ext.parse(...).find(data)`): child steps
(dot/bracket), field unions, wildcards, recursive descent (DFS pre-order),
slices with step, and filter scripts with comparisons, `&` conjunctions and
`@` current-node references. No third-party Mojo code is used or adapted, and
no jsonpath_ng source was read; the semantics contract was established by
probing the oracle and is encoded in the differential test suite.

One-shot C ABI: the wrapper ships (expression, JSON text) in and gets the
matched paths back — values are resolved by the wrapper against the original
Python objects, so number/string handling here only has to be exact for
filter comparisons (strtod gives correctly-rounded float64, matching
CPython's float()).

    int32_t  jsonpathmojo_abi_version(void)
    void*    jsonpathmojo_find(expr, expr_len, json, json_len, status*)
    int64_t  jsonpathmojo_count(handle, which)
    void     jsonpathmojo_copy(handle, kinds, vals, path_off, str_off, str_buf)
    void     jsonpathmojo_destroy(handle)

status: 0 ok; 1 parse error (aux = byte offset); 2 TypeError from int()
coercion (aux = value tag); 3 TypeError from an ordering comparison
(aux = op*64 + ltag*8 + rtag); 4 IndexError (aux = 0 list, 1 string); 5
KeyError (aux = index); 6 ValueError slice-step-zero; 7 NotImplementedError
(existence left operand of `&`); 8 out-of-domain input, wrapper must fall
back (duplicate keys / >63-bit integers / nesting too deep); 9 TypeError
"object of type has no len()" (aux = tag).

Result step kinds: 0 = object field (val = output string id), 1 = sequence
index (val, may be negative; resolves as node[val]), 2 = dict-position
index (filter on dicts; resolves as values()[val]), 3 = render-only `[val]`
step that keeps the current node during value resolution (`[*]` on
dicts/strings/scalars yields the node itself at path [val]), 4 = filter
match on a list, 5 = recursive-descent list edge. Kinds 4/5 render like 1
but tell the wrapper the step came from a list-only operation, so data
shapes the oracle treats differently there (e.g. tuples) reroute to the
pure-Python engine.
"""

from std.collections import List
from std.ffi import external_call
from std.math import trunc
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# JSON node tags (also used as scalar value tags in filter evaluation).
comptime TAG_NULL: UInt8 = 0
comptime TAG_FALSE: UInt8 = 1
comptime TAG_TRUE: UInt8 = 2
comptime TAG_INT: UInt8 = 3
comptime TAG_DBL: UInt8 = 4
comptime TAG_STR: UInt8 = 5
comptime TAG_ARR: UInt8 = 6
comptime TAG_OBJ: UInt8 = 7

# Plan step kinds.
comptime S_CHILD: UInt8 = 0  # a = start in pf, b = count (field string ids)
comptime S_WILD: UInt8 = 1  # .* (object values only)
comptime S_INDEX: UInt8 = 2  # a = index
comptime S_IDXWILD: UInt8 = 3  # [*]
comptime S_SLICE: UInt8 = 4  # a/b/c = start/end/step, ABSENT = missing
comptime S_DESC: UInt8 = 5  # a = substep index in substeps
comptime S_FILTER: UInt8 = 6  # a = filter id

# Filter conjunct ops.
comptime OP_EQ: UInt8 = 0
comptime OP_NE: UInt8 = 1
comptime OP_LT: UInt8 = 2
comptime OP_LE: UInt8 = 3
comptime OP_GT: UInt8 = 4
comptime OP_GE: UInt8 = 5
comptime OP_EXISTS: UInt8 = 6

# Arith node kinds.
comptime A_LIT_INT: UInt8 = 0  # a = value
comptime A_LIT_DBL: UInt8 = 1  # a = bits
comptime A_LIT_STR: UInt8 = 2  # a = plan string id
comptime A_PATH: UInt8 = 3  # a = start in fpsteps, b = count
comptime A_ADD: UInt8 = 4  # a = left arith id, b = right arith id
comptime A_SUB: UInt8 = 5
comptime A_MUL: UInt8 = 6

# Filter-path step kinds.
comptime F_FIELD: UInt8 = 0  # a = plan string id
comptime F_INDEX: UInt8 = 1  # a = index
comptime F_SLICE: UInt8 = 2  # a/b/c = start/end/step, ABSENT = missing
comptime F_WILD: UInt8 = 3  # @.* (object values)
comptime F_IDXWILD: UInt8 = 4  # @[*]
comptime F_UNION: UInt8 = 5  # a = start in pf, b = count

# Output path step kinds (cross-checked by the wrapper).
comptime OUT_FIELD: UInt8 = 0
comptime OUT_INDEX: UInt8 = 1
comptime OUT_DICTPOS: UInt8 = 2
comptime OUT_SELFIDX: UInt8 = 3
comptime OUT_FILTERIDX: UInt8 = 4  # filter match on a list (resolver: list-only)
comptime OUT_DESCIDX: UInt8 = 5  # recursive-descent list edge (resolver: list-only)

# find() status codes (ABI contract above).
comptime ST_OK: Int32 = 0
comptime ST_PARSE: Int32 = 1
comptime ST_TE_INT: Int32 = 2
comptime ST_TE_ORD: Int32 = 3
comptime ST_INDEXERROR: Int32 = 4
comptime ST_KEYERROR: Int32 = 5
comptime ST_VE_SLICE: Int32 = 6
comptime ST_NOTIMPL: Int32 = 7
comptime ST_FALLBACK: Int32 = 8
comptime ST_TE_LEN: Int32 = 9

# Document flags.
comptime FLAG_DUP_KEYS: Int64 = 1
comptime FLAG_BIG_INT: Int64 = 2
comptime FLAG_DEEP: Int64 = 4

comptime MAX_DEPTH: Int32 = 512
comptime ABSENT: Int64 = Int64(-9223372036854775807) - 1  # INT64_MIN sentinel
comptime MAX_STR_REPEAT: Int64 = 67108864  # 64 MiB cap on string repetition

comptime CH_TAB = 0x09
comptime CH_LF = 0x0A
comptime CH_CR = 0x0D
comptime CH_VT = 0x0B
comptime CH_FF = 0x0C
comptime CH_SPACE = 0x20
comptime CH_BANG = 0x21
comptime CH_DQUOTE = 0x22
comptime CH_DOLLAR = 0x24
comptime CH_AMP = 0x26
comptime CH_SQUOTE = 0x27
comptime CH_LPAREN = 0x28
comptime CH_RPAREN = 0x29
comptime CH_STAR = 0x2A
comptime CH_PLUS = 0x2B
comptime CH_COMMA = 0x2C
comptime CH_MINUS = 0x2D
comptime CH_DOT = 0x2E
comptime CH_SLASH = 0x2F
comptime CH_0 = 0x30
comptime CH_9 = 0x39
comptime CH_COLON = 0x3A
comptime CH_LT = 0x3C
comptime CH_EQ = 0x3D
comptime CH_GT = 0x3E
comptime CH_QMARK = 0x3F
comptime CH_AT = 0x40
comptime CH_A_UPPER = 0x41
comptime CH_F_UPPER = 0x46
comptime CH_Z_UPPER = 0x5A
comptime CH_LBRACKET = 0x5B
comptime CH_BACKSLASH = 0x5C
comptime CH_RBRACKET = 0x5D
comptime CH_UNDERSCORE = 0x5F
comptime CH_A = 0x61
comptime CH_B = 0x62
comptime CH_E = 0x65
comptime CH_F = 0x66
comptime CH_L = 0x6C
comptime CH_N = 0x6E
comptime CH_R = 0x72
comptime CH_S = 0x73
comptime CH_T = 0x74
comptime CH_U = 0x75
comptime CH_Z = 0x7A
comptime CH_LBRACE = 0x7B
comptime CH_RBRACE = 0x7D


struct JNode(ImplicitlyCopyable, Copyable, Movable):
    """One JSON value in the document arena."""

    var tag: UInt8
    var a: Int64  # INT: value; DBL: bits; STR: sbuf offset; ARR/OBJ: first child slot
    var b: Int64  # STR: byte length; ARR: item count; OBJ: member count

    def __init__(out self, tag: UInt8, a: Int64, b: Int64):
        self.tag = tag
        self.a = a
        self.b = b


struct Member(ImplicitlyCopyable, Copyable, Movable):
    """One object member: key bytes in sbuf plus the value node id."""

    var koff: Int64
    var klen: Int64
    var vnode: Int32

    def __init__(out self, koff: Int64, klen: Int64, vnode: Int32):
        self.koff = koff
        self.klen = klen
        self.vnode = vnode


struct PStep(ImplicitlyCopyable, Copyable, Movable):
    var kind: UInt8
    var a: Int64
    var b: Int64
    var c: Int64

    def __init__(out self, kind: UInt8, a: Int64, b: Int64, c: Int64):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c


struct Conj(ImplicitlyCopyable, Copyable, Movable):
    """One filter conjunct: left arith, op, right arith (unused for EXISTS)."""

    var left: Int32
    var op: UInt8
    var right: Int32

    def __init__(out self, left: Int32, op: UInt8, right: Int32):
        self.left = left
        self.op = op
        self.right = right


struct Arith(ImplicitlyCopyable, Copyable, Movable):
    var kind: UInt8
    var a: Int64
    var b: Int64

    def __init__(out self, kind: UInt8, a: Int64, b: Int64):
        self.kind = kind
        self.a = a
        self.b = b


struct FpStep(ImplicitlyCopyable, Copyable, Movable):
    var kind: UInt8
    var a: Int64
    var b: Int64
    var c: Int64

    def __init__(out self, kind: UInt8, a: Int64, b: Int64, c: Int64):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c


struct Filter(ImplicitlyCopyable, Copyable, Movable):
    var start: Int64  # first conjunct in conjs
    var count: Int64

    def __init__(out self, start: Int64, count: Int64):
        self.start = start
        self.count = count


struct V(ImplicitlyCopyable, Copyable, Movable):
    """A scalar value in filter evaluation; strings reference an arena."""

    var tag: UInt8  # TAG_*; TAG_ARR/TAG_OBJ mark container (arith) results
    var i: Int64  # int value / double bits / bool 0-1
    var so: Int64  # string offset
    var sl: Int64  # string byte length
    var src: UInt8  # 0 = doc sbuf, 1 = vbuf, 2 = plan string table

    def __init__(out self, tag: UInt8, i: Int64, so: Int64, sl: Int64, src: UInt8):
        self.tag = tag
        self.i = i
        self.so = so
        self.sl = sl
        self.src = src


struct Engine:
    """Whole per-call state: document arena, query plan, evaluation arenas."""

    # input buffers (borrowed)
    var expr: U8Ptr
    var elen: Int64
    var json: U8Ptr
    var jlen: Int64
    # cursor shared by both parsers (phases are sequential)
    var pos: Int64
    var status: Int32
    var aux: Int64
    var str_off: Int64  # last parsed string span (JSON parser)
    var str_len: Int64
    var big_coerce: Bool  # int() coercion hit |v| >= 2^63
    var arith_poison: Bool  # an arith pair raised (caught): operand yields nothing

    # document arena (children/members hold CONTIGUOUS per-node runs;
    # the scratch stacks keep runs contiguous while nested values parse)
    var nodes: List[JNode]
    var children: List[Int32]
    var members: List[Member]
    var arr_scratch: List[Int32]
    var obj_scratch: List[Member]
    var sbuf: List[UInt8]
    var flags: Int64

    # plan arena
    var steps: List[PStep]
    var substeps: List[PStep]  # descend substeps live outside the main chain
    var pf: List[Int32]  # field string ids for S_CHILD / F_UNION
    var pstr_off: List[Int64]  # plan string table offsets into pstr_buf
    var pstr_buf: List[UInt8]
    var filters: List[Filter]
    var conjs: List[Conj]
    var ariths: List[Arith]
    var fpsteps: List[FpStep]

    # output string table (plan fields + document keys), open-addressed dedup
    var ostr_off: List[Int64]
    var ostr_buf: List[UInt8]
    var ostr_hash: List[Int64]  # capacity power of two; <0 empty, else string id+1

    # path arena (linked list): node 0 is the empty root path
    var pprev: List[Int32]
    var pkind: List[UInt8]
    var pval: List[Int64]

    # current result set (parallel arrays)
    var cur_nodes: List[Int32]
    var cur_paths: List[Int32]
    var nxt_nodes: List[Int32]
    var nxt_paths: List[Int32]

    # filter evaluation scratch
    var vbuf: List[UInt8]
    var vals: List[V]

    def __init__(out self, expr: U8Ptr, elen: Int64, json: U8Ptr, jlen: Int64):
        self.expr = expr
        self.elen = elen
        self.json = json
        self.jlen = jlen
        self.pos = 0
        self.status = ST_OK
        self.aux = 0
        self.str_off = 0
        self.str_len = 0
        self.big_coerce = False
        self.arith_poison = False
        self.nodes = List[JNode]()
        self.children = List[Int32]()
        self.members = List[Member]()
        self.arr_scratch = List[Int32]()
        self.obj_scratch = List[Member]()
        self.sbuf = List[UInt8]()
        self.flags = 0
        self.steps = List[PStep]()
        self.substeps = List[PStep]()
        self.pf = List[Int32]()
        self.pstr_off = List[Int64]()
        self.pstr_buf = List[UInt8]()
        self.filters = List[Filter]()
        self.conjs = List[Conj]()
        self.ariths = List[Arith]()
        self.fpsteps = List[FpStep]()
        self.ostr_off = List[Int64]()
        self.ostr_buf = List[UInt8]()
        self.ostr_hash = List[Int64]()
        self.pprev = List[Int32]()
        self.pkind = List[UInt8]()
        self.pval = List[Int64]()
        self.cur_nodes = List[Int32]()
        self.cur_paths = List[Int32]()
        self.nxt_nodes = List[Int32]()
        self.nxt_paths = List[Int32]()
        self.vbuf = List[UInt8]()
        self.vals = List[V]()


# ---------------------------------------------------------------------------
# Output string interning (all reads go through locals so the source arena
# never aliases the output arena across a mutation)
# ---------------------------------------------------------------------------


comptime FNV_SEED: UInt64 = 1469598103934665603
comptime FNV_MULT: UInt64 = 1099511628211


def _field_byte(self: Engine, from_sbuf: Bool, i: Int64) -> UInt8:
    if from_sbuf:
        return self.sbuf[Int(i)]
    return self.pstr_buf[Int(i)]


def _ostr_rehash(mut self: Engine):
    var nmask = Int64(len(self.ostr_hash)) - 1
    for sid2 in range(len(self.ostr_off) - 1):
        var so2 = self.ostr_off[Int(sid2)]
        var se2 = self.ostr_off[Int(sid2 + 1)]
        var hh = FNV_SEED
        for i in range(so2, se2):
            hh = (hh ^ UInt64(self.ostr_buf[Int(i)])) * FNV_MULT
        var s2 = Int64(hh & UInt64(nmask))
        while self.ostr_hash[Int(s2)] >= 0:
            s2 = (s2 + 1) & nmask
        self.ostr_hash[Int(s2)] = Int64(sid2) + 1


def ostr_intern_field(mut self: Engine, from_sbuf: Bool, off: Int64, n: Int64) -> Int64:
    """Intern n bytes of sbuf (from_sbuf) or pstr_buf at off; return the id."""
    if len(self.ostr_off) == 0:
        self.ostr_off.append(0)
        self.ostr_hash.resize(64, -1)
    var h = FNV_SEED
    for i in range(off, off + n):
        h = (h ^ UInt64(_field_byte(self, from_sbuf, i))) * FNV_MULT
    var mask = Int64(len(self.ostr_hash)) - 1
    var slot = Int64(h & UInt64(mask))
    while True:
        var entry = self.ostr_hash[Int(slot)]
        if entry < 0:
            break
        var sid = entry - 1
        var so = self.ostr_off[Int(sid)]
        var se = self.ostr_off[Int(sid + 1)]
        if se - so == n:
            var same = True
            for i in range(n):
                if self.ostr_buf[Int(so + i)] != _field_byte(self, from_sbuf, off + i):
                    same = False
                    break
            if same:
                return sid
        slot = (slot + 1) & mask
    if (len(self.ostr_off) - 1) * 4 >= len(self.ostr_hash) * 3:
        self.ostr_hash.resize(len(self.ostr_hash) * 2, -1)
        _ostr_rehash(self)
        return ostr_intern_field(self, from_sbuf, off, n)
    var sid = Int64(len(self.ostr_off) - 1)
    for i in range(off, off + n):
        self.ostr_buf.append(_field_byte(self, from_sbuf, i))
    self.ostr_off.append(Int64(len(self.ostr_buf)))
    var mask2 = Int64(len(self.ostr_hash)) - 1
    var slot2 = Int64(h & UInt64(mask2))
    while self.ostr_hash[Int(slot2)] >= 0:
        slot2 = (slot2 + 1) & mask2
    self.ostr_hash[Int(slot2)] = sid + 1
    return sid


# ---------------------------------------------------------------------------
# JSON parser (document order preserved; strings decoded to UTF-8 bytes)
# ---------------------------------------------------------------------------


def _jpeek(self: Engine) -> Int32:
    if self.pos >= self.jlen:
        return -1
    return Int32(self.json[unsafe_offset=self.pos])


def _jws(mut self: Engine):
    while self.pos < self.jlen:
        var c = self.json[unsafe_offset=self.pos]
        if c == CH_SPACE or c == CH_TAB or c == CH_LF or c == CH_CR:
            self.pos += 1
        else:
            return


def _hex4(mut self: Engine, mut ok: Bool) -> Int32:
    """Parse 4 hex digits at pos; return the code unit (or -1)."""
    var v = Int32(0)
    for _ in range(4):
        if self.pos >= self.jlen:
            ok = False
            return -1
        var c = self.json[unsafe_offset=self.pos]
        self.pos += 1
        var d = Int32(-1)
        if c >= CH_0 and c <= CH_9:
            d = Int32(c - CH_0)
        elif c >= CH_A and c <= CH_F:
            d = Int32(c - CH_A) + 10
        elif c >= CH_A_UPPER and c <= CH_F_UPPER:
            d = Int32(c - CH_A_UPPER) + 10
        if d < 0:
            ok = False
            return -1
        v = v * 16 + d
    return v


def _append_codepoint(mut self: Engine, cp: Int32):
    """Append one Unicode scalar value as UTF-8; lone surrogates become
    CESU-8 (3-byte) so they round-trip through Python's surrogatepass."""
    var u = UInt32(cp)
    if u < 0x80:
        self.sbuf.append(UInt8(u))
    elif u < 0x800:
        self.sbuf.append(UInt8(0xC0 | (u >> 6)))
        self.sbuf.append(UInt8(0x80 | (u & 0x3F)))
    elif u < 0x10000:
        self.sbuf.append(UInt8(0xE0 | (u >> 12)))
        self.sbuf.append(UInt8(0x80 | ((u >> 6) & 0x3F)))
        self.sbuf.append(UInt8(0x80 | (u & 0x3F)))
    else:
        self.sbuf.append(UInt8(0xF0 | (u >> 18)))
        self.sbuf.append(UInt8(0x80 | ((u >> 12) & 0x3F)))
        self.sbuf.append(UInt8(0x80 | ((u >> 6) & 0x3F)))
        self.sbuf.append(UInt8(0x80 | (u & 0x3F)))


def j_parse_string(mut self: Engine) -> Bool:
    """Parse a JSON string at pos (at the quote); decoded span lands in
    self.str_off/self.str_len."""
    self.pos += 1
    self.str_off = Int64(len(self.sbuf))
    while True:
        if self.pos >= self.jlen:
            return False
        var c = self.json[unsafe_offset=self.pos]
        if c == CH_DQUOTE:
            self.pos += 1
            self.str_len = Int64(len(self.sbuf)) - self.str_off
            return True
        if c == CH_BACKSLASH:
            self.pos += 1
            if self.pos >= self.jlen:
                return False
            var e = self.json[unsafe_offset=self.pos]
            self.pos += 1
            if e == CH_DQUOTE or e == CH_BACKSLASH or e == CH_SLASH:
                self.sbuf.append(e)
            elif e == CH_B:
                self.sbuf.append(0x08)
            elif e == CH_F:
                self.sbuf.append(0x0C)
            elif e == CH_N:
                self.sbuf.append(0x0A)
            elif e == CH_R:
                self.sbuf.append(0x0D)
            elif e == CH_T:
                self.sbuf.append(0x09)
            elif e == CH_U:
                var ok = True
                var cp = _hex4(self, ok)
                if not ok:
                    return False
                if cp >= 0xD800 and cp <= 0xDBFF:
                    if (
                        self.pos + 1 < self.jlen
                        and self.json[unsafe_offset=self.pos] == CH_BACKSLASH
                        and self.json[unsafe_offset=self.pos + 1] == CH_U
                    ):
                        self.pos += 2
                        var ok2 = True
                        var lo = _hex4(self, ok2)
                        if not ok2:
                            return False
                        if lo >= 0xDC00 and lo <= 0xDFFF:
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                        else:
                            # lone high surrogate followed by a non-low unit:
                            # emit the high surrogate as CESU-8, then the unit
                            _append_codepoint(self, cp)
                            cp = lo
                _append_codepoint(self, cp)
            else:
                return False
        else:
            self.sbuf.append(c)
            self.pos += 1


def _strtod(text: U8Ptr, start: Int64, tok_len: Int64) -> Int64:
    """Correctly-rounded decimal→float64 via libc strtod; returns the IEEE
    bits. Tokens longer than the scratch are truncated (the big-int flag
    makes the wrapper fall back anyway, so the value is then irrelevant)."""
    var buf = unsafe_alloc[UInt8](128)
    var n = tok_len
    if n > 127:
        n = 127
    for i in range(n):
        buf[unsafe_offset=i] = text[unsafe_offset=start + i]
    buf[unsafe_offset=Int(n)] = 0
    var end = unsafe_alloc[U8Ptr](1)
    var v = external_call["strtod", Float64](buf, end)
    buf.unsafe_free()
    end.unsafe_free()
    return Pointer(to=v).unsafe_bitcast[Int64]()[]


def j_parse_number(mut self: Engine):
    """Parse a JSON number token at pos; push a TAG_INT or TAG_DBL node."""
    var start = self.pos
    var p = self.pos
    var is_float = False
    if p < self.jlen and self.json[unsafe_offset=p] == CH_MINUS:
        p += 1
    while p < self.jlen and self.json[unsafe_offset=p] >= CH_0 and self.json[unsafe_offset=p] <= CH_9:
        p += 1
    if p < self.jlen and self.json[unsafe_offset=p] == CH_DOT:
        is_float = True
        p += 1
        while p < self.jlen and self.json[unsafe_offset=p] >= CH_0 and self.json[unsafe_offset=p] <= CH_9:
            p += 1
    if p < self.jlen and (self.json[unsafe_offset=p] == CH_E or self.json[unsafe_offset=p] == CH_E + 32):
        is_float = True
        p += 1
        if p < self.jlen and (self.json[unsafe_offset=p] == CH_PLUS or self.json[unsafe_offset=p] == CH_MINUS):
            p += 1
        while p < self.jlen and self.json[unsafe_offset=p] >= CH_0 and self.json[unsafe_offset=p] <= CH_9:
            p += 1
    var tok_len = p - start
    self.pos = p
    if not is_float:
        var neg = self.json[unsafe_offset=start] == CH_MINUS
        var ds = start + (Int64(1) if neg else Int64(0))
        var ndig = p - ds
        if ndig <= 18:
            var v = Int64(0)
            for i in range(ds, p):
                v = v * 10 + Int64(self.json[unsafe_offset=i] - CH_0)
            self.nodes.append(JNode(TAG_INT, -v if neg else v, 0))
        elif ndig == 19:
            var v = UInt64(0)
            var ovf = False
            for i in range(ds, p):
                var d = UInt64(self.json[unsafe_offset=i] - CH_0)
                if v > (UInt64(18446744073709551615) - d) // 10:
                    ovf = True
                    break
                v = v * 10 + d
            var limit = UInt64(9223372036854775807) if not neg else UInt64(9223372036854775808)
            if ovf or v > limit:
                self.flags |= FLAG_BIG_INT
                self.nodes.append(JNode(TAG_DBL, _strtod(self.json, start, tok_len), 0))
            else:
                var vv = Int64(v)
                self.nodes.append(JNode(TAG_INT, -vv if neg else vv, 0))
        else:
            self.flags |= FLAG_BIG_INT
            self.nodes.append(JNode(TAG_DBL, _strtod(self.json, start, tok_len), 0))
    else:
        self.nodes.append(JNode(TAG_DBL, _strtod(self.json, start, tok_len), 0))


def j_parse_value(mut self: Engine, depth: Int32) -> Int32:
    """Parse one JSON value; return its node id, or -1 on error."""
    if depth > MAX_DEPTH:
        self.flags |= FLAG_DEEP
    _jws(self)
    var c = _jpeek(self)
    if c < 0:
        return -1
    var idx = Int32(len(self.nodes))
    if c == CH_LBRACE:
        self.pos += 1
        var scratch_start = Int64(len(self.obj_scratch))
        self.nodes.append(JNode(TAG_OBJ, 0, 0))  # patched below
        _jws(self)
        if _jpeek(self) == CH_RBRACE:
            self.pos += 1
        else:
            while True:
                _jws(self)
                if _jpeek(self) != CH_DQUOTE:
                    return -1
                if not j_parse_string(self):
                    return -1
                var koff = self.str_off
                var klen = self.str_len
                var mcount = Int64(len(self.obj_scratch)) - scratch_start
                for mi in range(scratch_start, scratch_start + mcount):
                    var m = self.obj_scratch[Int(mi)]
                    if m.klen == klen:
                        var same = True
                        for i in range(klen):
                            if self.sbuf[Int(m.koff + i)] != self.sbuf[Int(koff + i)]:
                                same = False
                                break
                        if same:
                            self.flags |= FLAG_DUP_KEYS
                            break
                _jws(self)
                if _jpeek(self) != CH_COLON:
                    return -1
                self.pos += 1
                var vnode = j_parse_value(self, depth + 1)
                if vnode < 0:
                    return -1
                self.obj_scratch.append(Member(koff, klen, vnode))
                _jws(self)
                var nc = _jpeek(self)
                if nc == CH_COMMA:
                    self.pos += 1
                    continue
                if nc == CH_RBRACE:
                    self.pos += 1
                    break
                return -1
        var mstart = Int64(len(self.members))
        var mcount = Int64(len(self.obj_scratch)) - scratch_start
        for mi in range(scratch_start, scratch_start + mcount):
            self.members.append(self.obj_scratch[Int(mi)])
        self.obj_scratch.resize(Int(scratch_start), Member(0, 0, 0))
        self.nodes[Int(idx)] = JNode(TAG_OBJ, mstart, mcount)
        return idx
    elif c == CH_LBRACKET:
        self.pos += 1
        var scratch_start = Int64(len(self.arr_scratch))
        self.nodes.append(JNode(TAG_ARR, 0, 0))
        _jws(self)
        if _jpeek(self) == CH_RBRACKET:
            self.pos += 1
        else:
            while True:
                var vnode = j_parse_value(self, depth + 1)
                if vnode < 0:
                    return -1
                self.arr_scratch.append(vnode)
                _jws(self)
                var nc = _jpeek(self)
                if nc == CH_COMMA:
                    self.pos += 1
                    continue
                if nc == CH_RBRACKET:
                    self.pos += 1
                    break
                return -1
        var cstart = Int64(len(self.children))
        var ccount = Int64(len(self.arr_scratch)) - scratch_start
        for ci in range(scratch_start, scratch_start + ccount):
            self.children.append(self.arr_scratch[Int(ci)])
        self.arr_scratch.resize(Int(scratch_start), 0)
        self.nodes[Int(idx)] = JNode(TAG_ARR, cstart, ccount)
        return idx
    elif c == CH_DQUOTE:
        if not j_parse_string(self):
            return -1
        self.nodes.append(JNode(TAG_STR, self.str_off, self.str_len))
        return idx
    elif c == CH_T:
        if self.pos + 4 <= self.jlen and self.json[unsafe_offset=self.pos + 1] == CH_R and self.json[unsafe_offset=self.pos + 2] == CH_U and self.json[unsafe_offset=self.pos + 3] == CH_E:
            self.pos += 4
            self.nodes.append(JNode(TAG_TRUE, 1, 0))
            return idx
        return -1
    elif c == CH_F:
        if self.pos + 5 <= self.jlen and self.json[unsafe_offset=self.pos + 1] == CH_A and self.json[unsafe_offset=self.pos + 2] == CH_L and self.json[unsafe_offset=self.pos + 3] == CH_S and self.json[unsafe_offset=self.pos + 4] == CH_E:
            self.pos += 5
            self.nodes.append(JNode(TAG_FALSE, 0, 0))
            return idx
        return -1
    elif c == CH_N:
        if self.pos + 4 <= self.jlen and self.json[unsafe_offset=self.pos + 1] == CH_U and self.json[unsafe_offset=self.pos + 2] == CH_L and self.json[unsafe_offset=self.pos + 3] == CH_L:
            self.pos += 4
            self.nodes.append(JNode(TAG_NULL, 0, 0))
            return idx
        return -1
    elif c == CH_MINUS or (c >= CH_0 and c <= CH_9):
        j_parse_number(self)
        return Int32(len(self.nodes)) - 1
    return -1


# ---------------------------------------------------------------------------
# Expression parser
# ---------------------------------------------------------------------------


def _peek(self: Engine) -> Int32:
    if self.pos >= self.elen:
        return -1
    return Int32(self.expr[unsafe_offset=self.pos])


def _peek_at(self: Engine, k: Int64) -> Int32:
    if self.pos + k >= self.elen:
        return -1
    return Int32(self.expr[unsafe_offset=self.pos + k])


def _ws(mut self: Engine):
    while self.pos < self.elen:
        var c = self.expr[unsafe_offset=self.pos]
        if c == CH_SPACE or c == CH_TAB or c == CH_LF or c == CH_CR:
            self.pos += 1
        else:
            return


def _is_id_start(c: Int32) -> Bool:
    return (c >= CH_A and c <= CH_Z) or (c >= CH_A_UPPER and c <= CH_Z_UPPER) or c == CH_UNDERSCORE


def _is_id_char(c: Int32) -> Bool:
    return _is_id_start(c) or (c >= CH_0 and c <= CH_9)


def _is_digit(c: Int32) -> Bool:
    return c >= CH_0 and c <= CH_9


def perr(mut self: Engine) -> Int32:
    self.status = ST_PARSE
    self.aux = self.pos
    return -1


def _pstr_begin(mut self: Engine):
    if len(self.pstr_off) == 0:
        self.pstr_off.append(0)


def _pstr_finish(mut self: Engine) -> Int32:
    self.pstr_off.append(Int64(len(self.pstr_buf)))
    return Int32(len(self.pstr_off) - 2)


def plan_str_is_star(self: Engine, sid: Int32) -> Bool:
    var so = self.pstr_off[Int(sid)]
    var se = self.pstr_off[Int(sid + 1)]
    return se - so == 1 and self.pstr_buf[Int(so)] == CH_STAR


def parse_quoted(mut self: Engine) -> Int32:
    """Parse a quoted string at pos into the plan table; return its id.
    A backslash escapes the next character literally (oracle rule)."""
    var quote = self.expr[unsafe_offset=self.pos]
    self.pos += 1
    _pstr_begin(self)
    while True:
        if self.pos >= self.elen:
            _ = perr(self)
            return -1
        var c = self.expr[unsafe_offset=self.pos]
        if c == quote:
            self.pos += 1
            return _pstr_finish(self)
        if c == CH_BACKSLASH:
            self.pos += 1
            if self.pos >= self.elen:
                _ = perr(self)
                return -1
            self.pstr_buf.append(self.expr[unsafe_offset=self.pos])
            self.pos += 1
        else:
            self.pstr_buf.append(c)
            self.pos += 1


def parse_ident(mut self: Engine) -> Int32:
    """Parse an ID token at pos into the plan table; return its id."""
    _pstr_begin(self)
    while self.pos < self.elen and _is_id_char(Int32(self.expr[unsafe_offset=self.pos])):
        self.pstr_buf.append(self.expr[unsafe_offset=self.pos])
        self.pos += 1
    return _pstr_finish(self)


def plan_str_eq(self: Engine, sid: Int32, lit: String) -> Bool:
    var so = self.pstr_off[Int(sid)]
    var se = self.pstr_off[Int(sid + 1)]
    var lb = lit.as_bytes()
    if se - so != Int64(len(lb)):
        return False
    for i in range(len(lb)):
        if self.pstr_buf[Int(so + Int64(i))] != lb[Int(i)]:
            return False
    return True


def plan_string_from_int(mut self: Engine, v: Int64) -> Int32:
    """Plan string id for the decimal rendering of v (dot-number children)."""
    _pstr_begin(self)
    var s = String(v)
    var sb = s.as_bytes()
    for i in range(len(sb)):
        self.pstr_buf.append(sb[i])
    return _pstr_finish(self)


def parse_uint(mut self: Engine) -> Int64:
    """Parse digits at pos (no sign); caller checks digit presence."""
    var v = Int64(0)
    while self.pos < self.elen and _is_digit(Int32(self.expr[unsafe_offset=self.pos])):
        v = v * 10 + Int64(self.expr[unsafe_offset=self.pos] - CH_0)
        self.pos += 1
    return v


def parse_number_token(mut self: Engine) -> Int32:
    """Parse a (possibly negative) INT or FLOAT literal in a filter;
    appends an arith node and returns its id. Exponents are rejected,
    matching the oracle lexer."""
    var neg = False
    if _peek(self) == CH_MINUS:
        neg = True
        self.pos += 1
    if not _is_digit(_peek(self)):
        _ = perr(self)
        return -1
    var iv = parse_uint(self)
    if _peek(self) == CH_DOT:
        self.pos += 1
        if not _is_digit(_peek(self)):
            _ = perr(self)
            return -1
        var start = self.pos
        var frac = parse_uint(self)
        var digits = self.pos - start
        var dv = Float64(iv)
        var scale = Float64(1.0)
        for _ in range(digits):
            dv *= 10.0
            scale *= 10.0
        dv = (dv + Float64(frac)) / scale
        if neg:
            dv = -dv
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_LIT_DBL, Pointer(to=dv).unsafe_bitcast[Int64]()[], 0))
        return aid
    if _is_id_start(_peek(self)) or _peek(self) == CH_E or _peek(self) == CH_E + 32:
        # e.g. 1e2 — the oracle lexes 'e2' as an ID and fails at parse time
        _ = perr(self)
        return -1
    if neg:
        iv = -iv
    var aid = Int32(len(self.ariths))
    self.ariths.append(Arith(A_LIT_INT, iv, 0))
    return aid


def parse_index_or_slice(mut self: Engine, in_filter: Bool) -> Int32:
    """Parse `[int]` or `[a:b[:c]]` body (cursor inside the brackets)."""
    var has_start = False
    var start_v = Int64(0)
    var neg = False
    if _peek(self) == CH_MINUS:
        neg = True
        self.pos += 1
    if _is_digit(_peek(self)):
        start_v = parse_uint(self)
        has_start = True
        if neg:
            start_v = -start_v
    elif neg:
        _ = perr(self)
        return -1
    _ws(self)
    if _peek(self) == CH_COLON:
        self.pos += 1
        _ws(self)
        var has_end = False
        var end_v = Int64(0)
        var neg2 = False
        if _peek(self) == CH_MINUS:
            neg2 = True
            self.pos += 1
        if _is_digit(_peek(self)):
            end_v = parse_uint(self)
            has_end = True
            if neg2:
                end_v = -end_v
        elif neg2:
            _ = perr(self)
            return -1
        _ws(self)
        var step_v = ABSENT
        if _peek(self) == CH_COLON:
            self.pos += 1
            _ws(self)
            var neg3 = False
            if _peek(self) == CH_MINUS:
                neg3 = True
                self.pos += 1
            if _is_digit(_peek(self)):
                step_v = parse_uint(self)
                if neg3:
                    step_v = -step_v
            elif neg3:
                _ = perr(self)
                return -1
            _ws(self)
        if _peek(self) != CH_RBRACKET:
            _ = perr(self)
            return -1
        self.pos += 1
        if in_filter:
            self.fpsteps.append(FpStep(F_SLICE, start_v if has_start else ABSENT, end_v if has_end else ABSENT, step_v))
        else:
            self.steps.append(PStep(S_SLICE, start_v if has_start else ABSENT, end_v if has_end else ABSENT, step_v))
        return 0
    if not has_start:
        _ = perr(self)
        return -1
    if _peek(self) != CH_RBRACKET:
        _ = perr(self)
        return -1
    self.pos += 1
    if in_filter:
        self.fpsteps.append(FpStep(F_INDEX, start_v, 0, 0))
    else:
        self.steps.append(PStep(S_INDEX, start_v, 0, 0))
    return 0


def parse_arith(mut self: Engine) -> Int32:
    """farith := fterm ((+|-) fterm)*"""
    var left = parse_term(self)
    if left < 0:
        return -1
    while True:
        _ws(self)
        var c = _peek(self)
        if c != CH_PLUS and c != CH_MINUS:
            return left
        self.pos += 1
        var right = parse_term(self)
        if right < 0:
            return -1
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_ADD if c == CH_PLUS else A_SUB, Int64(left), Int64(right)))
        left = aid


def parse_term(mut self: Engine) -> Int32:
    """fterm := ffactor ('*' ffactor)*"""
    var left = parse_factor(self)
    if left < 0:
        return -1
    while True:
        _ws(self)
        if _peek(self) != CH_STAR:
            return left
        self.pos += 1
        var right = parse_factor(self)
        if right < 0:
            return -1
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_MUL, Int64(left), Int64(right)))
        left = aid


def parse_factor(mut self: Engine) -> Int32:
    """ffactor := NUMBER | FLOAT | STRING | true | false | ID | @fpath | ( farith )"""
    _ws(self)
    var c = _peek(self)
    if c == CH_AT:
        self.pos += 1
        return parse_fpath(self)
    if c == CH_SQUOTE or c == CH_DQUOTE:
        var sid = parse_quoted(self)
        if sid < 0:
            return -1
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_LIT_STR, Int64(sid), 0))
        return aid
    if c == CH_LPAREN:
        self.pos += 1
        var inner = parse_arith(self)
        if inner < 0:
            return -1
        _ws(self)
        if _peek(self) != CH_RPAREN:
            _ = perr(self)
            return -1
        self.pos += 1
        return inner
    if c == CH_MINUS or _is_digit(c):
        return parse_number_token(self)
    if _is_id_start(c):
        var sid = parse_ident(self)
        if plan_str_eq(self, sid, "true"):
            var aid = Int32(len(self.ariths))
            self.ariths.append(Arith(A_LIT_INT, 1, 0))
            return aid
        if plan_str_eq(self, sid, "false"):
            var aid = Int32(len(self.ariths))
            self.ariths.append(Arith(A_LIT_INT, 0, 0))
            return aid
        # any other bare word (null, None, foo, ...) is a string literal
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_LIT_STR, Int64(sid), 0))
        return aid
    _ = perr(self)
    return -1


def parse_filter_literal(mut self: Engine) -> Int32:
    """The right operand of a filter comparison: a single literal — number,
    quoted string, true/false, or bare word (no parens, no @, no arith),
    matching the oracle grammar."""
    _ws(self)
    var c = _peek(self)
    if c == CH_MINUS or _is_digit(c):
        return parse_number_token(self)
    if c == CH_SQUOTE or c == CH_DQUOTE:
        var sid = parse_quoted(self)
        if sid < 0:
            return -1
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_LIT_STR, Int64(sid), 0))
        return aid
    if _is_id_start(c):
        var sid = parse_ident(self)
        if plan_str_eq(self, sid, "true"):
            var aid = Int32(len(self.ariths))
            self.ariths.append(Arith(A_LIT_INT, 1, 0))
            return aid
        if plan_str_eq(self, sid, "false"):
            var aid = Int32(len(self.ariths))
            self.ariths.append(Arith(A_LIT_INT, 0, 0))
            return aid
        var aid = Int32(len(self.ariths))
        self.ariths.append(Arith(A_LIT_STR, Int64(sid), 0))
        return aid
    _ = perr(self)
    return -1


def parse_fpath(mut self: Engine) -> Int32:
    """Parse the steps after `@`; appends an A_PATH arith node and returns
    its id. An empty path (`@` alone) refers to the current node itself."""
    var start = Int64(len(self.fpsteps))
    var count = Int64(0)
    while True:
        var c = _peek(self)
        if c == CH_DOT:
            if _peek_at(self, 1) == CH_DOT:
                break
            self.pos += 1
            _ws(self)
            var nc = _peek(self)
            if nc == CH_STAR:
                self.pos += 1
                self.fpsteps.append(FpStep(F_WILD, 0, 0, 0))
                count += 1
            elif _is_id_start(nc):
                var sid = parse_ident(self)
                self.fpsteps.append(FpStep(F_FIELD, Int64(sid), 0, 0))
                count += 1
            elif _is_digit(nc):
                var v = parse_uint(self)
                var sid = plan_string_from_int(self, v)
                self.fpsteps.append(FpStep(F_FIELD, Int64(sid), 0, 0))
                count += 1
            else:
                _ = perr(self)
                return -1
        elif c == CH_LBRACKET:
            self.pos += 1
            _ws(self)
            var nc = _peek(self)
            if nc == CH_STAR:
                self.pos += 1
                _ws(self)
                if _peek(self) != CH_RBRACKET:
                    _ = perr(self)
                    return -1
                self.pos += 1
                self.fpsteps.append(FpStep(F_IDXWILD, 0, 0, 0))
                count += 1
            elif nc == CH_SQUOTE or nc == CH_DQUOTE:
                var sid = parse_quoted(self)
                if sid < 0:
                    return -1
                if plan_str_is_star(self, sid):
                    # @['*'] == @.* (dict-values wildcard)
                    _ws(self)
                    if _peek(self) == CH_COMMA:
                        while _peek(self) == CH_COMMA:
                            self.pos += 1
                            _ws(self)
                            if _peek(self) != CH_SQUOTE and _peek(self) != CH_DQUOTE:
                                _ = perr(self)
                                return -1
                            var sid2 = parse_quoted(self)
                            if sid2 < 0:
                                return -1
                            _ws(self)
                    if _peek(self) != CH_RBRACKET:
                        _ = perr(self)
                        return -1
                    self.pos += 1
                    self.fpsteps.append(FpStep(F_WILD, 0, 0, 0))
                    count += 1
                elif _peek(self) == CH_COMMA:
                    var fs = Int64(len(self.pf))
                    self.pf.append(sid)
                    var has_star = False
                    while _peek(self) == CH_COMMA:
                        self.pos += 1
                        _ws(self)
                        if _peek(self) != CH_SQUOTE and _peek(self) != CH_DQUOTE:
                            _ = perr(self)
                            return -1
                        var sid2 = parse_quoted(self)
                        if sid2 < 0:
                            return -1
                        if plan_str_is_star(self, sid2):
                            has_star = True
                        self.pf.append(sid2)
                        _ws(self)
                    if _peek(self) != CH_RBRACKET:
                        _ = perr(self)
                        return -1
                    self.pos += 1
                    self.fpsteps.append(FpStep(F_WILD if has_star else F_UNION, fs, Int64(len(self.pf)) - fs, 0))
                    count += 1
                elif _peek(self) == CH_RBRACKET:
                    self.pos += 1
                    self.fpsteps.append(FpStep(F_FIELD, Int64(sid), 0, 0))
                    count += 1
                else:
                    _ = perr(self)
                    return -1
            elif nc == CH_MINUS or _is_digit(nc) or nc == CH_COLON:
                if parse_index_or_slice(self, in_filter=True) < 0:
                    return -1
                count += 1
            else:
                _ = perr(self)
                return -1
        else:
            break
    var aid = Int32(len(self.ariths))
    self.ariths.append(Arith(A_PATH, start, count))
    return aid


def arith_has_path(self: Engine, aid: Int32) -> Bool:
    var a = self.ariths[Int(aid)]
    if a.kind == A_PATH:
        return True
    if a.kind == A_ADD or a.kind == A_SUB or a.kind == A_MUL:
        return arith_has_path(self, Int32(a.a)) or arith_has_path(self, Int32(a.b))
    return False


def parse_filter(mut self: Engine) -> Int32:
    """Parse `(expr)` after `?`; append a Filter; return its id."""
    if _peek(self) != CH_LPAREN:
        _ = perr(self)
        return -1
    self.pos += 1
    var cstart = Int64(len(self.conjs))
    var count = Int64(0)
    while True:
        var left = parse_arith(self)
        if left < 0:
            return -1
        _ws(self)
        var c = _peek(self)
        var op = OP_EXISTS
        var right = Int32(-1)
        if c == CH_EQ and _peek_at(self, 1) == CH_EQ:
            op = OP_EQ
            self.pos += 2
        elif c == CH_BANG and _peek_at(self, 1) == CH_EQ:
            op = OP_NE
            self.pos += 2
        elif c == CH_LT and _peek_at(self, 1) == CH_EQ:
            op = OP_LE
            self.pos += 2
        elif c == CH_GT and _peek_at(self, 1) == CH_EQ:
            op = OP_GE
            self.pos += 2
        elif c == CH_LT:
            op = OP_LT
            self.pos += 1
        elif c == CH_GT:
            op = OP_GT
            self.pos += 1
        if op != OP_EXISTS:
            right = parse_filter_literal(self)
            if right < 0:
                return -1
        self.conjs.append(Conj(left, op, right))
        count += 1
        _ws(self)
        if _peek(self) == CH_AMP:
            self.pos += 1
            continue
        if _peek(self) == CH_RPAREN:
            self.pos += 1
            break
        _ = perr(self)
        return -1
    var fid = Int32(len(self.filters))
    self.filters.append(Filter(cstart, count))
    return fid


def parse_dot_child(mut self: Engine) -> Int32:
    """After '.': '*' | ID | digits."""
    _ws(self)
    var c = _peek(self)
    if c == CH_STAR:
        self.pos += 1
        self.steps.append(PStep(S_WILD, 0, 0, 0))
        return 0
    if _is_id_start(c):
        var sid = parse_ident(self)
        var fs = Int64(len(self.pf))
        self.pf.append(sid)
        self.steps.append(PStep(S_CHILD, fs, 1, 0))
        return 0
    if _is_digit(c):
        var v = parse_uint(self)
        var sid = plan_string_from_int(self, v)
        var fs = Int64(len(self.pf))
        self.pf.append(sid)
        self.steps.append(PStep(S_CHILD, fs, 1, 0))
        return 0
    _ = perr(self)
    return -1


def parse_bracket(mut self: Engine, after_descend: Bool) -> Int32:
    """After '[': '*', index, slice, quoted-field union, or filter (filters
    are rejected after '..', matching the oracle)."""
    _ws(self)
    var c = _peek(self)
    if c == CH_STAR:
        self.pos += 1
        _ws(self)
        if _peek(self) != CH_RBRACKET:
            _ = perr(self)
            return -1
        self.pos += 1
        self.steps.append(PStep(S_IDXWILD, 0, 0, 0))
        return 0
    if c == CH_QMARK:
        if after_descend:
            _ = perr(self)
            return -1
        self.pos += 1
        var fid = parse_filter(self)
        if fid < 0:
            return -1
        _ws(self)
        if _peek(self) != CH_RBRACKET:
            _ = perr(self)
            return -1
        self.pos += 1
        self.steps.append(PStep(S_FILTER, Int64(fid), 0, 0))
        return 0
    if c == CH_SQUOTE or c == CH_DQUOTE:
        var sid = parse_quoted(self)
        if sid < 0:
            return -1
        var fs = Int64(len(self.pf))
        self.pf.append(sid)
        var n = Int64(1)
        var has_star = plan_str_is_star(self, sid)
        _ws(self)
        while _peek(self) == CH_COMMA:
            self.pos += 1
            _ws(self)
            if _peek(self) != CH_SQUOTE and _peek(self) != CH_DQUOTE:
                _ = perr(self)
                return -1
            var sid2 = parse_quoted(self)
            if sid2 < 0:
                return -1
            if plan_str_is_star(self, sid2):
                has_star = True
            self.pf.append(sid2)
            n += 1
            _ws(self)
        if _peek(self) != CH_RBRACKET:
            _ = perr(self)
            return -1
        self.pos += 1
        # a quoted '*' — alone or inside a union — is a dict wildcard
        self.steps.append(PStep(S_WILD if has_star else S_CHILD, fs, n, 0))
        return 0
    if c == CH_MINUS or _is_digit(c) or c == CH_COLON:
        return parse_index_or_slice(self, in_filter=False)
    _ = perr(self)
    return -1


def parse_descend_substep(mut self: Engine) -> Int32:
    """Parse the step after '..' into the substeps arena; return its index."""
    _ws(self)
    var c = _peek(self)
    var idx = Int32(len(self.substeps))
    if c == CH_STAR:
        self.pos += 1
        self.substeps.append(PStep(S_WILD, 0, 0, 0))
        return idx
    if _is_id_start(c):
        var sid = parse_ident(self)
        var fs = Int64(len(self.pf))
        self.pf.append(sid)
        self.substeps.append(PStep(S_CHILD, fs, 1, 0))
        return idx
    if _is_digit(c):
        var v = parse_uint(self)
        var sid = plan_string_from_int(self, v)
        var fs = Int64(len(self.pf))
        self.pf.append(sid)
        self.substeps.append(PStep(S_CHILD, fs, 1, 0))
        return idx
    if c == CH_LBRACKET:
        self.pos += 1
        _ws(self)
        var nc = _peek(self)
        if nc == CH_QMARK:
            _ = perr(self)  # the oracle rejects `..[?(...)]`
            return -1
        if nc == CH_STAR:
            self.pos += 1
            _ws(self)
            if _peek(self) != CH_RBRACKET:
                _ = perr(self)
                return -1
            self.pos += 1
            self.substeps.append(PStep(S_IDXWILD, 0, 0, 0))
            return idx
        if nc == CH_SQUOTE or nc == CH_DQUOTE:
            var sid = parse_quoted(self)
            if sid < 0:
                return -1
            var fs = Int64(len(self.pf))
            self.pf.append(sid)
            var n = Int64(1)
            var has_star = plan_str_is_star(self, sid)
            _ws(self)
            while _peek(self) == CH_COMMA:
                self.pos += 1
                _ws(self)
                if _peek(self) != CH_SQUOTE and _peek(self) != CH_DQUOTE:
                    _ = perr(self)
                    return -1
                var sid2 = parse_quoted(self)
                if sid2 < 0:
                    return -1
                if plan_str_is_star(self, sid2):
                    has_star = True
                self.pf.append(sid2)
                n += 1
                _ws(self)
            if _peek(self) != CH_RBRACKET:
                _ = perr(self)
                return -1
            self.pos += 1
            self.substeps.append(PStep(S_WILD if has_star else S_CHILD, fs, n, 0))
            return idx
        if nc == CH_MINUS or _is_digit(nc) or nc == CH_COLON:
            var saved = Int32(len(self.steps))
            if parse_index_or_slice(self, in_filter=False) < 0:
                return -1
            # move the just-appended step into the substeps arena
            var st = self.steps[Int(saved)]
            _ = self.steps.pop()
            self.substeps.append(st)
            return idx
        _ = perr(self)
        return -1
    _ = perr(self)
    return -1


def parse_expr(mut self: Engine) -> Bool:
    """Parse the whole expression into the plan arena."""
    _ws(self)
    if _peek(self) == CH_DOLLAR:
        self.pos += 1
    while True:
        _ws(self)
        var c = _peek(self)
        if c < 0:
            return True
        if c == CH_DOT:
            if _peek_at(self, 1) == CH_DOT:
                self.pos += 2
                var sub = parse_descend_substep(self)
                if sub < 0:
                    return False
                self.steps.append(PStep(S_DESC, Int64(sub), 0, 0))
            else:
                self.pos += 1
                if parse_dot_child(self) < 0:
                    return False
        elif c == CH_LBRACKET:
            self.pos += 1
            if parse_bracket(self, after_descend=False) < 0:
                return False
        elif len(self.steps) == 0 and (c == CH_STAR or c == CH_SQUOTE or c == CH_DQUOTE or _is_id_start(c)):
            # implicit root: the oracle accepts a leading step without '$'
            if c == CH_STAR:
                self.pos += 1
                self.steps.append(PStep(S_WILD, 0, 0, 0))
            elif _is_id_start(c):
                var sid1 = parse_ident(self)
                var fs1 = Int64(len(self.pf))
                self.pf.append(sid1)
                self.steps.append(PStep(S_CHILD, fs1, 1, 0))
            else:
                var sid2 = parse_quoted(self)
                if sid2 < 0:
                    return False
                var fs2 = Int64(len(self.pf))
                self.pf.append(sid2)
                self.steps.append(PStep(S_CHILD, fs2, 1, 0))
        else:
            self.status = ST_PARSE
            self.aux = self.pos
            return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def new_path(mut self: Engine, parent: Int32, kind: UInt8, val: Int64) -> Int32:
    var pid = Int32(len(self.pprev))
    self.pprev.append(parent)
    self.pkind.append(kind)
    self.pval.append(val)
    return pid


def emit(mut self: Engine, node: Int32, path: Int32):
    self.nxt_nodes.append(node)
    self.nxt_paths.append(path)


def node_falsy(self: Engine, node: Int32) -> Bool:
    var n = self.nodes[Int(node)]
    if n.tag == TAG_NULL or n.tag == TAG_FALSE:
        return True
    if n.tag == TAG_TRUE:
        return False
    if n.tag == TAG_INT:
        return n.a == 0
    if n.tag == TAG_DBL:
        return (n.a & Int64(9223372036854775807)) == 0  # +/-0.0
    # STR / ARR / OBJ: falsy when empty
    return n.b == 0


def apply_child(mut self: Engine, step: PStep, node: Int32, path: Int32):
    var n = self.nodes[Int(node)]
    if n.tag != TAG_OBJ:
        return
    for fi in range(step.a, step.a + step.b):
        var sid = self.pf[Int(fi)]
        var fso = self.pstr_off[Int(sid)]
        var fse = self.pstr_off[Int(sid + 1)]
        var flen = fse - fso
        for mi in range(n.a, n.a + n.b):
            var m = self.members[Int(mi)]
            if m.klen == flen:
                var same = True
                for i in range(flen):
                    if self.sbuf[Int(m.koff + i)] != self.pstr_buf[Int(fso + i)]:
                        same = False
                        break
                if same:
                    var osid = ostr_intern_field(self, False, fso, flen)
                    emit(self, m.vnode, new_path(self, path, OUT_FIELD, osid))
                    break


def apply_wild(mut self: Engine, node: Int32, path: Int32):
    var n = self.nodes[Int(node)]
    if n.tag != TAG_OBJ:
        return
    for mi in range(n.a, n.a + n.b):
        var m = self.members[Int(mi)]
        var osid = ostr_intern_field(self, True, m.koff, m.klen)
        emit(self, m.vnode, new_path(self, path, OUT_FIELD, osid))


def make_char_node(mut self: Engine, off: Int64, blen: Int64, char_idx: Int64) -> Int32:
    """Node for the one-code-point string at code-point index char_idx."""
    var pos = off
    var cp = Int64(0)
    var start = off
    while cp <= char_idx and pos < off + blen:
        start = pos
        var ch = self.sbuf[Int(pos)]
        var adv = Int64(1)
        if ch >= 0xF0:
            adv = 4
        elif ch >= 0xE0:
            adv = 3
        elif ch >= 0xC0:
            adv = 2
        pos += adv
        cp += 1
    var idx = Int32(len(self.nodes))
    self.nodes.append(JNode(TAG_STR, start, pos - start))
    return idx


def apply_index(mut self: Engine, idx: Int64, node: Int32, path: Int32):
    if node_falsy(self, node):
        return
    var n = self.nodes[Int(node)]
    if n.tag == TAG_ARR or n.tag == TAG_STR:
        if idx >= 0:
            if idx < n.b:
                if n.tag == TAG_ARR:
                    var child = self.children[Int(n.a + idx)]
                    var cpath = new_path(self, path, OUT_INDEX, idx)
                    emit(self, child, cpath)
                else:
                    # code-point indexing on the decoded string
                    var cp = Int64(0)
                    var pos = n.a
                    var start = n.a
                    while cp <= idx and pos < n.a + n.b:
                        start = pos
                        var ch = self.sbuf[Int(pos)]
                        var adv = Int64(1)
                        if ch >= 0xF0:
                            adv = 4
                        elif ch >= 0xE0:
                            adv = 3
                        elif ch >= 0xC0:
                            adv = 2
                        pos += adv
                        cp += 1
                    var nid = Int32(len(self.nodes))
                    self.nodes.append(JNode(TAG_STR, start, pos - start))
                    var cpath = new_path(self, path, OUT_INDEX, idx)
                    emit(self, nid, cpath)
            return
        if idx < -n.b:
            self.status = ST_INDEXERROR
            self.aux = 0 if n.tag == TAG_ARR else 1
            return
        if n.tag == TAG_ARR:
            var child = self.children[Int(n.a + n.b + idx)]
            var cpath = new_path(self, path, OUT_INDEX, idx)
            emit(self, child, cpath)
        else:
            # negative code-point index: count from the end
            var total = Int64(0)
            var pos = n.a
            while pos < n.a + n.b:
                var ch = self.sbuf[Int(pos)]
                var adv = Int64(1)
                if ch >= 0xF0:
                    adv = 4
                elif ch >= 0xE0:
                    adv = 3
                elif ch >= 0xC0:
                    adv = 2
                pos += adv
                total += 1
            var target = total + idx
            if target < 0:
                self.status = ST_INDEXERROR
                self.aux = 1
                return
            var cp = Int64(0)
            var pos2 = n.a
            var start = n.a
            while cp <= target and pos2 < n.a + n.b:
                start = pos2
                var ch = self.sbuf[Int(pos2)]
                var adv = Int64(1)
                if ch >= 0xF0:
                    adv = 4
                elif ch >= 0xE0:
                    adv = 3
                elif ch >= 0xC0:
                    adv = 2
                pos2 += adv
                cp += 1
            var nid = Int32(len(self.nodes))
            self.nodes.append(JNode(TAG_STR, start, pos2 - start))
            var cpath = new_path(self, path, OUT_INDEX, idx)
            emit(self, nid, cpath)
        return
    if n.tag == TAG_OBJ:
        if idx >= 0 and idx >= n.b:
            return
        self.status = ST_KEYERROR
        self.aux = idx
        return
    # truthy scalar: len() fails
    self.status = ST_TE_LEN
    self.aux = Int64(n.tag)


def apply_idxwild(mut self: Engine, node: Int32, path: Int32):
    var n = self.nodes[Int(node)]
    if n.tag == TAG_ARR:
        for i in range(n.b):
            var child = self.children[Int(n.a + i)]
            var cp = new_path(self, path, OUT_INDEX, i)
            emit(self, child, cp)
        return
    if node_falsy(self, node):
        return
    if n.tag == TAG_DBL:
        self.status = ST_TE_LEN
        self.aux = Int64(TAG_DBL)
        return
    # truthy dict/str/int/bool: the node itself at [0]
    var cp2 = new_path(self, path, OUT_SELFIDX, 0)
    emit(self, node, cp2)


def slice_step(mut self: Engine, s: Int64, e: Int64, st: Int64, length: Int64, node: Int32, path: Int32, arr_start: Int64):
    """Emit array children over Python slice semantics (indices resolved)."""
    var ss = s
    var ee = e
    if st > 0:
        if ss == ABSENT:
            ss = 0
        if ee == ABSENT:
            ee = length
        if ss < 0:
            ss += length
            if ss < 0:
                ss = 0
        if ee < 0:
            ee += length
            if ee < 0:
                ee = 0
        if ss > length:
            ss = length
        if ee > length:
            ee = length
        var i = ss
        while i < ee:
            var child = self.children[Int(arr_start + i)]
            var cp = new_path(self, path, OUT_INDEX, i)
            emit(self, child, cp)
            i += st
    else:
        if ss == ABSENT:
            ss = length - 1
        if ee == ABSENT:
            ee = -length - 1
        if ss < 0:
            ss += length
            if ss < 0:
                ss = -1
        if ee < 0:
            ee += length
            if ee < 0:
                ee = -1
        if ss >= length:
            ss = length - 1
        if ee >= length:
            ee = length - 1
        var i = ss
        while i > ee:
            var child = self.children[Int(arr_start + i)]
            var cp = new_path(self, path, OUT_INDEX, i)
            emit(self, child, cp)
            i += st


def apply_slice(mut self: Engine, step: PStep, node: Int32, path: Int32):
    if node_falsy(self, node):
        return
    var stepv = step.c if step.c != ABSENT else Int64(1)
    if stepv == 0:
        self.status = ST_VE_SLICE
        return
    var n = self.nodes[Int(node)]
    if n.tag != TAG_ARR:
        return
    slice_step(self, step.a, step.b, stepv, n.b, node, path, n.a)


def apply_descend(mut self: Engine, sub: PStep, node: Int32, path: Int32):
    """DFS pre-order: apply the substep at the node, then descend into its
    children (dict values and array elements, in document order)."""
    var stack_nodes = List[Int32]()
    var stack_paths = List[Int32]()
    var stack_cidx = List[Int64]()
    var stack_applied = List[UInt8]()
    stack_nodes.append(node)
    stack_paths.append(path)
    stack_cidx.append(0)
    stack_applied.append(0)
    while len(stack_nodes) > 0:
        var top = len(stack_nodes) - 1
        if stack_applied[Int(top)] == 0:
            stack_applied[Int(top)] = 1
            apply_step(self, sub, stack_nodes[Int(top)], stack_paths[Int(top)])
            if self.status != ST_OK:
                return
        var tn = stack_nodes[Int(top)]
        var nd = self.nodes[Int(tn)]
        var nch = Int64(0)
        if nd.tag == TAG_OBJ or nd.tag == TAG_ARR:
            nch = nd.b
        var ci = stack_cidx[Int(top)]
        if ci < nch:
            stack_cidx[Int(top)] = ci + 1
            var tp = stack_paths[Int(top)]
            if nd.tag == TAG_OBJ:
                var m = self.members[Int(nd.a + ci)]
                var osid = ostr_intern_field(self, True, m.koff, m.klen)
                var cp = new_path(self, tp, OUT_FIELD, osid)
                stack_nodes.append(m.vnode)
                stack_paths.append(cp)
            else:
                var child = self.children[Int(nd.a + ci)]
                var cp = new_path(self, tp, OUT_DESCIDX, ci)
                stack_nodes.append(child)
                stack_paths.append(cp)
            stack_cidx.append(0)
            stack_applied.append(0)
        else:
            _ = stack_nodes.pop()
            _ = stack_paths.pop()
            _ = stack_cidx.pop()
            _ = stack_applied.pop()


def apply_step(mut self: Engine, step: PStep, node: Int32, path: Int32):
    if step.kind == S_CHILD:
        apply_child(self, step, node, path)
    elif step.kind == S_WILD:
        apply_wild(self, node, path)
    elif step.kind == S_INDEX:
        apply_index(self, step.a, node, path)
    elif step.kind == S_IDXWILD:
        apply_idxwild(self, node, path)
    elif step.kind == S_SLICE:
        apply_slice(self, step, node, path)
    elif step.kind == S_DESC:
        var sub = self.substeps[Int(step.a)]
        apply_descend(self, sub, node, path)
    elif step.kind == S_FILTER:
        var filt = self.filters[Int(step.a)]
        apply_filter(self, filt, node, path)


def apply_filter(mut self: Engine, filt: Filter, node: Int32, path: Int32):
    var n = self.nodes[Int(node)]
    if n.tag == TAG_ARR:
        for i in range(n.b):
            var cand = self.children[Int(n.a + i)]
            var matched = eval_filter(self, filt, cand)
            if matched:
                var cpath = new_path(self, path, OUT_FILTERIDX, i)
                emit(self, cand, cpath)
            if self.status != ST_OK:
                return
    elif n.tag == TAG_OBJ:
        var pos = Int64(0)
        for mi in range(n.a, n.a + n.b):
            var cand = self.members[Int(mi)].vnode
            if eval_filter(self, filt, cand):
                emit(self, cand, new_path(self, path, OUT_DICTPOS, pos))
            if self.status != ST_OK:
                return
            pos += 1


def shrink_vals(mut self: Engine, n: Int32):
    while Int32(len(self.vals)) > n:
        _ = self.vals.pop()


def eval_filter(mut self: Engine, filt: Filter, at: Int32) -> Bool:
    """Evaluate all conjuncts (the oracle does NOT short-circuit `&`:
    later conjuncts still run, and their TypeErrors propagate) and AND
    their results. Within a comparison the full cross product is evaluated
    for the same reason."""
    var all_ok = True
    for ci in range(filt.start, filt.start + filt.count):
        var conj = self.conjs[Int(ci)]
        if conj.op == OP_EXISTS:
            if ci == filt.start and filt.count > 1:
                # `?(@.a & ...)`: existence as the LEFT operand of `&`
                self.status = ST_NOTIMPL
                return False
            var ok = arith_has_path(self, conj.left)
            if ok:
                self.arith_poison = False
                var n0 = Int32(len(self.vals))
                eval_arith(self, conj.left, at)
                if self.status != ST_OK:
                    return False
                var produced = Int32(len(self.vals)) - n0
                if self.arith_poison:
                    produced = 0
                shrink_vals(self, n0)
                ok = produced > 0
            all_ok = all_ok and ok
        else:
            self.arith_poison = False
            var nl = Int32(len(self.vals))
            eval_arith(self, conj.left, at)
            if self.status != ST_OK:
                return False
            if self.arith_poison:
                shrink_vals(self, nl)
            self.arith_poison = False
            var nr = Int32(len(self.vals))
            eval_arith(self, conj.right, at)
            if self.status != ST_OK:
                return False
            if self.arith_poison:
                shrink_vals(self, nr)
            var nend = Int32(len(self.vals))
            var matched = False
            for li in range(nl, nr):
                for ri in range(nr, nend):
                    var lv = self.vals[Int(li)]
                    var rv = self.vals[Int(ri)]
                    if compare_vals(self, lv, conj.op, rv):
                        matched = True
                    if self.status != ST_OK:
                        return False
            shrink_vals(self, nl)
            all_ok = all_ok and matched
    return all_ok


def eval_arith(mut self: Engine, aid: Int32, at: Int32):
    """Append the values of the arith expression to self.vals. Arithmetic
    type errors are caught per pair (the pair yields no value), mirroring
    the oracle treating them as a non-match."""
    var a = self.ariths[Int(aid)]
    if a.kind == A_LIT_INT:
        self.vals.append(V(TAG_INT, a.a, 0, 0, 0))
    elif a.kind == A_LIT_DBL:
        self.vals.append(V(TAG_DBL, a.a, 0, 0, 0))
    elif a.kind == A_LIT_STR:
        var so = self.pstr_off[Int(a.a)]
        var se = self.pstr_off[Int(a.a + 1)]
        self.vals.append(V(TAG_STR, 0, so, se - so, 2))
    elif a.kind == A_PATH:
        eval_fpath(self, a, at)
    else:
        var nl = Int32(len(self.vals))
        eval_arith(self, Int32(a.a), at)
        if self.status != ST_OK:
            return
        var nr = Int32(len(self.vals))
        eval_arith(self, Int32(a.b), at)
        if self.status != ST_OK:
            return
        var ne = Int32(len(self.vals))
        var results = List[V]()
        for li in range(nl, nr):
            for ri in range(nr, ne):
                var ok = False
                var lv = self.vals[Int(li)]
                var rv = self.vals[Int(ri)]
                var v = arith_pair(self, a.kind, lv, rv, ok)
                if self.status != ST_OK:
                    return
                if ok:
                    results.append(v)
                else:
                    self.arith_poison = True
        shrink_vals(self, nl)
        for i in range(len(results)):
            self.vals.append(results[i])


def val_int(v: V) -> Int64:
    if v.tag == TAG_TRUE:
        return 1
    if v.tag == TAG_FALSE:
        return 0
    return v.i


def val_double(v: V) -> Float64:
    if v.tag == TAG_DBL:
        return Pointer(to=v.i).unsafe_bitcast[Float64]()[]
    return Float64(val_int(v))


def arith_pair(mut self: Engine, kind: UInt8, l: V, r: V, mut ok: Bool) -> V:
    """One binary arithmetic operation with Python scalar semantics; pairs
    that would raise in Python yield ok=False (caught, no value)."""
    ok = False
    var lnum = l.tag == TAG_INT or l.tag == TAG_TRUE or l.tag == TAG_FALSE or l.tag == TAG_DBL
    var rnum = r.tag == TAG_INT or r.tag == TAG_TRUE or r.tag == TAG_FALSE or r.tag == TAG_DBL
    if kind == A_ADD and l.tag == TAG_STR and r.tag == TAG_STR:
        var lo = Int64(len(self.vbuf))
        append_val_bytes(self, l)
        append_val_bytes(self, r)
        ok = True
        return V(TAG_STR, 0, lo, Int64(len(self.vbuf)) - lo, 1)
    if lnum and rnum:
        if l.tag == TAG_DBL or r.tag == TAG_DBL:
            var ld = val_double(l)
            var rd = val_double(r)
            var res = Float64(0)
            if kind == A_ADD:
                res = ld + rd
            elif kind == A_SUB:
                res = ld - rd
            else:
                res = ld * rd
            ok = True
            return V(TAG_DBL, Pointer(to=res).unsafe_bitcast[Int64]()[], 0, 0, 0)
        var li = val_int(l)
        var ri = val_int(r)
        var resi = Int64(0)
        if kind == A_ADD:
            var s = li + ri
            if (ri > 0 and s < li) or (ri < 0 and s > li):
                self.status = ST_FALLBACK  # Python bigint; defer to wrapper
                return V(TAG_INT, 0, 0, 0, 0)
            resi = s
        elif kind == A_SUB:
            var s = li - ri
            if (ri < 0 and s < li) or (ri > 0 and s > li):
                self.status = ST_FALLBACK
                return V(TAG_INT, 0, 0, 0, 0)
            resi = s
        else:
            var s = li * ri
            if li != 0 and s // li != ri:
                self.status = ST_FALLBACK
                return V(TAG_INT, 0, 0, 0, 0)
            resi = s
        ok = True
        return V(TAG_INT, resi, 0, 0, 0)
    if kind == A_MUL and l.tag == TAG_STR and (r.tag == TAG_INT or r.tag == TAG_TRUE or r.tag == TAG_FALSE):
        return str_repeat(self, l, val_int(r), ok)
    if kind == A_MUL and (l.tag == TAG_INT or l.tag == TAG_TRUE or l.tag == TAG_FALSE) and r.tag == TAG_STR:
        return str_repeat(self, r, val_int(l), ok)
    # containers: list repetition/concat only need the tag for comparisons
    if kind == A_MUL and l.tag == TAG_ARR and (r.tag == TAG_INT or r.tag == TAG_TRUE or r.tag == TAG_FALSE):
        ok = True
        return V(TAG_ARR, 0, 0, 0, 0)
    if kind == A_MUL and (l.tag == TAG_INT or l.tag == TAG_TRUE or l.tag == TAG_FALSE) and r.tag == TAG_ARR:
        ok = True
        return V(TAG_ARR, 0, 0, 0, 0)
    if kind == A_ADD and l.tag == TAG_ARR and r.tag == TAG_ARR:
        ok = True
        return V(TAG_ARR, 0, 0, 0, 0)
    return V(TAG_NULL, 0, 0, 0, 0)


def str_repeat(mut self: Engine, s: V, count: Int64, mut ok: Bool) -> V:
    ok = False
    var n = count
    if n < 0:
        n = 0
    if s.sl * n > MAX_STR_REPEAT:
        self.status = ST_FALLBACK  # absurdly long repetition: defer
        return V(TAG_STR, 0, 0, 0, 1)
    var lo = Int64(len(self.vbuf))
    for _ in range(n):
        append_val_bytes(self, s)
    ok = True
    return V(TAG_STR, 0, lo, Int64(len(self.vbuf)) - lo, 1)


def append_val_bytes(mut self: Engine, v: V):
    if v.src == 0:
        for i in range(v.sl):
            self.vbuf.append(self.sbuf[Int(v.so + i)])
    elif v.src == 1:
        for i in range(v.sl):
            self.vbuf.append(self.vbuf[Int(v.so + i)])
    else:
        for i in range(v.sl):
            self.vbuf.append(self.pstr_buf[Int(v.so + i)])


def eval_fpath(mut self: Engine, a: Arith, at: Int32):
    """Evaluate an A_PATH: apply the filter-path steps to the node set and
    append the resulting values to self.vals."""
    var cur = List[Int32]()
    var nxt = List[Int32]()
    cur.append(at)
    for si in range(a.a, a.a + a.b):
        var fp = self.fpsteps[Int(si)]
        nxt.clear()
        for ni in range(len(cur)):
            var node = cur[Int(ni)]
            if fp.kind == F_FIELD:
                fpath_field_one(self, Int32(fp.a), node, nxt)
            elif fp.kind == F_UNION:
                for fi in range(fp.a, fp.a + fp.b):
                    fpath_field_one(self, self.pf[Int(fi)], node, nxt)
            elif fp.kind == F_WILD:
                var n = self.nodes[Int(node)]
                if n.tag == TAG_OBJ:
                    for mi in range(n.a, n.a + n.b):
                        nxt.append(self.members[Int(mi)].vnode)
            elif fp.kind == F_INDEX:
                fpath_index(self, fp.a, node, nxt)
            elif fp.kind == F_SLICE:
                fpath_slice(self, fp, node, nxt)
            elif fp.kind == F_IDXWILD:
                fpath_idxwild(self, node, nxt)
            if self.status != ST_OK:
                return
        var tmp = cur^
        cur = nxt^
        nxt = tmp^
    for ni in range(len(cur)):
        self.vals.append(v_from_node(self, cur[Int(ni)]))


def fpath_field_one(mut self: Engine, sid: Int32, node: Int32, mut out: List[Int32]):
    var n = self.nodes[Int(node)]
    if n.tag != TAG_OBJ:
        return
    var fso = self.pstr_off[Int(sid)]
    var fse = self.pstr_off[Int(sid + 1)]
    var flen = fse - fso
    for mi in range(n.a, n.a + n.b):
        var m = self.members[Int(mi)]
        if m.klen == flen:
            var same = True
            for i in range(flen):
                if self.sbuf[Int(m.koff + i)] != self.pstr_buf[Int(fso + i)]:
                    same = False
                    break
            if same:
                out.append(m.vnode)
                return


def fpath_index(mut self: Engine, idx: Int64, node: Int32, mut out: List[Int32]):
    if node_falsy(self, node):
        return
    var n = self.nodes[Int(node)]
    if n.tag == TAG_ARR:
        if idx >= 0:
            if idx < n.b:
                out.append(self.children[Int(n.a + idx)])
            return
        if idx < -n.b:
            self.status = ST_INDEXERROR
            self.aux = 0
            return
        out.append(self.children[Int(n.a + n.b + idx)])
        return
    if n.tag == TAG_STR:
        # code-point indexing (positive) / from the end (negative)
        var total = Int64(0)
        var pos = n.a
        while pos < n.a + n.b:
            var ch = self.sbuf[Int(pos)]
            var adv = Int64(1)
            if ch >= 0xF0:
                adv = 4
            elif ch >= 0xE0:
                adv = 3
            elif ch >= 0xC0:
                adv = 2
            pos += adv
            total += 1
        var target = idx
        if idx < 0:
            target = total + idx
        if target < 0:
            self.status = ST_INDEXERROR
            self.aux = 1
            return
        if target >= total:
            return  # positive out of range: no match
        var cp = Int64(0)
        var pos2 = n.a
        var start = n.a
        while cp <= target and pos2 < n.a + n.b:
            start = pos2
            var ch = self.sbuf[Int(pos2)]
            var adv = Int64(1)
            if ch >= 0xF0:
                adv = 4
            elif ch >= 0xE0:
                adv = 3
            elif ch >= 0xC0:
                adv = 2
            pos2 += adv
            cp += 1
        var nid = Int32(len(self.nodes))
        self.nodes.append(JNode(TAG_STR, start, pos2 - start))
        out.append(nid)
        return
    if n.tag == TAG_OBJ:
        if idx >= 0 and idx >= n.b:
            return
        self.status = ST_KEYERROR
        self.aux = idx
        return
    self.status = ST_TE_LEN
    self.aux = Int64(n.tag)


def fpath_idxwild(mut self: Engine, node: Int32, mut out: List[Int32]):
    var n = self.nodes[Int(node)]
    if n.tag == TAG_ARR:
        for i in range(n.b):
            out.append(self.children[Int(n.a + i)])
        return
    if node_falsy(self, node):
        return
    if n.tag == TAG_DBL:
        self.status = ST_TE_LEN
        self.aux = Int64(TAG_DBL)
        return
    out.append(node)


def fpath_slice(mut self: Engine, fp: FpStep, node: Int32, mut out: List[Int32]):
    if node_falsy(self, node):
        return
    var stepv = fp.c if fp.c != ABSENT else Int64(1)
    if stepv == 0:
        self.status = ST_VE_SLICE
        return
    var n = self.nodes[Int(node)]
    if n.tag != TAG_ARR:
        return
    var length = n.b
    var s = fp.a
    var e = fp.b
    if stepv > 0:
        if s == ABSENT:
            s = 0
        if e == ABSENT:
            e = length
        if s < 0:
            s += length
            if s < 0:
                s = 0
        if e < 0:
            e += length
            if e < 0:
                e = 0
        if s > length:
            s = length
        if e > length:
            e = length
        var i = s
        while i < e:
            out.append(self.children[Int(n.a + i)])
            i += stepv
    else:
        if s == ABSENT:
            s = length - 1
        if e == ABSENT:
            e = -length - 1
        if s < 0:
            s += length
            if s < 0:
                s = -1
        if e < 0:
            e += length
            if e < 0:
                e = -1
        if s >= length:
            s = length - 1
        if e >= length:
            e = length - 1
        var i = s
        while i > e:
            out.append(self.children[Int(n.a + i)])
            i += stepv


def v_from_node(self: Engine, node: Int32) -> V:
    var n = self.nodes[Int(node)]
    if n.tag == TAG_INT or n.tag == TAG_DBL:
        return V(n.tag, n.a, 0, 0, 0)
    if n.tag == TAG_STR:
        return V(TAG_STR, 0, n.a, n.b, 0)
    if n.tag == TAG_TRUE or n.tag == TAG_FALSE:
        return V(n.tag, n.a, 0, 0, 0)
    return V(n.tag, 0, 0, 0, 0)  # null / containers


def byte_of(self: Engine, v: V, i: Int64) -> UInt8:
    if v.src == 0:
        return self.sbuf[Int(v.so + i)]
    if v.src == 1:
        return self.vbuf[Int(v.so + i)]
    return self.pstr_buf[Int(v.so + i)]


def val_bytes_eq(self: Engine, a: V, b: V) -> Bool:
    if a.sl != b.sl:
        return False
    for i in range(a.sl):
        if byte_of(self, a, i) != byte_of(self, b, i):
            return False
    return True


def val_str_lt(self: Engine, a: V, b: V) -> Bool:
    """Lexicographic byte order == Unicode code-point order (UTF-8)."""
    var n = a.sl if a.sl < b.sl else b.sl
    for i in range(n):
        var x = byte_of(self, a, i)
        var y = byte_of(self, b, i)
        if x != y:
            return x < y
    return a.sl < b.sl


def is_py_space(c: UInt8) -> Bool:
    return c == CH_SPACE or c == CH_TAB or c == CH_LF or c == CH_CR or c == CH_VT or c == CH_FF


def coerce_int(mut self: Engine, v: V, mut coerced: Int64, mut failed: Bool):
    """Python int(v) for the right-literal-int coercion: failed=True means
    ValueError (caught, no match); a status is set for TypeError (raises)."""
    failed = False
    coerced = 0
    if v.tag == TAG_INT:
        coerced = v.i
        return
    if v.tag == TAG_TRUE:
        coerced = 1
        return
    if v.tag == TAG_FALSE:
        coerced = 0
        return
    if v.tag == TAG_DBL:
        var d = Pointer(to=v.i).unsafe_bitcast[Float64]()[]
        var t = trunc(d)
        if t >= 9223372036854775808.0:
            coerced = Int64(9223372036854775807)
            self.big_coerce = True
        elif t <= -9223372036854775808.0:
            coerced = Int64(-9223372036854775807) - 1
            self.big_coerce = True
        else:
            coerced = Int64(t)
        return
    if v.tag == TAG_STR:
        # Python int(str): ASCII whitespace, optional sign, digits with
        # single underscores between digits.
        var i = Int64(0)
        var n = v.sl
        while i < n and is_py_space(byte_of(self, v, i)):
            i += 1
        var j = n
        while j > i and is_py_space(byte_of(self, v, j - 1)):
            j -= 1
        var neg = False
        if i < j and (byte_of(self, v, i) == CH_PLUS or byte_of(self, v, i) == CH_MINUS):
            neg = byte_of(self, v, i) == CH_MINUS
            i += 1
        if i >= j:
            failed = True
            return
        var val = UInt64(0)
        var last_digit = False
        var any = False
        var ovf = False
        while i < j:
            var c = byte_of(self, v, i)
            if c >= CH_0 and c <= CH_9:
                var d = UInt64(c - CH_0)
                if val > (UInt64(18446744073709551615) - d) // 10:
                    ovf = True
                else:
                    val = val * 10 + d
                last_digit = True
                any = True
            elif c == CH_UNDERSCORE:
                if not last_digit:
                    failed = True
                    return
                last_digit = False
            else:
                failed = True
                return
            i += 1
        if not any or not last_digit:
            failed = True
            return
        var limit = UInt64(9223372036854775807) if not neg else UInt64(9223372036854775808)
        if ovf or val > limit:
            # Python bigint: == / != / ordering against an i64 literal can
            # still be decided from the sign alone (see compare_vals).
            self.big_coerce = True
            coerced = Int64(-9223372036854775807) - 1 if neg else Int64(9223372036854775807)
            return
        coerced = -Int64(val) if neg else Int64(val)
        return
    # None / list / dict: int() raises TypeError
    self.status = ST_TE_INT
    self.aux = Int64(v.tag)


def compare_vals(mut self: Engine, l: V, op: UInt8, r: V) -> Bool:
    """One filter comparison under the oracle's coercion contract:
    - int literal right (incl. true/false): left coerced via int(); string
      parse failures are caught (no match), None/list/dict raise TypeError.
    - float literal right: direct comparison; mixed-type ordering raises
      TypeError.
    - string/bare-word right (incl. null): direct comparison; ordering on a
      non-string left raises TypeError."""
    self.big_coerce = False
    if r.tag == TAG_INT:
        var li = Int64(0)
        var failed = False
        coerce_int(self, l, li, failed)
        if self.status != ST_OK or failed:
            return False
        if self.big_coerce:
            # |left| >= 2^63 after coercion: no i64 literal equals it, so
            # == is False, != is True; ordering is decided by the sign.
            var lbig_pos = li > 0
            if op == OP_EQ:
                return False
            if op == OP_NE:
                return True
            if op == OP_LT or op == OP_LE:
                return not lbig_pos
            return lbig_pos
        var ri = r.i
        if op == OP_EQ:
            return li == ri
        if op == OP_NE:
            return li != ri
        if op == OP_LT:
            return li < ri
        if op == OP_LE:
            return li <= ri
        if op == OP_GT:
            return li > ri
        return li >= ri
    if r.tag == TAG_DBL:
        var lnum = l.tag == TAG_INT or l.tag == TAG_TRUE or l.tag == TAG_FALSE or l.tag == TAG_DBL
        if not lnum:
            if op == OP_EQ:
                return False
            if op == OP_NE:
                return True
            self.status = ST_TE_ORD
            self.aux = Int64(op) * 64 + Int64(l.tag) * 8 + Int64(r.tag)
            return False
        var ld = val_double(l)
        var rd = Pointer(to=r.i).unsafe_bitcast[Float64]()[]
        if op == OP_EQ:
            return ld == rd
        if op == OP_NE:
            return ld != rd
        if op == OP_LT:
            return ld < rd
        if op == OP_LE:
            return ld <= rd
        if op == OP_GT:
            return ld > rd
        return ld >= rd
    if r.tag == TAG_STR:
        if l.tag != TAG_STR:
            if op == OP_EQ:
                return False
            if op == OP_NE:
                return True
            self.status = ST_TE_ORD
            self.aux = Int64(op) * 64 + Int64(l.tag) * 8 + Int64(r.tag)
            return False
        var eq = val_bytes_eq(self, l, r)
        if op == OP_EQ:
            return eq
        if op == OP_NE:
            return not eq
        var lt = val_str_lt(self, l, r)
        if op == OP_LT:
            return lt
        if op == OP_LE:
            return lt or eq
        if op == OP_GT:
            return not lt and not eq
        return not lt or eq
    # container/null right literal: unreachable from parsed filters
    if op == OP_EQ:
        return l.tag == r.tag
    if op == OP_NE:
        return l.tag != r.tag
    self.status = ST_TE_ORD
    self.aux = Int64(op) * 64 + Int64(l.tag) * 8 + Int64(r.tag)
    return False


# ---------------------------------------------------------------------------
# find driver + ABI
# ---------------------------------------------------------------------------


def run_find(mut self: Engine) -> Bool:
    """Parse the expression, parse the document, evaluate."""
    self.pos = 0
    if not parse_expr(self):
        return False
    _ws(self)
    if self.pos != self.elen:
        self.status = ST_PARSE
        self.aux = self.pos
        return False
    self.pos = 0
    _jws(self)
    var root = j_parse_value(self, 0)
    if root < 0:
        # invalid JSON: the wrapper's json.dumps cannot have produced this
        self.status = ST_FALLBACK
        return False
    _jws(self)
    if self.pos != self.jlen:
        self.status = ST_FALLBACK
        return False
    if self.flags != 0:
        self.status = ST_FALLBACK
        return False
    self.pprev.append(-1)
    self.pkind.append(OUT_FIELD)
    self.pval.append(0)
    self.cur_nodes.append(root)
    self.cur_paths.append(0)
    for si in range(len(self.steps)):
        var step = self.steps[Int(si)]
        self.nxt_nodes.clear()
        self.nxt_paths.clear()
        for i in range(len(self.cur_nodes)):
            apply_step(self, step, self.cur_nodes[Int(i)], self.cur_paths[Int(i)])
            if self.status != ST_OK:
                return False
        var tn = self.cur_nodes^
        self.cur_nodes = self.nxt_nodes^
        self.nxt_nodes = tn^
        var tp = self.cur_paths^
        self.cur_paths = self.nxt_paths^
        self.nxt_paths = tp^
    return True


@export
def jsonpathmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def jsonpathmojo_find(
    expr: U8Ptr,
    expr_len: Int64,
    json: U8Ptr,
    json_len: Int64,
    status_out: I64Ptr,
) abi("C") -> Handle:
    """Run one query. status_out[0] = status code, status_out[1] = aux.
    Returns an owned result handle (even on error; it then holds zero
    matches) which the caller must destroy."""
    var eng = unsafe_alloc[Engine](1)
    eng[] = Engine(expr, expr_len, json, json_len)
    _ = run_find(eng[])
    status_out[unsafe_offset=0] = Int64(eng[].status)
    status_out[unsafe_offset=1] = eng[].aux
    return eng.unsafe_bitcast[UInt8]()


@export
def jsonpathmojo_count(handle: Handle, which: Int32) abi("C") -> Int64:
    """which: 0 = n_matches, 1 = total steps, 2 = n_strings, 3 = str bytes."""
    if not handle:
        return 0
    var eng = handle.value().unsafe_bitcast[Engine]()
    if which == 0:
        return Int64(len(eng[].cur_nodes))
    if which == 1:
        var total = Int64(0)
        for i in range(len(eng[].cur_paths)):
            var pid = eng[].cur_paths[Int(i)]
            while pid > 0:
                total += 1
                pid = eng[].pprev[Int(pid)]
        return total
    if which == 2:
        if len(eng[].ostr_off) == 0:
            return 0
        return Int64(len(eng[].ostr_off) - 1)
    return Int64(len(eng[].ostr_buf))


@export
def jsonpathmojo_copy(
    handle: Handle,
    kinds: U8Ptr,
    vals: I64Ptr,
    path_off: I64Ptr,
    str_off: I64Ptr,
    str_buf: U8Ptr,
) abi("C"):
    """Fill caller buffers sized by jsonpathmojo_count."""
    if not handle:
        return
    var eng = handle.value().unsafe_bitcast[Engine]()
    var n = Int64(len(eng[].cur_nodes))
    var step_pos = Int64(0)
    for i in range(n):
        path_off[unsafe_offset=i] = step_pos
        var pid = eng[].cur_paths[Int(i)]
        var depth = Int64(0)
        var p2 = pid
        while p2 > 0:
            depth += 1
            p2 = eng[].pprev[Int(p2)]
        var w = step_pos + depth
        p2 = pid
        while p2 > 0:
            w -= 1
            kinds[unsafe_offset=w] = eng[].pkind[Int(p2)]
            vals[unsafe_offset=w] = eng[].pval[Int(p2)]
            p2 = eng[].pprev[Int(p2)]
        step_pos += depth
    path_off[unsafe_offset=n] = step_pos
    var ns = Int64(len(eng[].ostr_off))
    for i in range(ns):
        str_off[unsafe_offset=i] = eng[].ostr_off[Int(i)]
    var nb = Int64(len(eng[].ostr_buf))
    for i in range(nb):
        str_buf[Int(i)] = eng[].ostr_buf[Int(i)]


@export
def jsonpathmojo_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var eng = handle.value().unsafe_bitcast[Engine]()
    eng[].ostr_hash = List[Int64]()  # release the big table first
    eng.unsafe_free()
