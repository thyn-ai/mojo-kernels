"""Clean-room JSON Schema (draft 2020-12 subset) validator kernel.

Written fresh from the JSON Schema 2020-12 specification. No third-party
Mojo code is used or adapted.

Exported C ABI:

    int32_t jsonschemamojo_abi_version(void)
    void*   jsonschemamojo_schema_compile(const uint8_t* text, int64_t len)
    int32_t jsonschemamojo_schema_flags(void* handle)
    void    jsonschemamojo_schema_destroy(void* handle)
    int32_t jsonschemamojo_validate(void* handle,
                                    const uint8_t* inst, int64_t inst_len,
                                    uint8_t* out, int64_t out_cap,
                                    int32_t* meta)

Both schema and instance arrive as UTF-8 JSON text (the wrapper serializes
with json.dumps, so the text is always ASCII with all non-ASCII escaped).
The kernel parses each into a flat node arena, compiles the schema into a
tree of keyword nodes, then walks instance against schema, appending error
records to `out` (see OutWriter for the record wire format). Numbers are
parsed with the C library's strtod, which is correctly rounded (round to
nearest, ties to even), exactly matching CPython's float().

Scope (enforced by the Python wrapper before the kernel ever sees a schema):
type, properties, required, items, additionalProperties, enum, const,
minimum, maximum, exclusiveMinimum, exclusiveMaximum, minLength, maxLength,
pattern, minItems, maxItems, uniqueItems, minProperties, maxProperties,
multipleOf, plus boolean subschemas. `pattern` checks and exact
multipleOf edge cases are emitted as DEFER records and resolved by the
wrapper (it owns the regex engine and Python's exact arithmetic).

Semantics mirrored from the reference implementation (PyPI jsonschema):
  * "integer" type accepts integral floats (1.0 is an integer);
  * numbers compare across int/float form (1 == 1.0), booleans never equal
    numbers, strings only equal strings;
  * multipleOf with an integer divisor uses Python's % (floored modulo)
    zero test, with a float divisor uses the int(quotient) != quotient
    test; non-trivial edge cases defer to the wrapper's exact path;
  * error identity is (keyword, instance path, schema path); ordering is a
    fixed canonical order (the reference orders by schema-dict insertion
    order — documented divergence, order is not part of the contract).

meta[0] = status (0 ok, 1 bad handle, 2 instance parse error, 3 depth limit)
meta[1] = number of error records written
meta[2] = number of defer records written
meta[3] = 1 if `out` overflowed (records are complete only up to that
          point; the wrapper grows the buffer and re-runs)
meta[4] = 1 if the instance contains a >15-digit integer literal (wrapper
          must re-validate with the exact fallback)
meta[5] = low 32 bits of bytes written to `out`
"""

from std.ffi import external_call
from std.math import trunc
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# JSON node tags.
comptime TAG_NULL: UInt8 = 0
comptime TAG_TRUE: UInt8 = 1
comptime TAG_FALSE: UInt8 = 2
comptime TAG_NUMBER: UInt8 = 3
comptime TAG_STRING: UInt8 = 4
comptime TAG_ARRAY: UInt8 = 5
comptime TAG_OBJECT: UInt8 = 6

# JNode nflags bits.
comptime NF_INTEGRAL: UInt8 = 1  # value satisfies the "integer" type check
comptime NF_TOKEN_INT: UInt8 = 2  # source literal had no '.'/exponent

# Type-bit indices for CNode.type_mask (bit i set = type name i allowed).
comptime T_NULL: UInt8 = 1 << 0
comptime T_BOOL: UInt8 = 1 << 1
comptime T_NUMBER: UInt8 = 1 << 2
comptime T_INTEGER: UInt8 = 1 << 3
comptime T_STRING: UInt8 = 1 << 4
comptime T_ARRAY: UInt8 = 1 << 5
comptime T_OBJECT: UInt8 = 1 << 6

# Keyword-present bits in CNode.kw (also the error-record keyword codes).
comptime KW_TYPE: UInt32 = 1 << 0
comptime KW_PROPS: UInt32 = 1 << 1
comptime KW_REQUIRED: UInt32 = 1 << 2
comptime KW_ITEMS: UInt32 = 1 << 3
comptime KW_AP: UInt32 = 1 << 4
comptime KW_ENUM: UInt32 = 1 << 5
comptime KW_CONST: UInt32 = 1 << 6
comptime KW_MIN: UInt32 = 1 << 7
comptime KW_MAX: UInt32 = 1 << 8
comptime KW_EXCLMIN: UInt32 = 1 << 9
comptime KW_EXCLMAX: UInt32 = 1 << 10
comptime KW_MINLEN: UInt32 = 1 << 11
comptime KW_MAXLEN: UInt32 = 1 << 12
comptime KW_PATTERN: UInt32 = 1 << 13
comptime KW_MINITEMS: UInt32 = 1 << 14
comptime KW_MAXITEMS: UInt32 = 1 << 15
comptime KW_UNIQUE: UInt32 = 1 << 16
comptime KW_MINPROPS: UInt32 = 1 << 17
comptime KW_MAXPROPS: UInt32 = 1 << 18
comptime KW_MULT: UInt32 = 1 << 19
comptime KW_FALSE_SCHEMA: UInt32 = 1 << 20  # boolean "false" subschema

# Record kinds in the out stream.
comptime REC_ERROR: UInt8 = 0
comptime REC_DEFER: UInt8 = 1

# Deferral kinds.
comptime DEFER_PATTERN: UInt8 = 1
comptime DEFER_MULT_EXACT: UInt8 = 2

# Document / handle flags.
comptime FLAG_BIG_NUMBER: UInt32 = 1 << 0

# Recursion budget for parsing and compiling (the wrapper's json.dumps
# output cannot nest deeper than CPython's recursion limit anyway).
comptime MAX_DEPTH: Int32 = 1024

# |instance| at or above 2^52 defers integer-divisor multipleOf to the
# wrapper's exact path.
comptime MULT_EXACT_LIMIT: Float64 = 4503599627370496.0


struct JNode(ImplicitlyCopyable, Copyable, Movable):
    var tag: UInt8
    var nflags: UInt8
    var num: Float64
    var a: Int64  # string: byte offset; array/object: first child/member slot
    var b: Int64  # string: byte length; array/object: child/member count

    def __init__(out self, tag: UInt8, nflags: UInt8, num: Float64, a: Int64, b: Int64):
        self.tag = tag
        self.nflags = nflags
        self.num = num
        self.a = a
        self.b = b


struct JMember(ImplicitlyCopyable, Copyable, Movable):
    var key_off: Int64
    var key_len: Int64
    var value: Int32  # node index

    def __init__(out self, key_off: Int64, key_len: Int64, value: Int32):
        self.key_off = key_off
        self.key_len = key_len
        self.value = value


struct StrRef(ImplicitlyCopyable, Copyable, Movable):
    var off: Int64
    var len: Int64

    def __init__(out self, off: Int64, len: Int64):
        self.off = off
        self.len = len


struct Doc(Movable):
    """One parsed JSON document: flat pre-order node arena + decoded strings."""

    var nodes: List[JNode]
    var strs: List[UInt8]
    var slots: List[Int32]  # array element node indices
    var members: List[JMember]  # object members
    var root: Int32
    var flags: UInt32

    def __init__(out self):
        self.nodes = List[JNode]()
        self.strs = List[UInt8]()
        self.slots = List[Int32]()
        self.members = List[JMember]()
        self.root = -1
        self.flags = 0


struct Parser:
    var text: U8Ptr
    var length: Int64
    var pos: Int64
    var failed: Bool
    var depth_exceeded: Bool
    # Scratch stacks: per-level children are staged here and committed to the
    # document arenas as one contiguous run per array/object.
    var node_scratch: List[Int32]
    var member_scratch: List[JMember]

    def __init__(out self, text: U8Ptr, length: Int64):
        self.text = text
        self.length = length
        self.pos = 0
        self.failed = False
        self.depth_exceeded = False
        self.node_scratch = List[Int32]()
        self.member_scratch = List[JMember]()

    def skip_ws(mut self):
        while self.pos < self.length:
            var c = self.text[unsafe_offset=self.pos]
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                return

    def peek(mut self) -> Int32:
        if self.pos >= self.length:
            return -1
        return Int32(self.text[unsafe_offset=self.pos])

    def expect(mut self, c: UInt8) -> Bool:
        if self.pos < self.length and self.text[unsafe_offset=self.pos] == c:
            self.pos += 1
            return True
        self.failed = True
        return False

    def parse_value(mut self, mut doc: Doc, depth: Int32) -> Int32:
        if depth > MAX_DEPTH:
            self.depth_exceeded = True
            self.failed = True
            return -1
        self.skip_ws()
        var c = self.peek()
        if c < 0:
            self.failed = True
            return -1
        if c == 0x7B:  # '{'
            return self.parse_object(doc, depth)
        if c == 0x5B:  # '['
            return self.parse_array(doc, depth)
        if c == 0x22:  # '"'
            var off = Int64(doc.strs.__len__())
            var n = self.parse_string(doc)
            if self.failed:
                return -1
            var idx = Int32(doc.nodes.__len__())
            doc.nodes.append(JNode(TAG_STRING, 0, 0.0, off, n))
            return idx
        if c == 0x74:  # 't' true
            if self.consume_lit("true"):
                var idx = Int32(doc.nodes.__len__())
                doc.nodes.append(JNode(TAG_TRUE, 0, 0.0, 0, 0))
                return idx
            return -1
        if c == 0x66:  # 'f' false
            if self.consume_lit("false"):
                var idx = Int32(doc.nodes.__len__())
                doc.nodes.append(JNode(TAG_FALSE, 0, 0.0, 0, 0))
                return idx
            return -1
        if c == 0x6E:  # 'n' null
            if self.consume_lit("null"):
                var idx = Int32(doc.nodes.__len__())
                doc.nodes.append(JNode(TAG_NULL, 0, 0.0, 0, 0))
                return idx
            return -1
        return self.parse_number(doc)

    def consume_lit(mut self, lit: String) -> Bool:
        var lb = lit.as_bytes()
        for i in range(lb.__len__()):
            if self.pos >= self.length or self.text[unsafe_offset=self.pos] != lb[i]:
                self.failed = True
                return False
            self.pos += 1
        return True

    def parse_object(mut self, mut doc: Doc, depth: Int32) -> Int32:
        # Placeholder first: the parent node index is lower than every
        # descendant's (pre-order arena).
        var idx = Int32(doc.nodes.__len__())
        doc.nodes.append(JNode(TAG_OBJECT, 0, 0.0, 0, 0))
        _ = self.expect(0x7B)
        var mark = self.member_scratch.__len__()
        self.skip_ws()
        if self.peek() == 0x7D:  # '}' immediately
            self.pos += 1
        else:
            while True:
                self.skip_ws()
                if self.peek() != 0x22:
                    self.failed = True
                    return -1
                var koff = Int64(doc.strs.__len__())
                var klen = self.parse_string(doc)
                if self.failed:
                    return -1
                self.skip_ws()
                if not self.expect(0x3A):  # ':'
                    return -1
                var v = self.parse_value(doc, depth + 1)
                if self.failed:
                    return -1
                self.member_scratch.append(JMember(koff, klen, v))
                self.skip_ws()
                var nc = self.peek()
                if nc == 0x2C:  # ','
                    self.pos += 1
                    continue
                if nc == 0x7D:  # '}'
                    self.pos += 1
                    break
                self.failed = True
                return -1
        var count = self.member_scratch.__len__() - mark
        var start = Int64(doc.members.__len__())
        for i in range(count):
            doc.members.append(self.member_scratch[mark + i])
        self.member_scratch.shrink(mark)
        doc.nodes[Int(idx)] = JNode(TAG_OBJECT, 0, 0.0, start, Int64(count))
        return idx

    def parse_array(mut self, mut doc: Doc, depth: Int32) -> Int32:
        var idx = Int32(doc.nodes.__len__())
        doc.nodes.append(JNode(TAG_ARRAY, 0, 0.0, 0, 0))
        _ = self.expect(0x5B)
        var mark = self.node_scratch.__len__()
        self.skip_ws()
        if self.peek() == 0x5D:  # ']' immediately
            self.pos += 1
        else:
            while True:
                var v = self.parse_value(doc, depth + 1)
                if self.failed:
                    return -1
                self.node_scratch.append(v)
                self.skip_ws()
                var nc = self.peek()
                if nc == 0x2C:
                    self.pos += 1
                    continue
                if nc == 0x5D:
                    self.pos += 1
                    break
                self.failed = True
                return -1
        var count = self.node_scratch.__len__() - mark
        var start = Int64(doc.slots.__len__())
        for i in range(count):
            doc.slots.append(self.node_scratch[mark + i])
        self.node_scratch.shrink(mark)
        doc.nodes[Int(idx)] = JNode(TAG_ARRAY, 0, 0.0, start, Int64(count))
        return idx

    def hex4(mut self) -> Int32:
        var v = Int32(0)
        for _ in range(4):
            if self.pos >= self.length:
                self.failed = True
                return 0
            var c = Int32(self.text[unsafe_offset=self.pos])
            self.pos += 1
            var d = Int32(-1)
            if c >= 0x30 and c <= 0x39:
                d = c - 0x30
            elif c >= 0x61 and c <= 0x66:
                d = c - 0x61 + 10
            elif c >= 0x41 and c <= 0x46:
                d = c - 0x41 + 10
            if d < 0:
                self.failed = True
                return 0
            v = v * 16 + d
        return v

    def emit_utf8(mut self, mut doc: Doc, cp: Int32):
        if cp < 0x80:
            doc.strs.append(UInt8(cp))
        elif cp < 0x800:
            doc.strs.append(UInt8(0xC0 | (cp >> 6)))
            doc.strs.append(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            doc.strs.append(UInt8(0xE0 | (cp >> 12)))
            doc.strs.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            doc.strs.append(UInt8(0x80 | (cp & 0x3F)))
        else:
            doc.strs.append(UInt8(0xF0 | (cp >> 18)))
            doc.strs.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            doc.strs.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            doc.strs.append(UInt8(0x80 | (cp & 0x3F)))

    def parse_string(mut self, mut doc: Doc) -> Int64:
        """Decode one string literal into doc.strs (UTF-8). Returns length."""
        _ = self.expect(0x22)
        var n = Int64(0)
        while True:
            if self.pos >= self.length:
                self.failed = True
                return 0
            var c = self.text[unsafe_offset=self.pos]
            self.pos += 1
            if c == 0x22:  # closing quote
                return n
            if c == 0x5C:  # backslash
                if self.pos >= self.length:
                    self.failed = True
                    return 0
                var e = self.text[unsafe_offset=self.pos]
                self.pos += 1
                if e == 0x75:  # 'u'
                    var cp = self.hex4()
                    if self.failed:
                        return 0
                    if (
                        cp >= 0xD800
                        and cp <= 0xDBFF
                        and self.pos + 1 < self.length
                        and self.text[unsafe_offset=self.pos] == 0x5C
                        and self.text[unsafe_offset=self.pos + 1] == 0x75
                    ):
                        # Possible surrogate pair; look ahead without committing.
                        var save = self.pos
                        self.pos += 2
                        var lo = self.hex4()
                        if self.failed:
                            return 0
                        if lo >= 0xDC00 and lo <= 0xDFFF:
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                        else:
                            self.pos = save  # lone surrogate: emit as-is
                    self.emit_utf8(doc, cp)
                    if cp < 0x80:
                        n += 1
                    elif cp < 0x800:
                        n += 2
                    elif cp < 0x10000:
                        n += 3
                    else:
                        n += 4
                    continue
                var simple = Int32(-1)
                if e == 0x22 or e == 0x5C or e == 0x2F:
                    simple = Int32(e)
                elif e == 0x62:
                    simple = 0x08
                elif e == 0x66:
                    simple = 0x0C
                elif e == 0x6E:
                    simple = 0x0A
                elif e == 0x72:
                    simple = 0x0D
                elif e == 0x74:
                    simple = 0x09
                if simple < 0:
                    self.failed = True
                    return 0
                doc.strs.append(UInt8(simple))
                n += 1
                continue
            if c < 0x20:
                self.failed = True
                return 0
            doc.strs.append(c)
            n += 1

    def parse_number(mut self, mut doc: Doc) -> Int32:
        var start = self.pos
        var neg = False
        if self.peek() == 0x2D:  # '-'
            self.pos += 1
            neg = True
        var digits = Int64(0)
        var int_val = Int64(0)
        var c = self.peek()
        if c == 0x30:  # leading zero is the whole integer part
            self.pos += 1
            digits = 1
        else:
            if c < 0x31 or c > 0x39:
                self.failed = True
                return -1
            while True:
                c = self.peek()
                if c < 0x30 or c > 0x39:
                    break
                if digits < 19:
                    int_val = int_val * 10 + Int64(c - 0x30)
                digits += 1
                self.pos += 1
        var is_float = False
        if self.peek() == 0x2E:  # '.'
            is_float = True
            self.pos += 1
            c = self.peek()
            if c < 0x30 or c > 0x39:
                self.failed = True
                return -1
            while True:
                c = self.peek()
                if c < 0x30 or c > 0x39:
                    break
                self.pos += 1
        c = self.peek()
        if c == 0x65 or c == 0x45:  # 'e' / 'E'
            is_float = True
            self.pos += 1
            c = self.peek()
            if c == 0x2B or c == 0x2D:
                self.pos += 1
            c = self.peek()
            if c < 0x30 or c > 0x39:
                self.failed = True
                return -1
            while True:
                c = self.peek()
                if c < 0x30 or c > 0x39:
                    break
                self.pos += 1
        var tok_len = self.pos - start
        var value = Float64(0.0)
        var nflags = UInt8(0)
        if not is_float:
            nflags |= NF_TOKEN_INT
            if digits <= 15:
                # Every integer with at most 15 digits is exactly a float64.
                value = Float64(-int_val if neg else int_val)
                nflags |= NF_INTEGRAL
            else:
                doc.flags |= FLAG_BIG_NUMBER
                value = self.strtod_token(start, tok_len)
                if value == trunc(value):
                    nflags |= NF_INTEGRAL
        else:
            value = self.strtod_token(start, tok_len)
            if value == trunc(value):
                nflags |= NF_INTEGRAL
        var idx = Int32(doc.nodes.__len__())
        doc.nodes.append(JNode(TAG_NUMBER, nflags, value, 0, 0))
        return idx

    def strtod_token(mut self, start: Int64, tok_len: Int64) -> Float64:
        # Copy the token into a NUL-terminated scratch and let the C library
        # do the correctly-rounded decimal-to-binary conversion.
        var buf = unsafe_alloc[UInt8](64)
        var n = tok_len
        if n > 63:
            n = 63
        for i in range(n):
            buf[i] = self.text[unsafe_offset=start + i]
        buf[Int(n)] = 0
        var end = unsafe_alloc[U8Ptr](1)
        var v = external_call["strtod", Float64](buf, end)
        buf.unsafe_free()
        end.unsafe_free()
        return v


# --------------------------------------------------------------------------
# Compiled schema
# --------------------------------------------------------------------------


struct CProp(ImplicitlyCopyable, Copyable, Movable):
    var name: StrRef
    var cnode: Int32

    def __init__(out self, name: StrRef, cnode: Int32):
        self.name = name
        self.cnode = cnode


struct CNode(ImplicitlyCopyable, Copyable, Movable):
    var mode: UInt8  # 0 = keyword object, 1 = true (accept), 2 = false (deny)
    var kw: UInt32
    var type_mask: UInt8
    var ap_mode: UInt8  # 0 absent, 1 subschema, 2 deny (false)
    var mof_token_int: UInt8
    var props_off: Int32
    var props_len: Int32
    var req_off: Int32
    var req_len: Int32
    var items: Int32
    var ap: Int32
    var enum_node: Int32
    var const_node: Int32
    var vmin: Float64
    var vmax: Float64
    var vexclmin: Float64
    var vexclmax: Float64
    var vmult: Float64
    var iminlen: Int64
    var imaxlen: Int64
    var iminitems: Int64
    var imaxitems: Int64
    var iminprops: Int64
    var imaxprops: Int64
    var pattern: StrRef
    var spath_off: Int32
    var spath_len: Int32

    def __init__(out self, mode: UInt8):
        self.mode = mode
        self.kw = 0
        self.type_mask = 0
        self.ap_mode = 0
        self.mof_token_int = 0
        self.props_off = 0
        self.props_len = 0
        self.req_off = 0
        self.req_len = 0
        self.items = -1
        self.ap = -1
        self.enum_node = -1
        self.const_node = -1
        self.vmin = 0.0
        self.vmax = 0.0
        self.vexclmin = 0.0
        self.vexclmax = 0.0
        self.vmult = 0.0
        self.iminlen = 0
        self.imaxlen = 0
        self.iminitems = 0
        self.imaxitems = 0
        self.iminprops = 0
        self.imaxprops = 0
        self.pattern = StrRef(0, 0)
        self.spath_off = 0
        self.spath_len = 0


struct Schema(Movable):
    var doc: Doc  # parsed schema JSON (enum/const values, names, pattern text)
    var cnodes: List[CNode]
    var cprops: List[CProp]
    var reqs: List[StrRef]
    var spath: List[StrRef]
    var root: Int32
    var flags: UInt32

    def __init__(out self, var doc: Doc):
        self.doc = doc^
        self.cnodes = List[CNode]()
        self.cprops = List[CProp]()
        self.reqs = List[StrRef]()
        self.spath = List[StrRef]()
        self.root = -1
        self.flags = 0


def bytes_eq(doc: Doc, r: StrRef, lit: String) -> Bool:
    var lb = lit.as_bytes()
    if r.len != Int64(lb.__len__()):
        return False
    for i in range(r.len):
        if doc.strs[Int(r.off + i)] != lb[Int(i)]:
            return False
    return True


def keys_eq(a: Doc, ra: StrRef, b: Doc, rb: StrRef) -> Bool:
    if ra.len != rb.len:
        return False
    for i in range(ra.len):
        if a.strs[Int(ra.off + i)] != b.strs[Int(rb.off + i)]:
            return False
    return True


def type_name_bit(doc: Doc, r: StrRef) -> UInt8:
    if bytes_eq(doc, r, "null"):
        return T_NULL
    if bytes_eq(doc, r, "boolean"):
        return T_BOOL
    if bytes_eq(doc, r, "number"):
        return T_NUMBER
    if bytes_eq(doc, r, "integer"):
        return T_INTEGER
    if bytes_eq(doc, r, "string"):
        return T_STRING
    if bytes_eq(doc, r, "array"):
        return T_ARRAY
    if bytes_eq(doc, r, "object"):
        return T_OBJECT
    return 0


struct Compiler:
    """Builds the CNode tree inside a Schema (scratch state only)."""

    var spath_scratch: List[StrRef]
    var prop_scratch: List[CProp]
    var req_scratch: List[StrRef]
    var depth_exceeded: Bool

    def __init__(out self):
        self.spath_scratch = List[StrRef]()
        self.prop_scratch = List[CProp]()
        self.req_scratch = List[StrRef]()
        self.depth_exceeded = False

    def lit_ref(mut self, mut s: Schema, lit: String) -> StrRef:
        """Append a compile-time keyword name into the string arena."""
        var lb = lit.as_bytes()
        var off = Int64(s.doc.strs.__len__())
        for i in range(lb.__len__()):
            s.doc.strs.append(lb[i])
        return StrRef(off, Int64(lb.__len__()))

    def compile_schema(mut self, mut s: Schema, jn: Int32, depth: Int32) -> Int32:
        if depth > MAX_DEPTH:
            self.depth_exceeded = True
            return -1
        var node = s.doc.nodes[Int(jn)]
        var idx = Int32(s.cnodes.__len__())
        if node.tag == TAG_TRUE:
            s.cnodes.append(CNode(1))
            self.commit_spath(s, idx)
            return idx
        if node.tag == TAG_FALSE:
            s.cnodes.append(CNode(2))
            self.commit_spath(s, idx)
            return idx
        if node.tag != TAG_OBJECT:
            # Defensive: the wrapper gate rejects non-object/bool schemas.
            s.cnodes.append(CNode(1))
            self.commit_spath(s, idx)
            return idx
        s.cnodes.append(CNode(0))
        self.commit_spath(s, idx)
        var base = Int(node.a)
        var count = Int(node.b)
        for mi in range(count):
            var m = s.doc.members[base + mi]
            var kref = StrRef(m.key_off, m.key_len)
            var v = m.value
            var vnode = s.doc.nodes[Int(v)]
            if bytes_eq(s.doc, kref, "type"):
                s.cnodes[Int(idx)].kw |= KW_TYPE
                var mask = UInt8(0)
                if vnode.tag == TAG_STRING:
                    mask = type_name_bit(s.doc, StrRef(vnode.a, vnode.b))
                elif vnode.tag == TAG_ARRAY:
                    for ei in range(Int(vnode.b)):
                        var el = s.doc.nodes[
                            Int(s.doc.slots[Int(vnode.a) + ei])
                        ]
                        if el.tag == TAG_STRING:
                            mask |= type_name_bit(s.doc, StrRef(el.a, el.b))
                s.cnodes[Int(idx)].type_mask = mask
            elif bytes_eq(s.doc, kref, "properties"):
                s.cnodes[Int(idx)].kw |= KW_PROPS
                if vnode.tag == TAG_OBJECT:
                    var mark = self.prop_scratch.__len__()
                    self.spath_scratch.append(self.lit_ref(s, "properties"))
                    for pi in range(Int(vnode.b)):
                        var pm = s.doc.members[Int(vnode.a) + pi]
                        self.spath_scratch.append(StrRef(pm.key_off, pm.key_len))
                        var child = self.compile_schema(s, pm.value, depth + 1)
                        _ = self.spath_scratch.pop()
                        if child < 0:
                            return -1
                        self.prop_scratch.append(
                            CProp(StrRef(pm.key_off, pm.key_len), child)
                        )
                    _ = self.spath_scratch.pop()
                    var n = self.prop_scratch.__len__() - mark
                    var off = Int32(s.cprops.__len__())
                    for i in range(n):
                        s.cprops.append(self.prop_scratch[mark + i])
                    self.prop_scratch.shrink(mark)
                    s.cnodes[Int(idx)].props_off = off
                    s.cnodes[Int(idx)].props_len = Int32(n)
            elif bytes_eq(s.doc, kref, "required"):
                s.cnodes[Int(idx)].kw |= KW_REQUIRED
                if vnode.tag == TAG_ARRAY:
                    var mark = self.req_scratch.__len__()
                    for ri in range(Int(vnode.b)):
                        var el = s.doc.nodes[Int(s.doc.slots[Int(vnode.a) + ri])]
                        if el.tag == TAG_STRING:
                            self.req_scratch.append(StrRef(el.a, el.b))
                    var n = self.req_scratch.__len__() - mark
                    var off = Int32(s.reqs.__len__())
                    for i in range(n):
                        s.reqs.append(self.req_scratch[mark + i])
                    self.req_scratch.shrink(mark)
                    s.cnodes[Int(idx)].req_off = off
                    s.cnodes[Int(idx)].req_len = Int32(n)
            elif bytes_eq(s.doc, kref, "items"):
                s.cnodes[Int(idx)].kw |= KW_ITEMS
                if vnode.tag == TAG_FALSE:
                    # Aggregate deny: one error for the whole array (the
                    # reference yields a single "extra items" error).
                    s.cnodes[Int(idx)].items = -2
                elif vnode.tag == TAG_TRUE:
                    pass  # no constraint
                else:
                    self.spath_scratch.append(self.lit_ref(s, "items"))
                    var child = self.compile_schema(s, v, depth + 1)
                    _ = self.spath_scratch.pop()
                    if child < 0:
                        return -1
                    s.cnodes[Int(idx)].items = child
            elif bytes_eq(s.doc, kref, "additionalProperties"):
                if vnode.tag == TAG_FALSE:
                    s.cnodes[Int(idx)].kw |= KW_AP
                    s.cnodes[Int(idx)].ap_mode = 2
                elif vnode.tag == TAG_TRUE:
                    pass  # no constraint
                else:
                    s.cnodes[Int(idx)].kw |= KW_AP
                    s.cnodes[Int(idx)].ap_mode = 1
                    self.spath_scratch.append(self.lit_ref(s, "additionalProperties"))
                    var child = self.compile_schema(s, v, depth + 1)
                    _ = self.spath_scratch.pop()
                    if child < 0:
                        return -1
                    s.cnodes[Int(idx)].ap = child
            elif bytes_eq(s.doc, kref, "enum"):
                s.cnodes[Int(idx)].kw |= KW_ENUM
                s.cnodes[Int(idx)].enum_node = v
            elif bytes_eq(s.doc, kref, "const"):
                s.cnodes[Int(idx)].kw |= KW_CONST
                s.cnodes[Int(idx)].const_node = v
            elif bytes_eq(s.doc, kref, "minimum"):
                s.cnodes[Int(idx)].kw |= KW_MIN
                s.cnodes[Int(idx)].vmin = vnode.num
            elif bytes_eq(s.doc, kref, "maximum"):
                s.cnodes[Int(idx)].kw |= KW_MAX
                s.cnodes[Int(idx)].vmax = vnode.num
            elif bytes_eq(s.doc, kref, "exclusiveMinimum"):
                s.cnodes[Int(idx)].kw |= KW_EXCLMIN
                s.cnodes[Int(idx)].vexclmin = vnode.num
            elif bytes_eq(s.doc, kref, "exclusiveMaximum"):
                s.cnodes[Int(idx)].kw |= KW_EXCLMAX
                s.cnodes[Int(idx)].vexclmax = vnode.num
            elif bytes_eq(s.doc, kref, "minLength"):
                s.cnodes[Int(idx)].kw |= KW_MINLEN
                s.cnodes[Int(idx)].iminlen = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "maxLength"):
                s.cnodes[Int(idx)].kw |= KW_MAXLEN
                s.cnodes[Int(idx)].imaxlen = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "pattern"):
                s.cnodes[Int(idx)].kw |= KW_PATTERN
                s.cnodes[Int(idx)].pattern = StrRef(vnode.a, vnode.b)
            elif bytes_eq(s.doc, kref, "minItems"):
                s.cnodes[Int(idx)].kw |= KW_MINITEMS
                s.cnodes[Int(idx)].iminitems = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "maxItems"):
                s.cnodes[Int(idx)].kw |= KW_MAXITEMS
                s.cnodes[Int(idx)].imaxitems = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "uniqueItems"):
                if vnode.tag == TAG_TRUE:
                    s.cnodes[Int(idx)].kw |= KW_UNIQUE
            elif bytes_eq(s.doc, kref, "minProperties"):
                s.cnodes[Int(idx)].kw |= KW_MINPROPS
                s.cnodes[Int(idx)].iminprops = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "maxProperties"):
                s.cnodes[Int(idx)].kw |= KW_MAXPROPS
                s.cnodes[Int(idx)].imaxprops = Int64(vnode.num)
            elif bytes_eq(s.doc, kref, "multipleOf"):
                s.cnodes[Int(idx)].kw |= KW_MULT
                s.cnodes[Int(idx)].vmult = vnode.num
                s.cnodes[Int(idx)].mof_token_int = vnode.nflags & NF_TOKEN_INT
            # Unknown keywords are ignored here; the wrapper's subset gate
            # rejects validation-affecting ones before compile.
        return idx

    def commit_spath(mut self, mut s: Schema, idx: Int32):
        var off = Int32(s.spath.__len__())
        var n = self.spath_scratch.__len__()
        for i in range(n):
            s.spath.append(self.spath_scratch[i])
        s.cnodes[Int(idx)].spath_off = off
        s.cnodes[Int(idx)].spath_len = Int32(n)


# --------------------------------------------------------------------------
# Deep equality (mirrors the reference's `equal`: bools never equal numbers,
# numbers compare by value across int/float form, containers recurse)
# --------------------------------------------------------------------------


def deep_equal(a: Doc, na: Int32, b: Doc, nb: Int32) -> Bool:
    var x = a.nodes[Int(na)]
    var y = b.nodes[Int(nb)]
    if x.tag == TAG_NUMBER and y.tag == TAG_NUMBER:
        return x.num == y.num
    if x.tag != y.tag:
        return False
    if x.tag == TAG_NULL or x.tag == TAG_TRUE or x.tag == TAG_FALSE:
        return True
    if x.tag == TAG_STRING:
        return keys_eq(a, StrRef(x.a, x.b), b, StrRef(y.a, y.b))
    if x.tag == TAG_ARRAY:
        if x.b != y.b:
            return False
        for i in range(Int(x.b)):
            if not deep_equal(
                a, a.slots[Int(x.a) + i], b, b.slots[Int(y.a) + i]
            ):
                return False
        return True
    # object: key sets must match pairwise (order-insensitive)
    if x.b != y.b:
        return False
    for i in range(Int(x.b)):
        var mx = a.members[Int(x.a) + i]
        var found = False
        for j in range(Int(y.b)):
            var my = b.members[Int(y.a) + j]
            if keys_eq(a, StrRef(mx.key_off, mx.key_len), b, StrRef(my.key_off, my.key_len)):
                if not deep_equal(a, mx.value, b, my.value):
                    return False
                found = True
                break
        if not found:
            return False
    return True


def fnv_mix(h: UInt64, byte: UInt8) -> UInt64:
    return (h ^ UInt64(byte)) * UInt64(0x100000001B3)


def structural_hash(d: Doc, n: Int32) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    var node = d.nodes[Int(n)]
    h = fnv_mix(h, node.tag)
    if node.tag == TAG_NUMBER:
        var v = node.num
        if v == 0.0:
            v = 0.0  # normalize -0.0 so it hashes like 0.0 (they are equal)
        var bits = UInt64(v.to_bits())
        for i in range(8):
            h = fnv_mix(h, UInt8((bits >> UInt64(i * 8)) & 0xFF))
    elif node.tag == TAG_STRING:
        for i in range(node.b):
            h = fnv_mix(h, d.strs[Int(node.a + Int64(i))])
    elif node.tag == TAG_ARRAY:
        h = fnv_mix(h, 0xA5)
        for i in range(node.b):
            h = h * UInt64(0x100000001B3) ^ structural_hash(
                d, d.slots[Int(node.a + i)]
            )
    elif node.tag == TAG_OBJECT:
        h = fnv_mix(h, 0x5A)
        # Order-insensitive: additively combine per-member mixes.
        var acc = UInt64(0)
        for i in range(node.b):
            var m = d.members[Int(node.a + i)]
            var kh = UInt64(0xCBF29CE484222325)
            for j in range(m.key_len):
                kh = fnv_mix(kh, d.strs[Int(m.key_off + Int64(j))])
            var vh = structural_hash(d, m.value)
            acc = acc + (kh ^ (vh * UInt64(0x9E3779B97F4A7C15)))
        h = h ^ acc
    return h


# --------------------------------------------------------------------------
# Output record writer
# --------------------------------------------------------------------------


struct PSeg(ImplicitlyCopyable, Copyable, Movable):
    var is_index: UInt8
    var key: StrRef  # when is_index == 0, into the instance doc's string arena
    var index: Int64

    def __init__(out self, is_index: UInt8, key: StrRef, index: Int64):
        self.is_index = is_index
        self.key = key
        self.index = index


struct OutCtx:
    var buf: U8Ptr
    var cap: Int64
    var length: Int64
    var overflow: Bool
    var n_err: Int32
    var n_defer: Int32
    var path: List[PSeg]

    def __init__(out self, buf: U8Ptr, cap: Int64):
        self.buf = buf
        self.cap = cap
        self.length = 0
        self.overflow = False
        self.n_err = 0
        self.n_defer = 0
        self.path = List[PSeg]()

    def put_u8(mut self, v: UInt8):
        self.buf[unsafe_offset=self.length] = v
        self.length += 1

    def put_u16(mut self, v: UInt16):
        self.buf[unsafe_offset=self.length] = UInt8(v & 0xFF)
        self.buf[unsafe_offset=self.length + 1] = UInt8((v >> 8) & 0xFF)
        self.length += 2

    def put_u32(mut self, v: UInt32):
        for i in range(4):
            self.buf[unsafe_offset=self.length + Int64(i)] = UInt8((v >> UInt32(i * 8)) & 0xFF)
        self.length += 4

    def put_i64(mut self, v: Int64):
        var uv = UInt64(v.to_bits())
        for i in range(8):
            self.buf[unsafe_offset=self.length + Int64(i)] = UInt8((uv >> UInt64(i * 8)) & 0xFF)
        self.length += 8

    def put_bytes[o: Origin](mut self, src: Pointer[UInt8, o], n: Int64):
        for i in range(n):
            self.buf[unsafe_offset=self.length + Int64(i)] = src[unsafe_offset=i]
        self.length += n

    def seg_size(self, is_index: UInt8, key_len: Int64) -> Int64:
        if is_index == 1:
            return 9
        return 5 + key_len

    def emit(
        mut self,
        kind: UInt8,
        kw: UInt32,
        idoc: Doc,
        sdoc: Doc,
        spath: List[StrRef],
        spath_off: Int32,
        spath_len: Int32,
        extra_kw_src: Pointer[UInt8, _],
        extra_kw_len: Int64,
        has_extra_kw: Bool,
    ) -> Bool:
        """Append one record. Returns False (and sets overflow) if it won't fit."""
        var n_schema = Int(spath_len) + (1 if has_extra_kw else 0)
        var needed = Int64(6)
        for i in range(self.path.__len__()):
            var p = self.path[i]
            needed += self.seg_size(p.is_index, p.key.len)
        for i in range(Int(spath_len)):
            needed += self.seg_size(0, spath[Int(spath_off) + i].len)
        if has_extra_kw:
            needed += self.seg_size(0, extra_kw_len)
        if self.length + needed > self.cap:
            self.overflow = True
            return False
        self.put_u8(kind)
        # keyword code as a single byte: the bit position of kw (0..20)
        var kw_byte = UInt8(0)
        var tmp = kw
        while tmp > 1:
            tmp >>= 1
            kw_byte += 1
        self.put_u8(kw_byte)
        self.put_u16(UInt16(self.path.__len__()))
        self.put_u16(UInt16(n_schema))
        for i in range(self.path.__len__()):
            var p = self.path[i]
            self.put_u8(p.is_index)
            if p.is_index == 1:
                self.put_i64(p.index)
            else:
                self.put_u32(UInt32(p.key.len))
                var src = idoc.strs.unsafe_ptr()
                self.put_bytes(src + Int(p.key.off), p.key.len)
        for i in range(Int(spath_len)):
            var r = spath[Int(spath_off) + i]
            self.put_u8(0)
            self.put_u32(UInt32(r.len))
            var src = sdoc.strs.unsafe_ptr()
            self.put_bytes(src + Int(r.off), r.len)
        if has_extra_kw:
            self.put_u8(0)
            self.put_u32(UInt32(extra_kw_len))
            self.put_bytes(extra_kw_src, extra_kw_len)
        if kind == REC_ERROR:
            self.n_err += 1
        else:
            self.n_defer += 1
        return True


# --------------------------------------------------------------------------
# Validation walk
# --------------------------------------------------------------------------


struct Walker:
    var s: Pointer[Schema, MutUntrackedOrigin]
    var idoc: Doc
    var out: OutCtx
    var kwn_bytes: List[UInt8]  # per-call arena holding the keyword names
    var kw_names: List[StrRef]  # kw bit position -> name ref into kwn_bytes

    def __init__(
        out self,
        s: Pointer[Schema, MutUntrackedOrigin],
        var idoc: Doc,
        var out: OutCtx,
    ):
        self.s = s
        self.idoc = idoc^
        self.out = out^
        self.kwn_bytes = List[UInt8]()
        self.kw_names = List[StrRef]()

    def add_kw_name(mut self, name: String):
        var lb = name.as_bytes()
        var off = Int64(self.kwn_bytes.__len__())
        for i in range(lb.__len__()):
            self.kwn_bytes.append(lb[i])
        self.kw_names.append(StrRef(off, Int64(lb.__len__())))

    def find_member(self, node: JNode, kdoc: Doc, key: StrRef) -> Int32:
        """Linear search for an object member by decoded key bytes.

        `key` points into `kdoc`'s string arena (the schema document for
        required/property names), member keys into the instance document.
        """
        for i in range(Int(node.b)):
            var m = self.idoc.members[Int(node.a) + i]
            if keys_eq(self.idoc, StrRef(m.key_off, m.key_len), kdoc, key):
                return m.value
        return -1

    def has_member(self, node: JNode, kdoc: Doc, key: StrRef) -> Bool:
        return self.find_member(node, kdoc, key) >= 0

    def in_props(self, cn: CNode, key: StrRef) -> Bool:
        for i in range(Int(cn.props_len)):
            if keys_eq(
                self.idoc,
                key,
                self.s[].doc,
                self.s[].cprops[Int(cn.props_off) + i].name,
            ):
                return True
        return False

    def emit_err(mut self, kw: UInt32, cn: CNode, kw_name_idx: Int) -> Bool:
        var r = self.kw_names[kw_name_idx]
        return self.out.emit(
            REC_ERROR,
            kw,
            self.idoc,
            self.s[].doc,
            self.s[].spath,
            cn.spath_off,
            cn.spath_len,
            self.kwn_bytes.unsafe_ptr() + Int(r.off),
            r.len,
            True,
        )

    def walk(mut self, cn_idx: Int32, jn: Int32) -> Bool:
        """False return = output buffer overflow; unwind immediately."""
        var cn = self.s[].cnodes[Int(cn_idx)]
        if cn.mode == 1:
            return True
        if cn.mode == 2:
            return self.out.emit(
                REC_ERROR,
                KW_FALSE_SCHEMA,
                self.idoc,
                self.s[].doc,
                self.s[].spath,
                cn.spath_off,
                cn.spath_len,
                self.s[].doc.strs.unsafe_ptr(),
                0,
                False,
            )
        var node = self.idoc.nodes[Int(jn)]

        if cn.kw & KW_TYPE != 0:
            var ok = False
            if node.tag == TAG_NULL:
                ok = cn.type_mask & T_NULL != 0
            elif node.tag == TAG_TRUE or node.tag == TAG_FALSE:
                ok = cn.type_mask & T_BOOL != 0
            elif node.tag == TAG_NUMBER:
                ok = (cn.type_mask & T_NUMBER != 0) or (
                    cn.type_mask & T_INTEGER != 0 and node.nflags & NF_INTEGRAL != 0
                )
            elif node.tag == TAG_STRING:
                ok = cn.type_mask & T_STRING != 0
            elif node.tag == TAG_ARRAY:
                ok = cn.type_mask & T_ARRAY != 0
            else:
                ok = cn.type_mask & T_OBJECT != 0
            if not ok and not self.emit_err(KW_TYPE, cn, 0):
                return False

        if cn.kw & KW_ENUM != 0:
            var enode = self.s[].doc.nodes[Int(cn.enum_node)]
            var any = False
            if enode.tag == TAG_ARRAY:
                for i in range(Int(enode.b)):
                    if deep_equal(
                        self.idoc,
                        jn,
                        self.s[].doc,
                        self.s[].doc.slots[Int(enode.a) + i],
                    ):
                        any = True
                        break
            if not any and not self.emit_err(KW_ENUM, cn, 5):
                return False

        if cn.kw & KW_CONST != 0:
            if not deep_equal(self.idoc, jn, self.s[].doc, cn.const_node):
                if not self.emit_err(KW_CONST, cn, 6):
                    return False

        if node.tag == TAG_NUMBER:
            var v = node.num
            if cn.kw & KW_MIN != 0 and v < cn.vmin:
                if not self.emit_err(KW_MIN, cn, 7):
                    return False
            if cn.kw & KW_MAX != 0 and v > cn.vmax:
                if not self.emit_err(KW_MAX, cn, 8):
                    return False
            if cn.kw & KW_EXCLMIN != 0 and v <= cn.vexclmin:
                if not self.emit_err(KW_EXCLMIN, cn, 9):
                    return False
            if cn.kw & KW_EXCLMAX != 0 and v >= cn.vexclmax:
                if not self.emit_err(KW_EXCLMAX, cn, 10):
                    return False
            if cn.kw & KW_MULT != 0:
                var d = cn.vmult
                if cn.mof_token_int != 0:
                    # Python `instance % dB` with an integer divisor: only
                    # zero/nonzero matters; IEEE remainder is exact and has
                    # the same zeros as floored modulo. Large dividends are
                    # deferred to the wrapper's exact path.
                    if not (v > -MULT_EXACT_LIMIT and v < MULT_EXACT_LIMIT):
                        if not self.emit_defer(DEFER_MULT_EXACT, cn, 19):
                            return False
                    else:
                        var r = v - trunc(v / d) * d
                        if r != 0.0 and not self.emit_err(KW_MULT, cn, 19):
                            return False
                else:
                    var q = v / d
                    if not (q > -1e308 and q < 1e308):  # non-finite quotient
                        if not self.emit_defer(DEFER_MULT_EXACT, cn, 19):
                            return False
                    elif trunc(q) != q:
                        if not self.emit_err(KW_MULT, cn, 19):
                            return False

        if node.tag == TAG_STRING:
            if cn.kw & (KW_MINLEN | KW_MAXLEN) != 0:
                var cps = Int64(0)
                for i in range(node.b):
                    if self.idoc.strs[Int(node.a + i)] & 0xC0 != 0x80:
                        cps += 1
                if cn.kw & KW_MINLEN != 0 and cps < cn.iminlen:
                    if not self.emit_err(KW_MINLEN, cn, 11):
                        return False
                if cn.kw & KW_MAXLEN != 0 and cps > cn.imaxlen:
                    if not self.emit_err(KW_MAXLEN, cn, 12):
                        return False
            if cn.kw & KW_PATTERN != 0:
                if not self.emit_defer(DEFER_PATTERN, cn, 13):
                    return False

        if node.tag == TAG_ARRAY:
            var n = Int(node.b)
            if cn.kw & KW_MINITEMS != 0 and Int64(n) < cn.iminitems:
                if not self.emit_err(KW_MINITEMS, cn, 14):
                    return False
            if cn.kw & KW_MAXITEMS != 0 and Int64(n) > cn.imaxitems:
                if not self.emit_err(KW_MAXITEMS, cn, 15):
                    return False
            if cn.kw & KW_UNIQUE != 0 and n > 1:
                if not self.check_unique(node):
                    if not self.emit_err(KW_UNIQUE, cn, 16):
                        return False
            if cn.kw & KW_ITEMS != 0:
                if cn.items == -2:
                    if n > 0 and not self.emit_err(KW_ITEMS, cn, 3):
                        return False
                elif cn.items >= 0:
                    for i in range(n):
                        self.out.path.append(PSeg(1, StrRef(0, 0), Int64(i)))
                        if not self.walk(cn.items, self.idoc.slots[Int(node.a) + i]):
                            return False
                        _ = self.out.path.pop()

        if node.tag == TAG_OBJECT:
            if cn.kw & KW_REQUIRED != 0:
                for i in range(Int(cn.req_len)):
                    var r = self.s[].reqs[Int(cn.req_off) + i]
                    if not self.has_member(node, self.s[].doc, r):
                        # required errors carry the missing property name;
                        # the wrapper reads it from the schema via schema_path.
                        if not self.emit_err(KW_REQUIRED, cn, 2):
                            return False
            if cn.kw & KW_MINPROPS != 0 and node.b < cn.iminprops:
                if not self.emit_err(KW_MINPROPS, cn, 17):
                    return False
            if cn.kw & KW_MAXPROPS != 0 and node.b > cn.imaxprops:
                if not self.emit_err(KW_MAXPROPS, cn, 18):
                    return False
            if cn.kw & KW_PROPS != 0:
                for i in range(Int(cn.props_len)):
                    var p = self.s[].cprops[Int(cn.props_off) + i]
                    # Descend into the matching member; the error path carries
                    # the instance's own key bytes (byte-equal to the schema's
                    # property name, but from the instance's string arena).
                    for mi in range(Int(node.b)):
                        var m = self.idoc.members[Int(node.a) + mi]
                        if keys_eq(
                            self.idoc,
                            StrRef(m.key_off, m.key_len),
                            self.s[].doc,
                            p.name,
                        ):
                            self.out.path.append(
                                PSeg(0, StrRef(m.key_off, m.key_len), 0)
                            )
                            if not self.walk(p.cnode, m.value):
                                return False
                            _ = self.out.path.pop()
                            break
            if cn.kw & KW_AP != 0:
                if cn.ap_mode == 2:
                    var extras = 0
                    for i in range(Int(node.b)):
                        var m = self.idoc.members[Int(node.a) + i]
                        if not self.in_props(cn, StrRef(m.key_off, m.key_len)):
                            extras += 1
                    if extras > 0 and not self.emit_err(KW_AP, cn, 4):
                        return False
                else:
                    for i in range(Int(node.b)):
                        var m = self.idoc.members[Int(node.a) + i]
                        var kref = StrRef(m.key_off, m.key_len)
                        if not self.in_props(cn, kref):
                            self.out.path.append(PSeg(0, kref, 0))
                            if not self.walk(cn.ap, m.value):
                                return False
                            _ = self.out.path.pop()
        return True

    def emit_defer(mut self, kind: UInt8, cn: CNode, kw_name_idx: Int) -> Bool:
        var r = self.kw_names[kw_name_idx]
        return self.out.emit(
            REC_DEFER,
            UInt32(1) << UInt32(kw_name_idx),
            self.idoc,
            self.s[].doc,
            self.s[].spath,
            cn.spath_off,
            cn.spath_len,
            self.kwn_bytes.unsafe_ptr() + Int(r.off),
            r.len,
            True,
        )

    def check_unique(mut self, node: JNode) -> Bool:
        """False when the array has duplicate elements (reference `equal`)."""
        var n = Int(node.b)
        var hashes = List[UInt64]()
        var order = List[Int32]()
        for i in range(n):
            hashes.append(structural_hash(self.idoc, self.idoc.slots[Int(node.a) + i]))
            order.append(Int32(i))
        # Deterministic quicksort of `order` by hash.
        self.sort_by_hash(order, hashes, 0, n - 1)
        var i = 0
        while i < n:
            var j = i + 1
            while j < n and hashes[Int(order[j])] == hashes[Int(order[i])]:
                j += 1
            if j - i > 1:
                # Same-hash run: pairwise deep compare.
                for a in range(i, j):
                    for b in range(a + 1, j):
                        if deep_equal(
                            self.idoc,
                            self.idoc.slots[Int(node.a) + Int(order[a])],
                            self.idoc,
                            self.idoc.slots[Int(node.a) + Int(order[b])],
                        ):
                            return False
            i = j
        return True

    def sort_by_hash(mut self, mut order: List[Int32], hashes: List[UInt64], lo: Int, hi: Int):
        if lo >= hi:
            return
        var pivot = hashes[Int(order[(lo + hi) // 2])]
        var i = lo
        var j = hi
        while i <= j:
            while hashes[Int(order[i])] < pivot:
                i += 1
            while hashes[Int(order[j])] > pivot:
                j -= 1
            if i <= j:
                var tmp = order[i]
                order[i] = order[j]
                order[j] = tmp
                i += 1
                j -= 1
        if lo < j:
            self.sort_by_hash(order, hashes, lo, j)
        if i < hi:
            self.sort_by_hash(order, hashes, i, hi)


# --------------------------------------------------------------------------
# Exported ABI
# --------------------------------------------------------------------------


@export
def jsonschemamojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def jsonschemamojo_schema_compile(
    text: U8Ptr, length: Int64
) abi("C") -> Handle:
    if length < 0 or length > 0x7FFF_FFFF:
        return None
    var parser = Parser(text, length)
    var doc = Doc()
    var root = parser.parse_value(doc, 0)
    if parser.failed:
        return None
    parser.skip_ws()
    if parser.pos != parser.length:
        return None
    doc.root = root
    var schema = Schema(doc^)
    var compiler = Compiler()
    var croot = compiler.compile_schema(schema, root, 0)
    if compiler.depth_exceeded or croot < 0:
        return None
    schema.root = croot
    schema.flags = schema.doc.flags
    var boxed = unsafe_alloc[Schema](1)
    boxed[] = schema^
    return boxed.unsafe_bitcast[UInt8]()


@export
def jsonschemamojo_schema_flags(handle: Handle) abi("C") -> Int32:
    if not handle:
        return 0
    var s = handle.value().unsafe_bitcast[Schema]()
    return Int32(s[].flags)


@export
def jsonschemamojo_schema_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var s = handle.value().unsafe_bitcast[Schema]()
    # Move the Schema out so its List fields' destructors release their
    # buffers, then free the box itself.
    var owned = s.take_pointee()
    _ = owned^
    s.unsafe_free()


@export
def jsonschemamojo_validate(
    handle: Handle,
    inst: U8Ptr,
    inst_len: Int64,
    out_buf: U8Ptr,
    out_cap: Int64,
    meta: I32Ptr,
) abi("C") -> Int32:
    if not handle:
        return 1
    if inst_len < 0 or inst_len > 0x7FFF_FFFF:
        return 2
    var s = handle.value().unsafe_bitcast[Schema]()

    var parser = Parser(inst, inst_len)
    var idoc = Doc()
    var root = parser.parse_value(idoc, 0)
    if parser.failed:
        var status = Int32(3) if parser.depth_exceeded else Int32(2)
        meta[unsafe_offset=0] = status
        return status
    parser.skip_ws()
    if parser.pos != parser.length:
        meta[unsafe_offset=0] = 2
        return 2

    var outctx = OutCtx(out_buf, out_cap)
    var walker = Walker(s, idoc^, outctx^)
    # Keyword names, indexed by bit position, kept in the walker's own arena
    # (validating must never mutate the shared compiled schema).
    walker.add_kw_name("type")  # 0
    walker.add_kw_name("properties")  # 1
    walker.add_kw_name("required")  # 2
    walker.add_kw_name("items")  # 3
    walker.add_kw_name("additionalProperties")  # 4
    walker.add_kw_name("enum")  # 5
    walker.add_kw_name("const")  # 6
    walker.add_kw_name("minimum")  # 7
    walker.add_kw_name("maximum")  # 8
    walker.add_kw_name("exclusiveMinimum")  # 9
    walker.add_kw_name("exclusiveMaximum")  # 10
    walker.add_kw_name("minLength")  # 11
    walker.add_kw_name("maxLength")  # 12
    walker.add_kw_name("pattern")  # 13
    walker.add_kw_name("minItems")  # 14
    walker.add_kw_name("maxItems")  # 15
    walker.add_kw_name("uniqueItems")  # 16
    walker.add_kw_name("minProperties")  # 17
    walker.add_kw_name("maxProperties")  # 18
    walker.add_kw_name("multipleOf")  # 19
    walker.add_kw_name("false")  # 20 (unused: false-schema errors carry no keyword)

    _ = walker.walk(s[].root, root)

    meta[unsafe_offset=0] = 0
    meta[unsafe_offset=1] = walker.out.n_err
    meta[unsafe_offset=2] = walker.out.n_defer
    meta[unsafe_offset=3] = 1 if walker.out.overflow else 0
    meta[unsafe_offset=4] = 1 if walker.idoc.flags & FLAG_BIG_NUMBER != 0 else 0
    meta[unsafe_offset=5] = Int32(walker.out.length & 0x7FFF_FFFF)
    return 0
