"""Clean-room TOML 1.0 parser kernel (tomlmojo).

Written fresh from the public TOML 1.0.0 specification (ABNF grammar) plus
black-box observed behavior of the CPython 3.12 `tomllib` reference parser
(accept/reject verdicts on edge cases, probed empirically — no parser source
was read or adapted). The kernel parses a UTF-8 TOML document in one pass and
emits a compact binary record stream; the Python wrapper assembles that
stream into the same typed objects `tomllib.loads` would return.

Exported C ABI (single-shot: parse a whole document per call):

    int32_t  tomlmojo_abi_version(void)
    void*    tomlmojo_parse(const uint8_t* data, int64_t length)
    int32_t  tomlmojo_result_status(void* handle)     -> 0 ok / 1 parse error
    int64_t  tomlmojo_result_error_pos(void* handle)  -> byte offset, -1 if ok
    int64_t  tomlmojo_result_size(void* handle)       -> stream byte length
    uint8_t* tomlmojo_result_data(void* handle)       -> stream bytes
    void     tomlmojo_result_destroy(void* handle)

Record stream format (all integers little-endian):

    "TMO1" magic, then records:
    0x01 STRUCT_TABLE: u16 nsegs, segs            # materialize {} at path
    0x02 AOT_ELEM:     u16 nsegs, segs, i32 index # append {} to list at path
    0x03 VALUE:        u16 nsegs, segs, value     # set value at path
    seg: u8 flags (bit0 = array-of-tables descent), [i32 elem index if bit0],
         u32 name byte length, name bytes
    value: tagged, recursive:
    0x01 string        u32 len + UTF-8 bytes
    0x02 int-dec       u32 len + ASCII literal (Python int() handles bignums)
    0x03 int-hex       u32 len + literal        0x04 int-oct, 0x05 int-bin
    0x06 float         u32 len + ASCII literal (Python float() is exact)
    0x07 bool          u8 0/1
    0x08 array         u32 count + values
    0x09 inline-table  u32 count + entries; entry =
                       u16 nsegs + (u32 keylen + key bytes)* + value
                       (dotted keys inside inline tables nest on decode)
    0x0A date          u16 year, u8 month, u8 day
    0x0B time          u8 hour, u8 min, u8 sec, u32 microsecond
    0x0C local-datetime    date + time fields
    0x0D offset-datetime   date + time fields + i16 offset minutes

TOML's table/duplicate-key validity rules are enforced during the parse with
a flat symbol table mapping normalized key paths (length-prefixed segments,
with array-of-tables segments carrying their current element index) to one of:
IMPLICIT (header-prefix table), DEFINED (explicit [table]), DOTTED (created
by a dotted key), VALUE (leaf incl. inline tables and static arrays), AOT.
The state machine mirrors tomllib's observable accept/reject behavior.
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

# Symbol-table states for normalized key paths.
comptime ST_MISSING: Int8 = 0
comptime ST_IMPLICIT: Int8 = 1  # table created as a [header] path prefix
comptime ST_DEFINED: Int8 = 2  # table explicitly defined by a [header]
comptime ST_DOTTED: Int8 = 3  # table created by a dotted key
comptime ST_VALUE: Int8 = 4  # leaf value (incl. inline tables, static arrays)
comptime ST_AOT: Int8 = 5  # array of tables

# Record tags.
comptime REC_TABLE: Int = 1
comptime REC_AOT_ELEM: Int = 2
comptime REC_VALUE: Int = 3

# Value tags.
comptime VT_STR: Int = 1
comptime VT_INT_DEC: Int = 2
comptime VT_INT_HEX: Int = 3
comptime VT_INT_OCT: Int = 4
comptime VT_INT_BIN: Int = 5
comptime VT_FLOAT: Int = 6
comptime VT_BOOL: Int = 7
comptime VT_ARRAY: Int = 8
comptime VT_INLINE: Int = 9
comptime VT_DATE: Int = 10
comptime VT_TIME: Int = 11
comptime VT_DT_LOCAL: Int = 12
comptime VT_DT_OFFSET: Int = 13

# Byte constants (ASCII).
comptime CH_TAB: Int = 0x09
comptime CH_LF: Int = 0x0A
comptime CH_CR: Int = 0x0D
comptime CH_SPACE: Int = 0x20
comptime CH_QUOTE: Int = 0x22  # "
comptime CH_HASH: Int = 0x23  # #
comptime CH_APOS: Int = 0x27  # '
comptime CH_PLUS: Int = 0x2B  # +
comptime CH_COMMA: Int = 0x2C  # ,
comptime CH_MINUS: Int = 0x2D  # -
comptime CH_DOT: Int = 0x2E  # .
comptime CH_ZERO: Int = 0x30  # 0
comptime CH_NINE: Int = 0x39  # 9
comptime CH_COLON: Int = 0x3A  # :
comptime CH_EQ: Int = 0x3D  # =
comptime CH_A_UP: Int = 0x41
comptime CH_E_UP: Int = 0x45  # E
comptime CH_T_UP: Int = 0x54  # T
comptime CH_Z_UP: Int = 0x5A  # Z
comptime CH_LBRACK: Int = 0x5B  # [
comptime CH_BSLASH: Int = 0x5C  # \
comptime CH_RBRACK: Int = 0x5D  # ]
comptime CH_UNDER: Int = 0x5F  # _
comptime CH_A: Int = 0x61  # a
comptime CH_B: Int = 0x62  # b
comptime CH_E: Int = 0x65  # e
comptime CH_F: Int = 0x66  # f
comptime CH_I: Int = 0x69  # i
comptime CH_L: Int = 0x6C  # l
comptime CH_N: Int = 0x6E  # n
comptime CH_O: Int = 0x6F  # o
comptime CH_R: Int = 0x72  # r
comptime CH_T: Int = 0x74  # t
comptime CH_U: Int = 0x75  # u
comptime CH_X: Int = 0x78  # x
comptime CH_Z: Int = 0x7A  # z
comptime CH_LBRACE: Int = 0x7B  # {
comptime CH_RBRACE: Int = 0x7D  # }
comptime CH_DEL: Int = 0x7F

# Value-nesting (array/inline-table) depth limit. Far above stdlib tomllib's
# own practical recursion limit (~450 array levels on a default 1000-frame
# Python stack), so no document tomllib can parse is rejected here — while
# pathological nesting can never overflow this kernel's native stack.
comptime MAX_VALUE_DEPTH: Int = 2000


def _is_digit(c: Int) -> Bool:
    return CH_ZERO <= c <= CH_NINE


def _is_hexdig(c: Int) -> Bool:
    return (
        _is_digit(c)
        or (CH_A_UP <= c <= 0x46)
        or (CH_A <= c <= 0x66)
    )


def _is_alpha(c: Int) -> Bool:
    return (CH_A_UP <= c <= 0x5A) or (CH_A <= c <= 0x7A)


def _is_bare_key(c: Int) -> Bool:
    return _is_alpha(c) or _is_digit(c) or c == CH_MINUS or c == CH_UNDER


def _is_ws(c: Int) -> Bool:
    return c == CH_SPACE or c == CH_TAB


struct Seg(Copyable, Movable):
    """One original (output-facing) key path segment.

    `aot_index` >= 0 means: descend into element `aot_index` of the
    array-of-tables named `name`; -1 means a plain table/key segment.
    """

    var name: String
    var aot_index: Int

    def __init__(out self, name: String, aot_index: Int):
        self.name = name
        self.aot_index = aot_index


struct ParseResult(Copyable, Movable):
    """Owned parse outcome, read back through the result accessors."""

    var data: U8Ptr  # [size] output record stream (may be null when size==0)
    var size: Int64
    var status: Int32  # 0 ok, 1 parse error
    var err_pos: Int64  # byte offset of the error, -1 when ok

    def __init__(
        out self, data: U8Ptr, size: Int64, status: Int32, err_pos: Int64
    ):
        self.data = data
        self.size = size
        self.status = status
        self.err_pos = err_pos


def _enc_seg(name: String) -> String:
    """Length-prefixed segment encoding for the normalized-path symbol table.

    Deterministic and collision-free for arbitrary (including NUL-containing
    and non-ASCII) key bytes; keys are only ever compared as whole strings.
    """
    return String(name.byte_length()) + ":" + name


def _encode_seg_into(mut dst: List[UInt8], name: String, aot_index: Int):
    """Append one path-segment encoding to a byte buffer (same layout as
    Parser.e_seg_parts)."""
    if aot_index >= 0:
        dst.append(1)
        var v = aot_index
        for shift in range(0, 32, 8):
            dst.append(UInt8((v >> shift) & 0xFF))
    else:
        dst.append(0)
    var n = name.byte_length()
    for shift in range(0, 32, 8):
        dst.append(UInt8((n >> shift) & 0xFF))
    for b in name.as_bytes():
        dst.append(b)


struct StateNode(Copyable, Movable):
    """One node of the key-path state trie.

    TOML's table/duplicate-key rules are enforced with this trie instead of
    path strings: each node is a table-ish thing with a state (IMPLICIT /
    DEFINED / DOTTED / VALUE / AOT), and children keyed by raw segment name
    (no path concatenation anywhere). An AOT node's `children` are scoped to
    its current (last) element and are cleared each time a new element is
    appended — exactly the per-element key scoping of TOML arrays of tables.
    """

    var state: Int8  # ST_*
    var aot_count: Int  # elements appended so far (arrays of tables only)
    var children: Dict[String, Int]  # segment name -> index into Parser.nodes

    def __init__(out self, state: Int8):
        self.state = state
        self.aot_count = 0
        self.children = Dict[String, Int]()


struct Parser(Movable):
    """Single-pass recursive-descent TOML parser over the input bytes."""

    var src: U8Ptr
    var n: Int
    var pos: Int
    var nodes: List[StateNode]  # state-trie arena; nodes[0] is the root table
    var ctx_node: Int  # nodes index of the current table context
    var out: List[UInt8]
    var err: Int  # byte offset of first error, -1 = none
    var depth: Int  # value-nesting depth (arrays / inline tables)
    var ctx_emit: List[UInt8]  # pre-encoded path segments of the context
    var ctx_nsegs: Int  # number of segments in ctx_emit

    def __init__(out self, src: U8Ptr, n: Int):
        self.src = src
        self.n = n
        self.pos = 0
        self.nodes = List[StateNode]()
        self.nodes.append(StateNode(ST_DEFINED))  # root table
        self.ctx_node = 0
        self.out = List[UInt8]()
        self.err = -1
        self.depth = 0
        self.ctx_emit = List[UInt8]()
        self.ctx_nsegs = 0

    # ---- input access -------------------------------------------------

    def at_end(self) -> Bool:
        return self.pos >= self.n

    def ch(self) -> Int:
        """Byte at the cursor, or -1 at end of input (NUL-safe)."""
        if self.pos >= self.n:
            return -1
        return Int(self.src[unsafe_offset=self.pos])

    def ch_at(self, off: Int) -> Int:
        var at = self.pos + off
        if at < 0 or at >= self.n:
            return -1
        return Int(self.src[unsafe_offset=at])

    def fail(mut self, pos: Int) raises:
        if self.err < 0:
            self.err = pos
        raise Error("invalid TOML")

    # ---- output helpers ------------------------------------------------

    def e_u8(mut self, v: Int):
        self.out.append(UInt8(v & 0xFF))

    def e_u16(mut self, v: Int):
        self.out.append(UInt8(v & 0xFF))
        self.out.append(UInt8((v >> 8) & 0xFF))

    def e_u32(mut self, v: Int):
        self.out.append(UInt8(v & 0xFF))
        self.out.append(UInt8((v >> 8) & 0xFF))
        self.out.append(UInt8((v >> 16) & 0xFF))
        self.out.append(UInt8((v >> 24) & 0xFF))

    def patch_u32(mut self, at: Int, v: Int):
        self.out[at] = UInt8(v & 0xFF)
        self.out[at + 1] = UInt8((v >> 8) & 0xFF)
        self.out[at + 2] = UInt8((v >> 16) & 0xFF)
        self.out[at + 3] = UInt8((v >> 24) & 0xFF)

    def e_bytes(mut self, data: List[UInt8]):
        """Bulk-copy a byte list into the output stream."""
        var n = len(data)
        if n == 0:
            return
        var old = len(self.out)
        self.out.resize(old + n, UInt8(0))
        unsafe_memcpy(dest=self.out.unsafe_ptr() + old, src=data.unsafe_ptr(), count=n)

    def e_str(mut self, s: String):
        if s.byte_length() == 0:
            return
        self.out.extend(s.as_bytes())

    def e_src_slice(mut self, start: Int, end: Int):
        """Bulk-copy src[start:end] into the output stream."""
        var n = end - start
        if n <= 0:
            return
        var sp = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.src + start, length=n
        )
        self.out.extend(sp)

    def e_seg_parts(mut self, name: String, aot_index: Int):
        if aot_index >= 0:
            self.e_u8(1)
            self.e_u32(aot_index)
        else:
            self.e_u8(0)
        self.e_u32(name.byte_length())
        self.e_str(name)

    def e_seg(mut self, s: Seg):
        self.e_seg_parts(s.name, s.aot_index)

    def e_path(mut self, segs: List[Seg]):
        self.e_u16(len(segs))
        for s in segs:
            self.e_seg(s)

    def _new_node(mut self, state: Int8) -> Int:
        self.nodes.append(StateNode(state))
        return len(self.nodes) - 1

    # ---- whitespace / comments / newlines ------------------------------

    def skip_ws(mut self):
        while _is_ws(self.ch()):
            self.pos += 1

    def skip_newline(mut self) raises -> Bool:
        """Consume one newline (LF or CRLF); a lone CR is an error."""
        var c = self.ch()
        if c == CH_LF:
            self.pos += 1
            return True
        if c == CH_CR:
            if self.ch_at(1) != CH_LF:
                self.fail(self.pos)
            self.pos += 2
            return True
        return False

    def skip_comment(mut self) raises:
        """Consume a comment (cursor at '#') up to (not incl.) the newline."""
        self.pos += 1  # '#'
        while True:
            var c = self.ch()
            if c < 0 or c == CH_LF or c == CH_CR:
                return
            # comment chars: tab, 0x20-0x7E, non-ASCII (>= 0x80)
            if c < 0x80 and (c < CH_SPACE or c > 0x7E) and c != CH_TAB:
                self.fail(self.pos)
            self.pos += 1

    def skip_trivia(mut self) raises:
        """Consume any mix of ws, newlines and comments (statement gaps and
        array interiors)."""
        while True:
            self.skip_ws()
            var c = self.ch()
            if c == CH_HASH:
                self.skip_comment()
            elif self.skip_newline():
                pass
            else:
                return

    def expect_stmt_end(mut self) raises:
        """After a statement: ws, optional comment, then newline or EOF."""
        self.skip_ws()
        if self.ch() == CH_HASH:
            self.skip_comment()
        if self.at_end():
            return
        if not self.skip_newline():
            self.fail(self.pos)

    # ---- string parsing ------------------------------------------------

    def _hex_val(mut self, c: Int) -> Int:
        if _is_digit(c):
            return c - CH_ZERO
        if CH_A_UP <= c <= 0x46:
            return c - CH_A_UP + 10
        if CH_A <= c <= 0x66:
            return c - CH_A + 10
        return -1

    def _utf8_encode(mut self, mut dst: List[UInt8], cp: Int):
        if cp < 0x80:
            dst.append(UInt8(cp))
        elif cp < 0x800:
            dst.append(UInt8(0xC0 | (cp >> 6)))
            dst.append(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            dst.append(UInt8(0xE0 | (cp >> 12)))
            dst.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            dst.append(UInt8(0x80 | (cp & 0x3F)))
        else:
            dst.append(UInt8(0xF0 | (cp >> 18)))
            dst.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            dst.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            dst.append(UInt8(0x80 | (cp & 0x3F)))

    def _escape(mut self, mut dst: List[UInt8]) raises:
        """Process one escape sequence (cursor just past the backslash)."""
        var c = self.ch()
        if c == 0x62:  # b
            dst.append(UInt8(0x08))
        elif c == CH_T:
            dst.append(UInt8(CH_TAB))
        elif c == CH_N:
            dst.append(UInt8(CH_LF))
        elif c == CH_F:
            dst.append(UInt8(0x0C))
        elif c == CH_R:
            dst.append(UInt8(CH_CR))
        elif c == CH_QUOTE:
            dst.append(UInt8(CH_QUOTE))
        elif c == CH_BSLASH:
            dst.append(UInt8(CH_BSLASH))
        elif c == CH_U or c == 0x55:  # u / U
            var ndig = 4
            if c == 0x55:
                ndig = 8
            var cp = 0
            for i in range(ndig):
                var h = self._hex_val(self.ch_at(1 + i))
                if h < 0:
                    self.fail(self.pos + 1 + i)
                cp = cp * 16 + h
            if cp > 0x10FFFF or (0xD800 <= cp <= 0xDFFF):
                self.fail(self.pos)
            self.pos += ndig
            self._utf8_encode(dst, cp)
        else:
            self.fail(self.pos)
        self.pos += 1

    def _basic_char_check(mut self, c: Int) raises:
        """Reject raw characters illegal in any string kind."""
        if c != CH_TAB and c < CH_SPACE:
            self.fail(self.pos)
        if c == CH_DEL:
            self.fail(self.pos)

    def _ml_char_check(mut self, c: Int) raises:
        """Multiline strings additionally allow raw LF (CR is handled by
        its own branch, which normalizes CRLF to LF)."""
        if c == CH_LF:
            return
        self._basic_char_check(c)

    def _flush_run(mut self, mut dst: List[UInt8], start: Int, end: Int):
        """Bulk-copy a raw source span into a string decode buffer."""
        var n = end - start
        if n <= 0:
            return
        var sp = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.src + start, length=n
        )
        dst.extend(sp)

    def parse_basic_to(mut self, mut dst: List[UInt8]) raises:
        """Parse a basic string (cursor at opening '"'), decode into dst."""
        self.pos += 1
        var run = self.pos
        while True:
            var c = self.ch()
            if c < 0 or c == CH_LF or c == CH_CR:
                self.fail(self.pos)
            if c == CH_QUOTE:
                self._flush_run(dst, run, self.pos)
                self.pos += 1
                return
            if c == CH_BSLASH:
                self._flush_run(dst, run, self.pos)
                self.pos += 1
                self._escape(dst)
                run = self.pos
                continue
            self._basic_char_check(c)
            self.pos += 1

    def parse_literal_to(mut self, mut dst: List[UInt8]) raises:
        """Parse a literal string (cursor at opening quote), raw into dst."""
        self.pos += 1
        var run = self.pos
        while True:
            var c = self.ch()
            if c < 0 or c == CH_LF or c == CH_CR:
                self.fail(self.pos)
            if c == CH_APOS:
                self._flush_run(dst, run, self.pos)
                self.pos += 1
                return
            self._basic_char_check(c)
            self.pos += 1

    def _ml_first_newline(mut self) raises:
        """Skip the one newline allowed right after a multiline delimiter."""
        _ = self.skip_newline()

    def parse_ml_basic_to(mut self, mut dst: List[UInt8]) raises:
        """Parse a multiline basic string (cursor at the opening delimiter)."""
        self.pos += 3
        self._ml_first_newline()
        var run = self.pos
        while True:
            var c = self.ch()
            if c < 0:
                self.fail(self.pos)
            if c == CH_QUOTE:
                var qrun = 0
                while self.ch_at(qrun) == CH_QUOTE:
                    qrun += 1
                if qrun < 3:
                    # Runs of one or two quotes are content; keep them in
                    # the current raw run.
                    self.pos += qrun
                    continue
                self._flush_run(dst, run, self.pos)
                if qrun > 5:
                    self.fail(self.pos)
                for _ in range(qrun - 3):
                    dst.append(UInt8(CH_QUOTE))
                self.pos += qrun
                return
            if c == CH_BSLASH:
                self._flush_run(dst, run, self.pos)
                self.pos += 1
                var e = self.ch()
                if (
                    e == 0x62
                    or e == CH_T
                    or e == CH_N
                    or e == CH_F
                    or e == CH_R
                    or e == CH_QUOTE
                    or e == CH_BSLASH
                    or e == CH_U
                    or e == 0x55
                ):
                    self._escape(dst)
                    run = self.pos
                    continue
                # Line-ending backslash: ws*, one newline, then all
                # whitespace/newlines are trimmed.
                if e < 0:
                    self.fail(self.pos)
                var save = self.pos
                self.skip_ws()
                if not self.skip_newline():
                    self.fail(save)
                while True:
                    self.skip_ws()
                    if not self.skip_newline():
                        break
                run = self.pos
                continue
            if c == CH_CR:
                self._flush_run(dst, run, self.pos)
                if self.ch_at(1) != CH_LF:
                    self.fail(self.pos)
                dst.append(UInt8(CH_LF))
                self.pos += 2
                run = self.pos
                continue
            self._ml_char_check(c)
            self.pos += 1

    def parse_ml_literal_to(mut self, mut dst: List[UInt8]) raises:
        """Parse a multiline literal string (cursor at opening "'''")."""
        self.pos += 3
        self._ml_first_newline()
        var run = self.pos
        while True:
            var c = self.ch()
            if c < 0:
                self.fail(self.pos)
            if c == CH_APOS:
                var qrun = 0
                while self.ch_at(qrun) == CH_APOS:
                    qrun += 1
                if qrun < 3:
                    self.pos += qrun
                    continue
                self._flush_run(dst, run, self.pos)
                if qrun > 5:
                    self.fail(self.pos)
                for _ in range(qrun - 3):
                    dst.append(UInt8(CH_APOS))
                self.pos += qrun
                return
            if c == CH_CR:
                self._flush_run(dst, run, self.pos)
                if self.ch_at(1) != CH_LF:
                    self.fail(self.pos)
                dst.append(UInt8(CH_LF))
                self.pos += 2
                run = self.pos
                continue
            self._ml_char_check(c)
            self.pos += 1

    def parse_string_value(mut self, mut scratch: List[UInt8]) raises:
        """Parse any string value; emit VT_STR + payload into out.

        The string is decoded into the reusable scratch buffer (bulk span
        copies inside) and emitted with a single memcpy.
        """
        var c = self.ch()
        scratch.clear()
        if c == CH_QUOTE:
            if self.ch_at(1) == CH_QUOTE and self.ch_at(2) == CH_QUOTE:
                self.parse_ml_basic_to(scratch)
            else:
                self.parse_basic_to(scratch)
        else:
            if self.ch_at(1) == CH_APOS and self.ch_at(2) == CH_APOS:
                self.parse_ml_literal_to(scratch)
            else:
                self.parse_literal_to(scratch)
        self.e_u8(VT_STR)
        self.e_u32(len(scratch))
        self.e_bytes(scratch)

    def parse_key_string(mut self) raises -> String:
        """Parse one quoted simple-key into a String."""
        var tmp = List[UInt8]()
        if self.ch() == CH_QUOTE:
            self.parse_basic_to(tmp)
        else:
            self.parse_literal_to(tmp)
        return String(unsafe_from_utf8=tmp^)

    # ---- keys ------------------------------------------------------------

    def parse_key_segs(mut self) raises -> List[String]:
        """Parse a (possibly dotted) key into its segment names."""
        var segs = List[String]()
        while True:
            self.skip_ws()
            var c = self.ch()
            if c == CH_QUOTE or c == CH_APOS:
                segs.append(self.parse_key_string())
            elif _is_bare_key(c):
                var start = self.pos
                while _is_bare_key(self.ch()):
                    self.pos += 1
                var raw = List[UInt8]()
                self.e_slice_to(raw, start, self.pos)
                segs.append(String(unsafe_from_utf8=raw^))
            else:
                self.fail(self.pos)
            self.skip_ws()
            if self.ch() == CH_DOT:
                self.pos += 1
                continue
            return segs^

    def e_slice_to(mut self, mut dst: List[UInt8], start: Int, end: Int):
        var i = start
        while i < end:
            dst.append(self.src[unsafe_offset=i])
            i += 1

    # ---- numbers / dates ---------------------------------------------------

    def scan_digits(mut self) raises -> Int:
        """Scan one digit group with TOML underscore rules; returns digits."""
        var cnt = 0
        var last_digit = False
        while True:
            var c = self.ch()
            if _is_digit(c):
                self.pos += 1
                cnt += 1
                last_digit = True
            elif c == CH_UNDER:
                if not last_digit or not _is_digit(self.ch_at(1)):
                    self.fail(self.pos)
                self.pos += 1
                last_digit = False
            else:
                break
        return cnt

    def scan_plain_digits(mut self) -> Int:
        """Scan digits with no underscores (time seconds fraction)."""
        var cnt = 0
        while _is_digit(self.ch()):
            self.pos += 1
            cnt += 1
        return cnt

    def fixed(self, off: Int, k: Int) -> Int:
        """Value of exactly k ASCII digits at pos+off, or -1 (no consume)."""
        var v = 0
        for i in range(k):
            var c = self.ch_at(off + i)
            if not _is_digit(c):
                return -1
            v = v * 10 + (c - CH_ZERO)
        return v

    def days_in_month(self, y: Int, m: Int) -> Int:
        if m == 2:
            var leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            return 29 if leap else 28
        if m == 4 or m == 6 or m == 9 or m == 11:
            return 30
        return 31

    def emit_time_fields(mut self) raises:
        """Parse ':'-separated time after the date part or standalone
        (cursor at the 2-digit hour). Emits h/m/s/us into out."""
        var h = self.fixed(0, 2)
        if h < 0 or self.ch_at(2) != CH_COLON:
            self.fail(self.pos)
        var mi = self.fixed(3, 2)
        if mi < 0 or self.ch_at(5) != CH_COLON:
            self.fail(self.pos)
        var s = self.fixed(6, 2)
        if s < 0:
            self.fail(self.pos)
        self.pos += 8
        if h > 23 or mi > 59 or s > 59:
            self.fail(self.pos)
        var us = 0
        if self.ch() == CH_DOT:
            self.pos += 1
            var start = self.pos
            var ndig = self.scan_plain_digits()
            if ndig == 0:
                self.fail(self.pos)
            var scale = 100000
            for i in range(6):
                if i < ndig:
                    us += (
                        Int(self.src[unsafe_offset=start + i]) - CH_ZERO
                    ) * scale
                scale //= 10
        self.e_u8(h)
        self.e_u8(mi)
        self.e_u8(s)
        self.e_u32(us)

    def parse_date_time_or_number(mut self) raises:
        """Parse a value starting with a digit: date/time/datetime or number."""
        # Date shape: exactly 4 digits, '-', 2 digits, '-', 2 digits.
        if self.fixed(0, 4) >= 0 and self.ch_at(4) == CH_MINUS:
            var mo = self.fixed(5, 2)
            var dy = self.fixed(8, 2)
            if mo >= 0 and self.ch_at(7) == CH_MINUS and dy >= 0:
                # Committed to the date shape; validate ranges strictly.
                var y = self.fixed(0, 4)
                if (
                    y < 1
                    or mo < 1
                    or mo > 12
                    or dy < 1
                    or dy > self.days_in_month(y, mo)
                ):
                    self.fail(self.pos)
                self.pos += 10
                var tag_at = len(self.out)
                self.e_u8(VT_DATE)
                self.e_u16(y)
                self.e_u8(mo)
                self.e_u8(dy)
                # Optional time part: 'T'/'t', or one space + HH:MM shape.
                var has_time = False
                var c = self.ch()
                if c == CH_T_UP or c == CH_T:
                    self.pos += 1
                    has_time = True
                elif (
                    c == CH_SPACE
                    and self.fixed(1, 2) >= 0
                    and self.ch_at(3) == CH_COLON
                ):
                    self.pos += 1
                    has_time = True
                if not has_time:
                    return
                # Upgrade the record tag to a datetime.
                self.out[tag_at] = UInt8(VT_DT_LOCAL)
                self.emit_time_fields()
                # Optional UTC offset.
                var o = self.ch()
                if o == CH_Z_UP or o == CH_Z:
                    self.pos += 1
                    self.out[tag_at] = UInt8(VT_DT_OFFSET)
                    self.e_u16(0)
                    return
                if o == CH_PLUS or o == CH_MINUS:
                    var oh = self.fixed(1, 2)
                    if oh < 0 or self.ch_at(3) != CH_COLON:
                        self.fail(self.pos)
                    var om = self.fixed(4, 2)
                    if om < 0 or oh > 23 or om > 59:
                        self.fail(self.pos)
                    var off = oh * 60 + om
                    if o == CH_MINUS:
                        off = -off
                    self.pos += 6
                    self.out[tag_at] = UInt8(VT_DT_OFFSET)
                    self.e_u16(off & 0xFFFF)
                    return
                return
        # Time-only shape: exactly 2 digits + ':'.
        if self.fixed(0, 2) >= 0 and self.ch_at(2) == CH_COLON:
            self.e_u8(VT_TIME)
            self.emit_time_fields()
            return
        self.parse_number()

    def parse_number(mut self) raises:
        """Parse an integer or float (cursor at optional sign or digit)."""
        var start = self.pos
        var c = self.ch()
        var neg_or_sign = False
        if c == CH_PLUS or c == CH_MINUS:
            neg_or_sign = True
            self.pos += 1
            c = self.ch()
        # inf / nan (sign already handled; word itself is unsigned).
        if c == CH_I or c == CH_N:
            if c == CH_I:
                if self.ch_at(1) != CH_N or self.ch_at(2) != CH_F:
                    self.fail(self.pos)
            else:
                if self.ch_at(1) != CH_A or self.ch_at(2) != CH_N:
                    self.fail(self.pos)
            self.pos += 3
            self.e_u8(VT_FLOAT)
            self.e_u32(self.pos - start)
            self.e_src_slice(start, self.pos)
            return
        if not _is_digit(c):
            self.fail(self.pos)
        # Radix-prefixed integers (unsigned only; a consumed sign leaves the
        # '0x..' as trailing garbage which the caller rejects).
        if c == CH_ZERO and not neg_or_sign:
            var nx = self.ch_at(1)
            if nx == CH_X or nx == CH_O or nx == CH_B:
                var tag = VT_INT_HEX
                if nx == CH_O:
                    tag = VT_INT_OCT
                elif nx == CH_B:
                    tag = VT_INT_BIN
                self.pos += 2
                var ndig = 0
                var last_digit = False
                while True:
                    var d = self.ch()
                    var ok = False
                    if tag == VT_INT_HEX:
                        ok = _is_hexdig(d)
                    elif tag == VT_INT_OCT:
                        ok = CH_ZERO <= d <= 0x37
                    else:
                        ok = d == CH_ZERO or d == 0x31
                    if ok:
                        self.pos += 1
                        ndig += 1
                        last_digit = True
                    elif d == CH_UNDER:
                        if not last_digit or not _is_hexdig(self.ch_at(1)):
                            self.fail(self.pos)
                        self.pos += 1
                        last_digit = False
                    else:
                        break
                if ndig == 0:
                    self.fail(self.pos)
                self.e_u8(tag)
                self.e_u32(self.pos - start)
                self.e_src_slice(start, self.pos)
                return
        # Decimal integer part.
        var int_start = self.pos
        var ndig = self.scan_digits()
        if ndig == 0:
            self.fail(self.pos)
        # Leading-zero rule: a zero-padded integer part is invalid.
        if (
            ndig > 1
            and Int(self.src[unsafe_offset=int_start]) == CH_ZERO
        ):
            self.fail(int_start)
        var is_float = False
        if self.ch() == CH_DOT:
            is_float = True
            self.pos += 1
            var f = self.scan_digits()
            if f == 0:
                self.fail(self.pos)
        var e = self.ch()
        if e == CH_E or e == CH_E_UP:
            is_float = True
            self.pos += 1
            var s2 = self.ch()
            if s2 == CH_PLUS or s2 == CH_MINUS:
                self.pos += 1
            var f2 = self.scan_digits()
            if f2 == 0:
                self.fail(self.pos)
        self.e_u8(VT_FLOAT if is_float else VT_INT_DEC)
        self.e_u32(self.pos - start)
        self.e_src_slice(start, self.pos)

    # ---- containers --------------------------------------------------------

    def parse_array(mut self, mut scratch: List[UInt8]) raises:
        """Array value with a nesting-depth guard (see parse_value)."""
        self.depth += 1
        if self.depth > MAX_VALUE_DEPTH:
            self.fail(self.pos)
        self.parse_array_body(scratch)
        self.depth -= 1

    def parse_array_body(mut self, mut scratch: List[UInt8]) raises:
        """Parse an array value (cursor at '['); emit VT_ARRAY payload."""
        self.pos += 1
        self.e_u8(VT_ARRAY)
        var count_at = len(self.out)
        self.e_u32(0)  # backpatched
        var count = 0
        self.skip_trivia()
        if self.ch() == CH_RBRACK:
            self.pos += 1
            return
        while True:
            self.parse_value(scratch)
            count += 1
            self.skip_trivia()
            var c = self.ch()
            if c == CH_COMMA:
                self.pos += 1
                self.skip_trivia()
                if self.ch() == CH_RBRACK:
                    self.pos += 1
                    self.patch_u32(count_at, count)
                    return
                continue
            if c == CH_RBRACK:
                self.pos += 1
                self.patch_u32(count_at, count)
                return
            self.fail(self.pos)

    def parse_inline_table(mut self, mut scratch: List[UInt8]) raises:
        """Inline table value with a nesting-depth guard (see parse_value)."""
        self.depth += 1
        if self.depth > MAX_VALUE_DEPTH:
            self.fail(self.pos)
        self.parse_inline_table_body(scratch)
        self.depth -= 1

    def parse_inline_table_body(mut self, mut scratch: List[UInt8]) raises:
        """Parse an inline table (cursor at '{'); emit VT_INLINE payload.

        Duplicate keys and dotted-key freezing inside an inline table are
        tracked in a fresh local symbol table: an inline table is a closed
        value, so nothing outside its braces may refer into it.
        """
        self.pos += 1
        self.e_u8(VT_INLINE)
        var count_at = len(self.out)
        self.e_u32(0)  # backpatched
        var count = 0
        var local = Dict[String, Int8]()
        self.skip_ws()
        if self.ch() == CH_RBRACE:
            self.pos += 1
            return
        while True:
            var segs = self.parse_key_segs()
            self.skip_ws()
            if self.ch() != CH_EQ:
                self.fail(self.pos)
            self.pos += 1
            self.skip_ws()
            # Validate the key against the local symbol table, then emit the
            # full (possibly dotted) relative key path for this entry.
            var cur = String()
            for i in range(len(segs)):
                var key = cur + _enc_seg(segs[i])
                var st = ST_MISSING
                if key in local:
                    st = local[key]
                if i == len(segs) - 1:
                    if st != ST_MISSING:
                        self.fail(self.pos)
                    local[key] = ST_VALUE
                else:
                    if st == ST_MISSING:
                        local[key] = ST_DOTTED
                    elif st != ST_DOTTED:
                        self.fail(self.pos)
                    cur = key
            self.e_u16(len(segs))
            for i in range(len(segs)):
                self.e_u32(segs[i].byte_length())
                self.e_str(segs[i])
            self.parse_value(scratch)
            count += 1
            self.skip_ws()
            var c = self.ch()
            if c == CH_COMMA:
                self.pos += 1
                self.skip_ws()
                continue
            if c == CH_RBRACE:
                self.pos += 1
                self.patch_u32(count_at, count)
                return
            self.fail(self.pos)

    # ---- values --------------------------------------------------------------

    def parse_value(mut self, mut scratch: List[UInt8]) raises:
        """Parse any value; append its tagged payload to out."""
        var c = self.ch()
        if c == CH_QUOTE or c == CH_APOS:
            self.parse_string_value(scratch)
        elif c == CH_LBRACK:
            self.parse_array(scratch)
        elif c == CH_LBRACE:
            self.parse_inline_table(scratch)
        elif c == CH_T:  # true
            if (
                self.ch_at(1) != CH_R
                or self.ch_at(2) != CH_U
                or self.ch_at(3) != CH_E
            ):
                self.fail(self.pos)
            self.pos += 4
            self.e_u8(VT_BOOL)
            self.e_u8(1)
        elif c == CH_F:  # false
            if (
                self.ch_at(1) != CH_A
                or self.ch_at(2) != CH_L
                or self.ch_at(3) != 0x73
                or self.ch_at(4) != CH_E
            ):
                self.fail(self.pos)
            self.pos += 5
            self.e_u8(VT_BOOL)
            self.e_u8(0)
        elif c == CH_I or c == CH_N or c == CH_PLUS or c == CH_MINUS:
            self.parse_number()
        elif _is_digit(c):
            self.parse_date_time_or_number()
        else:
            self.fail(self.pos)

    # ---- key/value pairs ------------------------------------------------------

    def parse_keyval(mut self, mut scratch: List[UInt8]) raises:
        """Parse `key = value` in the current context; emit a VALUE record."""
        var segs = self.parse_key_segs()
        self.skip_ws()
        if self.ch() != CH_EQ:
            self.fail(self.pos)
        self.pos += 1
        self.skip_ws()
        # Validate the key path against the state trie, walking from the
        # current context node.
        var node = self.ctx_node
        for i in range(len(segs)):
            var child = -1
            if segs[i] in self.nodes[node].children:
                child = self.nodes[node].children[segs[i]]
            if i == len(segs) - 1:
                if child >= 0:
                    self.fail(self.pos)
                var niv = self._new_node(ST_VALUE)
                self.nodes[node].children[segs[i]] = niv
            else:
                if child < 0:
                    var ni = self._new_node(ST_DOTTED)
                    self.nodes[node].children[segs[i]] = ni
                    node = ni
                else:
                    var st = self.nodes[child].state
                    if st == ST_DOTTED or st == ST_IMPLICIT:
                        node = child
                    else:
                        # DEFINED (redefine namespace), VALUE (overwrite), AOT.
                        self.fail(self.pos)
        # Emit the record path: pre-encoded context segments (copied out of
        # self to satisfy the aliasing rules) + key segments.
        self.e_u8(REC_VALUE)
        self.e_u16(self.ctx_nsegs + len(segs))
        var prefix = List[UInt8](copy=self.ctx_emit)
        self.e_bytes(prefix)
        for i in range(len(segs)):
            self.e_seg_parts(segs[i], -1)
        self.parse_value(scratch)

    # ---- table headers ---------------------------------------------------------

    def parse_header(mut self) raises:
        """Parse a [table] or [[array-of-tables]] header."""
        var is_aot = False
        self.pos += 1  # '['
        if self.ch() == CH_LBRACK:
            is_aot = True
            self.pos += 1
        self.skip_ws()
        var segs = self.parse_key_segs()
        self.skip_ws()
        if self.ch() != CH_RBRACK:
            self.fail(self.pos)
        self.pos += 1
        if is_aot:
            if self.ch() != CH_RBRACK:
                self.fail(self.pos)
            self.pos += 1
        # Walk/create the path from the root of the state trie.
        var node = 0
        var orig = List[Seg]()
        for i in range(len(segs)):
            var last = i == len(segs) - 1
            var child = -1
            if segs[i] in self.nodes[node].children:
                child = self.nodes[node].children[segs[i]]
            if not last:
                if child < 0:
                    var ni = self._new_node(ST_IMPLICIT)
                    self.nodes[node].children[segs[i]] = ni
                    node = ni
                    orig.append(Seg(segs[i], -1))
                else:
                    var st = self.nodes[child].state
                    if st == ST_IMPLICIT or st == ST_DEFINED or st == ST_DOTTED:
                        node = child
                        orig.append(Seg(segs[i], -1))
                    elif st == ST_AOT:
                        # Descend into the array's current (last) element;
                        # an AoT node's children are scoped to it.
                        node = child
                        orig.append(
                            Seg(segs[i], self.nodes[child].aot_count - 1)
                        )
                    else:  # VALUE
                        self.fail(self.pos)
                continue
            # Final segment.
            if not is_aot:
                if child < 0:
                    child = self._new_node(ST_DEFINED)
                    self.nodes[node].children[segs[i]] = child
                elif self.nodes[child].state == ST_IMPLICIT:
                    self.nodes[child].state = ST_DEFINED
                else:
                    # DEFINED/DOTTED (declared twice), VALUE, AOT.
                    self.fail(self.pos)
                orig.append(Seg(segs[i], -1))
                self.e_u8(REC_TABLE)
                self.e_path(orig)
                self.ctx_node = child
                var emit = List[UInt8]()
                for sg in orig:
                    _encode_seg_into(emit, sg.name, sg.aot_index)
                self.ctx_emit = emit^
                self.ctx_nsegs = len(orig)
            else:
                if child < 0:
                    child = self._new_node(ST_AOT)
                    self.nodes[node].children[segs[i]] = child
                elif self.nodes[child].state != ST_AOT:
                    self.fail(self.pos)
                var idx = self.nodes[child].aot_count
                self.nodes[child].aot_count = idx + 1
                # A new element starts with an empty scope: an AoT node's
                # children map is per-element (always the last one).
                self.nodes[child].children.clear()
                self.e_u8(REC_AOT_ELEM)
                orig.append(Seg(segs[i], -1))
                self.e_path(orig)
                self.e_u32(idx)
                orig[len(orig) - 1] = Seg(segs[i], idx)
                self.ctx_node = child
                var emit2 = List[UInt8]()
                for sg in orig:
                    _encode_seg_into(emit2, sg.name, sg.aot_index)
                self.ctx_emit = emit2^
                self.ctx_nsegs = len(orig)

    # ---- top level ---------------------------------------------------------

    def run(mut self) raises:
        """Parse the whole document into the output stream."""
        # The record stream is typically somewhat larger than the source
        # (typed values + paths); reserve once to avoid geometric regrowth.
        self.out.reserve(self.n + self.n // 2 + 64)
        # Stream magic.
        self.out.append(0x54)  # T
        self.out.append(0x4D)  # M
        self.out.append(0x4F)  # O
        self.out.append(0x31)  # 1
        var scratch = List[UInt8]()
        self.skip_trivia()
        while not self.at_end():
            if self.ch() == CH_LBRACK:
                self.parse_header()
            else:
                self.parse_keyval(scratch)
            self.expect_stmt_end()
            self.skip_trivia()


# ---- C ABI ---------------------------------------------------------------------


@export
def tomlmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def tomlmojo_parse(data: U8Ptr, length: Int64) abi("C") -> Handle:
    """Parse `length` bytes of TOML; returns an owned ParseResult handle."""
    var res = unsafe_alloc[ParseResult](1)
    var n = Int(length)
    if n < 0:
        n = 0
    var parser = Parser(data, n)
    try:
        parser.run()
    except:
        # One-byte dummy allocation so `data` is always freeable; size 0
        # signals "no stream".
        var dummy = unsafe_alloc[UInt8](1)
        res[] = ParseResult(dummy, 0, 1, Int64(parser.err))
        return res.unsafe_bitcast[UInt8]()
    var size = Int64(len(parser.out))
    var buf = unsafe_alloc[UInt8](Int(size))
    unsafe_memcpy(dest=buf, src=parser.out.unsafe_ptr(), count=Int(size))
    res[] = ParseResult(buf, size, 0, -1)
    return res.unsafe_bitcast[UInt8]()


@export
def tomlmojo_result_status(handle: Handle) abi("C") -> Int32:
    if not handle:
        return 2
    return handle.value().unsafe_bitcast[ParseResult]()[].status


@export
def tomlmojo_result_error_pos(handle: Handle) abi("C") -> Int64:
    if not handle:
        return -1
    return handle.value().unsafe_bitcast[ParseResult]()[].err_pos


@export
def tomlmojo_result_size(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    return handle.value().unsafe_bitcast[ParseResult]()[].size


@export
def tomlmojo_result_data(handle: Handle) abi("C") -> Handle:
    if not handle:
        return None
    return handle.value().unsafe_bitcast[ParseResult]()[].data


@export
def tomlmojo_result_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var res = handle.value().unsafe_bitcast[ParseResult]()
    # `data` is always an owned allocation (a 1-byte dummy on parse errors).
    res[].data.unsafe_free()
    res.unsafe_free()
