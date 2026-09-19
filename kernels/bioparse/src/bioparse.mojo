"""Clean-room FASTA / GenBank record scanner.

Written fresh from the FASTA and GenBank flat-file format conventions (NCBI
feature-table layout) and pinned by black-box differential tests against the
published `biopython` package (the test oracle). No third-party Mojo code is
used or adapted, and no biopython source is read or copied.

The kernel is a byte-level, single-pass line scanner. It never decodes text
and never builds per-record objects; it copies the byte ranges the Python
wrapper needs into growable arenas and returns concat-style offset arrays
(one offsets array per field category, n+1 entries, into one arena per
category). All Unicode handling, id/description splitting, location-string
normalization and qualifier unescaping lives in the Python wrapper, shared
by the native and pure-Python backends.

Byte-level semantics implemented here (all pinned against the oracle):

FASTA
  * The first line must start with '>', otherwise the parse fails with the
    "preamble" error (the oracle raises ValueError for any leading content,
    including blank lines; a completely empty file yields zero records).
  * A record's title is the raw bytes after '>' up to the line terminator
    (no stripping here; the wrapper rstrips after decoding, matching the
    oracle's str-level rstrip, including non-ASCII whitespace).
  * Sequence lines contribute their bytes with trailing ASCII whitespace
    removed (bytes.rstrip set) and then every remaining 0x20/0x09 byte
    dropped. Non-ASCII in the assembled sequence is rejected by the wrapper
    via bytes.decode("ascii"), reproducing the oracle's UnicodeDecodeError.
  * Lines are split on '\n', '\r\n' and lone '\r' (universal newlines).

GenBank (subset: LOCUS/DEFINITION/ACCESSION/VERSION/FEATURES/ORIGIN)
  * Records run from a column-0 'LOCUS' line to a column-0 '//' line; the
    final record may end at EOF. Missing ORIGIN is an error. A new LOCUS
    line before '//' is an error. Junk lines between records are skipped.
  * LOCUS name = first whitespace token after the 'LOCUS' keyword.
  * DEFINITION content starts at column 12; continuation lines (indented,
    column 12) are joined with a single space. ACCESSION/VERSION keep only
    their first line's column-12 content (the wrapper takes the first token
    and applies the id rule: VERSION else ACCESSION else LOCUS name).
  * FEATURES uses the standard 5/21-column layout: a feature key starts with
    a non-space byte in column 5, its location starts at column 21 and may
    continue on >=21-space-indented lines (concatenated verbatim); a content
    line starting with '/' opens a qualifier; while a quoted qualifier value
    is still open, content lines are value continuations joined with ' '
    (empty separator for /translation). A feature whose location is empty is
    dropped (with its qualifiers), matching the oracle.
  * ORIGIN lines must carry a numeric coordinate prefix; sequence bytes are
    uppercased with digits and whitespace removed.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

# Error codes returned through out_err (0 = success).
comptime ERR_NONE: Int32 = 0
comptime ERR_BAD_ARGS: Int32 = 2
comptime ERR_FASTA_PREAMBLE: Int32 = 3
comptime ERR_GB_NO_ORIGIN: Int32 = 4
comptime ERR_GB_NO_TERMINATOR: Int32 = 5
comptime ERR_GB_BAD_SEQ_LINE: Int32 = 6
comptime ERR_GB_MISPLACED_FEATURE_LINE: Int32 = 7

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U8PtrPtr = Pointer[U8Ptr, MutUntrackedOrigin]
comptime I64PtrPtr = Pointer[I64Ptr, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

comptime NL: UInt8 = 10
comptime CR: UInt8 = 13
comptime SPACE: UInt8 = 32
comptime TAB: UInt8 = 9
comptime GT: UInt8 = 62  # '>'
comptime QUOTE: UInt8 = 34  # '"'
comptime SLASH: UInt8 = 47  # '/'
comptime TRANSLATION = "translation"
comptime EQ: UInt8 = 61  # '='


struct ByteBuf(Copyable, Movable):
    """Growable byte arena. Contents are stable only until the next grow."""

    var ptr: U8Ptr
    var len: Int64
    var cap: Int64

    def __init__(out self):
        self.cap = 4096
        self.ptr = unsafe_alloc[UInt8](Int(self.cap))
        self.len = 0

    def ensure(mut self, extra: Int64):
        if self.len + extra <= self.cap:
            return
        var ncap = self.cap
        while ncap < self.len + extra:
            ncap = ncap * 2
        var np = unsafe_alloc[UInt8](Int(ncap))
        unsafe_memcpy(dest=np, src=self.ptr, count=Int(self.len))
        self.ptr.unsafe_free()
        self.ptr = np
        self.cap = ncap

    def append_byte(mut self, b: UInt8):
        self.ensure(1)
        self.ptr[unsafe_offset=Int(self.len)] = b
        self.len += 1

    def append(mut self, src: U8Ptr, n: Int64):
        if n <= 0:
            return
        self.ensure(n)
        unsafe_memcpy(dest=self.ptr.unsafe_offset(Int(self.len)), src=src, count=Int(n))
        self.len += n

    def free(mut self):
        self.ptr.unsafe_free()
        self.len = 0
        self.cap = 0


struct I64Vec(Copyable, Movable):
    """Growable int64 vector used for concat offsets."""

    var ptr: I64Ptr
    var len: Int64
    var cap: Int64

    def __init__(out self):
        self.cap = 256
        self.ptr = unsafe_alloc[Int64](Int(self.cap))
        self.len = 0

    def push(mut self, v: Int64):
        if self.len >= self.cap:
            var ncap = self.cap * 2
            var np = unsafe_alloc[Int64](Int(ncap))
            unsafe_memcpy(dest=np, src=self.ptr, count=Int(self.len))
            self.ptr.unsafe_free()
            self.ptr = np
            self.cap = ncap
        self.ptr[unsafe_offset=Int(self.len)] = v
        self.len += 1

    def free(mut self):
        self.ptr.unsafe_free()
        self.len = 0
        self.cap = 0


def _is_ws(b: UInt8) -> Bool:
    """Python bytes.rstrip() whitespace set."""
    return (
        b == SPACE
        or b == TAB
        or b == NL
        or b == CR
        or b == 11  # \x0b
        or b == 12  # \x0c
    )


def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


def _find_eol(data: U8Ptr, start: Int, length: Int) -> Int:
    """Index of the next NL or CR at/after `start`; `length` if none."""
    var i = start
    comptime W = simd_width_of[DType.uint8]()
    while i + W <= length:
        var chunk = data.unsafe_load[width=W](i)
        var x_nl = chunk ^ SIMD[DType.uint8, W](NL)
        var x_cr = chunk ^ SIMD[DType.uint8, W](CR)
        if x_nl.reduce_min() == 0 or x_cr.reduce_min() == 0:
            for lane in range(W):
                var b = data[unsafe_offset=i + lane]
                if b == NL or b == CR:
                    return i + lane
        i += W
    while i < length:
        var b = data[unsafe_offset=i]
        if b == NL or b == CR:
            return i
        i += 1
    return length


def _past_eol(data: U8Ptr, eol: Int, length: Int) -> Int:
    """First index after the line terminator at `eol` (handles CRLF)."""
    if eol >= length:
        return length
    if (
        data[unsafe_offset=eol] == CR
        and eol + 1 < length
        and data[unsafe_offset=eol + 1] == NL
    ):
        return eol + 2
    return eol + 1


def _rstrip_end(data: U8Ptr, s: Int, e: Int) -> Int:
    """End index after removing trailing ASCII whitespace (bytes.rstrip)."""
    var end = e
    while end > s and _is_ws(data[unsafe_offset=end - 1]):
        end -= 1
    return end


def _slice_all_space(data: U8Ptr, s: Int, e: Int) -> Bool:
    for i in range(s, e):
        if data[unsafe_offset=i] != SPACE:
            return False
    return True


# --------------------------------------------------------------------------
# FASTA
# --------------------------------------------------------------------------


struct FastaResult(Copyable, Movable):
    var titles: ByteBuf
    var seqs: ByteBuf
    var title_offs: I64Vec
    var seq_offs: I64Vec
    var n: Int64

    def __init__(out self):
        self.titles = ByteBuf()
        self.seqs = ByteBuf()
        self.title_offs = I64Vec()
        self.seq_offs = I64Vec()
        self.n = 0

    def free(mut self):
        self.titles.free()
        self.seqs.free()
        self.title_offs.free()
        self.seq_offs.free()


def _fasta_parse(
    data: U8Ptr, length: Int64, mut err: Int32, mut err_line: Int64
) -> FastaResult:
    var res = FastaResult()
    err = ERR_NONE
    err_line = 0
    var n = Int(length)
    var pos = 0
    var lineno = Int64(0)
    var in_record = False
    while pos < n:
        var eol = _find_eol(data, pos, n)
        var s = pos
        var e = eol
        pos = _past_eol(data, eol, n)
        lineno += 1
        if e > s and data[unsafe_offset=s] == GT:
            # New record: title is the raw bytes after '>'.
            res.title_offs.push(res.titles.len)
            if e > s + 1:
                res.titles.append(data.unsafe_offset(s + 1), Int64(e - s - 1))
            res.seq_offs.push(res.seqs.len)
            in_record = True
        else:
            if not in_record:
                # Any content before the first '>' line (including blank
                # lines) is a hard error, matching the oracle.
                err = ERR_FASTA_PREAMBLE
                err_line = lineno
                return res^
            # Sequence line: rstrip, then drop every space/tab byte.
            e = _rstrip_end(data, s, e)
            var i = s
            while i < e:
                var b = data[unsafe_offset=i]
                if b != SPACE and b != TAB:
                    res.seqs.append_byte(b)
                i += 1
    res.title_offs.push(res.titles.len)
    res.seq_offs.push(res.seqs.len)
    res.n = res.seq_offs.len - 1
    return res^


# --------------------------------------------------------------------------
# GenBank
# --------------------------------------------------------------------------

comptime ST_BETWEEN: Int32 = 0  # between records (junk skipped)
comptime ST_HEAD: Int32 = 1  # record header sections
comptime ST_FEATURES: Int32 = 2
comptime ST_ORIGIN: Int32 = 3

comptime SEC_OTHER: Int32 = 0
comptime SEC_DEF: Int32 = 1


struct GbResult(Copyable, Movable):
    # Per-record fields (one offsets entry per record per category).
    var names: ByteBuf
    var defs: ByteBuf
    var accs: ByteBuf
    var vers: ByteBuf
    var seqs: ByteBuf
    var name_offs: I64Vec
    var def_offs: I64Vec
    var acc_offs: I64Vec
    var ver_offs: I64Vec
    var seq_offs: I64Vec
    var rec_feat_offs: I64Vec  # n_records + 1 entries into feature arrays
    # Per-feature fields.
    var keys: ByteBuf
    var locs: ByteBuf
    var key_offs: I64Vec
    var loc_offs: I64Vec
    var feat_qual_offs: I64Vec  # n_features + 1 entries into qualifier arrays
    # Per-qualifier fields.
    var qnames: ByteBuf
    var qvals: ByteBuf
    var qquoted: ByteBuf  # 1 flag byte per qualifier: bit0 quoted, bit1 has '='
    var qname_offs: I64Vec
    var qval_offs: I64Vec
    var n_rec: Int64
    var n_feat: Int64
    var n_qual: Int64

    def __init__(out self):
        self.names = ByteBuf()
        self.defs = ByteBuf()
        self.accs = ByteBuf()
        self.vers = ByteBuf()
        self.seqs = ByteBuf()
        self.name_offs = I64Vec()
        self.def_offs = I64Vec()
        self.acc_offs = I64Vec()
        self.ver_offs = I64Vec()
        self.seq_offs = I64Vec()
        self.rec_feat_offs = I64Vec()
        self.keys = ByteBuf()
        self.locs = ByteBuf()
        self.key_offs = I64Vec()
        self.loc_offs = I64Vec()
        self.feat_qual_offs = I64Vec()
        self.qnames = ByteBuf()
        self.qvals = ByteBuf()
        self.qquoted = ByteBuf()
        self.qname_offs = I64Vec()
        self.qval_offs = I64Vec()
        self.n_rec = 0
        self.n_feat = 0
        self.n_qual = 0

    def free(mut self):
        self.names.free()
        self.defs.free()
        self.accs.free()
        self.vers.free()
        self.seqs.free()
        self.name_offs.free()
        self.def_offs.free()
        self.acc_offs.free()
        self.ver_offs.free()
        self.seq_offs.free()
        self.rec_feat_offs.free()
        self.keys.free()
        self.locs.free()
        self.key_offs.free()
        self.loc_offs.free()
        self.feat_qual_offs.free()
        self.qnames.free()
        self.qvals.free()
        self.qquoted.free()
        self.qname_offs.free()
        self.qval_offs.free()


def _kw[word: StaticString](line: U8Ptr, s: Int, e: Int) -> Bool:
    """True if the line starts with `word` followed by space or EOL."""
    comptime k = word.byte_length()
    if e - s < k:
        return False
    for i in range(k):
        if line[unsafe_offset=s + i] != word.as_bytes()[i]:
            return False
    if e - s == k:
        return True
    var b = line[unsafe_offset=s + k]
    return b == SPACE or b == TAB


def _is_col0(line: U8Ptr, s: Int, e: Int) -> Bool:
    return e > s and line[unsafe_offset=s] != SPACE and line[unsafe_offset=s] != TAB


struct _GbState(Copyable, Movable):
    """Mutable per-parse state so the helpers stay small."""

    var res: GbResult
    var state: Int32
    var section: Int32
    var wrote_def: Bool
    var wrote_acc: Bool
    var wrote_ver: Bool
    var wrote_seq: Bool
    # Open feature accumulators.
    var feat_open: Bool
    var cur_key: ByteBuf
    var cur_loc: ByteBuf
    # Rollback marks for a dropped feature (empty location).
    var mark_qname: Int64
    var mark_qval: Int64
    var mark_qquoted: Int64
    var mark_nqual: Int64
    # Open qualifier state.
    var qual_open: Bool
    var in_quote: Bool

    def __init__(out self):
        self.res = GbResult()
        self.state = ST_BETWEEN
        self.section = SEC_OTHER
        self.wrote_def = False
        self.wrote_acc = False
        self.wrote_ver = False
        self.wrote_seq = False
        self.feat_open = False
        self.cur_key = ByteBuf()
        self.cur_loc = ByteBuf()
        self.mark_qname = 0
        self.mark_qval = 0
        self.mark_qquoted = 0
        self.mark_nqual = 0
        self.qual_open = False
        self.in_quote = False

    def free(mut self):
        self.res.free()
        self.cur_key.free()
        self.cur_loc.free()


def _scan_quote(data: U8Ptr, s: Int, e: Int, mut in_quote: Bool):
    """Advance the quote state across a raw value segment.

    A doubled quote '""' is an escaped literal and does not toggle the state.
    """
    var i = s
    while i < e:
        if data[unsafe_offset=i] == QUOTE:
            if i + 1 < e and data[unsafe_offset=i + 1] == QUOTE:
                i += 2
                continue
            in_quote = not in_quote
        i += 1


def _gb_close_qualifier(mut st: _GbState):
    st.qual_open = False
    st.in_quote = False


def _gb_close_feature(mut st: _GbState):
    """Flush the open feature, or roll it back when its location is empty.

    The oracle drops features whose location string is empty, qualifiers
    included; we restore the qualifier buffers to the marks taken when the
    feature opened.
    """
    if not st.feat_open:
        return
    _gb_close_qualifier(st)
    st.feat_open = False
    if st.cur_loc.len == 0 or _slice_all_space(
        st.cur_loc.ptr, 0, Int(st.cur_loc.len)
    ):
        st.res.qnames.len = st.mark_qname
        st.res.qvals.len = st.mark_qval
        st.res.qquoted.len = st.mark_qquoted
        st.res.n_qual = st.mark_nqual
        st.cur_key.len = 0
        st.cur_loc.len = 0
        return
    st.res.key_offs.push(st.res.keys.len)
    st.res.keys.append(st.cur_key.ptr, st.cur_key.len)
    st.res.loc_offs.push(st.res.locs.len)
    st.res.locs.append(st.cur_loc.ptr, st.cur_loc.len)
    st.res.feat_qual_offs.push(st.res.n_qual)
    st.res.n_feat += 1
    st.cur_key.len = 0
    st.cur_loc.len = 0


def _gb_close_record(mut st: _GbState, mut err: Int32, mut err_line: Int64, lineno: Int64) -> Bool:
    """Close the current record. Returns False (with err set) on failure."""
    _gb_close_feature(st)
    if not st.wrote_seq:
        err = ERR_GB_NO_ORIGIN
        err_line = lineno
        return False
    if not st.wrote_def:
        st.res.def_offs.push(st.res.defs.len)
    if not st.wrote_acc:
        st.res.acc_offs.push(st.res.accs.len)
    if not st.wrote_ver:
        st.res.ver_offs.push(st.res.vers.len)
    st.res.rec_feat_offs.push(st.res.n_feat)
    st.res.n_rec += 1
    st.wrote_def = False
    st.wrote_acc = False
    st.wrote_ver = False
    st.wrote_seq = False
    st.section = SEC_OTHER
    return True


def _gb_start_locus(mut st: _GbState, data: U8Ptr, s: Int, e: Int):
    # Name = first whitespace token after the 'LOCUS' keyword.
    var i = s + 5
    while i < e and (data[unsafe_offset=i] == SPACE or data[unsafe_offset=i] == TAB):
        i += 1
    var t0 = i
    while i < e and data[unsafe_offset=i] != SPACE and data[unsafe_offset=i] != TAB:
        i += 1
    st.res.name_offs.push(st.res.names.len)
    st.res.names.append(data.unsafe_offset(t0), Int64(i - t0))
    st.state = ST_HEAD
    st.section = SEC_OTHER


def _gb_col12(data: U8Ptr, s: Int, e: Int) -> Tuple[Int, Int]:
    """Column-12 content of a header line, rstripped (may be empty)."""
    var c0 = s + 12
    if c0 > e:
        c0 = e
    var c1 = _rstrip_end(data, c0, e)
    return Tuple(c0, c1)


def _gb_finish(mut st: _GbState) -> GbResult:
    """Release the per-parse scratch buffers and hand back the result."""
    st.cur_key.free()
    st.cur_loc.free()
    # Shallow copy out of the state (no destructors anywhere; ownership of
    # the arenas moves to the caller, which frees them via *_free).
    return st.res.copy()


def _gb_parse(
    data: U8Ptr, length: Int64, mut err: Int32, mut err_line: Int64
) -> GbResult:
    var st = _GbState()
    err = ERR_NONE
    err_line = 0
    st.res.rec_feat_offs.push(0)
    st.res.feat_qual_offs.push(0)
    var n = Int(length)
    var pos = 0
    var lineno = Int64(0)
    while pos < n:
        var eol = _find_eol(data, pos, n)
        var s = pos
        var e = eol
        pos = _past_eol(data, eol, n)
        lineno += 1
        if e <= s:
            continue  # blank line: skipped in every section

        # Column-0 lines are section boundaries, EXCEPT inside ORIGIN where
        # every line that is not '//' or 'LOCUS' is sequence data (the
        # oracle errors on malformed sequence lines there).
        var col0_boundary = False
        if _is_col0(data, s, e):
            # ---- column-0 keyword line: section boundary ----
            if _kw["//"](data, s, e):
                col0_boundary = True
                if st.state == ST_BETWEEN:
                    continue  # stray terminator between records: junk
                if not _gb_close_record(st, err, err_line, lineno):
                    return _gb_finish(st)
                st.state = ST_BETWEEN
            elif _kw["LOCUS"](data, s, e):
                col0_boundary = True
                if st.state != ST_BETWEEN:
                    err = ERR_GB_NO_TERMINATOR
                    err_line = lineno
                    return _gb_finish(st)
                _gb_start_locus(st, data, s, e)
            elif st.state == ST_BETWEEN:
                continue  # junk between records
            elif _kw["DEFINITION"](data, s, e) and st.state == ST_HEAD and not st.wrote_def:
                col0_boundary = True
                var c = _gb_col12(data, s, e)
                st.res.def_offs.push(st.res.defs.len)
                st.res.defs.append(data.unsafe_offset(c[0]), Int64(c[1] - c[0]))
                st.wrote_def = True
                st.section = SEC_DEF
            elif _kw["ACCESSION"](data, s, e) and st.state == ST_HEAD and not st.wrote_acc:
                col0_boundary = True
                var c = _gb_col12(data, s, e)
                st.res.acc_offs.push(st.res.accs.len)
                st.res.accs.append(data.unsafe_offset(c[0]), Int64(c[1] - c[0]))
                st.wrote_acc = True
                st.section = SEC_OTHER
            elif _kw["VERSION"](data, s, e) and st.state == ST_HEAD and not st.wrote_ver:
                col0_boundary = True
                var c = _gb_col12(data, s, e)
                st.res.ver_offs.push(st.res.vers.len)
                st.res.vers.append(data.unsafe_offset(c[0]), Int64(c[1] - c[0]))
                st.wrote_ver = True
                st.section = SEC_OTHER
            elif _kw["FEATURES"](data, s, e) and st.state == ST_HEAD:
                col0_boundary = True
                st.state = ST_FEATURES
                st.section = SEC_OTHER
            elif _kw["ORIGIN"](data, s, e) and (
                st.state == ST_HEAD or st.state == ST_FEATURES
            ):
                col0_boundary = True
                _gb_close_feature(st)
                st.state = ST_ORIGIN
                st.wrote_seq = True
                st.res.seq_offs.push(st.res.seqs.len)
            elif st.state != ST_ORIGIN:
                # CONTIG / KEYWORDS / SOURCE / REFERENCE / COMMENT / ... :
                # sections outside the supported subset are skipped.
                col0_boundary = True
                if st.state == ST_FEATURES:
                    _gb_close_feature(st)
                st.state = ST_HEAD
                st.section = SEC_OTHER
        if col0_boundary:
            continue

        # ---- indented line (or in-ORIGIN col0 data line): dispatch ----
        if st.state == ST_HEAD:
            if st.section == SEC_DEF and st.wrote_def:
                var c = _gb_col12(data, s, e)
                st.res.defs.append_byte(SPACE)
                st.res.defs.append(data.unsafe_offset(c[0]), Int64(c[1] - c[0]))
            # other header continuations are ignored
        elif st.state == ST_FEATURES:
            var is_feat = (
                e - s > 5
                and data[unsafe_offset=s] == SPACE
                and _slice_all_space(data, s, s + 5)
                and data[unsafe_offset=s + 5] != SPACE
                and data[unsafe_offset=s + 5] != TAB
            )
            if is_feat:
                _gb_close_feature(st)
                st.feat_open = True
                st.mark_qname = st.res.qnames.len
                st.mark_qval = st.res.qvals.len
                st.mark_qquoted = st.res.qquoted.len
                st.mark_nqual = st.res.n_qual
                # Key = token starting at column 5 (ends at first space or
                # column 21); location = column 21 onward, rstripped.
                var kend = s + 21
                if kend > e:
                    kend = e
                var i = s + 5
                while i < kend and data[unsafe_offset=i] != SPACE:
                    i += 1
                st.cur_key.append(data.unsafe_offset(s + 5), Int64(i - (s + 5)))
                if e > s + 21:
                    var l1 = _rstrip_end(data, s + 21, e)
                    st.cur_loc.append(data.unsafe_offset(s + 21), Int64(l1 - (s + 21)))
            elif e - s >= 21 and _slice_all_space(data, s, s + 21):
                var c1 = _rstrip_end(data, s + 21, e)
                var c0 = s + 21
                if not st.feat_open:
                    continue  # content before any feature line: skipped
                if st.in_quote and st.qual_open:
                    # Quoted value continuation: ' ' separator, except for
                    # /translation which concatenates directly (oracle rule).
                    var qstart = Int(st.res.qname_offs.ptr[unsafe_offset=Int(st.res.n_qual - 1)])
                    var qlen = Int(st.res.qnames.len) - qstart
                    var is_translation = qlen == 11
                    if is_translation:
                        for i in range(11):
                            if st.res.qnames.ptr[unsafe_offset=qstart + i] != TRANSLATION.as_bytes()[i]:
                                is_translation = False
                    if not is_translation:
                        st.res.qvals.append_byte(SPACE)
                    st.res.qvals.append(data.unsafe_offset(c0), Int64(c1 - c0))
                    _scan_quote(data, c0, c1, st.in_quote)
                elif c1 > c0 and data[unsafe_offset=c0] == SLASH:
                    # New qualifier.
                    var eq = c0 + 1
                    while eq < c1 and data[unsafe_offset=eq] != EQ:
                        eq += 1
                    st.res.qname_offs.push(st.res.qnames.len)
                    st.res.qnames.append(data.unsafe_offset(c0 + 1), Int64(eq - (c0 + 1)))
                    st.res.qval_offs.push(st.res.qvals.len)
                    var flags = UInt8(0)
                    if eq < c1:
                        flags = 2  # bit1: '=' present (value-bearing)
                        var v0 = eq + 1
                        if v0 < c1 and data[unsafe_offset=v0] == QUOTE:
                            flags = flags | 1  # bit0: quoted value
                            st.in_quote = True
                            _scan_quote(data, v0 + 1, c1, st.in_quote)
                        st.res.qvals.append(data.unsafe_offset(v0), Int64(c1 - v0))
                    st.res.qquoted.append_byte(flags)
                    st.res.n_qual += 1
                    st.qual_open = True
                else:
                    # Location continuation: concatenated verbatim, but only
                    # while the location's parentheses are unbalanced (the
                    # oracle crashes on continuations of a complete
                    # location; we fail explicitly instead).
                    var balance = Int64(0)
                    for i in range(Int(st.cur_loc.len)):
                        var b = st.cur_loc.ptr[unsafe_offset=i]
                        if b == 40:  # '('
                            balance += 1
                        elif b == 41:  # ')'
                            balance -= 1
                    if balance <= 0:
                        err = ERR_GB_MISPLACED_FEATURE_LINE
                        err_line = lineno
                        return _gb_finish(st)
                    st.cur_loc.append(data.unsafe_offset(c0), Int64(c1 - c0))
            # anything else inside FEATURES (e.g. the Location/Qualifiers
            # header) is skipped
        elif st.state == ST_ORIGIN:
            # Sequence line: optional leading spaces, numeric coordinate,
            # then the sequence; uppercase, drop digits and whitespace.
            var i = s
            while i < e and (data[unsafe_offset=i] == SPACE or data[unsafe_offset=i] == TAB):
                i += 1
            if i < e:
                if not _is_digit(data[unsafe_offset=i]):
                    err = ERR_GB_BAD_SEQ_LINE
                    err_line = lineno
                    return _gb_finish(st)
                while i < e and data[unsafe_offset=i] != SPACE and data[unsafe_offset=i] != TAB:
                    i += 1
                if i >= e:
                    err = ERR_GB_BAD_SEQ_LINE
                    err_line = lineno
                    return _gb_finish(st)
                i += 1
                while i < e:
                    var b = data[unsafe_offset=i]
                    if _is_digit(b) or _is_ws(b):
                        i += 1
                        continue
                    if b >= 97 and b <= 122:
                        b = b - 32
                    st.res.seqs.append_byte(b)
                    i += 1

    # EOF: an unterminated final record is accepted iff it has its ORIGIN.
    if st.state != ST_BETWEEN:
        if not _gb_close_record(st, err, err_line, lineno):
            return _gb_finish(st)
        st.state = ST_BETWEEN

    # Final sentinels for every concat category.
    st.res.name_offs.push(st.res.names.len)
    st.res.def_offs.push(st.res.defs.len)
    st.res.acc_offs.push(st.res.accs.len)
    st.res.ver_offs.push(st.res.vers.len)
    st.res.seq_offs.push(st.res.seqs.len)
    st.res.key_offs.push(st.res.keys.len)
    st.res.loc_offs.push(st.res.locs.len)
    st.res.qname_offs.push(st.res.qnames.len)
    st.res.qval_offs.push(st.res.qvals.len)
    return _gb_finish(st)


# --------------------------------------------------------------------------
# Exported C ABI
# --------------------------------------------------------------------------


@export
def bioparse_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def bioparse_fasta_parse(
    data: U8Ptr, length: Int64, out_err: I32Ptr, out_err_line: I64Ptr
) abi("C") -> Handle:
    """Parse a whole FASTA buffer. NULL on error (code in out_err)."""
    if length < 0:
        out_err[] = ERR_BAD_ARGS
        out_err_line[] = 0
        return None
    var err = Int32(0)
    var err_line = Int64(0)
    var res = _fasta_parse(data, length, err, err_line)
    out_err[] = err
    out_err_line[] = err_line
    if err != ERR_NONE:
        res.free()
        return None
    var boxed = unsafe_alloc[FastaResult](1)
    boxed[] = res^
    return boxed.unsafe_bitcast[UInt8]()


@export
def bioparse_fasta_count(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    var r = handle.value().unsafe_bitcast[FastaResult]()
    return r[].n


@export
def bioparse_fasta_arrays(
    handle: Handle,
    titles: U8PtrPtr,
    title_offs: I64PtrPtr,
    seqs: U8PtrPtr,
    seq_offs: I64PtrPtr,
) abi("C") -> Int32:
    """Expose the concat arenas; pointers stay valid until _free."""
    if not handle:
        return 1
    var r = handle.value().unsafe_bitcast[FastaResult]()
    titles[] = r[].titles.ptr
    title_offs[] = r[].title_offs.ptr
    seqs[] = r[].seqs.ptr
    seq_offs[] = r[].seq_offs.ptr
    return 0


@export
def bioparse_fasta_free(handle: Handle) abi("C"):
    if not handle:
        return
    var r = handle.value().unsafe_bitcast[FastaResult]()
    r[].free()
    r.unsafe_free()


@export
def bioparse_gb_parse(
    data: U8Ptr, length: Int64, out_err: I32Ptr, out_err_line: I64Ptr
) abi("C") -> Handle:
    """Parse a whole GenBank buffer. NULL on error (code in out_err)."""
    if length < 0:
        out_err[] = ERR_BAD_ARGS
        out_err_line[] = 0
        return None
    var err = Int32(0)
    var err_line = Int64(0)
    var res = _gb_parse(data, length, err, err_line)
    out_err[] = err
    out_err_line[] = err_line
    if err != ERR_NONE:
        res.free()
        return None
    var boxed = unsafe_alloc[GbResult](1)
    boxed[] = res^
    return boxed.unsafe_bitcast[UInt8]()


@export
def bioparse_gb_count(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    var r = handle.value().unsafe_bitcast[GbResult]()
    return r[].n_rec


@export
def bioparse_gb_record_arrays(
    handle: Handle,
    names: U8PtrPtr,
    name_offs: I64PtrPtr,
    defs: U8PtrPtr,
    def_offs: I64PtrPtr,
    accs: U8PtrPtr,
    acc_offs: I64PtrPtr,
    vers: U8PtrPtr,
    ver_offs: I64PtrPtr,
    seqs: U8PtrPtr,
    seq_offs: I64PtrPtr,
    rec_feat_offs: I64PtrPtr,
) abi("C") -> Int32:
    if not handle:
        return 1
    var r = handle.value().unsafe_bitcast[GbResult]()
    names[] = r[].names.ptr
    name_offs[] = r[].name_offs.ptr
    defs[] = r[].defs.ptr
    def_offs[] = r[].def_offs.ptr
    accs[] = r[].accs.ptr
    acc_offs[] = r[].acc_offs.ptr
    vers[] = r[].vers.ptr
    ver_offs[] = r[].ver_offs.ptr
    seqs[] = r[].seqs.ptr
    seq_offs[] = r[].seq_offs.ptr
    rec_feat_offs[] = r[].rec_feat_offs.ptr
    return 0


@export
def bioparse_gb_feature_arrays(
    handle: Handle,
    keys: U8PtrPtr,
    key_offs: I64PtrPtr,
    locs: U8PtrPtr,
    loc_offs: I64PtrPtr,
    feat_qual_offs: I64PtrPtr,
) abi("C") -> Int32:
    if not handle:
        return 1
    var r = handle.value().unsafe_bitcast[GbResult]()
    keys[] = r[].keys.ptr
    key_offs[] = r[].key_offs.ptr
    locs[] = r[].locs.ptr
    loc_offs[] = r[].loc_offs.ptr
    feat_qual_offs[] = r[].feat_qual_offs.ptr
    return 0


@export
def bioparse_gb_qualifier_arrays(
    handle: Handle,
    qnames: U8PtrPtr,
    qname_offs: I64PtrPtr,
    qvals: U8PtrPtr,
    qval_offs: I64PtrPtr,
    qquoted: U8PtrPtr,
) abi("C") -> Int32:
    if not handle:
        return 1
    var r = handle.value().unsafe_bitcast[GbResult]()
    qnames[] = r[].qnames.ptr
    qname_offs[] = r[].qname_offs.ptr
    qvals[] = r[].qvals.ptr
    qval_offs[] = r[].qval_offs.ptr
    qquoted[] = r[].qquoted.ptr
    return 0


@export
def bioparse_gb_free(handle: Handle) abi("C"):
    if not handle:
        return
    var r = handle.value().unsafe_bitcast[GbResult]()
    r[].free()
    r.unsafe_free()
