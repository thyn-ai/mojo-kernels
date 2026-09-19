"""Vendored pure-Python FASTA/GenBank scanner (fallback backend).

This is the fallback used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``BIO_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
FASTA and GenBank flat-file formats, written line-by-line parallel to
``kernels/bioparse/src/bioparse.mojo`` so the two backends produce
byte-identical :class:`~bio_mojo._raw` structures for every input —
the differential suite asserts exactly that, on top of oracle equality.

Like the kernel, this module never decodes text and never builds public
record objects; all Unicode handling and normalization lives in
``bio_mojo.core``.
"""

from __future__ import annotations

from bio_mojo._raw import (
    ERR_FASTA_PREAMBLE,
    ERR_GB_BAD_SEQ_LINE,
    ERR_GB_NO_ORIGIN,
    ERR_GB_NO_TERMINATOR,
    ERR_GB_MISPLACED_FEATURE_LINE,
    RawFasta,
    RawFeature,
    RawGbRecord,
    RawQualifier,
    raise_for_error,
)

_ST_BETWEEN, _ST_HEAD, _ST_FEATURES, _ST_ORIGIN = 0, 1, 2, 3
_SEC_OTHER, _SEC_DEF = 0, 1

_KEYWORDS = (b"//", b"LOCUS", b"DEFINITION", b"ACCESSION", b"VERSION",
             b"FEATURES", b"ORIGIN")


def _iter_lines(data: bytes):
    """Yield (start, end) of each line, universal newlines, no terminator.

    Every ``find`` is bounded to the current line so the whole scan stays
    O(len(data)) (an unbounded ``find(b"\\r")`` on LF-only data degenerates
    to O(n^2)).
    """
    pos = 0
    n = len(data)
    while pos < n:
        i_nl = data.find(b"\n", pos)
        limit = i_nl if i_nl != -1 else n
        i_cr = data.find(b"\r", pos, limit)
        eol = i_cr if i_cr != -1 else limit
        yield pos, eol
        if eol >= n:
            pos = n
        elif data[eol : eol + 2] == b"\r\n":
            pos = eol + 2
        else:
            pos = eol + 1


def _kw(line: bytes, word: bytes) -> bool:
    """True if the line starts with `word` followed by space, tab, or EOL."""
    if not line.startswith(word):
        return False
    if len(line) == len(word):
        return True
    return line[len(word)] in (0x20, 0x09)


def fasta_parse_bytes(data: bytes) -> RawFasta:
    """Parse a whole FASTA buffer into concat arenas (kernel-equivalent)."""
    titles = bytearray()
    seqs = bytearray()
    title_offs: list[int] = []
    seq_offs: list[int] = []
    in_record = False
    lineno = 0
    for s, e in _iter_lines(data):
        lineno += 1
        line = data[s:e]
        if line.startswith(b">"):
            title_offs.append(len(titles))
            titles += line[1:]
            seq_offs.append(len(seqs))
            in_record = True
        else:
            if not in_record:
                raise_for_error(ERR_FASTA_PREAMBLE, lineno)
            seqs += line.rstrip().translate(None, b" \t")
    title_offs.append(len(titles))
    seq_offs.append(len(seqs))
    return RawFasta(bytes(titles), title_offs, bytes(seqs), seq_offs)


class _FeatureScratch:
    __slots__ = ("key", "loc_parts", "qualifiers", "qual_open", "in_quote")

    def __init__(self, key: bytes):
        self.key = key
        self.loc_parts: list[bytes] = []
        self.qualifiers: list[RawQualifier] = []
        self.qual_open = False
        self.in_quote = False


def _scan_quote(value: bytes, in_quote: bool) -> bool:
    i = 0
    n = len(value)
    while i < n:
        if value[i] == 0x22:  # '"'
            if i + 1 < n and value[i + 1] == 0x22:
                i += 2
                continue
            in_quote = not in_quote
        i += 1
    return in_quote


def gb_parse_bytes(data: bytes) -> list[RawGbRecord]:
    """Parse a whole GenBank buffer into raw records (kernel-equivalent)."""
    records: list[RawGbRecord] = []
    state = _ST_BETWEEN
    section = _SEC_OTHER
    name = definition = accession = version = seq = b""
    def_parts: list[bytes] = []
    seq_parts: list[bytes] = []
    features: list[RawFeature] = []
    feat: _FeatureScratch | None = None
    wrote_def = wrote_acc = wrote_ver = wrote_seq = False
    lineno = 0

    def close_feature() -> None:
        nonlocal feat
        if feat is None:
            return
        location = b"".join(feat.loc_parts)
        # The oracle drops features whose location is empty, qualifiers
        # included.
        if location.strip(b" "):
            features.append(RawFeature(feat.key, location, feat.qualifiers))
        feat = None

    for s, e in _iter_lines(data):
        lineno += 1
        if e <= s:
            continue  # blank line: skipped in every section
        line = data[s:e]
        # Column-0 lines are section boundaries, EXCEPT inside ORIGIN where
        # every line that is not '//' or 'LOCUS' is sequence data (the
        # oracle errors on malformed sequence lines there).
        col0 = line[0] not in (0x20, 0x09)
        boundary = False
        if col0:
            if _kw(line, b"//"):
                boundary = True
                if state == _ST_BETWEEN:
                    continue  # stray terminator between records: junk
                close_feature()
                if not wrote_seq:
                    raise_for_error(ERR_GB_NO_ORIGIN, lineno)
                records.append(
                    RawGbRecord(
                        name,
                        b" ".join(def_parts),
                        accession,
                        version,
                        b"".join(seq_parts),
                        features,
                    )
                )
                state = _ST_BETWEEN
                def_parts, seq_parts, features = [], [], []
                wrote_def = wrote_acc = wrote_ver = wrote_seq = False
                section = _SEC_OTHER
            elif _kw(line, b"LOCUS"):
                boundary = True
                if state != _ST_BETWEEN:
                    raise_for_error(ERR_GB_NO_TERMINATOR, lineno)
                # Name = first whitespace token after the 'LOCUS' keyword.
                name = line[5:].lstrip(b" \t").split(b" ", 1)[0].split(b"\t", 1)[0]
                state = _ST_HEAD
                section = _SEC_OTHER
                accession = version = b""
                def_parts, seq_parts, features = [], [], []
                wrote_def = wrote_acc = wrote_ver = wrote_seq = False
            elif state == _ST_BETWEEN:
                continue  # junk between records
            elif _kw(line, b"DEFINITION") and state == _ST_HEAD and not wrote_def:
                boundary = True
                def_parts = [line[12:].rstrip()]
                wrote_def = True
                section = _SEC_DEF
            elif _kw(line, b"ACCESSION") and state == _ST_HEAD and not wrote_acc:
                boundary = True
                accession = line[12:].rstrip()
                wrote_acc = True
                section = _SEC_OTHER
            elif _kw(line, b"VERSION") and state == _ST_HEAD and not wrote_ver:
                boundary = True
                version = line[12:].rstrip()
                wrote_ver = True
                section = _SEC_OTHER
            elif _kw(line, b"FEATURES") and state == _ST_HEAD:
                boundary = True
                state = _ST_FEATURES
                section = _SEC_OTHER
            elif _kw(line, b"ORIGIN") and state in (_ST_HEAD, _ST_FEATURES):
                boundary = True
                close_feature()
                state = _ST_ORIGIN
                wrote_seq = True
            elif state != _ST_ORIGIN:
                # CONTIG / KEYWORDS / SOURCE / REFERENCE / COMMENT / ... :
                # sections outside the supported subset are skipped.
                boundary = True
                if state == _ST_FEATURES:
                    close_feature()
                state = _ST_HEAD
                section = _SEC_OTHER
        if boundary:
            continue

        # ---- indented line (or in-ORIGIN col0 data line) ----
        if state == _ST_HEAD:
            if section == _SEC_DEF and wrote_def:
                def_parts.append(line[12:].rstrip())
        elif state == _ST_FEATURES:
            is_feat = (
                len(line) > 5
                and line[:5] == b"     "
                and line[5] not in (0x20, 0x09)
            )
            if is_feat:
                close_feature()
                key_end = min(21, len(line))
                key = line[5:key_end]
                sp = key.find(b" ")
                if sp != -1:
                    key = key[:sp]
                feat = _FeatureScratch(key)
                if len(line) > 21:
                    feat.loc_parts.append(line[21:].rstrip())
            elif len(line) >= 21 and line[:21] == b" " * 21:
                content = line[21:].rstrip()
                if feat is None:
                    continue  # content before any feature line: skipped
                if feat.in_quote and feat.qual_open:
                    prev = feat.qualifiers[-1]
                    sep = b"" if prev.name == b"translation" else b" "
                    feat.qualifiers[-1] = prev._replace(
                        value=prev.value + sep + content
                    )
                    feat.in_quote = _scan_quote(content, feat.in_quote)
                elif content.startswith(b"/"):
                    body = content[1:]
                    eq = body.find(b"=")
                    if eq == -1:
                        qname, value, quoted, has_value = body, b"", False, False
                    else:
                        qname = body[:eq]
                        value = body[eq + 1 :]
                        quoted = value.startswith(b'"')
                        has_value = True
                    feat_in_quote = False
                    if quoted:
                        feat_in_quote = _scan_quote(value[1:], True)
                    feat.qualifiers.append(
                        RawQualifier(qname, value, quoted, has_value)
                    )
                    feat.qual_open = True
                    feat.in_quote = feat_in_quote
                else:
                    # Location continuation: concatenated verbatim, but only
                    # while the location's parentheses are unbalanced (the
                    # oracle crashes on continuations of a complete
                    # location; we fail explicitly instead).
                    loc_so_far = b"".join(feat.loc_parts)
                    if loc_so_far.count(b"(") <= loc_so_far.count(b")"):
                        raise_for_error(ERR_GB_MISPLACED_FEATURE_LINE, lineno)
                    feat.loc_parts.append(content)
            # anything else inside FEATURES (e.g. the Location/Qualifiers
            # header) is skipped
        elif state == _ST_ORIGIN:
            stripped = line.lstrip(b" \t")
            if stripped:
                if not stripped[0:1].isdigit():
                    raise_for_error(ERR_GB_BAD_SEQ_LINE, lineno)
                i = 0
                while i < len(stripped) and stripped[i] not in (0x20, 0x09):
                    i += 1
                if i >= len(stripped):
                    raise_for_error(ERR_GB_BAD_SEQ_LINE, lineno)
                seq_parts.append(
                    stripped[i + 1 :]
                    .translate(None, b"0123456789 \t\x0b\x0c\r")
                    .upper()
                )

    if state != _ST_BETWEEN:
        # An unterminated final record is accepted iff it has its ORIGIN.
        close_feature()
        if not wrote_seq:
            raise_for_error(ERR_GB_NO_ORIGIN, lineno)
        records.append(
            RawGbRecord(
                name,
                b" ".join(def_parts),
                accession,
                version,
                b"".join(seq_parts),
                features,
            )
        )
    return records
