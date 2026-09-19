"""Raw byte-level interchange between the parse backends and ``core``.

Both backends — the native Mojo kernel (``bio_mojo._native``) and the
vendored pure-Python parser (``bio_mojo._reference``) — produce exactly
these structures from an input buffer, so record assembly, validation and
normalization happen once, in ``bio_mojo.core``, and can never diverge
between backends.
"""

from __future__ import annotations

from typing import NamedTuple

from bio_mojo.errors import ParseError

# Kernel error codes (mirror kernels/bioparse/src/bioparse.mojo).
ERR_FASTA_PREAMBLE = 3
ERR_GB_NO_ORIGIN = 4
ERR_GB_NO_TERMINATOR = 5
ERR_GB_BAD_SEQ_LINE = 6
ERR_GB_MISPLACED_FEATURE_LINE = 7

_ERR_MESSAGES = {
    ERR_FASTA_PREAMBLE: (
        "FASTA file contains comments or other content before the first "
        "record: lines before the first '>' header are not allowed"
    ),
    ERR_GB_NO_ORIGIN: (
        "GenBank record ended before its ORIGIN sequence section was found"
    ),
    ERR_GB_NO_TERMINATOR: (
        "GenBank record is missing the '//' terminator before the next "
        "LOCUS line"
    ),
    ERR_GB_BAD_SEQ_LINE: (
        "malformed GenBank sequence line (expected leading coordinate digits)"
    ),
    ERR_GB_MISPLACED_FEATURE_LINE: (
        "misplaced GenBank feature-table line (a location may only "
        "continue while its parentheses are unbalanced)"
    ),
}


def raise_for_error(code: int, line: int) -> None:
    """Raise the shared ParseError for a backend error code."""
    message = _ERR_MESSAGES.get(code, f"parse failed with error code {code}")
    raise ParseError(message, line=line if line > 0 else None)


class RawFasta(NamedTuple):
    """Whole-buffer FASTA parse: concat arenas plus n+1 offsets per arena.

    Record i's title is ``titles[title_offs[i]:title_offs[i+1]]`` (raw bytes
    after '>', unstripped; core rstrips post-decode) and its sequence is
    ``seqs[seq_offs[i]:seq_offs[i+1]]`` (cleaned bytes, ASCII-whitespace
    rules applied).
    """

    titles: bytes
    title_offs: list[int]
    seqs: bytes
    seq_offs: list[int]


class RawQualifier(NamedTuple):
    name: bytes
    value: bytes  # raw joined value, quotes still present when quoted=True
    quoted: bool
    has_value: bool  # False for bare '/flag' qualifiers (no '=')


class RawFeature(NamedTuple):
    key: bytes
    location: bytes  # concatenated location string (may be empty -> dropped)
    qualifiers: list[RawQualifier]


class RawGbRecord(NamedTuple):
    name: bytes  # LOCUS name token
    definition: bytes  # DEFINITION lines joined with single spaces
    accession: bytes  # ACCESSION line content (column 12)
    version: bytes  # VERSION line content (column 12; may be empty)
    seq: bytes  # ORIGIN sequence, uppercased, digits/whitespace removed
    features: list[RawFeature]
