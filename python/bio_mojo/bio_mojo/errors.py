"""Structured error types for bio_mojo.

Both backends (native kernel and vendored pure-Python parser) raise these
exact types for the same malformed inputs, so callers get identical failure
behaviour regardless of the active backend. ``ParseError`` and
``StreamModeError`` subclass ``ValueError`` to match the oracle's
(``Bio.SeqIO``) exception hierarchy.
"""

from __future__ import annotations


class BioMojoError(Exception):
    """Base class for all bio_mojo errors."""


class ParseError(BioMojoError, ValueError):
    """Malformed input data (the oracle raises ValueError for these)."""

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        if line is not None:
            message = f"{message} (line {line})"
        super().__init__(message)


class StreamModeError(BioMojoError, ValueError):
    """A binary stream was passed where a text stream is required.

    Mirrors the oracle: ``Bio.SeqIO.parse`` raises ``Bio.StreamModeError``
    (a ValueError) for binary handles on both formats.
    """


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native bioparse kernel could not be found, loaded, or verified.

    Never propagates to callers of ``parse_fasta``/``parse_genbank``: those
    transparently fall back to the vendored pure-Python parser.
    """
