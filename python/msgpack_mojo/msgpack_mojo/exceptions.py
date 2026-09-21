"""Exception hierarchy mirroring ``msgpack.exceptions``.

Same class names and the same inheritance shapes (``FormatError`` is a
``ValueError`` and an ``UnpackException``; ``ExtraData`` is a ``ValueError``),
so ``except msgpack.ValueError``-style handlers keep working when the import
is switched to ``msgpack_mojo``.
"""

from __future__ import annotations


class PackException(Exception):
    """Base class for pack-side failures."""


class UnpackException(Exception):
    """Base class for unpack-side failures."""


class FormatError(ValueError, UnpackException):
    """The input bytes are not valid MessagePack (e.g. the reserved 0xc1)."""


class StackError(ValueError, UnpackException):
    """Reserved for parity with ``msgpack.exceptions.StackError``."""


class ExtraData(ValueError):
    """``unpackb`` received trailing bytes after one complete value."""


class OutOfData(UnpackException):
    """Reserved for streaming-unpacker parity (not raised by this package)."""


class BufferFull(Exception):
    """Reserved for streaming-unpacker parity (not raised by this package)."""
