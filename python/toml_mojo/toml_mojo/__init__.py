"""toml-mojo: a drop-in faster replacement for the stdlib ``tomllib`` parser.

``toml_mojo.loads(text)`` returns exactly what ``tomllib.loads(text)``
returns — the same typed objects (``dict``/``list``/``str``/``int``/
``float``/``bool``/``datetime``) — parsed by a clean-room Mojo kernel where
the platform supports it (macOS arm64, Linux x86_64), and by stdlib
``tomllib`` itself everywhere else (the fallback is the reference parser, so
it is correct by construction).

    import toml_mojo

    doc = toml_mojo.loads("[tool.example]\\nversion = 1\\n")
    assert doc == {"tool": {"example": {"version": 1}}}

Like ``tomllib`` (and unlike ``tomlkit``), this package returns plain
``tomllib``-shaped data: comments and formatting trivia are discarded and
documents cannot be round-tripped back to text.

Set ``TOML_MOJO_DISABLE_NATIVE=1`` to force the ``tomllib`` fallback.
"""

from __future__ import annotations

import tomllib

from toml_mojo._decode import assemble
from toml_mojo._native import NativeUnavailable, backend_info, native_available, parse_bytes

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "TOMLDecodeError",
    "backend_info",
    "load",
    "loads",
    "native_available",
    "__version__",
]


class TOMLDecodeError(ValueError):
    """An invalid TOML document was passed to :func:`loads` (same base class
    as ``tomllib.TOMLDecodeError``)."""


def _line_col(data: bytes, offset: int) -> tuple[int, int]:
    """1-based (line, column) of a byte offset, for error messages."""
    line = data.count(b"\n", 0, offset) + 1
    col = offset - data.rfind(b"\n", 0, offset)
    return line, col


def _loads_native(s: str) -> dict:
    data = s.encode("utf-8", "surrogatepass")
    status, err_pos, stream = parse_bytes(data)
    if status != 0:
        line, col = _line_col(data, max(err_pos, 0))
        raise TOMLDecodeError(f"Invalid TOML (at line {line}, column {col})")
    return assemble(stream)  # raises NativeUnavailable on a corrupt stream


def loads(s: str, /) -> dict:
    """Parse a TOML document from a string (``tomllib.loads`` compatible).

    Raises :class:`TOMLDecodeError` (a ``ValueError``) on invalid input,
    and ``TypeError`` if ``s`` is not a ``str`` — both like ``tomllib``.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected str object, not '{type(s).__name__}'")
    try:
        return _loads_native(s)
    except NativeUnavailable:
        pass
    return tomllib.loads(s)


def load(fp, /) -> dict:
    """Parse a TOML document from a binary file object (``tomllib.load``
    compatible)."""
    data = fp.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return loads(data)
