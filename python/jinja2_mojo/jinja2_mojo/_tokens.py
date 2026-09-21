"""Build a jinja2-compatible token list from native kernel records.

The native kernel performs the structural scan (token boundaries, kinds,
line numbers, bracket balance); this module applies the value semantics
probed from the reference package (PyPI jinja2 3.1.6), used strictly as a
black-box oracle:

* data tokens: ``\\r\\n`` / ``\\r`` normalised to ``\\n``; whitespace-control
  trimming (``-`` strips all adjacent whitespace, ``+`` disables the
  trim/lstrip environment flags); ``trim_blocks`` drops one newline after a
  block or comment end; ``lstrip_blocks`` strips line-leading spaces/tabs
  before a block or comment begin; a single trailing newline is dropped
  unless ``keep_trailing_newline``; empty data tokens are never emitted.
* strings: Python string-literal escapes (``\\n \\t \\r \\b \\f \\v \\a \\\\
  \\' \\" \\0-\\777 \\xHH \\uHHHH \\UHHHHHHHH \\N{...}`` and backslash-newline
  continuation); unknown escapes keep their backslash.
* numbers: decimal / hex / octal / binary integers with PEP-515 underscores
  (base taken from the prefix, like the oracle), floats with fraction and/or
  exponent.
* names: validated against the oracle's word-character semantics; anything
  else declines the template.

Any anomaly returns ``None`` — never a guess — and the caller falls back to
the stock lexer for that template.
"""

from __future__ import annotations

import re
import unicodedata

from jinja2.lexer import Token

from . import _native

K_DATA = 1
K_NAME = 2
K_INTEGER = 3
K_FLOAT = 4
K_STRING = 5
K_OPERATOR = 6
K_BLOCK_BEGIN = 7
K_BLOCK_END = 8
K_VARIABLE_BEGIN = 9
K_VARIABLE_END = 10
K_COMMENT = 11

# Operator text -> token type name, probed from the oracle.
OPERATORS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "**": "pow",
    "/": "div",
    "//": "floordiv",
    "%": "mod",
    "==": "eq",
    "!=": "ne",
    ">": "gt",
    ">=": "gteq",
    "<": "lt",
    "<=": "lteq",
    "=": "assign",
    "~": "tilde",
    ":": "colon",
    ".": "dot",
    "|": "pipe",
    ",": "comma",
    ";": "semicolon",
    "(": "lparen",
    ")": "rparen",
    "[": "lbracket",
    "]": "rbracket",
    "{": "lbrace",
    "}": "rbrace",
}

_MINUS = ord("-")
_PLUS = ord("+")

_NAME_RE = re.compile(r"\w+")
_LSTRIP_LINE_RE = re.compile(r"[ \t]+$")
_WS_PREFIX_RE = re.compile(r"^\s+")
_WS_SUFFIX_RE = re.compile(r"\s+$")
_NEWLINES_RE = re.compile(r"\r\n|\r")

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "a": "\a",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


class _Decline(ValueError):
    """The template cannot be tokenised natively with oracle parity."""


def _unescape(raw: str, quote: str) -> str:
    """Unescape one string literal's raw content (Python-literal semantics)."""
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            raise _Decline("dangling backslash in string literal")
        e = raw[i]
        simple = _SIMPLE_ESCAPES.get(e)
        if simple is not None:
            out.append(simple)
            i += 1
        elif e == "\n":
            i += 1  # backslash-newline continuation
        elif e == "\r":
            i += 1
            if i < n and raw[i] == "\n":
                i += 1
        elif e == "x":
            h = raw[i + 1 : i + 3]
            if len(h) != 2 or any(ch not in _HEX_DIGITS for ch in h):
                raise _Decline("bad \\x escape")
            out.append(chr(int(h, 16)))
            i += 3
        elif e == "u":
            h = raw[i + 1 : i + 5]
            if len(h) != 4 or any(ch not in _HEX_DIGITS for ch in h):
                raise _Decline("bad \\u escape")
            out.append(chr(int(h, 16)))
            i += 5
        elif e == "U":
            h = raw[i + 1 : i + 9]
            if len(h) != 8 or any(ch not in _HEX_DIGITS for ch in h):
                raise _Decline("bad \\U escape")
            cp = int(h, 16)
            if cp > 0x10FFFF:
                raise _Decline("bad \\U escape")
            out.append(chr(cp))
            i += 9
        elif e == "N":
            if i + 1 >= n or raw[i + 1] != "{":
                raise _Decline("bad \\N escape")
            j = raw.find("}", i + 2)
            if j < 0:
                raise _Decline("bad \\N escape")
            try:
                out.append(unicodedata.lookup(raw[i + 2 : j]))
            except KeyError as exc:
                raise _Decline("unknown \\N escape") from exc
            i = j + 1
        elif "0" <= e <= "7":
            j = i
            while j < n and j < i + 3 and "0" <= raw[j] <= "7":
                j += 1
            out.append(chr(int(raw[i:j], 8)))
            i = j
        else:
            # Unknown escape: the oracle keeps the backslash literally.
            out.append("\\" + e)
            i += 1
    return "".join(out)


def _convert_integer(text: str) -> int:
    if text[:2] in ("0x", "0X", "0b", "0B", "0o", "0O"):
        return int(text, 0)
    return int(text)


class _Item:
    """One intermediate token (kind, signs, lineno, value) before trimming."""

    __slots__ = ("kind", "aux", "aux2", "lineno", "value")

    def __init__(self, kind: int, aux: int, aux2: int, lineno: int, value) -> None:
        self.kind = kind
        self.aux = aux
        self.aux2 = aux2
        self.lineno = lineno
        self.value = value


def _decode_items(src: bytes, records: list[tuple]) -> list[_Item]:
    """Decode raw records into items with typed values (data still raw text)."""
    items: list[_Item] = []
    for kind, aux, aux2, _pad, lineno, off, length in records:
        if kind == K_DATA:
            value = _NEWLINES_RE.sub("\n", src[off : off + length].decode("utf-8"))
        elif kind == K_NAME:
            value = src[off : off + length].decode("utf-8")
            if _NAME_RE.fullmatch(value) is None:
                raise _Decline(f"name {value!r} outside oracle word semantics")
        elif kind == K_INTEGER:
            value = _convert_integer(src[off : off + length].decode("ascii"))
        elif kind == K_FLOAT:
            value = float(src[off : off + length].decode("ascii"))
        elif kind == K_STRING:
            value = _unescape(src[off : off + length].decode("utf-8"), chr(aux))
        elif kind == K_OPERATOR:
            text = src[off : off + length].decode("ascii")
            op_type = OPERATORS.get(text)
            if op_type is None:
                raise _Decline(f"unknown operator {text!r}")
            items.append(_Item(K_OPERATOR, 0, 0, lineno, (op_type, text)))
            continue
        elif kind in (K_BLOCK_BEGIN, K_VARIABLE_BEGIN, K_COMMENT):
            sign = chr(aux) if aux else ""
            opener = {K_BLOCK_BEGIN: "{%", K_VARIABLE_BEGIN: "{{", K_COMMENT: "{#"}[kind]
            value = opener + sign
        else:  # K_BLOCK_END / K_VARIABLE_END
            sign = chr(aux) if aux else ""
            closer = "%}" if kind == K_BLOCK_END else "}}"
            value = sign + closer
        items.append(_Item(kind, aux, aux2, lineno, value))
    return items


def _apply_trimming(items: list[_Item], *, trim_blocks: bool, lstrip_blocks: bool) -> None:
    """Whitespace control, matching the oracle's lex-time trimming.

    End-side passes run first (they can only shrink a following data token
    from the left), then begin-side passes (shrinking a preceding data token
    from the right); emptied data tokens are removed between the two so a
    begin marker never trims across a tag.
    """
    # End of tag -> strip the start of the following data token. The oracle
    # consumes that whitespace as part of the end marker, so the data token's
    # line number advances past every newline removed here.
    for i, it in enumerate(items):
        if it.kind not in (K_BLOCK_END, K_VARIABLE_END, K_COMMENT):
            continue
        sign = it.aux if it.kind != K_COMMENT else it.aux2
        j = i + 1
        if j >= len(items) or items[j].kind != K_DATA:
            continue
        v = items[j].value
        if sign == _MINUS:
            new = _WS_PREFIX_RE.sub("", v)
        elif (
            trim_blocks
            and it.kind != K_VARIABLE_END
            and sign != _PLUS
            and v.startswith("\n")
        ):
            new = v[1:]
        else:
            continue
        items[j].lineno += v[: len(v) - len(new)].count("\n")
        items[j].value = new
    items[:] = [it for it in items if it.kind != K_DATA or it.value != ""]
    # Begin of tag -> strip the end of the preceding data token.
    for i, it in enumerate(items):
        if it.kind not in (K_BLOCK_BEGIN, K_VARIABLE_BEGIN, K_COMMENT):
            continue
        j = i - 1
        if j < 0 or items[j].kind != K_DATA:
            continue
        if it.aux == _MINUS:
            items[j].value = _WS_SUFFIX_RE.sub("", items[j].value)
        elif (
            lstrip_blocks
            and it.kind != K_VARIABLE_BEGIN
            and it.aux != _PLUS
        ):
            v = items[j].value
            m = _LSTRIP_LINE_RE.search(v)
            if m and (m.start() == 0 or v[m.start() - 1] == "\n"):
                items[j].value = v[: m.start()]
    items[:] = [it for it in items if it.kind != K_DATA or it.value != ""]


def build_tokens(
    source: str,
    *,
    trim_blocks: bool = False,
    lstrip_blocks: bool = False,
    keep_trailing_newline: bool = False,
) -> list[Token] | None:
    """Tokenise ``source`` natively; ``None`` when the wrapper must fall back.

    The returned list is token-for-token identical to the oracle lexer's
    output for the supported subset (differential tests assert type, value
    and line number agreement; block/variable end values may differ from the
    oracle's, which folds trimmed-away whitespace into them — the parser
    never reads end-token values).
    """
    try:
        src = source.encode("utf-8")
    except UnicodeEncodeError:
        return None  # lone surrogates: let the stock lexer produce its error
    try:
        records = _native.scan(src)
    except _native.NativeUnavailable:
        raise
    except Exception:
        return None
    if records is None:
        return None
    try:
        items = _decode_items(src, records)
        _apply_trimming(items, trim_blocks=trim_blocks, lstrip_blocks=lstrip_blocks)
    except (_Decline, ValueError, OverflowError):
        return None
    if not keep_trailing_newline and items and items[-1].kind == K_DATA:
        v = items[-1].value
        if v.endswith("\n"):
            items[-1].value = v[:-1]
            if not items[-1].value:
                items.pop()
    out: list[Token] = []
    for it in items:
        if it.kind == K_COMMENT:
            continue
        if it.kind == K_DATA:
            out.append(Token(it.lineno, "data", it.value))
        elif it.kind == K_NAME:
            out.append(Token(it.lineno, "name", it.value))
        elif it.kind == K_INTEGER:
            out.append(Token(it.lineno, "integer", it.value))
        elif it.kind == K_FLOAT:
            out.append(Token(it.lineno, "float", it.value))
        elif it.kind == K_STRING:
            out.append(Token(it.lineno, "string", it.value))
        elif it.kind == K_OPERATOR:
            out.append(Token(it.lineno, it.value[0], it.value[1]))
        elif it.kind == K_BLOCK_BEGIN:
            out.append(Token(it.lineno, "block_begin", it.value))
        elif it.kind == K_BLOCK_END:
            out.append(Token(it.lineno, "block_end", it.value))
        elif it.kind == K_VARIABLE_BEGIN:
            out.append(Token(it.lineno, "variable_begin", it.value))
        elif it.kind == K_VARIABLE_END:
            out.append(Token(it.lineno, "variable_end", it.value))
    return out
