"""Pure-Python implementation of the python-slugify 9.1.0 pipeline stages.

This module is the fallback engine (used where the native kernel is
unavailable, and for the configurations the kernel does not accelerate:
``allow_unicode=True``, custom ``regex_pattern``, non-text-unidecode
backends, Unicode-digit numeric references). It is a clean-room
re-implementation written from the observed behavior of the reference
package; the differential test suite asserts byte-identical output against
the oracle on every stage combination.

The legacy pipeline is the reference default (``algorithm='legacy'``); the
modern pipeline differs in entity-decode order, numeric-reference error
handling (per-reference instead of all-or-nothing), argument validation and
truncation semantics.
"""

from __future__ import annotations

import re
import unicodedata  # noqa: F401  (re-exported for the orchestration layer)
from html.entities import name2codepoint

from slugify_mojo._data import SlugData

QUOTE_PATTERN = re.compile(r"[']+")
DISALLOWED_CHARS_PATTERN = re.compile(r"[^-a-zA-Z0-9]+")
DISALLOWED_UNICODE_CHARS_PATTERN = re.compile(r"[\W_]+")
DUPLICATE_DASH_PATTERN = re.compile(r"-{2,}")
NUMBERS_PATTERN = re.compile(r"(?<=\d),(?=\d)")
DECIMAL_PATTERN = re.compile(r"&#(\d+);")
HEX_PATTERN_LEGACY = re.compile(r"&#x([\da-fA-F]+);")
HEX_PATTERN_MODERN = re.compile(r"&#[xX]([\da-fA-F]+);")
DEFAULT_SEPARATOR = "-"


def default_entities() -> dict[str, int]:
    """The stdlib HTML4 entity-name table the reference decodes against."""
    return dict(name2codepoint)


def transliterate(data: SlugData, text: str) -> str:
    """Map every codepoint through the table; cp 0 -> NUL, codepoints past
    the table are dropped (the reference catches IndexError and skips)."""
    replaces = data.replaces
    n = len(replaces)
    out: list[str] = []
    append = out.append
    for ch in text:
        cp = ord(ch)
        if cp == 0:
            append("\x00")
        elif cp - 1 < n:
            append(replaces[cp - 1])
    return "".join(out)


_NAMED_PATTERN: re.Pattern[str] | None = None


def _named_entity_pattern(entities: dict[str, int]) -> re.Pattern[str]:
    global _NAMED_PATTERN
    if _NAMED_PATTERN is None:
        # Same alternation the reference builds from name2codepoint, in the
        # dict's insertion order. The stdlib table is process-constant, so a
        # module-level cache is exact.
        _NAMED_PATTERN = re.compile(r"&(%s);" % "|".join(entities))
    return _NAMED_PATTERN


def decode_entities(
    entities_map: dict[str, int],
    text: str,
    entities: bool,
    decimal: bool,
    hexadecimal: bool,
    legacy: bool,
) -> str:
    """Decode HTML entity references with the exact per-algorithm semantics.

    Legacy numeric passes are all-or-nothing: a single reference whose
    int()/chr() raises leaves every reference of that base literal. Modern
    numeric passes leave only the offending reference literal and never
    decode surrogates.
    """
    if entities:
        text = _named_entity_pattern(entities_map).sub(
            lambda m: chr(entities_map[m.group(1)]), text
        )
    if decimal:
        if legacy:
            try:
                text = DECIMAL_PATTERN.sub(lambda m: chr(int(m.group(1))), text)
            except (ValueError, OverflowError):
                pass
        else:
            text = DECIMAL_PATTERN.sub(lambda m: _numeric_reference(m, 10), text)
    if hexadecimal:
        if legacy:
            try:
                text = HEX_PATTERN_LEGACY.sub(lambda m: chr(int(m.group(1), 16)), text)
            except (ValueError, OverflowError):
                pass
        else:
            text = HEX_PATTERN_MODERN.sub(lambda m: _numeric_reference(m, 16), text)
    return text


def _numeric_reference(match: re.Match[str], base: int) -> str:
    """Modern per-reference decoding: invalid references stay literal."""
    try:
        value = int(match.group(1), base)
        if 0xD800 <= value <= 0xDFFF:
            return match.group(0)
        return chr(value)
    except (ValueError, OverflowError):
        return match.group(0)


def filter_quote_numbers(text: str) -> str:
    """Apostrophe removal, then digit-comma joining (both algorithms)."""
    text = QUOTE_PATTERN.sub("", text)
    return NUMBERS_PATTERN.sub("", text)


def filter_default_ascii(text: str) -> str:
    """The non-unicode default-character filter, dash dedup and trim."""
    text = filter_quote_numbers(text)
    text = DISALLOWED_CHARS_PATTERN.sub(DEFAULT_SEPARATOR, text)
    return DUPLICATE_DASH_PATTERN.sub(DEFAULT_SEPARATOR, text).strip(DEFAULT_SEPARATOR)


def filter_default_unicode(text: str) -> str:
    """The allow_unicode default-character filter (\\W_ runs), dash dedup, trim."""
    text = filter_quote_numbers(text)
    text = DISALLOWED_UNICODE_CHARS_PATTERN.sub(DEFAULT_SEPARATOR, text)
    return DUPLICATE_DASH_PATTERN.sub(DEFAULT_SEPARATOR, text).strip(DEFAULT_SEPARATOR)


def filter_custom_pattern(text: str, pattern: re.Pattern[str] | str) -> str:
    """Quote/number cleanup with a user-supplied disallowed-chars pattern."""
    text = filter_quote_numbers(text)
    text = re.sub(pattern, DEFAULT_SEPARATOR, text)
    return DUPLICATE_DASH_PATTERN.sub(DEFAULT_SEPARATOR, text).strip(DEFAULT_SEPARATOR)


def smart_truncate(
    string: str,
    max_length: int = 0,
    word_boundary: bool = False,
    separator: str = " ",
    save_order: bool = False,
) -> str:
    """Legacy public truncation behavior, including character-set stripping.

    Zero means unlimited; negative limits retain slicing semantics.
    """
    string = string.strip(separator)
    if not max_length:
        return string
    if len(string) < max_length:
        return string
    if not word_boundary:
        return string[:max_length].strip(separator)
    if separator not in string:
        return string[:max_length]
    truncated = ""
    for word in string.split(separator):
        if word:
            next_len = len(truncated) + len(word)
            if next_len < max_length:
                truncated += "{}{}".format(word, separator)
            elif next_len == max_length:
                truncated += "{}".format(word)
                break
            elif save_order:
                break
    if not truncated:
        truncated = string[:max_length]
    return truncated.strip(separator)


def modern_truncate(
    text: str,
    max_length: int,
    word_boundary: bool,
    separator: str,
    save_order: bool,
) -> str:
    """Modern truncation: budget internal dash-separated tokens before the
    output delimiter mapping; a hard cut never emits a partial delimiter."""
    if max_length <= 0:
        return text.replace(DEFAULT_SEPARATOR, separator)
    output_length = len(text) + text.count(DEFAULT_SEPARATOR) * (len(separator) - 1)
    if output_length <= max_length:
        return text.replace(DEFAULT_SEPARATOR, separator)
    tokens = text.split(DEFAULT_SEPARATOR)
    if word_boundary:
        words: list[str] = []
        length = 0
        for word in tokens:
            if not word:
                continue
            next_length = length + len(word) + (len(separator) if words else 0)
            if next_length <= max_length:
                words.append(word)
                length = next_length
            elif save_order:
                break
        if words:
            return separator.join(words)
    parts: list[str] = []
    length = 0
    for index, word in enumerate(tokens):
        delimiter = separator if index else ""
        remaining = max_length - length - len(delimiter)
        if remaining <= 0:
            break
        parts.append(delimiter + word[:remaining])
        length += len(delimiter) + min(len(word), remaining)
        if len(word) > remaining:
            break
    return "".join(parts)
