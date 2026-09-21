"""Drop-in slug generation, API-compatible with `python-slugify` 9.1.0.

`slugify(text, ...)` returns byte-identical strings to the reference package
for both algorithms ('legacy', the reference default, and 'modern'), powered
by a Mojo kernel where the platform supports it (macOS arm64, Linux x86_64),
with a pure-Python fallback everywhere else. The kernel accelerates the hot
stages — the per-character transliteration table lookup, entity-reference
decoding and the ASCII disallowed-character filter — while every
Unicode-aware step the reference delegates to CPython (NFKD/NFKC
normalization, case folding, user regex patterns, the allow_unicode path)
runs in CPython here too, so behavior matches exactly.

Backend resolution mirrors the reference: 'auto' uses the GPL `unidecode`
package when it is importable, else `text-unidecode` (whose table this
package loads at runtime). Set SLUGIFY_MOJO_DISABLE_NATIVE=1 to force the
pure-Python fallback.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import unicodedata
from collections.abc import Iterable
from importlib import import_module
from typing import Literal

from slugify_mojo import _pipeline
from slugify_mojo._data import SlugDataError, get_data
from slugify_mojo._native import (
    F_ENT_DECIMAL,
    F_ENT_HEX,
    F_ENT_NAMED,
    F_FILTER,
    F_LOWER,
    F_NUMERIC_LEGACY,
    F_QUOTE_DASH,
    F_TRANS,
    NativeTable,
    NativeUnavailable,
)

__all__ = [
    "slugify",
    "slugify_column",
    "smart_truncate",
    "backend",
    "Backend",
    "ReplacementStage",
    "Algorithm",
    "DEFAULT_SEPARATOR",
]

Backend = Literal["auto", "text-unidecode", "unidecode", "anyascii"]
ReplacementStage = Literal["both", "pre", "post"]
Algorithm = Literal["legacy", "modern"]

DEFAULT_SEPARATOR = "-"

_VALID_BACKENDS = ("auto", "text-unidecode", "unidecode", "anyascii")
_VALID_STAGES = ("both", "pre", "post")
_VALID_ALGORITHMS = ("legacy", "modern")


def _encode(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "surrogatepass")


# --------------------------------------------------------------------------
# Backend state (table snapshot + native model), resolved lazily
# --------------------------------------------------------------------------

_LOCK = threading.Lock()
_STATE: tuple[object, NativeTable | None] | None = None


def _get_state():
    """Load the tables once; build the native model when the kernel is up.

    A missing text-unidecode table is tolerated here (data=None): the
    allow_unicode and module-backend paths do not need it, matching the
    reference, which only touches text-unidecode when it actually
    transliterates. The table path re-raises via get_data() when reached.
    """
    global _STATE
    if _STATE is not None:
        return _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE
        try:
            data = get_data()
        except SlugDataError:
            data = None
        model: NativeTable | None = None
        if data is not None:
            try:
                model = NativeTable(data.blob())
            except NativeUnavailable:
                model = None
        _STATE = (data, model)
        return _STATE


def _backend():
    data, model = _get_state()
    # Forcing the fallback must work even after a native model was cached
    # (the differential suite runs the whole suite both ways).
    if os.environ.get("SLUGIFY_MOJO_DISABLE_NATIVE") == "1":
        return data, None
    return data, model


def backend() -> str:
    """Which engine serves the transliteration path right now."""
    _, model = _backend()
    return "native" if model is not None else "fallback"


# --------------------------------------------------------------------------
# Transliterator resolution (reference-compatible backend selection)
# --------------------------------------------------------------------------

_UNIDECODE_PROBE: bool | None = None  # None=not tried, False=absent, True=present


def _resolve_transliterator(backend_name: str):
    """Return 'table' for the text-unidecode table path, else a callable.

    Mirrors the reference: 'auto' prefers an importable `unidecode`, falling
    back to text-unidecode. The importable check is cached (the reference
    pays a failed-import penalty per call); a module appearing in
    ``sys.modules`` later is still picked up.
    """
    if backend_name == "text-unidecode":
        return "table"
    if backend_name == "auto":
        module = sys.modules.get("unidecode")
        if module is not None:
            return getattr(module, "unidecode")
        global _UNIDECODE_PROBE
        if _UNIDECODE_PROBE is None:
            try:
                module = import_module("unidecode")
            except ModuleNotFoundError as error:
                if error.name != "unidecode":
                    raise
                _UNIDECODE_PROBE = False
            else:
                _UNIDECODE_PROBE = True
                return getattr(module, "unidecode")
        return "table"
    module = import_module(backend_name.replace("-", "_"))
    return getattr(module, "anyascii" if backend_name == "anyascii" else "unidecode")


def _entities_map() -> dict:
    """The stdlib entity-name table (no table-package dependency)."""
    return _pipeline.default_entities()


def _legacy_entity_flags(entities: bool, decimal: bool, hexadecimal: bool) -> int:
    flags = F_NUMERIC_LEGACY
    if entities:
        flags |= F_ENT_NAMED
    if decimal:
        flags |= F_ENT_DECIMAL
    if hexadecimal:
        flags |= F_ENT_HEX
    return flags


def _decimal_kernel_safe(decimal: bool) -> bool:
    """The kernel models CPython's default 4300-digit int() ceiling; when the
    interpreter is configured otherwise, decimal references decode in Python."""
    return (not decimal) or sys.get_int_max_str_digits() == 4300


def _has_amp_cp(text: str, data) -> bool:
    """True if text contains a codepoint that transliterates to '&' (the
    fused fast path must not skip entity decoding for those)."""
    return any(c in text for c in data.amp_chars)


def _finish(text: str, lowercase, regex_pattern, model: NativeTable | None, allow_unicode: bool) -> str:
    """Shared tail of both pipelines: second normalization, case folding and
    the disallowed-character filter.

    NFKD/NFKC and str.lower() stay in CPython (they are Unicode-defined);
    the kernel's F_LOWER is selected only for pure-ASCII text, where the
    reference's .lower() reduces to the A-Z mapping the kernel performs.
    NFKD after transliteration is provably the identity on pure-ASCII text,
    so it is skipped there.
    """
    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
        if lowercase:
            text = text.lower()
        if regex_pattern is None:
            return _pipeline.filter_default_unicode(text)
        return _pipeline.filter_custom_pattern(text, regex_pattern)
    if text.isascii():
        kernel_lower = bool(lowercase)
    else:
        text = unicodedata.normalize("NFKD", text)
        if lowercase:
            text = text.lower()
        kernel_lower = False
    if regex_pattern is None:
        if model is not None:
            flags = F_FILTER | (F_LOWER if kernel_lower else 0)
            return _decode(model.transform(_encode(text), flags))
        if kernel_lower:
            text = text.lower()
        return _pipeline.filter_default_ascii(text)
    if kernel_lower:
        text = text.lower()
    return _pipeline.filter_custom_pattern(text, regex_pattern)


# --------------------------------------------------------------------------
# Legacy pipeline (reference default algorithm)
# --------------------------------------------------------------------------


def _legacy_slugify(
    text,
    entities,
    decimal,
    hexadecimal,
    max_length,
    word_boundary,
    separator,
    save_order,
    stopwords,
    regex_pattern,
    lowercase,
    replacements,
    allow_unicode,
    replacement_stage,
    backend_name,
) -> str:
    if not isinstance(text, str):
        if not isinstance(text, (bytes, bytearray)):
            raise TypeError(f"text must be str, bytes or bytearray, not {type(text).__name__}")
        text = text.decode("utf-8", "ignore")
    if replacement_stage not in _VALID_STAGES:
        raise ValueError("replacement_stage must be 'both', 'pre' or 'post'")
    if backend_name not in _VALID_BACKENDS:
        raise ValueError("backend must be 'auto', 'text-unidecode', 'unidecode' or 'anyascii'")

    if replacements and replacement_stage in ("both", "pre"):
        for old, new in replacements:
            text = text.replace(old, new)

    if allow_unicode:
        text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
        text = unicodedata.normalize("NFKC", text)
        text = _pipeline.decode_entities(
            _entities_map(), text, entities, decimal, hexadecimal, legacy=True
        )
        text = _finish(text, lowercase, regex_pattern, None, allow_unicode=True)
    else:
        transliterator = _resolve_transliterator(backend_name)
        data, model = _backend()
        if transliterator == "table" and data is None:
            get_data()  # unreachable after _get_state(); re-raises SlugDataError
        finished = False
        if transliterator == "table" and model is not None and _decimal_kernel_safe(decimal):
            ent_flags = F_TRANS | _legacy_entity_flags(entities, decimal, hexadecimal)
            if text.isascii():
                if "&" not in text and regex_pattern is None and data.fusion_safe:
                    # Fused fast path: NFKD is the identity on ASCII, and no
                    # entity reference can appear (no '&' in the input, and
                    # fusion_safe certifies only U+0026 could transliterate
                    # to one), so the quote fold, transliteration and ASCII
                    # filter run as ONE kernel call. Transliteration output
                    # is pure ASCII, so the second normalization is the
                    # identity as well.
                    flags = F_TRANS | F_QUOTE_DASH | F_FILTER
                    if lowercase:
                        flags |= F_LOWER
                    text = _decode(model.transform(_encode(text), flags))
                    finished = True
                else:
                    flags = ent_flags | F_QUOTE_DASH
                    text = _decode(model.transform(_encode(text), flags))
            else:
                text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
                text = unicodedata.normalize("NFKD", text)
                if (
                    "&" not in text
                    and regex_pattern is None
                    and data.fusion_safe
                    and not _has_amp_cp(text, data)
                ):
                    # Fused, entity-free: no reference can appear even after
                    # transliteration (the amp-producing codepoints are absent),
                    # so transliteration and the ASCII filter run as one call.
                    flags = F_TRANS | F_FILTER
                    if lowercase:
                        flags |= F_LOWER
                    text = _decode(model.transform(_encode(text), flags))
                    finished = True
                else:
                    text = _decode(model.transform(_encode(text), ent_flags))
        else:
            text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
            text = unicodedata.normalize("NFKD", text)
            if transliterator == "table":
                text = _pipeline.transliterate(data, text)
            else:
                text = transliterator(text)
            if "&" in text:
                text = _pipeline.decode_entities(
                    _entities_map(), text, entities, decimal, hexadecimal, legacy=True
                )
        if not finished:
            text = _finish(text, lowercase, regex_pattern, model, allow_unicode=False)

    if stopwords:
        excluded = [word.lower() for word in stopwords] if lowercase else stopwords
        text = DEFAULT_SEPARATOR.join(
            word for word in text.split(DEFAULT_SEPARATOR) if word not in excluded
        )
    if replacements and replacement_stage in ("both", "post"):
        for old, new in replacements:
            text = text.replace(old, new)

    if max_length > 0:
        text = _pipeline.smart_truncate(text, max_length, word_boundary, DEFAULT_SEPARATOR, save_order)
    if separator != DEFAULT_SEPARATOR:
        return text.replace(DEFAULT_SEPARATOR, separator)
    return text


# --------------------------------------------------------------------------
# Modern pipeline
# --------------------------------------------------------------------------


def _modern_slugify(
    text,
    entities,
    decimal,
    hexadecimal,
    max_length,
    word_boundary,
    separator,
    save_order,
    stopwords,
    regex_pattern,
    lowercase,
    replacements,
    allow_unicode,
    replacement_stage,
    backend_name,
) -> str:
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError(f"max_length must be an int, not {type(max_length).__name__}")
    if not isinstance(separator, str):
        raise TypeError(f"separator must be str, not {type(separator).__name__}")
    if not isinstance(text, str):
        if not isinstance(text, (bytes, bytearray)):
            raise TypeError(f"text must be str, bytes or bytearray, not {type(text).__name__}")
        text = text.decode("utf-8", "ignore")
    if replacement_stage not in _VALID_STAGES:
        raise ValueError("replacement_stage must be 'both', 'pre' or 'post'")
    if backend_name not in _VALID_BACKENDS:
        raise ValueError("backend must be 'auto', 'text-unidecode', 'unidecode' or 'anyascii'")

    rules = tuple((old, new) for old, new in replacements) if replacements else ()
    if rules and replacement_stage in ("both", "pre"):
        for old, new in rules:
            text = text.replace(old, new)

    # Entity decoding on the raw text stays in CPython on every backend: the
    # numeric patterns match Unicode decimal digits, which the byte-level
    # kernel does not model.
    text = _pipeline.decode_entities(
        _entities_map(), text, entities, decimal, hexadecimal, legacy=False
    )
    if allow_unicode:
        text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
        text = unicodedata.normalize("NFKC", text)
        text = _finish(text, lowercase, regex_pattern, None, allow_unicode=True)
    else:
        transliterator = _resolve_transliterator(backend_name)
        data, model = _backend()
        if transliterator == "table" and data is None:
            get_data()  # re-raises SlugDataError
        if transliterator == "table" and model is not None:
            finished = False
            if regex_pattern is None:
                # Entity decoding already ran in CPython above (modern
                # order); a '&' produced here by transliteration is never
                # entity-decoded in the reference either — it is filtered to
                # a dash like any other symbol — so transliteration and the
                # filter fuse into one call.
                flags = F_TRANS | F_FILTER
                if lowercase:
                    flags |= F_LOWER
                if text.isascii():
                    # NFKD is the identity on ASCII; the quote fold rides along.
                    text = _decode(model.transform(_encode(text), flags | F_QUOTE_DASH))
                else:
                    text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
                    text = unicodedata.normalize("NFKD", text)
                    text = _decode(model.transform(_encode(text), flags))
                finished = True
            elif text.isascii():
                # NFKD is the identity on ASCII, so the quote fold commutes
                # into the kernel call.
                text = _decode(model.transform(_encode(text), F_TRANS | F_QUOTE_DASH))
            else:
                text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
                text = unicodedata.normalize("NFKD", text)
                text = _decode(model.transform(_encode(text), F_TRANS))
        else:
            text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
            text = unicodedata.normalize("NFKD", text)
            if transliterator == "table":
                text = _pipeline.transliterate(data, text)
            else:
                text = transliterator(text)
            finished = False
        if not finished:
            text = _finish(text, lowercase, regex_pattern, model, allow_unicode=False)

    if stopwords:
        excluded = {word.lower() if lowercase else word for word in stopwords}
        text = DEFAULT_SEPARATOR.join(
            word for word in text.split(DEFAULT_SEPARATOR) if word not in excluded
        )
    if rules and replacement_stage in ("both", "post"):
        for old, new in rules:
            text = text.replace(old, new)

    return _pipeline.modern_truncate(text, max_length, word_boundary, separator, save_order)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def slugify(
    text: str | bytes | bytearray,
    entities: bool = True,
    decimal: bool = True,
    hexadecimal: bool = True,
    max_length: int = 0,
    word_boundary: bool = False,
    separator: str = DEFAULT_SEPARATOR,
    save_order: bool = False,
    stopwords: Iterable[str] = (),
    regex_pattern: re.Pattern[str] | str | None = None,
    lowercase: bool = True,
    replacements: Iterable[Iterable[str]] = (),
    allow_unicode: bool = False,
    *,
    replacement_stage: ReplacementStage = "both",
    backend: Backend = "auto",
    algorithm: Algorithm = "legacy",
) -> str:
    """Make a slug; byte-identical to python-slugify 9.1.0's slugify().

    algorithm='legacy' (the reference default) runs the frozen legacy
    pipeline; algorithm='modern' runs the reference's modern pipeline.
    """
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError("algorithm must be 'legacy' or 'modern'")
    if algorithm == "legacy":
        return _legacy_slugify(
            text, entities, decimal, hexadecimal, max_length, word_boundary,
            separator, save_order, stopwords, regex_pattern, lowercase,
            replacements, allow_unicode, replacement_stage, backend,
        )
    return _modern_slugify(
        text, entities, decimal, hexadecimal, max_length, word_boundary,
        separator, save_order, stopwords, regex_pattern, lowercase,
        replacements, allow_unicode, replacement_stage, backend,
    )


def _batch_eligible(kwargs: dict) -> bool:
    """The batched-FFI fast path covers the legacy, non-unicode,
    default-pattern, table-transliteration configuration; anything else goes
    through the per-item loop (identical output, one call per item)."""
    if kwargs.get("algorithm", "legacy") not in _VALID_ALGORITHMS:
        return False  # loop raises the same ValueError
    if kwargs.get("algorithm", "legacy") != "legacy":
        return False
    if kwargs.get("allow_unicode", False):
        return False
    if kwargs.get("regex_pattern", None) is not None:
        return False
    if kwargs.get("backend", "auto") not in _VALID_BACKENDS:
        return False  # loop raises the same ValueError
    if _resolve_transliterator(kwargs.get("backend", "auto")) != "table":
        return False
    if kwargs.get("replacement_stage", "both") not in _VALID_STAGES:
        return False  # loop raises the same ValueError
    return _decimal_kernel_safe(kwargs.get("decimal", True))


def slugify_column(texts: Iterable[str | bytes | bytearray], **kwargs) -> list[str]:
    """Batch slug generation: one slug per input, same per-item semantics as
    slugify(). On the native backend with a fast-path configuration the
    transliteration/entity and filter stages run as two FFI calls for the
    whole column; elsewhere it is a per-item loop with identical output.
    """
    items = list(texts)
    if not items:
        return []
    if not _batch_eligible(kwargs):
        return [slugify(t, **kwargs) for t in items]
    if not all(isinstance(t, str) for t in items):
        return [slugify(t, **kwargs) for t in items]
    entities = kwargs.get("entities", True)
    decimal = kwargs.get("decimal", True)
    hexadecimal = kwargs.get("hexadecimal", True)
    lowercase = kwargs.get("lowercase", True)
    replacements = kwargs.get("replacements", ())
    replacement_stage = kwargs.get("replacement_stage", "both")
    stopwords = kwargs.get("stopwords", ())
    max_length = kwargs.get("max_length", 0)
    word_boundary = kwargs.get("word_boundary", False)
    save_order = kwargs.get("save_order", False)
    separator = kwargs.get("separator", DEFAULT_SEPARATOR)

    data, model = _backend()
    if data is None or model is None:
        return [slugify(t, **kwargs) for t in items]

    base_flags = F_TRANS | _legacy_entity_flags(entities, decimal, hexadecimal)
    prepared: list[str] = []
    flags_a: list[int] = []
    for text in items:
        if replacements and replacement_stage in ("both", "pre"):
            for old, new in replacements:
                text = text.replace(old, new)
        if text.isascii():
            if "&" not in text and data.fusion_safe:
                # Fused: quote fold + transliterate + filter in one pass;
                # the second batch call is a flags=0 identity for this item.
                flags_a.append(base_flags | F_QUOTE_DASH | F_FILTER | (F_LOWER if lowercase else 0))
            else:
                # NFKD is the identity on ASCII; the quote fold rides along.
                flags_a.append(base_flags | F_QUOTE_DASH)
            prepared.append(text)
        else:
            text = _pipeline.QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)
            text = unicodedata.normalize("NFKD", text)
            if "&" not in text and data.fusion_safe and not _has_amp_cp(text, data):
                flags_a.append(F_TRANS | F_FILTER | (F_LOWER if lowercase else 0))
            else:
                flags_a.append(base_flags)
            prepared.append(text)

    stage1 = [
        _decode(raw)
        for raw in model.transform_batch([_encode(t) for t in prepared], flags_a)
    ]

    if all(fl & F_FILTER for fl in flags_a):
        # Every item fused: the first pass already produced final slugs.
        stage3 = stage1
    else:
        middle: list[str] = []
        flags_b: list[int] = []
        for i, text in enumerate(stage1):
            if flags_a[i] & F_FILTER:
                # Fused item: already fully processed by the first pass.
                middle.append(text)
                flags_b.append(0)
            elif text.isascii():
                middle.append(text)
                flags_b.append(F_FILTER | (F_LOWER if lowercase else 0))
            else:
                text = unicodedata.normalize("NFKD", text)
                if lowercase:
                    text = text.lower()
                middle.append(text)
                flags_b.append(F_FILTER)

        stage3 = [
            _decode(raw)
            for raw in model.transform_batch([_encode(t) for t in middle], flags_b)
        ]

    out: list[str] = []
    for text in stage3:
        if stopwords:
            excluded = [word.lower() for word in stopwords] if lowercase else stopwords
            text = DEFAULT_SEPARATOR.join(
                word for word in text.split(DEFAULT_SEPARATOR) if word not in excluded
            )
        if replacements and replacement_stage in ("both", "post"):
            for old, new in replacements:
                text = text.replace(old, new)
        if max_length > 0:
            text = _pipeline.smart_truncate(
                text, max_length, word_boundary, DEFAULT_SEPARATOR, save_order
            )
        if separator != DEFAULT_SEPARATOR:
            text = text.replace(DEFAULT_SEPARATOR, separator)
        out.append(text)
    return out


def smart_truncate(
    string: str,
    max_length: int = 0,
    word_boundary: bool = False,
    separator: str = " ",
    save_order: bool = False,
) -> str:
    """The reference's public legacy truncation helper, unchanged."""
    return _pipeline.smart_truncate(string, max_length, word_boundary, separator, save_order)
