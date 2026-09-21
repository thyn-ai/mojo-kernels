"""Differential tests: slugify_mojo must match PyPI python-slugify exactly.

The oracle is the published PyPI package (python-slugify==9.1.0, with its
text-unidecode 1.3 transliteration backend). The suite runs twice via
``scripts/test_all_slugify.sh``: once against the native Mojo kernel and
once with SLUGIFY_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback). On
both backends every output must be byte-identical to the oracle — the
assertions are exact string equality, not tolerance-based.

The corpus is generated deterministically from fixed seeds (no network, no
datasets): unicode ranges (Latin/Cyrillic/Greek/CJK/emoji/accents/combining
marks/fullwidth forms), punctuation and dash runs, HTML entity references
(named/decimal/hex, valid/invalid/boundary, nested), reserved-word
stopwords, max_length truncation with and without word_boundary/save_order,
separator variants, lowercase=False, allow_unicode=True/False, replacements
at all stages, custom regex_pattern (str and compiled), both algorithms,
and bytes/bytearray input.
"""

from __future__ import annotations

import random
import re

import pytest
from slugify import slugify as oracle_slugify
from slugify import smart_truncate as oracle_truncate

import slugify_mojo

# --------------------------------------------------------------------------
# Deterministic corpus generation
# --------------------------------------------------------------------------

# Representative spans: (lo, hi) inclusive codepoint ranges the generator
# samples from. Covers the scope list plus table-boundary codepoints
# (text-unidecode's table ends at U+FFFF) and lone-surrogate producers.
UNICODE_SPANS = [
    (0x0020, 0x007E),  # printable ASCII
    (0x00C0, 0x00FF),  # Latin-1 letters (accents)
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x04FF),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic (incl. Arabic-Indic digits)
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x2000, 0x206F),  # general punctuation (dashes, quotes)
    (0x20A0, 0x20BF),  # currency
    (0x2100, 0x214F),  # letterlike symbols (NFKD folds to ASCII)
    (0x2460, 0x24FF),  # enclosed alphanumerics (NFKD digits)
    (0x3000, 0x30FF),  # CJK punctuation, hiragana, katakana
    (0x4E00, 0x4E80),  # CJK ideographs
    (0xAC00, 0xAC80),  # Hangul
    (0xFE00, 0xFE6F),  # variation selectors + small form variants
    (0xFF00, 0xFF65),  # fullwidth forms (NFKD -> ASCII, & # ; ' !)
    (0x0300, 0x036F),  # combining marks
    (0x1D400, 0x1D4FF),  # mathematical alphanumerics (NFKD letters)
    (0x1F300, 0x1F640),  # emoji (dropped by the table)
    (0xFFF0, 0xFFFF),  # table boundary
]

ENTITY_SNIPPETS = [
    "&amp;", "&AMP;", "&eacute;", "&notin;", "&not;", "&notit;", "&unknown;",
    "&amp", "&;", "&", "&&amp;", "&amp&amp;", "&amp=&amp;",
    "&#65;", "&#0;", "&#9;", "&#1114111;", "&#1114112;", "&#99999999;",
    "&#00065;", "&#65", "&#;", "&#x41;", "&#x1F600;", "&#X41;", "&#x;",
    "&#x110000;", "&#xD800;", "&#xDC00;", "&#55296;", "&#38;amp;", "&#38;#65;",
    "&lt;tag&gt;", "&rsquo;", "&apos;", "&#" + "9" * 4500 + ";",
    "&#" + "0" * 4500 + "1;", "&#x" + "f" * 100 + ";",
]

PUNCT_SNIPPETS = [
    "...", " - ", "--", "—", "–", "―", "- -", "'a'", "''", "'''",
    "a'b''c", "1,000", "1,00,00", "a,b", "1,2,3,4", "12,345.67",
    "   ", "a  b   c", "_", "__", "a_b_c", "a.b,c;d:e", "(test)",
    "[x]", "{y}", "a/b\\c", "line\nbreak", "tab\there", "nul\x00char",
    "\u200bzero-width", "\ufeffbom", "a\u0301", "e\u0301\u0302",
]

WORD_SNIPPETS = [
    "the quick brown fox", "a an the and or but", "resume résumé",
    "İstanbul i̇stanbul", "Œuvre œuvre", "ß ßß ẞ", "ﬁle ﬂag",
    "①②③", "½ + ¼ = ¾", "™ ® ©", "№ 5", "Ⅻ roman",
]


def _random_text(rng: random.Random) -> str:
    parts: list[str] = []
    for _ in range(rng.randint(0, 6)):
        kind = rng.random()
        if kind < 0.55:
            lo, hi = rng.choice(UNICODE_SPANS)
            span = "".join(
                chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 12))
            )
            parts.append(span)
        elif kind < 0.75:
            parts.append(rng.choice(ENTITY_SNIPPETS))
        elif kind < 0.9:
            parts.append(rng.choice(PUNCT_SNIPPETS))
        else:
            parts.append(rng.choice(WORD_SNIPPETS))
        if rng.random() < 0.5:
            parts.append(rng.choice([" ", "-", "_", "'", ".", "&", ";", ""]))
    return "".join(parts)


def _config_matrix(rng: random.Random) -> list[dict]:
    """Cross-product of the scope's configuration space, deterministically
    sampled to a tractable size."""
    configs: list[dict] = [{}]
    for max_length in (1, 4, 10, 25):
        configs.append({"max_length": max_length})
        configs.append({"max_length": max_length, "word_boundary": True})
        configs.append({"max_length": max_length, "word_boundary": True, "save_order": True})
        configs.append({"max_length": max_length, "save_order": True})
    for separator in ("_", "", " ", "+", "--", "×", "0"):
        configs.append({"separator": separator})
        configs.append({"separator": separator, "max_length": 12})
        configs.append({"separator": separator, "max_length": 12, "word_boundary": True})
    configs.append({"lowercase": False})
    configs.append({"lowercase": False, "max_length": 12, "word_boundary": True})
    configs.append({"allow_unicode": True})
    configs.append({"allow_unicode": True, "lowercase": False})
    configs.append({"allow_unicode": True, "max_length": 10, "word_boundary": True})
    configs.append({"allow_unicode": True, "separator": "_"})
    configs.append({"stopwords": ["the", "a", "and"]})
    configs.append({"stopwords": ["The", "A"], "lowercase": False})
    configs.append({"stopwords": ["the"], "allow_unicode": True})
    configs.append({"stopwords": ("de", "la"), "separator": "_"})
    configs.append({"replacements": [("foo", "bar"), ("a", "A")]})
    configs.append({"replacements": [("o", "0")], "replacement_stage": "pre"})
    configs.append({"replacements": [("o", "0"), ("-", "=")], "replacement_stage": "post"})
    configs.append({"replacements": [("e", "3")], "replacement_stage": "both"})
    configs.append({"replacements": [("x", "y")], "replacement_stage": "pre", "allow_unicode": True})
    configs.append({"regex_pattern": r"[^-a-zA-Z0-9_]+"})
    configs.append({"regex_pattern": r"[^-a-zA-Z0-9.]+"})
    configs.append({"regex_pattern": re.compile(r"[^a-z0-9]+")})
    configs.append({"regex_pattern": r"[^-a-zA-Z0-9_]+", "allow_unicode": True})
    configs.append({"entities": False})
    configs.append({"decimal": False})
    configs.append({"hexadecimal": False})
    configs.append({"entities": False, "decimal": False, "hexadecimal": False})
    configs.append({"entities": False, "allow_unicode": True})
    configs.append({"algorithm": "modern"})
    configs.append({"algorithm": "modern", "max_length": 12})
    configs.append({"algorithm": "modern", "max_length": 12, "word_boundary": True})
    configs.append({"algorithm": "modern", "max_length": 12, "word_boundary": True, "save_order": True})
    configs.append({"algorithm": "modern", "max_length": 12, "save_order": True})
    configs.append({"algorithm": "modern", "separator": "_", "max_length": 10})
    configs.append({"algorithm": "modern", "allow_unicode": True})
    configs.append({"algorithm": "modern", "lowercase": False})
    configs.append({"algorithm": "modern", "entities": False})
    configs.append({"algorithm": "modern", "decimal": False, "hexadecimal": False})
    configs.append({"algorithm": "modern", "replacements": [("a", "A")], "replacement_stage": "post"})
    configs.append({"algorithm": "modern", "stopwords": ["the", "a"]})
    configs.append({"algorithm": "modern", "regex_pattern": r"[^-a-zA-Z0-9_]+"})
    configs.append({"backend": "text-unidecode"})
    configs.append({"backend": "text-unidecode", "algorithm": "modern"})
    rng.shuffle(configs)
    return configs


def _build_cases() -> list[tuple[str, dict]]:
    rng = random.Random(20260920)
    configs = _config_matrix(random.Random(1729))
    cases: list[tuple[str, dict]] = []

    # Targeted snippets under the default config.
    for snippet in ENTITY_SNIPPETS + PUNCT_SNIPPETS + WORD_SNIPPETS:
        cases.append((snippet, {}))
        cases.append((f"prefix {snippet} suffix", {}))

    # Every named entity the stdlib knows, embedded in text.
    from html.entities import name2codepoint

    for name in sorted(name2codepoint):
        cases.append((f"a &{name}; z", {}))

    # Random texts under the default config.
    for _ in range(500):
        cases.append((_random_text(rng), {}))

    # Random texts under sampled configurations.
    for _ in range(700):
        cases.append((_random_text(rng), rng.choice(configs)))

    # Deterministic edge inputs.
    cases.extend(
        [
            ("", {}),
            (" ", {}),
            ("-", {}),
            ("--a--", {}),
            ("-a-b-", {"separator": "_"}),
            ("\x00", {}),
            ("\x00a\x00b\x00", {}),
            ("😀", {}),
            ("😀😀", {"separator": "_"}),
            ("￠", {}),  # U+FFE0
            ("￾", {}),  # U+FFFE (past-table)
            ("\U00010000", {}),  # first codepoint past the table
            ("﷐", {}),  # U+FDD0 noncharacter, in-table
            ("café", {"max_length": 3}),
            ("café", {"max_length": 4}),
            ("one two three", {"max_length": 8, "word_boundary": True}),
            ("one two three", {"max_length": 8, "word_boundary": True, "save_order": True}),
            ("one  two", {"max_length": 6, "word_boundary": True}),
            ("a" * 300, {"max_length": 100}),
            ("word " * 100, {"max_length": 50, "word_boundary": True}),
            (b"bytes input caf\xc3\xa9", {}),
            (b"invalid \xff\xfe bytes", {}),
            (bytearray(b"bytearray \xe2\x80\x94 dash"), {}),
            (b"\xed\xa0\x80 lone surrogate utf8", {}),  # decodes to surrogate via ignore? no
            ("lone \ud800 surrogate", {}),
            ("lone \udfff surrogate", {"allow_unicode": True}),
            ("&#55296;", {}),  # decimal surrogate ref (legacy keeps, modern drops)
            ("&#xD800;", {}),
            ("&#xD800;", {"algorithm": "modern"}),
            ("&#1114112;&#65;", {}),  # invalid first: legacy abandons both
            ("&#65;&#1114112;", {}),  # invalid second: legacy abandons both
            ("&#65;&#1114112;", {"algorithm": "modern"}),
            ("&#x41;&#x110000;", {}),
            ("&#x41;&#x110000;", {"algorithm": "modern"}),
            ("&#X41;", {}),  # uppercase X: legacy leaves literal
            ("&#X41;", {"algorithm": "modern"}),
            ("١٢٣", {}),  # Arabic-Indic digits
            ("&#١٢٣;", {"algorithm": "modern"}),  # Unicode-digit reference
            ("&#١٢٣;", {}),  # transliterated first: digits become ASCII
            ("&٤٢;", {}),
            ("ﬁle", {"allow_unicode": True}),
            ("①②③", {"max_length": 2}),
            ("＆＃６５；", {}),  # fullwidth &#65; -> NFKD -> decoded
            ("＆lt；", {}),  # fullwidth &lt; -> NFKD -> decoded
            # Codepoints whose table replacement is '&': entity decoding must
            # see the introduced '&' (fusion fast path must not skip it).
            ("ϗamp;", {}),  # U+03D7 -> '&' then 'amp;'
            ("ϗ#65;", {}),
            ("۽amp;", {}),
            ("⠯amp;", {}),
            ("﹠amp;", {}),
            ("＆amp;", {}),
            ("xϗamp;y", {}),
            ("ϗunknown;", {}),
            ("ϗ", {}),
            ("aϗb", {}),
            ("x" * 5000, {}),
            ("&amp;" * 200, {}),
        ]
    )
    return cases


CASES = _build_cases()


def _case_id(case: tuple[str, dict]) -> str:
    text, kwargs = case
    label = repr(text[:24]) if text else "''"
    if kwargs:
        label += "|" + ",".join(f"{k}={v!r}"[:18] for k, v in sorted(kwargs.items()))
    return label


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_slugify_matches_oracle(case):
    text, kwargs = case
    expected = oracle_slugify(text, **kwargs)
    got = slugify_mojo.slugify(text, **kwargs)
    assert got == expected, (
        f"slugify({text!r}, {kwargs!r}):\n  mojo   {got!r}\n  oracle {expected!r}"
    )
    assert type(got) is str


def test_corpus_is_substantial():
    assert len(CASES) >= 1500
    # Sanity: the corpus must actually exercise the table, not just ASCII.
    def has_non_ascii(text) -> bool:
        if isinstance(text, (bytes, bytearray)):
            return any(b > 127 for b in bytes(text))
        return any(ord(c) > 127 for c in text)

    assert sum(1 for text, _ in CASES if has_non_ascii(text)) > 800
    assert sum(1 for text, _ in CASES if "&" in str(text)) > 300


# --------------------------------------------------------------------------
# Argument validation parity (same exception type AND message)
# --------------------------------------------------------------------------

ERROR_CASES = [
    (123, {}),
    (None, {}),
    (["list"], {}),
    ("text", {"algorithm": "unknown"}),
    ("text", {"algorithm": "Legacy"}),
    ("text", {"replacement_stage": "neither"}),
    ("text", {"backend": "pyphen"}),
    ("text", {"algorithm": "modern", "max_length": True}),
    ("text", {"algorithm": "modern", "max_length": "10"}),
    ("text", {"algorithm": "modern", "separator": 5}),
    ("text", {"algorithm": "modern", "replacement_stage": "x"}),
    ("text", {"algorithm": "modern", "backend": "x"}),
]


@pytest.mark.parametrize("text,kwargs", ERROR_CASES)
def test_validation_errors_match_oracle(text, kwargs):
    oracle_exc = None
    try:
        oracle_slugify(text, **kwargs)
    except Exception as exc:  # noqa: BLE001
        oracle_exc = exc
    assert oracle_exc is not None, f"oracle accepted {text!r} {kwargs!r}"
    mojo_exc = None
    try:
        slugify_mojo.slugify(text, **kwargs)
    except Exception as exc:  # noqa: BLE001
        mojo_exc = exc
    assert mojo_exc is not None, f"slugify_mojo accepted {text!r} {kwargs!r}"
    assert type(mojo_exc) is type(oracle_exc)
    assert str(mojo_exc) == str(oracle_exc)


# --------------------------------------------------------------------------
# smart_truncate parity
# --------------------------------------------------------------------------

TRUNCATE_CASES = [
    ("one two three four", 0, False, " ", False),
    ("one two three four", 10, False, " ", False),
    ("one two three four", 10, True, " ", False),
    ("one two three four", 10, True, " ", True),
    ("one two three four", 11, True, " ", False),
    ("one-two-three-four", 10, True, "-", False),
    ("--one--two--", 6, True, "-", False),
    ("one two", -3, False, " ", False),
    ("", 5, False, " ", False),
    ("onewordonly", 4, True, " ", False),
    ("one two three", 3, True, " ", False),
    ("a b c d e", 5, True, " ", True),
]


@pytest.mark.parametrize("string,max_length,word_boundary,separator,save_order", TRUNCATE_CASES)
def test_smart_truncate_matches_oracle(string, max_length, word_boundary, separator, save_order):
    expected = oracle_truncate(string, max_length, word_boundary, separator, save_order)
    got = slugify_mojo.smart_truncate(string, max_length, word_boundary, separator, save_order)
    assert got == expected


# --------------------------------------------------------------------------
# slugify_column parity (batch fast path + per-item loop path)
# --------------------------------------------------------------------------

COLUMN_CONFIGS = [
    {},
    {"max_length": 12, "word_boundary": True},
    {"separator": "_"},
    {"lowercase": False},
    {"stopwords": ["the", "a"]},
    {"replacements": [("o", "0")], "replacement_stage": "post"},
    {"allow_unicode": True},  # loop path
    {"regex_pattern": r"[^-a-z0-9]+"},  # loop path
    {"algorithm": "modern"},  # loop path
    {"algorithm": "modern", "max_length": 10, "word_boundary": True},  # loop path
]


def _column_corpus() -> list[str]:
    rng = random.Random(4711)
    texts = [_random_text(rng) for _ in range(120)]
    texts += ["", "plain ascii", "Déjà Vu — Café", "日本語", "&#65;&amp;", "😀"]
    return texts


@pytest.mark.parametrize("kwargs", COLUMN_CONFIGS, ids=lambda k: str(k) or "default")
def test_slugify_column_matches_oracle(kwargs):
    texts = _column_corpus()
    expected = [oracle_slugify(t, **kwargs) for t in texts]
    got = slugify_mojo.slugify_column(texts, **kwargs)
    assert got == expected


def test_slugify_column_empty_and_bytes():
    assert slugify_mojo.slugify_column([]) == []
    texts = [b"caf\xc3\xa9 bytes", "plain", 42]
    expected = []
    for t in texts:
        try:
            expected.append(oracle_slugify(t))
        except TypeError:
            expected.append(TypeError)
    got = []
    for t in texts:
        try:
            got.append(slugify_mojo.slugify(t))
        except TypeError:
            got.append(TypeError)
    assert got == expected


def test_slugify_column_generator_input():
    texts = _column_corpus()[:20]
    got = slugify_mojo.slugify_column(iter(texts))
    assert got == [oracle_slugify(t) for t in texts]
