"""Runtime loading of language profiles and data tables from the installed
`langdetect` package, plus the binary blob handed to the native kernel.

Profiles are read from the installed reference package at runtime (never
re-derived, never vendored into this repository), in the same directory order
and with the same probability convention as the reference factory:
``prob = 1.0 * freq / n_words[len(word) - 1]`` per 1..3-gram.

The blob is a little-endian, 8-byte-aligned binary document:

    header: 20 x u64
        [0]  magic 0x4C444D4F4A4F3031 ("LDMOJO01")
        [1]  format version (1)
        [2]  n_langs        [3]  n_keys        [4]  n_entries
        [5]  off_langs      (n_langs x (u64 bytelen + utf8 bytes, padded to 8))
        [6]  off_keys       (n_keys x u64 packed n-grams, strictly ascending)
        [7]  off_kid_off    ((n_keys + 1) x u64 CSR offsets into entries)
        [8]  off_entries    (n_entries x 16B: u32 lang idx, u32 pad, f64 prob)
        [9]  off_cjk        [10] n_cjk      (pairs u32 from, u32 to; sorted)
        [11] off_latin1     [12] n_latin1   (u32 codepoints, padded to 8)
        [13] off_vi_alpha   [14] n_vi_alpha (u32 codepoints)
        [15] off_vi_dmark   [16] n_vi_dmark (u32 codepoints)
        [17] off_vi_rows    (5 * n_vi_alpha u32, row-major per diacritic)
        [18] off_upper      [19] n_upper    (pairs u32 lo, u32 hi; sorted)

An n-gram of 1..3 codepoints is packed into one u64 as
``cp0 | cp1 << 21 | cp2 << 42`` (codepoints are < 2**21), which the kernel
reproduces on the normalized text window; the same packing is applied here to
profile keys, so membership is an exact integer comparison.
"""

from __future__ import annotations

import json
import os
import struct
import threading

from langdetect_mojo._uppercase_table import UPPER_RANGES

BLOB_MAGIC = 0x4C444D4F4A4F3031
BLOB_VERSION = 1
_HDR_WORDS = 20


class ProfileLoadError(RuntimeError):
    """The installed langdetect package could not provide profiles/tables."""


def _pack_key(word: str) -> int:
    key = 0
    for i, ch in enumerate(word):
        key |= ord(ch) << (21 * i)
    return key


class Profiles:
    """Immutable snapshot of the reference profiles + normalization tables.

    ``langlist`` preserves the reference factory's load order (os.listdir of
    the profiles directory, dotfiles and non-files skipped); the per-n-gram
    sparse probability entries reference language indexes in that order.
    """

    __slots__ = (
        "langlist",
        "keys",
        "kid_off",
        "ent_lang",
        "ent_prob",
        "key_to_kid",
        "cjk_pairs",
        "latin1",
        "vi_alpha",
        "vi_dmark",
        "vi_rows",
        "_blob",
    )

    def __init__(self) -> None:
        self.langlist: list[str] = []
        self.keys: list[int] = []  # sorted packed keys
        self.kid_off: list[int] = []  # CSR offsets, len(keys) + 1
        self.ent_lang: list[int] = []
        self.ent_prob: list[float] = []
        self.key_to_kid: dict[int, int] = {}
        self.cjk_pairs: list[tuple[int, int]] = []
        self.latin1: list[int] = []
        self.vi_alpha: list[int] = []
        self.vi_dmark: list[int] = []
        self.vi_rows: list[int] = []
        self._blob: bytes | None = None

    @property
    def n_langs(self) -> int:
        return len(self.langlist)

    def blob(self) -> bytes:
        if self._blob is None:
            self._blob = _build_blob(self)
        return self._blob


def load_profiles() -> Profiles:
    """Load profiles and tables from the installed langdetect package."""
    try:
        import langdetect
    except ImportError as exc:
        raise ProfileLoadError(
            "the 'langdetect' package is required at runtime for its shipped "
            "55-language profiles; install it (pip install langdetect)"
        ) from exc
    from langdetect.utils.ngram import NGram  # data tables (read, not executed)

    profiles_dir = os.path.join(os.path.dirname(langdetect.__file__), "profiles")
    if not os.path.isdir(profiles_dir):
        raise ProfileLoadError(f"profiles directory not found: {profiles_dir}")

    out = Profiles()
    vectors: dict[int, dict[int, float]] = {}
    # Same enumeration as the reference DetectorFactory.load_profile.
    for filename in os.listdir(profiles_dir):
        if filename.startswith("."):
            continue
        full = os.path.join(profiles_dir, filename)
        if not os.path.isfile(full):
            continue
        with open(full, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        name = data["name"]
        if name in out.langlist:
            raise ProfileLoadError(f"duplicate language profile: {name}")
        index = len(out.langlist)
        out.langlist.append(name)
        n_words = data["n_words"]
        for word, freq in data["freq"].items():
            length = len(word)
            if 1 <= length <= 3:
                prob = 1.0 * freq / n_words[length - 1]
                vectors.setdefault(_pack_key(word), {})[index] = prob

    keys = sorted(vectors)
    out.keys = keys
    out.key_to_kid = {k: i for i, k in enumerate(keys)}
    out.kid_off = [0] * (len(keys) + 1)
    for kid, key in enumerate(keys):
        entries = vectors[key]
        for lang in sorted(entries):
            out.ent_lang.append(lang)
            out.ent_prob.append(entries[lang])
        out.kid_off[kid + 1] = len(out.ent_lang)

    # Normalization tables shipped with the reference package.
    out.cjk_pairs = sorted((ord(k), ord(v)) for k, v in NGram.CJK_MAP.items())
    out.latin1 = [ord(c) for c in NGram.LATIN1_EXCLUDED]
    out.vi_alpha = [ord(c) for c in NGram.TO_NORMALIZE_VI_CHARS]
    out.vi_dmark = [ord(c) for c in NGram.DMARK_CLASS]
    n_alpha = len(out.vi_alpha)
    if len(NGram.NORMALIZED_VI_CHARS) != 5 or any(
        len(row) != n_alpha for row in NGram.NORMALIZED_VI_CHARS
    ):
        raise ProfileLoadError("unexpected Vietnamese normalization table shape")
    for row in NGram.NORMALIZED_VI_CHARS:
        out.vi_rows.extend(ord(c) for c in row)
    return out


def _pad8(buf: bytearray) -> None:
    while len(buf) % 8 != 0:
        buf.append(0)


def _build_blob(p: Profiles) -> bytes:
    body = bytearray(_HDR_WORDS * 8)  # header placeholder; patched at the end
    offs: list[int] = []

    def begin_section() -> int:
        _pad8(body)
        offs.append(len(body))
        return len(body)

    # langs: n_langs x (u64 byte length + utf8 bytes, padded)
    begin_section()
    for name in p.langlist:
        raw = name.encode("utf-8")
        body += struct.pack("<Q", len(raw))
        body += raw
        _pad8(body)

    begin_section()
    body += struct.pack(f"<{len(p.keys)}Q", *p.keys)
    begin_section()
    body += struct.pack(f"<{len(p.kid_off)}Q", *p.kid_off)
    begin_section()
    ent = bytearray()
    for lang, prob in zip(p.ent_lang, p.ent_prob):
        ent += struct.pack("<IId", lang, 0, prob)
    body += ent
    begin_section()
    for frm, to in p.cjk_pairs:
        body += struct.pack("<II", frm, to)
    begin_section()
    for cp in p.latin1:
        body += struct.pack("<I", cp)
    begin_section()
    for cp in p.vi_alpha:
        body += struct.pack("<I", cp)
    begin_section()
    for cp in p.vi_dmark:
        body += struct.pack("<I", cp)
    begin_section()
    for cp in p.vi_rows:
        body += struct.pack("<I", cp)
    begin_section()
    for lo, hi in UPPER_RANGES:
        body += struct.pack("<II", lo, hi)
    _pad8(body)

    struct.pack_into(
        "<20Q",
        body,
        0,
        BLOB_MAGIC,
        BLOB_VERSION,
        p.n_langs,
        len(p.keys),
        len(p.ent_lang),
        offs[0],  # langs
        offs[1],  # keys
        offs[2],  # kid_off
        offs[3],  # entries
        offs[4],
        len(p.cjk_pairs),
        offs[5],
        len(p.latin1),
        offs[6],
        len(p.vi_alpha),
        offs[7],
        len(p.vi_dmark),
        offs[8],  # vi_rows
        offs[9],
        len(UPPER_RANGES),
    )
    return bytes(body)


_LOCK = threading.Lock()
_CACHED: Profiles | None = None


def get_profiles() -> Profiles:
    """Process-wide singleton (profiles are read once per process)."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    with _LOCK:
        if _CACHED is None:
            _CACHED = load_profiles()
    return _CACHED
