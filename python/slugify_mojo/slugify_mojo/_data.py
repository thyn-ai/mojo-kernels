"""Runtime loading of the transliteration and entity tables, plus the binary
blob handed to the native kernel.

The transliteration table is read at runtime from the installed
`text-unidecode` package's own data file (never re-derived, never vendored
into this repository), and the entity table is the standard library's
`html.entities.name2codepoint` (252 HTML4 names). Reading the installed
data keeps this package byte-identical to the oracle by construction and
license-clean: nothing is copied into the wheel.

The blob is a little-endian, 8-byte-aligned binary document. The
transliteration section is the package's raw data.bin payload (NUL-separated
ASCII entries, codepoints 1..n_trans); the kernel indexes it at create time,
so building the blob costs a header write plus one bytes copy:

    header: 16 x u64
        [0]  magic 0x534C4D4F4A4F3031 ("SLMOJO01")
        [1]  format version (1)
        [2]  n_trans         (transliteration entries; codepoints 1..n_trans)
        [3]  off_trans_bytes (raw data.bin payload)
        [4]  trans_bytes_len
        [5]  n_ent           (entity names)
        [6]  off_ent_name_off ((n_ent + 1) x u32 offsets into name bytes)
        [7]  off_ent_names   (concatenated entity names, sorted bytewise)
        [8]  ent_names_len
        [9]  off_ent_cps     (n_ent x u32 codepoints, parallel to names)
        [10..15] reserved (0)
"""

from __future__ import annotations

import struct
import threading
from html.entities import name2codepoint

BLOB_MAGIC = 0x534C4D4F4A4F3031
BLOB_VERSION = 1
_HDR_WORDS = 16


class SlugDataError(ModuleNotFoundError):
    """The installed text-unidecode package could not provide its table.

    Subclasses ModuleNotFoundError (with .name set) because that is exactly
    what the reference raises in this situation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "text_unidecode"


def _pad8(buf: bytearray) -> None:
    while len(buf) % 8 != 0:
        buf.append(0)


class SlugData:
    """Immutable snapshot of the transliteration + entity tables.

    ``raw`` is the data.bin payload: NUL-separated ASCII replacements for
    codepoints 1..n_trans (codepoints past the table are dropped by the
    pipeline, matching the reference's IndexError skip). ``replaces`` is the
    decoded list form, built lazily on first fallback use. ``entities`` is
    html.entities.name2codepoint in its original dict order.
    ``amp_cps`` is the set of codepoints whose replacement contains '&'
    (the fused fast path must not skip entity decoding for them), and
    ``fusion_safe`` certifies that among ASCII inputs only U+0026 itself can
    produce a '&'.
    """

    __slots__ = (
        "raw",
        "entities",
        "amp_cps",
        "amp_chars",
        "fusion_safe",
        "_replaces",
        "_blob",
    )

    def __init__(self, raw: bytes, entities: dict[str, int]) -> None:
        self.raw = raw
        self.entities = entities
        self.amp_cps = _amp_cps(raw)
        self.fusion_safe = all(cp == 0x26 for cp in self.amp_cps if cp <= 0x7F)
        # Non-ASCII codepoints that transliterate to '&': the fused fast path
        # scans for these before skipping entity decoding.
        self.amp_chars = "".join(chr(cp) for cp in sorted(self.amp_cps) if cp > 0x7F)
        self._replaces: list[str] | None = None
        self._blob: bytes | None = None

    @property
    def n_trans(self) -> int:
        # The payload joins n_trans entries with n_trans - 1 NUL separators
        # (split('\x00') yields count + 1 items); UTF-8 never emits a NUL
        # byte inside a multi-byte sequence, so byte counting is exact.
        return self.raw.count(b"\x00") + 1

    @property
    def replaces(self) -> list[str]:
        """Decoded per-codepoint replacements (fallback engine; lazy)."""
        if self._replaces is None:
            self._replaces = self.raw.decode("utf-8").split("\x00")
        return self._replaces

    def blob(self) -> bytes:
        if self._blob is None:
            self._blob = _build_blob(self)
        return self._blob


def _amp_cps(raw: bytes) -> frozenset:
    """Codepoints whose table replacement contains '&'. Entry i (0-based)
    maps codepoint i+1; a NUL scan locates the entry of each '&' byte. The
    table has a handful of these, so the loop is a few iterations."""
    cps: set[int] = set()
    start = 0
    while True:
        pos = raw.find(b"&", start)
        if pos < 0:
            return frozenset(cps)
        cps.add(raw.count(b"\x00", 0, pos) + 1)
        start = pos + 1


def load_data() -> SlugData:
    """Load the tables from the installed text-unidecode package + stdlib."""
    try:
        import pkgutil

        raw = pkgutil.get_data("text_unidecode", "data.bin")
    except (ImportError, FileNotFoundError) as exc:
        raise SlugDataError(
            "the 'text-unidecode' package is required at runtime for its "
            "shipped transliteration table; install it "
            "(pip install text-unidecode)"
        ) from exc
    if raw is None:
        raise SlugDataError("text-unidecode data.bin not found in the installed package")
    if not raw or raw.count(b"\x00") == 0:
        raise SlugDataError("text-unidecode data.bin has an unexpected shape")
    return SlugData(raw, dict(name2codepoint))


def _build_blob(data: SlugData) -> bytes:
    body = bytearray(_HDR_WORDS * 8)  # header placeholder; patched at the end

    off_trans_bytes = len(body)
    body += data.raw

    # entity names sorted bytewise (all ASCII), parallel codepoint array
    items = sorted(data.entities.items(), key=lambda kv: kv[0].encode("ascii"))
    ent_names = bytearray()
    ent_name_off: list[int] = [0]
    ent_cps: list[int] = []
    for name, cp in items:
        ent_names += name.encode("ascii")
        ent_name_off.append(len(ent_names))
        ent_cps.append(cp)

    _pad8(body)
    off_ent_name_off = len(body)
    body += struct.pack(f"<{len(ent_name_off)}I", *ent_name_off)
    _pad8(body)
    off_ent_names = len(body)
    body += ent_names
    _pad8(body)
    off_ent_cps = len(body)
    if ent_cps:
        body += struct.pack(f"<{len(ent_cps)}I", *ent_cps)

    header = struct.pack(
        "<16Q",
        BLOB_MAGIC,
        BLOB_VERSION,
        data.n_trans,
        off_trans_bytes,
        len(data.raw),
        len(items),
        off_ent_name_off,
        off_ent_names,
        len(ent_names),
        off_ent_cps,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    body[: _HDR_WORDS * 8] = header
    return bytes(body)


_LOCK = threading.Lock()
_DATA: SlugData | None = None


def get_data() -> SlugData:
    """Process-wide cached snapshot; the installed table never changes."""
    global _DATA
    if _DATA is not None:
        return _DATA
    with _LOCK:
        if _DATA is None:
            _DATA = load_data()
        return _DATA
