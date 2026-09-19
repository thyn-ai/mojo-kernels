"""Public parse API: ``parse_fasta`` / ``parse_genbank``.

Both functions accept a path (``str``/``os.PathLike``) or a text-mode file
object, scan the whole input on the active backend — the native Mojo kernel
when its shared library is available (macOS arm64 / Linux x86_64 wheels),
otherwise the vendored pure-Python parser — and yield records that are equal
to ``Bio.SeqIO.parse(source, "fasta"|"genbank")`` records on the compared
fields (``id``, ``name``, ``description``, ``str(seq)``, feature count,
``str(feature.location)``, ``feature.qualifiers``).

Both backends produce the same raw byte structures (see
:mod:`bio_mojo._raw`), so every normalization rule below applies identically
regardless of backend; the differential suite asserts oracle equality on
both paths. Set ``BIO_MOJO_DISABLE_NATIVE=1`` to force the fallback.
"""

from __future__ import annotations

import os
from typing import Iterator

from bio_mojo import _reference
from bio_mojo._location import parse_location
from bio_mojo._native import NativeUnavailable, _load
from bio_mojo._raw import RawGbRecord
from bio_mojo.errors import StreamModeError
from bio_mojo.records import (
    CompoundLocation,
    FastaRecord,
    Feature,
    GenBankRecord,
    SimpleLocation,
)

__all__ = [
    "parse_fasta",
    "parse_genbank",
    "FastaRecord",
    "GenBankRecord",
    "Feature",
    "SimpleLocation",
    "CompoundLocation",
]


def _read_source(source, format_label: str) -> bytes:
    """Read the whole input as bytes.

    Paths are read in binary (the parser implements universal newlines
    itself). File objects must be opened in text mode — mirroring the
    oracle, which raises ``Bio.StreamModeError`` for binary handles on both
    formats.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            return fh.read()
    read = getattr(source, "read", None)
    if read is None:
        raise TypeError(
            f"expected a path (str/os.PathLike) or a text-mode file object, "
            f"got {type(source).__name__}"
        )
    data = read()
    if isinstance(data, str):
        return data.encode("utf-8")
    raise StreamModeError(f"{format_label} files must be opened in text mode.")


def _native_backend():
    """The native parse module if usable right now, else None."""
    try:
        _load()
    except NativeUnavailable:
        return None
    from bio_mojo import _native

    return _native


def backend() -> str:
    """Which backend serves parses right now: "native" or "fallback"."""
    return "native" if _native_backend() is not None else "fallback"


def _fasta_record(title_b: bytes, seq_b: bytes) -> FastaRecord:
    # The oracle reads text-mode handles: the title is decoded text, then
    # rstripped at str level (which also removes non-ASCII whitespace), and
    # id is the first whitespace-split token. The sequence is assembled as
    # bytes and must be ASCII (the oracle raises UnicodeDecodeError for
    # non-ASCII sequence content, at the same offset in the same string).
    title = title_b.decode("utf-8").rstrip()
    tokens = title.split(None, 1)
    rec_id = tokens[0] if tokens else ""
    seq = seq_b.decode("ascii")
    return FastaRecord(id=rec_id, name=rec_id, description=title, seq=seq)


def _unquote(raw_value: bytes, quoted: bool) -> str:
    """Undo GenBank value quoting: strip outer quotes, '""' -> '"'."""
    value = raw_value
    if quoted and len(value) >= 1 and value[:1] == b'"':
        value = value[1:]
        if value.endswith(b'"'):
            value = value[:-1]
    if quoted:
        value = value.replace(b'""', b'"')
    return value.decode("utf-8")


def _genbank_record(raw: RawGbRecord) -> GenBankRecord:
    name = raw.name.decode("utf-8")
    # DEFINITION lines were space-joined by the backend; the oracle strips
    # exactly one trailing period from the assembled definition.
    description = raw.definition.decode("utf-8")
    if description.endswith("."):
        description = description[:-1]
    # id rule (oracle): VERSION's accession.version token, else the first
    # ACCESSION, else the LOCUS name.
    accession = raw.accession.decode("utf-8").split()[0] if raw.accession.split() else ""
    version = raw.version.decode("utf-8").split()[0] if raw.version.split() else ""
    rec_id = version or accession or name
    seq = raw.seq.decode("ascii")
    features: list[Feature] = []
    for raw_feat in raw.features:
        qualifiers: dict[str, list[str]] = {}
        for qual in raw_feat.qualifiers:
            key = qual.name.decode("utf-8")
            if qual.has_value:
                qualifiers.setdefault(key, []).append(
                    _unquote(qual.value, qual.quoted)
                )
            else:
                # Bare '/flag' qualifiers are idempotent in the oracle:
                # repeats do not extend the [''] value list.
                qualifiers.setdefault(key, [""])
        loc_text = raw_feat.location.decode("ascii").translate(
            {0x20: None, 0x09: None}
        )
        features.append(
            Feature(
                type=raw_feat.key.decode("utf-8"),
                location=parse_location(loc_text),
                qualifiers=qualifiers,
            )
        )
    return GenBankRecord(
        id=rec_id, name=name, description=description, seq=seq, features=features
    )


def parse_fasta(source) -> Iterator[FastaRecord]:
    """Yield FASTA records equal to ``Bio.SeqIO.parse(source, "fasta")``.

    Laziness matches the oracle: input is consumed and validated on the
    first ``next()`` call, and parse errors surface there, not at the
    ``parse_fasta`` call itself.
    """
    data = _read_source(source, "Fasta")
    native = _native_backend()
    if native is not None:
        raw = native.fasta_parse_bytes(data)
    else:
        raw = _reference.fasta_parse_bytes(data)
    titles, title_offs, seqs, seq_offs = raw
    # Tight per-record loop (this is the hot path for >100k-record corpora):
    # the title is decoded and rstripped at str level and the id is the first
    # whitespace-split token (oracle rules); the sequence must decode as
    # ASCII (the oracle raises UnicodeDecodeError otherwise, at the same
    # offset in the same string).
    record_cls = FastaRecord
    for i in range(len(title_offs) - 1):
        title = titles[title_offs[i] : title_offs[i + 1]].decode("utf-8").rstrip()
        tokens = title.split(None, 1)
        rec_id = tokens[0] if tokens else ""
        seq = seqs[seq_offs[i] : seq_offs[i + 1]].decode("ascii")
        yield record_cls(id=rec_id, name=rec_id, description=title, seq=seq)


def parse_genbank(source) -> Iterator[GenBankRecord]:
    """Yield GenBank records equal to ``Bio.SeqIO.parse(source, "genbank")``
    on the supported subset (LOCUS/DEFINITION/ACCESSION/VERSION/FEATURES/
    ORIGIN)."""
    data = _read_source(source, "GenBank")
    native = _native_backend()
    if native is not None:
        raw_records = native.gb_parse_bytes(data)
    else:
        raw_records = _reference.gb_parse_bytes(data)
    for raw in raw_records:
        yield _genbank_record(raw)
