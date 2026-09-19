"""Public record types returned by ``bio_mojo.parse_fasta``/``parse_genbank``.

These are intentionally lightweight value objects (``__slots__``, plain
``str`` sequences) that carry exactly the fields the differential suite
compares against ``Bio.SeqIO`` records: ``id``, ``name``, ``description``,
``seq`` and, for GenBank, ``features`` with ``type``, ``location`` and
``qualifiers``.

``str(location)`` reproduces the oracle's normalization byte-for-byte for
the supported location subset, e.g.::

    123..456                    -> [122:456](+)
    complement(123..456)        -> [122:456](-)
    <123..>456                  -> [<122:>456](+)
    join(1..10,20..30)          -> join{[0:10](+), [19:30](+)}
    complement(join(1..10,...)) -> join{[19:30](-), [0:10](-)}
    50^51                       -> [50:50](+)

Locations outside the supported subset (``one-of``, non-adjacent ``a^b``,
nested compound operators, reversed ranges, unparseable strings) compare as
``None`` — the same value the oracle assigns to ``feature.location``.
"""

from __future__ import annotations


class FastaRecord:
    """One FASTA record. ``seq`` is a plain ``str`` (ASCII)."""

    __slots__ = ("id", "name", "description", "seq")

    def __init__(self, id: str, name: str, description: str, seq: str):
        self.id = id
        self.name = name
        self.description = description
        self.seq = seq

    def __repr__(self) -> str:
        return (
            f"FastaRecord(id={self.id!r}, description={self.description!r}, "
            f"seq=({len(self.seq)} letters))"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FastaRecord):
            return NotImplemented
        return (
            self.id == other.id
            and self.name == other.name
            and self.description == other.description
            and self.seq == other.seq
        )


class SimpleLocation:
    """One contiguous range, already normalized to 0-based half-open bounds.

    ``start_fuzz``/``end_fuzz`` are ``''``, ``'<'`` or ``'>'``; ``strand`` is
    1 or -1; ``accession`` is set for remote locations (``J00194.1:100..200``).
    """

    __slots__ = ("start", "start_fuzz", "end", "end_fuzz", "strand", "accession")

    def __init__(
        self,
        start: int,
        end: int,
        strand: int,
        start_fuzz: str = "",
        end_fuzz: str = "",
        accession: str | None = None,
    ):
        self.start = start
        self.end = end
        self.strand = strand
        self.start_fuzz = start_fuzz
        self.end_fuzz = end_fuzz
        self.accession = accession

    def __str__(self) -> str:
        prefix = self.accession or ""
        strand = {1: "(+)", -1: "(-)"}.get(self.strand, "")
        return (
            f"{prefix}[{self.start_fuzz}{self.start}:"
            f"{self.end_fuzz}{self.end}]{strand}"
        )

    def __repr__(self) -> str:
        return f"SimpleLocation({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleLocation):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.strand == other.strand
            and self.start_fuzz == other.start_fuzz
            and self.end_fuzz == other.end_fuzz
            and self.accession == other.accession
        )


class CompoundLocation:
    """A ``join{...}`` or ``order{...}`` of simple parts (flat, normalized)."""

    __slots__ = ("operator", "parts")

    def __init__(self, operator: str, parts: tuple[SimpleLocation, ...]):
        self.operator = operator
        self.parts = parts

    def __str__(self) -> str:
        inner = ", ".join(str(p) for p in self.parts)
        return f"{self.operator}{{{inner}}}"

    def __repr__(self) -> str:
        return f"CompoundLocation({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompoundLocation):
            return NotImplemented
        return self.operator == other.operator and self.parts == other.parts


class Feature:
    """One GenBank feature-table entry.

    ``location`` is ``SimpleLocation``/``CompoundLocation`` for the supported
    subset, or ``None`` for locations the oracle also leaves unparsed.
    ``qualifiers`` maps names to ordered lists of values (flags map to
    ``['']``), exactly like the oracle's ``feature.qualifiers``.
    """

    __slots__ = ("type", "location", "qualifiers")

    def __init__(
        self,
        type: str,
        location: SimpleLocation | CompoundLocation | None,
        qualifiers: dict[str, list[str]],
    ):
        self.type = type
        self.location = location
        self.qualifiers = qualifiers

    def __repr__(self) -> str:
        return f"Feature(type={self.type!r}, location={self.location!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return NotImplemented
        return (
            self.type == other.type
            and self.location == other.location
            and self.qualifiers == other.qualifiers
        )


class GenBankRecord:
    """One GenBank record (LOCUS/DEFINITION/ACCESSION/VERSION/FEATURES/ORIGIN)."""

    __slots__ = ("id", "name", "description", "seq", "features")

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        seq: str,
        features: list[Feature],
    ):
        self.id = id
        self.name = name
        self.description = description
        self.seq = seq
        self.features = features

    def __repr__(self) -> str:
        return (
            f"GenBankRecord(id={self.id!r}, name={self.name!r}, "
            f"seq=({len(self.seq)} letters), features={len(self.features)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenBankRecord):
            return NotImplemented
        return (
            self.id == other.id
            and self.name == other.name
            and self.description == other.description
            and self.seq == other.seq
            and self.features == other.features
        )
