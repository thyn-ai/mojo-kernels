"""bio_mojo: FASTA/GenBank parsers matching Bio.SeqIO, powered by a Mojo kernel.

Clean-room parsers for the FASTA format and a GenBank subset (LOCUS,
DEFINITION, ACCESSION, VERSION, FEATURES locations+qualifiers, ORIGIN),
byte-exact with ``Bio.SeqIO.parse`` on the compared fields — powered by a
Mojo kernel where the platform supports it (macOS arm64, Linux x86_64),
with a vendored pure-Python fallback everywhere else (including Windows).

    import bio_mojo

    for record in bio_mojo.parse_fasta("reads.fa"):
        print(record.id, len(record.seq))

    for record in bio_mojo.parse_genbank("genome.gbk"):
        print(record.id, len(record.features))

Set BIO_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback; inspect
the active backend with bio_mojo.backend_info().
"""

from bio_mojo._native import backend_info, native_available
from bio_mojo.core import (
    CompoundLocation,
    FastaRecord,
    Feature,
    GenBankRecord,
    SimpleLocation,
    backend,
    parse_fasta,
    parse_genbank,
)
from bio_mojo.errors import BioMojoError, ParseError, StreamModeError

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "parse_fasta",
    "parse_genbank",
    "FastaRecord",
    "GenBankRecord",
    "Feature",
    "SimpleLocation",
    "CompoundLocation",
    "BioMojoError",
    "ParseError",
    "StreamModeError",
    "backend",
    "backend_info",
    "native_available",
    "__version__",
]
