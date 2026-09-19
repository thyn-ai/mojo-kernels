"""bio_mojo end-user quickstart (also the wheel smoke test).

Parses embedded FASTA and GenBank samples and prints deterministic
checksums; CI runs this once on the native backend and once with
BIO_MOJO_DISABLE_NATIVE=1 and compares the checksums. Works from an
installed wheel with no repository and no biopython install.
"""

from __future__ import annotations

import bio_mojo

FASTA_SAMPLE = """\
>seq0001 first seeded record
ACGTUNRYMK
acgtun.*-x

>seq0002 second record with  trailing spaces  
TT TT
GG\tGG
>seq0003
CC
"""

GENBANK_SAMPLE = """\
LOCUS       DEMO1         60 bp    DNA             PLN       01-JAN-2000
DEFINITION  Demonstration record for the bio_mojo quickstart.
ACCESSION   DM000001
VERSION     DM000001.1  GI:4242
FEATURES             Location/Qualifiers
     source          1..60
                     /organism="Syntheticus constructus"
                     /mol_type="genomic DNA"
     gene            complement(<10..>25)
                     /gene="demo"
     CDS             join(10..15,complement(20..25),40..45)
                     /gene="demo"
                     /translation="MKLA
                     TWRVVA"
ORIGIN
        1 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
//
"""


def main() -> None:
    info = bio_mojo.backend_info()
    print(f"bio_mojo {bio_mojo.__version__} backend: "
          f"{'native' if info['native_available'] else 'fallback'}")

    fasta = list(bio_mojo.parse_fasta(_as_text_handle(FASTA_SAMPLE)))
    print(f"fasta records: {len(fasta)}")
    for r in fasta:
        print(f"  id={r.id!r} desc={r.description!r} seq={r.seq!r}")

    gb = list(bio_mojo.parse_genbank(_as_text_handle(GENBANK_SAMPLE)))
    rec = gb[0]
    print(f"genbank: id={rec.id!r} name={rec.name!r} desc={rec.description!r}")
    print(f"  seq={rec.seq!r}")
    for f in rec.features:
        print(f"  {f.type} {f.location} {f.qualifiers}")

    # Deterministic checksum for the CI native-vs-fallback comparison.
    checksum = sum(len(r.seq) for r in fasta) + sum(len(r.seq) for r in gb)
    loc_checksum = ";".join(str(f.location) for r in gb for f in r.features)
    print(f"checksum: {checksum} locations: {loc_checksum}")


def _as_text_handle(text: str):
    import io

    return io.StringIO(text)


if __name__ == "__main__":
    main()
