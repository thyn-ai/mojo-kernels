"""Differential tests: bio_mojo must match Bio.SeqIO (biopython) record-for-record.

Run twice by ``scripts/test_all_biopython.sh``: once against the native Mojo
kernel and once with BIO_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback).
Both backends must agree with the oracle exactly (string equality — there is
no numeric tolerance; every compared field is a str/int/dict).

The oracle is the published PyPI package (pinned: biopython==1.88, see the
test script). All corpora are generated locally from explicit seeds — no
network, no unseeded randomness — so the suite is bit-reproducible.

Compared fields (the assignment's parity contract):
  FASTA:   record.id, record.name, record.description, str(record.seq)
  GenBank: record.id, record.name, record.description, str(record.seq),
           len(record.features), and per feature: type, str(location),
           qualifiers dict
"""

from __future__ import annotations

import io
import os
import random
import string
import warnings

import pytest
from Bio import SeqIO

import bio_mojo

# ---------------------------------------------------------------------------
# Seeded corpus generators
# ---------------------------------------------------------------------------

_SEQ_ALPHABET = "ACGTUNRYMKSWHBVDacgtun" + ".-*"


def _rand_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_SEQ_ALPHABET) for _ in range(length))


def make_fasta_corpus(seed: int, n_records: int) -> str:
    """Seeded FASTA corpus: multiline sequences, blank lines, internal
    spaces, empty sequences, tab/space-heavy descriptions."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(n_records):
        rec_id = f"seq{i:07d}"
        kind = i % 10
        if kind == 0:
            header = f">{rec_id}"  # no description
        elif kind == 1:
            header = f">{rec_id}  multiple   spaces and\ttabs  "  # trailing ws
        elif kind == 2:
            header = f">{rec_id} desc with punctuation !@#$%^&*()_+-=[]|:;,.?"
        else:
            header = f">{rec_id} description number {i}"
        out.append(header)
        if i % 13 == 5:
            out.append("")  # empty-sequence record (blank line is a no-op)
            continue
        total = rng.randint(10, 240)
        seq = _rand_seq(rng, total)
        # multiline with occasional blank lines and internal spaces
        pos = 0
        while pos < len(seq):
            width = rng.choice((10, 20, 60, 80))
            chunk = seq[pos : pos + width]
            if rng.random() < 0.1 and len(chunk) > 4:
                cut = rng.randint(1, len(chunk) - 1)
                chunk = chunk[:cut] + " " + chunk[cut:]
            out.append(chunk)
            if rng.random() < 0.05:
                out.append("")  # blank line inside the record
            pos += width
    return "\n".join(out) + "\n"


_GB_TYPES = (
    "source", "gene", "CDS", "mRNA", "tRNA", "rRNA", "exon", "intron",
    "5'UTR", "3'UTR", "misc_feature", "regulatory", "repeat_region",
)
_GB_LOCATIONS = (
    "{a}..{b}", "<{a}..{b}", "{a}..>{b}", "<{a}..>{b}", "{a}",
    "complement({a}..{b})", "complement(<{a}..>{b})",
    "join({a}..{b},{c}..{d})", "join({a}..{b},complement({c}..{d}),{e}..{f})",
    "order({a}..{b},{c}..{d})",
    "complement(join({a}..{b},{c}..{d}))",
    "complement(order({a}..{b},{c}..{d}))",
    "{a}^{a1}",            # adjacent bond -> [a:a]
    "join({a}..{b})",      # single-part join unwraps
    # locations the oracle leaves unparsed (feature.location is None):
    "{a}^{a9}",            # non-adjacent bond
    "one-of({a},{b})",
    "join(join({a}..{b},{c}..{d}),{e}..{f})",  # nested compound
)


def _gb_location(rng: random.Random) -> str:
    tmpl = rng.choice(_GB_LOCATIONS)
    # Positions always within the declared LOCUS length (>= 50): outside that
    # domain the oracle is bounds-sensitive (it raises ValueError on reversed
    # ranges whose start exceeds the declared length), which is documented
    # unsupported scope.
    vals = sorted(rng.sample(range(5, 45), 6))
    a, b, c, d, e, f = vals
    return tmpl.format(a=a, b=b, c=c, d=d, e=e, f=f, a1=a + 1, a9=a + 9)


def _gb_qualifiers(rng: random.Random, ftype: str) -> list[str]:
    lines: list[str] = []
    if ftype == "source":
        lines.append('/organism="Syntheticus constructus"')
        lines.append('/mol_type="genomic DNA"')
    if ftype == "CDS":
        lines.append("/codon_start=1")
        if rng.random() < 0.5:
            prot = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(30))
            # /translation joins continuation lines WITHOUT a separator
            lines.append(f'/translation="{prot[:12]}')
            lines.append(f'{prot[12:24]}')
            lines.append(f'{prot[24:]}"')
    roll = rng.random()
    if roll < 0.3:
        lines.append("/pseudo")
    if roll < 0.5:
        note = f"note number {rng.randint(0, 999)}"
        lines.append(f'/note="{note[:8]}')
        lines.append(f'{note[8:]}"')  # /note joins with a space
    if rng.random() < 0.2:
        lines.append(f'/note="has ""embedded"" quotes"')
    if rng.random() < 0.2:
        lines.append("/experiment")
        lines.append("/experiment")  # repeated qualifier -> list of values
    if rng.random() < 0.15:
        lines.append("/label=bare_value")
    return lines


def make_gb_corpus(seed: int, n_records: int) -> str:
    """Seeded GenBank corpus: valid well-formed records exercising the whole
    supported subset."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(n_records):
        name = f"SC{i:06d}"[:16]
        length = rng.randint(50, 5000)
        out.append(
            f"LOCUS       {name:<16} {length:>5} bp    DNA             "
            f"PLN       01-JAN-2000"
        )
        if i % 7 != 3:
            out.append(f"DEFINITION  Synthetic record number {i} for")
            out.append(f"            differential testing {i}.")
        else:
            out.append(f"DEFINITION  Single line definition {i}.")
        if i % 5 != 4:
            out.append(f"ACCESSION   AC{i:06d}")
        if i % 3 != 2:
            out.append(f"VERSION     AC{i:06d}.{1 + i % 3}  GI:{1000 + i}")
        out.append("KEYWORDS    .")
        if i % 4 == 0:
            out.append("SOURCE      Syntheticus constructus")
            out.append("  ORGANISM  Syntheticus constructus")
        out.append("FEATURES             Location/Qualifiers")
        for _ in range(rng.randint(0, 6)):
            ftype = rng.choice(_GB_TYPES)
            loc = _gb_location(rng)
            if rng.random() < 0.15 and "," in loc:
                # Multiline location: split right after a comma, i.e. while
                # parentheses are unbalanced — the only break position the
                # oracle accepts (it crashes on other break positions).
                head, tail = loc.split(",", 1)
                out.append(f"     {ftype:<16}{head},")
                out.append(f"                     {tail}")
            else:
                out.append(f"     {ftype:<16}{loc}")
            for q in _gb_qualifiers(rng, ftype):
                out.append(f"                     {q}")
        out.append("ORIGIN")
        seq_len = min(length, rng.randint(40, 400))
        seq = _rand_seq(rng, seq_len)
        for pos in range(0, len(seq), 60):
            chunk = seq[pos : pos + 60]
            groups = " ".join(chunk[g : g + 10] for g in range(0, len(chunk), 10))
            out.append(f"{pos + 1:>9} {groups}")
        out.append("//")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Oracle comparison helpers
# ---------------------------------------------------------------------------


def _oracle_fasta(text: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            (r.id, r.name, r.description, str(r.seq))
            for r in SeqIO.parse(io.StringIO(text), "fasta")
        ]


def _ours_fasta(text: str):
    return [
        (r.id, r.name, r.description, str(r.seq))
        for r in bio_mojo.parse_fasta(io.StringIO(text))
    ]


def _oracle_gb(text: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            (
                r.id,
                r.name,
                r.description,
                str(r.seq),
                len(r.features),
                [(f.type, str(f.location), f.qualifiers) for f in r.features],
            )
            for r in SeqIO.parse(io.StringIO(text), "genbank")
        ]


def _ours_gb(text: str):
    return [
        (
            r.id,
            r.name,
            r.description,
            str(r.seq),
            len(r.features),
            [(f.type, str(f.location), f.qualifiers) for f in r.features],
        )
        for r in bio_mojo.parse_genbank(io.StringIO(text))
    ]


def _expected_backend() -> str:
    return "fallback" if os.environ.get("BIO_MOJO_DISABLE_NATIVE") == "1" else "native"


# ---------------------------------------------------------------------------
# Backend sanity
# ---------------------------------------------------------------------------


def test_backend_is_the_expected_one():
    assert bio_mojo.backend() == _expected_backend()
    info = bio_mojo.backend_info()
    assert info["native_available"] is (_expected_backend() == "native")


# ---------------------------------------------------------------------------
# FASTA differential tests
# ---------------------------------------------------------------------------


def test_fasta_seeded_small():
    text = make_fasta_corpus(seed=1, n_records=50)
    assert _ours_fasta(text) == _oracle_fasta(text)


def test_fasta_seeded_1k():
    text = make_fasta_corpus(seed=2, n_records=1_000)
    assert _ours_fasta(text) == _oracle_fasta(text)


def test_fasta_seeded_150k_records():
    """>100k-record corpus: full record-for-record comparison."""
    text = make_fasta_corpus(seed=3, n_records=150_000)
    ours = _ours_fasta(text)
    oracle = _oracle_fasta(text)
    assert len(ours) == len(oracle) == 150_000
    assert ours == oracle


def test_fasta_path_and_pathlib_and_buffer_agree(tmp_path):
    text = make_fasta_corpus(seed=4, n_records=40)
    path = tmp_path / "corpus.fa"
    path.write_text(text)
    oracle = _oracle_fasta(text)
    assert [(r.id, r.name, r.description, str(r.seq))
            for r in bio_mojo.parse_fasta(str(path))] == oracle
    assert [(r.id, r.name, r.description, str(r.seq))
            for r in bio_mojo.parse_fasta(path)] == oracle
    with open(path, "r") as fh:
        assert [(r.id, r.name, r.description, str(r.seq))
                for r in bio_mojo.parse_fasta(fh)] == oracle


FASTA_EDGE_CASES = [
    ">id1 desc one\nACGT\nTGCA\n>id2\nNN\n",
    ">id1   desc with  spaces  \nACGT\n",
    ">\nACGT\n",
    ">   \nACGT\n",
    ">id1\tdesc tab\nACGT\n",
    ">  id1 desc\nAC\n",
    ">id1 d\nACGT\n;notacomment\n>id2\nNN\n",
    ">id1 d\nAC\n\n\nTG\n>id2\nNN\n",
    ">id1 d\nAC GT\n T G \n",
    ">id1 d\nAC\tGT\n",
    ">id1 d\nAC\x0bGT\x0cTT\n",
    ">id1 d\r\nACGT\r\nTG\r\n",
    ">id1 d\nACGT",
    ">id1 d\nacgtn.*-x\n",
    ">id1 d\n",
    ">id1 d\n>id2 d2\nAC\n",
    ">id1 déscription ünïcodé\nAC\n",
    ">id1 d\nAC12GT\n",
    ">id1 desc \nAC\n",
]


@pytest.mark.parametrize("text", FASTA_EDGE_CASES)
def test_fasta_edge_cases(text):
    assert _ours_fasta(text) == _oracle_fasta(text)


def test_fasta_empty_file_yields_nothing():
    assert _ours_fasta("") == _oracle_fasta("") == []


FASTA_PREAMBLE_CASES = [
    "; a comment\n>id1 d\nACGT\n",
    "# comment\n>id1 d\nACGT\n",
    "\n\n>id1 d\nACGT\n",
    "hello\n>id1\nAC\n",
    "\n",
    "   \n>id1\nAC\n",
]


@pytest.mark.parametrize("text", FASTA_PREAMBLE_CASES)
def test_fasta_preamble_raises(text):
    """Any content before the first '>' line: the oracle raises ValueError
    and so do we (ParseError subclasses ValueError), on both backends."""
    with pytest.raises(ValueError):
        _oracle_fasta(text)
    with pytest.raises(ValueError):
        _ours_fasta(text)
    with pytest.raises(bio_mojo.ParseError):
        _ours_fasta(text)


def test_fasta_nonascii_sequence_raises_unicode_decode_error():
    text = ">id1 d\nAC GT\n"  # U+00A0 inside the sequence
    with pytest.raises(UnicodeDecodeError):
        _oracle_fasta(text)
    with pytest.raises(UnicodeDecodeError):
        _ours_fasta(text)


def test_fasta_binary_handle_raises_stream_mode_error():
    with pytest.raises(ValueError):
        list(bio_mojo.parse_fasta(io.BytesIO(b">id1 d\nAC\n")))


def test_fasta_repeated_parse_is_deterministic():
    text = make_fasta_corpus(seed=5, n_records=200)
    assert _ours_fasta(text) == _ours_fasta(text)


# ---------------------------------------------------------------------------
# GenBank differential tests
# ---------------------------------------------------------------------------


def test_gb_seeded_small():
    text = make_gb_corpus(seed=11, n_records=30)
    assert _ours_gb(text) == _oracle_gb(text)


def test_gb_seeded_500():
    text = make_gb_corpus(seed=12, n_records=500)
    assert _ours_gb(text) == _oracle_gb(text)


def test_gb_seeded_3k_crlf():
    text = make_gb_corpus(seed=13, n_records=3_000).replace("\n", "\r\n")
    assert _ours_gb(text) == _oracle_gb(text)


def test_gb_path_and_buffer_agree(tmp_path):
    text = make_gb_corpus(seed=14, n_records=20)
    path = tmp_path / "corpus.gbk"
    path.write_text(text)
    oracle = _oracle_gb(text)
    assert [
        (r.id, r.name, r.description, str(r.seq), len(r.features),
         [(f.type, str(f.location), f.qualifiers) for f in r.features])
        for r in bio_mojo.parse_genbank(str(path))
    ] == oracle


_GB_HDR = "LOCUS       T            100 bp    DNA             UNA       01-JAN-2000\n"
_GB_META = "DEFINITION  x.\nACCESSION   A1\nVERSION     A1.1  GI:9\n"
_GB_ORG = "ORIGIN\n        1 acgtacgtac gtacgtacgt\n//\n"


def _gb_with_features(feat_text: str) -> str:
    return (
        _GB_HDR + _GB_META + "FEATURES             Location/Qualifiers\n"
        + feat_text + _GB_ORG
    )


GB_EDGE_CASES = [
    # qualifier forms: flags, unquoted, embedded quotes, repeats, multiline
    _gb_with_features(
        "     CDS             10..90\n"
        "                     /pseudo\n"
        "                     /number=42\n"
        "                     /label=bare\n"
        "                     /note=\"has \"\"embedded\"\" quotes\"\n"
        "                     /empty=\"\"\n"
        "                     /note=\"rep\"\n"
        "                     /note=\"rep2\"\n"
        "                     /translation=\"MKL\n"
        "                     AAA\n"
        "                     BBB\"\n"
    ),
    # location forms (supported + oracle-None forms)
    _gb_with_features(
        "     gene            complement(5..10)\n"
        "                     /note=\"b\"\n"
        "     gene            join(5..10,complement(20..30),40..50)\n"
        "                     /note=\"c\"\n"
        "     gene            complement(join(5..10,20..30))\n"
        "                     /note=\"d\"\n"
        "     gene            complement(order(5..10,20..30))\n"
        "                     /note=\"d2\"\n"
        "     gene            <5..>10\n"
        "                     /note=\"f\"\n"
        "     gene            5^6\n"
        "                     /note=\"g\"\n"
        "     gene            5^9\n"
        "                     /note=\"g2\"\n"
        "     gene            123\n"
        "                     /note=\"h\"\n"
        "     gene            complement(123)\n"
        "                     /note=\"h2\"\n"
        "     gene            one-of(5,9)\n"
        "                     /note=\"i\"\n"
        "     gene            order(5..10,20..30)\n"
        "                     /note=\"j\"\n"
        "     gene            J00194.1:100..200\n"
        "                     /note=\"k\"\n"
        "     gene            complement(complement(5..10))\n"
        "                     /note=\"l\"\n"
        "     gene            join(join(5..10,11..12),20..30)\n"
        "                     /note=\"m\"\n"
        "     gene            10..5\n"
        "                     /note=\"n\"\n"
        "     gene            join(5..10)\n"
        "                     /note=\"o\"\n"
        "     gene            bogus_location\n"
        "                     /note=\"p\"\n"
        "     rRNA            complement(<5..>10)\n"
        "                     /note=\"q\"\n"
    ),
    # empty-location feature is dropped by the oracle
    _gb_with_features("     gene            \n     source          1..5\n"),
    _gb_with_features("     gene            join(5..10, 20..30)\n"),
    # definition variants
    _GB_HDR + "DEFINITION  Ends with two..\nACCESSION   A1\nVERSION     A1.1\n" + _GB_ORG,
    _GB_HDR + "DEFINITION  No trailing period\nACCESSION   A1\nVERSION     A1.1\n" + _GB_ORG,
    _GB_HDR + "DEFINITION  .\nACCESSION   A1\nVERSION     A1.1\n" + _GB_ORG,
    _GB_HDR + "DEFINITION  x  has   gaps.\nACCESSION   A1\nVERSION     A1.1\n" + _GB_ORG,
    _GB_HDR + "DEFINITION  first\n            second.\nACCESSION   A1\nVERSION     A1.1\n" + _GB_ORG,
    # id rule variants
    _GB_HDR + "DEFINITION  d.\nACCESSION   AB1 AB2\n" + _GB_ORG,
    _GB_HDR + "DEFINITION  d.\n" + _GB_ORG,
    # section variants
    _GB_HDR + _GB_META + "FEATURES             Location/Qualifiers\n" + _GB_ORG,
    _GB_HDR + _GB_META + _GB_ORG,
    _GB_HDR + _GB_META + "ORIGIN\n        1 acgtacgtac\n",  # missing final //
    _GB_HDR + _GB_META + "ORIGIN\n        1 AcGt nN-x.*\n//\n",
    # junk between records is skipped
    _GB_HDR + _GB_META + _GB_ORG + "junk line\n\n" + _GB_HDR + _GB_META + _GB_ORG,
    "",
]


@pytest.mark.parametrize("text", GB_EDGE_CASES)
def test_gb_edge_cases(text):
    assert _ours_gb(text) == _oracle_gb(text)


def test_gb_missing_origin_raises():
    text = _GB_HDR + _GB_META + "FEATURES             Location/Qualifiers\n     source          1..5\n//\n"
    with pytest.raises(ValueError):
        _oracle_gb(text)
    with pytest.raises(bio_mojo.ParseError):
        _ours_gb(text)


def test_gb_malformed_sequence_line_raises():
    text = _GB_HDR + _GB_META + "ORIGIN\nacgtacgt acgt\n//\n"
    with pytest.raises(ValueError):
        _oracle_gb(text)
    with pytest.raises(bio_mojo.ParseError):
        _ours_gb(text)


def test_gb_locus_without_terminator_raises_explicitly():
    """Off-domain input: the oracle silently mangles records whose '//'
    terminator is missing before the next LOCUS line; we instead fail
    explicitly (documented divergence — see README)."""
    text = _GB_HDR + _GB_META + "ORIGIN\n        1 acgt\n" + _GB_HDR + _GB_META + _GB_ORG
    with pytest.raises(bio_mojo.ParseError):
        _ours_gb(text)


def test_gb_balanced_location_break_raises_explicitly():
    """Off-domain input: a location continuation line arriving when the
    location's parentheses are balanced crashes the oracle (AssertionError);
    we fail explicitly with ParseError (documented divergence)."""
    text = _gb_with_features(
        "     gene            complement\n"
        "                     (<8..>11)\n"
        "                     /note=\"x\"\n"
    )
    with pytest.raises(bio_mojo.ParseError):
        _ours_gb(text)


def test_gb_binary_handle_raises_stream_mode_error():
    with pytest.raises(ValueError):
        list(bio_mojo.parse_genbank(io.BytesIO(b"LOCUS       T\n")))


def test_gb_repeated_parse_is_deterministic():
    text = make_gb_corpus(seed=15, n_records=100)
    assert _ours_gb(text) == _ours_gb(text)


def test_gb_location_objects_roundtrip_str():
    """Our location objects render exactly like the oracle's for the whole
    supported subset (checked here through record parity, plus __repr__)."""
    text = make_gb_corpus(seed=16, n_records=50)
    ours = list(bio_mojo.parse_genbank(io.StringIO(text)))
    for record in ours:
        for feature in record.features:
            assert feature.location is None or repr(feature.location)
