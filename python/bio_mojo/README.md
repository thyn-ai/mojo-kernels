# bio-mojo

FASTA and GenBank parsers that are **record-for-record identical to
[`Bio.SeqIO`](https://biopython.org/)** (biopython) — powered by a clean-room
Mojo kernel, with a vendored pure-Python fallback for platforms without a
native build (including Windows). No runtime dependencies; biopython itself
is used only as the differential-test oracle, never at runtime.

```python
import bio_mojo

for record in bio_mojo.parse_fasta("reads.fa"):       # Bio.SeqIO "fasta"
    print(record.id, record.description, len(record.seq))

for record in bio_mojo.parse_genbank("genome.gbk"):   # Bio.SeqIO "genbank"
    print(record.id, len(record.features))
    for feature in record.features:
        print(feature.type, feature.location, feature.qualifiers)
```

`source` may be a path (`str`/`os.PathLike`) or a text-mode file object.
Both parsers are lazy generators like the oracle: input is consumed on the
first `next()`, and parse errors surface there.

## What "identical" means (parity contract)

The differential suite compares, on both backends (native and forced
fallback), against `Bio.SeqIO.parse` from the pinned published package
(`biopython==1.88`):

- **FASTA**: `record.id`, `record.name`, `record.description`,
  `str(record.seq)` — including multiline sequences, blank lines, comment
  lines inside records (they are sequence data), empty sequences, empty
  headers, CRLF, missing trailing newline, UTF-8 titles, and the oracle's
  error behavior (any content before the first `>` line raises ValueError;
  non-ASCII sequence content raises UnicodeDecodeError; binary handles
  raise StreamModeError). Tested up to 150k-record corpora.
- **GenBank** (subset: LOCUS, DEFINITION, ACCESSION, VERSION, FEATURES
  locations+qualifiers for the common feature types, ORIGIN): `record.id`,
  `record.name`, `record.description`, `str(record.seq)`,
  `len(record.features)`, and per feature `type`, `str(location)`,
  `qualifiers` — including multiline definitions, the
  VERSION-else-ACCESSION-else-LOCUS id rule, multiline locations/qualifiers,
  `/translation` no-space joining, `""` quote escaping, flag qualifiers, and
  the oracle's location normalization (`[122:456](+)`, `[<122:>456](+)`,
  `join{...}`, `order{...}`, complement reversal, `[50:50]` bonds, remote
  accessions, and `None` for locations the oracle leaves unparsed).

Parity is **exact string/dict equality** — there is no numeric tolerance.
134 differential assertions (67 native + 67 fallback) plus ABI/loader tests
run in CI on macOS and Linux.

## Measured performance

Whole-file parse into records (median of 5 warm runs; cold = first call in
the process incl. kernel load), vs `Bio.SeqIO.parse`, measured on this
machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14, Mojo 1.1.0,
biopython 1.88, 2026-09-19; seeded synthetic corpora — reproduce with
`benchmarks/bench_biopython.py` in the repository):

| corpus | file size | records | cold Bio.SeqIO | cold bio_mojo | cold speedup | warm Bio.SeqIO | warm bio_mojo | warm speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FASTA 10,000 | 1.9 MB | 10,000 | 10.6 ms | 7.6 ms | 1.4x | 10.5 ms | 6.9 ms | 1.5x |
| FASTA 150,000 | 29.2 MB | 150,000 | 173.6 ms | 114.7 ms | 1.5x | 176.7 ms | 116.9 ms | 1.5x |
| GenBank 300 | 0.4 MB | 300 | 12.3 ms | 5.9 ms | 2.1x | 13.0 ms | 5.6 ms | 2.3x |
| GenBank 3,000 | 3.7 MB | 3,000 | 142.6 ms | 75.7 ms | 1.9x | 145.9 ms | 67.4 ms | 2.2x |

Warm bio_mojo throughput: FASTA ~250-280 MB/s (~1.3-1.5M records/s),
GenBank ~54-65 MB/s.

Note what these numbers mean: biopython's per-line FASTA loop is already
tight, so the kernel's win is bounded by the per-record Python object
assembly both sides must do; GenBank's feature-table handling gives the
kernel more room. The pure-Python fallback runs at roughly oracle speed
(FASTA ~0.4x-0.5x, GenBank ~0.6x — it exists for correctness, not speed).

## Backend selection

Per-platform wheels ship the compiled kernel (macOS arm64, Linux x86_64);
everywhere else the package transparently uses its vendored pure-Python
parser, with identical results.

- Force the fallback: `BIO_MOJO_DISABLE_NATIVE=1`
- Inspect the active backend: `bio_mojo.backend_info()`, `bio_mojo.backend()`
- Explicit kernel path override (development): `BIO_MOJO_NATIVE_LIB=/path/libbioparse.dylib`

## Unsupported scope (explicit divergences from the oracle)

For well-formed input within the formats above there are no known
divergences. The following off-domain behaviors differ by design — we fail
explicitly (`bio_mojo.ParseError`, a ValueError) rather than mirror the
oracle's junk recovery:

- Records whose `//` terminator is missing before the next LOCUS line, and
  LOCUS lines inside a record (the oracle silently merges/mangles such
  records).
- A feature-location continuation line arriving when the location's
  parentheses are balanced (the oracle crashes with AssertionError).
- Reversed ranges `a..b` with `a > b` where the start exceeds the declared
  sequence length (the oracle raises ValueError; we return `location=None`,
  matching the oracle whenever both bounds are within the declared length).
- Non-standard LOCUS line layouts (the oracle validates fixed columns; we
  accept the first token as the record name).
- Text streams containing lone-`\r` line breaks inside a line (e.g. a
  `StringIO` with embedded carriage returns): we always apply universal
  newlines, matching real text-mode files.
- GenBank records without an ORIGIN section (CONTIG-style assemblies): we
  raise ParseError, like the oracle's ValueError; sequence-less records are
  outside the supported subset.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
