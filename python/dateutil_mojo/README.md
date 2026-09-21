# dateutil-mojo

Drop-in faster replacement for [`dateutil.parser.parse`](https://dateutil.readthedocs.io/en/stable/parser.html) (python-dateutil 2.9.0.post0 — the most-installed datetime parser on PyPI), powered by a Mojo kernel, with a vendored pure-Python fallback that is *itself* ~2× faster than the oracle on common formats.

```python
from dateutil_mojo import parse

parse("2025-07-08T14:30:00+02:00")        # datetime(2025, 7, 8, 14, 30, tzinfo=tzoffset(None, 7200))
parse("Tue, 08 Jul 2025 14:30:00 +0200")  # same
parse("07/08/2025")                       # datetime(2025, 7, 8)  (US order)
parse("March 1st, 2025", fuzzy=True)      # datetime(2025, 3, 1)
parse("12:30", default=datetime(2000, 1, 1))

from dateutil_mojo import parse_column    # ETL batch API
parse_column(["2025-07-08", "2025-07-09 12:00", "07/10/2025"])
```

Same call shape, same results, same errors — verified bit-for-bit against the
oracle on the oracle's own test-corpus inputs plus thousands of generated
variants, on **both** backends (native and forced fallback).

## Install

```bash
pip install dateutil_mojo-0.1.0-py3-none-<platform>.whl
```

Per-platform wheels (macOS arm64, Linux x86_64) bundle the Mojo kernel.
Everywhere else (Windows, other architectures) the wheel falls back to the
vendored pure-Python implementation — same API, same results, still ~2×
faster than python-dateutil on common formats. Zero runtime dependencies.

## Backends

| backend | where | notes |
|---|---|---|
| Mojo kernel | macOS arm64, Linux x86_64 | hot formats (ISO 8601, RFC 2822, US/EU numeric, named-month) parsed natively; anything else falls through to the Python reference per string |
| pure-Python reference | everywhere (incl. Windows) | full `parse()` semantics; used for all non-hot inputs and when the kernel can't load |

The backend is chosen transparently. Set `DATEUTIL_MOJO_DISABLE_NATIVE=1`
to force the fallback (this is how the differential test suite exercises
both paths). `dateutil_mojo.backend_info()` reports the active backend.

## API

- `parse(timestr, parserinfo=None, *, dayfirst=None, yearfirst=None, ignoretz=False, tzinfos=None, default=None, fuzzy=False, fuzzy_with_tokens=False)`
  — drop-in for `dateutil.parser.parse`. Returns a `datetime`.
- `parse_column(strings, **same_kwargs)` — batch API for ETL: parses a list
  of strings in one native-kernel call where possible, returning
  `list[datetime]`. Errors (and non-hot rows) are handled per row exactly
  like `parse()`.
- `dateutil_mojo.ParserError` — `ValueError` subclass matching the oracle's.
- `dateutil_mojo.tzutc`, `tzoffset`, `tzlocal` — tzinfo classes with the
  same `utcoffset()` / `tzname()` / `dst()` behavior as `dateutil.tz`'s
  (the package has no dependency on python-dateutil).

`default` semantics, `dayfirst`/`yearfirst` disambiguation, the sliding
2-digit-year pivot (a 2-digit year maps into `[current_year-50,
current_year+50)`, exactly like the oracle), fuzzy mode, `ignoretz`,
`tzinfos` (dict or callable), am/pm rules, `h`/`m`/`s` suffix times
(`01h02m03s`), weekday handling, ordinal suffixes, and the oracle's exact
error messages (`Unknown string format: …`, `String does not contain a
date: …`, `bad month number N; must be 1-12: …`, raw `OverflowError` /
`ValueError` where the oracle lets them escape) are all reproduced.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS arm64, Python 3.12.14,
Mojo 1.1.0), median of 5 runs, correctness gate asserted first
(`benchmarks/bench_dateutil.py`, 2026-09-20):

| workload | python-dateutil (us/call) | dateutil_mojo native (us/call) | speedup |
|---|---:|---:|---:|
| cold parse: iso-8601 | 16.88 | 5.63 | 3.00x |
| cold parse: iso-date-only | 7.29 | 4.91 | 1.49x |
| cold parse: rfc-2822 | 22.45 | 14.87 | 1.51x |
| cold parse: us-numeric | 12.81 | 4.83 | 2.65x |
| cold parse: named-month | 16.96 | 5.21 | 3.25x |
| cold parse: compact | 6.26 | 4.68 | 1.34x |
| warm parse: iso-8601 | 15.07 | 5.30 | 2.84x |
| warm parse: iso-date-only | 6.97 | 4.80 | 1.45x |
| warm parse: rfc-2822 | 22.66 | 15.52 | 1.46x |
| warm parse: us-numeric | 12.85 | 4.84 | 2.65x |
| warm parse: named-month | 17.36 | 4.96 | 3.50x |
| warm parse: compact | 6.19 | 4.67 | 1.32x |
| parse_column 100k ISO rows (us/row) | 16.26 | 1.51 | 10.75x |

The pure-Python fallback (used where the kernel is unavailable or the input
is outside the hot formats) is ~0.37–0.52× the oracle's time (i.e. ~2×
faster than the oracle) on the same workloads. `parse_column` is the ETL
fast path: one FFI call for the whole column.

Reproduce: `PYTHONPATH=python/dateutil_mojo PYTHONNOUSERSITE=1 pixi run python benchmarks/bench_dateutil.py`

## Parity

- Differential suite: **620 tests × 2 backends**, all green: the oracle's
  own test-corpus inputs (extracted from the python-dateutil 2.9.0.post0
  sdist, compared black-box), a ~250-case hand-picked probe corpus, and
  seeded generators covering ISO/RFC2822/US-EU/named-month/h:m:s/fuzzy
  families with dayfirst/yearfirst/ignoretz/tzinfos/default variants.
- Comparison is exact: datetime fields, `fold`, `tzname()`, `utcoffset()`,
  exception class + exact message, and emitted warnings.
- Tolerance: none. Datetimes compare bit-for-bit; `utcoffset()` equality
  (offsets ≥ 24h compare by the CPython rejection signature, as the oracle
  stores them).

## Unsupported scope (honest list)

- **Custom `parserinfo` subclasses** — passing one raises
  `NotImplementedError`. The default parserinfo behavior is fully supported.
- **`fuzzy_with_tokens` skipped-token tuple** — the returned tuple matches
  the oracle on common inputs; on some inputs with multiple skipped runs
  before 2+ kept tokens (e.g. `'spam … ham'`-style with extra kept tokens,
  and apostrophe-year inputs) whitespace attachment inside the tuple may
  differ. The returned **datetime is always bit-identical**.
- **`tzinfos` values that are tz strings** — resolved via
  `zoneinfo.ZoneInfo`; the oracle's `tzstr` parsing of POSIX TZ strings is
  not replicated (raises `TypeError` like the oracle for unresolvable ones).
- **Locale-specific month/weekday names** — English only, like the oracle's
  default parserinfo.
- **Windows native kernel** — no Mojo toolchain target; the pure-Python
  fallback ships instead and is silently correct.
- `parse()` of non-`str` inputs raises `TypeError` (the oracle fails less
  cleanly downstream; this is a deliberate deviation).

## Development

```bash
bash kernels/dateutil-parse/build.sh            # build the Mojo kernel
PYTHONPATH=python/dateutil_mojo PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_dateutil.sh
bash python/dateutil_mojo/build_wheel.sh        # wheel (delocate/auditwheel repaired)
```

Clean-room: the implementation was written from black-box observation of
python-dateutil 2.9.0.post0 behavior (probed input/output pairs and the
oracle's published test-corpus inputs used as differential fixtures). No
oracle source was read or adapted.

Attribution: Algenta.
