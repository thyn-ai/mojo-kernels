# slugify-mojo

Drop-in faster replacement for the [`python-slugify`](https://pypi.org/project/python-slugify/)
slug generator, powered by a Mojo kernel — with a pure-Python fallback that
keeps the package correct everywhere else.

```python
from slugify_mojo import slugify, slugify_column

slugify("Déjà Vu — Café naïve résumé")
# 'deja-vu-cafe-naive-resume'

slugify_column(["Hello World!", "Москва столица", "日本語のテキスト"])
# ['hello-world', 'moskva-stolitsa', 'ri-ben-yu-notekisuto']
```

- **Same output, byte for byte.** `slugify(...)` returns exactly the same
  string as `python-slugify` 9.1.0 for both pipelines — the default
  `algorithm='legacy'` and `algorithm='modern'` — across unicode ranges
  (Cyrillic, Greek, CJK, emoji, accents, combining marks, fullwidth forms),
  HTML entity references (named/decimal/hex, valid, invalid and boundary
  values), punctuation runs, `max_length` truncation with `word_boundary` /
  `save_order`, separator variants, `lowercase=False`, `allow_unicode`,
  `replacements` at all stages, stopwords, custom `regex_pattern` (str or
  compiled), and bytes/bytearray input. Verified by a ~1,700-case
  differential suite against the PyPI oracle, run twice: once on the native
  kernel and once on the forced fallback. Agreement is exact string
  equality — there is no tolerance to tune.
- **Same errors.** Invalid argument types raise the same exception type
  with the same message as the oracle (the modern pipeline's strict
  `max_length`/`separator` validation included).
- **Same transliteration table.** The table is read at runtime from the
  installed `text-unidecode` package's own data file (the same package
  python-slugify depends on; never re-derived, never vendored), so
  transliteration parity holds by construction.

## Install

```bash
pip install slugify-mojo
```

Wheels are published for macOS arm64 and Linux x86_64 (with the Mojo kernel
inside, repaired with delocate/auditwheel so they are self-contained). On any
other platform the same code installs from a pure wheel and works via the
pure-Python fallback — same results, no native code.

## Quickstart

```python
import slugify_mojo

slugify_mojo.slugify("The Quick Brown Fox — Déjà Vu!")
# 'the-quick-brown-fox-deja-vu'

# Every python-slugify 9.1.0 keyword works the same way:
slugify_mojo.slugify("A very long title indeed", max_length=16, word_boundary=True)
# 'a-very-long'

slugify_mojo.slugify("Déjà Vu", allow_unicode=True)
# 'déjà-vu'

slugify_mojo.slugify("Café", separator="_", lowercase=False)
# 'Cafe'

slugify_mojo.slugify("Article about the Cats", stopwords=["the", "about"])
# 'article-cats'

# Batch API for CMS/e-commerce pipelines:
slugify_mojo.slugify_column(["Fish &amp; Chips", "Crème brûlée 😀"])
# ['fish-chips', 'creme-brulee']

# The reference's public helpers and the modern pipeline:
slugify_mojo.smart_truncate("one two three four", 10, word_boundary=True)
# 'one two'
slugify_mojo.slugify("Déjà &amp; &#65;", algorithm="modern")
# 'deja-a'
```

Check which engine is serving slugs:

```python
slugify_mojo.backend()        # 'native' or 'fallback'
slugify_mojo.backend_info()   # resolver diagnostics (ABI handshake, source path)
```

Set `SLUGIFY_MOJO_DISABLE_NATIVE=1` to force the pure-Python fallback
(same output, useful for debugging).

## How it works

The reference pipeline spends most of its time on per-call import machinery
(the default `backend='auto'` retries a failed `unidecode` import on every
call), the per-character transliteration loop in pure Python, and a chain of
regular-expression substitutions. slugify-mojo keeps every Unicode-defined
step in CPython where the reference uses it (NFKD/NFKC normalization, case
folding, user regex patterns, the `allow_unicode` path) and moves the
byte-defined hot stages into the Mojo kernel:

1. **transliteration** — one pass over the UTF-8 bytes through the
   text-unidecode table (the table blob is handed to the kernel at init),
2. **entity decoding** — `&name;`, `&#123;`, `&#x1a;` with the reference's
   exact per-algorithm semantics (legacy numeric decoding is all-or-nothing
   per base; modern leaves only the offending reference literal),
3. **the ASCII filter** — apostrophe removal, digit-comma joining
   (`1,000` → `1000`), disallowed-run collapsing, dash dedup and trim, fused
   into a single pass.

When the input is pure ASCII and contains no `&` (the common CMS case), the
whole pipeline is *one* kernel call: NFKD is provably the identity on ASCII,
and the reference's `.lower()` reduces to the A-Z mapping the kernel
performs. A load-time check over the table certifies that no ASCII
codepoint other than `&` itself can transliterate to `&`, so the fused path
can never skip a real entity reference. `slugify_column` batches the kernel
calls over the whole column with per-item stage flags.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.5,
Mojo 1.1.0), 2026-09-20, with `benchmarks/bench_slugify.py` (median of 5;
correctness gate byte-exact vs the oracle on every corpus text first).
"Fallback" is the same package with `SLUGIFY_MOJO_DISABLE_NATIVE=1`.

Warm steady-state `slugify()`:

| workload | PyPI python-slugify | slugify-mojo (native) | slugify-mojo (fallback) | native speedup |
|---|---:|---:|---:|---:|
| 20× ascii titles, per call | 25.15 µs | 3.08 µs | 5.12 µs | 8.2x |
| 20× accented titles, per call | 25.23 µs | 3.94 µs | 5.13 µs | 6.4x |
| 20× mixed-script titles, per call | 26.17 µs | 3.86 µs | 6.07 µs | 6.8x |
| 20× entity-heavy titles, per call | 28.44 µs | 5.82 µs | 8.68 µs | 4.9x |
| long text (1164 chars), per call | 145.9 µs | 34.2 µs | 132.9 µs | 4.3x |
| `slugify_column`, 1000 mixed titles | 26.42 ms | 3.53 ms | 6.55 ms | 7.5x |

Cold first-call (fresh process: import + table load + first slugify, median
of 5):

| package | import + first slugify |
|---|---:|
| PyPI python-slugify | 16.5 ms |
| slugify-mojo (native) | 23.0 ms |
| slugify-mojo (fallback) | 18.2 ms |

Honest notes on the numbers:

- The oracle's per-call cost is dominated by its own `import_module`
  resolution, which varies with system state; all numbers above are from one
  benchmark process. Re-run `benchmarks/bench_slugify.py` to reproduce.
- Cold start is *slower* than the oracle today (~1.4x): the wrapper loads
  the transliteration table and hands it to the kernel on first use. This is
  a per-process cost, amortized over every subsequent call.

## Unsupported scope / honest limits

- **`backend='unidecode'` and `backend='anyascii'`** work exactly like the
  oracle (the named package is imported and called through its public API,
  per the reference's own resolution rules, including `backend='auto'`
  preferring an installed `unidecode`), but those paths are not
  kernel-accelerated — only the text-unidecode table path is.
- **`allow_unicode=True`** is byte-identical but runs the same CPython code
  on both backends (the `[\W_]+` filter is Unicode-defined); the kernel
  gives it no speedup beyond avoiding the per-call import machinery.
- **Custom `regex_pattern`** is likewise byte-identical and CPython-run.
- A transliteration table with non-ASCII replacements would make the native
  kernel reject the table and fall back automatically (text-unidecode 1.3's
  table is pure ASCII; the kernel validates this at load).
- `sys.get_int_max_str_digits()` values other than the default 4300 (e.g.
  via `PYTHONINTMAXSTRDIGITS`) are detected at call time; decimal entity
  decoding then runs in CPython to stay exact.
- Windows: no Mojo toolchain builds the kernel there today; the package
  installs and works via the pure-Python fallback (tested in CI on
  windows-latest).
- The differential suite pins `python-slugify==9.1.0` with
  `text-unidecode==1.3`. Other oracle versions are not covered.

## Development

```bash
# from the repository root, inside the pixi environment
bash kernels/slugify/build.sh                     # compile the Mojo kernel
bash scripts/test_all_slugify.sh                  # differential suite, both backends
python benchmarks/bench_slugify.py                # reproduce the benchmark table
bash python/slugify_mojo/build_wheel.sh           # build + repair the wheel
```

Layout: `kernels/slugify/` (Mojo kernel), `python/slugify_mojo/` (package,
wheel tooling, quickstart), `tests/test_slugify_*.py` (differential + loader
suites), `benchmarks/bench_slugify.py`, `.github/workflows/ci-slugify.yml`.

## License

Apache-2.0. The transliteration table is loaded at runtime from the
user-installed `text-unidecode` package (Artistic/GPL dual-licensed); no
reference code or data is vendored into this package.
