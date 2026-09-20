# minisearch-mojo

**Drop-in faster replacement for [MiniSearch](https://github.com/lucaong/minisearch)
fuzzy/prefix search and auto-suggestions** — the same API, the same results,
powered by a clean-room bounded-Levenshtein + BM25 kernel written in
[Mojo](https://www.modular.com/mojo). Prebuilt per-platform native libraries,
a thin Node wrapper, and the vendored MiniSearch itself as an automatic
fallback on platforms without a native build (e.g. Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)). No Mojo
toolchain required at install time.

```js
// npm install @minisearch-mojo/core   →   then use it exactly like MiniSearch
import MiniSearch from '@minisearch-mojo/core'

const documents = [
  { id: 1, title: 'Moby Dick', text: 'Call me Ishmael. Some years ago...' },
  { id: 2, title: 'Zen and the Art of Motorcycle Maintenance', text: 'I can see by my watch...' },
  { id: 3, title: 'Neuromancer', text: 'The sky above the port was...' },
  { id: 4, title: 'Zen in the Art of Archery', text: 'At first sight it must seem...' },
]

const miniSearch = new MiniSearch({
  fields: ['title', 'text'],
  searchOptions: { fuzzy: 0.2 },
})
miniSearch.addAll(documents)

miniSearch.search('zen art motorcycle')
// → [{ id: 2, score: 2.77..., terms: [...], match: {...} }, ...]

miniSearch.autoSuggest('neromancer', { fuzzy: 0.2 })
// → [ { suggestion: 'neuromancer', terms: ['neuromancer'], score: 1.03... } ]
```

CommonJS works too: `const MiniSearch = require('@minisearch-mojo/core')`.

## Benchmark

Measured with `benchmarks/bench_minisearch_fuzzy.mjs` (a correctness gate
asserting scores within 1e-9, consistent ranking, and identical suggestions
runs before every timing pass). Corpora are generated locally from fixed
seeds: 10k / 50k / 100k documents (title + text fields) over a ~3000-token
vocabulary; 40 queries per cell, median of 5 runs after a warmup round.
**Cold** is the first-call experience: a fresh `addAll` (native index build
included) plus its first query, median of 5 runs × 3 queries. Environment:
**Apple M4 Max (16 threads), macOS arm64, Node v23.10.0, minisearch 7.2.0
(npm), Mojo 1.1.0**, 2026-09-19.

### Autopsy (why MiniSearch is slow here — measured, not assumed)

V8 CPU profile (`node --cpu-prof`) of MiniSearch 7.2.0 running the bench's
fuzzy/prefix/autoSuggest query mix over the 100k corpus (query loop 4.3s),
self time by function:

| self time | function | what it is |
|---:|---|---|
| 36% | `termResults` (dist/cjs/index.cjs:1708) | per-query-term resolution: the fuzzy/prefix expansion walk over the radix-tree vocabulary and per-variant posting fetch |
| 13% | `executeQuerySpec` (dist/cjs/index.cjs:1579) | query-term orchestration around the same resolution loop |
| 11% | `search` (dist/cjs/index.cjs:1281) | per-document score combination in JS |
| 11% | (garbage collector) | allocation churn in the JS index structures |
| 3% | `autoSuggest` (dist/cjs/index.cjs:1373) | suggestion assembly on top of the same machinery |
| rest | tokenize / byScore / sort | noise |

The hot path is term resolution — the fuzzy/prefix expansion DP plus
posting-list score accumulation executed in JavaScript over every candidate
term, every query. minisearch-mojo moves exactly that work (the bounded
edit-distance DP scan, the prefix scan, and the BM25 posting accumulation)
into a compiled Mojo kernel; tokenization, option handling, result assembly,
and the vendored fallback stay in JS, behavior-compatibly. The remaining JS
(index flattening, radix-order bookkeeping, FFI marshalling) is why speedups
are largest on suggest-heavy mixes (3x) and modest on exact single-term
search (1x) — and why tiny 10k fuzzy cells can even dip below 1x (FFI
overhead dominates sub-60us queries).

### Warm steady state (median of 5 runs, ms per query)

| corpus | query mix | n queries | MiniSearch ms | mojo ms | MiniSearch q/s | mojo q/s | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | search exact (OR) | 40 | 0.019 | 0.020 | 53,673.3 | 50,172.5 | 0.9x |
| 10,000 | search fuzzy 0.2 | 40 | 0.051 | 0.066 | 19,483.3 | 15,168.5 | 0.8x |
| 10,000 | search fuzzy+prefix AND | 40 | 0.192 | 0.165 | 5,195.2 | 6,077.7 | 1.2x |
| 10,000 | search prefix | 40 | 1.021 | 0.772 | 979.7 | 1,295.5 | 1.3x |
| 10,000 | autoSuggest fuzzy 0.2 | 40 | 0.057 | 0.057 | 17,654.5 | 17,609.5 | 1.0x |
| 10,000 | autoSuggest default (prefix last) | 40 | 0.992 | 0.362 | 1,008.2 | 2,759.0 | 2.7x |
| 50,000 | search exact (OR) | 40 | 0.162 | 0.154 | 6,155.3 | 6,493.9 | 1.1x |
| 50,000 | search fuzzy 0.2 | 40 | 0.097 | 0.105 | 10,353.1 | 9,565.2 | 0.9x |
| 50,000 | search fuzzy+prefix AND | 40 | 0.486 | 0.212 | 2,057.5 | 4,721.5 | 2.3x |
| 50,000 | search prefix | 40 | 7.730 | 6.613 | 129.4 | 151.2 | 1.2x |
| 50,000 | autoSuggest fuzzy 0.2 | 40 | 0.095 | 0.087 | 10,473.9 | 11,551.8 | 1.1x |
| 50,000 | autoSuggest default (prefix last) | 40 | 8.110 | 2.622 | 123.3 | 381.4 | 3.1x |
| 100,000 | search exact (OR) | 40 | 0.338 | 0.223 | 2,957.8 | 4,483.0 | 1.5x |
| 100,000 | search fuzzy 0.2 | 40 | 0.223 | 0.182 | 4,490.2 | 5,488.5 | 1.2x |
| 100,000 | search fuzzy+prefix AND | 40 | 1.060 | 0.338 | 943.8 | 2,962.7 | 3.1x |
| 100,000 | search prefix | 40 | 20.734 | 14.827 | 48.2 | 67.4 | 1.4x |
| 100,000 | autoSuggest fuzzy 0.2 | 40 | 0.232 | 0.142 | 4,313.7 | 7,021.6 | 1.6x |
| 100,000 | autoSuggest default (prefix last) | 40 | 21.319 | 6.779 | 46.9 | 147.5 | 3.1x |

### Cold first call (fresh addAll + first query, median of 5 runs × 3 queries)

| corpus | query mix | MiniSearch cold (ms) | mojo cold (ms) | speedup |
|---:|---|---:|---:|---:|
| 10,000 | search exact (OR) | 65.5 | 36.0 | 1.8x |
| 10,000 | search fuzzy 0.2 | 67.8 | 36.2 | 1.9x |
| 10,000 | search fuzzy+prefix AND | 68.0 | 36.4 | 1.9x |
| 10,000 | search prefix | 69.2 | 37.4 | 1.9x |
| 10,000 | autoSuggest fuzzy 0.2 | 61.6 | 32.8 | 1.9x |
| 10,000 | autoSuggest default (prefix last) | 63.8 | 33.3 | 1.9x |
| 50,000 | search exact (OR) | 379.0 | 170.3 | 2.2x |
| 50,000 | search fuzzy 0.2 | 413.7 | 178.5 | 2.3x |
| 50,000 | search fuzzy+prefix AND | 378.8 | 161.8 | 2.3x |
| 50,000 | search prefix | 378.2 | 162.4 | 2.3x |
| 50,000 | autoSuggest fuzzy 0.2 | 386.0 | 173.5 | 2.2x |
| 50,000 | autoSuggest default (prefix last) | 353.8 | 164.2 | 2.2x |
| 100,000 | search exact (OR) | 777.6 | 326.0 | 2.4x |
| 100,000 | search fuzzy 0.2 | 765.9 | 336.4 | 2.3x |
| 100,000 | search fuzzy+prefix AND | 792.1 | 344.2 | 2.3x |
| 100,000 | search prefix | 852.2 | 348.2 | 2.4x |
| 100,000 | autoSuggest fuzzy 0.2 | 764.8 | 316.9 | 2.4x |
| 100,000 | autoSuggest default (prefix last) | 805.5 | 332.7 | 2.4x |

### Index construction (median of 5 builds)

| corpus | MiniSearch build (ms) | mojo build (ms) | ratio |
|---:|---:|---:|---:|
| 10,000 | 66.4 | 36.2 | 0.54x |
| 50,000 | 437.2 | 194.0 | 0.44x |
| 100,000 | 680.5 | 304.9 | 0.45x |

Correctness gate: max |Δscore| = max |Δscore| = 1.71e-13 (at 50000/search fuzzy+prefix AND/larenmoor larenmoor paschigu)

## How it works

- `npm install @minisearch-mojo/core` installs the JS wrapper plus the
  matching optional platform package (`@minisearch-mojo/darwin-arm64` or
  `@minisearch-mojo/linux-x64`), which carries a prebuilt shared library with
  its Mojo runtime vendored inside (self-contained: `delocate`/`patchelf`
  repaired, loads on a machine with no Mojo toolchain).
- The wrapper resolves the native kernel at first use, verifies an ABI
  version handshake, and then serves `add`/`addAll`/`search`/`autoSuggest`
  from it. Index building is incremental; the native index rebuilds lazily
  after new documents.
- Per query, the kernel runs a bounded Levenshtein DP scan (plain
  ins/del/sub costs, no transposition) and a prefix scan over the whole
  vocabulary in reference-faithful traversal orders, then accumulates the
  reference's BM25 variant over the resolved variants' posting lists.
- On any load failure (missing/unsupported platform, ABI mismatch) the
  package transparently falls back to the vendored MiniSearch 7.2.0 (MIT,
  see `vendor/` and `NOTICE`), so results are correct on every platform.
- Set `MINISEARCH_MOJO_DISABLE_NATIVE=1` to force the fallback (the
  differential suite runs twice: native and forced fallback).
  `MINISEARCH_MOJO_NATIVE_LIB=/path/to/libmsmojo.(dylib|so)` overrides the
  native library resolution for development.

## Supported options (behavior-parity vs MiniSearch 7.2.0)

- Index options: `fields` (required), `storeFields`, `idField`,
  `searchOptions` (defaults merged into per-call options).
- Search options: `prefix` (boolean or per-term function), `fuzzy` (boolean,
  number, or per-term function), `boost`, `fields`, `combineWith`
  (`'AND'`/`'OR'`).
- `add`, `addAll`, `search`, `autoSuggest` with the reference's result
  shapes (`{id, score, terms, queryTerms, match, ...stored}` and
  `{suggestion, terms, score}`).
- Replicated semantics (all extracted black-box from the reference):
  - default tokenizer splitting on Unicode space/punctuation, lowercasing;
    the edge-empty token from leading/trailing separators counts in document
    field lengths (it never matches queries);
  - fuzzy edit budget `min(Math.round(fuzzy * term.length), 6)` for
    `fuzzy < 1` (the relative budget is capped at 6), absolute budget for
    `fuzzy >= 1`, plain Levenshtein distance (no transposition);
  - fuzzy weight `0.45 * len / (len + distance)`, prefix weight
    `15 * len / (52 * len - 12 * qlen)`, exact matches always weight 1 and
    prefix beats fuzzy on overlap;
  - per-field document frequency in `idf = 1.5 * ln(1 + (N - df + 0.5) /
    (df + 0.5))`, tf component `F = (45*tf + 3 + 7r) / (25*tf + 9 + 21r)`
    with `r = dl/avgdl`, per-field unique-term lengths, and running-mean
    average lengths (absent fields inherit the current average);
  - document score = sum of weighted per-variant scores × the number of
    matched distinct query terms; ties keep first-encountered order;
  - `autoSuggest`: last-term prefix expansion by default (explicit `prefix`
    applies to all terms), `AND` combination by default, suggestion score =
    mean of the generating documents' query scores, variant enumeration in
    radix-tree traversal order (prefix: reverse-map DFS, exact term first;
    fuzzy: forward-map DFS).
- **Parity tolerance:** search scores and suggestion scores agree with the
  reference within **1e-9** (typically far inside; the only remaining
  deviation is ±1-2 ulps from floating-point operation order in score
  accumulation). Result sets, `terms`/`queryTerms`/`match`, and suggestion
  strings are exactly identical. Ranking order is identical wherever scores
  differ by more than ~1e-12 — see "Deviations" below.
- Not supported (throws `UnsupportedOptionError`): custom
  `tokenize`/`processTerm`/`extractField`, `filter`, `remove`/`discard`/
  `vacuum`, `loadJSON`/`loadJS`, logical query objects, > 32 distinct query
  terms.

## Deviations

- **Ulp-level score ties:** scores agree within 1e-9, but where the
  reference's own float operation order makes two documents differ by 1-2
  ulps, the relative order of those two documents can occasionally flip
  (their scores are genuine float ties at the 1e-12 level). The differential
  suite asserts ranking consistency on all pairs separated by > 1e-12.
- The relative fuzzy budget cap (6 edits) means very long terms at high
  fuzzy values match less than a naive `round(fuzzy * len)` would suggest —
  this matches the reference exactly (verified at len 13 / 15 / 20).

## Differential tests

`npm test` in `typescript/minisearch_fuzzy_mojo` runs 35 differential cells
against the published MiniSearch 7.2.0 plus 7 unit tests, over seeded
corpora (600 docs, ~1200-token vocabulary, unicode tokens) with ~200
generated queries per cell, covering exact/fuzzy/prefix search, AND/OR
combination, field boost, field restriction, `searchOptions` defaults,
autoSuggest across the option matrix, unicode-heavy corpora, edge
collections, incremental adds, and string/number id types. The same suite
runs twice per CI job: once native, once with the fallback forced
(`MINISEARCH_MOJO_DISABLE_NATIVE=1`).

## Repository layout (this kernel)

```
kernels/minisearch_fuzzy/
  build.sh                 # mojo build --emit shared-lib
  src/msmojo.mojo          # clean-room kernel (scans + BM25 accumulator)
typescript/minisearch_fuzzy_mojo/
  packages/core/           # @minisearch-mojo/core (wrapper + vendored fallback)
  packages/darwin-arm64/   # prebuilt macOS arm64 kernel package
  packages/linux-x64/      # prebuilt Linux x64 kernel package
  tests/                   # differential + unit suites
  scripts/                 # pack-platform / smoke / test_all
benchmarks/bench_minisearch_fuzzy.mjs
.github/workflows/ci-minisearch_fuzzy.yml
```

## License

Apache-2.0 for the original code (Algenta). The vendored fallback in
`packages/core/vendor/` is MiniSearch 7.2.0 by Luca Ongaro, MIT license (see
`NOTICE` and `vendor/minisearch.LICENSE.txt`).
