# fuse-mojo

**Drop-in faster replacement for [Fuse.js](https://fusejs.io) fuzzy search** —
the same API, the same results, powered by a clean-room Bitap kernel written
in [Mojo](https://www.modular.com/mojo). Prebuilt per-platform native
libraries, a thin Node wrapper, and the vendored Fuse.js itself as an
automatic fallback on platforms without a native build (e.g. Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)). No
Mojo toolchain required at install time.

```js
// npm install @fuse-mojo/core   →   then use it exactly like fuse.js
import Fuse from '@fuse-mojo/core'

const books = [
  { title: "Old Man's War", author: { firstName: 'John', lastName: 'Scalzi' } },
  { title: 'The Lock Artist', author: { firstName: 'Steve', lastName: 'Hamilton' } },
]

const fuse = new Fuse(books, { keys: ['title', 'author.firstName'] })
fuse.search('lock')
// → [{ item: {...}, refIndex: 1 }]
```

CommonJS works too: `const Fuse = require('@fuse-mojo/core')`.

## Benchmark

Measured with `benchmarks/bench_fuse.mjs` (run `pixi run bench-fuse` to
reproduce; a correctness gate asserting identical `refIndex` order and scores
within 1e-9 runs before every timing pass — measured agreement here is
**exactly 0**, the scores are bit-identical, not approximations). Corpora are
generated locally from fixed seeds: 10k / 50k / 100k documents of 5-15
word-like tokens (~3000-word vocabulary, a unicode token in every 11th
document); patterns of length 3 / 8 / 16 code units drawn from corpus tokens,
half with 1-2 seeded typos; 15 patterns per cell; median of 5 runs after a
warmup round. **Cold** is the first-call experience per §11: a fresh instance
(index build included) plus its first query, median of 5 runs × 3 patterns.
Environment: **Apple M4 Max (16 threads), macOS arm64, Node
v23.10.0, fuse.js 7.1.0 (npm), Mojo 1.1.0**, 2026-09-19.

### Autopsy (why Fuse.js is slow — measured, not assumed)

V8 CPU profile (`node --cpu-prof`) of Fuse.js 7.1.0 searching the 100k
corpus, self time by function:

| self time | function | what it is |
|---:|---|---|
| 48.1% | `search` (dist/fuse.cjs:713) | the Bitap bitmask DP itself |
| 45.2% | `chunks.forEach` closure (dist/fuse.cjs:988) | `BitapSearch.searchIn` driving the DP per document (inlined DP work attributed here) |
| 1.8% | (garbage collector) | allocation churn |
| 1.1% | `searchIn` (dist/fuse.cjs:957) | per-document dispatch, `toLowerCase` |
| <1% | everything else | scoring, sort, format |

~94% of query CPU is the Bitap dynamic program executed in JavaScript over
every document, every query — a pure interpreter loop, so the §11
classification is "pure-interpreter loop: 10-100x available". (Field-norm
tokenization, scoring and sorting are all noise; document lowercasing per
query is measurable but small at these document lengths.) fuse-mojo moves
exactly that loop into a compiled Mojo kernel and parallelizes it across
threads; everything else stays in JS, bit-compatibly.

### Search — warm steady state (median of 5 runs)

| corpus | pattern len | Fuse.js ms/query | fuse-mojo ms/query | Fuse.js q/s | fuse-mojo q/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 3 | 14.813 | 1.629 | 67.5 | 613.9 | 9.1x |
| 10,000 | 8 | 34.502 | 1.584 | 29.0 | 631.2 | 21.8x |
| 10,000 | 16 | 75.916 | 3.080 | 13.2 | 324.7 | 24.6x |
| 50,000 | 3 | 72.545 | 8.270 | 13.8 | 120.9 | 8.8x |
| 50,000 | 8 | 184.715 | 5.731 | 5.4 | 174.5 | 32.2x |
| 50,000 | 16 | 414.061 | 13.077 | 2.4 | 76.5 | 31.7x |
| 100,000 | 3 | 168.540 | 18.851 | 5.9 | 53.0 | 8.9x |
| 100,000 | 8 | 419.301 | 16.232 | 2.4 | 61.6 | 25.8x |
| 100,000 | 16 | 1,239.508 | 32.003 | 0.8 | 31.2 | 38.7x |

### Search — cold first call (fresh index build + first query, median of 5 runs × 3 patterns)

| corpus | pattern len | Fuse.js cold (ms) | fuse-mojo cold (ms) | speedup |
|---:|---:|---:|---:|---:|
| 10,000 | 3 | 18.9 | 6.6 | 2.9x |
| 10,000 | 8 | 37.5 | 6.1 | 6.2x |
| 10,000 | 16 | 75.3 | 6.7 | 11.2x |
| 50,000 | 3 | 84.8 | 29.5 | 2.9x |
| 50,000 | 8 | 195.7 | 27.3 | 7.2x |
| 50,000 | 16 | 402.6 | 30.4 | 13.3x |
| 100,000 | 3 | 171.4 | 55.5 | 3.1x |
| 100,000 | 8 | 417.7 | 60.5 | 6.9x |
| 100,000 | 16 | 945.4 | 70.3 | 13.4x |

Index construction is one-time; ours is ~1.4-1.7x slower than Fuse.js's
(we additionally lowercase and copy every searchable string into the native
index once, instead of re-lowercasing every document on every query) and pays
for itself within the first couple of queries at these corpus sizes:

| corpus | Fuse.js build (ms) | fuse-mojo build (ms) | ratio |
|---:|---:|---:|---:|
| 10,000 | 3.2 | 5.3 | 1.65x |
| 50,000 | 14.6 | 20.6 | 1.41x |
| 100,000 | 28.4 | 40.6 | 1.43x |

Where the speed comes from: Fuse.js runs the Bitap bitmask DP over every
document for every query, in JavaScript, and re-lowercases every document on
every search. fuse-mojo runs a compiled Mojo kernel over a pre-lowered UTF-16
index, word-parallel (32 text positions per 32-bit word in the Shift-And DP)
**and** thread-parallel across documents (pthreads via a small C shim,
deterministic: jobs own disjoint document ranges and results are assembled in
fixed document order). Short (3-code-unit) patterns gain least — per-query
fixed costs dominate; length 8+ patterns gain ~30x.

## How it works

```
npm install @fuse-mojo/core
        │
        ▼
@fuse-mojo/core (thin JS wrapper)
        │  index build: extract searchable strings, lowercase once,
        │  flatten to a UTF-16 buffer — per Fuse instance
        ▼
@fuse-mojo/darwin-arm64 | @fuse-mojo/linux-x64   (optionalDependencies)
        │  libfusemojo.{dylib,so} (Mojo kernel) + libfusemojoshim (pthreads)
        │  loaded with koffi, ABI-version handshaked
        ▼
batch C ABI: search_begin → search_range × threads → search_end
        │  scores all texts per pattern chunk in one call
        ▼
JS: chunk combine → key weights × field-norm scoring → sort → format
(on any load failure: transparent fallback to the vendored Fuse.js)
```

- **Batch-shaped C ABI**: one `search_begin` per pattern chunk sets up a
  pattern alphabet and per-thread scratch, `fusemojo_search_parallel` (the
  shim) fans the collection across pthreads, `search_end` folds results.
  FFI overhead is per-query, not per-document.
- **Bit-exact arithmetic**: all DP arithmetic is 32-bit two's complement and
  all scoring IEEE-754 float64 in the reference's operation order; the final
  key-weight/field-norm exponentiation runs in JS on both backends. The
  differential suite asserts scores within 1e-9; measured agreement on the
  benchmark corpus is 0 (bit-identical).
- **ABI handshake**: the wrapper checks `fusemojo_abi_version()` against its
  own expected version before searching; a mismatch falls back cleanly.
- **Self-contained platform packages**: `delocate` (macOS) / `patchelf`
  (Linux) vendor the Mojo runtime libraries into the package and rewrite load
  paths to be package-relative, so no Mojo toolchain or build-machine paths
  remain. (Redistribution terms for Modular's runtime binaries should be
  confirmed with Modular before any public release.)

## Supported options (the 90% case, bit-exact vs Fuse.js 7.1.0)

`keys` (strings, dotted paths, `{name, weight}` objects, array-valued
properties), `threshold`, `location`, `distance`, `minMatchCharLength`,
`includeScore`, `includeMatches`, `shouldSort` (default comparator),
`ignoreLocation`, `isCaseSensitive`, `findAllMatches`, `ignoreFieldNorm`,
`fieldNormWeight`, and the `limit` search parameter. Both string lists and
arrays of objects (including nested dotted key paths and array values, with
the reference's sub-record ordering) are supported.

**Unsupported** (constructor throws `UnsupportedOptionError` listing the
supported set; same behavior on both backends):

- extended search (`useExtendedSearch: true`: `'`-exact, `^` prefix, `!`
  negation, `|` OR, space-AND) and logical `$and`/`$or` query objects
- `ignoreDiacritics`
- custom `getFn` (including key-level `getFn`), custom `sortFn`
- external indices (`Fuse.createIndex` / `Fuse.parseIndex` / constructor
  `index` argument)
- non-integer `location` / `minMatchCharLength`, non-finite `threshold`

The vendored fallback is Fuse.js's *basic* build, which has the same
restrictions, so both backends behave identically.

## Unicode: UTF-16 code units, honestly

Fuse.js itself operates on **UTF-16 code units** (`String.length`,
`charAt`, `indexOf` are UTF-16 based), not Unicode code points. The kernel
does the same, so scores and match spans are bit-identical to the reference
with no index conversion — including the reference's own behavior on astral
characters (an emoji counts as two units and can match as two units; e.g.
pattern `😀b` matches text `a😀bc` at span `[1, 3]`). Match spans index into
the (case-folded) UTF-16 string exactly like Fuse.js, so `String.slice`
works on them directly. The differential suite covers accented, CJK,
Cyrillic, and emoji documents and patterns, case-sensitively and
case-insensitively.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so any load or ABI failure transparently selects the **vendored
Fuse.js 7.1.0 basic build** (`vendor/fuse.basic.cjs`, Apache-2.0, see
`NOTICE`):

- Resolution order: `$FUSE_MOJO_NATIVE_LIB` → the platform package
  (`@fuse-mojo/<platform>-<arch>` optionalDependency) → the repository
  development build output.
- `FUSE_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs this
  way as its second pass).
- `FUSE_MOJO_THREADS=N` caps native threads (default: online CPUs, max 64);
  `FUSE_MOJO_MIN_CHUNK=N` overrides the minimum documents-per-thread (2048).
- Inspect what's active: `Fuse.backendInfo()`, `Fuse.nativeAvailable()`, and
  the per-instance `fuse.backend` (`"native"` or `"fallback"`).
- The fallback is silent: no new failure modes, no new dependencies, results
  are identical (asserted by the differential suite on both backends).

## Differential tests

```
npm install            # in typescript/fuse-mojo
npm test               # native backend
npm run test:fallback  # FUSE_MOJO_DISABLE_NATIVE=1, vendored fallback
# or from the repo root: pixi run test-fuse (both passes)
```

`tests/` compares `@fuse-mojo/core` against the published `fuse.js` 7.1.0
package on deterministic seeded corpora: ~200 generated patterns (exact
tokens, typo'd tokens, substrings, multi-token spans, absent tokens, unicode,
>32-code-unit chunked patterns) across 20 option cells (threshold
0.0/0.3/1.0, location/distance, distance 0, ignoreLocation, findAllMatches,
minMatchCharLength 2/4, ignoreFieldNorm, fieldNormWeight, case
sensitivity, string lists and object lists with single/weighted/nested/array
keys), asserting identical `refIndex` **order** (the reference sort is a
total order over (score, idx), so ties cannot reorder), scores within 1e-9,
and structurally identical match spans — plus a forced-multi-job cell that
exercises the threaded stash concatenation, edge collections (blank docs,
empty pattern, single chars), the `limit` parameter, and unit tests for the
error surfaces. 42 tests pass on the native backend and 40 on the
forced-fallback backend (the 2 native-only cells skip there by design).

`tests/parity.property.test.js` adds property-based parity with
[fast-check](https://fast-check.dev): the collection (string lists, and
object lists with nested arrays, missing and blank values), the patterns
(corpus words, substrings, typos, joins, free Unicode text, chunk-boundary
lengths), every supported option and the `limit` parameter are drawn at
random, and each search is held to the same contract as above. A failure
shrinks to a minimal counterexample and prints the seed and path that replay
it (`FC_SEED` / `FC_PATH`). `npm test` runs 200 scenarios per property;
`npm run test:property` (or `pixi run fuzz-fuse` for both backends) honours
`FC_NUM_RUNS`, which the nightly `fuzz` workflow raises to 50,000.

## Repository layout (this kernel)

```
kernels/fuse/src/fusemojo.mojo        # clean-room Bitap Mojo kernel, batch C ABI
kernels/fuse/src/shim.c               # pthread fan-out shim (deterministic ranges)
kernels/fuse/build.sh                 # mojo build --emit shared-lib + cc shim
typescript/fuse-mojo/packages/core/   # @fuse-mojo/core: JS API, koffi loader,
                                      #   vendored Fuse.js fallback, NOTICE
typescript/fuse-mojo/packages/<platform>-<arch>/  # prebuilt native libs (packed +
                                      #   repaired by scripts/pack-platform.sh)
typescript/fuse-mojo/tests/           # differential suite vs published Fuse.js
typescript/fuse-mojo/quickstart.mjs   # Fuse.js README example (smoke test)
benchmarks/bench_fuse.mjs             # seeded reproducible benchmark
.github/workflows/ci-fuse.yml         # ubuntu + macOS: build → test → pack → hermetic smoke
```

## License

Apache-2.0, © 2026 Algenta The kernel is a clean-room implementation of
the textbook Bitap/Shift-And algorithm; Fuse.js is vendored as the fallback
backend and test oracle under its own Apache-2.0 license (see `NOTICE`).
