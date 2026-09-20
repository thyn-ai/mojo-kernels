# natural-mojo

**Drop-in faster replacement for [natural](https://github.com/NaturalNode/natural)'s
`LevenshteinDistance` and `DamerauLevenshteinDistance`** — the same API, the
same results, powered by a clean-room edit-distance kernel written in
[Mojo](https://www.modular.com/mojo). Prebuilt per-platform native libraries,
a thin Node wrapper, and the vendored natural implementation itself as an
automatic fallback on platforms without a native build (e.g. Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)). No
Mojo toolchain required at install time.

```js
// npm install @natural-mojo/core   →   then use it exactly like natural
const { LevenshteinDistance, DamerauLevenshteinDistance } = require('@natural-mojo/core')

LevenshteinDistance('kitten', 'sitting')                          // 3
LevenshteinDistance('kitten', 'sitting', { substitution_cost: 2 }) // 5
DamerauLevenshteinDistance('ca', 'abc')                           // 2 (unrestricted)
DamerauLevenshteinDistance('ca', 'abc', { restricted: true })     // 3 (OSA)
```

ESM works too: `import { LevenshteinDistance } from '@natural-mojo/core'`.

## Benchmark

Measured with `benchmarks/bench_natural.mjs` (a correctness gate asserting
**exact** float64 equality across ~3,000 distance comparisons — all variants,
integer and fractional costs — runs before every timing pass; measured
agreement is exactly 0, the distances are bit-identical, not approximations).
Pairs are generated locally from fixed seeds; warm steady state is the median
of 5 runs of per-pair calls; **cold** is a fresh subprocess with module load
and the first call after load measured separately, median of 5 subprocesses.
Environment: **Apple M4 Max (16 threads), macOS arm64, Node v23.10.0,
natural 8.1.1 (npm), Mojo 1.1.0**, 2026-09-19.

### Why natural is slow here (measured, not assumed)

natural's distance function builds an (n+1)×(m+1) matrix of **JavaScript
objects** — one heap-allocated `{cost, parentCell}` cell per entry, plus a
3-4 element `possibleParents` array of objects per cell and an `underscore`
`_.min` call per cell. At 62 code units that is ~3,800 cells and ~460 µs per
call; at 2,000 code units ~4M cells and ~1.5 s per call. The parent-cell
bookkeeping exists only for the substring-search variants
(`LevenshteinDistanceSearch`), which plain distance calls never use — but
every call pays for it. natural-mojo moves exactly that dynamic program into
a compiled Mojo kernel over two float64 rows (three for OSA; a full matrix
only where the unrestricted Damerau recurrence actually needs one), with
identical arithmetic and no per-cell allocation.

### Distance — warm steady state (median of 5 runs)

| pair size (code units) | variant | natural ms/call | natural-mojo ms/call | speedup |
|---:|---|---:|---:|---:|
| 8 | lev | 0.015 | 0.002 | 7.6x |
| 8 | dlu | 0.013 | 0.003 | 3.7x |
| 8 | dlr | 0.010 | 0.001 | 6.4x |
| 64 | lev | 0.437 | 0.013 | 33.0x |
| 64 | dlu | 0.768 | 0.020 | 38.7x |
| 64 | dlr | 0.489 | 0.014 | 34.5x |
| 512 | lev | 70.163 | 1.380 | 50.9x |
| 512 | dlu | 110.026 | 1.706 | 64.5x |
| 512 | dlr | 80.267 | 1.311 | 61.2x |
| 1,024 | lev | 491.630 | 14.410 | 34.1x |
| 1,024 | dlu | 461.303 | 7.362 | 62.7x |
| 1,024 | dlr | 293.795 | 5.183 | 56.7x |

### Distance — cold (fresh process, median of 5 subprocesses): module load vs first call after load

| pair size (code units) | natural require (ms) | natural-mojo require (ms) | natural first call (ms) | natural-mojo first call (ms) |
|---:|---:|---:|---:|---:|
| 8 | 3,160.5 | 16.8 | 0.440 | 57.784 |
| 512 | 1,880.6 | 5.1 | 122.516 | 19.518 |

`require('natural')` loads the whole natural dependency tree (~100 packages,
including dotenv) and is environment-dependent — on this machine it costs
seconds of mostly I/O wait; an end user without that tree's quirks pays less.
The kernel-relevant cold cost is the first call after load, where
natural-mojo pays dlopen + ABI handshake + scratch allocation once (then
every subsequent call is the warm number above).

Where the speed comes from: natural allocates several objects per matrix
cell in an interpreter loop; the kernel computes the same recurrence over
contiguous float64 rows with zero per-cell allocation. Tiny pairs (8 code
units) are bounded by FFI crossing (~1 µs), so the gap is smallest there;
from ~64 code units up the allocation-dominated oracle falls behind
quadratically while the kernel scales linearly per cell.

## How it works

```
npm install @natural-mojo/core
        │
        ▼
@natural-mojo/core (thin JS wrapper)
        │  validate strings, merge options with the reference's exact
        │  semantics (isNaN cost defaults, NaN-disables-transpositions)
        ▼
@natural-mojo/darwin-arm64 | @natural-mojo/linux-x64   (optionalDependencies)
        │  libnaturalmojo.{dylib,so} (Mojo kernel), loaded with koffi,
        │  ABI-version handshaked
        ▼
stateless C ABI: one naturalmojo_distance() call per pair
        │  UTF-16 code units in, float64 distance out
        ▼
(on any load failure: transparent fallback to the vendored natural code)
```

- **Stateless C ABI**: one FFI call per distance; the kernel allocates and
  frees its scratch inside the call, so there is no handle lifecycle, no
  finalizer, and no thread-safety concern. Strings are passed as UTF-16 code
  units — exactly what JavaScript compares — so astral characters behave
  identically to the reference (an emoji is two units).
- **Bit-exact arithmetic**: all arithmetic is IEEE-754 float64 in the
  reference's operation order; the differential suite asserts exact equality
  (not a tolerance). With the default unit costs every result is an integer.
- **ABI handshake**: the wrapper checks `naturalmojo_abi_version()` against
  its own expected version before computing; a mismatch falls back cleanly.
- **Self-contained platform packages**: `delocate` (macOS) / `patchelf`
  (Linux) vendor the Mojo runtime libraries into the package and rewrite load
  paths to be package-relative, so no Mojo toolchain or build-machine paths
  remain. (Redistribution terms for Modular's runtime binaries should be
  confirmed with Modular before any public release.)

## Supported surface (bit-exact vs natural 8.1.1)

- `LevenshteinDistance(source, target, options?)` with `insertion_cost`,
  `deletion_cost`, `substitution_cost` (defaulted with the reference's
  `isNaN` coercion: undefined/NaN → 1, `null` → 0, `true` → 1)
- `DamerauLevenshteinDistance(source, target, options?)` with the above plus
  `transposition_cost` (default 1 **only when the key is absent** — an
  explicit `undefined`/`NaN` disables transpositions, exactly like the
  reference's NaN-min behavior) and `restricted` (`true` = OSA adjacent
  transpositions, `false`/default = unrestricted Lowrance-Wagner)
- diacritics compared as-is (no normalization), empty strings, astral
  characters as two UTF-16 code units, fractional and zero costs
- `damerau` / `search` keys in user options are overridden per function,
  exactly like the reference; unknown option keys are ignored

**Unsupported** (fail loudly, same on both backends):

- `LevenshteinDistanceSearch` / `DamerauLevenshteinDistanceSearch`
  (substring search) — stubs throw `UnsupportedOptionError`; everything else
  in the `natural` package (tokenizers, stemmers, classifiers, …) is out of
  scope for this package
- non-string inputs throw `TypeError` (the reference's behavior on
  non-strings is an unspecified crash and is not mirrored); cost values that
  are non-numeric strings (e.g. `substitution_cost: "2"`) corrupt the
  reference's matrix into string arithmetic — natural-mojo coerces them with
  `Number()` instead. Both are garbage-in cases outside the parity contract.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so any load or ABI failure transparently selects the **vendored
natural 8.1.1 distance module** (`vendor/natural_distance.cjs`, MIT, see
`NOTICE`; the only adaptation is inlining `underscore`'s `extend`/`min` so
the package is self-contained):

- Resolution order: `$NATURAL_MOJO_NATIVE_LIB` → the platform package
  (`@natural-mojo/<platform>-<arch>` optionalDependency) → the repository
  development build output.
- `NATURAL_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Inspect what's active: `naturalMojo.backendInfo()` and
  `naturalMojo.nativeAvailable()`.
- The fallback is silent: no new failure modes, no new dependencies, results
  are identical (asserted by the differential suite on both backends).

## Differential tests

```
npm install            # in typescript/natural-mojo
npm test               # native backend
npm run test:fallback  # NATURAL_MOJO_DISABLE_NATIVE=1, vendored fallback
# or from the repo root: bash scripts/test_all_natural.sh (both passes)
```

`tests/` compares `@natural-mojo/core` against the published `natural`
8.1.1 package on deterministic seeded pairs: ~200 generated word pairs
(identical, typo'd, unrelated, prefix/truncated, empty, unicode, block
moves) across 15 option cells (default, `substitution_cost` 0/2,
insertion/deletion cost splits, fractional costs, `transposition_cost`
3/5, restricted and unrestricted), 80 transposition-heavy pairs engineered
to make OSA and unrestricted Damerau disagree, unicode-heavy pairs
(diacritics, emoji, CJK), an all-pairs edge sweep over 20 edge strings
(empty, single chars, digraph reversals, `café`/`cafe`, astral pairs,
64/65-code-unit runs) × 5 option cells, long strings (300-1200 code units),
and a curated option-merge cell pinning the reference's exact quirks
(explicit `undefined`/`NaN`/`null` costs, truthy/falsy `restricted`,
overridden `damerau`/`search` keys, non-object options) — all asserted for
**exact equality** on both backends, plus unit tests for the error surfaces,
backend selection, and resolver behavior. 47 tests pass on the native
backend and 44 on the forced-fallback backend (the 3 native-only tests skip
there by design).

## Repository layout (this kernel)

```
kernels/natural/src/naturalmojo.mojo    # clean-room edit-distance Mojo kernel, stateless C ABI
kernels/natural/build.sh                # mojo build --emit shared-lib
typescript/natural-mojo/packages/core/  # @natural-mojo/core: JS API, koffi loader,
                                        #   vendored natural fallback, NOTICE
typescript/natural-mojo/packages/<platform>-<arch>/  # prebuilt native libs (packed +
                                        #   repaired by scripts/pack-platform.sh)
typescript/natural-mojo/tests/          # differential suite vs published natural
typescript/natural-mojo/quickstart.mjs  # natural README examples (smoke test)
benchmarks/bench_natural.mjs            # seeded reproducible benchmark
scripts/test_all_natural.sh             # both test passes from the repo root
.github/workflows/ci-natural.yml        # ubuntu + macOS: build → test → pack → hermetic smoke
```

## License

Apache-2.0, © 2026 Algenta The kernel is a clean-room implementation of the
textbook Levenshtein / OSA / Lowrance-Wagner dynamic programs; natural's
distance module is vendored as the fallback backend and test oracle under
its own MIT license (see `NOTICE`).
