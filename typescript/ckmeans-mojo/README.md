# ckmeans-mojo

**Drop-in faster `ckmeans` for [simple-statistics](https://simple-statistics.github.io/)** —
the same API, exactly the same cluster assignments, powered by a clean-room
Ckmeans.1d.dp kernel written in [Mojo](https://www.modular.com/mojo). Prebuilt
per-platform native libraries, a thin Node wrapper, and the vendored
simple-statistics implementation itself as an automatic fallback on platforms
without a native build (e.g. Windows). No Mojo toolchain required at install
time.

```js
// npm install @ckmeans-mojo/core   →   then use it exactly like simple-statistics' ckmeans
import ckmeans from '@ckmeans-mojo/core'

ckmeans([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3)
// → [[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]]
```

CommonJS works too: `const ckmeans = require('@ckmeans-mojo/core')`.

## Benchmark

Measured with `benchmarks/bench_ckmeans.mjs` (a correctness gate asserting
**exact cluster-assignment equality** runs before every timing pass — measured
agreement here is exactly zero differences, on every cell). Datasets are
generated locally from fixed seeds: 4-blob mixtures (realistic clustering) and
small-integer grids (heavy DP ties), at n = 1k / 5k / 20k with k = 3 / 10;
median of 5 timed calls after a 2-call warmup. **Cold** is a fresh node
process loading the package and making its first call (koffi `dlopen` + JIT
included), median of 5 spawns. Environment: **Apple M4 Max (16 threads),
macOS arm64, Node v23.10.0, simple-statistics 7.12.0 (npm), Mojo 1.1.0**,
2026-09-19.

### Autopsy (why simple-statistics' ckmeans is slow — measured, not assumed)

V8 CPU profile (`node --cpu-prof`) of simple-statistics 7.12.0 clustering the
20k mixture dataset (k = 10) 60 times, self time by function:

| self time | function | what it is |
|---:|---|---|
| 69.8% | `fillMatrixColumn` (simple-statistics.mjs:182) | the DP cell fill (all V8 code tiers combined) |
| 7.1% | `numericSort` (simple-statistics.mjs:103) | comparator-callback sort of the input |
| 5.1% | `makeMatrix` (simple-statistics.mjs:74) | per-call k×n boxed-array matrix allocation |
| 3.4% | (garbage collector) | allocation churn |
| <5% | everything else | ssq inlining residue, backtrack, slicing |

Per the §11 classification this is **not** a naive interpreter loop: V8's JIT
compiles the tight float64 DP to decent machine code, so the honest available
win is single-digit, not 10-100x. ckmeans-mojo removes the parts V8 cannot
optimize away — the comparator sort (native stable merge sort inside the
kernel), the per-call k×n boxed-matrix allocation and GC churn (flat native
buffers, two cost rows), and the remaining interpreter/JIT overhead — and
keeps the arithmetic bit-identical (`--fp-mode contract=off`: no FMA fusion,
so every split-point comparison sees the same float64 operands as V8).

### ckmeans — warm steady state (median of 5 timed calls)

| dataset | n | k | simple-statistics ms/call | ckmeans-mojo ms/call | simple-statistics calls/s | ckmeans-mojo calls/s | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixture | 1,000 | 3 | 0.333 | 0.208 | 3,003.4 | 4,809.6 | 1.6x |
| mixture | 1,000 | 10 | 1.101 | 0.398 | 908.5 | 2,513.1 | 2.8x |
| mixture | 5,000 | 3 | 1.733 | 0.719 | 577.0 | 1,390.9 | 2.4x |
| mixture | 5,000 | 10 | 4.875 | 1.428 | 205.1 | 700.5 | 3.4x |
| mixture | 20,000 | 3 | 7.075 | 2.641 | 141.3 | 378.7 | 2.7x |
| mixture | 20,000 | 10 | 16.977 | 5.624 | 58.9 | 177.8 | 3.0x |
| int-grid | 1,000 | 3 | 0.154 | 0.065 | 6,509.4 | 15,384.6 | 2.4x |
| int-grid | 1,000 | 10 | 0.408 | 0.160 | 2,453.5 | 6,238.7 | 2.5x |
| int-grid | 5,000 | 3 | 0.782 | 0.379 | 1,279.1 | 2,636.8 | 2.1x |
| int-grid | 5,000 | 10 | 2.948 | 1.026 | 339.2 | 974.4 | 2.9x |
| int-grid | 20,000 | 3 | 4.853 | 2.970 | 206.1 | 336.7 | 1.6x |
| int-grid | 20,000 | 10 | 15.763 | 5.200 | 63.4 | 192.3 | 3.0x |

### ckmeans — cold first call (fresh node process: module load + first call, median of 5 spawns)

| dataset | n | k | simple-statistics cold (ms) | ckmeans-mojo cold (ms) | speedup |
|---|---:|---:|---:|---:|---:|
| mixture | 1,000 | 3 | 7.6 | 54.7 | 0.1x |
| mixture | 1,000 | 10 | 15.4 | 26.2 | 0.6x |
| mixture | 5,000 | 3 | 14.9 | 16.3 | 0.9x |
| mixture | 5,000 | 10 | 18.0 | 22.3 | 0.8x |
| mixture | 20,000 | 3 | 21.9 | 19.0 | 1.2x |
| mixture | 20,000 | 10 | 37.2 | 23.3 | 1.6x |
| int-grid | 1,000 | 3 | 6.8 | 24.9 | 0.3x |
| int-grid | 1,000 | 10 | 9.9 | 65.5 | 0.2x |
| int-grid | 5,000 | 3 | 31.9 | 74.9 | 0.4x |
| int-grid | 5,000 | 10 | 30.9 | 51.5 | 0.6x |
| int-grid | 20,000 | 3 | 36.2 | 31.1 | 1.2x |
| int-grid | 20,000 | 10 | 53.2 | 64.5 | 0.8x |

Cold reads honestly: the first native call pays a one-time ~11 ms
(`dlopen` of the kernel through koffi plus JIT; `require` itself is ~5 ms —
measured on this machine, median of 5), so a process that clusters one small
array once is slower end-to-end; any workload past the first calls recovers
it within milliseconds (steady-state calls at n = 1,000 cost 0.02-0.4 ms).
The cold columns above also include node-process startup noise (fresh spawn
per measurement), which is why adjacent cells vary.

## How it works

```
npm install @ckmeans-mojo/core
        │
        ▼
@ckmeans-mojo/core (thin JS wrapper)
        │  validation identical to the reference (k > n error, ceil row
        │  count, single-unique-value fast path), one conversion pass
        ▼
@ckmeans-mojo/darwin-arm64 | @ckmeans-mojo/linux-x64   (optionalDependencies)
        │  libckmeansmojo.{dylib,so} (Mojo kernel), loaded with koffi,
        │  ABI-version handshaked
        ▼
batch C ABI: stable merge sort (native) → median-shifted Ckmeans.1d.dp
        │  divide-and-conquer DP → backtrack → cluster-left boundaries
        ▼
JS: slice the kernel-sorted buffer into plain-array clusters
(on any load failure: transparent fallback to the vendored simple-statistics)
```

- **Exact parity by construction**: the kernel replicates the reference
  algorithm (Wang & Song Ckmeans.1d.dp 3.4.6 divide-and-conquer fill:
  midpoint-first cell order, backtrack-narrowed split-point scans,
  descending scan with strict `<` updates, so ties resolve to the largest
  split point exactly like the reference), including the median shift and
  the two-branch within-cluster sum-of-squares with its negative clamp. All
  float64 operations run in the reference's operation order with FMA
  contraction disabled, so assignments agree exactly — not within a
  tolerance.
- **Native stable sort**: finite data is sorted inside the kernel by a
  stable merge sort (±0 compares equal, original relative order kept), which
  is bit-identical to the reference's stable `Array` sort but ~5x faster
  than comparator sorting. NaN-containing input takes a JS-sort path so NaN
  ordering keeps the engine's own comparator semantics.
- **ABI handshake**: the wrapper checks `ckmeansmojo_abi_version()` against
  its own expected version before clustering; a mismatch falls back cleanly.
- **Self-contained platform packages**: `delocate` (macOS) / `patchelf`
  (Linux) vendor the Mojo runtime libraries into the package and rewrite load
  paths to be package-relative, so no Mojo toolchain or build-machine paths
  remain. (Redistribution terms for Modular's runtime binaries should be
  confirmed with Modular before any public release.)

## API surface (full parity with simple-statistics 7.12.0 `ckmeans`)

`ckmeans(x, nClusters)` — clusters `x` (not mutated) into `nClusters`
groups, returned as ascending plain arrays. The reference's exact behavior
is preserved, including:

- `k = 1` (one cluster), `k = n` (singletons), unsorted input, duplicates,
  negative values, large offsets (the median shift is replicated)
- constant arrays collapse to a single cluster for any valid `k`
  (`[[7, 7, 7]]` for `k = 2`)
- non-integer `k` behaves like the reference's matrix row count
  (`k = 2.5` → 3 clusters; `k = 3.5` with `n = 3` throws)
- errors: `k > n` throws `Error: cannot generate more classes than there are
  data values`; `k <= 0` / `NaN` / `null` throw the reference's
  `TypeError: Cannot read properties of undefined (reading 'length')`

**Out of scope**: the rest of simple-statistics (this package is exactly
`ckmeans`), and Windows/native-unsupported platforms (the vendored fallback
is the product there, bit-identical results). NaN/±Infinity/non-numeric
array elements are garbage-in for the reference itself; ckmeans-mojo
reproduces the reference's observable behavior on them (via the JS-sort
path) but treats them as unsupported input.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so any load or ABI failure transparently selects the **vendored
simple-statistics 7.12.0 ckmeans** (`vendor/`, ISC license, see `NOTICE`):

- Resolution order: `$CKMEANS_MOJO_NATIVE_LIB` → the platform package
  (`@ckmeans-mojo/<platform>-<arch>` optionalDependency) → the repository
  development build output.
- `CKMEANS_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Inspect what's active: `ckmeans.nativeAvailable()`, `ckmeans.backendInfo()`,
  and the `ckmeans.backend` getter (`"native"` or `"fallback"`).
- The fallback is silent: no new failure modes, no new dependencies, results
  are identical (asserted by the differential suite on both backends).

## Differential tests

```
npm install            # in typescript/ckmeans-mojo
npm test               # native backend
npm run test:fallback  # CKMEANS_MOJO_DISABLE_NATIVE=1, vendored fallback
# or from the repo root: bash typescript/ckmeans-mojo/scripts/test_all.sh (both passes)
```

`tests/` compares `@ckmeans-mojo/core` against the published
`simple-statistics` 7.12.0 package and asserts **exact cluster-assignment
equality** (identical cluster count, lengths, and SameValue-equal elements —
partitions are discrete and values pass through unmodified, so equality is
exact, not tolerant): every array of length 1-6 over {0,1,2,3} with every
valid k (30,948 cells — pins the tie-resolution convention), 2,400 seeded
random datasets across six kinds (integer grids, floats, half-step grids,
mixtures, 1e10 offsets, negatives), a 2,000-point mixture at k = 8, the
documented edge cases (k = 1, k = n, constant arrays n = 1-8, non-integer
and string k, signed zeros, error class+message parity, input non-mutation).
29 tests pass on the native backend and 29 on the forced-fallback backend.

## Repository layout (this kernel)

```
kernels/ckmeans/src/ckmeansmojo.mojo  # clean-room Ckmeans.1d.dp Mojo kernel, batch C ABI
kernels/ckmeans/build.sh              # mojo build --emit shared-lib (--fp-mode contract=off)
typescript/ckmeans-mojo/packages/core/      # @ckmeans-mojo/core: JS API, koffi
                                            #   loader, vendored fallback, NOTICE
typescript/ckmeans-mojo/packages/<platform>-<arch>/  # prebuilt native libs (packed +
                                            #   repaired by scripts/pack-platform.sh)
typescript/ckmeans-mojo/tests/        # differential suite vs published simple-statistics
typescript/ckmeans-mojo/quickstart.mjs # README example + seeded workload (smoke test)
benchmarks/bench_ckmeans.mjs          # seeded reproducible benchmark
.github/workflows/ci-ckmeans.yml      # ubuntu + macOS: build → test → pack → hermetic smoke
```

## License

Apache-2.0, © 2026 Algenta. The kernel is a clean-room implementation of the
published Ckmeans.1d.dp algorithm (Wang & Song, The R Journal Vol. 3/2,
2011); simple-statistics is vendored as the fallback backend and used as the
test oracle under its own ISC license (see `NOTICE`).
