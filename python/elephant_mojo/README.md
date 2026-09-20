# elephant-mojo

**Fast SPADE surrogate dithering and p-value spectra for
[elephant](https://python-elephant.org/)** — the Electrophysiology
Analysis Toolkit's SPADE synchronous-pattern analysis — powered by a Mojo
kernel, with a vendored NumPy fallback. It accelerates the two functions
behind SPADE's surrogate-based significance evaluation:

- `dither(...)` — the `_generate_binned_surrogates` surrogate path
  (`dither_spikes`, optionally with a refractory period, followed by
  binning);
- `pvalue_spectrum(...)` — the deterministic p-value-spectrum reduction
  (`_get_pvalue_spec`) over the per-surrogate maximal-occurrence matrix.

Prebuilt per-platform binaries mean **no Mojo toolchain is ever required**
on an end user's machine. elephant itself is *not* a dependency: it is
the test oracle, never imported at runtime. NumPy is the only hard
dependency.

## The parity contract (read this first)

The two functions have different parity contracts, by construction:

1. **`pvalue_spectrum` is deterministic and BIT-EXACT** against
   elephant's `spade._get_pvalue_spec` on both backends: same signature
   integers, same float64 p-values (compared with `==` in the
   differential suite, including an end-to-end pass through elephant's
   real `spade.pvalue_spectrum` pipeline with a pinned surrogate
   generator). The histogram bin edges, the int16 truncation of the
   column maximum, the closed last bin, the reverse cumsum, and the
   division by the *given* `n_surr` are all mirrored operation for
   operation.
2. **`dither` is stochastic — RNG-sequence parity across implementations
   is impossible**, so the contract is (a) *same-seed reproducibility
   within a backend* (identical output arrays for identical seeds) and
   (b) *statistical equivalence with the reference*: KS tests on
   single-spike displacement and survivor-count distributions
   (p > 1e-3), per-bin occupancy two-proportion z-tests with Bonferroni
   correction, edge drop/clamp rates within 2%, and exact structural
   invariants (shape, dtype, rate, refractory bin separation). The
   kernels draw from an in-kernel xoshiro256\*\* stream (seeded per
   surrogate/train via SplitMix64); the fallback draws from NumPy's
   PCG64 — sequences differ between backends, distributions do not.

## Install

```
pip install elephant-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows, where CI runs this package's fallback suite
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)) —
the same wheel API runs on the vendored NumPy fallback, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import numpy as np
from elephant_mojo import dither, pvalue_spectrum

# Two spike trains (seconds), sorted ascending within [0, 1]:
spiketrains = [np.array([0.012, 0.105, 0.240, 0.402, 0.555, 0.731, 0.908]),
               np.array([0.040, 0.180, 0.333, 0.510, 0.666, 0.845])]

# 100 binned surrogates per train — the dither_spikes + binning pipeline
# SPADE's significance evaluation consumes (5 ms bins, 15 ms dither):
surrogates = dither(spiketrains, bin_size=0.005, dither=0.015,
                    n_surrogates=100, t_start=0.0, t_stop=1.0, seed=42)
# -> bool array (n_surrogates=100, n_trains=2, n_bins=200), the
#    `to_bool_array()` form SPADE's pattern mining consumes.

# P-value spectrum from the per-surrogate maximal-occurrence matrix
# (n_surr=100, pattern sizes 2..4) — deterministic, bit-exact vs elephant:
max_occs = ...  # see the recipe below
pv = pvalue_spectrum(max_occs, min_spikes=2, max_spikes=4, min_occ=2)
# -> [[pattern_size, pattern_occ, p_value], ...] for spectrum='#'
#    [[pattern_size, pattern_occ, pattern_dur, p_value], ...] for '3d#'
```

A runnable version is `examples/quickstart.py` in this package directory:

```
python quickstart.py
```

### Where `max_occs` comes from (elephant recipe)

`pvalue_spectrum` mirrors elephant's `_get_pvalue_spec`, whose input is
the per-surrogate matrix of *maximal pattern occurrences* — for every
surrogate, the highest occurrence count of any mined pattern of a given
size (and duration), cascaded over sizes. With elephant's own mining:

```python
import elephant.spade as espade

max_occs = np.zeros((n_surr, max_spikes - min_spikes + 1))
for k, binned_surrogate in enumerate(surrogates):  # e.g. from dither(...)
    bst = elephant.conversion.BinnedSpikeTrain(
        binned_surrogate, bin_size=bin_size, t_start=t_start, t_stop=t_stop,
        tolerance=None)
    concepts = espade.concepts_mining(
        bst, bin_size, winlen, min_spikes=min_spikes, max_spikes=max_spikes,
        min_occ=min_occ, min_neu=1, report=spectrum)[0][:, :-1]
    max_occs[k] = espade._get_max_occ(
        concepts, min_spikes, max_spikes, winlen, spectrum)
```

Concept mining (FIM/FCA) itself stays in elephant — it is not the hot
loop this package accelerates.

## API

### `dither(spiketrains, bin_size, dither, n_surrogates=1, *, t_start=0.0, t_stop=None, method='dither_spikes', edges=True, refractory_period=None, seed=None)`

- `spiketrains`: list of 1-D arrays of spike times (sorted ascending,
  within `[t_start, t_stop]`), same time unit as the other arguments.
- `bin_size` (> 0), `dither` (>= 0), `t_start`, `t_stop` (required):
  binning geometry. `n_bins = int((t_stop - t_start) / bin_size)` —
  truncation, mirroring `BinnedSpikeTrain(tolerance=None)`; a spike
  landing exactly on `t_stop` is discarded by the binning, exactly like
  the reference.
- `method='dither_spikes'` (default): every spike is displaced by an
  independent uniform offset in `[-dither, +dither)`. `edges=True` (the
  reference default) drops spikes dithered outside `(t_start, t_stop)`;
  `edges=False` clamps them to the range ends.
- `method='dither_spikes_with_refractory_period'` (requires
  `refractory_period`): each spike's dither range is shrunk so it cannot
  enter the effective refractory period
  `min(refractory_period, smallest ISI)` of its current neighbours;
  spikes are perturbed in a uniform random order and cannot cross.
- `seed`: explicit integer for reproducible output on the same backend;
  `None` draws OS entropy (the one non-deterministic input).
- Returns a boolean array `(n_surrogates, n_trains, n_bins)`.

### `pvalue_spectrum(max_occs, min_spikes, max_spikes, min_occ, n_surr=None, winlen=1, spectrum='#')`

- `max_occs`: float matrix `(n_surr, max_spikes - min_spikes + 1)` for
  `'#'`, or cube `(n_surr, n_sizes, winlen)` for `'3d#'`.
- `n_surr`: p-value denominator; defaults to the row count, which is what
  the reference pipeline passes. An explicit override divides by the
  given count, exactly like the reference.
- For `'#'` the reference forces `winlen` to 1; so does this function.
- Returns the entries in the reference emission order (size, then
  duration, then occurrence ascending).

Introspection: `elephant_mojo.backend_info()` and
`elephant_mojo.native_available()` report the active backend.
`ELEPHANT_MOJO_DISABLE_NATIVE=1` forces the fallback;
`ELEPHANT_MOJO_NATIVE_LIB` overrides the native library path.

## Benchmark

Measured with `benchmarks/bench_elephant_surrogates.py` in this
repository. Workloads: plain dither **10 trains x 500 spikes x 200
surrogates** (1000 ms, 5 ms bins); refractory dither **5 trains x 500
spikes x 50 surrogates**; p-value spectrum on a **(2000, 20)** maximal-
occurrence matrix (~6000 entries). Median of 5 warm runs; cold = first
call in a fresh interpreter (import + dlopen + first call). Environment:
**Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14, numpy 2.5.3, Mojo
1.1.0, elephant 1.2.1**, 2026-09-19.

| workload | elephant 1.2.1 warm (s) | elephant-mojo cold (s) | elephant-mojo warm (s) | warm speedup |
|---|---:|---:|---:|---:|
| plain dither | 0.371 | 0.0247 | 0.001087 | 342x |
| refractory dither | 0.114 | 0.0237 | 0.000522 | 218x |
| pvalue spectrum | 0.002 | 0.0223 | 0.001131 | 2x |
| plain dither — NumPy fallback | 0.371 | — | 0.0175 | 21.2x |
| refractory dither — NumPy fallback | 0.114 | — | 0.0893 | 1.3x |
| pvalue spectrum — NumPy fallback | 0.002 | — | 0.003388 | 0.6x |

Correctness gate (asserted before every timing run): survivor-rate
agreement within 1% relative against the oracle for both dither cells
(measured diff 2.5e-04 and 1.0e-03), and bit-exact spectrum equality.

Honest framing:

- The dither cells are the hot loops: elephant builds a full
  `BinnedSpikeTrain` per surrogate (plain path) and runs a pure-Python
  loop over surrogates x spikes (refractory path); the Mojo kernel does
  one pass over flat arrays with an in-kernel RNG — hence 200-350x.
- The p-value spectrum is tiny compute (histograms over at most a few
  thousand values); both sides are millisecond-scale and the measured
  speedup (1.7x) mostly removes Python-loop overhead. The NumPy fallback
  is *slower* than the oracle here (0.6x) because the wrapper validates
  and converts inputs the oracle does not — reported as measured.
- The refractory fallback (1.3x) mirrors the reference's own
  Python-loop structure, so it lands in the same cost class; the plain
  fallback vectorizes the per-surrogate draws (21x).

## How it works

```
pip install elephant-mojo
        │
        ▼
elephant_mojo (thin Python wrapper)
        │  validates trains/matrices, flattens to typed arrays
        ▼
libelephantsurrogatesmojo.dylib / .so     (Mojo kernel, C ABI v1)
        │  elephantsurrogatesmojo_dither: whole surrogate set per call
        │  elephantsurrogatesmojo_pvalue_spec_{count,fill}: whole matrix
        ▼
bool surrogate block / spectrum entries in the reference order
```

- **Batch-shaped C ABI**: one call generates all surrogates for all
  trains (or the whole spectrum); FFI overhead is per-call, not
  per-spike.
- **In-kernel xoshiro256\*\* RNG**, seeded per (surrogate, train) via
  SplitMix64: deterministic for a given seed, no global state, safe under
  concurrent use.
- **Reference-matching semantics**: bin indices by truncation
  (`BinnedSpikeTrain(tolerance=None)`), spikes on `t_stop` discarded,
  strict `(t_start, t_stop)` edge dropping, clamp-then-bin for
  `edges=False`, effective refractory `min(given, smallest ISI)` with
  uniform perturbation order, int16-truncated histogram upper edge,
  closed last bin, reverse-cumulated counts divided by the given
  `n_surr`.
- **ABI handshake**: the wrapper checks
  `elephantsurrogatesmojo_abi_version()` before calling; a mismatch falls
  back cleanly.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored NumPy reference**
(`elephant_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$ELEPHANT_MOJO_NATIVE_LIB` → the library bundled in
  the wheel → the repo development build output.
- `ELEPHANT_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite
  runs this way as its second pass).
- Both backends share validation and flattening in
  `elephant_mojo/core.py`, so they cannot disagree about inputs; the
  differential suite asserts both against elephant.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `ELEPHANT_MOJO_ALLOW_PURE_WHEEL=1` (the wheel CI builds and tests on Windows: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

## Differential tests

```
PYTHONPATH=python/elephant_mojo pixi run bash scripts/test_all_elephant_surrogates.sh
# builds nothing itself: compile the kernel first with
#   pixi run bash kernels/elephant-surrogates/build.sh
# runs the suite twice: once native, once with ELEPHANT_MOJO_DISABLE_NATIVE=1
```

The suite (`tests/test_elephant_surrogates_*.py`) compares elephant-mojo
against the published oracle (elephant==1.2.1, installed by the test
runner; never a runtime dependency). 44 tests pass per backend (88 per
full run):

- **Bit-exact spectrum parity** (tolerance: exact equality): randomized
  matrices and cubes, degenerate columns (maximum below `min_occ`,
  all-zero), bin-edge values, single-surrogate and `n_surr` overrides,
  large occurrence counts (30 000), and an end-to-end pass through
  elephant's real `spade.pvalue_spectrum` pipeline with the surrogate
  generator pinned to this package's `dither` output.
- **Statistical dither equivalence** (KS p > 1e-3; occupancy z-tests,
  Bonferroni alpha 0.05/#bins; drop/clamp rates within 2%):
  single-spike displacement at center and both edges, survivor counts,
  per-bin occupancy, `edges=False` clamping, refractory dithering, plus
  exact invariants (shape/dtype/rate, refractory bin separation,
  same-seed reproducibility).
- **Validation and loader behaviour**: structured fail-fast errors, ABI
  handshake, fallback forcing, broken-override resilience.

## Scope and limitations

- Surrogate methods: `dither_spikes` (with both edge modes) and
  `dither_spikes_with_refractory_period` — the SPADE dithering paths.
  elephant's other surrogate methods (`bin_shuffling`,
  `joint_isi_dithering`, `isi_dithering`, `uniform_randomisation`, ...)
  are out of scope.
- Concept mining (FIM/FCA) and the maximal-occurrence reduction stay in
  elephant; this package accelerates the surrogate generation and the
  deterministic spectrum reduction around them (recipe above).
- Spectrum parity is guaranteed for maximal occurrences <= 32 765: past
  that, the reference's own int16 arithmetic overflows (numpy emits a
  RuntimeWarning and produces a degenerate edge array); the native
  kernel computes in wide arithmetic and does not reproduce the overflow
  artifact. The vendored fallback mirrors the reference even there.
- `spectrum='#'` always forces `winlen = 1`, like the reference.
- Single-threaded kernel (determinism first); multithreading is future
  work and would multiply the dither speedup.
- elephant is *not* a dependency: inputs are plain arrays, so the package
  also works standalone for any binned spike-train surrogate task.

## Citing

If you use elephant-mojo in academic work, please cite both elephant and
this package:

> elephant-mojo: Mojo-accelerated SPADE surrogate dithering and p-value
> spectra for elephant. thyn-ai, 2026.
> https://github.com/thyn-ai/mojo-kernels
> (citation file with DOI forthcoming)

elephant itself: the SPADE method is from Torre et al., *Statistical
evaluation of synchronous spike patterns extracted by frequent item set
mining*, Front. Comput. Neurosci. 7:132 (2013), and Quaglio et al.,
*Multiple-comparison correction for spike pattern analysis*, Front.
Comput. Neurosci. 11:41 (2017).

## License

Apache-2.0, © 2026 Algenta The kernel and wrapper are clean-room
implementations of the published surrogate-dithering and p-value-spectrum
algorithms (Torre et al. 2013; Quaglio et al. 2017). elephant is used
only as the test/benchmark oracle, never as a runtime dependency.
