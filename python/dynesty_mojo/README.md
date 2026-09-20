# dynesty-mojo

**Nested sampling with a [dynesty](https://dynesty.readthedocs.io/)-shaped
API, powered by a Mojo bounding/proposal kernel** — for the cheap-likelihood
regime where sampler-side Python overhead dominates the run (dynesty issue
\#432). The single-ellipsoid bound fit (mean, covariance, max-Mahalanobis
containment scaling, Cholesky factor) and the uniform-in-ellipsoid batched
proposals — the per-iteration numerical machinery — run in compiled Mojo
where the platform supports it (macOS arm64, Linux x86_64), with a vendored
NumPy fallback everywhere else, silently and correctly. **Your likelihood
and prior transform stay ordinary Python callables**, exactly as in dynesty;
prebuilt per-platform wheels mean **no Mojo toolchain is ever required** on
an end user's machine.

Statistical parity with the PyPI dynesty sampler is asserted by the
differential test suite: evidence `logz` within combined Monte Carlo error
and weighted-sample means/covariances within their MC standard errors, on
problems with analytic ground truth (correlated Gaussian, Gaussian shells).
Nested sampling is a Monte Carlo algorithm, so parity is on posterior
statistics, never on trajectories.

## Install

```
pip install dynesty-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform
the same wheel API runs on the vendored NumPy fallback. There is no sdist:
a source tarball cannot rebuild the native library. numpy is the only
runtime dependency; **dynesty itself is not a dependency** (it is the
test/benchmark oracle only).

## Quickstart

```python
import numpy as np
from dynesty_mojo import sample

ndim = 5
cov = ...                      # target covariance (ndim, ndim)
cov_inv = np.linalg.inv(cov)

def loglike(v):                # ordinary Python, as in dynesty
    return -0.5 * float(v @ cov_inv @ v)

def prior_transform(u):        # unit cube -> physical parameters
    return 20.0 * u - 10.0

res = sample(loglike, prior_transform, ndim,
             nlive=500, dlogz=0.1, seed=42)

res.logz, res.logz_err         # evidence estimate and MC error
res.samples, res.weights       # posterior-weighted samples
mean = res.weights @ res.samples
```

A runnable version (3-D Gaussian with analytic evidence check) is
`quickstart.py` next to this README:

```
python quickstart.py
```

`sample()` mirrors the essentials of dynesty's `DynamicNestedSampler` API:
`loglike`, `prior_transform`, `ndim`, `nlive=500`, `dlogz=0.01`,
`logl_max`, `maxiter`, `maxcall`, plus `enlarge=1.25` (bounding enlargement
factor), `batch=64` (candidates per kernel call — statistics are provably
independent of it), `seed` (uint64; bit-reproducible runs), and
`vectorized=False` (set True when both callbacks accept `(k, ndim)`
batches). The result is a dict-with-attributes (`NSResults`, dynesty's
convention) with `samples`, `samples_u`, `logl`, `logwt`, `weights`,
`logz`, `logz_err`, `information`, `niter`, `ncall`, `eff`, and the run
configuration.

## Benchmark

Measured with `benchmarks/bench_dynesty.py` in this repository (median of
5 runs; cold = first call in a fresh process, warm = steady state in one
process; fixed seeds; the correctness gate `|logz - logz_true| <=
4*logz_err + 0.15` passes on every arm before it is timed). Environment:
**Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14, numpy 2.5.3, Mojo
1.1.0, dynesty 3.1.0 (PyPI oracle)**, 2026-09-19.

`gauss10`: 10-D correlated Gaussian, nlive=500, dlogz=0.25
(logz_true = −40.0569):

| arm | cold (s) | warm (s) | ncall | speedup (warm) |
|---|---:|---:|---:|---:|
| dynesty 3.1.0 (oracle, defaults) | 11.342 | 11.026 | 812,809 | 1.00x |
| dynesty-mojo native (scalar) | 2.456 | 2.433 | 109,965 | 4.53x |
| dynesty-mojo native (vectorized) | 2.377 | 2.589 | 1,309,748 | 4.26x |
| dynesty-mojo NumPy fallback (vectorized) | 5.666 | 6.476 | 1,299,124 | 1.70x |

`shells2`: two Gaussian shells in 2-D (bimodal), nlive=250, dlogz=0.2
(logz_true = −3.1293; oracle arm uses its documented `bootstrap=0` remedy
for multi-modal bounds — with the default bootstrap the oracle's own
`UserWarning` fires and the same run takes ~20x longer, measured):

| arm | cold (s) | warm (s) | ncall | speedup (warm) |
|---|---:|---:|---:|---:|
| dynesty 3.1.0 (oracle, bootstrap=0) | 3.274 | 3.380 | 56,166 | 1.00x |
| dynesty-mojo native (scalar) | 0.109 | 0.097 | 12,883 | 34.71x |
| dynesty-mojo native (vectorized) | 0.074 | 0.065 | 78,394 | 51.87x |
| dynesty-mojo NumPy fallback (vectorized) | 0.181 | 0.149 | 81,082 | 22.76x |

Honest framing: this is an **overhead-class reduction**, not an algorithmic
claim. The oracle's per-iteration machinery (bound updates, proposal
generation, bookkeeping) runs as interpreted Python/NumPy around every
likelihood call; for likelihoods that cost microseconds, that machinery is
the run. dynesty-mojo moves the same machinery into the compiled kernel and
batches proposals, so one Python iteration covers a whole batch of
candidates. The scalar arm also *reduces* likelihood calls (lazy
first-passing-candidate evaluation); the vectorized arm *trades* more calls
(batched evaluation overshoots within a batch) for far fewer interpreter
steps — pick per problem. What does not change: your likelihood stays
Python, so for expensive likelihoods (milliseconds and up) the advantage
shrinks toward 1x — the kernel accelerates the sampler, not your model.

## How it works

```
pip install dynesty-mojo
        │
        ▼
dynesty_mojo (thin Python wrapper)
        │  validates args/callbacks, owns the NS loop, evidence bookkeeping
        │  (running logZ, weights, stopping rule) — all O(1) per iteration
        ▼
libdynestymojo.dylib / .so        (Mojo kernel, C ABI v1)
        │  fit_ellipsoid: mean + cov + max-Mahalanobis scaling + Cholesky
        │  propose_batch: K iid uniform-in-ellipsoid∩cube candidates/call
        ▼
Python calls back: prior_transform + loglike on candidates
(first candidate above the threshold = exact constrained-prior draw)
```

- **Batch-shaped C ABI**: one FFI call fits the ellipsoid over all live
  points; one call draws up to `batch` candidates. FFI overhead is
  per-iteration, not per-candidate.
- **Exactness of batching**: candidates within a batch are iid uniform in
  the bound, so taking the first one above the likelihood threshold is
  distribution-identical to sequential rejection sampling. `batch` changes
  interpreter overhead, never the posterior.
- **Seeded, deterministic**: xoshiro256\*\* in the kernel, PCG64 in the
  fallback; `seed=` gives bit-reproducible runs per backend (the two
  backends are statistically equivalent, not bit-identical).
- **ABI handshake**: the wrapper checks `dynestymojo_abi_version()` before
  sampling; a mismatch falls back cleanly.
- **Algorithm provenance** (clean-room, from the literature): Skilling
  (2006) for the nested-sampling recursion and evidence/information
  estimators; Mukherjee, Parkinson & Liddle (2006) and Feroz, Hobson &
  Bridges (2009, MULTINEST) for ellipsoidal rejection sampling with the
  covariance bound scaled to contain every live point, times an
  enlargement factor. No dynesty code is read, copied, or linked.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored NumPy reference**
(`dynesty_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$DYNESTY_MOJO_NATIVE_LIB` → the library bundled in
  the wheel → the repo development build output.
- `DYNESTY_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Both backends expose identical `fit_ellipsoid` / `propose_batch`
  semantics; the sampler driver is shared, so the backends cannot disagree
  about evidence bookkeeping.
- Inspect what's active: `dynesty_mojo.backend_info()` and
  `dynesty_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `DYNESTY_MOJO_ALLOW_PURE_WHEEL=1`. That wheel is not tested on Windows
  in CI, and this package makes no Windows claim: the differential
  suite's sanity check on the *oracle* itself (dynesty's own `logz`
  against the analytic truth, within 4σ + 0.15) does not hold on
  `windows-latest` for the shared seed, so the suite cannot go green
  there even though the fallback matched both the oracle and the truth
  in that run.

## Differential tests

```
# the oracle (dynesty==3.1.0) comes from the repo pixi environment
# (pixi.toml [pypi-dependencies], pinned and lock-verified):
pixi run bash kernels/dynesty/build.sh
PYTHONPATH=python/dynesty_mojo PYTHONNOUSERSITE=1 \
  pixi run bash scripts/test_all_dynesty.sh   # native pass, then forced fallback
```

The suite (`tests/test_dynesty_differential.py`,
`tests/test_dynesty_loader.py`) compares dynesty-mojo against the real
PyPI dynesty sampler AND against analytic/quadrature ground truth, so the
gates never rest on the oracle alone. Parity gates (documented in the test
module; all runs seeded, so they hold deterministically):

- `|logz_ours − logz_ref| ≤ 4·√(err_ours² + err_ref²) + 0.15`, and each
  side vs the analytic `logz` within `4·err + 0.15`;
- weighted means within `5·√(C_jj/ess)` and covariances within
  `6·√((C_jj·C_kk + C_jk²)/ess)` of truth, `ess = 1/Σw²`.

Measured agreement on this machine (2026-09-19, seeds as in the suite;
NLIVE=150, dlogz=0.5): correlated-Gaussian logz pulls vs truth −0.69σ
(native) / −0.45σ (fallback), |Δlogz vs oracle| 0.58 / 0.50 (1.5σ / 1.3σ
combined); shells +0.47σ / −1.76σ vs truth, |Δlogz vs oracle| 0.07 / 0.24
(0.5σ / 1.7σ combined); posterior means/covs all inside the gates. The
native pass runs 35 tests (30 loader/unit + 5 differential); the forced
fallback pass runs the same 35 with the 5 native-only kernel-unit tests
skipped (30 passed, 5 skipped).

## Scope and limitations

- **Single-ellipsoid bounding** (dynesty `bound='single'`-equivalent).
  Multi-ellipsoid (`bound='multi'`) decomposition, and the slice/rwalk
  proposal families, are future work — single ellipsoids stay correct on
  multi-modal posteriors (rejection via the likelihood threshold), just
  less efficient, which is why `shells2` above uses fewer live points.
- **Static-core nested sampling**: constant `nlive` with the standard
  remaining-evidence stopping rule and mean-shrinkage prior volumes
  (`log X_k = −k/nlive`). dynesty's dynamic `nlive` refinement batches are
  not reimplemented; the API mirrors `DynamicNestedSampler`'s essentials,
  not its full option surface.
- **User likelihoods stay Python** (scoped claim: sampler-overhead class
  reduction). For expensive likelihoods the speedup shrinks toward 1x.
- `logz_err` is the run-internal `sqrt(H/nlive)` information estimate
  (Skilling 2006); like all such estimates it excludes shrinkage variance.
- Single-threaded kernel (determinism first).
- Periodic/reflective boundary conditions and `logl_args`-style extra
  callback arguments are not supported (wrap with `functools.partial`).

## Citing

If you use dynesty-mojo in academic work, please cite both dynesty and
this package:

> dynesty-mojo: Mojo-accelerated bounding/proposal machinery for nested
> sampling with a dynesty-shaped API. thyn-ai, 2026.
> https://github.com/thyn-ai/mojo-kernels (citation file with DOI
> forthcoming)

dynesty itself: J. S. Speagle, *dynesty: a dynamic nested sampling package
for estimating Bayesian posteriors and evidences*, MNRAS 493 (2020) 3132.
Nested sampling: J. Skilling, *Nested sampling for general Bayesian
computation*, Bayesian Analysis 1 (2006) 833.

## License

Apache-2.0, © 2026 Algenta. The kernel and wrapper are clean-room
implementations of the published nested-sampling algorithm (Skilling 2006;
Mukherjee, Parkinson & Liddle 2006; Feroz, Hobson & Bridges 2009). dynesty
is used only as a test/benchmark oracle, never as a runtime dependency.
