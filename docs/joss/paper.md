---
title: "cclib-mojo: Mojo-accelerated electron-density and wavefunction-on-grid evaluation for cclib"
tags:
  - Python
  - Mojo
  - computational chemistry
  - quantum chemistry
  - electron density
  - Gaussian basis functions
  - cube files
authors:
  # JOSS requires named individuals (not organizations) with an ORCID.
  # Set the author of record's name and ORCID before submission.
  - name: "AUTHOR OF RECORD TBD"
    orcid: "0000-0000-0000-0000"
    affiliation: "1"
affiliations:
  - name: Algenta
    index: 1
date: 19 September 2026
bibliography: paper.bib
---

# Summary

Computational-chemistry programs describe the electrons in a molecule as
orbitals and densities: smooth mathematical fields defined over
three-dimensional space. To visualize or analyze these fields, chemists
evaluate them on regular three-dimensional grids and store the result as
"cube files". The open-source library cclib [@oboyle2008cclib] is the
standard, parser-agnostic tool for turning the output of many different
quantum-chemistry programs into such grids, but its grid-evaluation path
runs an interpreted inner loop and is slow — a single production-size
orbital grid can take minutes. `cclib-mojo` is a drop-in accelerator for
that path: a compiled kernel written in the Mojo programming language
[@mojo] behind a thin Python wrapper, producing results identical to the
reference implementation and falling back to a pure-NumPy path on
platforms without a prebuilt binary. On a reference benzene workload it
evaluates an orbital grid 72× faster than the fastest backend cclib
ships today, while agreeing with that backend to better than $10^{-12}$.

# Statement of need

cclib's volume module evaluates molecular-orbital amplitudes
(`Volume.wavefunction()`) and electron densities
(`Volume.electrondensity()`) by summing contracted Gaussian basis
functions at every point of a 3-D grid, dispatching to optional PyQuante
backends [@pyquante; @pyquante2]. With the PyQuante 1 backend, each grid
point triggers a chain of pure-Python calls per basis function and per
primitive — hundreds of millions of interpreter steps for a production
grid; we measure 345 s for a single 100^3^ orbital grid of benzene.
PyQuante 1 additionally cannot execute on Python 3 at all, and the
NumPy-vectorized pyquante2 backend that can still materializes large
per-(basis-function, grid) temporaries. Grid evaluation is therefore the
bottleneck in cclib-based visualization and analysis pipelines.

The software serves computational and theoretical chemists who use cclib
for orbital and density visualization, cube-file generation, and
grid-based post-analysis of density-functional-theory (DFT) and
wavefunction calculations; educators preparing visual material; and
developers embedding cclib in higher-throughput pipelines. `cclib-mojo`
removes the bottleneck without changing any cclib convention — unit
conversions, shell ordering, and grid layout are mirrored exactly, so
generated grids and cube files are drop-in equivalent — and it adds no
new hard dependency (NumPy only) and no Mojo toolchain at install time.

# State of the field

Within cclib, volume evaluation uses whichever optional PyQuante backend
is importable; pyquante2's vectorized `cgbf.mesh` is the fastest backend
cclib ships today and is the honest performance baseline. Outside cclib,
orbital and density grids are produced by program-native utilities
(Gaussian's `cubegen`, ORCA's `orca_plot`), by standalone analyzers such
as Multiwfn, or by PySCF's grid tools. None of these serves cclib's core
use case — parser-agnostic post-processing of any supported package's
output through one interface — so the build-vs-contribute decision was
to contribute an additional optional backend to the existing, widely
cited community tool, following the extension pattern cclib already
uses, rather than to start a new ecosystem. Upstreaming a Mojo backend
to cclib has been proposed to its maintainers
([cclib/cclib#1909](https://github.com/cclib/cclib/issues/1909)); the
independent package remains useful either way.

# Software design

Four decisions define the package:

- **Batch-shaped C ABI.** One FFI call evaluates a whole 3-D grid for
  one orbital, or accumulates the density over all occupied orbitals, so
  foreign-function overhead is paid per call rather than per grid point.
- **Separable, deterministic kernel.** The kernel precomputes the
  exactly separable axis factors
  $\exp(-a\,dx^2)\,\exp(-a\,dy^2)\,\exp(-a\,dz^2)$ once per call, then
  runs one SIMD fused multiply-add per (grid point, primitive) in
  compiled IEEE-754 float64, with the grid row under update kept
  cache-resident. It is single-threaded by design so results are
  reproducible run to run; multithreading is future work and would
  multiply the speedup.
- **Reference-matching arithmetic.** Normalization constants are
  computed in Python using the reference's exact formulas and operation
  order (Taketa–Huzinaga–O-ohata, eqs. 2.2 and 2.12 [@taketa1966]); the
  kernel evaluates the textbook contracted Cartesian Gaussian and skips
  exactly-zero orbital coefficients — the same inclusion rule cclib
  uses. Both backends share this flattening code, so they cannot
  disagree about constants.
- **Fail-safe dual backend.** The wrapper verifies an ABI version
  handshake before evaluating; a mismatch or a missing shared library
  falls back cleanly to a vendored clean-room NumPy reference
  (`CCLIB_MOJO_DISABLE_NATIVE=1` forces it, and the test suite runs
  both paths).

Per-platform wheels (macOS arm64, Linux x86_64) are self-contained — the
Mojo runtime is vendored into the wheel with `delocate` / `auditwheel
repair` — so end users need no compiler or Mojo toolchain; on other
platforms (e.g. Windows) the same API runs on the NumPy fallback,
silently and correctly. The kernel and wrapper are clean-room
implementations: cclib and PyQuante are test and benchmark references
only, never runtime dependencies.

# Performance and correctness

\autoref{benchmarks} reports measured timings. Workload: benzene,
6-31G\* (12 atoms, 102 contracted Cartesian basis functions, 192
primitives; basis data from PyQuante 1.6.5's published basis library),
seeded orbital coefficients, median of 5 runs, single-threaded.
Environment: Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14, NumPy
2.5.3 [@numpy2020], Mojo 1.1.0.

: Measured performance of `cclib-mojo` against both reference baselines on the benzene 6-31G\* workload (median of 5 runs, single-threaded). \label{benchmarks} []{label="benchmarks"}

| Workload (baseline) | Baseline (s) | `cclib-mojo` (s) | Speedup |
|:---|---:|---:|---:|
| Wavefunction, 1 MO, 50^3^ grid (cclib `Volume.wavefunction()`, pyquante2 backend) | 0.689 | 0.0095 | 72× |
| Wavefunction, 1 MO, 50^3^ grid (PyQuante 1 amplitude path) | 32.23 | 0.0126 | 2,567× |
| Wavefunction, 1 MO, 100^3^ grid (PyQuante 1 amplitude path) | 345.37 | 0.0491 | 7,033× |
| Electron density, 3 MOs, 50^3^ grid (PyQuante 1 amplitude path) | 63.59 | 0.0335 | 1,898× |
| Wavefunction, 1 MO, 50^3^ grid, `cclib-mojo` NumPy fallback (PyQuante 1 amplitude path) | 32.23 | 0.1691 | 191× |

Two baselines are reported honestly. The pyquante2 baseline is cclib's
real `Volume.wavefunction()`, unmodified. Because PyQuante 1 cannot
execute on Python 3, its baseline is a verbatim Python-3 transcription
of its `CGBF.amp` amplitude path, driven exactly the way cclib drives it
(same formulas, operation order, and per-point interpreter cost
profile); both baselines and the benchmark script are in the repository.
Per-call setup is comparable on both sides (2.09 ms vs 1.48 ms), so the
table reflects the evaluation loop rather than setup asymmetry.
Correctness is gated, not assumed: a maximum absolute difference of
1.5e-12 against the PyQuante reference is asserted before every timing
run, and end-to-end equivalence with cclib's own
`wavefunction()`/`electrondensity()` on the pyquante2 backend agrees to
8.4e-13. Reproduce with `pixi run bench-cclib`.

# Research use cases

The use cases are those of cclib's volume path, now one to three orders
of magnitude cheaper: (i) orbital and electron-density cube files for
visualization in tools such as VMD, PyMOL, ChimeraX, or Jmol — including
fine grids (100^3^ and above) that are impractical from stock cclib;
(ii) grid-based post-analysis of parsed wavefunctions, e.g. numerical
integration, density differencing, and electrostatic-potential
evaluation in DFT post-processing pipelines; (iii) teaching, where
interactive exploration of orbital shape versus basis set breaks down
when each evaluation takes minutes; and (iv) batch cube generation
across many logfiles for screening or dataset construction — cclib's
parser-agnostic core use case.

# Research impact statement

The impact evidence is concrete and reproducible: a measured benchmark
(\autoref{benchmarks}) with a correctness gate asserted before every
timing run; a differential test suite of 19 tests executed on both
backends (38 test runs) against the PyQuante reference within 1e-10
relative — covering s/p/d/f shells (STO-3G H~2~O, 6-31G\* d on O,
cc-pVTZ f on C), multiple atoms, orbital amplitude versus summed
density, anisotropic and origin-offset grids, exact-zero coefficient
handling, grid-point ordering, and validation errors, plus end-to-end
equivalence tests against cclib itself; and continuous integration that
builds the kernel, runs both-backend differential tests, repairs and
checks wheel self-containment, and smoke-tests a hermetic install. An
optional-backend proposal is open with the cclib maintainers, a first
step toward upstream integration.

# Availability

Source code: <https://github.com/thyn-ai/mojo-kernels>
(`python/cclib_mojo`, kernel `kernels/gaussgrid`), Apache-2.0, © 2026
Algenta. Install with `pip install cclib-mojo`: per-platform
self-contained wheels for macOS arm64 and Linux x86_64, with the NumPy
fallback everywhere else; NumPy is the only hard dependency.
Documentation and a runnable quickstart ship in the package README and
`quickstart.py`; tests live in `tests/test_gaussgrid_*.py` and the
benchmark in `benchmarks/bench_gaussgrid.py`. A Zenodo archive DOI for
the release tag will be minted at submission.

# AI usage disclosure

The kernel, wrapper, tests, documentation, and this paper were produced
with generative-AI assistance within Algenta's agent-based kernel
program, under human direction and review. Every numerical claim is
machine-verified rather than model-generated: correctness is established
by the differential test suite against the PyQuante and cclib
references, run on both backends and gated in CI, and all performance
numbers are machine-measured and reproducible with
`pixi run bench-cclib`.

# Acknowledgements

Development was supported by Algenta; no external funding was received.
We thank the cclib and PyQuante authors for the reference
implementations used as test and benchmark oracles.

# References
