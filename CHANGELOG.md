# Changelog

All notable changes to the mojo-kernels packages (`bm25-mojo` and
`cclib-mojo` on PyPI, `@fuse-mojo/core` on npm) and the Mojo kernels under
`kernels/` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
Entries after 0.1.0 are written by [release-please](https://github.com/googleapis/release-please)
from the Conventional Commits merged since the previous tag, in the release
pull request that cuts the version ([RELEASING.md](RELEASING.md)).

## [0.1.5](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.4...v0.1.5) (2026-09-26)


### Fixed

* **bm25:** score BM25Plus queries in reference order when the floor decomposition can overflow ([#63](https://github.com/thyn-ai/mojo-kernels/issues/63)) ([225640a](https://github.com/thyn-ai/mojo-kernels/commit/225640aeb631d32fa297f363d4e5fdc557ecb255))

## [0.1.4](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.3...v0.1.4) (2026-09-21)


### Fixed

* **fuzz:** classify ill-conditioned raw-parameter cases as a documented known issue in the bm25 harness ([#59](https://github.com/thyn-ai/mojo-kernels/issues/59)) ([b3cd900](https://github.com/thyn-ai/mojo-kernels/commit/b3cd900599b59d9e2ec2f62f3c8646fc2288c8b8))


### Performance

* **bm25:** ABI v3 native vocab map + string-token batch path — the eviction-bound Xeon M cell ([#62](https://github.com/thyn-ai/mojo-kernels/issues/62)) ([dd86c8e](https://github.com/thyn-ai/mojo-kernels/commit/dd86c8ebf44be85ee724fd7de165c7ecf4293693))
* **bm25:** flat batch packing + cheapest FFI marshal — Xeon arena warm margin 1.0-1.4x → 1.5-2.1x ([#61](https://github.com/thyn-ai/mojo-kernels/issues/61)) ([238f666](https://github.com/thyn-ai/mojo-kernels/commit/238f666cce38a25253825def554fc78f52db2e86))

## [0.1.3](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.2...v0.1.3) (2026-09-21)


### Fixed

* **build:** compile the shipped kernels for a fixed CPU baseline (x86-64-v3, apple-m1) ([#55](https://github.com/thyn-ai/mojo-kernels/issues/55)) ([ea13998](https://github.com/thyn-ai/mojo-kernels/commit/ea13998e7cc1a7992c06f7342cf22dd7e49a82a6))

## [0.1.2](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.1...v0.1.2) (2026-09-20)


### Fixed

* **fuzz:** classify the overflowing Okapi idf floor as the known degenerate-NaN divergence ([#44](https://github.com/thyn-ai/mojo-kernels/issues/44)) ([c870f60](https://github.com/thyn-ai/mojo-kernels/commit/c870f602fed2c014ddd86077862c8896b9e169dc))

## [0.1.1](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.1.1) (2026-09-20)


### Fixed

* **bm25:** evaluate a BM25Plus term in reference order when its floor overflows ([#28](https://github.com/thyn-ai/mojo-kernels/issues/28)) ([1381151](https://github.com/thyn-ai/mojo-kernels/commit/1381151dc2c72a5d33ef8ceb300248c7fd826391))
* **elephant:** duplicate LLVM OpenMP runtime abort on macOS CI + de-flaked occupancy z-tests ([#24](https://github.com/thyn-ai/mojo-kernels/issues/24)) ([6c6b6c5](https://github.com/thyn-ai/mojo-kernels/commit/6c6b6c58d4561590158071233d5a5427028c2a3e))

## [0.1.0] - 2026-09-20

### Added

- `bm25-mojo` (Python) — drop-in replacement for `rank_bm25` (`BM25Okapi`,
  `BM25L`, `BM25Plus`) backed by the `kernels/bm25` Mojo kernel; measured
  116x–8,769x on seeded 1k–100k document corpora, max abs diff 3.6e-15 vs
  `rank_bm25` 0.2.2. Pure-Python fallback via `BM25_MOJO_DISABLE_NATIVE=1`.
- `@fuse-mojo/core` (TypeScript/Node) — drop-in replacement for Fuse.js 7.1.0
  fuzzy search backed by the `kernels/fuse` Bitap kernel; measured 8.8x–38.7x
  warm and up to 13x cold, bit-identical scores and `refIndex` order.
  Vendored Fuse.js fallback via `FUSE_MOJO_DISABLE_NATIVE=1`; unsupported
  options throw `UnsupportedOptionError` on both backends.
- `cclib-mojo` (Python) — `density_on_grid` / `wavefunction_on_grid` for
  cclib-style Gaussian basis sets backed by the `kernels/gaussgrid` kernel,
  plus `cclib_mojo.cclib_integration` mirroring `cclib.method.volume`;
  measured 72x–7,033x on benzene 6-31G*, max abs diff 1.5e-12 vs the
  PyQuante1 reference. NumPy fallback via `CCLIB_MOJO_DISABLE_NATIVE=1`.
- Kernel factory layout (`kernels/<name>/` + wrapper + differential tests +
  seeded benchmark + per-kernel CI workflow) and a pinned pixi toolchain
  (Mojo via `max 26.6.*`, Python 3.12) for reproducible builds on
  macOS arm64 and Linux x86_64.
- Self-contained per-platform wheels and npm platform packages: Mojo runtime
  libraries vendored and load paths rewritten with `delocate` (macOS) and
  `auditwheel` / `patchelf` (Linux); an ABI-version handshake before every
  native call, falling back cleanly on mismatch.
- Signed, SLSA-attested releases: every GitHub Release asset ships with a
  keyless Sigstore signature bundle (`<asset>.sigstore.json`) and SLSA build
  provenance (`multiple.intoto.jsonl`); registry publishing (PyPI, npm) uses
  Trusted Publishing behind repository-variable gates. Verification steps in
  [RELEASING.md](RELEASING.md).
- Differential fuzzing (`fuzz/`): atheris harnesses for `bm25-mojo` and
  `cclib-mojo` that compare the native kernel, the vendored fallback and the
  reference package on generated corpora, parameters, basis sets and grids
  (including NaN/infinite parameters, empty documents and malformed input,
  which must raise the documented exceptions), with a fuzzer-free regression
  mode over checked-in seed corpora that runs inside the normal test suites;
  fast-check parity properties for `@fuse-mojo/core` against Fuse.js 7.1.0;
  a `fuzz` workflow (bounded budget per pull request, larger nightly).

### Fixed

- pixi toolchain (macOS arm64): the environment's OpenBLAS is now the
  pthreads build instead of the OpenMP build, so importing numpy no longer
  loads LLVM's OpenMP runtime into the process. pip-installed test oracles
  that vendor their own copy of that runtime — elephant's `fim` mining
  extension does — aborted the interpreter on the first mining call
  (`OMP: Error #15`, two copies of libomp), which took the `elephant-mojo`
  differential suite down on macOS while Linux, where the manylinux wheel
  vendors GCC's libgomp instead, passed. The suite now runs a preflight that
  reports the runtime's own message instead of an opaque abort.

- `LICENSE` is the verbatim Apache License 2.0 text (with the appendix naming
  Algenta as the copyright holder). The previous copy paraphrased sections 4,
  5, 7 and 9, so GitHub and license scanners could not identify it as
  Apache-2.0; every package's declared `license` field was always Apache-2.0.

### Known issues

- `bm25-mojo`: for degenerate parameters (`k1 == 0`, `b == 1` with an empty
  document, `b > 1`, non-finite `k1`/`b`) the native kernel scores documents
  that do not contain a query term as 0 where `rank_bm25` and the fallback
  yield NaN ([#15](https://github.com/thyn-ai/mojo-kernels/issues/15)).
- `cclib-mojo`: extreme magnitudes are not validated. Gaussian exponents
  beyond ~1e68 or below ~1e-100 bohr^-2 and atom coordinates beyond ~1e170
  Angstrom raise `OverflowError` / `ZeroDivisionError` from basis
  normalisation instead of `BasisError`, and intermediates at the edge of
  the double range make the kernel's and the NumPy fallback's evaluation
  orders disagree (inf x 0 = NaN on one side, 0 on the other)
  ([#16](https://github.com/thyn-ai/mojo-kernels/issues/16)).

[0.1.0]: https://github.com/thyn-ai/mojo-kernels/releases/tag/v0.1.0
