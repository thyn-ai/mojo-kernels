# Changelog

All notable changes to the mojo-kernels packages (`bm25-mojo` and
`cclib-mojo` on PyPI, `@fuse-mojo/core` on npm) and the Mojo kernels under
`kernels/` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/thyn-ai/mojo-kernels/commits/main
