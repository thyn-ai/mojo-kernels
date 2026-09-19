# Contributing to mojo-kernels

Thank you for your interest in contributing. This repository is the kernel
factory behind Algenta's drop-in accelerator packages: clean-room
[Mojo](https://www.modular.com/mojo) kernels, thin wrappers that load them
(`ctypes` for Python, `koffi` for Node), and a vendored pure-language
fallback for every package, so an end user never needs a Mojo toolchain.
Three kernels ship today — `bm25-mojo` (`rank_bm25` drop-in), `cclib-mojo`
(cclib electron-density grids) and `fuse-mojo` (Fuse.js drop-in) — and the
root [README](./README.md#the-kernel-factory) describes the template a new
one follows.

## Repository layout

| Path | What it is |
|---|---|
| `kernels/<name>/src/*.mojo` | The Mojo kernel, exposing a C ABI; `kernels/<name>/build.sh` compiles it into a shared library under `kernels/<name>/build/` |
| `kernels/fuse/src/shim.c` | The one C file: a pthread shim the fuse kernel loads at runtime |
| `python/bm25_mojo/`, `python/cclib_mojo/` | The Python wrapper packages (`hatchling` builds, `hatch_build.py` vendors the kernel into the wheel, `build_wheel.sh` repairs it with `delocate` / `auditwheel`) |
| `typescript/fuse-mojo/` | An npm workspace: `packages/core` (wrapper + vendored Fuse.js fallback) and the per-platform binary packages `packages/darwin-arm64`, `packages/linux-x64` |
| `tests/` | The Python differential suites — every kernel is asserted element-wise against its reference package, once on the native backend and once with the fallback forced |
| `typescript/fuse-mojo/tests/` | The same for fuse-mojo (`node --test`), driven by `typescript/fuse-mojo/scripts/test_all.sh` |
| `benchmarks/` | Reproducible benchmarks (median of 5 runs, correctness gate before every timing pass) |
| `scripts/` | `test_all.sh` / `test_all_cclib.sh`, the two-pass (native, then forced-fallback) suite runners |
| `pixi.toml`, `pixi.lock` | The one reproducible toolchain: the Mojo compiler (from Modular's `max` channel), Python, numpy, pytest and the test oracles |
| `.github/` | CI (one workflow per kernel), CodeQL, Scorecard, Dependabot, the security gate, `CODEOWNERS` |

## Development setup

Prerequisites: [pixi](https://pixi.sh) (everything else — the Mojo compiler
included — comes from `pixi.lock`), and for fuse-mojo, Node.js ≥ 18 (CI runs
Node 20) with npm from your host. Mojo has a toolchain for macOS Apple
Silicon and Linux x86_64 only; on any other platform the wrappers run their
fallback and the differential suites cannot be run natively.

```bash
git clone https://github.com/thyn-ai/mojo-kernels
cd mojo-kernels
pixi install          # resolves pixi.lock into .pixi/ (first run downloads MAX; be patient)
```

### Running the same commands CI runs

The three CI workflows are thin wrappers around pixi tasks, so green
locally means green in CI. Each task builds its kernel first (`depends-on`):

| Workflow | What it runs | Locally |
|---|---|---|
| `ci.yml` | bm25 differential suite, both backends; wheel build + repair; wheel smoke in a clean venv | `pixi run test` then `pixi run wheel-bm25` |
| `ci-cclib.yml` | gaussgrid / cclib-mojo differential suite, both backends; wheel build + repair; `quickstart.py` smoke, native vs. fallback checksum | `pixi run test-cclib` then `pixi run wheel-cclib` |
| `ci-fuse.yml` | fuse-mojo differential + unit tests, both backends; npm platform package pack + repair; end-user smoke from the packed tarballs | `pixi run test-fuse` then `pixi run pack-fuse` and `pixi run smoke-fuse` |

Single passes, when iterating:

```bash
pixi run build-kernel-bm25           # or build-kernel-gaussgrid / build-kernel-fuse
pixi run test-native                 # bm25, native backend only
pixi run test-fallback               # bm25, BM25_MOJO_DISABLE_NATIVE=1
pixi run test-cclib-native           # and test-cclib-fallback (CCLIB_MOJO_DISABLE_NATIVE=1)
cd typescript/fuse-mojo && npm test  # and npm run test:fallback (FUSE_MOJO_DISABLE_NATIVE=1)
```

Benchmarks: `pixi run bench`, `pixi run bench-cclib`, `pixi run bench-fuse`.
The README tables are regenerated from these; please do not edit numbers by
hand.

### Pre-commit and the security gate

The repository's [pre-commit](./.pre-commit-config.yaml) configuration is
managed by [thyn-ai/security-toolchain](https://github.com/thyn-ai/security-toolchain):
secret scanning (gitleaks), Opengrep, actionlint, dependency vulnerability
(OSV) and IaC checks run at commit and push time, and the same policy re-runs
on a clean checkout in CI (`security.yml`). Mojo sources are excluded from
Opengrep via [`.semgrepignore`](./.semgrepignore) — it cannot parse them; the
differential suites are the kernels' safety net.

```bash
uv tool install pre-commit   # or: pipx install pre-commit
pre-commit install            # installs both the pre-commit and pre-push hooks
```

## What a kernel change needs

- **Parity, asserted.** Every behavior change to a kernel or wrapper ships
  with differential coverage: the suite compares the native backend and the
  forced fallback against the reference package element-wise, at the
  tolerance documented for that kernel (`bm25` 1e-8 absolute, `gaussgrid`
  1e-10 relative, `fuse` scores within 1e-9 with identical match spans). A
  change that widens a tolerance needs to say why in the PR.
- **Both backends, both platforms.** CI runs every suite on `ubuntu-latest`
  and `macos-latest`, native then fallback. Windows has no Mojo toolchain;
  the wrappers' fallback is the Windows path and the fuse smoke simulates it.
- **The C ABI is versioned.** Kernels export `<name>_index_create` /
  `<name>_score` / `<name>_index_destroy`-style entry points behind an ABI
  version; changing a signature means bumping it and teaching the wrapper's
  loader to refuse a mismatched library.
- **Clean-room only.** Kernels implement the published algorithm from its
  description, not from the reference package's source; the reference
  packages are test and benchmark oracles, pinned in `pixi.toml`, never
  runtime dependencies. Fuse.js is the one vendored dependency (the fuse-mojo
  fallback) and carries its own `NOTICE`.
- **No credentials, no network.** Nothing in this repository talks to a
  network at test time; hard-coded tokens or endpoints of any kind are out.

## Commit messages and pull requests

We follow [Conventional Commits](https://www.conventionalcommits.org/), with
the scope naming the kernel or package you changed:

```
feat(bm25): SIMD path for get_batch_scores
fix(fuse): honor FUSE_MOJO_THREADS=1 on the pthread shim
docs(cclib): document the 1e-10 tolerance
ci(fuse): pin patchelf
```

- Branch from `main` as `feat/short-description`, `fix/short-description`
  or `docs/short-description`.
- Keep the diff focused on one change; unrelated refactors go in their own
  pull request.
- All CI checks must pass, including on forked-repository pull requests —
  CI runs with no secrets and a read-only token, so it is safe to run
  automatically on every PR. Review follows [`CODEOWNERS`](./.github/CODEOWNERS):
  changes to `pixi.lock`, the packaging scripts or anything under `.github/`
  always get deliberate maintainer review.
- Releases are cut by maintainers from a `vX.Y.Z` tag;
  [RELEASING.md](./RELEASING.md) describes the pipeline, the version-bump
  checklist, the registry gates and how anyone verifies a published asset.

## Reporting issues and getting help

- **Security vulnerabilities** → see [SECURITY.md](./SECURITY.md) (do NOT
  open a public issue)
- **Bugs and parity breaks** → [GitHub Issues](https://github.com/thyn-ai/mojo-kernels/issues)
  with the failing differential case (corpus / query / grid) if you have one
- **Kernel requests** → GitHub Issues; the
  [factory template](./README.md#the-kernel-factory) says what a candidate
  library needs to look like

## Licensing

This repository is licensed under [Apache-2.0](./LICENSE). Contributions
are inbound = outbound: by submitting a pull request, you license your
contribution under the project's existing license, consistent with
section D.6 of the [GitHub Terms of
Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license).
There is no Contributor License Agreement (CLA) to sign and no DCO sign-off
is required. Please only contribute work you have the right to submit under
these terms — in particular, kernels must be clean-room implementations, not
ports of the reference package's source.

## Code of conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md).
Reports go to `conduct@algenta.ai`.
