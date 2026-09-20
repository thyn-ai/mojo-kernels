# Contributing to mojo-kernels

Thank you for your interest in contributing. This repository is the kernel
factory behind the Algenta team's drop-in accelerator packages: clean-room
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
| `typescript/fuse-mojo/tests/` | The same for fuse-mojo (`node --test`), driven by `typescript/fuse-mojo/scripts/test_all.sh`; `parity.property.test.js` adds fast-check properties over generated corpora and options |
| `fuzz/` | Differential fuzz harnesses (atheris, with a fuzzer-free regression mode) and their seed corpora; see [`fuzz/README.md`](./fuzz/README.md) |
| `benchmarks/` | Reproducible benchmarks (median of 5 runs, correctness gate before every timing pass) |
| `scripts/` | `test_all.sh` / `test_all_cclib.sh`, the two-pass (native, then forced-fallback) suite runners |
| `pixi.toml`, `pixi.lock` | The one reproducible toolchain: the Mojo compiler (from Modular's `max` channel), Python, numpy, pytest and the test oracles |
| `.github/` | CI (one workflow per kernel), CodeQL, Scorecard, Dependabot, the security gate, `release-please.yml` (opens the release pull request; tags and publishes the Release when it merges) and `release.yml` (builds, signs and attests the assets onto it), `CODEOWNERS` |

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
| `fuzz.yml` | coverage-guided atheris runs of the bm25 and cclib differential harnesses (60 s each on a PR, 20 min nightly) plus the seed-corpus replay; fast-check parity properties for fuse-mojo on both backends (2,000 scenarios per property on a PR, 50,000 nightly) | `pixi run -e fuzz fuzz-bm25 -- -max_total_time=60` (Linux x86_64), `pixi run fuzz-regression`, `FC_NUM_RUNS=2000 pixi run fuzz-fuse` |

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

### Fuzzing

Every kernel also has a differential fuzz harness ([`fuzz/`](./fuzz/README.md)):
`fuzz/fuzz_bm25.py` and `fuzz/fuzz_cclib.py` decode arbitrary bytes into a
corpus/query or basis/grid scenario and compare the native kernel, the
vendored fallback and the reference package at the documented tolerance,
including the validation paths (NaN and infinite parameters, empty
documents, malformed basis sets must raise the documented exceptions and
nothing else). `typescript/fuse-mojo/tests/parity.property.test.js` does the
same for fuse-mojo with [fast-check](https://fast-check.dev) properties.

```bash
pixi run fuzz-regression                           # replay fuzz/corpus/ (any platform; also in `pixi run test*`)
pixi run -e fuzz fuzz-bm25 -- -max_total_time=60   # coverage-guided (Linux x86_64: atheris has no macOS wheel)
FC_NUM_RUNS=5000 pixi run fuzz-fuse                # fast-check, native then forced fallback
```

A divergence the fuzzer finds is a bug, not noise: minimise it, check the
input in under `fuzz/corpus/<name>/` (through `fuzz/seed_corpus.py`, which
owns that directory) so it is replayed forever, and either fix it or open an
issue and register it as a `KnownIssue` in the harness
(the replay then insists the seed keeps reproducing until the fix removes
both). Never narrow a generator to avoid a finding.

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
  change that widens a tolerance needs to say why in the PR. The fuzz
  harnesses assert the same tolerances on generated input (the `bm25`
  harness adds a 1e-13 relative term for out-of-domain parameters that push
  scores past ~1e5, where 1e-8 is below one ulp; see `fuzz/README.md`); a
  change that makes the seed-corpus replay fail is a parity break until
  proven otherwise.
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

### Adding a kernel workflow

Every `.github/workflows/ci-<kernel>.yml` runs on every push and pull
request, with no path filter, and the `main` branch ruleset ("Codna Review
required (mojo-kernels)", id `23705899`) lists each of its job names — both
matrix legs, spelled exactly as the check run appears on a commit, for
example `uproot-mojo differential tests + wheel (ubuntu-latest)` — as a
required status check under the strict (branch up to date) policy, next to
`codna review`, `full / security gate`, the fuzz jobs and CodeQL. A pull
request can never merge while any required check is red or pending, and a
branch that is behind `main` has to be brought up to date and re-run first.
A job the ruleset does not name still runs, but does not gate the merge, so
a new workflow is a two-part change:

1. Add the workflow (copy a sibling; keep `on: [push, pull_request]` and the
   `<name>-mojo differential tests + wheel (${{ matrix.os }})` job name so
   the check exists on every pull request and on `main`).
2. Add both job names to the ruleset. This needs repository admin and sends
   the whole ruleset back — the PUT replaces it, so start from the GET and
   strip only the read-only fields (a denylist, so an updatable field GitHub
   adds later is kept rather than silently dropped), keep every other rule,
   and send the bypass list as an empty array even if the GET returns `null`:

   ```bash
   gh api repos/thyn-ai/mojo-kernels/rulesets/23705899 \
     | jq 'del(.id, .node_id, .source, .source_type, .current_user_can_bypass,
               .created_at, .updated_at, ._links)
           | .bypass_actors |= (. // [])
           | .rules |= map(if .type == "required_status_checks" then
               .parameters.required_status_checks += [
                 {context: "<name>-mojo differential tests + wheel (ubuntu-latest)", integration_id: 15368},
                 {context: "<name>-mojo differential tests + wheel (macos-latest)", integration_id: 15368}]
             else . end)' > ruleset.json
   gh api -X PUT repos/thyn-ai/mojo-kernels/rulesets/23705899 --input ruleset.json
   gh api repos/thyn-ai/mojo-kernels/rulesets/23705899 \
     --jq '.rules[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context'
   ```

   `integration_id` 15368 is GitHub Actions. Compare the last GET with the
   first one: apart from `updated_at` and the two new contexts, nothing may
   differ. Renaming a job or a matrix leg is the same change: the old name
   stays required until the ruleset is updated, and no pull request can
   satisfy it in the meantime.

A new package joins the version lockstep by following the layout, with no
edit to the release configuration: `release-please-config.json` globs
`python/*/pyproject.toml` (`[project] version`), `python/*/*/__init__.py`
(the line `__version__ = "X.Y.Z"  # x-release-please-version` — the
trailing marker is what release-please rewrites, so keep it),
`typescript/*/package.json`, `typescript/*/packages/*/package.json` (plus
core's `optionalDependencies` pins) and `typescript/*/package-lock.json`.
Anything else that must carry the version gets the same marker on its line
and an entry in that file. `release.yml`'s preflight lists only the
packages a release ships.

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
- The pull request title becomes the squash commit on `main`, and that is
  what release-please reads to cut the next release: `feat` bumps the minor
  version; `fix`, `perf`, `deps`, `security` and `revert` bump the patch
  version; a breaking change (`!` after the type, or a `BREAKING CHANGE:`
  footer) bumps the minor version while the project is on 0.x
  (`bump-minor-pre-major`) and the major version from 1.0.0 on. `docs`,
  `ci`, `chore`, `build`, `refactor`, `test` and `style` never cut a release
  and do not appear in `CHANGELOG.md`. Write the title as the changelog
  line you want users to read; the scope becomes its bold prefix.
- The squash commit's body is the pull request description, and
  release-please parses the whole message with a strict Conventional Commits
  grammar. A message it cannot parse is dropped silently: no changelog line,
  no part in the version bump, no warning on the pull request. The one
  construct known to trip it is a `(` on a line that also contains a
  backtick, without its `)` later on the same line — which is what a
  backtick-quoted call becomes when it is hard-wrapped mid-argument
  (`` `f(a, `` on one line, `` b)` `` on the next). Keep such a call on one
  line or close the parenthesis before wrapping. Plain prose parentheses,
  Markdown headings, tables, bullets and `#123` references are all fine.
- Keep the diff focused on one change; unrelated refactors go in their own
  pull request.
- All CI checks must pass, including on forked-repository pull requests —
  CI runs with no secrets and a read-only token, so it is safe to run
  automatically on every PR. Review follows [`CODEOWNERS`](./.github/CODEOWNERS):
  changes to `pixi.lock`, the packaging scripts or anything under `.github/`
  always get deliberate maintainer review.
- Releases are cut by merging the release pull request that release-please
  opens on `main` (`chore(release): vX.Y.Z`); nobody pushes a tag by hand.
  [RELEASING.md](./RELEASING.md) describes the pipeline, the version
  lockstep, the registry gates and how anyone verifies a published asset.

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
