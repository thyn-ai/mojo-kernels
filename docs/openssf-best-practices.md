# OpenSSF Best Practices badge: answer sheet for the `passing` level

This document is the prepared submission for registering
`thyn-ai/mojo-kernels` at [bestpractices.dev](https://www.bestpractices.dev)
(the OpenSSF Best Practices badge, formerly CII Best Practices). It walks
every criterion of the `passing` level, states whether this repository meets
it today, gives a justification that can be pasted into the form, and links
the evidence in this repository. Nothing here is aspirational: a criterion is
marked **Met** only when the evidence exists on `main`, and everything not yet
met says exactly what would meet it.

Status as of 2026-09-19. The criteria text is the upstream
[criteria definition](https://github.com/coreinfrastructure/best-practices-badge/blob/main/criteria/criteria.yml)
and its English descriptions; this sheet does not reword the requirements.

## Summary

| | Met | Not applicable | Owner attestation | Not met |
|---|---:|---:|---:|---:|
| MUST (43) | 31 | 9 | 2 | 1 |
| SHOULD (10) | 6 | 3 | 0 | 1 |
| SUGGESTED (14) | 11 | 0 | 0 | 3 |
| **Total (67)** | **48** | **12** | **2** | **5** |

The `passing` level requires every MUST criterion to be Met (or N/A where the
criterion allows it), every SHOULD criterion to be Met or Unmet with a
justification, and every SUGGESTED criterion to be answered.

What stands between this repository and `passing`:

1. **`release_notes` (MUST)** - no release has been published yet, so there
   are no release notes to point at. `CHANGELOG.md` already carries the
   human-written `[Unreleased]` summary and release-drafter maintains a
   categorized draft; the criterion becomes Met the moment the first release
   (`v0.1.0`) is published with that summary as its notes. Registration can
   proceed now (the badge shows *in progress*), and this criterion is
   flipped to Met at the first release.
2. **`know_secure_design` and `know_common_errors` (MUST)** are statements
   about the maintainer, not the repository. The owner attests to them in
   the form; this sheet cannot supply evidence for them.
3. **`build_floss_tools` (SHOULD)** is honestly Unmet: the Mojo compiler and
   MAX runtime in `pixi.lock` are `LicenseRef-Modular-Proprietary`. The
   justification below explains it; a SHOULD may be Unmet with justification.
4. Three SUGGESTED criteria are Unmet (`version_tags`, `warnings_strict`,
   `dynamic_analysis_unsafe`). Each row names the change that would meet it.

### Registration steps (owner only; needs the GitHub login)

1. Sign in at <https://www.bestpractices.dev/en/login> with GitHub.
2. Open <https://www.bestpractices.dev/en/projects/new> and enter the
   repository URL `https://github.com/thyn-ai/mojo-kernels`. The form
   pre-fills the name, description and license from GitHub's API, which is
   why `LICENSE` must be detected as `Apache-2.0`
   (`gh api repos/thyn-ai/mojo-kernels --jq .license.spdx_id`).
3. For each criterion, set the status from this sheet and paste the
   justification. Criteria marked **URL required** below must contain a URL
   in the justification text.
4. Once the badge is issued, add it next to the Scorecard badge in
   `README.md`:
   `[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/<id>/badge)](https://www.bestpractices.dev/projects/<id>)`.
   Scorecard's `CII-Best-Practices` check (currently 0) reads the badge
   registry by repository URL and scores *in progress* 2, *passing* 5,
   *silver* 7, *gold* 10.

### Project-level fields

| Field | Value |
|---|---|
| Name | mojo-kernels |
| Description | Clean-room Mojo kernels as drop-in accelerators for popular Python and TypeScript libraries, with bit-exact parity against the reference packages and pure-language fallbacks. |
| Project URL | <https://github.com/thyn-ai/mojo-kernels> |
| Repository URL | <https://github.com/thyn-ai/mojo-kernels> |
| License | Apache-2.0 |
| Implementation languages | Mojo, Python, JavaScript, C, Shell |
| CPE | none (no CVEs have been assigned to this project) |

Evidence links below point at `main`; `R` abbreviates
`https://github.com/thyn-ai/mojo-kernels`.

## Basics

| Criterion | Status | Justification and evidence |
|---|---|---|
| `description_good` (MUST) | Met | The README opens with a one-paragraph, jargon-light statement of what the software does: clean-room Mojo kernels that are drop-in accelerators for `rank_bm25`, Fuse.js and cclib, same API and results, prebuilt binaries so no Mojo toolchain is needed, pure-language fallback everywhere. `R/blob/main/README.md` |
| `interact` (MUST) | Met | The README says how to obtain each package ("Install & quickstart"), how to report bugs and request enhancements ("Contributing and security", `SUPPORT.md`) and how to contribute (`CONTRIBUTING.md`). `R/blob/main/README.md#install--quickstart`, `R/blob/main/SUPPORT.md` |
| `contribution` (MUST, URL required) | Met | `CONTRIBUTING.md` states the process: branch from `main`, Conventional Commits, one change per pull request, all CI checks must pass (including on fork PRs), review by `CODEOWNERS`. `R/blob/main/CONTRIBUTING.md#commit-messages-and-pull-requests` |
| `contribution_requirements` (SHOULD, URL required) | Met | "What a kernel change needs" lists the acceptance requirements: differential parity asserted on both backends at the documented tolerance, both platforms, versioned C ABI, clean-room only, no credentials or network at test time; commit format is Conventional Commits. `R/blob/main/CONTRIBUTING.md#what-a-kernel-change-needs` |
| `floss_license` (MUST) | Met | Apache License 2.0, verbatim text in `LICENSE`; every published package declares `license = "Apache-2.0"` (`python/*/pyproject.toml`, `typescript/fuse-mojo/packages/*/package.json`). `R/blob/main/LICENSE` |
| `floss_license_osi` (SUGGESTED) | Met | Apache-2.0 is OSI-approved. |
| `license_location` (MUST, URL required) | Met | Top-level `LICENSE` (full text) and `NOTICE` (copyright and third-party attribution); the vendored Fuse.js fallback carries its own `NOTICE`. `R/blob/main/LICENSE`, `R/blob/main/NOTICE` |
| `documentation_basics` (MUST) | Met | Install, start and use: the README's per-package quickstarts and the package READMEs (`python/bm25_mojo/README.md`, `typescript/fuse-mojo/README.md`, `python/cclib_mojo/README.md`) plus the runnable `quickstart.py` / `typescript/fuse-mojo/quickstart.mjs`. Secure use: "Fallback semantics" documents the native-library resolution order and the `*_DISABLE_NATIVE` / `*_NATIVE_LIB` environment switches, and `SECURITY.md` states what is in scope. `R/blob/main/README.md#install--quickstart`, `R/blob/main/README.md#fallback-semantics-every-package` |
| `documentation_interface` (MUST) | Met | Each package documents its external interface: bm25-mojo's `rank_bm25` API parity (`BM25Okapi`, `BM25L`, `BM25Plus`, constructor arguments, attributes, methods, `backend_info()`); fuse-mojo's supported-options matrix and `UnsupportedOptionError` contract, with TypeScript declarations in `packages/core/index.d.ts`; cclib-mojo's `density_on_grid` / `wavefunction_on_grid` signatures and the `cclib_integration` module. `R/blob/main/python/bm25_mojo/README.md`, `R/blob/main/typescript/fuse-mojo/README.md#supported-options-the-90-case-bit-exact-vs-fusejs-710`, `R/blob/main/python/cclib_mojo/README.md#quickstart` |
| `sites_https` (MUST) | Met | Project site and repository are `https://github.com/thyn-ai/mojo-kernels`; there is no separate website. |
| `discussion` (MUST) | Met | GitHub Issues, pull requests and GitHub Discussions are enabled: searchable, URL-addressable, open to new participants, no proprietary client. `R/issues`, `R/discussions`, `R/pulls` |
| `english` (SHOULD) | Met | All documentation, issue templates and code comments are in English; bug reports and reviews are accepted in English. |
| `maintained` (MUST) | Met | Actively developed (every change lands through reviewed pull requests with CI on macOS and Linux); Dependabot, CodeQL and Scorecard run on a schedule; `GOVERNANCE.md` names the maintainer and the decision process; `SECURITY.md` commits to acknowledging vulnerability reports within 3 business days. `R/pulse`, `R/blob/main/GOVERNANCE.md` |

## Change Control

| Criterion | Status | Justification and evidence |
|---|---|---|
| `repo_public` (MUST) | Met | Public git repository at `https://github.com/thyn-ai/mojo-kernels`. |
| `repo_track` (MUST) | Met | git history records author, date and content of every change; every change arrives as a pull request with its review thread. `R/commits/main` |
| `repo_interim` (MUST) | Met | `main` receives every merged pull request between releases; there are no squashed release-only drops. `R/commits/main` |
| `repo_distributed` (SUGGESTED) | Met | git. |
| `version_unique` (MUST) | Met | Each package carries a Semantic Versioning identifier (`version = "0.1.0"` in `python/*/pyproject.toml`, `"version": "0.1.0"` in `typescript/*/packages/*/package.json`); `GOVERNANCE.md` requires releases to be cut from a tagged commit and release-drafter names releases `v$RESOLVED_VERSION`. No release has shipped yet; the first will be `v0.1.0`. `R/blob/main/GOVERNANCE.md#releases` |
| `version_semver` (SUGGESTED) | Met | SemVer is the stated policy in `GOVERNANCE.md` and `CHANGELOG.md`. `R/blob/main/GOVERNANCE.md#releases` |
| `version_tags` (SUGGESTED) | **Not met** | The repository has no git tags yet (release-drafter holds an unpublished draft, currently resolved to `v0.0.1`). Would meet it: push the `v0.1.0` tag as `RELEASING.md` describes; `release.yml` then builds, signs and attests every asset and adopts the open draft as the Release, rewriting the draft's version to the tag and keeping its notes, so the tag that is pushed (not the draft's `v0.0.1`) is what has to match the packages' `0.1.0`. `R/blob/main/RELEASING.md` |
| `release_notes` (MUST, URL required) | **Not met** (no release yet) | `CHANGELOG.md` follows Keep a Changelog with a human-written `[Unreleased]` summary of each package, and release-drafter maintains a categorized draft release on every merge - but no release has been published, so there are no release notes *in a release*. Would meet it: at the first release, move `[Unreleased]` to `## [0.1.0] - <date>`, put that summary in the draft's body, then push the tag; `release.yml` publishes the draft with its notes kept (the drafter's list of pull-request titles is not a substitute for the summary). `R/blob/main/CHANGELOG.md`, `R/releases` |
| `release_notes_vulns` (MUST) | N/A | No release has been published and no vulnerability has been publicly reported against this project (0 security advisories). When releases exist, every fixed CVE will be listed in the release's `Security` section, which `CHANGELOG.md` and the drafter's categories already provide for. `R/security/advisories` |

## Reporting

| Criterion | Status | Justification and evidence |
|---|---|---|
| `report_process` (MUST, URL required) | Met | Bugs and parity breaks are reported through GitHub Issues using the bug-report form (`.github/ISSUE_TEMPLATE/bug_report.yml`); `SUPPORT.md` says what to include (package version, active backend, minimal reproduction). `R/issues/new/choose`, `R/blob/main/SUPPORT.md` |
| `report_tracker` (SHOULD) | Met | GitHub Issues, with bug / enhancement / question forms and a stale-issue policy (`.github/workflows/stale.yml`). `R/issues` |
| `report_responses` (MUST) | Met (no reports yet) | No bug report has been filed to date, so there is nothing unacknowledged; `SUPPORT.md` states the triage priorities and the first-interaction workflow acknowledges every new issue immediately with the next steps. State "no bug reports received yet" in the justification. `R/issues?q=is%3Aissue`, `R/blob/main/SUPPORT.md#response-expectations` |
| `enhancement_responses` (SHOULD) | Met (no requests yet) | No enhancement request has been filed to date. Kernel requests have a dedicated form (`feature_request.yml`) and `GOVERNANCE.md` describes how they are decided (lazy consensus, 72 hours). `R/issues?q=is%3Aissue+label%3Aenhancement` |
| `report_archive` (MUST, URL required) | Met | Issues, pull requests and discussions are public and searchable. `R/issues?q=`, `R/pulls?q=` |
| `vulnerability_report_process` (MUST, URL required) | Met | `SECURITY.md` publishes the process, scope, and response timeline (acknowledgement within 3 business days, triage within 7, coordinated disclosure). `R/blob/main/SECURITY.md` |
| `vulnerability_report_private` (MUST, URL required) | Met | Two private channels: GitHub private vulnerability reporting is enabled on the repository (Security -> Report a vulnerability), and `security@algenta.ai`. `R/security/advisories/new`, `R/blob/main/SECURITY.md#reporting-a-vulnerability` |
| `vulnerability_report_response` (MUST) | N/A | No vulnerability report has been received in the last 6 months. |

## Quality

| Criterion | Status | Justification and evidence |
|---|---|---|
| `build` (MUST) | Met | One reproducible build system: `pixi.toml` / `pixi.lock` pin the Mojo compiler, Python and the test oracles of the first three packages (the other kernels pin their oracle by exact version in `ci-<kernel>.yml` or `scripts/test_all_<kernel>.sh`); `pixi run build-kernel-{bm25,gaussgrid,fuse}` and `pixi run bash kernels/<kernel>/build.sh` compile the kernels, `pixi run wheel-{bm25,cclib}` and the per-package `build_wheel.sh` scripts build and repair the wheels, `pixi run pack-fuse` and its siblings build the npm platform packages. Each `ci*.yml` workflow runs exactly these commands. `R/blob/main/pixi.toml`, `R/blob/main/CONTRIBUTING.md#running-the-same-commands-ci-runs` |
| `build_common_tools` (SUGGESTED) | Met | pixi (conda ecosystem), `python -m build` with hatchling, npm, the system C compiler, `delocate` / `auditwheel` / `patchelf`. |
| `build_floss_tools` (SHOULD) | **Not met** | The wrappers and their fallbacks build with FLOSS tools only, and end users never need a Mojo toolchain, but the kernels themselves require Modular's Mojo compiler and MAX runtime, which `pixi.lock` records as `LicenseRef-Modular-Proprietary`. Justification for the form: "the accelerated kernels depend on the Mojo compiler, which is not yet FLOSS; the packages remain fully usable, and buildable, in their pure-Python / pure-JavaScript fallback form with FLOSS tools." Would meet it: a FLOSS Mojo compiler release from Modular, then re-pin. `R/blob/main/pixi.lock` |
| `test` (MUST) | Met | Automated suites in the repository, under the project's Apache-2.0 license: a pytest differential + loader suite for every Python package (`tests/`) and a `node --test` differential + unit suite for every npm package (`typescript/<package>/tests/`), each run twice (native kernel, then forced fallback) against the real reference package. How to run them is documented (`pixi run test`, `pixi run test-cclib`, `pixi run test-fuse`; every other kernel's workflow names its `scripts/test_all_<kernel>.sh`) and is what CI runs. `R/blob/main/CONTRIBUTING.md#running-the-same-commands-ci-runs`, `R/tree/main/tests` |
| `test_invocation` (SHOULD) | Met | `pytest` for the Python suites, `npm test` for the Node suite, `pixi run test` as the one-command entry point. `R/blob/main/pixi.toml` |
| `test_most` (SUGGESTED) | Met (unmeasured) | Every public entry point of every wrapper is exercised element-wise against the reference implementation on both backends, plus loader and ABI-handshake tests; unsupported options are asserted to raise on both backends. Coverage is not instrumented; a coverage report in CI would turn this into a measured claim. `R/tree/main/tests`, `R/tree/main/typescript/fuse-mojo/tests` |
| `test_continuous_integration` (SUGGESTED) | Met | 26 workflows (`ci.yml`, `ci-cclib.yml`, `ci-fuse.yml` and one `ci-<kernel>.yml` per remaining kernel) run on every pull request and every push to `main`, on `ubuntu-latest` and `macos-latest`; `fuzz.yml` runs on every pull request and nightly. The first three workflows and the four CodeQL jobs are required status checks in the branch ruleset. `R/actions` |
| `test_policy` (MUST) | Met | Policy in writing: "Every behavior change to a kernel or wrapper ships with differential coverage" (`CONTRIBUTING.md`), and `GOVERNANCE.md` makes parity a standing rule not subject to lazy consensus. `R/blob/main/CONTRIBUTING.md#what-a-kernel-change-needs`, `R/blob/main/GOVERNANCE.md#decision-making` |
| `tests_are_added` (MUST) | Met | Every kernel landed together with its differential suite and loader tests (`tests/test_differential.py` + `tests/test_loader.py` for bm25, `tests/test_gaussgrid_differential.py` + `tests/test_gaussgrid_loader.py` for cclib, `tests/test_<kernel>_differential.py` + `tests/test_<kernel>_loader.py` for the other Python packages, `typescript/<package>/tests/*.test.cjs` for the npm packages), and CI runs them on both backends. `R/tree/main/tests` |
| `tests_documented_added` (SUGGESTED) | Met | The pull-request template's first checklist item is "Differential tests added or updated, and passing on both backends". `R/blob/main/.github/pull_request_template.md` |
| `warnings` (MUST) | Met | Every first-party CommonJS module under `typescript/` runs in JavaScript strict mode (`'use strict'` in 30 of the 34 `.cjs`/`.js` files; the four without it are the simple-statistics modules vendored under `typescript/ckmeans-mojo/packages/core/vendor/`, converted mechanically from ESM and otherwise unchanged; `index.mjs` files are ES modules, strict by definition); the Mojo compiler's diagnostics are on for every kernel build; the Python wrappers guard inputs with explicit checks. Known gap, not required for this criterion: the one C file (`kernels/fuse/src/shim.c`) is compiled without `-Wall`, and there is no Python linter yet - see `warnings_strict`. `R/blob/main/typescript/fuse-mojo/packages/core/index.cjs`, `R/blob/main/kernels/fuse/build.sh` |
| `warnings_fixed` (MUST) | Met | CI builds are warning-free at the enabled levels; CodeQL's quality queries report 0 open alerts. `R/actions`, `R/security/code-scanning` |
| `warnings_strict` (SUGGESTED) | **Not met** | Would meet it: add `-Wall -Wextra` (and `-Werror` in CI) to the `cc` line in `kernels/fuse/build.sh`, and add a Python linter (for example ruff) to the pre-commit configuration the security-toolchain manages. |

## Security

| Criterion | Status | Justification and evidence |
|---|---|---|
| `know_secure_design` (MUST) | Owner attestation | A statement about the primary developer's knowledge of secure design (Saltzer and Schroeder's principles); the maintainer attests to it in the form. Supporting evidence of the principles in practice: least-privilege `permissions: contents: read` on every workflow, fail-safe fallback when a native library or ABI handshake does not match, `SECURITY.md` scope written around the trust boundary (library loading, FFI marshalling, packaging). |
| `know_common_errors` (MUST) | Owner attestation | A statement about knowledge of common vulnerability classes for this kind of software (memory-safety errors across the C ABI, integer overflow in index arithmetic, library-loading and path-handling bugs); the maintainer attests to it. `SECURITY.md` enumerates exactly these classes as in scope. `R/blob/main/SECURITY.md#scope` |
| `crypto_published` (MUST) | N/A | The software implements and uses no cryptography: the kernels score, search and evaluate grids; the wrappers load a shared library and marshal arrays. There are no hashes, ciphers, keys, tokens or authentication anywhere in `kernels/`, `python/` or `typescript/`. |
| `crypto_call` (SHOULD) | N/A | No cryptography (see `crypto_published`). |
| `crypto_floss` (MUST) | N/A | No cryptography. |
| `crypto_keylength` (MUST) | N/A | No cryptography. |
| `crypto_working` (MUST) | N/A | No cryptography. |
| `crypto_weaknesses` (SHOULD) | N/A | No cryptography. |
| `crypto_pfs` (SHOULD) | N/A | No key agreement protocols. |
| `crypto_password_storage` (MUST) | N/A | The software authenticates no users and stores no passwords. |
| `crypto_random` (MUST) | N/A | The software generates no keys or nonces (the benchmarks' seeded corpora are test data, not security mechanisms). |
| `delivery_mitm` (MUST) | Met | Source is delivered over HTTPS (`git clone https://github.com/thyn-ai/mojo-kernels`); CI build artifacts are downloaded from GitHub over HTTPS; the planned distribution channels (PyPI, npm) are HTTPS-only. |
| `delivery_unsigned` (MUST) | Met | No hash is ever fetched over plain HTTP: every pinned artifact (`pixi.lock`, `typescript/fuse-mojo/package-lock.json`, the SHA-pinned GitHub Actions) is retrieved over HTTPS and verified against the hash recorded in the repository; the end-user smoke installs with `npm ci` so the tarballs and koffi are hash-verified too. The release pipeline (`release.yml`) signs every Release asset with a keyless Sigstore bundle, attaches SLSA build provenance and publishes to PyPI and npm through Trusted Publishing; no release has been published yet, so Scorecard's `Signed-Releases` check reads *no releases found* until the first tag. `R/blob/main/typescript/fuse-mojo/scripts/smoke.sh`, `R/blob/main/.github/workflows/release.yml`, `R/blob/main/RELEASING.md` |
| `vulnerabilities_fixed_60_days` (MUST) | Met | No vulnerability of any severity is publicly known against this project: 0 security advisories, 0 open Dependabot alerts, 0 open CodeQL alerts (checked 2026-09-19). The one open code-scanning alert is Opengrep's `dangerous-subprocess-use-tainted-env-args` on `benchmarks/bench_jmespath.py`, a benchmark harness that no published package ships, opened 2026-09-19 and inside the 60-day window. Dependabot security updates are enabled with weekly version updates for GitHub Actions, npm and the pip manifests it lists. `R/security`, `R/blob/main/.github/dependabot.yml` |
| `vulnerabilities_critical_fixed` (SHOULD) | Met | Policy and timelines are published in `SECURITY.md` (acknowledgement in 3 business days, triage in 7, coordinated disclosure with a fix); none has been reported. `R/blob/main/SECURITY.md#what-to-expect` |
| `no_leaked_credentials` (MUST) | Met | The repository contains no credentials: CI runs with no stored secrets, every workflow defaults to a read-only token, and the release pipeline's only write scopes are `contents: write` for the Release and short-lived OIDC `id-token: write` for Sigstore and Trusted Publishing. GitHub secret scanning, push protection, validity checks and non-provider patterns are enabled on the repository; `CONTRIBUTING.md` forbids hard-coded tokens or endpoints. `R/blob/main/CONTRIBUTING.md#what-a-kernel-change-needs` |

## Analysis

| Criterion | Status | Justification and evidence |
|---|---|---|
| `static_analysis` (MUST) | Met | CodeQL (advanced setup) analyses the four languages CodeQL supports here - `python`, `javascript-typescript`, `c-cpp`, `actions` - on every pull request to `main`, every push to `main`, and weekly. The branch ruleset makes the four `CodeQL (<language>)` jobs required checks and blocks merging on any error/warning alert or any medium-or-higher security alert. The `security.yml` workflow (thyn-ai/security-toolchain) adds an Opengrep scan whose results also land in code scanning. Mojo is not a CodeQL language; the kernels are covered by the differential suites, which is stated in the workflow. `R/blob/main/.github/workflows/codeql.yml`, `R/blob/main/.github/workflows/security.yml`, `R/security/code-scanning` |
| `static_analysis_common_vulnerabilities` (SUGGESTED) | Met | CodeQL's default code-scanning suite includes its security queries (CWE-mapped) for all four analysed languages. `R/blob/main/.github/workflows/codeql.yml` |
| `static_analysis_fixed` (MUST) | Met | 0 open CodeQL alerts and one open Opengrep alert in a benchmark harness (see `vulnerabilities_fixed_60_days`); the ruleset's code-scanning rule prevents merging while a CodeQL error or warning, or a medium-or-higher security alert, exists on the pull request. `R/security/code-scanning` |
| `static_analysis_often` (SUGGESTED) | Met | On every pull request and push to `main`, plus a weekly scheduled run. `R/blob/main/.github/workflows/codeql.yml` |
| `dynamic_analysis` (SUGGESTED) | Met | `fuzz.yml` runs coverage-guided differential fuzzing on every pull request and nightly: atheris harnesses for bm25-mojo and cclib-mojo (`fuzz/fuzz_bm25.py`, `fuzz/fuzz_cclib.py`; 60 s per harness per pull request, 20 minutes nightly) that compare the native kernel, the fallback and the reference package, and fast-check parity properties for fuse-mojo against Fuse.js 7.1.0 on both backends (2,000 scenarios per property per pull request, 50,000 nightly). The seed corpora under `fuzz/corpus/` are replayed by the regular suites (`tests/test_fuzz_regression_*.py`). `R/blob/main/.github/workflows/fuzz.yml`, `R/blob/main/fuzz/README.md` |
| `dynamic_analysis_unsafe` (SUGGESTED) | **Not met** | The project contains one C file (`kernels/fuse/src/shim.c`) and Mojo kernels that use raw pointers across the C ABI, and no sanitizer run exists. Would meet it: a CI job that builds the shim with `-fsanitize=address,undefined` and runs the fuse suite under it (Mojo's `debug_assert` bounds checks in a debug build would extend it to the kernels). |
| `dynamic_analysis_enable_assertions` (SUGGESTED) | Met | The suites run with assertions enabled: pytest executes the Python wrappers' `assert` and explicit `raise` guards (43 in `python/*/`) un-optimised, the fuse suite uses `node:assert` (`assert.deepStrictEqual` on every result), and the quickstarts assert the active backend and the top hit before comparing outputs. `R/tree/main/tests`, `R/blob/main/typescript/fuse-mojo/quickstart.mjs` |
| `dynamic_analysis_fixed` (MUST) | N/A | The fuzz runs to date are green and no vulnerability has been found by dynamic analysis (the criterion allows N/A in that case). Becomes Met the first time a fuzz finding is fixed within the timeline in `SECURITY.md`. |

## Keeping this sheet accurate

Re-check the counted facts before each badge update: open alerts
(`gh api repos/thyn-ai/mojo-kernels/code-scanning/alerts?state=open`,
`.../dependabot/alerts?state=open`), advisories (`.../security-advisories`),
issue response times, and whether a release has been published
(`gh release list -R thyn-ai/mojo-kernels`). When the first release ships,
flip `release_notes` and `version_tags` to Met with the release URL.
