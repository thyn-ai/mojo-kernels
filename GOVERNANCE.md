# Governance

This document describes how decisions are made for mojo-kernels
(`thyn-ai/mojo-kernels`). It is intentionally lightweight and will evolve as
the contributor community grows.

## Roles

- **Contributors** — anyone who opens issues, participates in discussions, or
  submits pull requests.
- **Maintainers** — people with merge and release authority on this
  repository. The project currently has a single maintainer: the `thyn-ai`
  organization owner (see [CODEOWNERS](./CODEOWNERS)), who also holds final
  decision authority on all matters not explicitly delegated.

## Decision-making

Day-to-day decisions are made by **lazy consensus**:

1. Propose the change as a GitHub issue or pull request.
2. Maintainers and contributors discuss in the open.
3. If no maintainer objects within 72 hours (three business days), the
   proposal is considered accepted and may proceed.

Maintainers may fast-track obvious, low-risk changes (typo fixes, CI repairs,
dependency security bumps) without waiting out the window. Any maintainer may
pause lazy consensus by raising an objection, in which case the change waits
until the objection is resolved in discussion. Where consensus cannot be
reached, the organization owner makes the final call.

Two standing rules are not subject to lazy consensus, because they are the
guarantees every package makes to its users:

- **Parity with the reference library.** Every kernel ships with a
  differential test suite against the real reference package (rank_bm25,
  Fuse.js, cclib/PyQuante), run on both the native and the forced-fallback
  backend. A change that weakens or skips that suite is not mergeable.
- **Clean-room implementation.** Kernels implement published algorithms from
  their specifications; reference-library source is used only as a test and
  benchmark oracle, never copied into a kernel. (The vendored Fuse.js
  fallback is redistributed under its own Apache-2.0 license, unmodified in
  substance, and is not part of the kernel.)

## Adding a kernel

New kernels follow the factory template described in the
[README](./README.md#the-kernel-factory): a `kernels/<name>/` Mojo source
with the batch-shaped C ABI and `<name>mojo_abi_version` handshake, a
wrapper package under `python/` or `typescript/` with a vendored
pure-language fallback, differential tests on both backends, a seeded
benchmark with a correctness gate, and a per-kernel CI workflow. Proposals
start as an `enhancement` issue naming the reference library and the hot
loop; acceptance follows lazy consensus.

## Becoming a maintainer

External contributors can become maintainers. The path:

1. **Sustained contribution** — a track record of merged pull requests,
   thoughtful reviews, and issue triage over several months.
2. **Nomination** — an existing maintainer nominates the contributor, citing
   specific contributions, in a GitHub discussion visible to all maintainers.
3. **Lazy consensus** — if no maintainer objects within 14 days, the
   nomination carries. The organization owner confirms and grants access.

Maintainers are expected to review pull requests, triage issues, uphold the
[Code of Conduct](./CODE_OF_CONDUCT.md), follow the
[security policy](./SECURITY.md), and keep CI green on `main`. Maintainers
who become inactive for more than a year may be moved to emeritus status by
the organization owner; emeritus maintainers can regain commit access on
request.

## Releases

- Each wrapper package (`bm25-mojo` and `cclib-mojo` on PyPI, `@fuse-mojo/core`
  and its platform packages on npm) follows
  [Semantic Versioning](https://semver.org/); notable changes are recorded in
  [CHANGELOG.md](./CHANGELOG.md).
- Releases are cut from `main` by a maintainer. Per-platform binaries
  (wheels and npm platform packages) are built and repaired by CI — never
  uploaded from a developer machine — so that every published artifact is
  reproducible from a tagged commit. Release authorization currently rests
  solely with the organization owner.
- Patch releases may be cut whenever a worthwhile fix lands. Minor releases
  are cut when backward-compatible additions have accumulated (a new kernel,
  a newly supported option). Major releases require an announced deprecation
  window for any removal from a wrapper's public API; the drop-in contract
  with the reference library's API is never narrowed in a minor or patch
  release.
- Release decisions — what ships and when — are made by maintainers through
  lazy consensus as described above, with the organization owner holding
  final authority.

## Changing this document

Amendments to this file follow the same lazy-consensus process as any other
change, with one difference: the review window is 14 days, and the
organization owner must approve the merged pull request.
