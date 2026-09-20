# Releasing

A release of this repository is one `vX.Y.Z` tag that ships every
distributable at that version: the `bm25-mojo` and `cclib-mojo` platform
wheels (macOS arm64, Linux x86_64) and the `@fuse-mojo/core`,
`@fuse-mojo/darwin-arm64` and `@fuse-mojo/linux-x64` npm tarballs. Two
workflows cut it: [`release-please.yml`](./.github/workflows/release-please.yml)
opens the release pull request and, when that merges, creates the tag and
the GitHub Release; [`release.yml`](./.github/workflows/release.yml) builds,
signs and attests the assets onto that Release. Nobody pushes a tag by
hand. This document is what a maintainer needs to know, what the owner sets
up once, and how anyone verifies what it produced.

## What a release contains

For tag `vX.Y.Z` the workflow builds from the tagged commit with the exact
steps the per-kernel CI workflows use (pixi tasks, differential suites,
`delocate` / `auditwheel` / `patchelf` repair, clean-environment smoke tests)
and attaches to the [GitHub Release](https://github.com/thyn-ai/mojo-kernels/releases)
for that tag:

| Asset | What it is |
| --- | --- |
| `bm25_mojo-X.Y.Z-py3-none-{manylinux_2_35_x86_64,macosx_14_0_arm64}.whl` | bm25-mojo wheels, Mojo runtime vendored |
| `cclib_mojo-X.Y.Z-py3-none-{manylinux_2_35_x86_64,macosx_14_0_arm64}.whl` | cclib-mojo wheels, Mojo runtime vendored |
| `fuse-mojo-core-X.Y.Z.tgz` | `@fuse-mojo/core` (wrapper + vendored Fuse.js fallback) |
| `fuse-mojo-linux-x64-X.Y.Z.tgz`, `fuse-mojo-darwin-arm64-X.Y.Z.tgz` | `@fuse-mojo/*` platform packages, one kernel each |
| `SHA256SUMS` | SHA-256 of the seven files above |
| `<asset>.sigstore.json` (one per asset, `SHA256SUMS` included) | keyless [Sigstore](https://www.sigstore.dev/) signature bundle, signed by the `release.yml` run itself |
| `multiple.intoto.jsonl` | [SLSA](https://slsa.dev/) build provenance covering all eight assets, from the [SLSA generic generator](https://github.com/slsa-framework/slsa-github-generator) |

The Release is the one release-please published when the release pull
request merged, with that pull request's CHANGELOG entry as its notes.
Creating it is what pushes the tag that starts `release.yml`, which attaches
the assets to it and never creates a second one. (A tag with no release at
all, which is not a supported path, gets one with generated notes.)

Registries are gated. The wheels are published to PyPI and the tarballs to
npm **only** when the repository variables `PUBLISH_PYPI` / `PUBLISH_NPM`
hold the string `true`. Until the owner flips a variable ([one-time
setup](#one-time-setup-owner)), that job is skipped and the GitHub Release is
the only channel. Both registry paths use Trusted Publishing (OpenID
Connect): no long-lived registry token lives in this repository.

## Cutting a release

Nobody bumps a version or pushes a tag. The release is a pull request that
release-please writes and a maintainer merges.

1. **Merge changes with Conventional Commit titles.** The pull request title
   becomes the squash commit on `main` and decides the next version: `feat`
   bumps the minor version; `fix`, `perf`, `deps`, `security` and `revert`
   bump the patch version; a breaking change (`!` after the type or a
   `BREAKING CHANGE:` footer) bumps the minor version while the project is
   on 0.x (`bump-minor-pre-major`) and the major version from 1.0.0 on.
   `docs`, `ci`, `chore`, `build`, `refactor`, `test` and `style` never cut
   a release on their own and do not appear in the changelog.

2. **The release pull request.** On every push to `main`,
   [`release-please.yml`](./.github/workflows/release-please.yml) opens or
   rewrites `chore(release): vX.Y.Z` (branch `release-please--branches--main`,
   label `autorelease: pending`) from the commits merged since the previous
   tag. Its diff is exactly: the next entry at the top of
   [`CHANGELOG.md`](./CHANGELOG.md), `version.txt`,
   `.release-please-manifest.json`, and every version location listed under
   [What release-please bumps](#what-release-please-bumps), all set to the
   same `X.Y.Z`. Its body is the CHANGELOG entry. It follows `main`: another
   merge re-runs release-please, which rewrites it on top of the new head, so
   it is never behind.

   To override the computed version, merge a commit whose message carries a
   `Release-As: X.Y.Z` footer; release-please uses that version for the next
   release pull request.

3. **Merge it like any other pull request.** It is opened by the
   `algenta-sdk-sync` App (a different identity from codna), so codna reviews
   it and its approval counts; every required check runs on it (no workflow
   in this repository is path-filtered). Once codna has approved and all
   required checks are green, squash-merge it. Read the version in the title
   and the CHANGELOG entry first: merging is the release decision, and that
   version is the one that ships.

   When it merges, the `release-please` run on that push creates the
   `vX.Y.Z` tag on the merge commit and publishes the GitHub Release with the
   CHANGELOG entry as its notes (the label flips to `autorelease: tagged`).
   Creating the tag starts `release.yml`.

4. **Watch the run** on the Actions tab (`release` workflow). The jobs, in
   order:

   | Job | Runs when | What it does |
   | --- | --- | --- |
   | `preflight` | always | validates the tag (and, on a re-run, that the run was dispatched on it), checks that every package version equals the tag's |
   | `build` (ubuntu, macos) | always | kernels, differential suites, wheels, npm tarballs, smoke tests; each runner keeps the assets it can vouch for |
   | `sign` | always | `SHA256SUMS`; `cosign sign-blob` per asset with the job's OIDC identity; verifies every bundle; computes the provenance subjects |
   | `release` | not on a dry run | finds / adopts / creates the GitHub Release, uploads assets and bundles |
   | `provenance` | always | SLSA generic generator; `multiple.intoto.jsonl` to the Release (or a workflow artifact on a dry run) |
   | `publish-pypi` | `PUBLISH_PYPI == "true"` | `pypa/gh-action-pypi-publish`, Trusted Publishing, PEP 740 attestations, `skip-existing` |
   | `publish-npm` | `PUBLISH_NPM == "true"` | `npm publish --provenance`, platform packages then core, versions already on the registry skipped |

   Nothing reaches a registry before the Release and its provenance exist.

### What release-please bumps

Every published unit carries the same version and `preflight` refuses a tag
that disagrees with any of them. The release pull request writes the new
`X.Y.Z` into all of these ([`release-please-config.json`](./release-please-config.json),
`extra-files`):

| Location | How release-please finds it |
| --- | --- |
| `version.txt`, `.release-please-manifest.json` | its own anchor and the last released version |
| `pixi.toml` (`[workspace] version`) | TOML path `$.workspace.version` |
| `python/*/pyproject.toml` (`[project] version`) | TOML path `$.project.version`, globbed |
| `python/*/*/__init__.py` (`__version__ = "X.Y.Z"  # x-release-please-version`) | the line marker |
| `typescript/*/package.json`, `typescript/*/packages/*/package.json` (`version`), core's `optionalDependencies` pins | JSON paths, globbed |
| `typescript/*/package-lock.json` (root `version`, `packages[""]`, `packages/core` and its pins) | JSON paths, globbed |
| `typescript/fuse-mojo/tests/unit.test.cjs`, `typescript/natural-mojo/tests/unit.test.cjs` (`assert.equal(..., 'X.Y.Z') // x-release-please-version`) | the line marker |

A new package that follows the layout is picked up by the globs. Anything
else that must carry the version gets an `x-release-please-version` marker
on its line and an entry in `extra-files`; without one it drifts. The smoke
scripts under `typescript/*/scripts/` read the version from
`packages/core/package.json` and need nothing. `preflight` checks the
packages a release ships (bm25-mojo, cclib-mojo, fuse-mojo) together with
`pixi.toml`, `version.txt` and the manifest.

### Re-running a release

Every step is idempotent (assets are replaced, registry versions already
present are skipped), so a failed or partial run is repeated by dispatching
the workflow **on the tag itself**, naming the tag again as confirmation:

```bash
gh workflow run release.yml -R thyn-ai/mojo-kernels --ref vX.Y.Z -f tag=vX.Y.Z
```

A re-run builds the tag's commit with the tag's own copy of `release.yml`,
exactly like the original tag push, and signs with the same
`@refs/tags/vX.Y.Z` identity. No job ever checks out a ref chosen by an
input: a dispatch on a branch without `dry_run`, or a `tag` that differs
from the dispatched ref, is refused in `preflight`. The trade-off is that a
fix to the pipeline made after a tag ships with the next tag, not with a
re-run.

### Dry run

To exercise the pipeline without a tag, a Release or a registry, dispatch it
on any branch with `dry_run`:

```bash
gh workflow run release.yml -R thyn-ai/mojo-kernels --ref <branch> -f dry_run=true
```

It builds, signs and attests the branch tip and leaves two workflow
artifacts on the run: `release-assets` (the seven distributables,
`SHA256SUMS`, and one `.sigstore.json` per file) and `multiple.intoto.jsonl`.
Download them with `gh run download <run-id>` and verify them as below, with
`--certificate-identity .../release.yml@refs/heads/<branch>` and
`--source-branch <branch>`.

### What the pipeline refuses

- A tag that is not `vMAJOR.MINOR.PATCH` (an optional pre-release suffix
  is allowed and marks the Release as a pre-release); a re-run dispatched
  on a branch, or whose `tag` input differs from the ref it was dispatched
  on; a dry run dispatched on a tag.
- A tag whose version differs from any package version in the tree, or a
  tree whose packages disagree with each other.
- An asset set that is not exactly four wheels and three tarballs, all
  carrying the version.
- A signature bundle that does not verify against the run's own identity.

## One-time setup (owner)

### The release pull request: App credentials

release-please opens a pull request, and this organization's enterprise
policy does not let GitHub Actions' own token create or approve pull
requests (`gh api repos/thyn-ai/mojo-kernels/actions/permissions/workflow`
reports `can_approve_pull_request_reviews: false`, and the setting cannot be
changed per repository). A pull request opened with that token would also
start none of the required workflows. So `release-please.yml` mints a
short-lived installation token of the `algenta-sdk-sync` GitHub App
(Integration 4614082, the App that already authors thyn-ai/algenta-sdk's
sync pull requests and thyn-ai/algenta-integrations' release commits),
scoped to this repository. Until the credentials exist the workflow skips
with a notice in the run summary and nothing is tagged.

1. Grant the App's installation access to this repository (Organization
   settings → GitHub Apps → `algenta-sdk-sync` → Configure → Repository
   access → add `mojo-kernels`).
2. Add its App ID and private key as repository secrets, under the names
   thyn-ai/algenta and thyn-ai/algenta-integrations already use:

   ```bash
   gh secret set ALGENTA_SDK_SYNC_APP_ID -R thyn-ai/mojo-kernels          # paste the App ID
   gh secret set ALGENTA_SDK_SYNC_APP_PRIVATE_KEY -R thyn-ai/mojo-kernels < app-private-key.pem
   ```

3. Start the first run: `gh workflow run release-please.yml -R thyn-ai/mojo-kernels`.
   It opens `chore(release): vX.Y.Z` for everything merged since the last
   tag.

The labels release-please attaches (`autorelease: pending`,
`autorelease: tagged`) already exist in the repository.

### Registries

None of the rest is needed for the GitHub Release itself. It enables the two
registry jobs and it is done once.

### Before anything is published

The wheels and platform packages vendor Modular's Mojo runtime libraries
(that is what makes them self-contained). Confirm the redistribution terms
for those binaries with Modular before flipping either variable; the build
scripts carry the same note.

### PyPI: `bm25-mojo` and `cclib-mojo`

Register a *pending* Trusted Publisher for each project (neither exists on
PyPI yet; a pending publisher creates the project on first use):

1. Open <https://pypi.org/manage/account/publishing/>, section
   **GitHub**, and add, once per project:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `bm25-mojo`, then `cclib-mojo` |
   | Owner | `thyn-ai` |
   | Repository name | `mojo-kernels` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. Enable the job:

   ```bash
   gh variable set PUBLISH_PYPI --body true -R thyn-ai/mojo-kernels
   ```

The `pypi` GitHub environment is created by the first run that reaches the
job. Adding a required reviewer to it (Settings → Environments → `pypi`)
puts a manual approval in front of every PyPI upload without touching the
workflow.

### npm: `@fuse-mojo/core`, `@fuse-mojo/darwin-arm64`, `@fuse-mojo/linux-x64`

Scoped packages need their scope to exist: create the `fuse-mojo`
organization on npmjs.com (or confirm it is owned) before the first publish.

npm can only be told to trust a workflow for a package that already exists,
so the very first publish of each package uses a short-lived token; every
later release uses OpenID Connect and no token at all.

1. First publish (token). On npmjs.com create a **granular access token**
   with *Read and write* on packages in the `@fuse-mojo` scope, *Bypass
   2FA* enabled (the workflow cannot answer a one-time code) and the
   shortest expiry offered. Store it and enable the job:

   ```bash
   gh secret set NPM_TOKEN -R thyn-ai/mojo-kernels    # paste the token when prompted
   gh variable set PUBLISH_NPM --body true -R thyn-ai/mojo-kernels
   ```

   Cut the release (or re-run an existing tag as above). The `publish-npm`
   job tries OpenID Connect first, finds no trusted publisher yet, and falls
   back to the token.

2. Switch to Trusted Publishing. For each of the three packages on
   npmjs.com: **Settings → Trusted Publisher → GitHub Actions**, with

   | Field | Value |
   | --- | --- |
   | Organization or user | `thyn-ai` |
   | Repository | `mojo-kernels` |
   | Workflow filename | `release.yml` |
   | Environment name | `npm` |

3. Retire the token:

   ```bash
   gh secret delete NPM_TOKEN -R thyn-ai/mojo-kernels
   ```

   and revoke it on npmjs.com. From now on the job authenticates with the
   run's OIDC token only, and `npm publish --provenance` attaches a
   Sigstore attestation of the run to every published version.

## Verify a release

Every asset can be checked independently of GitHub's UI: the digest in
`SHA256SUMS`, the Sigstore bundle (who signed it, from which workflow, at
which ref) and the SLSA provenance (which source commit and which builder
produced it). With `TAG=vX.Y.Z`, using the Linux bm25-mojo wheel as the
example asset:

```bash
TAG=v0.1.0
VERSION="${TAG#v}"
ASSET="bm25_mojo-${VERSION}-py3-none-manylinux_2_35_x86_64.whl"

gh release download "$TAG" -R thyn-ai/mojo-kernels \
  -p "$ASSET" -p "$ASSET.sigstore.json" \
  -p SHA256SUMS -p SHA256SUMS.sigstore.json -p multiple.intoto.jsonl

# 1. The bytes match the published digest list.
sha256sum --check --ignore-missing SHA256SUMS      # macOS: shasum -a 256 --check --ignore-missing SHA256SUMS

# 2. The signature was made by release.yml in this repository, at this tag.
cosign verify-blob "$ASSET" \
  --bundle "$ASSET.sigstore.json" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity "https://github.com/thyn-ai/mojo-kernels/.github/workflows/release.yml@refs/tags/${TAG}"

cosign verify-blob SHA256SUMS \
  --bundle SHA256SUMS.sigstore.json \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity "https://github.com/thyn-ai/mojo-kernels/.github/workflows/release.yml@refs/tags/${TAG}"

# 3. The SLSA provenance names this asset and this repository at this tag.
slsa-verifier verify-artifact "$ASSET" \
  --provenance-path multiple.intoto.jsonl \
  --source-uri github.com/thyn-ai/mojo-kernels \
  --source-tag "$TAG"
```

`cosign` (v3 or later) is at <https://docs.sigstore.dev/cosign/system_config/installation/>
and `slsa-verifier` at <https://github.com/slsa-framework/slsa-verifier#installation>.
Add `--print-provenance` to the last command to see the full statement:
the `buildDefinition` records the workflow, the exact commit and, for a
`workflow_dispatch` run, the inputs it was dispatched with.

Two details of the identity string:

- Both release paths (a pushed tag and a re-run dispatched on the tag) run
  on `refs/tags/vX.Y.Z`, so every release asset carries the identity
  above. Only a [dry run](#dry-run) is signed from a branch; its identity
  ends in `@refs/heads/<branch>` and `slsa-verifier` takes
  `--source-branch <branch>` instead of `--source-tag`.
- `multiple.intoto.jsonl` is signed by the SLSA generator's own identity
  (`slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@refs/tags/v2.1.0`),
  not by this repository; `slsa-verifier` checks that for you.

On the registries: PyPI shows the PEP 740 attestations uploaded with each
wheel on the file's page (the **Provenance** entry links back to the
`release.yml` run), and `npm audit signatures` in a project that depends on
`@fuse-mojo/core` verifies the registry signature and the `--provenance`
attestation of every installed `@fuse-mojo/*` version.
