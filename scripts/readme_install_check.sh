#!/usr/bin/env bash
# The README install commands, run exactly as printed, must install the
# packages from the GitHub Release assets and load the native kernels.
#
# Extracts every ```bash block that carries an `x-release-please-version`
# line from the README files below, runs each one verbatim (bash, and zsh
# where it is installed) in a fresh Python 3.12 venv and a fresh npm project,
# then asserts that bm25_mojo, cclib_mojo and @fuse-mojo/core report the
# native backend at the README's version. Finally it downloads the assets
# the blocks installed and checks them against the Release's SHA256SUMS.
#
# readme-install.yml runs this on ubuntu-latest and macos-latest; run it
# locally from the repository root:  bash scripts/readme_install_check.sh
#
# README_INSTALL_ALLOW_UNPUBLISHED=1: when the Release the README points at
# does not exist yet, report that and exit 0 instead of failing. The workflow
# sets it only on release-please's own pull request, whose assets release.yml
# publishes after the merge; the weekly run then checks them for real.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
readmes=(README.md python/cclib_mojo/README.md typescript/fuse-mojo/README.md)

work="$(mktemp -d "${TMPDIR:-/tmp}/readme-install.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "== extract the install blocks =="
python3 - "$root" "$work" "${readmes[@]}" <<'PY'
import pathlib, re, sys
root, work, *files = sys.argv[1:]
n = 0
for rel in files:
    text = (pathlib.Path(root) / rel).read_text()
    for match in re.finditer(r"^```bash\n(.*?)^```", text, re.S | re.M):
        block = match.group(1)
        if "x-release-please-version" not in block:
            continue
        n += 1
        pathlib.Path(work, f"block-{n:02d}.sh").write_text(block)
        line = text[: match.start()].count("\n") + 2
        print(f"  block {n}: {rel}:{line}")
if n == 0:
    sys.exit("no install block carries the x-release-please-version marker")
PY

# Every block pins the Release on its own `V=X.Y.Z  # x-release-please-version`
# line. A block without a well-formed line, or two blocks that disagree, is an
# error with a message, never an empty or garbage version.
version=""
for block in "$work"/block-*.sh; do
  v="$(awk '/x-release-please-version/ { if (match($0, /^V=[0-9]+\.[0-9]+\.[0-9]+[^ ]*/)) { print substr($0, 3, RLENGTH - 2); exit } }' "$block")"
  if [ -z "$v" ]; then
    echo "::error::$(basename "$block") has no 'V=X.Y.Z  # x-release-please-version' line; every install block must pin the Release version that way." >&2
    exit 1
  fi
  if [ -n "$version" ] && [ "$v" != "$version" ]; then
    echo "::error::the install blocks pin different versions ($version and $v in $(basename "$block")); they must agree." >&2
    exit 1
  fi
  version="$v"
done
url="https://github.com/thyn-ai/mojo-kernels/releases/download/v${version}"
echo "README version: $version"
if ! curl -fsSLI -o /dev/null "$url/SHA256SUMS"; then
  if [ "${README_INSTALL_ALLOW_UNPUBLISHED:-}" = "1" ]; then
    echo "Release v${version} has no assets yet (release.yml publishes them after the release pull request merges); nothing to install. Skipping."
    exit 0
  fi
  echo "::error::Release v${version} has no SHA256SUMS at $url; the README points at a Release that does not exist." >&2
  exit 1
fi

case "$(uname -sm)" in
  "Darwin arm64")
    assets=("bm25_mojo-${version}-py3-none-macosx_14_0_arm64.whl" "cclib_mojo-${version}-py3-none-macosx_14_0_arm64.whl" "fuse-mojo-darwin-arm64-${version}.tgz" "fuse-mojo-core-${version}.tgz")
    sha256="shasum -a 256" ;;
  "Linux x86_64")
    assets=("bm25_mojo-${version}-py3-none-manylinux_2_35_x86_64.whl" "cclib_mojo-${version}-py3-none-manylinux_2_35_x86_64.whl" "fuse-mojo-linux-x64-${version}.tgz" "fuse-mojo-core-${version}.tgz")
    sha256="sha256sum" ;;
  *) echo "::error::no prebuilt assets for $(uname -sm); this check runs on macOS arm64 and Linux x86_64 only." >&2; exit 1 ;;
esac

echo "== fresh Python venv and npm project =="
python3 -m venv "$work/venv"
export PATH="$work/venv/bin:$PATH"
python --version
pip --version | cut -d' ' -f1-2
mkdir "$work/npm"
cd "$work/npm"
npm init -y >/dev/null
node --version
echo "npm $(npm --version)"

shells=(bash)
if command -v zsh >/dev/null 2>&1; then shells+=(zsh); fi
for shell in "${shells[@]}"; do
  for block in "$work"/block-*.sh; do
    echo "== $shell $(basename "$block") =="
    sed 's/^/    /' "$block"
    # The blocks run as a reader would paste them, but with unset variables and
    # failing pipelines fatal too, so a block that loses its `${VAR:?}` guard
    # cannot expand to an empty string and install a malformed URL unnoticed.
    "$shell" -e -u -o pipefail "$block"
  done
done

echo "== native backends =="
python - "$version" <<'PY'
import sys
import bm25_mojo, cclib_mojo
version = sys.argv[1]
for module in (bm25_mojo, cclib_mojo):
    info = module.backend_info()
    print(f"  {module.__name__} {module.__version__}: native_available={info['native_available']} native_source={info['native_source']}")
    assert module.__version__ == version, f"{module.__name__} is {module.__version__}, README says {version}"
    assert info["native_available"], f"{module.__name__} fell back: {info}"
bm25 = bm25_mojo.BM25Okapi([["hello", "world"], ["hello", "mojo"], ["goodbye", "sun"]])
scores = bm25.get_scores(["world"])
assert scores.shape == (3,) and scores[0] > 0 and scores[1] == 0 and scores[2] == 0, scores
PY
node - "$version" <<'JS'
const version = process.argv[2]
const Fuse = require('@fuse-mojo/core')
const info = Fuse.backendInfo()
console.log(`  @fuse-mojo/core ${Fuse.version}: native_available=${info.native_available} native_source=${info.native_source}`)
if (Fuse.version !== version) throw new Error(`@fuse-mojo/core is ${Fuse.version}, README says ${version}`)
if (!info.native_available) throw new Error(`@fuse-mojo/core fell back: ${JSON.stringify(info)}`)
const hits = new Fuse([{ title: 'The Lock Artist' }, { title: 'Syrup' }], { keys: ['title'] }).search('lock')
if (hits.length !== 1 || hits[0].refIndex !== 0) throw new Error(`unexpected search result ${JSON.stringify(hits)}`)
JS

echo "== SHA256SUMS =="
mkdir "$work/sums"
cd "$work/sums"
for asset in SHA256SUMS "${assets[@]}"; do
  curl -fsSL -o "$asset" "$url/$asset"
done
$sha256 --check --ignore-missing SHA256SUMS
echo "readme install OK: v${version} on $(uname -sm), native backends loaded, ${#assets[@]} assets match SHA256SUMS"
