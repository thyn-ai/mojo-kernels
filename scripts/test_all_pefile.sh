#!/usr/bin/env bash
# Full pefile-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The oracle (PyPI pefile) is not part of the repo's pixi lockfile (kernel
# packages must not touch pixi.toml), so this script bootstraps it into the
# environment: the exact pinned wheel is downloaded from PyPI and its SHA-256
# verified before installation. This also repairs the env after a `pixi
# install` has pruned pip-installed packages.
#
# Usage (from the repository root, pixi env's python on PATH or auto-detected):
#   bash scripts/test_all_pefile.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PEFILE_VERSION="2024.8.26"
PEFILE_WHEEL="pefile-${PEFILE_VERSION}-py3-none-any.whl"
PEFILE_URL="https://files.pythonhosted.org/packages/54/16/12b82f791c7f50ddec566873d5bdd245baa1491bac11d15ffb98aecc8f8b/${PEFILE_WHEEL}"
PEFILE_SHA256="76f8b485dcd3b1bb8166f1128d395fa3d87af26360c2358fb75b80019b957c6f"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x ".pixi/envs/default/bin/python" ]; then
    PYTHON=".pixi/envs/default/bin/python"
  else
    PYTHON="python3"
  fi
fi

# 1. Ensure the oracle is importable; install it hash-verified if not.
if ! "$PYTHON" -c "import pefile" 2>/dev/null; then
  echo "== installing oracle pefile==${PEFILE_VERSION} (hash-verified) =="
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -sSfL -o "$tmp/$PEFILE_WHEEL" "$PEFILE_URL"
  echo "${PEFILE_SHA256}  $tmp/$PEFILE_WHEEL" | shasum -a 256 -c -
  "$PYTHON" -m pip install --quiet "$tmp/$PEFILE_WHEEL"
  rm -rf "$tmp"
  trap - EXIT
fi
"$PYTHON" -c "import pefile; assert pefile.__version__ == '${PEFILE_VERSION}', pefile.__version__"

# 2. Ensure the native kernel is built (skipped when forcing the fallback).
if [ "${PEFILE_MOJO_DISABLE_NATIVE:-0}" != "1" ]; then
  if [ ! -f kernels/pefile/build/libpefilemojo.dylib ] && [ ! -f kernels/pefile/build/libpefilemojo.so ]; then
    echo "== building the pefile Mojo kernel =="
    if [ -x ".pixi/envs/default/bin/mojo" ]; then
      export PATH="$PWD/.pixi/envs/default/bin:$PATH"
    fi
    bash kernels/pefile/build.sh
  fi
fi

SUITE="tests/test_pefile_fixtures.py tests/test_pefile_differential.py tests/test_pefile_loader.py"
export PYTHONPATH="python/pefile_mojo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "== pefile-mojo differential suite: native backend =="
"$PYTHON" -m pytest $SUITE -q

echo
echo "== pefile-mojo differential suite: forced fallback (PEFILE_MOJO_DISABLE_NATIVE=1) =="
PEFILE_MOJO_DISABLE_NATIVE=1 "$PYTHON" -m pytest $SUITE -q
