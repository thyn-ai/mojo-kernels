#!/usr/bin/env bash
# Full uproot-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. The oracle deps (uproot/awkward)
# and the wrapper's runtime dep (cramjam) come from the repo pixi environment
# (pixi.toml [pypi-dependencies], pinned and lock-verified).
#
# Run from the repo root inside the pixi environment:
#   pixi run bash scripts/test_all_uproot-branches.sh
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import uproot, awkward, cramjam" 2>/dev/null || {
  echo "error: the uproot/awkward/cramjam oracle packages are not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

export PYTHONPATH="python/uproot-branches_mojo"
export PYTHONNOUSERSITE=1

SUITE="tests/test_uproot-branches_differential.py tests/test_uproot-branches_loader.py"

echo "== uproot-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== uproot-mojo differential suite: forced fallback (UPROOT_MOJO_DISABLE_NATIVE=1) =="
UPROOT_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
