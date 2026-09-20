#!/usr/bin/env bash
# Full ase-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The ASE oracle (ase==3.26.0, test-only dependency) comes from the repo pixi
# environment (pixi.toml [pypi-dependencies], pinned and lock-verified).
# Run from the repository root inside the pixi environment:
#   pixi run bash scripts/test_all_ase_neighborlist.sh
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import ase.neighborlist" 2>/dev/null || {
  echo "error: the ASE oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

export PYTHONPATH="python/ase_mojo${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

SUITE="tests/test_ase_neighborlist_differential.py tests/test_ase_neighborlist_loader.py"

echo "== ase-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== ase-mojo differential suite: forced fallback (ASE_MOJO_DISABLE_NATIVE=1) =="
ASE_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
