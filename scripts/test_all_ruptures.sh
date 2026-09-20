#!/usr/bin/env bash
# Full ruptures-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. The oracle is the published PyPI
# package ruptures==1.1.10, provided by the repo pixi environment (pixi.toml
# [pypi-dependencies], pinned and lock-verified); nothing is downloaded or
# installed here.
#
# Run from the repository root with the wrapper on PYTHONPATH:
#   PYTHONPATH=python/ruptures_mojo PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_ruptures.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_ruptures_differential.py tests/test_ruptures_loader.py"

python -c "import ruptures" 2>/dev/null || {
  echo "error: the ruptures oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (RUPTURES_MOJO_DISABLE_NATIVE=1) =="
RUPTURES_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
