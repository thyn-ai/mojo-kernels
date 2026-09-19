#!/usr/bin/env bash
# Full pykalman-mojo differential suite: once against the native kernel,
# once with the pure-NumPy fallback forced on.
# Run with the wrapper on PYTHONPATH, e.g.:
#   PYTHONPATH=python/pykalman_mojo pixi run bash scripts/test_all_pykalman.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_pykalman_differential.py tests/test_pykalman_loader.py"

echo "== pykalman-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== pykalman-mojo differential suite: forced fallback (PYKALMAN_MOJO_DISABLE_NATIVE=1) =="
PYKALMAN_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
