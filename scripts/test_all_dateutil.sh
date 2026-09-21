#!/usr/bin/env bash
# Full dateutil-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The python-dateutil oracle (2.9.0.post0) comes from the repo pixi
# environment (pixi.lock, pinned and lock-verified).
# Run from the repository root with explicit env, e.g.:
#   PYTHONPATH=python/dateutil_mojo PYTHONNOUSERSITE=1 ~/.pixi/bin/pixi run bash scripts/test_all_dateutil.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_dateutil_differential.py tests/test_dateutil_loader.py"

echo "== differential suite: native backend =="
DATEUTIL_MOJO_TEST_NATIVE=1 pytest $SUITE -q

echo
echo "== differential suite: forced fallback (DATEUTIL_MOJO_DISABLE_NATIVE=1) =="
DATEUTIL_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
