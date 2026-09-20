#!/usr/bin/env bash
# Full spatialmath-mojo differential suite: once against the native
# kernel, once with the pure-Python fallback forced on.
# Run with PYTHONPATH=python/spatialmath_mojo (plus the oracle directory)
# inside the repo pixi environment.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_spatialmath_differential.py tests/test_spatialmath_loader.py"

echo "== spatialmath-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== spatialmath-mojo differential suite: forced fallback (SPATIALMATH_MOJO_DISABLE_NATIVE=1) =="
SPATIALMATH_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
