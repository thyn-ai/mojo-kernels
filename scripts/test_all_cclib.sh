#!/usr/bin/env bash
# Full gaussgrid/cclib-mojo differential suite: once against the native
# kernel, once with the pure-Python fallback forced on.
# Run via `pixi run test-cclib`.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_gaussgrid_differential.py tests/test_gaussgrid_loader.py"

echo "== cclib-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== cclib-mojo differential suite: forced fallback (CCLIB_MOJO_DISABLE_NATIVE=1) =="
CCLIB_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
