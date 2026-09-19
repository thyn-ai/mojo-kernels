#!/usr/bin/env bash
# Full croniter-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# Requires the croniter oracle (pip croniter==6.2.4) in the test environment.
# Run from the repository root with explicit env, e.g.:
#   PYTHONPATH=python/croniter_mojo PYTHONNOUSERSITE=1 ~/.pixi/bin/pixi run bash scripts/test_all_croniter.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_croniter_differential.py tests/test_croniter_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (CRONITER_MOJO_DISABLE_NATIVE=1) =="
CRONITER_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
