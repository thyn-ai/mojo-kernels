#!/usr/bin/env bash
# Full statistics-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
# Run via `PYTHONPATH=python/statistics_mojo pixi run bash scripts/test_all_statistics.sh`.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_statistics_differential.py tests/test_statistics_loader.py"

echo "== statistics-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== statistics-mojo differential suite: forced fallback (STATISTICS_MOJO_DISABLE_NATIVE=1) =="
STATISTICS_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
