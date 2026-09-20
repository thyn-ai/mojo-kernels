#!/usr/bin/env bash
# Full motmetrics-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on. Run from the repository root
# with PYTHONPATH covering the wrapper and the pinned oracle, e.g.
#   PYTHONPATH="python/motmetrics_mojo:.oracle-motmetrics" PYTHONNOUSERSITE=1 \
#     pixi run bash scripts/test_all_motmetrics.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_motmetrics_differential.py tests/test_motmetrics_loader.py"

echo "== motmetrics-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== motmetrics-mojo differential suite: forced fallback (MOTMETRICS_MOJO_DISABLE_NATIVE=1) =="
MOTMETRICS_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
