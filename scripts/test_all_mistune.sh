#!/usr/bin/env bash
# Full mistune-mojo differential suite: once against the native kernel, once
# with the pure-Python engine forced on.
# Run via `pixi run bash scripts/test_all_mistune.sh`.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_mistune_differential.py tests/test_mistune_loader.py"

echo "== mistune-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== mistune-mojo differential suite: forced fallback (MISTUNE_MOJO_DISABLE_NATIVE=1) =="
MISTUNE_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
