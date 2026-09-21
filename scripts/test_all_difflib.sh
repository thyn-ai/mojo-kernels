#!/usr/bin/env bash
# Full difflib-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
# Run from the repository root, e.g.
#   PYTHONPATH=python/difflib_mojo bash scripts/test_all_difflib.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_difflib_differential.py tests/test_difflib_loader.py"

echo "== difflib-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== difflib-mojo differential suite: forced fallback (DIFFLIB_MOJO_DISABLE_NATIVE=1) =="
DIFFLIB_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
