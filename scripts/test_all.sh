#!/usr/bin/env bash
# Full bm25-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. Run via `pixi run test`.
# Scoped to the bm25 test files; the cclib-mojo suite lives in
# scripts/test_all_cclib.sh (`pixi run test-cclib`).
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_differential.py tests/test_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (BM25_MOJO_DISABLE_NATIVE=1) =="
BM25_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
