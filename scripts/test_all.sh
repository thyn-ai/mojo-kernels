#!/usr/bin/env bash
# Full differential suite: once against the native kernel, once with the
# pure-Python fallback forced on. Run via `pixi run test`.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== differential suite: native backend =="
pytest tests -q

echo
echo "== differential suite: forced fallback (BM25_MOJO_DISABLE_NATIVE=1) =="
BM25_MOJO_DISABLE_NATIVE=1 pytest tests -q
