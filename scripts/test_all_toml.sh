#!/usr/bin/env bash
# Full toml-mojo differential suite: once against the native kernel, once
# with the stdlib-tomllib fallback forced on. Scoped to the toml test files;
# the other kernels' suites live in their own scripts.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_toml_differential.py tests/test_toml_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (TOML_MOJO_DISABLE_NATIVE=1) =="
TOML_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
