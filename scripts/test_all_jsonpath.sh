#!/usr/bin/env bash
# Full jsonpath-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The oracle (jsonpath_ng==1.7.0) comes from the repo pixi environment
# (pixi.toml [pypi-dependencies], pinned and lock-verified). Run from the
# repository root inside the pixi environment:
#   pixi run bash scripts/test_all_jsonpath.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="python/jsonpath_mojo"
export PYTHONNOUSERSITE=1

python -c "import jsonpath_ng" 2>/dev/null || {
  echo "error: the jsonpath_ng oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

SUITE="tests/test_jsonpath_differential.py tests/test_jsonpath_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (JSONPATH_MOJO_DISABLE_NATIVE=1) =="
JSONPATH_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
