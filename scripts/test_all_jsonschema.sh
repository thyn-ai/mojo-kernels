#!/usr/bin/env bash
# Full jsonschema-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# Run from the repository root inside the pixi environment:
#   ~/.pixi/bin/pixi run bash scripts/test_all_jsonschema.sh
#
# The oracle (PyPI jsonschema==4.26.0) comes from the repo pixi environment
# (pixi.toml [pypi-dependencies], pinned and lock-verified).
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="python/jsonschema_mojo"
export PYTHONNOUSERSITE=1

python -c "import jsonschema" 2>/dev/null || {
  echo "error: the jsonschema oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

SUITE="tests/test_jsonschema_differential.py tests/test_jsonschema_loader.py"

echo "== jsonschema-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== jsonschema-mojo differential suite: forced fallback (JSONSCHEMA_MOJO_DISABLE_NATIVE=1) =="
JSONSCHEMA_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
