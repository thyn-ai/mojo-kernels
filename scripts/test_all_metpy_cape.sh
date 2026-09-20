#!/usr/bin/env bash
# Full metpy-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. The metpy oracle package must be
# importable (e.g. `pip install metpy` into the environment).
#
# Run from the repository root:
#   PYTHONPATH=python/metpy_mojo bash scripts/test_all_metpy_cape.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_metpy_cape_differential.py tests/test_metpy_cape_loader.py"

echo "== metpy-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== metpy-mojo differential suite: forced fallback (METPY_MOJO_DISABLE_NATIVE=1) =="
METPY_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
