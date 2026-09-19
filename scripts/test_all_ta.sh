#!/usr/bin/env bash
# Full ta-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# Requires the wrapper and the oracle on PYTHONPATH. The oracle
# (pandas-ta-classic + pandas, pinned) is installed into .oracle-ta/ — a
# plain `pip --target` directory, NOT the pixi env, because `pixi run`
# prunes pip-installed packages it does not manage:
#   pixi run python -m ensurepip --upgrade   # once per fresh env
#   pixi run python -m pip install --target .oracle-ta "pandas==3.0.6" "pandas-ta-classic==0.8.32"
#   PYTHONPATH="python/ta_mojo:.oracle-ta" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_ta.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_ta_differential.py tests/test_ta_loader.py"

echo "== ta-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== ta-mojo differential suite: forced fallback (TA_MOJO_DISABLE_NATIVE=1) =="
TA_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
