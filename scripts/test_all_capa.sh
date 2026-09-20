#!/usr/bin/env bash
# Full capa-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# Requires the wrapper and the oracle on PYTHONPATH. The oracle
# (flare-capa, pinned) is installed into .oracle-capa/ — a plain
# `pip --target` directory, NOT the pixi env, because `pixi run`
# prunes pip-installed packages it does not manage:
#   pixi run python -m ensurepip --upgrade   # once per fresh env
#   pixi run python -m pip install --target .oracle-capa "flare-capa==9.2.0"
#   PYTHONPATH="python/capa_mojo:.oracle-capa" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_capa.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_capa_match_differential.py tests/test_capa_match_loader.py"

echo "== capa-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== capa-mojo differential suite: forced fallback (CAPA_MOJO_DISABLE_NATIVE=1) =="
CAPA_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
