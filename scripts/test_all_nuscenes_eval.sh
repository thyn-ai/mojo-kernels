#!/usr/bin/env bash
# Full nuscenes-eval-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# Requires the wrapper and the oracle on PYTHONPATH. The oracle
# (nuscenes-devkit, pinned) is installed into .oracle-nuscenes-eval/ — a plain
# `pip --target` directory, NOT the pixi env, because `pixi run` prunes
# pip-installed packages it does not manage:
#   pixi run python -m ensurepip --upgrade   # once per fresh env
#   pixi run python -m pip install --target .oracle-nuscenes-eval "nuscenes-devkit==1.2.0"
#   PYTHONPATH="python/nuscenes_eval_mojo:.oracle-nuscenes-eval:tests" PYTHONNOUSERSITE=1 \
#     pixi run bash scripts/test_all_nuscenes_eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_nuscenes_eval_differential.py tests/test_nuscenes_eval_loader.py"

echo "== nuscenes-eval-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== nuscenes-eval-mojo differential suite: forced fallback (NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1) =="
NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
