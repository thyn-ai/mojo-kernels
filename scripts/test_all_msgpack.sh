#!/usr/bin/env bash
# Full msgpack-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback engine forced on.
#
# Requires the wrapper and the oracle on PYTHONPATH. The oracle (published
# PyPI msgpack, pinned) is installed into .oracle-msgpack/ — a plain
# `pip --target` directory, NOT the pixi env, because `pixi run` prunes
# pip-installed packages it does not manage:
#   pixi run python -m ensurepip --upgrade   # once per fresh env
#   pixi run python -m pip install --target .oracle-msgpack "msgpack==1.1.2"
#   PYTHONPATH="python/msgpack_mojo:.oracle-msgpack" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_msgpack.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_msgpack_differential.py tests/test_msgpack_loader.py"

echo "== msgpack-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== msgpack-mojo differential suite: forced fallback (MSGPACK_MOJO_DISABLE_NATIVE=1) =="
MSGPACK_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
