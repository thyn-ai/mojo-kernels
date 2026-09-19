#!/usr/bin/env bash
# Full obspy-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The oracle (obspy) is not a pixi dependency (pixi.toml is shared); install
# it into .oracle-python once:
#   uv pip install --python .pixi/envs/default/bin/python --target .oracle-python "obspy==1.5.1"
# then run, from the repository root:
#   ~/.pixi/bin/pixi run bash scripts/test_all_obspy_konno.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .oracle-python/obspy ]; then
  echo "error: oracle package missing. Install it first:" >&2
  echo "  uv pip install --python .pixi/envs/default/bin/python --target .oracle-python \"obspy==1.5.1\"" >&2
  exit 1
fi

bash kernels/obspy-konno/build.sh

export PYTHONPATH=".oracle-python:python/obspy_mojo"
export PYTHONNOUSERSITE=1
SUITE="tests/test_obspy_konno.py tests/test_obspy_konno_loader.py"

echo "== obspy-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== obspy-mojo differential suite: forced fallback (OBSPY_MOJO_DISABLE_NATIVE=1) =="
OBSPY_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
