#!/usr/bin/env bash
# Full oletools-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# The oracle is the published PyPI package oletools==0.60.2. pixi.toml is
# shared by every kernel leg in this repo, so the oracle is installed
# imperatively into a --target directory that pixi never touches, and put on
# PYTHONPATH:
#   uv pip install --python .pixi/envs/default/bin/python --target .oracle-oletools "oletools==0.60.2"
#
# Run from the repository root:
#   PYTHONPATH="python/oletools_mojo" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_oletools_vba.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .oracle-oletools/oletools ]; then
  echo "error: the oletools oracle is not installed into .oracle-oletools/;" >&2
  echo "  uv pip install --python .pixi/envs/default/bin/python --target .oracle-oletools \"oletools==0.60.2\"" >&2
  exit 1
fi

export PYTHONPATH=".oracle-oletools:python/oletools_mojo${PYTHONPATH:+:$PYTHONPATH}"

SUITE="tests/test_oletools_vba_differential.py tests/test_oletools_vba_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (OLETOOLS_MOJO_DISABLE_NATIVE=1) =="
OLETOOLS_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
