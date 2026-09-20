#!/usr/bin/env bash
# Full vol-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# Requires the wrapper and the oracle on PYTHONPATH. The oracle
# (volatility3, pinned) is installed into .oracle-volatility3/ — a plain
# `pip --target` directory, NOT the pixi env, because `pixi run` prunes
# pip-installed packages it does not manage:
#   pixi run python -m ensurepip --upgrade   # once per fresh env
#   pixi run python -m pip install --target .oracle-volatility3 "volatility3==2.28.2"
#   PYTHONPATH="python/vol_mojo:.oracle-volatility3" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_volatility3_hive.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Make the in-repo wrapper importable.
export PYTHONPATH="python/vol_mojo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Test oracle: the published PyPI release volatility3==2.28.2, installed
# into .oracle-volatility3/ (see the header comment). It is never a runtime
# dependency of the wrapper — only the differential suite compares against
# it. An installation elsewhere can be put first via VOL_ORACLE_SITE
# (prepended to PYTHONPATH); the version check below applies all the same.
if [ -n "${VOL_ORACLE_SITE:-}" ]; then
  export PYTHONPATH="${VOL_ORACLE_SITE}:$PYTHONPATH"
fi
python -c "import volatility3" 2>/dev/null || {
  echo "error: the volatility3 oracle package is not importable;" >&2
  echo "       install it with: pixi run python -m pip install --target .oracle-volatility3 volatility3==2.28.2" >&2
  echo "       and run with PYTHONPATH=python/vol_mojo:.oracle-volatility3" >&2
  exit 1
}
python -c "
import importlib.metadata as m
v = m.version('volatility3')
assert v == '2.28.2', v
"

SUITE="tests/test_volatility3_hive_differential.py tests/test_volatility3_hive_loader.py"

echo "== vol-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== vol-mojo differential suite: forced fallback (VOL_MOJO_DISABLE_NATIVE=1) =="
VOL_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
