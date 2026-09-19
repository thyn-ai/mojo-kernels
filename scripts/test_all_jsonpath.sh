#!/usr/bin/env bash
# Full jsonpath-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The oracle (jsonpath_ng==1.7.0) is not in pixi.toml (this leg may not edit
# that file); it is bootstrapped into a throwaway target dir if missing.
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_DIR="${JSONPATH_ORACLE_DIR:-/tmp/jporacle-lib}"
if ! PYTHONPATH="${ORACLE_DIR}" python -c "import jsonpath_ng" >/dev/null 2>&1; then
  echo "bootstrapping oracle jsonpath_ng==1.7.0 into ${ORACLE_DIR}"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$(command -v python)" --target "${ORACLE_DIR}" "jsonpath_ng==1.7.0"
  else
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
    python -m pip install --quiet --target "${ORACLE_DIR}" "jsonpath_ng==1.7.0"
  fi
fi

export PYTHONPATH="python/jsonpath_mojo:${ORACLE_DIR}"
export PYTHONNOUSERSITE=1

SUITE="tests/test_jsonpath_differential.py tests/test_jsonpath_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (JSONPATH_MOJO_DISABLE_NATIVE=1) =="
JSONPATH_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
