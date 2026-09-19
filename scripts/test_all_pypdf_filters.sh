#!/usr/bin/env bash
# Full pypdf-filters-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# Run inside the repo pixi environment with the wrapper on PYTHONPATH:
#   PYTHONPATH=python/pypdf_filters_mojo pixi run bash scripts/test_all_pypdf_filters.sh
#
# The oracle is the published PyPI package pypdf==6.19.0. pixi.toml is shared
# by every kernel in this repo and must NOT gain per-kernel oracle entries,
# so the oracle is pip-installed on demand into a directory OUTSIDE the pixi
# environment (a pixi re-solve prunes unknown packages from site-packages).
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_VERSION="6.19.0"
ORACLE_DIR="${TMPDIR:-/tmp}/pypdf_oracle_${ORACLE_VERSION}"

if ! python -c "import pypdf" 2>/dev/null; then
  if [ ! -d "${ORACLE_DIR}/pypdf" ]; then
    echo "== installing pypdf==${ORACLE_VERSION} oracle into ${ORACLE_DIR} =="
    python -m pip install --quiet --target "${ORACLE_DIR}" "pypdf==${ORACLE_VERSION}"
  fi
  export PYTHONPATH="${ORACLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

SUITE="tests/test_pypdf_filters_differential.py tests/test_pypdf_filters_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (PDF_MOJO_DISABLE_NATIVE=1) =="
PDF_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
