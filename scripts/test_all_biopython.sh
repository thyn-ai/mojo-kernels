#!/usr/bin/env bash
# Full bio-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. Both runs compare against the
# real biopython package (the oracle), pinned to biopython==1.88.
#
# The oracle is a test-only dependency — the bio_mojo package itself never
# imports biopython. This script self-provisions it into a scratch directory
# OUTSIDE the pixi environment (pixi env operations can prune pip-installed
# packages there) using uv; override the location with BIO_MOJO_ORACLE_DIR.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_biopython_differential.py tests/test_biopython_loader.py"
ORACLE_VERSION="1.88"
ORACLE_DIR="${BIO_MOJO_ORACLE_DIR:-/tmp/bio-mojo-oracle}"

if python -c "import Bio; assert Bio.__version__ == '${ORACLE_VERSION}'" 2>/dev/null; then
  : # oracle already importable from the environment
else
  if [ ! -f "${ORACLE_DIR}/Bio/__init__.py" ]; then
    echo "== provisioning the biopython ${ORACLE_VERSION} oracle into ${ORACLE_DIR} =="
    if ! command -v uv >/dev/null 2>&1; then
      echo "error: biopython==${ORACLE_VERSION} (the differential-test oracle) is not" >&2
      echo "importable and 'uv' is unavailable to provision it. Install it with:" >&2
      echo "  uv pip install --target ${ORACLE_DIR} \"biopython==${ORACLE_VERSION}\"" >&2
      echo "(or make it importable in the active python environment and re-run.)" >&2
      exit 1
    fi
    uv pip install --quiet --target "${ORACLE_DIR}" "biopython==${ORACLE_VERSION}"
  fi
  export PYTHONPATH="${ORACLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (BIO_MOJO_DISABLE_NATIVE=1) =="
BIO_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
