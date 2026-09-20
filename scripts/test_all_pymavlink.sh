#!/usr/bin/env bash
# Full pymavlink-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on. Both runs compare against
# the real pymavlink package (the oracle), pinned to pymavlink==2.4.47.
#
# The oracle is a test-only dependency — the pymavlink_mojo package itself
# never imports pymavlink. This script self-provisions it into a scratch
# directory OUTSIDE the pixi environment (pixi env operations can prune
# pip-installed packages there) using uv; override the location with
# PYMAVLINK_MOJO_ORACLE_DIR.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_pymavlink_differential.py tests/test_pymavlink_loader.py"
ORACLE_VERSION="2.4.47"
ORACLE_DIR="${PYMAVLINK_MOJO_ORACLE_DIR:-/tmp/pymavlink-mojo-oracle}"

if python -c "from importlib.metadata import version; assert version('pymavlink') == '${ORACLE_VERSION}'" 2>/dev/null; then
  : # oracle already importable from the environment
else
  if [ ! -f "${ORACLE_DIR}/pymavlink/__init__.py" ]; then
    echo "== provisioning the pymavlink ${ORACLE_VERSION} oracle into ${ORACLE_DIR} =="
    if ! command -v uv >/dev/null 2>&1; then
      echo "error: pymavlink==${ORACLE_VERSION} (the differential-test oracle) is not" >&2
      echo "importable and 'uv' is unavailable to provision it. Install it with:" >&2
      echo "  uv pip install --target ${ORACLE_DIR} \"pymavlink==${ORACLE_VERSION}\"" >&2
      echo "(or make it importable in the active python environment and re-run.)" >&2
      exit 1
    fi
    uv pip install --quiet --target "${ORACLE_DIR}" "pymavlink==${ORACLE_VERSION}"
  fi
  export PYTHONPATH="${ORACLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (PYMAVLINK_MOJO_DISABLE_NATIVE=1) =="
PYMAVLINK_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
