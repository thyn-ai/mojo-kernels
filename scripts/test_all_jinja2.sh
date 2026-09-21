#!/usr/bin/env bash
# Full jinja2-mojo differential suite: once against the native kernel, once
# with the stock-lexer fallback forced on.
#
# The oracle is the published PyPI jinja2 package, pinned to 3.1.6 (the same
# release the parity claims were measured against). It is provisioned into
# .oracle-jinja2/ (a pip --target directory, not committed) because the repo
# pixi manifest is shared by every kernel and must not be edited per-kernel;
# jinja2 is also the wrapper's own runtime dependency, so the same pin is
# used for both roles.
#
# Run from the repository root:
#   PYTHONPATH=python/jinja2_mojo PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_jinja2.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_DIR=".oracle-jinja2"
if [ ! -d "$ORACLE_DIR/jinja2" ]; then
  echo "== provisioning oracle jinja2==3.1.6 into $ORACLE_DIR =="
  rm -rf "$ORACLE_DIR"
  python -m pip install --quiet --target "$ORACLE_DIR" "jinja2==3.1.6"
fi

export PYTHONPATH="python/jinja2_mojo:$PWD/$ORACLE_DIR${PYTHONPATH:+:$PYTHONPATH}"

SUITE="tests/test_jinja2_differential.py tests/test_jinja2_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (JINJA2_MOJO_DISABLE_NATIVE=1) =="
JINJA2_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
