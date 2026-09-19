#!/usr/bin/env bash
# Full dynesty-mojo differential suite: once against the native Mojo kernel,
# once with the pure-Python fallback forced on.
#
# The interpreter on PATH must provide numpy + pytest + dynesty (the
# differential oracle; the wrapper itself never depends on dynesty), e.g.:
#
#   python3 -m venv /tmp/dynesty-venv
#   /tmp/dynesty-venv/bin/pip install "dynesty==3.1.0" numpy pytest
#   PATH=/tmp/dynesty-venv/bin:$PATH PYTHONPATH=python/dynesty_mojo \
#     bash scripts/test_all_dynesty.sh
#
# Build the kernel first: `pixi run bash kernels/dynesty/build.sh`
# (without it, the native pass resolves no library and falls back, which
# the loader tests report as a failure on macOS/Linux).
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_dynesty_differential.py tests/test_dynesty_loader.py"

echo "== dynesty-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== dynesty-mojo differential suite: forced fallback (DYNESTY_MOJO_DISABLE_NATIVE=1) =="
DYNESTY_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
