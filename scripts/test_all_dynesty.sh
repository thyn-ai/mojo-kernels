#!/usr/bin/env bash
# Full dynesty-mojo differential suite: once against the native Mojo kernel,
# once with the pure-Python fallback forced on.
#
# The oracle (dynesty==3.1.0; the wrapper itself never depends on dynesty)
# comes from the repo pixi environment (pixi.toml [pypi-dependencies], pinned
# and lock-verified). Run from the repository root with the wrapper on
# PYTHONPATH:
#
#   PYTHONPATH=python/dynesty_mojo PYTHONNOUSERSITE=1 \
#     pixi run bash scripts/test_all_dynesty.sh
#
# Build the kernel first: `pixi run bash kernels/dynesty/build.sh`
# (without it, the native pass resolves no library and falls back, which
# the loader tests report as a failure on macOS/Linux).
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_dynesty_differential.py tests/test_dynesty_loader.py"

# The differential tests `importorskip` the oracle; refuse to run a suite
# whose parity gates would silently skip.
python -c "import dynesty" 2>/dev/null || {
  echo "error: the dynesty oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

echo "== dynesty-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== dynesty-mojo differential suite: forced fallback (DYNESTY_MOJO_DISABLE_NATIVE=1) =="
DYNESTY_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
