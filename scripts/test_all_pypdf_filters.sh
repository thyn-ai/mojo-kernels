#!/usr/bin/env bash
# Full pypdf-filters-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# Run inside the repo pixi environment with the wrapper on PYTHONPATH:
#   PYTHONPATH=python/pypdf_filters_mojo pixi run bash scripts/test_all_pypdf_filters.sh
#
# The oracle is the published PyPI package pypdf==6.19.0, provided by the
# repo pixi environment (pixi.toml [pypi-dependencies], pinned and
# lock-verified).
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import pypdf" 2>/dev/null || {
  echo "error: the pypdf oracle package is not importable;" >&2
  echo "       run inside the pixi environment (pixi install)" >&2
  exit 1
}

SUITE="tests/test_pypdf_filters_differential.py tests/test_pypdf_filters_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (PDF_MOJO_DISABLE_NATIVE=1) =="
PDF_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
