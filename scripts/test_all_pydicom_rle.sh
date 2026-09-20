#!/usr/bin/env bash
# Full pydicom-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# The oracle is the published PyPI package pydicom==3.0.2 (plus numpy, which
# the repository's shared tests/conftest.py imports). Neither is in pixi.toml
# — per-kernel paths are disjoint and pixi.toml is off-limits — so provide
# them on the active interpreter, e.g.:
#
#   python3 -m venv /tmp/pydicom-oracle
#   /tmp/pydicom-oracle/bin/pip install pydicom==3.0.2 numpy pytest
#   PATH="/tmp/pydicom-oracle/bin:$PATH" PYTHONPATH=python/pydicom_mojo \
#     bash scripts/test_all_pydicom_rle.sh
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import pydicom" 2>/dev/null || {
  echo "error: the pydicom oracle package is not importable;" >&2
  echo "       install pydicom==3.0.2 on the active interpreter (see header)" >&2
  exit 1
}

SUITE="tests/test_pydicom_rle_differential.py tests/test_pydicom_rle_loader.py"

echo "== differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== differential suite: forced fallback (PYDICOM_MOJO_DISABLE_NATIVE=1) =="
PYDICOM_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
