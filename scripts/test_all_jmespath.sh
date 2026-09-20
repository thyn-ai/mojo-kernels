#!/usr/bin/env bash
# Full jmespath-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The oracle is the published PyPI package (jmespath==1.0.1). It must be
# importable: install it into the pixi env (`pip install jmespath==1.0.1`)
# or point PYTHONPATH at a target dir (`pip install --target=/tmp/jm-oracle
# jmespath==1.0.1`). The wrapper package also comes from PYTHONPATH, e.g.:
#
#   PYTHONPATH="python/jmespath_mojo:/tmp/jm-oracle" \
#     pixi run bash scripts/test_all_jmespath.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_jmespath_differential.py tests/test_jmespath_loader.py"

python -c "import jmespath" 2>/dev/null || {
  echo "error: the jmespath oracle package is not importable (see header)" >&2
  exit 1
}

echo "== jmespath-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== jmespath-mojo differential suite: forced fallback (JMESPATH_MOJO_DISABLE_NATIVE=1) =="
JMESPATH_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
