#!/usr/bin/env bash
# Full elephant-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on. Run from anywhere:
#   bash scripts/test_all_elephant_surrogates.sh
# (inside the repo pixi environment, e.g. `pixi run bash scripts/...`).
set -euo pipefail

cd "$(dirname "$0")/.."

# Make the in-repo wrapper importable, exactly like the hard-rule example.
export PYTHONPATH="python/elephant_mojo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Test oracle: the published PyPI release of elephant. It is never a
# runtime dependency of the wrapper — only the differential suite compares
# against it. The pixi env may be re-solved by other work, so self-heal:
# ensure pip exists, then ensure the pinned oracle is importable. A
# pre-seeded site can be supplied via ELEPHANT_ORACLE_SITE (appended to
# PYTHONPATH) to avoid reinstalling.
if [ -n "${ELEPHANT_ORACLE_SITE:-}" ]; then
  export PYTHONPATH="${ELEPHANT_ORACLE_SITE}:$PYTHONPATH"
fi
if ! python -c "import elephant" 2>/dev/null; then
  python -c "import pip" 2>/dev/null || python -m ensurepip
  python -m pip install --quiet "elephant==1.2.1"
fi
python -c "import elephant; assert elephant.__version__ == '1.2.1', elephant.__version__"

SUITE="tests/test_elephant_surrogates_differential.py tests/test_elephant_surrogates_loader.py"

echo "== elephant-mojo differential suite: native backend =="
pytest $SUITE -q

echo
echo "== elephant-mojo differential suite: forced fallback (ELEPHANT_MOJO_DISABLE_NATIVE=1) =="
ELEPHANT_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
