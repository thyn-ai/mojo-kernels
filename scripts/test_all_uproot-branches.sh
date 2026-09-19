#!/usr/bin/env bash
# Full uproot-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. The oracle deps (uproot/awkward)
# and the wrapper's runtime dep (cramjam) are not in the repo's pixi
# manifest, so this script bootstraps them into a throwaway target dir
# (pinned versions; override with UPROOT_MOJO_DEPS_DIR to reuse one).
#
# Run from the repo root:  bash scripts/test_all_uproot-branches.sh
set -euo pipefail

cd "$(dirname "$0")/.."

DEPS_DIR="${UPROOT_MOJO_DEPS_DIR:-/tmp/uproot-branches-deps}"
if ! PYTHONPATH="$DEPS_DIR" python -c "import uproot, awkward, cramjam" >/dev/null 2>&1; then
  echo "== bootstrapping oracle deps into $DEPS_DIR =="
  python -m pip install --quiet --target "$DEPS_DIR" \
    "uproot==5.7.6" "awkward==2.14.0" "cramjam==2.12.1"
fi

export PYTHONPATH="python/uproot-branches_mojo:$DEPS_DIR"
export PYTHONNOUSERSITE=1

SUITE="tests/test_uproot-branches_differential.py tests/test_uproot-branches_loader.py"

echo "== uproot-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== uproot-mojo differential suite: forced fallback (UPROOT_MOJO_DISABLE_NATIVE=1) =="
UPROOT_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
