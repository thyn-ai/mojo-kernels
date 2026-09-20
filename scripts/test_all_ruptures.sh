#!/usr/bin/env bash
# Full ruptures-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. The oracle is the published PyPI
# package ruptures==1.1.10.
#
# The oracle must be importable by the repo pixi Python. If `import ruptures`
# fails, this script installs ruptures==1.1.10 (with its own numpy/scipy, into
# a directory outside the repo so pixi re-syncs cannot touch it) with uv and
# puts it on PYTHONPATH ahead of the pixi site-packages.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_ruptures_differential.py tests/test_ruptures_loader.py"
ORACLE_DIR="${TMPDIR:-/tmp}/ruptures-oracle-1.1.10"

if ! python -c "import ruptures" 2>/dev/null; then
  if [ ! -d "$ORACLE_DIR/ruptures" ]; then
    UV="$(command -v uv || true)"
    if [ -z "$UV" ]; then
      echo "installing uv (needed once to fetch the ruptures oracle)..." >&2
      curl -fsSL https://astral.sh/uv/install.sh | sh >&2
      UV="$HOME/.local/bin/uv"
    fi
    "$UV" pip install --target "$ORACLE_DIR" "ruptures==1.1.10"
  fi
  export PYTHONPATH="$ORACLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (RUPTURES_MOJO_DISABLE_NATIVE=1) =="
RUPTURES_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
