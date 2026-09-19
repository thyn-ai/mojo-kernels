#!/usr/bin/env bash
# Full langdetect-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# The test oracle is the PyPI langdetect package (pinned below). It is a
# test/benchmark-only dependency: langdetect_mojo itself reads its shipped
# 55-language profiles at runtime, so the oracle presence doubles as the
# runtime dependency check.
#
# Usage (from the repository root):
#   ~/.pixi/bin/pixi run bash scripts/test_all_langdetect.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_PIN="langdetect==1.0.9"

# --- interpreter + oracle bootstrap ---------------------------------------
# Preferred: the repo pixi env. On shared/development machines the pixi env
# may lack pip or a previously pip-installed oracle (environment churn); in
# that case fall back to a dedicated venv so the suite stays reproducible.
PY=()
if ~/.pixi/bin/pixi run python -c "import langdetect" 2>/dev/null; then
  PY=(~/.pixi/bin/pixi run python)
elif ~/.pixi/bin/pixi run python -m pip --version >/dev/null 2>&1; then
  ~/.pixi/bin/pixi run python -m pip install --quiet "$ORACLE_PIN"
  PY=(~/.pixi/bin/pixi run python)
else
  VENV=/tmp/langdetect-mojo-venv
  if [ ! -x "$VENV/bin/python" ]; then
    ~/.pixi/bin/pixi run python -m venv "$VENV"
    "$VENV/bin/pip" install --quiet "$ORACLE_PIN" pytest numpy  # numpy: tests/conftest.py
  fi
  PY=("$VENV/bin/python")
fi

echo "== interpreter: ${PY[*]} =="
"${PY[@]}" -c "import langdetect, sys; print('oracle:', langdetect.__file__); print('python:', sys.version.split()[0])"

# --- kernel build (native run needs the shared library) -------------------
bash kernels/langdetect/build.sh

export PYTHONPATH="python/langdetect_mojo"
export PYTHONNOUSERSITE=1

SUITE="tests/test_langdetect_differential.py tests/test_langdetect_loader.py"

echo "== differential suite: native backend =="
"${PY[@]}" -m pytest $SUITE -q

echo
echo "== differential suite: forced fallback (LANGDETECT_MOJO_DISABLE_NATIVE=1) =="
LANGDETECT_MOJO_DISABLE_NATIVE=1 "${PY[@]}" -m pytest $SUITE -q
