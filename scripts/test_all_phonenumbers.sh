#!/usr/bin/env bash
# Full phonenumbers-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# The test oracle is the PyPI phonenumbers package (pinned below). It is a
# test/benchmark-only dependency: phonenumbers_mojo reads its own compiled
# metadata blob at runtime, so the oracle presence doubles as the runtime
# dependency check.
#
# Usage (from the repository root):
#   ~/.pixi/bin/pixi run bash scripts/test_all_phonenumbers.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_PIN="phonenumbers==9.0.39"

# --- interpreter + oracle bootstrap ---------------------------------------
# Preferred: the repo pixi env. On shared/development machines the pixi env
# may lack pip or a previously pip-installed oracle (environment churn); in
# that case fall back to a dedicated venv so the suite stays reproducible.
PY=()
if ~/.pixi/bin/pixi run python -c "import phonenumbers" 2>/dev/null; then
  PY=(~/.pixi/bin/pixi run python)
elif ~/.pixi/bin/pixi run python -m pip --version >/dev/null 2>&1; then
  ~/.pixi/bin/pixi run python -m pip install --quiet "$ORACLE_PIN"
  PY=(~/.pixi/bin/pixi run python)
else
  VENV=/tmp/phonenumbers-mojo-venv
  if [ ! -x "$VENV/bin/python" ]; then
    ~/.pixi/bin/pixi run python -m venv "$VENV"
    "$VENV/bin/pip" install --quiet "$ORACLE_PIN" pytest numpy  # numpy: tests/conftest.py
  fi
  PY=("$VENV/bin/python")
fi

echo "== interpreter: ${PY[*]} =="
"${PY[@]}" -c "import phonenumbers, sys; print('oracle:', phonenumbers.__file__); print('python:', sys.version.split()[0])"

# --- kernel build (native run needs the shared library) -------------------
bash kernels/phonenumbers/build.sh

export PYTHONPATH="python/phonenumbers_mojo"
export PYTHONNOUSERSITE=1

SUITE="tests/test_phonenumbers_differential.py tests/test_phonenumbers_loader.py"

echo "== differential suite: native backend =="
"${PY[@]}" -m pytest $SUITE -q

echo
echo "== differential suite: forced fallback (PHONENUMBERS_MOJO_DISABLE_NATIVE=1) =="
PHONENUMBERS_MOJO_DISABLE_NATIVE=1 "${PY[@]}" -m pytest $SUITE -q
