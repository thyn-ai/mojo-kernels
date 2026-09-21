#!/usr/bin/env bash
# Full ipaddress-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on.
#
# The oracle is the CPython standard library `ipaddress` module itself, so
# nothing needs to be downloaded or pinned: parity is asserted against the
# interpreter that runs the suite (3.12 in the repo's pixi environment).
#
# Usage (from the repository root, pixi env's python on PATH or auto-detected):
#   bash scripts/test_all_ipaddress.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x ".pixi/envs/default/bin/python" ]; then
    PYTHON=".pixi/envs/default/bin/python"
  else
    PYTHON="python3"
  fi
fi

# Ensure the native kernel is built (skipped when forcing the fallback).
if [ "${IPADDRESS_MOJO_DISABLE_NATIVE:-0}" != "1" ]; then
  if [ ! -f kernels/ipaddress/build/libipaddressmojo.dylib ] && [ ! -f kernels/ipaddress/build/libipaddressmojo.so ]; then
    echo "== building the ipaddress Mojo kernel =="
    if [ -x ".pixi/envs/default/bin/mojo" ]; then
      export PATH="$PWD/.pixi/envs/default/bin:$PATH"
    fi
    bash kernels/ipaddress/build.sh
  fi
fi

SUITE="tests/test_ipaddress_differential.py tests/test_ipaddress_loader.py"
export PYTHONPATH="python/ipaddress_mojo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "== ipaddress-mojo differential suite: native backend =="
"$PYTHON" -m pytest $SUITE -q

echo
echo "== ipaddress-mojo differential suite: forced fallback (IPADDRESS_MOJO_DISABLE_NATIVE=1) =="
IPADDRESS_MOJO_DISABLE_NATIVE=1 "$PYTHON" -m pytest $SUITE -q
