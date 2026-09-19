#!/usr/bin/env bash
# Full ase-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on.
#
# The ASE oracle (test-only dependency) is installed into a scratch
# directory so it never touches the repo toolchain env:
#   bash scripts/test_all_ase_neighborlist.sh
# (run with the repo toolchain on PATH, e.g. `pixi run bash
# scripts/test_all_ase_neighborlist.sh`).
set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_SITE="${ASE_ORACLE_SITE:-$(mktemp -d)/oracle}"
if [ ! -d "${ORACLE_SITE}/ase" ]; then
  echo "== installing ASE oracle into ${ORACLE_SITE} =="
  python -m pip install --quiet --target "${ORACLE_SITE}" \
    "ase==${ASE_ORACLE_VERSION:-3.26.0}" "scipy==1.16.3"
fi

export PYTHONPATH="python/ase_mojo:${ORACLE_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

SUITE="tests/test_ase_neighborlist_differential.py tests/test_ase_neighborlist_loader.py"

echo "== ase-mojo differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== ase-mojo differential suite: forced fallback (ASE_MOJO_DISABLE_NATIVE=1) =="
ASE_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
