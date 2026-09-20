#!/usr/bin/env bash
# Full sacrebleu-mojo differential suite: once against the native kernel,
# once with the pure-Python fallback forced on. The sacrebleu oracle
# (sacrebleu==2.5.1) comes from the repo pixi environment (pixi.toml
# [pypi-dependencies], pinned and lock-verified); SACREBLEU_ORACLE_PATH=<dir>
# can point at an alternative checkout of the package.
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE="tests/test_sacrebleu_differential.py tests/test_sacrebleu_loader.py"

export PYTHONPATH="python/sacrebleu_mojo${SACREBLEU_ORACLE_PATH:+:${SACREBLEU_ORACLE_PATH}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

echo "== differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== differential suite: forced fallback (SACREBLEU_MOJO_DISABLE_NATIVE=1) =="
SACREBLEU_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
