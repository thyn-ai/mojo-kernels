#!/usr/bin/env bash
# fast-check parity properties for fuse-mojo: native backend, then the
# vendored fallback forced on (mirrors scripts/test_all.sh). Run via
# `pixi run fuzz-fuse` or `npm run test:property:all`.
#
#   FC_NUM_RUNS  scenarios per property (default 200; the nightly fuzz
#                workflow raises it)
#   FC_SEED      replay the seed a failing run printed
#   FC_PATH      replay one counterexample path (together with FC_SEED)
set -euo pipefail

cd "$(dirname "$0")/.."
npm ci --no-audit --no-fund
echo "== parity properties: native backend (FC_NUM_RUNS=${FC_NUM_RUNS:-200}) =="
npm run test:property
echo
echo "== parity properties: FUSE_MOJO_DISABLE_NATIVE=1, vendored fallback (FC_NUM_RUNS=${FC_NUM_RUNS:-200}) =="
npm run test:property:fallback
