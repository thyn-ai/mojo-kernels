#!/usr/bin/env bash
# Full fuse-mojo test pass: differential suite on the native backend, then
# again with the pure-JS fallback forced on (mirrors scripts/test_all.sh for
# the Python kernels).
set -euo pipefail

cd "$(dirname "$0")/.."
npm install --no-audit --no-fund
echo "== differential + unit tests (native backend) =="
npm test
echo "== differential + unit tests (FUSE_MOJO_DISABLE_NATIVE=1, vendored fallback) =="
npm run test:fallback
