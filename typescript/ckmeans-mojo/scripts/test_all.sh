#!/usr/bin/env bash
# Full ckmeans-mojo test pass: differential suite on the native backend, then
# again with the pure-JS fallback forced on (mirrors scripts/test_all.sh for
# the Python kernels).
set -euo pipefail

cd "$(dirname "$0")/.."
# `npm ci` installs exactly what package-lock.json records and verifies every
# package against its integrity hash, so the suite runs against the pinned
# koffi and oracle package rather than whatever a fresh resolution returns.
npm ci --no-audit --no-fund
echo "== differential + unit tests (native backend) =="
npm test
echo "== differential + unit tests (CKMEANS_MOJO_DISABLE_NATIVE=1, vendored fallback) =="
npm run test:fallback
