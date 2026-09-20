#!/usr/bin/env bash
# Full natural-mojo test pass: differential suite on the native backend, then
# again with the pure-JS fallback forced on.
set -euo pipefail

cd "$(dirname "$0")/.."
# `npm ci` installs exactly what package-lock.json records and verifies every
# package against its integrity hash, so the suite runs against the pinned
# koffi and oracle package rather than whatever a fresh resolution returns.
npm ci --no-audit --no-fund
echo "== differential + unit tests (native backend) =="
npm test
echo "== differential + unit tests (NATURAL_MOJO_DISABLE_NATIVE=1, vendored fallback) =="
npm run test:fallback
