#!/usr/bin/env bash
# Full minisearch-mojo test pass: differential + unit suite on the native
# backend, then again with the pure-JS fallback forced on.
set -euo pipefail

cd "$(dirname "$0")/.."
npm install --no-audit --no-fund
echo "== differential + unit tests (native backend) =="
npm test
echo "== differential + unit tests (MINISEARCH_MOJO_DISABLE_NATIVE=1, vendored fallback) =="
npm run test:fallback
