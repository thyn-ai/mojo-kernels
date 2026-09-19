#!/usr/bin/env bash
# natural-mojo differential suite: native backend, then forced fallback.
# Mirrors scripts/test_all_cclib.sh for the TypeScript kernels.
set -euo pipefail

cd "$(dirname "$0")/.."
bash typescript/natural-mojo/scripts/test_all.sh
