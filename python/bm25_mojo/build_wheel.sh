#!/usr/bin/env bash
# Build the bm25-mojo platform wheel (wheel-only; no sdist).
# Run inside the repo pixi environment (`pixi run wheel-bm25`).
set -euo pipefail

cd "$(dirname "$0")"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
rm -rf dist
python -m build --wheel --no-isolation --outdir dist .
ls -la dist/
