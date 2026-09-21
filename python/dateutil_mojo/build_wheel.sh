#!/usr/bin/env bash
# Build the dateutil-mojo platform wheel (wheel-only; no sdist), then repair it:
# vendor the Mojo runtime shared libraries into the wheel and rewrite the
# kernel library's load paths to be wheel-relative ($ORIGIN / @loader_path).
# Without this step the wheel's libdateutilmojo resolves its runtime dependencies
# through an absolute rpath into the build machine's pixi environment, so it
# only ever loads on the machine that built it — everywhere else the wrapper
# silently falls back. dist/ contains the repaired, self-contained wheel.
#
# Run inside the repo pixi environment.
#
# Note: the repair vendors Modular's Mojo runtime libraries into the wheel.
# Redistribution terms for those binaries should be confirmed with Modular
# before any public release of the wheels.
set -euo pipefail

cd "$(dirname "$0")"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
rm -rf dist dist-repaired
python -m build --wheel --no-isolation --outdir dist .
raw="$(ls dist/*.whl)"

case "$(uname -s)" in
  Darwin) delocate-wheel -v -w dist-repaired "$raw" ;;
  Linux)  auditwheel repair "$raw" -w dist-repaired ;;
  *)      echo "no wheel repair tool for $(uname -s); shipping unrepaired wheel" >&2
          mkdir -p dist-repaired && cp "$raw" dist-repaired/ ;;
esac

rm -rf dist
mv dist-repaired dist
ls -la dist/
