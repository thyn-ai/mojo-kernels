#!/usr/bin/env bash
# Compile the langdetect Mojo kernel into a shared library (macOS .dylib / Linux .so).
# Run inside the repo pixi environment, e.g.
#   ~/.pixi/bin/pixi run bash kernels/langdetect/build.sh
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="liblangdetectmojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="liblangdetectmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --emit shared-lib -o "build/${lib}" src/langdetectmojo.mojo
echo "built kernels/langdetect/build/${lib}"
