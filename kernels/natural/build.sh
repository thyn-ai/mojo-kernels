#!/usr/bin/env bash
# Compile the natural-distance Mojo kernel into a shared library
# (macOS .dylib / Linux .so). Run inside the repo pixi environment:
#   ~/.pixi/bin/pixi run bash kernels/natural/build.sh
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libnaturalmojo.dylib"
    # macOS floor for the released platform packages (matches the pixi
    # platform pin). If the library ever fails to load on an older system,
    # the wrapper's pure-JS fallback engages, so this only widens
    # compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libnaturalmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the JS wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --emit shared-lib -o "build/${lib}" src/naturalmojo.mojo
echo "built kernels/natural/build/${lib}"
