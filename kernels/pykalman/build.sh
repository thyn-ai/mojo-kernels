#!/usr/bin/env bash
# Compile the pykalman Mojo kernel into a shared library (macOS .dylib / Linux .so).
# Run directly (`bash kernels/pykalman/build.sh`) inside the repo pixi environment.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libpykalmanmojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libpykalmanmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --emit shared-lib -o "build/${lib}" src/pykalmanmojo.mojo
echo "built kernels/pykalman/build/${lib}"
