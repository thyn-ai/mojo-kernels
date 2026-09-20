#!/usr/bin/env bash
# Compile the pymavlink Mojo kernel into a shared library (macOS .dylib /
# Linux .so). Run inside the repo pixi environment
# (`pixi run bash kernels/pymavlink/build.sh`).
#
# The kernel is a single self-contained module (the dialect tables are
# passed in through the C ABI; see gen_tables_pymavlink.py).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libpymavmojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libpymavmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --emit shared-lib -o "build/${lib}" src/pymavmojo.mojo
echo "built kernels/pymavlink/build/${lib}"
