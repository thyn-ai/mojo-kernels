#!/usr/bin/env bash
# Compile the jinja2 lexer's Mojo kernel into a shared library
# (macOS .dylib / Linux .so). Run inside the repo pixi environment
# (`bash kernels/jinja2/build.sh` from the repository root).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libjinja2mojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # stock-jinja2 fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libjinja2mojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --emit shared-lib -o "build/${lib}" src/jinja2mojo.mojo
echo "built kernels/jinja2/build/${lib}"
