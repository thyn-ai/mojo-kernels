#!/usr/bin/env bash
# Compile the Ckmeans.1d.dp Mojo kernel into a shared library
# (macOS .dylib / Linux .so). Run inside the repo pixi environment, e.g.
# `pixi run bash kernels/ckmeans/build.sh`.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libckmeansmojo.dylib"
    # macOS floor for the released platform packages (matches the pixi
    # platform pin). If the library ever fails to load on an older system,
    # the wrapper's pure-JS fallback engages, so this only widens
    # compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)
    lib="libckmeansmojo.so"
    ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the JS wrapper will use its fallback" >&2
    exit 1
    ;;
esac

# contract=off: keep IEEE-754 float64 operation order bit-identical to the
# JS reference (the default contract=fast fuses a+b*c into FMAs, which
# perturbs last-ulp results and can flip split-point ties).
mojo build --fp-mode contract=off --emit shared-lib -o "build/${lib}" src/ckmeansmojo.mojo
echo "built kernels/ckmeans/build/${lib}"
