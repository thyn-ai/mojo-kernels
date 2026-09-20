#!/usr/bin/env bash
# Compile the networkx-graph Mojo kernel into a shared library
# (macOS .dylib / Linux .so). Run inside the repo pixi environment, e.g.
#   ~/.pixi/bin/pixi run bash kernels/networkx-graph/build.sh
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libnxgraphmojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libnxgraphmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

# `--fp-mode contract=off` is load-bearing: the default `contract=fast`
# fuses a + b*c into a single-rounded FMA, which breaks bit-for-bit parity
# with the oracle's CPython float64 evaluation (mul and add round
# separately). Strict IEEE double-rounding is what the differential suite
# pins; FMA buys nothing in these pointer-chasing loops anyway.
mojo build --emit shared-lib --fp-mode contract=off -o "build/${lib}" src/nxgraphmojo.mojo
echo "built kernels/networkx-graph/build/${lib}"
