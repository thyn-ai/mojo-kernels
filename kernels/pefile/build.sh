#!/usr/bin/env bash
# Compile the pefile Mojo kernel into a shared library (macOS .dylib / Linux .so).
# Run inside the repo pixi environment (PATH must contain `mojo`, e.g. via
# `export PATH="$PWD/.pixi/envs/default/bin:$PATH"` from the repository root).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libpefilemojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libpefilemojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

# The compiler locates its stdlib via MODULAR_HOME. `pixi run` activation sets
# it, but a bare PATH invocation does not; derive it from the toolchain layout.
if [ -z "${MODULAR_HOME:-}" ]; then
  mojo_bin="$(command -v mojo)"
  export MODULAR_HOME="$(cd "$(dirname "$mojo_bin")/../share/max" && pwd)"
fi

mojo build --emit shared-lib -o "build/${lib}" src/pefilemojo.mojo
echo "built kernels/pefile/build/${lib}"