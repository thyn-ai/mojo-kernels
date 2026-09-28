#!/usr/bin/env bash
# Compile the JSONPath Mojo kernel into a shared library (macOS .dylib / Linux .so).
# Run inside the repo pixi environment (see python/jsonpath_mojo/README.md).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libjsonpathmojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  Linux)  lib="libjsonpathmojo.so" ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

# Pin the target CPU the same way kernels/bm25 and kernels/fuse do. Without a target the compiler
# emits for the build host: a runner with AVX-512 then produces a library that SIGILLs on any CPU
# without it. x86-64-v3 (AVX2, FMA, BMI2) is every x86-64 CPU since Intel Haswell / AMD Excavator;
# scripts/check_x86_64_baseline.sh proves the Linux build stays on it.
case "$(uname -s)" in
  Darwin) target_cpu="apple-m1" ;;
  Linux)  target_cpu="x86-64-v3" ;;
  *) echo "error: no target CPU for $(uname -s)" >&2; exit 1 ;;
esac
mojo build --target-cpu "${target_cpu}" --emit shared-lib -o "build/${lib}" src/jsonpathmojo.mojo
if [ "$(uname -s)" = "Linux" ]; then
  bash ../../scripts/check_x86_64_baseline.sh "build/${lib}"
fi
echo "built kernels/jsonpath/build/${lib}"
