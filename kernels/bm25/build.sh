#!/usr/bin/env bash
# Compile the BM25 Mojo kernel into a shared library (macOS .dylib / Linux .so).
# Run inside the repo pixi environment (`pixi run build-kernel-bm25`).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libbm25mojo.dylib"
    # macOS floor for the released wheels (matches the pixi platform pin).
    # If the library ever fails to load on an older system, the wrapper's
    # pure-Python fallback engages, so this only widens compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    # Every Apple silicon Mac runs apple-m1 code; the host default would
    # be the build machine's own generation.
    target_cpu="apple-m1"
    ;;
  Linux)
    lib="libbm25mojo.so"
    # x86-64-v3 (AVX2, FMA, BMI2): every x86-64 CPU since Intel Haswell
    # and AMD Zen, and every GitHub-hosted runner. The host default is the
    # build machine's CPU; a kernel built on an AVX-512 runner dies with
    # SIGILL on any CPU without AVX-512 (scripts/check_x86_64_baseline.sh).
    target_cpu="x86-64-v3"
    ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the Python wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --target-cpu "${target_cpu}" --emit shared-lib -o "build/${lib}" src/bm25mojo.mojo
if [ "$(uname -s)" = "Linux" ]; then
  bash ../../scripts/check_x86_64_baseline.sh "build/${lib}"
fi
echo "built kernels/bm25/build/${lib}"
