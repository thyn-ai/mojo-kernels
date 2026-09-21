#!/usr/bin/env bash
# Compile the Fuse Bitap Mojo kernel + pthread shim into shared libraries
# (macOS .dylib / Linux .so). Run inside the repo pixi environment
# (`pixi run build-kernel-fuse`).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
  Darwin)
    lib="libfusemojo.dylib"
    shim="libfusemojoshim.dylib"
    shared_flag="-dynamiclib"
    rpath_flag="-Wl,-rpath,@loader_path"
    # macOS floor for the released platform packages (matches the pixi
    # platform pin). If the library ever fails to load on an older system,
    # the wrapper's pure-JS fallback engages, so this only widens
    # compatibility.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    # Every Apple silicon Mac runs apple-m1 code; the host default would
    # be the build machine's own generation.
    target_cpu="apple-m1"
    ;;
  Linux)
    lib="libfusemojo.so"
    shim="libfusemojoshim.so"
    shared_flag="-shared"
    rpath_flag="-Wl,-rpath,\$ORIGIN"
    # x86-64-v3 (AVX2, FMA, BMI2): every x86-64 CPU since Intel Haswell
    # and AMD Zen, and every GitHub-hosted runner. The host default is the
    # build machine's CPU; a kernel built on an AVX-512 runner dies with
    # SIGILL on any CPU without AVX-512 (scripts/check_x86_64_baseline.sh).
    target_cpu="x86-64-v3"
    ;;
  *)
    echo "error: no Mojo toolchain for $(uname -s); the JS wrapper will use its fallback" >&2
    exit 1
    ;;
esac

mojo build --target-cpu "${target_cpu}" --emit shared-lib -o "build/${lib}" src/fusemojo.mojo
if [ "$(uname -s)" = "Linux" ]; then
  bash ../../scripts/check_x86_64_baseline.sh "build/${lib}"
fi
cc -O2 -fPIC ${shared_flag} -o "build/${shim}" src/shim.c -Lbuild -lfusemojo ${rpath_flag}
if [ "$(uname -s)" = "Darwin" ]; then
  # The link records the kernel library by its relative build path; both
  # libraries always ship in the same directory, so anchor it there.
  install_name_tool -change "build/${lib}" "@loader_path/${lib}" "build/${shim}"
fi
echo "built kernels/fuse/build/${lib} and ${shim}"
