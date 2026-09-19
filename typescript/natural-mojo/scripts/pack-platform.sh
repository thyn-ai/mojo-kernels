#!/usr/bin/env bash
# Populate the platform package (packages/<platform>-<arch>/lib/) with the
# compiled naturalmojo kernel, then repair it: vendor the Mojo runtime shared
# libraries into the package and rewrite the kernel library's load paths to
# be package-relative (@loader_path / $ORIGIN). Without this step the kernel
# library resolves its runtime dependencies through an absolute rpath into
# the build machine's pixi environment, so it only ever loads on the machine
# that built it — everywhere else the wrapper silently falls back.
#
# Run inside the repo pixi environment.
#
# Note: the repair vendors Modular's Mojo runtime libraries into the npm
# package. Redistribution terms for those binaries should be confirmed with
# Modular before any public release of the packages.
set -euo pipefail

cd "$(dirname "$0")/.."
repo_root="$(cd ../.. && pwd)"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  pkg_dir="packages/darwin-arm64"; lib="libnaturalmojo.dylib" ;;
  Linux-x86_64)  pkg_dir="packages/linux-x64";  lib="libnaturalmojo.so" ;;
  *) echo "error: no platform package for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

src="${NATURAL_MOJO_NATIVE_SRC:-$repo_root/kernels/natural/build/$lib}"
[ -f "$src" ] || { echo "error: kernel not built: $src" >&2; exit 1; }

mkdir -p "$pkg_dir/lib"
cp "$src" "$pkg_dir/lib/$lib"

case "$(uname -s)" in
  Darwin)
    # delocate-path scans the directory, copies non-system dylib dependencies
    # (the Mojo runtime from the pixi env) into it, and rewrites load paths
    # to @loader_path-relative ones.
    delocate-path -v "$pkg_dir/lib"
    ;;
  Linux)
    # Vendor every non-system shared library the kernel needs and point its
    # rpath at $ORIGIN so it resolves them from its own directory.
    pixi_lib="$repo_root/.pixi/envs/default/lib"
    while read -r dep; do
      case "$dep" in
        "$pixi_lib"/*)
          cp "$dep" "$pkg_dir/lib/"
          ;;
      esac
    done < <(ldd "$pkg_dir/lib/$lib" | awk '/=>/ && $3 != "" { print $3 }')
    patchelf --set-rpath '$ORIGIN' "$pkg_dir/lib/$lib"
    for f in "$pkg_dir/lib"/*.so; do
      [ "$f" = "$pkg_dir/lib/$lib" ] || patchelf --set-rpath '$ORIGIN' "$f" || true
    done
    ;;
esac

echo "packed $pkg_dir:"
ls -la "$pkg_dir/lib"
