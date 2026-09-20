#!/usr/bin/env bash
# End-user bar for fuse-mojo: pack the npm tarballs, install them into a
# fresh project exactly like an end user would, and run the Fuse.js README
# quick-start — once with the native platform package (macOS arm64 / Linux
# x64) and once with the core package alone (simulated Windows / unsupported
# platform, where the vendored Fuse.js fallback IS the product). Both runs
# must print byte-identical search results.
#
# Each fresh project is installed with `npm ci` against a lockfile written by
# typescript/scripts/consumer-lockfile.cjs: the tarballs pinned by the sha512
# of what was just packed, koffi pinned to the entry in the committed
# workspace lockfile.
# Every package is hash-verified on install; nothing is resolved at install
# time.
#
# Run from anywhere: bash typescript/fuse-mojo/scripts/smoke.sh
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${FUSE_MOJO_DIST:-}" ]; then
  dist="$here/../../$FUSE_MOJO_DIST"
  mkdir -p "$dist"
  trap 'rm -rf /tmp/fuse-mojo-smoke-native /tmp/fuse-mojo-smoke-win' EXIT
else
  dist="$(mktemp -d /tmp/fuse-mojo-dist.XXXXXX)"
  trap 'rm -rf "$dist" /tmp/fuse-mojo-smoke-native /tmp/fuse-mojo-smoke-win' EXIT
fi

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform_pkg="fuse-mojo-darwin-arm64-0.1.0.tgz" ;;
  Linux-x86_64) platform_pkg="fuse-mojo-linux-x64-0.1.0.tgz" ;;
  *) echo "error: no platform package for $(uname -s)-$(uname -m); run the core-only fallback path manually" >&2; exit 1 ;;
esac

echo "== npm pack =="
for pkg in core darwin-arm64 linux-x64; do
  (cd "$here/packages/$pkg" && npm pack --silent --pack-destination "$dist" >/dev/null)
done
ls "$dist"

echo "== native smoke (fresh project, core + platform tarballs) =="
rm -rf /tmp/fuse-mojo-smoke-native
mkdir -p /tmp/fuse-mojo-smoke-native/vendor
cp "$dist/fuse-mojo-core-0.1.0.tgz" "$dist/$platform_pkg" /tmp/fuse-mojo-smoke-native/vendor/
cd /tmp/fuse-mojo-smoke-native
node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-native > native.json
head -3 native.json

echo "== simulated-Windows smoke (core tarball only; fallback engages) =="
rm -rf /tmp/fuse-mojo-smoke-win
mkdir -p /tmp/fuse-mojo-smoke-win/vendor
cp "$dist/fuse-mojo-core-0.1.0.tgz" /tmp/fuse-mojo-smoke-win/vendor/
cd /tmp/fuse-mojo-smoke-win
node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-fallback > fallback.json
head -3 fallback.json
# The forced-fallback env switch must behave identically.
FUSE_MOJO_DISABLE_NATIVE=1 node quickstart.mjs --assert-fallback > fallback-env.json

echo "== comparing native vs fallback results =="
node -e '
const a = require("/tmp/fuse-mojo-smoke-native/native.json")
const b = require("/tmp/fuse-mojo-smoke-win/fallback.json")
const c = require("/tmp/fuse-mojo-smoke-win/fallback-env.json")
if (a.backend !== "native") throw new Error("native run did not use the native backend")
if (b.backend !== "fallback" || c.backend !== "fallback") throw new Error("fallback run did not use the fallback backend")
const strip = ({ backend, ...rest }) => rest
if (JSON.stringify(strip(a)) !== JSON.stringify(strip(b))) throw new Error("native and fallback results differ")
if (JSON.stringify(strip(b)) !== JSON.stringify(strip(c))) throw new Error("env-forced fallback results differ")
console.log("smoke OK: native and fallback produce identical results")
'
