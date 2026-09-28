#!/usr/bin/env bash
# End-user bar for ckmeans-mojo: pack the npm tarballs, install them into a
# fresh project exactly like an end user would, and run the simple-statistics
# README quick-start — once with the native platform package (macOS arm64 /
# Linux x64) and once with the core package alone (simulated Windows /
# unsupported platform, where the vendored simple-statistics fallback IS the
# product). Both runs must print byte-identical cluster results.
#
# Each fresh project is installed with `npm ci` against a lockfile written by
# typescript/scripts/consumer-lockfile.cjs: the tarballs pinned by the sha512
# of what was just packed, koffi pinned to the entry in the committed
# workspace lockfile. Every package is hash-verified on install; nothing is
# resolved at install time.
#
# Run from anywhere: bash typescript/ckmeans-mojo/scripts/smoke.sh
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${CKMEANS_MOJO_DIST:-}" ]; then
  dist="$here/../../$CKMEANS_MOJO_DIST"
  mkdir -p "$dist"
  trap 'rm -rf /tmp/thyn-ai-ckmeans-mojo-smoke-native /tmp/thyn-ai-ckmeans-mojo-smoke-win' EXIT
else
  dist="$(mktemp -d /tmp/thyn-ai-ckmeans-mojo-dist.XXXXXX)"
  trap 'rm -rf "$dist" /tmp/thyn-ai-ckmeans-mojo-smoke-native /tmp/thyn-ai-ckmeans-mojo-smoke-win' EXIT
fi

# npm pack names every tarball <name>-<version>.tgz from its package.json, so the
# version the smoke installs by exact filename is read from the same place.
version="$(node -p "require('$here/packages/core/package.json').version")"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform_pkg="thyn-ai-ckmeans-mojo-darwin-arm64-${version}.tgz" ;;
  Linux-x86_64) platform_pkg="thyn-ai-ckmeans-mojo-linux-x64-${version}.tgz" ;;
  *) echo "error: no platform package for $(uname -s)-$(uname -m); run the core-only fallback path manually" >&2; exit 1 ;;
esac

echo "== npm pack =="
for pkg in core darwin-arm64 linux-x64; do
  (cd "$here/packages/$pkg" && npm pack --silent --pack-destination "$dist" >/dev/null)
done
ls "$dist"

echo "== native smoke (fresh project, core + platform tarballs) =="
rm -rf /tmp/thyn-ai-ckmeans-mojo-smoke-native
mkdir -p /tmp/thyn-ai-ckmeans-mojo-smoke-native/vendor
cp "$dist/thyn-ai-ckmeans-mojo-core-${version}.tgz" "$dist/$platform_pkg" /tmp/thyn-ai-ckmeans-mojo-smoke-native/vendor/
cd /tmp/thyn-ai-ckmeans-mojo-smoke-native
node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-native > native.json
head -3 native.json

echo "== simulated-Windows smoke (core tarball only; fallback engages) =="
rm -rf /tmp/thyn-ai-ckmeans-mojo-smoke-win
mkdir -p /tmp/thyn-ai-ckmeans-mojo-smoke-win/vendor
cp "$dist/thyn-ai-ckmeans-mojo-core-${version}.tgz" /tmp/thyn-ai-ckmeans-mojo-smoke-win/vendor/
cd /tmp/thyn-ai-ckmeans-mojo-smoke-win
CONSUMER_LOCKFILE_EXCLUDE="@thyn-ai/ckmeans-mojo-linux-x64,@thyn-ai/ckmeans-mojo-darwin-arm64" node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-fallback > fallback.json
head -3 fallback.json
# The forced-fallback env switch must behave identically.
CKMEANS_MOJO_DISABLE_NATIVE=1 node quickstart.mjs --assert-fallback > fallback-env.json

echo "== comparing native vs fallback results =="
node -e '
const a = require("/tmp/thyn-ai-ckmeans-mojo-smoke-native/native.json")
const b = require("/tmp/thyn-ai-ckmeans-mojo-smoke-win/fallback.json")
const c = require("/tmp/thyn-ai-ckmeans-mojo-smoke-win/fallback-env.json")
if (a.backend !== "native") throw new Error("native run did not use the native backend")
if (b.backend !== "fallback" || c.backend !== "fallback") throw new Error("fallback run did not use the fallback backend")
const strip = ({ backend, ...rest }) => rest
if (JSON.stringify(strip(a)) !== JSON.stringify(strip(b))) throw new Error("native and fallback results differ")
if (JSON.stringify(strip(b)) !== JSON.stringify(strip(c))) throw new Error("env-forced fallback results differ")
console.log("smoke OK: native and fallback produce identical results")
'
