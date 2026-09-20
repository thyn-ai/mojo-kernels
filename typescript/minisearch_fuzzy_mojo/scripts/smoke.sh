#!/usr/bin/env bash
# End-user bar for minisearch-mojo: pack the npm tarballs, install them into
# a fresh project exactly like an end user would, and run the README
# quick-start — once with the native platform package (macOS arm64 / Linux
# x64) and once with the core package alone (simulated Windows / unsupported
# platform, where the vendored MiniSearch fallback IS the product). Both
# runs must print identical results.
#
# Each fresh project is installed with `npm ci` against a lockfile written by
# typescript/scripts/consumer-lockfile.cjs: the tarballs pinned by the sha512
# of what was just packed, koffi pinned to the entry in the committed
# workspace lockfile. Every package is hash-verified on install; nothing is
# resolved at install time.
#
# Run from anywhere: bash typescript/minisearch_fuzzy_mojo/scripts/smoke.sh
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${MINISEARCH_MOJO_DIST:-}" ]; then
  dist="$here/../../$MINISEARCH_MOJO_DIST"
  mkdir -p "$dist"
  trap 'rm -rf /tmp/ms-mojo-smoke-native /tmp/ms-mojo-smoke-win' EXIT
else
  dist="$(mktemp -d /tmp/ms-mojo-dist.XXXXXX)"
  trap 'rm -rf "$dist" /tmp/ms-mojo-smoke-native /tmp/ms-mojo-smoke-win' EXIT
fi

# npm pack names every tarball <name>-<version>.tgz from its package.json, so the
# version the smoke installs by exact filename is read from the same place.
version="$(node -p "require('$here/packages/core/package.json').version")"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform_pkg="minisearch-mojo-darwin-arm64-${version}.tgz" ;;
  Linux-x86_64) platform_pkg="minisearch-mojo-linux-x64-${version}.tgz" ;;
  *) echo "error: no platform package for $(uname -s)-$(uname -m); run the core-only fallback path manually" >&2; exit 1 ;;
esac

echo "== npm pack =="
for pkg in core darwin-arm64 linux-x64; do
  (cd "$here/packages/$pkg" && npm pack --silent --pack-destination "$dist" >/dev/null)
done
ls "$dist"

echo "== native smoke (fresh project, core + platform tarballs) =="
rm -rf /tmp/ms-mojo-smoke-native
mkdir -p /tmp/ms-mojo-smoke-native/vendor
cp "$dist/minisearch-mojo-core-${version}.tgz" "$dist/$platform_pkg" /tmp/ms-mojo-smoke-native/vendor/
cd /tmp/ms-mojo-smoke-native
node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-native > native.json
head -5 native.json

echo "== simulated-Windows smoke (core tarball only; fallback engages) =="
rm -rf /tmp/ms-mojo-smoke-win
mkdir -p /tmp/ms-mojo-smoke-win/vendor
cp "$dist/minisearch-mojo-core-${version}.tgz" /tmp/ms-mojo-smoke-win/vendor/
cd /tmp/ms-mojo-smoke-win
node "$here/../scripts/consumer-lockfile.cjs" "$here/package-lock.json" vendor/*.tgz
npm ci --no-audit --no-fund --loglevel=error
cp "$here/quickstart.mjs" .
node quickstart.mjs --assert-fallback > fallback.json
head -5 fallback.json
# The forced-fallback env switch must behave identically.
MINISEARCH_MOJO_DISABLE_NATIVE=1 node quickstart.mjs --assert-fallback > fallback-env.json

echo "== comparing native vs fallback results =="
node -e '
const a = require("/tmp/ms-mojo-smoke-native/native.json")
const b = require("/tmp/ms-mojo-smoke-win/fallback.json")
const c = require("/tmp/ms-mojo-smoke-win/fallback-env.json")
if (a.backend !== "native") throw new Error("native run did not use the native backend")
if (b.backend !== "fallback" || c.backend !== "fallback") throw new Error("fallback run did not use the fallback backend")
const strip = ({ backend, ...rest }) => rest
const round = (x) => JSON.parse(JSON.stringify(x, (k, v) => (typeof v === "number" ? Math.round(v * 1e9) / 1e9 : v)))
// The vendored fallback is the reference implementation; scores are
// identical by construction, but round to the 1e-9 differential gate for
// robust cross-engine comparison.
if (JSON.stringify(round(strip(a))) !== JSON.stringify(round(strip(b)))) throw new Error("native and fallback results differ")
if (JSON.stringify(round(strip(b))) !== JSON.stringify(round(strip(c)))) throw new Error("env-forced fallback results differ")
console.log("smoke OK: native and fallback produce identical results")
'
