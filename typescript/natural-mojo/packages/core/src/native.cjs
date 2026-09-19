'use strict'

/**
 * koffi loader for the naturalmojo native kernel, with an ABI-version
 * handshake.
 *
 * Resolution order:
 *
 *   1. `$NATURAL_MOJO_NATIVE_LIB` (explicit path override, for development)
 *   2. the platform package (`@natural-mojo/<platform>-<arch>`) installed as
 *      an optional dependency of `@natural-mojo/core`
 *   3. the repository development build output `kernels/natural/build/`
 *
 * If the kernel cannot be found, fails to load, or reports an ABI version
 * this package does not understand, `NativeUnavailable` is thrown and the
 * caller falls back to the vendored natural implementation. Set
 * `NATURAL_MOJO_DISABLE_NATIVE=1` to force that fallback (used by the
 * differential test suite and on platforms without a prebuilt library).
 *
 * Stable C ABI (v1):
 *
 *   int32_t naturalmojo_abi_version(void)
 *   double  naturalmojo_distance(const uint16_t* source, int32_t source_len,
 *                                const uint16_t* target, int32_t target_len,
 *                                double insertion_cost, double deletion_cost,
 *                                double substitution_cost,
 *                                double transposition_cost,
 *                                int32_t damerau, int32_t restricted)
 *
 * The kernel is stateless: one call computes one pair's distance and all
 * scratch is allocated and freed inside the call, so there is no handle
 * lifecycle to manage here.
 */

const fs = require('node:fs')
const path = require('node:path')

// Must equal ABI_VERSION in kernels/natural/src/naturalmojo.mojo. A mismatch
// means the installed package and the resolved shared library disagree;
// fall back.
const ABI_VERSION = 1

const ENV_LIB = 'NATURAL_MOJO_NATIVE_LIB'
const ENV_DISABLE = 'NATURAL_MOJO_DISABLE_NATIVE'

class NativeUnavailable extends Error {
  constructor(message) {
    super(message)
    this.name = 'NativeUnavailable'
    this.code = 'NATURAL_MOJO_NATIVE_UNAVAILABLE'
  }
}

function libBasename() {
  if (process.platform === 'darwin') return 'libnaturalmojo.dylib'
  if (process.platform === 'linux') return 'libnaturalmojo.so'
  if (process.platform === 'win32') return 'naturalmojo.dll' // no Mojo toolchain builds this today
  return 'libnaturalmojo.so'
}

function candidatePaths() {
  /** [sourceLabel, absolutePath] candidates, in resolver order. */
  const out = []
  const override = process.env[ENV_LIB]
  if (override) {
    out.push([`env ${ENV_LIB}`, override])
  }
  // Platform package, resolved relative to this package so npm's nested or
  // hoisted layouts both work.
  const platformPkg = `@natural-mojo/${process.platform}-${process.arch}`
  try {
    const pkgJson = require.resolve(`${platformPkg}/package.json`, { paths: [__dirname] })
    out.push([`platform package ${platformPkg}`, path.join(path.dirname(pkgJson), 'lib', libBasename())])
  } catch {
    // optional dependency not installed (unsupported platform or pruned)
  }
  out.push([
    'repo-dev build output',
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'kernels', 'natural', 'build', libBasename()),
  ])
  return out
}

function bindAbi(lib) {
  // koffi parses C prototypes; TypedArrays passed to pointer parameters are
  // pinned for the duration of the call.
  return {
    abi_version: lib.func('int32_t naturalmojo_abi_version(void)'),
    distance: lib.func(`double naturalmojo_distance(
      const uint16_t* source, int32_t source_len,
      const uint16_t* target, int32_t target_len,
      double insertion_cost, double deletion_cost,
      double substitution_cost, double transposition_cost,
      int32_t damerau, int32_t restricted)`),
  }
}

let _koffi = null
let _lib = null
let _libSource = null
let _loadError = null

function loadKoffi() {
  if (_koffi) return _koffi
  try {
    _koffi = require('koffi')
  } catch (err) {
    throw new NativeUnavailable(`koffi is not available: ${err.message}`)
  }
  return _koffi
}

function load() {
  /** Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure. */
  if (process.env[ENV_DISABLE] === '1') {
    throw new NativeUnavailable(`native kernel disabled by ${ENV_DISABLE}=1`)
  }
  if (_lib !== null) {
    return _lib
  }
  const koffi = loadKoffi()
  const errors = []
  for (const [label, candidate] of candidatePaths()) {
    let resolved
    try {
      if (!candidate || !fs.existsSync(candidate)) {
        continue
      }
      try {
        resolved = koffi.load(candidate)
      } catch (err) {
        errors.push(`${label} (${candidate}): ${err.message}`)
        continue
      }
      let abi
      try {
        const bound = bindAbi(resolved)
        abi = bound.abi_version()
        if (abi !== ABI_VERSION) {
          errors.push(`${label} (${candidate}): native ABI v${abi} != wrapper ABI v${ABI_VERSION}`)
          continue
        }
        _lib = bound
        _libSource = `${label} (${candidate})`
        _loadError = null
        return bound
      } catch (err) {
        errors.push(`${label} (${candidate}): ABI not recognized: ${err.message}`)
        continue
      }
    } catch (err) {
      errors.push(`${label}: ${err.message}`)
    }
  }
  _loadError = errors.join('; ') || 'no native kernel found on any resolver path'
  throw new NativeUnavailable(_loadError)
}

function nativeAvailable() {
  /** True if the native kernel can compute distances right now. Never throws. */
  try {
    load()
    return true
  } catch {
    return false
  }
}

function backendInfo() {
  /** Diagnostics for the active backend. Never throws. */
  const info = {
    native_available: false,
    native_source: null,
    abi_version_expected: ABI_VERSION,
    abi_version_native: null,
    disabled_by_env: process.env[ENV_DISABLE] === '1',
    platform: process.platform,
    arch: process.arch,
    error: null,
  }
  try {
    const lib = load()
    info.native_available = true
    info.native_source = _libSource
    info.abi_version_native = lib.abi_version()
  } catch (err) {
    info.error = err.message
  }
  return info
}

/**
 * Compute one pair's distance natively. Throws NativeUnavailable when the
 * kernel cannot be loaded; the caller then falls back.
 *
 * `sourceU16`/`targetU16` are Uint16Arrays of UTF-16 code units; costs are
 * finalized float64 values (NaN transposition_cost disables transpositions,
 * matching the reference's NaN-min semantics).
 */
function distanceNative(sourceU16, targetU16, costs, damerau, restricted) {
  const lib = load() // throws NativeUnavailable
  return lib.distance(
    sourceU16,
    sourceU16.length,
    targetU16,
    targetU16.length,
    costs.insertion_cost,
    costs.deletion_cost,
    costs.substitution_cost,
    costs.transposition_cost,
    damerau ? 1 : 0,
    restricted ? 1 : 0
  )
}

module.exports = {
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
  distanceNative,
  ABI_VERSION,
  ENV_LIB,
  ENV_DISABLE,
}
