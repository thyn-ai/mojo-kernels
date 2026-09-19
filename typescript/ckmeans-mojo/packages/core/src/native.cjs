'use strict'

/**
 * koffi loader for the ckmeansmojo native kernel, with an ABI-version
 * handshake.
 *
 * Resolution order:
 *
 *   1. `$CKMEANS_MOJO_NATIVE_LIB` (explicit path override, for development)
 *   2. the platform package (`@ckmeans-mojo/<platform>-<arch>`) installed as
 *      an optional dependency of `@ckmeans-mojo/core`
 *   3. the repository development build output `kernels/ckmeans/build/`
 *
 * If the kernel cannot be found, fails to load, or reports an ABI version
 * this package does not understand, `NativeUnavailable` is thrown and the
 * caller falls back to the vendored simple-statistics implementation. Set
 * `CKMEANS_MOJO_DISABLE_NATIVE=1` to force that fallback (used by the
 * differential test suite and on platforms without a prebuilt library).
 *
 * Stable C ABI (v1):
 *
 *   int32_t ckmeansmojo_abi_version(void)
 *   int32_t ckmeansmojo_cluster(const double* sorted, int32_t n, int32_t k,
 *                               int32_t* out_lefts)
 */

const fs = require('node:fs')
const path = require('node:path')

// Must equal ABI_VERSION in kernels/ckmeans/src/ckmeansmojo.mojo. A mismatch
// means the installed package and the resolved shared library disagree;
// fall back.
const ABI_VERSION = 1

const ENV_LIB = 'CKMEANS_MOJO_NATIVE_LIB'
const ENV_DISABLE = 'CKMEANS_MOJO_DISABLE_NATIVE'

class NativeUnavailable extends Error {
  constructor(message) {
    super(message)
    this.name = 'NativeUnavailable'
    this.code = 'CKMEANS_MOJO_NATIVE_UNAVAILABLE'
  }
}

function libBasename() {
  if (process.platform === 'darwin') return 'libckmeansmojo.dylib'
  if (process.platform === 'linux') return 'libckmeansmojo.so'
  if (process.platform === 'win32') return 'ckmeansmojo.dll' // no Mojo toolchain builds this today
  return 'libckmeansmojo.so'
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
  const platformPkg = `@ckmeans-mojo/${process.platform}-${process.arch}`
  try {
    const pkgJson = require.resolve(`${platformPkg}/package.json`, { paths: [__dirname] })
    out.push([`platform package ${platformPkg}`, path.join(path.dirname(pkgJson), 'lib', libBasename())])
  } catch {
    // optional dependency not installed (unsupported platform or pruned)
  }
  out.push([
    'repo-dev build output',
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'kernels', 'ckmeans', 'build', libBasename()),
  ])
  return out
}

function bindAbi(lib) {
  // koffi parses C prototypes; TypedArrays passed to pointer parameters are
  // pinned for the duration of the call.
  return {
    abi_version: lib.func('int32_t ckmeansmojo_abi_version(void)'),
    cluster: lib.func(
      'int32_t ckmeansmojo_cluster(const double* sorted, int32_t n, int32_t k, int32_t* out_lefts)'
    ),
    cluster_unsorted: lib.func(
      'int32_t ckmeansmojo_cluster_unsorted(const double* data, int32_t n, int32_t k, int32_t* out_lefts, double* out_sorted)'
    ),
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
    if (!candidate || !fs.existsSync(candidate)) {
      continue
    }
    let resolved
    try {
      resolved = koffi.load(candidate)
    } catch (err) {
      errors.push(`${label} (${candidate}): ${err.message}`)
      continue
    }
    try {
      const bound = bindAbi(resolved)
      const abi = bound.abi_version()
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
  }
  _loadError = errors.join('; ') || 'no native kernel found on any resolver path'
  throw new NativeUnavailable(_loadError)
}

function nativeAvailable() {
  /** True if the native kernel can cluster right now. Never throws. */
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
 * Cluster `sortedValues` (an ascending-sorted JS array) into k clusters on
 * the native kernel. Returns the Int32Array of k cluster-left indices.
 * Throws NativeUnavailable if the kernel is missing or rejects the input.
 */
function clusterNative(sortedValues, k) {
  const lib = load() // throws NativeUnavailable
  const n = sortedValues.length
  // Element conversion matches the reference's arithmetic on the raw values:
  // non-numbers become NaN here exactly where the reference would produce
  // NaN in its sums, and IEEE comparisons then behave identically.
  const data = Float64Array.from(sortedValues)
  const lefts = new Int32Array(k)
  const rc = lib.cluster(data, n, k, lefts)
  if (rc !== 0) {
    // Unreachable by contract (the caller pre-validates 1 <= k <= n); a
    // status here means wrapper and kernel disagree — treat the kernel as
    // unavailable rather than trust a suspect result.
    throw new NativeUnavailable(`native kernel rejected the input (status ${rc})`)
  }
  return lefts
}

/**
 * Full native pipeline for finite data (the common path): the kernel
 * stable-sorts natively — bit-identical order to the reference's stable
 * Array sort — and runs the DP in the same call. `data` must contain no
 * NaN (the caller scans and routes NaN input to clusterNative instead, so
 * NaN sort semantics stay the engine's own).
 *
 * Returns { lefts, sorted } (kernel-sorted values plus the k cluster-left
 * indices into them), or { singleUnique: true, sorted } when the data has
 * one unique value. Throws NativeUnavailable on load/contract failure.
 */
function clusterNativeUnsorted(data, k) {
  const lib = load() // throws NativeUnavailable
  const n = data.length
  const lefts = new Int32Array(k)
  const sorted = new Float64Array(n)
  const rc = lib.cluster_unsorted(data, n, k, lefts, sorted)
  if (rc === 10) {
    return { singleUnique: true, sorted }
  }
  if (rc !== 0) {
    throw new NativeUnavailable(`native kernel rejected the input (status ${rc})`)
  }
  return { singleUnique: false, lefts, sorted }
}

module.exports = {
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
  clusterNative,
  clusterNativeUnsorted,
  ABI_VERSION,
  ENV_LIB,
  ENV_DISABLE,
}
