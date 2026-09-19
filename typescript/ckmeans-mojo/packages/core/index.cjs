'use strict'

/**
 * @ckmeans-mojo/core — drop-in faster ckmeans for simple-statistics.
 *
 * The Ckmeans.1d.dp dynamic program runs on a native Mojo kernel when its
 * shared library is available (macOS arm64 / Linux x64 platform packages);
 * on any load failure the package transparently falls back to the vendored
 * simple-statistics implementation (ISC, see vendor/ and NOTICE), so
 * results are identical on every platform.
 */

const { ckmeansNative } = require('./src/engine.cjs')
const {
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
} = require('./src/native.cjs')
const vendorCkmeans = require('./vendor/ckmeans.cjs')

/**
 * Optimal 1-D k-means clustering, API-compatible with
 * simple-statistics' ckmeans(x, nClusters).
 *
 * @param {Array<number>} x input data (not mutated)
 * @param {number} nClusters number of clusters; must not exceed x.length
 * @returns {Array<Array<number>>} clustered input, ascending cluster order
 */
function ckmeans(x, nClusters) {
  try {
    return ckmeansNative(x, nClusters)
  } catch (err) {
    if (!(err instanceof NativeUnavailable)) {
      throw err
    }
    return vendorCkmeans(x, nClusters)
  }
}

/** True if the native kernel can cluster right now. Never throws. */
ckmeans.nativeAvailable = nativeAvailable

/** Diagnostics for the active backend. Never throws. */
ckmeans.backendInfo = backendInfo

/** Which backend serves calls on this platform: "native" or "fallback". */
Object.defineProperty(ckmeans, 'backend', {
  enumerable: true,
  get() {
    return nativeAvailable() ? 'native' : 'fallback'
  },
})

ckmeans.version = require('./package.json').version

module.exports = ckmeans
module.exports.default = ckmeans
module.exports.ckmeans = ckmeans
module.exports.nativeAvailable = nativeAvailable
module.exports.backendInfo = backendInfo
