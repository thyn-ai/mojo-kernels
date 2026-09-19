'use strict'

/**
 * @fuse-mojo/core — drop-in faster replacement for Fuse.js v7 fuzzy search.
 *
 * The heavy Bitap matching runs on a native Mojo kernel when its shared
 * library is available (macOS arm64 / Linux x64 platform packages); on any
 * load failure the package transparently falls back to the vendored Fuse.js
 * implementation (Apache-2.0, see vendor/ and NOTICE), so results are
 * identical on every platform.
 */

const {
  DEFAULT_CONFIG,
  UnsupportedOptionError,
  validateOptions,
  createKeyStore,
} = require('./src/options.cjs')
const { buildEngine, nativeSearch } = require('./src/engine.cjs')
const {
  NativeIndex,
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
} = require('./src/native.cjs')
const VendorFuse = require('./vendor/fuse.basic.cjs')

class FuseMojo {
  /**
   * @param {Array} docs string list or array of objects
   * @param {object} options Fuse.js-compatible options (see README for the
   *   supported subset; unsupported options throw UnsupportedOptionError)
   */
  constructor(docs, options = {}, index) {
    if (index !== undefined) {
      throw new UnsupportedOptionError('external index (Fuse.createIndex/parseIndex)')
    }
    this.options = { ...DEFAULT_CONFIG, ...options }
    if (this.options.keys === undefined) {
      this.options.keys = []
    }
    validateOptions(this.options)
    this._keyStore = createKeyStore(this.options.keys)
    this._fallback = null
    this._engine = null
    this._nativeIndex = null
    this.setCollection(docs)
  }

  setCollection(docs, index) {
    if (index !== undefined) {
      throw new UnsupportedOptionError('external index (Fuse.createIndex/parseIndex)')
    }
    this._docs = docs
    this._destroyNative()
    this._fallback = null
    this._engine = null
    this._nativeIndex = null
    try {
      this._engine = buildEngine(docs, this._keyStore, this.options)
      const nativeOptions = {
        location: this.options.location,
        distance: this.options.distance,
        threshold: this.options.threshold,
        minMatchCharLength: Math.max(this.options.minMatchCharLength, 0),
        findAllMatches: this.options.findAllMatches,
        ignoreLocation: this.options.ignoreLocation,
        computeMatches: this.options.minMatchCharLength > 1 || this.options.includeMatches,
        includeMatches: this.options.includeMatches,
      }
      this._nativeIndex = new NativeIndex(this._engine.chars, this._engine.offsets, nativeOptions)
      this._backend = 'native'
    } catch (err) {
      if (!(err instanceof NativeUnavailable)) {
        throw err
      }
      this._engine = null
      this._nativeIndex = null
      this._fallback = new VendorFuse(docs, this.options)
      this._backend = 'fallback'
      this._fallbackError = err.message
    }
    return this
  }

  /**
   * @param {string} pattern search pattern (extended-search syntax is not
   *   supported; non-string patterns throw)
   * @param {{limit?: number}} opts
   * @returns {Array<{item: any, refIndex: number, score?: number, matches?: Array}>}
   */
  search(pattern, { limit = -1 } = {}) {
    if (typeof pattern !== 'string') {
      throw new UnsupportedOptionError(
        `non-string query (${typeof pattern === 'object' && pattern !== null ? 'logical $and/$or query' : typeof pattern})`
      )
    }
    if (this._backend === 'native') {
      return nativeSearch(this._engine, this._nativeIndex, pattern, limit, this.options)
    }
    return this._fallback.search(pattern, { limit })
  }

  /** Which backend serves this instance: "native" or "fallback". */
  get backend() {
    return this._backend
  }

  /** Free the native index handle (also freed automatically on GC). */
  destroy() {
    this._destroyNative()
  }

  _destroyNative() {
    if (this._nativeIndex) {
      this._nativeIndex.destroy()
      this._nativeIndex = null
    }
  }

  static createIndex() {
    throw new UnsupportedOptionError('Fuse.createIndex')
  }

  static parseIndex() {
    throw new UnsupportedOptionError('Fuse.parseIndex')
  }

  static nativeAvailable() {
    return nativeAvailable()
  }

  static backendInfo() {
    return backendInfo()
  }
}

FuseMojo.config = DEFAULT_CONFIG
FuseMojo.version = require('./package.json').version

module.exports = FuseMojo
module.exports.default = FuseMojo
module.exports.Fuse = FuseMojo
module.exports.FuseMojo = FuseMojo
module.exports.nativeAvailable = nativeAvailable
module.exports.backendInfo = backendInfo
module.exports.UnsupportedOptionError = UnsupportedOptionError
