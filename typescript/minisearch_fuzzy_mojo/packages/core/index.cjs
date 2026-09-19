'use strict'

/**
 * @minisearch-mojo/core — drop-in faster replacement for MiniSearch v7
 * fuzzy/prefix search and auto-suggestions.
 *
 * Term resolution (bounded edit-distance DP and prefix scans over the
 * vocabulary) and BM25 accumulation run on a native Mojo kernel when its
 * shared library is available (macOS arm64 / Linux x64 platform packages);
 * on any load failure the package transparently falls back to the vendored
 * MiniSearch implementation (MIT, see vendor/ and NOTICE), so results are
 * identical on every platform.
 */

const {
  validateIndexOptions,
  validateSearchOptions,
  tokenizeAndProcess,
  UnsupportedOptionError,
} = require('./src/options.cjs')
const { Engine, EngineBuildError } = require('./src/engine.cjs')
const {
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
} = require('./src/native.cjs')
const VendorMiniSearch = require('./vendor/minisearch.umd.cjs')

class MiniSearchMojo {
  /**
   * @param {object} options MiniSearch-compatible options (see README for
   *   the supported subset; unsupported options throw UnsupportedOptionError)
   */
  constructor(options = {}) {
    validateIndexOptions(options)
    this.options = { ...options }
    this._defaultSearchOptions = options.searchOptions || {}
    this._fallback = null
    this._fallbackError = null
    this._engine = null
    this._docsSnapshot = []
    this._seenIds = new Set()
    try {
      this._engine = new Engine(this.options)
      // Eagerly prove the kernel loads (the index itself builds lazily on
      // first query); anything else engages the vendored fallback.
      if (!nativeAvailable()) {
        throw new NativeUnavailable(backendInfo().error || 'native kernel not loadable')
      }
      this._backend = 'native'
    } catch (err) {
      if (!(err instanceof NativeUnavailable)) {
        throw err
      }
      this._engine = null
      this._fallback = new VendorMiniSearch(this.options)
      this._backend = 'fallback'
      this._fallbackError = err.message
    }
  }

  /** Merge per-call search options over the constructor's searchOptions. */
  _searchOptions(options) {
    const merged = { ...this._defaultSearchOptions, ...(options || {}) }
    validateSearchOptions(merged)
    return merged
  }

  /** Query validation shared by both backends (term-count limits are a
   *  wrapper-level constraint, also enforced on the fallback). */
  _checkQueryTerms(query) {
    const raw = tokenizeAndProcess(query)
    if (raw.length > 64) {
      throw new UnsupportedOptionError('more than 64 query terms')
    }
    if (new Set(raw).size > 32) {
      throw new UnsupportedOptionError('more than 32 distinct query terms')
    }
  }

  _ensureFallback() {
    if (this._backend === 'fallback') {
      return
    }
    if (this._engine) {
      this._engine.destroy()
    }
    this._engine = null
    this._fallback = new VendorMiniSearch(this.options)
    this._fallback.addAll(this._docsSnapshot)
    this._backend = 'fallback'
  }

  add(doc) {
    if (doc === null || typeof doc !== 'object') {
      throw new EngineBuildError('document must be an object')
    }
    const id = doc[this.options.idField || 'id']
    if (id === undefined || id === null) {
      throw new EngineBuildError(`document is missing the id field "${this.options.idField || 'id'}"`)
    }
    if (this._seenIds.has(id)) {
      throw new EngineBuildError(`duplicate document id: ${String(id)}`)
    }
    this._seenIds.add(id)
    if (this._backend === 'fallback') {
      return this._fallback.add(doc)
    }
    this._docsSnapshot.push(doc)
    try {
      this._engine.add(doc)
    } catch (err) {
      if (!(err instanceof NativeUnavailable)) {
        throw err
      }
      this._ensureFallback()
    }
    return this
  }

  addAll(docs) {
    for (const doc of docs) {
      this.add(doc)
    }
    return this
  }

  /**
   * @param {string} query
   * @param {object} options search options (prefix/fuzzy/boost/fields/combineWith)
   * @returns {Array<{id: any, score: number, terms: string[], queryTerms: string[], match: object}>}
   */
  search(query, options) {
    if (typeof query !== 'string') {
      throw new UnsupportedOptionError(`non-string query (${typeof query})`)
    }
    const opts = this._searchOptions(options)
    this._checkQueryTerms(query)
    if (this._backend === 'fallback') {
      return this._fallback.search(query, opts)
    }
    try {
      const nativeIndex = this._engine._buildNative()
      return this._engine.search(nativeIndex, query, opts)
    } catch (err) {
      if (!(err instanceof NativeUnavailable)) {
        throw err
      }
      this._ensureFallback()
      return this._fallback.search(query, opts)
    }
  }

  /**
   * @param {string} query
   * @param {object} options same options as search(); by default the last
   *   query term is prefix-expanded and suggestions require all query terms
   *   (combineWith 'AND')
   * @returns {Array<{suggestion: string, terms: string[], score: number}>}
   */
  autoSuggest(query, options) {
    if (typeof query !== 'string') {
      throw new UnsupportedOptionError(`non-string query (${typeof query})`)
    }
    const opts = this._searchOptions(options)
    this._checkQueryTerms(query)
    if (this._backend === 'fallback') {
      return this._fallback.autoSuggest(query, opts)
    }
    try {
      const nativeIndex = this._engine._buildNative()
      return this._engine.autoSuggest(nativeIndex, query, opts)
    } catch (err) {
      if (!(err instanceof NativeUnavailable)) {
        throw err
      }
      this._ensureFallback()
      return this._fallback.autoSuggest(query, opts)
    }
  }

  /** Which backend serves this instance: "native" or "fallback". */
  get backend() {
    return this._backend
  }

  /** Free the native index handle (also freed automatically on GC). */
  destroy() {
    if (this._engine) {
      this._engine.destroy()
    }
  }

  static nativeAvailable() {
    return nativeAvailable()
  }

  static backendInfo() {
    return backendInfo()
  }
}

MiniSearchMojo.version = require('./package.json').version

module.exports = MiniSearchMojo
module.exports.default = MiniSearchMojo
module.exports.MiniSearch = MiniSearchMojo
module.exports.MiniSearchMojo = MiniSearchMojo
module.exports.nativeAvailable = nativeAvailable
module.exports.backendInfo = backendInfo
module.exports.UnsupportedOptionError = UnsupportedOptionError
