'use strict'

/**
 * koffi loader for the msmojo native kernel, with an ABI-version handshake.
 *
 * Resolution order:
 *
 *   1. `$MINISEARCH_MOJO_NATIVE_LIB` (explicit path override, for development)
 *   2. the platform package (`@minisearch-mojo/<platform>-<arch>`) installed
 *      as an optional dependency of `@minisearch-mojo/core`
 *   3. the repository development build output `kernels/minisearch_fuzzy/build/`
 *
 * If the kernel cannot be found, fails to load, or reports an ABI version
 * this package does not understand, `NativeUnavailable` is thrown and the
 * caller falls back to the vendored MiniSearch implementation. Set
 * `MINISEARCH_MOJO_DISABLE_NATIVE=1` to force that fallback (used by the
 * differential test suite and on platforms without a prebuilt library).
 *
 * Stable C ABI (v1):
 *
 *   int32_t  msmojo_abi_version(void)
 *   void*    msmojo_index_create(const uint16_t* vocab_chars, const int32_t* vocab_offsets,
 *                                int32_t n_terms, const int32_t* term_df,
 *                                const int32_t* post_offsets, const int32_t* post_doc,
 *                                const int32_t* post_field, const int32_t* post_tf,
 *                                const int32_t* docfield_len, const double* avgdl,
 *                                int32_t n_docs, int32_t n_fields, int32_t doc_count)
 *   void     msmojo_index_destroy(void* handle)
 *   int32_t  msmojo_fuzzy_scan(void* handle, const uint16_t* query, int32_t qlen,
 *                              int32_t max_edits, int32_t* out_ids, int32_t* out_dist,
 *                              int32_t cap)
 *   int32_t  msmojo_prefix_scan(void* handle, const uint16_t* query, int32_t qlen,
 *                               int32_t* out_ids, int32_t cap)
 *   int32_t  msmojo_score(void* handle, const int32_t* qv_offsets, const int32_t* qv_ids,
 *                         const double* qv_weight, const int32_t* qv_qt, int32_t n_raw_q,
 *                         const double* field_boost, int32_t allowed_fields_mask,
 *                         double* out_scores, int32_t* out_mask, int32_t* out_records,
 *                         int32_t record_cap)
 */

const fs = require('node:fs')
const path = require('node:path')

// Must equal ABI_VERSION in kernels/minisearch_fuzzy/src/msmojo.mojo. A
// mismatch means the installed package and the resolved shared library
// disagree; fall back.
const ABI_VERSION = 1

const ENV_LIB = 'MINISEARCH_MOJO_NATIVE_LIB'
const ENV_DISABLE = 'MINISEARCH_MOJO_DISABLE_NATIVE'

class NativeUnavailable extends Error {
  constructor(message) {
    super(message)
    this.name = 'NativeUnavailable'
    this.code = 'MINISEARCH_MOJO_NATIVE_UNAVAILABLE'
  }
}

function libBasename() {
  if (process.platform === 'darwin') return 'libmsmojo.dylib'
  if (process.platform === 'linux') return 'libmsmojo.so'
  if (process.platform === 'win32') return 'msmojo.dll' // no Mojo toolchain builds this today
  return 'libmsmojo.so'
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
  const platformPkg = `@minisearch-mojo/${process.platform}-${process.arch}`
  try {
    const pkgJson = require.resolve(`${platformPkg}/package.json`, { paths: [__dirname] })
    out.push([`platform package ${platformPkg}`, path.join(path.dirname(pkgJson), 'lib', libBasename())])
  } catch {
    // optional dependency not installed (unsupported platform or pruned)
  }
  out.push([
    'repo-dev build output',
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'kernels', 'minisearch_fuzzy', 'build', libBasename()),
  ])
  return out
}

function bindAbi(lib) {
  // koffi parses C prototypes; TypedArrays passed to pointer parameters are
  // pinned for the duration of the call.
  return {
    abi_version: lib.func('int32_t msmojo_abi_version(void)'),
    index_create: lib.func(`void* msmojo_index_create(
      const uint16_t* vocab_chars, const int32_t* vocab_offsets, int32_t n_terms,
      const int32_t* post_offsets, const int32_t* post_doc,
      const int32_t* post_field, const int32_t* post_tf,
      const int32_t* docfield_len, const double* avgdl,
      int32_t n_docs, int32_t n_fields)`),
    index_destroy: lib.func('void msmojo_index_destroy(void* handle)'),
    fuzzy_scan: lib.func(`int32_t msmojo_fuzzy_scan(
      void* handle, const uint16_t* query, int32_t qlen, int32_t max_edits,
      int32_t* out_ids, int32_t* out_dist, int32_t cap)`),
    prefix_scan: lib.func(`int32_t msmojo_prefix_scan(
      void* handle, const uint16_t* query, int32_t qlen,
      int32_t* out_ids, int32_t cap)`),
    score: lib.func(`int32_t msmojo_score(
      void* handle, const int32_t* qv_offsets, const int32_t* qv_ids,
      const double* qv_weight, const double* qv_idf, const int32_t* qv_qt, int32_t n_raw_q,
      const double* field_boost, int32_t allowed_fields_mask,
      double* out_scores, int32_t* out_mask, int32_t* out_records, int32_t record_cap)`),
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
    }
  }
  _loadError = errors.join('; ') || 'no native kernel found on any resolver path'
  throw new NativeUnavailable(_loadError)
}

function nativeAvailable() {
  /** True if the native kernel can score right now. Never throws. */
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

const registry = new FinalizationRegistry((handle) => {
  try {
    if (_lib) _lib.index_destroy(handle)
  } catch {
    // best-effort cleanup during GC; never throw
  }
})

class NativeIndex {
  /** Owned handle to a native inverted index, plus per-query scratch buffers. */
  constructor(buffers) {
    const lib = load() // throws NativeUnavailable
    const handle = lib.index_create(
      buffers.vocabChars,
      buffers.vocabOffsets,
      buffers.nTerms,
      buffers.postOffsets,
      buffers.postDoc,
      buffers.postField,
      buffers.postTf,
      buffers.docfieldLen,
      buffers.avgdl,
      buffers.nDocs,
      buffers.nFields
    )
    if (handle === null || handle === undefined) {
      throw new NativeUnavailable(
        'native kernel rejected the index (invalid sizes); falling back to the vendored MiniSearch'
      )
    }
    // The kernel copies every buffer; the JS arrays may be GC'd.
    this._lib = lib
    this._handle = handle
    this.nTerms = buffers.nTerms
    this.nDocs = buffers.nDocs
    // Per-query scratch (single-threaded per instance).
    this._scanIds = new Int32Array(Math.max(buffers.nTerms, 1))
    this._scanDist = new Int32Array(Math.max(buffers.nTerms, 1))
    this._scores = new Float64Array(Math.max(buffers.nDocs, 1))
    this._masks = new Int32Array(Math.max(buffers.nDocs, 1))
    this._records = new Int32Array(16)
    registry.register(this, handle, this)
  }

  /** Bounded-Levenshtein scan over the vocabulary, ascending term ids. */
  fuzzyScan(queryU16, maxEdits) {
    if (this._handle === null) throw new NativeUnavailable('native index is closed')
    const n = this._lib.fuzzy_scan(
      this._handle,
      queryU16,
      queryU16.length,
      maxEdits,
      this._scanIds,
      this._scanDist,
      this.nTerms
    )
    if (n < 0) throw new NativeUnavailable(`native fuzzy scan failed with status ${n}`)
    return { count: n, ids: this._scanIds, dist: this._scanDist }
  }

  /** Prefix scan over the vocabulary, descending term ids (reverse insertion). */
  prefixScan(queryU16) {
    if (this._handle === null) throw new NativeUnavailable('native index is closed')
    const n = this._lib.prefix_scan(
      this._handle,
      queryU16,
      queryU16.length,
      this._scanIds,
      this.nTerms
    )
    if (n < 0) throw new NativeUnavailable(`native prefix scan failed with status ${n}`)
    return { count: n, ids: this._scanIds }
  }

  /**
   * Accumulate one query's scores. recordCap must upper-bound the emitted
   * records (sum of df over all variants works). Returns
   * { scores, masks, records, nRecords } views over shared buffers.
   */
  score(plan) {
    if (this._handle === null) throw new NativeUnavailable('native index is closed')
    if (plan.recordCap * 4 > this._records.length) {
      this._records = new Int32Array(plan.recordCap * 4)
    }
    const n = this._lib.score(
      this._handle,
      plan.qvOffsets,
      plan.qvIds,
      plan.qvWeight,
      plan.qvIdf,
      plan.qvQt,
      plan.nRawQ,
      plan.fieldBoost,
      plan.allowedFieldsMask,
      this._scores,
      this._masks,
      this._records,
      plan.recordCap
    )
    if (n < 0) throw new NativeUnavailable(`native scoring failed with status ${n}`)
    return { scores: this._scores, masks: this._masks, records: this._records, nRecords: n }
  }

  destroy() {
    const handle = this._handle
    this._handle = null
    if (handle) {
      registry.unregister(this)
      this._lib.index_destroy(handle)
    }
  }
}

module.exports = {
  NativeIndex,
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
  ABI_VERSION,
  ENV_LIB,
  ENV_DISABLE,
}
