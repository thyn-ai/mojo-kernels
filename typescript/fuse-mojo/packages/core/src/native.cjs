'use strict'

/**
 * koffi loader for the fusemojo native kernel, with an ABI-version handshake.
 *
 * Resolution order:
 *
 *   1. `$FUSE_MOJO_NATIVE_LIB` (explicit path override, for development; the
 *      pthread shim is expected next to it)
 *   2. the platform package (`@fuse-mojo/<platform>-<arch>`) installed as an
 *      optional dependency of `@fuse-mojo/core`
 *   3. the repository development build output `kernels/fuse/build/`
 *
 * The pthread shim (`libfusemojoshim`) is loaded from the same directory as
 * the kernel; without it, queries run sequentially on the calling thread
 * (same results, less speed).
 *
 * If the kernel cannot be found, fails to load, or reports an ABI version
 * this package does not understand, `NativeUnavailable` is thrown and the
 * caller falls back to the vendored Fuse.js implementation. Set
 * `FUSE_MOJO_DISABLE_NATIVE=1` to force that fallback (used by the
 * differential test suite and on platforms without a prebuilt library).
 * `FUSE_MOJO_THREADS=N` caps the native thread count (1 = sequential).
 *
 * Stable C ABI (v1)::
 *
 *   int32_t  fusemojo_abi_version(void)
 *   void*    fusemojo_index_create(const uint16_t* chars, const int32_t* offsets,
 *                                  int32_t n_texts, int32_t location, double distance,
 *                                  double threshold, int32_t min_match_char_length,
 *                                  int32_t find_all_matches, int32_t ignore_location,
 *                                  int32_t compute_matches, int32_t include_matches)
 *   void*    fusemojo_search_begin(void* handle, const uint16_t* pattern,
 *                                  int32_t pattern_len, int32_t location_offset,
 *                                  int32_t exact_check, double* out_scores,
 *                                  int32_t* out_is_match, int32_t* out_idx_offsets,
 *                                  int32_t n_jobs)
 *   int32_t  fusemojo_search_range(void* ctx, int32_t job_id, int32_t start, int32_t end)
 *   int32_t  fusemojo_search_end(void* ctx)
 *   void     fusemojo_copy_indices(void* handle, int32_t* out_pairs)
 *   void     fusemojo_index_destroy(void* handle)
 *   int32_t  fusemojo_search_parallel(void* ctx, int32_t n_texts, int32_t max_threads)  [shim]
 */

const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

// Must equal ABI_VERSION in kernels/fuse/src/fusemojo.mojo. A mismatch means
// the installed package and the resolved shared library disagree; fall back.
const ABI_VERSION = 1

const ENV_LIB = 'FUSE_MOJO_NATIVE_LIB'
const ENV_DISABLE = 'FUSE_MOJO_DISABLE_NATIVE'
const ENV_THREADS = 'FUSE_MOJO_THREADS'
const ENV_MIN_CHUNK = 'FUSE_MOJO_MIN_CHUNK'

class NativeUnavailable extends Error {
  constructor(message) {
    super(message)
    this.name = 'NativeUnavailable'
    this.code = 'FUSE_MOJO_NATIVE_UNAVAILABLE'
  }
}

function libBasename() {
  if (process.platform === 'darwin') return 'libfusemojo.dylib'
  if (process.platform === 'linux') return 'libfusemojo.so'
  if (process.platform === 'win32') return 'fusemojo.dll' // no Mojo toolchain builds this today
  return 'libfusemojo.so'
}

function shimBasename() {
  if (process.platform === 'darwin') return 'libfusemojoshim.dylib'
  if (process.platform === 'linux') return 'libfusemojoshim.so'
  if (process.platform === 'win32') return 'fusemojoshim.dll'
  return 'libfusemojoshim.so'
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
  const platformPkg = `@fuse-mojo/${process.platform}-${process.arch}`
  try {
    const pkgJson = require.resolve(`${platformPkg}/package.json`, { paths: [__dirname] })
    out.push([`platform package ${platformPkg}`, path.join(path.dirname(pkgJson), 'lib', libBasename())])
  } catch {
    // optional dependency not installed (unsupported platform or pruned)
  }
  out.push([
    'repo-dev build output',
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'kernels', 'fuse', 'build', libBasename()),
  ])
  return out
}

function bindAbi(lib) {
  // koffi parses C prototypes; TypedArrays passed to pointer parameters are
  // pinned for the duration of the call.
  return {
    abi_version: lib.func('int32_t fusemojo_abi_version(void)'),
    index_create: lib.func(`void* fusemojo_index_create(
      const uint16_t* chars, const int32_t* offsets, int32_t n_texts,
      int32_t location, double distance, double threshold,
      int32_t min_match_char_length, int32_t find_all_matches,
      int32_t ignore_location, int32_t compute_matches, int32_t include_matches)`),
    search_begin: lib.func(`void* fusemojo_search_begin(
      void* handle, const uint16_t* pattern, int32_t pattern_len,
      int32_t location_offset, int32_t exact_check,
      double* out_scores, int32_t* out_is_match, int32_t* out_idx_offsets,
      int32_t n_jobs)`),
    search_range: lib.func(
      'int32_t fusemojo_search_range(void* ctx, int32_t job_id, int32_t start, int32_t end)'
    ),
    search_end: lib.func('int32_t fusemojo_search_end(void* ctx)'),
    copy_indices: lib.func('void fusemojo_copy_indices(void* handle, int32_t* out_pairs)'),
    index_destroy: lib.func('void fusemojo_index_destroy(void* handle)'),
  }
}

let _koffi = null
let _lib = null
let _shim = null // null = not tried; false = unavailable; function = parallel runner
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

function threadCap() {
  const raw = process.env[ENV_THREADS]
  if (raw !== undefined) {
    const n = Number.parseInt(raw, 10)
    if (Number.isInteger(n) && n > 0) return n
  }
  const cpus = os.cpus().length || 1
  return Math.min(cpus, 64)
}

function minChunk() {
  const raw = process.env[ENV_MIN_CHUNK]
  if (raw !== undefined) {
    const n = Number.parseInt(raw, 10)
    if (Number.isInteger(n) && n > 0) return n
  }
  return 2048
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
        // The pthread shim is optional; without it queries run sequentially.
        _shim = false
        const shimPath = path.join(path.dirname(candidate), shimBasename())
        try {
          if (fs.existsSync(shimPath)) {
            const shimLib = koffi.load(shimPath)
            _shim = shimLib.func(
              'int32_t fusemojo_search_parallel(void* ctx, int32_t n_texts, int32_t max_threads, int32_t min_chunk)'
            )
          }
        } catch {
          _shim = false
        }
        _lib = bound
        _libSource = `${label} (${candidate})${_shim ? ' +pthread shim' : ''}`
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
    threads: 0,
    platform: process.platform,
    arch: process.arch,
    error: null,
  }
  try {
    const lib = load()
    info.native_available = true
    info.native_source = _libSource
    info.abi_version_native = lib.abi_version()
    info.threads = threadCap()
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
  /** Owned handle to a native text-collection index. */
  constructor(chars, offsets, options) {
    const lib = load() // throws NativeUnavailable
    const nTexts = offsets.length - 1
    const handle = lib.index_create(
      chars,
      offsets,
      nTexts,
      options.location,
      options.distance,
      options.threshold,
      options.minMatchCharLength,
      options.findAllMatches ? 1 : 0,
      options.ignoreLocation ? 1 : 0,
      options.computeMatches ? 1 : 0,
      options.includeMatches ? 1 : 0
    )
    if (handle === null || handle === undefined) {
      throw new NativeUnavailable(
        'native kernel rejected the index (invalid sizes); falling back to the vendored Fuse.js'
      )
    }
    // The kernel copies every buffer; the JS arrays may be GC'd.
    this._lib = lib
    this._handle = handle
    this.nTexts = nTexts
    this._threads = threadCap()
    this._minChunk = minChunk()
    // Reusable per-query buffers (single-threaded per instance).
    this._scores = new Float64Array(nTexts)
    this._isMatch = new Int32Array(nTexts)
    this._idxOffsets = new Int32Array(nTexts + 1)
    registry.register(this, handle, this)
  }

  /**
   * Score one pattern chunk (Uint16Array of length 1..32) against every text.
   * Returns { totalPairs, scores, isMatch, idxOffsets } views over the
   * instance's shared buffers (valid until the next searchChunk call).
   */
  searchChunk(patternU16, locationOffset, exactCheck) {
    if (this._handle === null) {
      throw new NativeUnavailable('native index is closed')
    }
    const ctx = this._lib.search_begin(
      this._handle,
      patternU16,
      patternU16.length,
      locationOffset,
      exactCheck ? 1 : 0,
      this._scores,
      this._isMatch,
      this._idxOffsets,
      this._threads
    )
    if (ctx === null || ctx === undefined) {
      throw new NativeUnavailable('native kernel rejected the pattern chunk')
    }
    let rc
    if (_shim) {
      rc = _shim(ctx, this.nTexts, this._threads, this._minChunk)
    } else {
      rc = this._lib.search_range(ctx, 0, 0, this.nTexts)
    }
    if (rc !== 0) {
      // Drain the context so buffers/stash stay consistent, then fail.
      this._lib.search_end(ctx)
      throw new NativeUnavailable(`native scoring failed with status ${rc}`)
    }
    const totalPairs = this._lib.search_end(ctx)
    if (totalPairs < 0) {
      throw new NativeUnavailable(`native scoring finalize failed with status ${totalPairs}`)
    }
    return {
      totalPairs,
      scores: this._scores,
      isMatch: this._isMatch,
      idxOffsets: this._idxOffsets,
    }
  }

  copyIndices(totalPairs) {
    const out = new Int32Array(totalPairs * 2)
    if (totalPairs > 0) {
      this._lib.copy_indices(this._handle, out)
    }
    return out
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
  ENV_THREADS,
  ENV_MIN_CHUNK,
}
