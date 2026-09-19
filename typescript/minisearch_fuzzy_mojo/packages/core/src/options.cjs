'use strict'

/**
 * Option handling, tokenization, and term-weight rules for
 * @minisearch-mojo/core.
 *
 * These functions replicate the *observable* behavior of the corresponding
 * MiniSearch v7 code paths (default tokenizer splitting on Unicode
 * space/punctuation, default term lowercasing, fuzzy edit-budget rule
 * `Math.round(fuzzy * term.length)` for fuzzy < 1 / absolute budget for
 * fuzzy >= 1, and the fuzzy/prefix match weight curves) so that results are
 * identical on both backends. Written clean-room against the published
 * documentation and black-box observed behavior of the reference package.
 */

/** Default tokenizer: split on Unicode space / punctuation / CR / LF. */
const SPLIT_RE = /[\p{Z}\p{P}\n\r]+/u

/**
 * Raw tokenization: split on the separator runs. Leading/trailing runs leave
 * empty edge tokens, which the reference counts in document field lengths
 * (the empty term has no postings and never matches queries).
 */
function tokenizeRaw(value) {
  return String(value).split(SPLIT_RE)
}

function tokenizeString(value) {
  const parts = String(value).split(SPLIT_RE)
  const out = []
  for (const part of parts) {
    if (part.length > 0) {
      out.push(part)
    }
  }
  return out
}

/** Default term processing: lowercase; empty terms are discarded. */
function processTermDefault(term) {
  return term.toLowerCase()
}

function tokenizeAndProcess(value) {
  const raw = tokenizeString(value)
  const out = []
  for (const part of raw) {
    const term = processTermDefault(part)
    if (term.length > 0) {
      out.push(term)
    }
  }
  return out
}

class UnsupportedOptionError extends Error {
  constructor(what) {
    super(
      `minisearch-mojo: unsupported option or input: ${what}. ` +
        'Supported index options are: fields, storeFields, idField, searchOptions. ' +
        'Supported search options are: prefix, fuzzy, boost, fields, combineWith. ' +
        'Custom tokenize/processTerm/extractField functions, the filter option, ' +
        'remove/discard/vacuum, and external serialization are not supported.'
    )
    this.name = 'UnsupportedOptionError'
    this.code = 'MINISEARCH_MOJO_UNSUPPORTED_OPTION'
  }
}

const INDEX_OPTION_KEYS = ['fields', 'storeFields', 'idField', 'searchOptions']
const SEARCH_OPTION_KEYS = ['prefix', 'fuzzy', 'boost', 'fields', 'combineWith']

function validateIndexOptions(options) {
  for (const key of Object.keys(options)) {
    if (!INDEX_OPTION_KEYS.includes(key)) {
      throw new UnsupportedOptionError(`index option ${key}`)
    }
  }
  if (!Array.isArray(options.fields) || options.fields.length === 0) {
    throw new UnsupportedOptionError('index option fields must be a non-empty array')
  }
  if (options.fields.length > 31) {
    throw new UnsupportedOptionError('more than 31 fields (match field bitmask is 32-bit)')
  }
  for (const f of options.fields) {
    if (typeof f !== 'string') throw new UnsupportedOptionError('non-string field name')
  }
  if (options.storeFields !== undefined && !Array.isArray(options.storeFields)) {
    throw new UnsupportedOptionError('storeFields must be an array')
  }
  if (options.searchOptions !== undefined) {
    if (typeof options.searchOptions !== 'object' || options.searchOptions === null) {
      throw new UnsupportedOptionError('searchOptions must be an object')
    }
    validateSearchOptions(options.searchOptions)
  }
}

function validateSearchOptions(options) {
  for (const key of Object.keys(options)) {
    if (!SEARCH_OPTION_KEYS.includes(key)) {
      throw new UnsupportedOptionError(`search option ${key}`)
    }
  }
  if (
    options.combineWith !== undefined &&
    options.combineWith !== 'AND' &&
    options.combineWith !== 'OR'
  ) {
    throw new UnsupportedOptionError(`combineWith ${options.combineWith}`)
  }
  if (options.boost !== undefined) {
    if (typeof options.boost !== 'object' || options.boost === null) {
      throw new UnsupportedOptionError('boost must be an object mapping field to number')
    }
    for (const [field, value] of Object.entries(options.boost)) {
      if (typeof value !== 'number' || Number.isNaN(value)) {
        throw new UnsupportedOptionError(`boost.${field} must be a number`)
      }
    }
  }
  if (options.fields !== undefined && !Array.isArray(options.fields)) {
    throw new UnsupportedOptionError('search option fields must be an array')
  }
  for (const [name, spec] of [
    ['fuzzy', options.fuzzy],
    ['prefix', options.prefix],
  ]) {
    if (
      spec !== undefined &&
      typeof spec !== 'boolean' &&
      typeof spec !== 'number' &&
      typeof spec !== 'function'
    ) {
      throw new UnsupportedOptionError(`${name} must be a boolean, number, or function`)
    }
  }
}

/**
 * Fuzzy edit budget for one query term: fuzzy < 1 means
 * min(Math.round(fuzzy * term.length), 6) — the relative budget is capped
 * at 6 edits, as observed from the reference (a 13-char term at fuzzy 0.5
 * allows 6 edits, not 7); fuzzy >= 1 is an absolute budget with no cap;
 * fuzzy === true is 0.2. Functions are evaluated on the (processed) term and
 * may return any of these. Returns -1 when fuzzy matching is disabled.
 */
function resolveMaxEdits(fuzzy, term) {
  if (fuzzy === undefined || fuzzy === null || fuzzy === false) {
    return -1
  }
  if (typeof fuzzy === 'function') {
    return resolveMaxEdits(fuzzy(term), term)
  }
  if (fuzzy === true) {
    return Math.min(Math.round(0.2 * term.length), 6)
  }
  if (typeof fuzzy === 'number') {
    if (Number.isNaN(fuzzy) || fuzzy < 0) {
      throw new UnsupportedOptionError(`fuzzy ${fuzzy}`)
    }
    if (fuzzy < 1) {
      return Math.min(Math.round(fuzzy * term.length), 6)
    }
    return Math.floor(fuzzy)
  }
  throw new UnsupportedOptionError(`fuzzy of type ${typeof fuzzy}`)
}

/** Whether prefix matching applies to one query term (functions evaluated). */
function resolvePrefix(prefix, term) {
  if (prefix === undefined || prefix === null) {
    return false
  }
  if (typeof prefix === 'function') {
    return Boolean(prefix(term))
  }
  return Boolean(prefix)
}

/** Fuzzy match weight: 9*len / (20 * (len + distance)), len = variant length. */
function fuzzyWeight(termLen, distance) {
  return (9 * termLen) / (20 * (termLen + distance))
}

/** Prefix match weight: 15*len / (52*len - 12*qlen), len = variant length. */
function prefixWeight(queryLen, termLen) {
  return (15 * termLen) / (52 * termLen - 12 * queryLen)
}

/** 32-bit population count. */
function popcount32(x) {
  x = x | 0
  x = x - ((x >>> 1) & 0x55555555)
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
  x = (x + (x >>> 4)) & 0x0f0f0f0f
  return (x * 0x01010101) >>> 24
}

function toU16(str) {
  const out = new Uint16Array(str.length)
  for (let i = 0; i < str.length; i += 1) {
    out[i] = str.charCodeAt(i)
  }
  return out
}

module.exports = {
  tokenizeRaw,
  tokenizeString,
  tokenizeAndProcess,
  processTermDefault,
  UnsupportedOptionError,
  validateIndexOptions,
  validateSearchOptions,
  resolveMaxEdits,
  resolvePrefix,
  fuzzyWeight,
  prefixWeight,
  popcount32,
  toU16,
}
