'use strict'

/**
 * Option handling for @fuse-mojo/core: defaults, validation, key store,
 * property-path getter, and the field-length norm.
 *
 * These functions replicate the *observable* behavior of the corresponding
 * Fuse.js v7 configuration code paths (default values, key weight
 * normalization, dotted-path value extraction, `1/sqrt(tokens)` field norm
 * rounded to 3 decimals) so that results are identical on both backends.
 * Written clean-room against the published documentation and observed
 * behavior of the reference package.
 */

const SPACE_RUN = /[^ ]+/g

/** Default options, identical to Fuse.js v7. */
const DEFAULT_CONFIG = {
  isCaseSensitive: false,
  includeMatches: false,
  findAllMatches: false,
  minMatchCharLength: 1,
  includeScore: false,
  keys: [],
  shouldSort: true,
  sortFn: defaultSortFn,
  location: 0,
  threshold: 0.6,
  distance: 100,
  useExtendedSearch: false,
  getFn: fuseGet,
  ignoreLocation: false,
  ignoreFieldNorm: false,
  fieldNormWeight: 1,
  ignoreDiacritics: false,
}

function defaultSortFn(a, b) {
  return a.score === b.score ? (a.idx < b.idx ? -1 : 1) : a.score < b.score ? -1 : 1
}

const SUPPORTED_OPTIONS = [
  'keys',
  'threshold',
  'location',
  'distance',
  'minMatchCharLength',
  'includeScore',
  'includeMatches',
  'shouldSort',
  'ignoreLocation',
  'isCaseSensitive',
  'findAllMatches',
  'ignoreFieldNorm',
  'fieldNormWeight',
]

class UnsupportedOptionError extends Error {
  constructor(what) {
    super(
      `fuse-mojo: unsupported option or input: ${what}. ` +
        `Supported options are: ${SUPPORTED_OPTIONS.join(', ')} ` +
        '(plus the `limit` search parameter). Extended search ' +
        '(`useExtendedSearch`, logical `$and`/`$or` queries), ' +
        '`ignoreDiacritics`, custom `getFn`, custom `sortFn`, and external ' +
        'indices (`Fuse.createIndex`/`parseIndex`) are not supported.'
    )
    this.name = 'UnsupportedOptionError'
    this.code = 'FUSE_MOJO_UNSUPPORTED_OPTION'
  }
}

function validateOptions(options) {
  if (options.useExtendedSearch) {
    throw new UnsupportedOptionError('useExtendedSearch')
  }
  if (options.ignoreDiacritics) {
    throw new UnsupportedOptionError('ignoreDiacritics')
  }
  if (options.getFn !== undefined && options.getFn !== fuseGet) {
    throw new UnsupportedOptionError('custom getFn')
  }
  if (options.sortFn !== undefined && options.sortFn !== defaultSortFn) {
    throw new UnsupportedOptionError('custom sortFn')
  }
  if (!Array.isArray(options.keys)) {
    throw new UnsupportedOptionError('keys must be an array of strings or {name, weight} objects')
  }
  if (!Number.isFinite(options.threshold)) {
    throw new UnsupportedOptionError('non-finite threshold')
  }
  if (typeof options.distance !== 'number' || Number.isNaN(options.distance)) {
    throw new UnsupportedOptionError('non-numeric distance')
  }
  if (!Number.isInteger(options.location)) {
    throw new UnsupportedOptionError('non-integer location')
  }
  if (!Number.isInteger(options.minMatchCharLength)) {
    throw new UnsupportedOptionError('non-integer minMatchCharLength')
  }
  for (const key of options.keys) {
    if (typeof key === 'object' && key !== null && !Array.isArray(key) && key.getFn) {
      throw new UnsupportedOptionError('key getFn')
    }
  }
}

/**
 * Key store for standard (non-logical) search. Weights are the RAW weights
 * from the key specifications: the reference only normalizes weights (to sum
 * to 1) inside the KeyStore used by logical $and/$or queries; the regular
 * object-list search path reads the un-normalized weights from the index's
 * key entries. Logical queries are unsupported, so no normalization happens
 * here either.
 */
function createKeyStore(keys) {
  const store = []
  for (const key of keys) {
    let path
    let id
    let src
    let weight = 1
    if (typeof key === 'string' || Array.isArray(key)) {
      src = key
      path = Array.isArray(key) ? key : key.split('.')
      id = Array.isArray(key) ? key.join('.') : key
    } else if (typeof key === 'object' && key !== null) {
      if (!Object.prototype.hasOwnProperty.call(key, 'name')) {
        throw new Error('Missing "name" property in key object')
      }
      const name = key.name
      src = name
      if (Object.prototype.hasOwnProperty.call(key, 'weight')) {
        weight = key.weight
        if (weight <= 0) {
          throw new Error(`Invalid "weight" property in key object: ${name}`)
        }
      }
      path = typeof name === 'string' ? name.split('.') : name
      id = Array.isArray(name) ? name.join('.') : name
    } else {
      throw new UnsupportedOptionError(`key of type ${typeof key}`)
    }
    store.push({ path, id, src, weight })
  }
  return store
}

const INFINITY = 1 / 0

function baseToString(value) {
  if (typeof value === 'string') {
    return value
  }
  const result = value + ''
  return result === '0' && 1 / value === -INFINITY ? '-0' : result
}

function toString(value) {
  return value == null ? '' : baseToString(value)
}

/**
 * Dotted-path property getter with array descent. Behavioral replica of the
 * reference default getFn: string/number/boolean leaves are stringified,
 * arrays are flattened depth-first in ascending index order, and the result
 * is the single value, or the list of values when any array was traversed.
 */
function fuseGet(obj, path) {
  const list = []
  let arr = false

  const deepGet = (obj, path, index) => {
    if (obj === undefined || obj === null) {
      return
    }
    if (!path[index]) {
      // No path left (or an empty path segment): the object itself.
      list.push(obj)
      return
    }
    const key = path[index]
    const value = obj[key]
    if (value === undefined || value === null) {
      return
    }
    if (
      index === path.length - 1 &&
      (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')
    ) {
      list.push(toString(value))
    } else if (Array.isArray(value)) {
      arr = true
      for (let i = 0, len = value.length; i < len; i += 1) {
        deepGet(value[i], path, index + 1)
      }
    } else if (path.length) {
      deepGet(value, path, index + 1)
    }
  }

  deepGet(obj, typeof path === 'string' ? path.split('.') : path, 0)
  return arr ? list : list[0]
}

function isBlank(value) {
  return !value.trim().length
}

/**
 * Field-length norm: 1 / numTokens^(0.5 * weight), rounded to 3 decimals,
 * cached by token count — identical to the reference norm generator.
 */
function createNormGetter(weight, mantissa = 3) {
  const cache = new Map()
  const m = Math.pow(10, mantissa)
  return (value) => {
    const numTokens = value.match(SPACE_RUN).length
    if (cache.has(numTokens)) {
      return cache.get(numTokens)
    }
    const norm = 1 / Math.pow(numTokens, 0.5 * weight)
    const n = parseFloat(Math.round(norm * m) / m)
    cache.set(numTokens, n)
    return n
  }
}

module.exports = {
  DEFAULT_CONFIG,
  SUPPORTED_OPTIONS,
  UnsupportedOptionError,
  validateOptions,
  createKeyStore,
  fuseGet,
  isBlank,
  createNormGetter,
  defaultSortFn,
}
