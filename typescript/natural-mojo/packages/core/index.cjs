'use strict'

/**
 * @natural-mojo/core — drop-in faster replacement for natural's
 * LevenshteinDistance and DamerauLevenshteinDistance.
 *
 * The dynamic program runs on a native Mojo kernel when its shared library
 * is available (macOS arm64 / Linux x64 platform packages); on any load
 * failure the package transparently falls back to the vendored natural
 * implementation (MIT, see vendor/ and NOTICE), so results are identical on
 * every platform.
 *
 * Option semantics replicate natural v8.1.1 exactly:
 *   - insertion_cost / deletion_cost / substitution_cost are defaulted to 1
 *     with the reference's `isNaN` coercion (undefined, NaN, non-numeric → 1;
 *     null → 0, true → 1)
 *   - transposition_cost (Damerau only) defaults to 1 only when the key is
 *     absent; an explicit undefined/NaN value disables transpositions via the
 *     reference's NaN-min behavior
 *   - restricted: true selects the OSA (adjacent-transposition) variant; the
 *     default false is the unrestricted Lowrance-Wagner Damerau-Levenshtein
 *   - damerau / search keys in user options are overridden per function,
 *     exactly like the reference
 */

const {
  NativeUnavailable,
  nativeAvailable,
  backendInfo,
  distanceNative,
} = require('./src/native.cjs')
const VendorNatural = require('./vendor/natural_distance.cjs')

class UnsupportedOptionError extends Error {
  constructor(feature) {
    super(
      `natural-mojo does not support ${feature}. Supported functions are: ` +
        `LevenshteinDistance, DamerauLevenshteinDistance.`
    )
    this.name = 'UnsupportedOptionError'
    this.code = 'NATURAL_MOJO_UNSUPPORTED_OPTION'
  }
}

function toU16(str) {
  const out = new Uint16Array(str.length)
  for (let i = 0; i < str.length; i += 1) {
    out[i] = str.charCodeAt(i)
  }
  return out
}

function assertStrings(source, target) {
  if (typeof source !== 'string' || typeof target !== 'string') {
    throw new TypeError(
      `distance functions require string arguments, got ${typeof source} and ${typeof target}`
    )
  }
}

// The reference defaults a cost to 1 when `isNaN(cost)` (which coerces with
// Number()): undefined/NaN/'abc' → 1, null → 0, true → 1.
function defaultedCost(raw) {
  const v = Number(raw)
  return Number.isNaN(v) ? 1 : v
}

function nativeLevenshtein(source, target, options) {
  // Reference: _.extend({}, options || {}, {damerau: false, search: false}).
  const opts = options || {}
  const costs = {
    insertion_cost: defaultedCost(opts.insertion_cost),
    deletion_cost: defaultedCost(opts.deletion_cost),
    substitution_cost: defaultedCost(opts.substitution_cost),
    transposition_cost: 1, // unused: damerau is false
  }
  return distanceNative(toU16(source), toU16(target), costs, false, false)
}

function nativeDamerau(source, target, options) {
  // Reference: _.extend({transposition_cost: 1, restricted: false},
  //                     options || {}, {damerau: true, search: false}).
  const merged = Object.assign({ transposition_cost: 1, restricted: false }, options || {})
  const costs = {
    insertion_cost: defaultedCost(merged.insertion_cost),
    deletion_cost: defaultedCost(merged.deletion_cost),
    substitution_cost: defaultedCost(merged.substitution_cost),
    // Not isNaN-defaulted by the reference: an explicit undefined/NaN value
    // disables transpositions through the reference's NaN-min behavior.
    transposition_cost: Number(merged.transposition_cost),
  }
  return distanceNative(toU16(source), toU16(target), costs, true, !!merged.restricted)
}

/**
 * Levenshtein distance between two strings (insertions, deletions,
 * substitutions). Drop-in for natural.LevenshteinDistance.
 *
 * @param {string} source
 * @param {string} target
 * @param {object} [options] {insertion_cost, deletion_cost, substitution_cost}
 * @returns {number} integer when all costs are integers
 */
function LevenshteinDistance(source, target, options) {
  assertStrings(source, target)
  try {
    return nativeLevenshtein(source, target, options)
  } catch (err) {
    if (!(err instanceof NativeUnavailable)) {
      throw err
    }
    return VendorNatural.LevenshteinDistance(source, target, options)
  }
}

/**
 * Damerau-Levenshtein distance between two strings (Levenshtein plus
 * transpositions). Drop-in for natural.DamerauLevenshteinDistance.
 *
 * @param {string} source
 * @param {string} target
 * @param {object} [options] {insertion_cost, deletion_cost, substitution_cost,
 *   transposition_cost, restricted} — restricted: true is the OSA variant,
 *   the default false is the unrestricted Lowrance-Wagner variant.
 * @returns {number} integer when all costs are integers
 */
function DamerauLevenshteinDistance(source, target, options) {
  assertStrings(source, target)
  try {
    return nativeDamerau(source, target, options)
  } catch (err) {
    if (!(err instanceof NativeUnavailable)) {
      throw err
    }
    return VendorNatural.DamerauLevenshteinDistance(source, target, options)
  }
}

function LevenshteinDistanceSearch() {
  throw new UnsupportedOptionError('LevenshteinDistanceSearch (substring search)')
}

function DamerauLevenshteinDistanceSearch() {
  throw new UnsupportedOptionError('DamerauLevenshteinDistanceSearch (substring search)')
}

module.exports = {
  LevenshteinDistance,
  DamerauLevenshteinDistance,
  LevenshteinDistanceSearch,
  DamerauLevenshteinDistanceSearch,
  nativeAvailable,
  backendInfo,
  UnsupportedOptionError,
  version: require('./package.json').version,
}
module.exports.default = module.exports
