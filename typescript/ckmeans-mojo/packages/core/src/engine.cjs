'use strict'

/**
 * Orchestration shell for the native backend: replicates the observable
 * behavior of simple-statistics' ckmeans() around the native DP kernel —
 * argument coercion, the too-many-clusters error, the non-mutating numeric
 * sort, the single-unique-value fast path, and the reference's matrix
 * row-count semantics for non-integer k.
 *
 * Two native paths behind one behavior:
 *   - finite data (the common case): the kernel stable-sorts natively —
 *     bit-identical order to the reference's stable Array sort — and runs
 *     the DP in one call;
 *   - NaN-containing data: sorted in JS first so NaN ordering keeps the
 *     engine's own comparator semantics, then the sorted-entry kernel call.
 */

const { clusterNative, clusterNativeUnsorted } = require('./native.cjs')
const numericSort = require('../vendor/numeric_sort.cjs')
const uniqueCountSorted = require('../vendor/unique_count_sorted.cjs')

const ZERO_ROW_TYPE_ERROR = "Cannot read properties of undefined (reading 'length')"

/**
 * Number of rows the reference's makeMatrix(k, n) builds: one row per
 * non-negative integer strictly below k (so a non-integer k like 2.5
 * produces 3 rows, and k <= 0 or NaN produces none).
 */
function matrixRowCount(kNum) {
  return kNum > 0 ? Math.ceil(kNum) : 0
}

/**
 * ckmeans(x, nClusters) on the native backend. Throws NativeUnavailable
 * (only) when the kernel cannot serve the call; every other throw is a
 * reference-compatible validation error.
 */
function ckmeansNative(x, nClusters) {
  // Read x.length before coercing nClusters: the reference's first statement
  // is `nClusters > x.length`, so a missing x fails here (with the engine's
  // own TypeError) before any coercion of nClusters happens.
  const n = x.length
  // Number() matches the reference's implicit coercion of nClusters in its
  // `>` comparison and loop `<` (a Symbol throws the same TypeError here).
  const kNum = Number(nClusters)
  if (kNum > n) {
    throw new Error('cannot generate more classes than there are data values')
  }

  // One conversion pass; the scan doubles as the NaN gate. Non-number
  // elements become NaN here exactly where the reference's arithmetic would
  // produce NaN, and such input takes the JS-sort path below.
  const data = Float64Array.from(x)
  let hasNaN = false
  for (let i = 0; i < n; i += 1) {
    if (Number.isNaN(data[i])) {
      hasNaN = true
      break
    }
  }

  const k = matrixRowCount(kNum)

  if (hasNaN) {
    // JS-sort path: the reference's comparator semantics for NaN are the
    // engine's own, so sort exactly like the reference does.
    const sorted = numericSort(x)
    if (uniqueCountSorted(sorted) === 1) {
      return [sorted]
    }
    if (k === 0) {
      // The reference builds a zero-row matrix and reads matrix[0].length.
      throw new TypeError(ZERO_ROW_TYPE_ERROR)
    }
    const lefts = clusterNative(sorted, k) // throws NativeUnavailable
    const clusters = new Array(k)
    for (let c = k - 1, right = n - 1; c >= 0; c--) {
      const left = lefts[c]
      clusters[c] = sorted.slice(left, right + 1)
      if (c > 0) {
        right = left - 1
      }
    }
    return clusters
  }

  if (k === 0) {
    // Zero-row matrix: the single-unique-value fast path still applies.
    const sorted = numericSort(x)
    if (uniqueCountSorted(sorted) === 1) {
      return [sorted]
    }
    throw new TypeError(ZERO_ROW_TYPE_ERROR)
  }

  // Finite fast path: native stable sort + DP in one kernel call.
  const result = clusterNativeUnsorted(data, k) // throws NativeUnavailable
  if (result.singleUnique) {
    return [Array.from(result.sorted)]
  }
  const { lefts, sorted } = result
  const clusters = new Array(k)
  for (let c = k - 1, right = n - 1; c >= 0; c--) {
    const left = lefts[c]
    clusters[c] = Array.from(sorted.subarray(left, right + 1))
    if (c > 0) {
      right = left - 1
    }
  }
  return clusters
}

module.exports = { ckmeansNative, matrixRowCount }
