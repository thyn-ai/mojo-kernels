'use strict'

/**
 * Deterministic seeded dataset generators for the differential suite. Same
 * seeds on every run (and every backend) produce identical datasets, so any
 * mismatch is a real behavioral difference.
 */

/** mulberry32: tiny deterministic PRNG. */
function rng(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const KINDS = ['int-grid', 'float', 'half-grid', 'mixture', 'big-offset', 'negative']

/**
 * One seeded dataset. Kinds:
 *   int-grid:   small integers — exact DP ties everywhere
 *   float:      uniform floats in [0, 100) — ties measure-zero
 *   half-grid:  multiples of 0.5 — exact in binary, engineered ties
 *   mixture:    well-separated gaussian-ish blobs (realistic clustering)
 *   big-offset: integers around 1e10 — exercises the median shift
 *   negative:   floats in [-50, 50)
 *   duplicates: heavy duplication drawn from {0,1,2}
 */
function makeDataset(seed, n, kind) {
  const rand = rng(seed)
  const out = new Array(n)
  for (let i = 0; i < n; i += 1) {
    if (kind === 'int-grid') {
      out[i] = Math.floor(rand() * 5)
    } else if (kind === 'float') {
      out[i] = rand() * 100
    } else if (kind === 'half-grid') {
      out[i] = Math.floor(rand() * 20) / 2
    } else if (kind === 'mixture') {
      const blob = Math.floor(rand() * 4)
      out[i] = blob * 25 + (rand() + rand() + rand()) * 4
    } else if (kind === 'big-offset') {
      out[i] = 1e10 + Math.floor(rand() * 100)
    } else if (kind === 'negative') {
      out[i] = -50 + rand() * 100
    } else if (kind === 'duplicates') {
      out[i] = Math.floor(rand() * 3)
    } else {
      throw new Error(`unknown dataset kind: ${kind}`)
    }
  }
  return out
}

/** All arrays of length `len` over `alphabet`, in lexicographic order. */
function* gridArrays(len, alphabet) {
  if (len === 0) {
    yield []
    return
  }
  for (const rest of gridArrays(len - 1, alphabet)) {
    for (const v of alphabet) {
      yield [v, ...rest]
    }
  }
}

/**
 * Compare two clusterings exactly: same cluster count, same lengths, and
 * SameValue equality per element (NaN matches NaN; -0 does not match +0).
 * Returns null when identical, else a diagnostic string.
 */
function compareClusters(a, b) {
  if (a.length !== b.length) {
    return `cluster count differs: ${a.length} vs ${b.length}`
  }
  for (let c = 0; c < a.length; c += 1) {
    if (a[c].length !== b[c].length) {
      return `cluster ${c} length differs: ${a[c].length} vs ${b[c].length} (${JSON.stringify(a)} vs ${JSON.stringify(b)})`
    }
    for (let i = 0; i < a[c].length; i += 1) {
      const va = a[c][i]
      const vb = b[c][i]
      if (Object.is(va, vb)) continue
      if (Number.isNaN(va) && Number.isNaN(vb)) continue
      return `cluster ${c} element ${i} differs: ${va} vs ${vb} (${JSON.stringify(a)} vs ${JSON.stringify(b)})`
    }
  }
  return null
}

module.exports = { rng, makeDataset, KINDS, gridArrays, compareClusters }
