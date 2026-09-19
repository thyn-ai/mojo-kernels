'use strict'

/**
 * Differential suite: @ckmeans-mojo/core (native backend, or forced fallback
 * when CKMEANS_MOJO_DISABLE_NATIVE=1) vs the published simple-statistics
 * 7.12.0 package, on exhaustive small grids, deterministic seeded datasets,
 * and the documented edge cases.
 *
 * Asserted per case: EXACT cluster-assignment parity — identical cluster
 * count, lengths, and SameValue-equal elements (partitions are discrete;
 * values pass through unmodified, so equality is exact, not tolerant).
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const ckmeans = require('@ckmeans-mojo/core')
const reference = require('simple-statistics')
const { rng, makeDataset, KINDS, gridArrays, compareClusters } = require('./helpers.cjs')

const BACKEND = process.env.CKMEANS_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

function assertParity(x, k, context) {
  const a = ckmeans(x, k)
  const b = reference.ckmeans(x, k)
  const problem = compareClusters(a, b)
  assert.equal(problem, null, `${context}: ${problem}`)
}

// --- exhaustive small grids: every array of length 1..6 over {0,1,2,3},
// every valid k — pins the tie-resolution convention exactly ---
for (let len = 1; len <= 6; len += 1) {
  test(`differential [${BACKEND}] exhaustive grid len ${len} over {0,1,2,3}`, () => {
    assert.equal(ckmeans.backend, BACKEND, `expected ${BACKEND} backend, got ${ckmeans.backend}`)
    let count = 0
    for (const arr of gridArrays(len, [0, 1, 2, 3])) {
      for (let k = 1; k <= len; k += 1) {
        assertParity(arr, k, `grid ${JSON.stringify(arr)} k=${k}`)
        count += 1
      }
    }
    assert.equal(count, len * 4 ** len)
  })
}

// --- seeded random datasets across all kinds, random valid k ---
for (const kind of KINDS) {
  test(`differential [${BACKEND}] seeded ${kind} datasets`, () => {
    const rand = rng(9001 + KINDS.indexOf(kind))
    for (let t = 0; t < 400; t += 1) {
      const n = 2 + Math.floor(rand() * 80)
      const x = makeDataset(1000 + t * 17, n, kind)
      const k = 1 + Math.floor(rand() * Math.min(n, 9))
      assertParity(x, k, `${kind} seed=${1000 + t * 17} n=${n} k=${k}`)
    }
  })
}

// --- larger realistic workload (guards float divergence at bigger sums) ---
test(`differential [${BACKEND}] mixture n=2000 k=8`, () => {
  const x = makeDataset(4242, 2000, 'mixture')
  assertParity(x, 8, 'mixture n=2000 k=8')
})

// --- documented edge cases ---
test(`differential [${BACKEND}] edge cases (k=1, k=n, constant, non-integer k)`, () => {
  const cases = [
    [[42], 1],
    [[3.14], 1],
    [[5, 3, 1, 4, 2], 1], // k=1
    [[5, 3, 1, 4, 2], 5], // k=n
    [[9, 9], 1],
    [[9, 9], 2], // constant, k=n
    [[7, 7, 7], 1],
    [[7, 7, 7], 2], // constant, k < n
    [[7, 7, 7], 3], // constant, k = n
    [[1, 1, 2, 2], 3], // duplicates
    [[1, 1, 1, 2], 2],
    [[1, 1, 1, 2], 3],
    [[3, 1, 2], 2], // unsorted input
    [[-5, -1, 2, 10], 2], // negatives
    [[-2, -1, 1, 2], 3],
    [[0.1, 0.2, 0.9, 1.0, 5.5], 2], // floats
    [[1e10, 1e10 + 2, 1e10 + 4], 2], // big offset
    [[1, 2, 3, 4], 2.5], // non-integer k (ceil rows)
    [[1, 2, 3, 4], 2.1],
    [[1, 2, 3, 4], 1.5],
    [[1, 2, 3, 4], 0.5],
    [[1, 2, 3, 4], 3.499],
    [[1, 2, 3, 4], 3.5], // ceil(3.5) = 4 = n
    [[1, 2, 3, 4], '2'], // string k coerces
    [[1, 2, 3], 2.9], // ceil(2.9) = 3 = n
  ]
  for (const [x, k] of cases) {
    assertParity(x, k, `edge ${JSON.stringify(x)} k=${k}`)
  }
})

// --- constant arrays of every size, every valid k ---
test(`differential [${BACKEND}] constant arrays n=1..8`, () => {
  for (let n = 1; n <= 8; n += 1) {
    const x = new Array(n).fill(7)
    for (let k = 1; k <= n; k += 1) {
      assertParity(x, k, `constant n=${n} k=${k}`)
    }
  }
})

// --- error parity: identical error class and message on both backends ---
test(`differential [${BACKEND}] error surfaces match the reference`, () => {
  const cases = [
    [[1, 2, 3], 4], // k > n
    [[], 1], // empty input
    [[], 0],
    [[1, 2, 3], 0],
    [[1, 2, 3], -1],
    [[1, 2, 3, 4], 2.5 + 2], // ceil > n
    [[1, 2, 3, 4], NaN],
    [[1, 2, 3, 4], Infinity],
    [[1, 2, 3, 4], null],
    [[1, 2, 3, 4], 'abc'],
  ]
  for (const [x, k] of cases) {
    let mine
    let theirs
    try {
      ckmeans(x, k)
    } catch (err) {
      mine = `${err.constructor.name}: ${err.message}`
    }
    try {
      reference.ckmeans(x, k)
    } catch (err) {
      theirs = `${err.constructor.name}: ${err.message}`
    }
    assert.equal(mine, theirs, `error mismatch for ${JSON.stringify(x)} k=${String(k)}`)
    assert.notEqual(mine, undefined, `expected a throw for ${JSON.stringify(x)} k=${String(k)}`)
  }
})

// --- input is never mutated ---
test(`differential [${BACKEND}] input array is not mutated`, () => {
  const x = [9, 3, 7, 1, 3, 3]
  const snapshot = x.slice()
  ckmeans(x, 3)
  assert.deepEqual(x, snapshot)
})

// --- -0/+0 and duplicate-value ordering through the sort ---
test(`differential [${BACKEND}] signed zeros and duplicates`, () => {
  const rand = rng(5150)
  for (let t = 0; t < 100; t += 1) {
    const n = 2 + Math.floor(rand() * 20)
    const x = []
    for (let i = 0; i < n; i += 1) {
      const v = Math.floor(rand() * 5) - 2
      x.push(rand() < 0.5 ? v : v * (rand() < 0.3 ? -0 : 1)) // sprinkle -0
    }
    const k = 1 + Math.floor(rand() * Math.min(n, 4))
    assertParity(x, k, `signed zeros seed=${t} k=${k}`)
  }
})
