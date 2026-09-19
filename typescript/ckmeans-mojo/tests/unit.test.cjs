'use strict'

/**
 * Unit tests for @ckmeans-mojo/core: API surface, backend reporting, and
 * wrapper-level behavior that does not involve the reference package.
 * Runs on the native backend by default and on the forced fallback when
 * CKMEANS_MOJO_DISABLE_NATIVE=1.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const ckmeans = require('@ckmeans-mojo/core')

const BACKEND = process.env.CKMEANS_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

test('API surface', () => {
  assert.equal(typeof ckmeans, 'function')
  assert.equal(typeof ckmeans.nativeAvailable, 'function')
  assert.equal(typeof ckmeans.backendInfo, 'function')
  assert.equal(typeof ckmeans.version, 'string')
  assert.equal(ckmeans.default, ckmeans)
  assert.equal(ckmeans.ckmeans, ckmeans)
  assert.equal(ckmeans.nativeAvailable(), BACKEND === 'native')
  assert.equal(ckmeans.backend, BACKEND)
})

test('backendInfo shape', () => {
  const info = ckmeans.backendInfo()
  assert.equal(info.native_available, BACKEND === 'native')
  assert.equal(info.abi_version_expected, 1)
  assert.equal(info.disabled_by_env, BACKEND === 'fallback')
  assert.equal(info.platform, process.platform)
  assert.equal(info.arch, process.arch)
  if (BACKEND === 'native') {
    assert.equal(info.abi_version_native, 1)
    assert.match(info.native_source, /libckmeansmojo/)
  } else {
    assert.equal(info.abi_version_native, null)
  }
})

test('basic clustering (simple-statistics README example)', () => {
  const clusters = ckmeans([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3)
  assert.deepEqual(clusters, [[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]])
})

test('clusters are ascending and cover the input', () => {
  const x = [10, 1, 5, 2, 9, 6, 1, 8]
  const clusters = ckmeans(x, 3)
  assert.equal(clusters.flat().length, x.length)
  for (const cluster of clusters) {
    for (let i = 1; i < cluster.length; i += 1) {
      assert.ok(cluster[i] >= cluster[i - 1], 'cluster is sorted')
    }
  }
  for (let c = 1; c < clusters.length; c += 1) {
    assert.ok(clusters[c][0] >= clusters[c - 1][clusters[c - 1].length - 1], 'clusters ascend')
  }
})

test('returned clusters are fresh arrays (mutating them is safe)', () => {
  const x = [3, 1, 2, 10, 11, 12]
  const a = ckmeans(x, 2)
  a[0][0] = -999
  const b = ckmeans(x, 2)
  assert.notEqual(b[0][0], -999)
})

test('k > n throws the reference error', () => {
  assert.throws(() => ckmeans([1, 2, 3], 4), {
    name: 'Error',
    message: 'cannot generate more classes than there are data values',
  })
})

test('k of zero or less throws the reference TypeError', () => {
  for (const k of [0, -1, -2.5, NaN, null]) {
    assert.throws(() => ckmeans([1, 2, 3], k), {
      name: 'TypeError',
      message: "Cannot read properties of undefined (reading 'length')",
    })
  }
})

test('undefined input throws a TypeError', () => {
  assert.throws(() => ckmeans(undefined, 2), TypeError)
})

test('constant array collapses to one cluster for any k', () => {
  assert.deepEqual(ckmeans([7, 7, 7], 1), [[7, 7, 7]])
  assert.deepEqual(ckmeans([7, 7, 7], 2), [[7, 7, 7]])
  assert.deepEqual(ckmeans([7, 7, 7], 3), [[7, 7, 7]])
})

test('non-integer k behaves like the reference (ceil rows)', () => {
  assert.deepEqual(ckmeans([1, 2, 3, 4], 2.5), [[1, 2], [3], [4]])
  assert.throws(() => ckmeans([1, 2, 3], 3.5), {
    message: 'cannot generate more classes than there are data values',
  })
})

test('native kernel result equals a manually verified partition', () => {
  // DP optimum for [1,2,10,11,100] k=2 is {1,2,10,11}|{100} (withinss 42.5),
  // beating {1,2}|{10,11,100} (withinss 4020.5).
  assert.deepEqual(ckmeans([100, 11, 1, 10, 2], 2), [[1, 2, 10, 11], [100]])
})
