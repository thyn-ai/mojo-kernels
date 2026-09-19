'use strict'

/**
 * Unit tests: option validation, unsupported-option errors, backend
 * reporting, and structural behaviors that do not need the oracle.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const MiniSearchMojo = require('@minisearch-mojo/core')
const { UnsupportedOptionError } = require('@minisearch-mojo/core')

const BACKEND = process.env.MINISEARCH_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

function mk() {
  const ms = new MiniSearchMojo({ fields: ['t'] })
  ms.addAll([
    { id: 1, t: 'hello world' },
    { id: 2, t: 'hello there' },
  ])
  return ms
}

test(`[${BACKEND}] backend reporting`, () => {
  const ms = mk()
  assert.equal(ms.backend, BACKEND)
  const info = MiniSearchMojo.backendInfo()
  assert.equal(typeof info.native_available, 'boolean')
  assert.equal(info.disabled_by_env, process.env.MINISEARCH_MOJO_DISABLE_NATIVE === '1')
  assert.equal(info.native_available, BACKEND === 'native')
  ms.destroy()
})

test(`[${BACKEND}] unsupported options throw UnsupportedOptionError`, () => {
  assert.throws(() => new MiniSearchMojo(), UnsupportedOptionError) // no fields
  assert.throws(() => new MiniSearchMojo({ fields: [] }), UnsupportedOptionError)
  assert.throws(
    () => new MiniSearchMojo({ fields: ['t'], tokenize: (s) => [s] }),
    UnsupportedOptionError
  )
  assert.throws(
    () => new MiniSearchMojo({ fields: ['t'], processTerm: (t) => t }),
    UnsupportedOptionError
  )
  assert.throws(
    () => new MiniSearchMojo({ fields: ['t'], extractField: (d) => d.t }),
    UnsupportedOptionError
  )
  const ms = mk()
  assert.throws(() => ms.search('x', { filter: () => true }), UnsupportedOptionError)
  assert.throws(() => ms.search('x', { combineWith: 'XOR' }), UnsupportedOptionError)
  assert.throws(() => ms.search('x', { boost: { t: 'high' } }), UnsupportedOptionError)
  assert.throws(() => ms.search('x', { fuzzy: 'yes' }), UnsupportedOptionError)
  assert.throws(() => ms.search(42), UnsupportedOptionError)
  assert.throws(() => ms.autoSuggest('x', { filter: () => true }), UnsupportedOptionError)
  // > 32 distinct query terms
  const manyTerms = Array.from({ length: 33 }, (_, i) => `term${i}`).join(' ')
  assert.throws(() => ms.search(manyTerms), UnsupportedOptionError)
  ms.destroy()
})

test(`[${BACKEND}] document validation errors`, () => {
  const ms = new MiniSearchMojo({ fields: ['t'] })
  assert.throws(() => ms.add({ t: 'no id' }), /id field/)
  ms.add({ id: 1, t: 'hello' })
  assert.throws(() => ms.add({ id: 1, t: 'dup' }), /duplicate/)
  ms.destroy()
})

test(`[${BACKEND}] idField option`, () => {
  const ms = new MiniSearchMojo({ fields: ['t'], idField: 'uid' })
  ms.addAll([{ uid: 'x1', t: 'hello' }])
  const res = ms.search('hello')
  assert.equal(res.length, 1)
  assert.equal(res[0].id, 'x1')
  ms.destroy()
})

test(`[${BACKEND}] result shape`, () => {
  const ms = new MiniSearchMojo({ fields: ['t'], storeFields: ['category'] })
  ms.addAll([{ id: 1, t: 'hello world', category: 'greeting', extra: 'not-stored' }])
  const res = ms.search('hello')
  assert.equal(res.length, 1)
  const r = res[0]
  assert.equal(r.id, 1)
  assert.equal(typeof r.score, 'number')
  assert.deepEqual(r.terms, ['hello'])
  assert.deepEqual(r.queryTerms, ['hello'])
  assert.deepEqual(r.match, { hello: ['t'] })
  assert.equal(r.category, 'greeting')
  assert.equal(r.extra, undefined)
  ms.destroy()
})

test(`[${BACKEND}] empty index`, () => {
  const ms = new MiniSearchMojo({ fields: ['t'] })
  assert.deepEqual(ms.search('hello'), [])
  assert.deepEqual(ms.autoSuggest('hello'), [])
  ms.destroy()
})

test(`[${BACKEND}] version + module shape`, () => {
  assert.equal(typeof MiniSearchMojo.version, 'string')
  assert.equal(MiniSearchMojo.MiniSearch, MiniSearchMojo)
  assert.equal(typeof MiniSearchMojo.nativeAvailable, 'function')
})
