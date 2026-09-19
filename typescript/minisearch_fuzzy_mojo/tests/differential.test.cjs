'use strict'

/**
 * Differential suite: @minisearch-mojo/core (native backend, or forced
 * fallback when MINISEARCH_MOJO_DISABLE_NATIVE=1) vs the published
 * MiniSearch 7.2.0 package, on deterministic seeded corpora and ~200
 * generated queries per cell.
 *
 * Asserted per query: identical result id ORDER, scores within 1e-9, and
 * structurally identical terms/queryTerms/match; autoSuggest additionally
 * asserts identical suggestion strings (exact parity).
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const MiniSearchMojo = require('@minisearch-mojo/core')
const MiniSearchRef = require('minisearch')
const {
  generateVocab,
  generateCorpus,
  generateQueries,
  compareSearch,
  compareSuggest,
} = require('./helpers.cjs')

const BACKEND = process.env.MINISEARCH_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

const vocab = generateVocab(1337, 1200)
const corpus = generateCorpus(1337, 600, vocab, { unicodeEvery: 9 })
const queries = generateQueries(1337, vocab, 200)
const unicodeVocab = generateVocab(4242, 300)
const unicodeCorpus = generateCorpus(4242, 250, unicodeVocab, { unicodeEvery: 2 })
const unicodeQueries = generateQueries(4242, unicodeVocab, 80)

const INDEX_OPTIONS = { fields: ['title', 'text'], storeFields: ['title', 'category'] }

function runCell(name, docs, queries, searchOptions) {
  test(`differential search [${BACKEND}] ${name} (${queries.length} queries)`, () => {
    const mojo = new MiniSearchMojo(INDEX_OPTIONS)
    const ref = new MiniSearchRef(INDEX_OPTIONS)
    mojo.addAll(docs)
    ref.addAll(docs)
    assert.equal(mojo.backend, BACKEND, `expected ${BACKEND} backend, got ${mojo.backend}`)
    const failures = []
    for (const query of queries) {
      const a = mojo.search(query, searchOptions)
      const b = ref.search(query, searchOptions)
      const problem = compareSearch(a, b)
      if (problem) {
        failures.push(`query ${JSON.stringify(query)}: ${problem}`)
        if (failures.length >= 3) break
      }
    }
    assert.deepEqual(failures, [], failures.join('\n'))
  })
}

function runSuggestCell(name, docs, queries, suggestOptions) {
  test(`differential autoSuggest [${BACKEND}] ${name} (${queries.length} queries)`, () => {
    const mojo = new MiniSearchMojo(INDEX_OPTIONS)
    const ref = new MiniSearchRef(INDEX_OPTIONS)
    mojo.addAll(docs)
    ref.addAll(docs)
    const failures = []
    for (const query of queries) {
      const a = mojo.autoSuggest(query, suggestOptions)
      const b = ref.autoSuggest(query, suggestOptions)
      const problem = compareSuggest(a, b)
      if (problem) {
        failures.push(`query ${JSON.stringify(query)}: ${problem}`)
        if (failures.length >= 3) break
      }
    }
    assert.deepEqual(failures, [], failures.join('\n'))
  })
}

// --- search cells over the option matrix ---
runCell('defaults (OR)', corpus, queries, {})
runCell('combineWith AND', corpus, queries, { combineWith: 'AND' })
runCell('fuzzy 0.2', corpus, queries, { fuzzy: 0.2 })
runCell('fuzzy 0.2 AND', corpus, queries, { fuzzy: 0.2, combineWith: 'AND' })
runCell('fuzzy 0.5', corpus, queries, { fuzzy: 0.5 })
runCell('fuzzy 1 absolute', corpus, queries, { fuzzy: 1 })
runCell('fuzzy fn', corpus, queries, { fuzzy: (term) => (term.length > 5 ? 0.3 : false) })
runCell('prefix true', corpus, queries, { prefix: true })
runCell('prefix fn', corpus, queries, { prefix: (term) => term.length >= 4 })
runCell('fuzzy+prefix', corpus, queries, { fuzzy: 0.2, prefix: true })
runCell('fuzzy+prefix AND', corpus, queries, { fuzzy: 0.2, prefix: true, combineWith: 'AND' })
runCell('fuzzy 0.4 + prefix AND', corpus, queries, { fuzzy: 0.4, prefix: true, combineWith: 'AND' })
runCell('boost title x2', corpus, queries, { boost: { title: 2 } })
runCell('boost title x2 fuzzy', corpus, queries, { boost: { title: 2 }, fuzzy: 0.2 })
runCell('fields title only', corpus, queries, { fields: ['title'] })
runCell('fields title fuzzy+prefix', corpus, queries, {
  fields: ['title'],
  fuzzy: 0.2,
  prefix: true,
})
runCell('searchOptions defaults at ctor', corpus, queries, undefined === 'never' ? {} : {})

// unicode-heavy corpus (UTF-16 code-unit semantics)
runCell('unicode defaults', unicodeCorpus, unicodeQueries, {})
runCell('unicode fuzzy+prefix', unicodeCorpus, unicodeQueries, { fuzzy: 0.2, prefix: true })
runCell('unicode AND fuzzy', unicodeCorpus, unicodeQueries, { fuzzy: 0.2, combineWith: 'AND' })

// --- autoSuggest cells ---
runSuggestCell('suggest defaults', corpus, queries, {})
runSuggestCell('suggest fuzzy 0.2', corpus, queries, { fuzzy: 0.2 })
runSuggestCell('suggest fuzzy 0.3', corpus, queries, { fuzzy: 0.3 })
runSuggestCell('suggest prefix false', corpus, queries, { prefix: false })
runSuggestCell('suggest explicit prefix true', corpus, queries, { prefix: true })
runSuggestCell('suggest fuzzy+prefix', corpus, queries, { fuzzy: 0.2, prefix: true })
runSuggestCell('suggest OR', corpus, queries, { combineWith: 'OR' })
runSuggestCell('suggest boost title', corpus, queries, { boost: { title: 2 } })
runSuggestCell('suggest fields title', corpus, queries, { fields: ['title'] })
runSuggestCell('suggest unicode fuzzy', unicodeCorpus, unicodeQueries, { fuzzy: 0.2 })
runSuggestCell('suggest unicode default', unicodeCorpus, unicodeQueries, {})

// --- constructor searchOptions defaults are honored ---
test(`differential [${BACKEND}] constructor searchOptions`, () => {
  const opts = { ...INDEX_OPTIONS, searchOptions: { fuzzy: 0.2, combineWith: 'AND' } }
  const mojo = new MiniSearchMojo(opts)
  const ref = new MiniSearchRef(opts)
  mojo.addAll(corpus)
  ref.addAll(corpus)
  for (const query of queries.slice(0, 80)) {
    const a = mojo.search(query)
    const b = ref.search(query)
    const problem = compareSearch(a, b)
    assert.equal(problem, null, `query=${JSON.stringify(query)}: ${problem}`)
    const sa = mojo.autoSuggest(query)
    const sb = ref.autoSuggest(query)
    const sproblem = compareSuggest(sa, sb)
    assert.equal(sproblem, null, `suggest query=${JSON.stringify(query)}: ${sproblem}`)
  }
})

// --- edge collections ---
test(`differential [${BACKEND}] edge collections`, () => {
  const edgeDocs = [
    { id: 'a', t: 'hello world' },
    { id: 'b', t: '' },
    { id: 'c' },
    { id: 'd', t: 'a ab abc abcd abcde' },
    { id: 'e', t: 'x'.repeat(60) },
    { id: 'f', t: 'a😀b zzz' },
    { id: 'g', t: "don't stop_believing c++ test123" },
    { id: 'h', t: 12345 },
    { id: 'i', t: 'hello hello hello' },
    { id: 'j', t: 'ABCDEF mixedCASE' },
  ]
  const edgeQueries = [
    '', '   ', '!!!', 'hello', 'world', 'abc', 'abcd', 'abcde', 'ab',
    'a😀b', 'a😀', "don't", 'stop', 'believing', 'c++', 'test123', '12345',
    'hello hello', 'hello world', 'HELLO', 'abcdef', 'mixedcase', 'x'.repeat(30),
    'zzz', 'hello zzz absent',
  ]
  const optionSets = [
    {},
    { combineWith: 'AND' },
    { fuzzy: 0.2 },
    { fuzzy: 0.5, prefix: true },
    { prefix: true },
  ]
  for (const options of optionSets) {
    const mojo = new MiniSearchMojo({ fields: ['t'] })
    const ref = new MiniSearchRef({ fields: ['t'] })
    mojo.addAll(edgeDocs)
    ref.addAll(edgeDocs)
    for (const query of edgeQueries) {
      const a = mojo.search(query, options)
      const b = ref.search(query, options)
      const problem = compareSearch(a, b)
      assert.equal(problem, null, `opts=${JSON.stringify(options)} query=${JSON.stringify(query)}: ${problem}`)
      const sa = mojo.autoSuggest(query, options)
      const sb = ref.autoSuggest(query, options)
      const sproblem = compareSuggest(sa, sb)
      assert.equal(sproblem, null, `suggest opts=${JSON.stringify(options)} query=${JSON.stringify(query)}: ${sproblem}`)
    }
  }
})

// --- incremental adds (index rebuilt between queries) ---
test(`differential [${BACKEND}] incremental addAll`, () => {
  const mojo = new MiniSearchMojo(INDEX_OPTIONS)
  const ref = new MiniSearchRef(INDEX_OPTIONS)
  const third = Math.floor(corpus.length / 3)
  mojo.addAll(corpus.slice(0, third))
  ref.addAll(corpus.slice(0, third))
  const sample = queries.slice(0, 30)
  for (const query of sample) {
    compareCheck(mojo, ref, query)
  }
  mojo.addAll(corpus.slice(third, 2 * third))
  ref.addAll(corpus.slice(third, 2 * third))
  for (const query of sample) {
    compareCheck(mojo, ref, query)
  }
  mojo.addAll(corpus.slice(2 * third))
  ref.addAll(corpus.slice(2 * third))
  for (const query of sample) {
    compareCheck(mojo, ref, query)
  }
  function compareCheck(m, r, query) {
    for (const options of [{}, { fuzzy: 0.2, prefix: true, combineWith: 'AND' }]) {
      const problem = compareSearch(m.search(query, options), r.search(query, options))
      assert.equal(problem, null, `query=${JSON.stringify(query)}: ${problem}`)
      const sproblem = compareSuggest(m.autoSuggest(query, options), r.autoSuggest(query, options))
      assert.equal(sproblem, null, `suggest query=${JSON.stringify(query)}: ${sproblem}`)
    }
  }
})

// --- string ids + numeric ids mixed corpora ---
test(`differential [${BACKEND}] id types`, () => {
  const docs = [
    { id: 'doc-1', t: 'apple banana' },
    { id: 42, t: 'apple cherry' },
    { id: 'doc-3', t: 'banana cherry apple' },
  ]
  const mojo = new MiniSearchMojo({ fields: ['t'] })
  const ref = new MiniSearchRef({ fields: ['t'] })
  mojo.addAll(docs)
  ref.addAll(docs)
  for (const query of ['apple', 'apple cherry', 'apple banana cherry']) {
    for (const options of [{}, { combineWith: 'AND' }, { fuzzy: 0.2, prefix: true }]) {
      const problem = compareSearch(mojo.search(query, options), ref.search(query, options))
      assert.equal(problem, null, `query=${JSON.stringify(query)}: ${problem}`)
    }
  }
})

// --- case-duplicated tokens: field lengths count unique RAW tokens
// ('The' and 'the' are two entries), postings stay lowercased ---
test(`differential [${BACKEND}] case-duplicated tokens`, () => {
  const docs = [
    { id: 1, t: 'The quick THE brown fox jumps over THE lazy dog' },
    { id: 2, t: 'the THE Quick Brown Fox' },
    { id: 3, t: 'Apples APPLES apples aPPles' },
    { id: 4, t: 'trailing words here...' },
    { id: 5, t: '...leading dots the THE' },
  ]
  const queries = ['the', 'THE', 'The quick', 'apples', 'trailing', 'the apples', 'lazy THE']
  for (const options of [
    {},
    { combineWith: 'AND' },
    { fuzzy: 0.2 },
    { fuzzy: 0.5, prefix: true },
    { prefix: true },
  ]) {
    const mojo = new MiniSearchMojo({ fields: ['t'] })
    const ref = new MiniSearchRef({ fields: ['t'] })
    mojo.addAll(docs)
    ref.addAll(docs)
    for (const query of queries) {
      const a = mojo.search(query, options)
      const b = ref.search(query, options)
      const problem = compareSearch(a, b)
      assert.equal(problem, null, `opts=${JSON.stringify(options)} query=${JSON.stringify(query)}: ${problem}`)
      const sa = mojo.autoSuggest(query, options)
      const sb = ref.autoSuggest(query, options)
      const sproblem = compareSuggest(sa, sb)
      assert.equal(sproblem, null, `suggest opts=${JSON.stringify(options)} query=${JSON.stringify(query)}: ${sproblem}`)
    }
  }
})
