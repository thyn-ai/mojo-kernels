'use strict'

/**
 * Differential suite: @fuse-mojo/core (native backend, or forced fallback
 * when FUSE_MOJO_DISABLE_NATIVE=1) vs the published Fuse.js 7.1.0 package,
 * on deterministic seeded corpora and ~200 generated patterns per cell.
 *
 * Asserted per pattern: identical refIndex ORDER (the reference sort is a
 * total order over (score, idx), so ties cannot reorder), scores within
 * 1e-9, and structurally identical match spans.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const Fuse = require('@fuse-mojo/core')
const ReferenceFuse = require('fuse.js')
const {
  generateCorpus,
  generatePatterns,
  toObjectDocs,
  compareResults,
} = require('./helpers.cjs')

const BACKEND = process.env.FUSE_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

const corpus = generateCorpus(1337, 800, { unicodeEvery: 9 })
const patterns = generatePatterns(1337, corpus, 200)
const unicodeCorpus = generateCorpus(4242, 300, { unicodeEvery: 2 })
const unicodePatterns = generatePatterns(4242, unicodeCorpus, 60)
const objectDocs = toObjectDocs(1337, corpus)

function runCell(name, docs, patterns, options) {
  test(`differential [${BACKEND}] ${name} (${patterns.length} patterns)`, () => {
    const mojo = new Fuse(docs, options)
    const ref = new ReferenceFuse(docs, options)
    assert.equal(mojo.backend, BACKEND, `expected ${BACKEND} backend, got ${mojo.backend}`)
    const failures = []
    for (const pattern of patterns) {
      const a = mojo.search(pattern)
      const b = ref.search(pattern)
      const problem = compareResults(a, b, options)
      if (problem) {
        failures.push(`pattern ${JSON.stringify(pattern)}: ${problem}`)
        if (failures.length >= 3) break
      }
    }
    assert.deepEqual(failures, [], failures.join('\n'))
  })
}

// --- string-list cells over the option matrix ---
runCell('defaults', corpus, patterns, {})
runCell('score+matches', corpus, patterns, { includeScore: true, includeMatches: true })
runCell('threshold 0.0', corpus, patterns, { threshold: 0.0, includeScore: true })
runCell('threshold 0.3', corpus, patterns, { threshold: 0.3, includeScore: true })
runCell('threshold 1.0', corpus, patterns, {
  threshold: 1.0,
  includeScore: true,
  includeMatches: true,
})
runCell('location+distance', corpus, patterns, {
  location: 4,
  distance: 12,
  threshold: 0.5,
  includeScore: true,
  includeMatches: true,
})
runCell('distance 0', corpus, patterns, { distance: 0, threshold: 0.9, includeScore: true })
runCell('ignoreLocation', corpus, patterns, {
  ignoreLocation: true,
  includeScore: true,
  includeMatches: true,
})
runCell('findAllMatches', corpus, patterns, {
  findAllMatches: true,
  includeScore: true,
  includeMatches: true,
})
runCell('minMatchCharLength 2', corpus, patterns, {
  minMatchCharLength: 2,
  includeScore: true,
  includeMatches: true,
  threshold: 0.8,
})
runCell('minMatchCharLength 4', corpus, patterns, {
  minMatchCharLength: 4,
  includeScore: true,
  includeMatches: true,
  threshold: 0.9,
})
runCell('no sort, no score', corpus, patterns, { shouldSort: false, threshold: 0.5 })
runCell('ignoreFieldNorm', corpus, patterns, {
  ignoreFieldNorm: true,
  includeScore: true,
  threshold: 0.8,
})
runCell('fieldNormWeight 0.5', corpus, patterns, {
  fieldNormWeight: 0.5,
  includeScore: true,
  threshold: 0.8,
})

// --- unicode-heavy corpus (UTF-16 code-unit semantics, emoji, accents, CJK) ---
runCell('unicode', unicodeCorpus, unicodePatterns, {
  includeScore: true,
  includeMatches: true,
  threshold: 0.7,
})
runCell('unicode case-sensitive', unicodeCorpus, unicodePatterns, {
  isCaseSensitive: true,
  includeScore: true,
  includeMatches: true,
  threshold: 0.7,
})

// --- mixed-case corpus (case folding of pattern and text) ---
const mixedCase = corpus.map((doc, i) =>
  i % 3 === 0 ? doc.toUpperCase() : i % 3 === 1 ? doc.replace(/(^| )[a-z]/g, (c) => c.toUpperCase()) : doc
)
const mixedPatterns = generatePatterns(777, mixedCase, 80).map((p, i) =>
  i % 2 === 0 ? p.toUpperCase() : p
)
runCell('mixed case insensitive', mixedCase, mixedPatterns, {
  includeScore: true,
  includeMatches: true,
  threshold: 0.7,
})
runCell('mixed case sensitive', mixedCase, mixedPatterns, {
  isCaseSensitive: true,
  includeScore: true,
  includeMatches: true,
  threshold: 0.7,
})

// --- object-list cells ---
runCell('object single key', objectDocs, patterns, {
  keys: ['title'],
  includeScore: true,
  includeMatches: true,
  threshold: 0.6,
})
runCell('object multi key weighted', objectDocs, patterns, {
  keys: [{ name: 'title', weight: 2 }, 'author', { name: 'meta.code', weight: 0.5 }],
  includeScore: true,
  includeMatches: true,
  threshold: 0.6,
})
runCell('object array values', objectDocs, patterns, {
  keys: ['tags', 'title'],
  includeScore: true,
  includeMatches: true,
  threshold: 0.5,
})
runCell('object nested path + number leaf', objectDocs, patterns, {
  keys: ['meta.code', 'year'],
  includeScore: true,
  includeMatches: true,
  threshold: 0.4,
})

// --- empty/edge collections ---
test(`differential [${BACKEND}] edge collections`, () => {
  const edgeDocs = ['', '   ', 'a', 'ab', 'abc', 'x'.repeat(100), 'a😀b', 'z']
  const edgePatterns = ['', 'a', 'ab', 'abc', 'x'.repeat(40), '😀', ' ', 'zz', 'a😀b', 'a😀']
  for (const options of [
    { includeScore: true, includeMatches: true, threshold: 1.0 },
    { threshold: 0.0, includeScore: true },
    { minMatchCharLength: 3, includeMatches: true, threshold: 0.5 },
  ]) {
    const mojo = new Fuse(edgeDocs, options)
    const ref = new ReferenceFuse(edgeDocs, options)
    for (const pattern of edgePatterns) {
      const a = mojo.search(pattern)
      const b = ref.search(pattern)
      const problem = compareResults(a, b, options)
      assert.equal(problem, null, `opts=${JSON.stringify(options)} pattern=${JSON.stringify(pattern)}: ${problem}`)
    }
  }
})

// --- search limit parameter ---
test(`differential [${BACKEND}] search limit`, () => {
  const options = { threshold: 0.4, includeScore: true }
  const mojo = new Fuse(corpus, options)
  const ref = new ReferenceFuse(corpus, options)
  for (const limit of [0, 1, 3, 50]) {
    for (const pattern of patterns.slice(0, 40)) {
      const a = mojo.search(pattern, { limit })
      const b = ref.search(pattern, { limit })
      const problem = compareResults(a, b, options)
      assert.equal(problem, null, `limit=${limit} pattern=${JSON.stringify(pattern)}: ${problem}`)
    }
  }
})
