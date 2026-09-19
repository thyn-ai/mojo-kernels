'use strict'

/**
 * Property-based parity: @fuse-mojo/core vs the published Fuse.js 7.1.0 on
 * generated corpora, patterns and option sets (fast-check).
 *
 * Where tests/differential.test.cjs replays fixed seeded corpora through a
 * hand-picked option matrix, this file lets fast-check draw the collection
 * (string lists and object lists with nested arrays, missing and blank
 * values), the patterns (corpus words, substrings, typos, joins, free
 * Unicode text, chunk-boundary lengths) and the full supported option set
 * at random, and asserts the same contract per search: identical refIndex
 * order, scores within 1e-9, structurally identical match spans (see
 * compareResults in helpers.cjs). Scenarios are plain data (word pools and
 * index lists), so a failure shrinks to a minimal counterexample that
 * fast-check prints together with the seed and path that replay it.
 *
 * Runs on whichever backend the environment selects (FUSE_MOJO_DISABLE_NATIVE=1
 * forces the vendored fallback), exactly like the rest of the suite.
 *
 *   FC_NUM_RUNS   scenarios per property (default 200; the nightly fuzz
 *                 workflow runs more)
 *   FC_SEED       replay a reported seed
 *   FC_PATH       replay one reported counterexample path (with FC_SEED)
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fc = require('fast-check')
const Fuse = require('@fuse-mojo/core')
const ReferenceFuse = require('fuse.js')
const { compareResults } = require('./helpers.cjs')

const BACKEND = process.env.FUSE_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'
const NUM_RUNS = Number.parseInt(process.env.FC_NUM_RUNS ?? '200', 10)
const SEED =
  process.env.FC_SEED !== undefined
    ? Number.parseInt(process.env.FC_SEED, 10)
    : Math.floor(Math.random() * 0x7fffffff)
const PATH = process.env.FC_PATH

const RUN_PARAMS = { numRuns: NUM_RUNS, seed: SEED, path: PATH }

// ---------------------------------------------------------------- data ---

const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('')
const UNICODE_TOKENS = [
  'café', 'naïve', 'résumé', 'über', 'sœur', '日本語', '東京', 'Москва',
  'a😀b', '🚀launch', 'Æsop', 'ﬁsh', 'İstanbul', 'ßeta', 'Ω', '́x',
]

const asciiWordArb = fc.string({
  unit: fc.constantFrom(...LETTERS),
  minLength: 1,
  maxLength: 8,
})

/** One corpus token: ASCII, Unicode, mixed case, odd whitespace, or free graphemes. */
const wordArb = fc.oneof(
  { weight: 8, arbitrary: asciiWordArb },
  { weight: 2, arbitrary: fc.constantFrom(...UNICODE_TOKENS) },
  { weight: 1, arbitrary: asciiWordArb.map((w) => w.toUpperCase()) },
  { weight: 1, arbitrary: asciiWordArb.map((w) => w[0].toUpperCase() + w.slice(1)) },
  {
    weight: 1,
    arbitrary: fc.string({
      unit: fc.constantFrom(...LETTERS, ' ', '\t', '0', '9', '-', "'", '.'),
      minLength: 0,
      maxLength: 6,
    }),
  },
  { weight: 1, arbitrary: fc.string({ unit: 'grapheme', minLength: 1, maxLength: 4 }) }
)

const pick = (pool, i) => pool[i % pool.length]

/** A document is a list of pool indices joined by single spaces. */
const docSpecArb = fc.array(fc.nat(31), { minLength: 0, maxLength: 10 })

function materializeDoc(pool, spec) {
  return spec.map((i) => pick(pool, i)).join(' ')
}

/**
 * Pattern specs are pure data so shrinking stays effective:
 *   word       one pool word
 *   substring  a slice of one pool word
 *   typo       one pool word with a substitution / deletion / insertion / swap
 *   join       several pool words glued together (crosses the 32-unit chunk boundary)
 *   free       arbitrary text, up to 40 code units
 */
const patternSpecArb = fc.oneof(
  { weight: 3, arbitrary: fc.record({ kind: fc.constant('word'), i: fc.nat(31) }) },
  {
    weight: 3,
    arbitrary: fc.record({ kind: fc.constant('substring'), i: fc.nat(31), a: fc.nat(12), b: fc.nat(12) }),
  },
  {
    weight: 4,
    arbitrary: fc.record({
      kind: fc.constant('typo'),
      i: fc.nat(31),
      pos: fc.nat(12),
      op: fc.constantFrom('sub', 'del', 'ins', 'swap'),
      ch: fc.constantFrom(...LETTERS, ' ', 'É', '😀'),
    }),
  },
  {
    weight: 1,
    arbitrary: fc.record({ kind: fc.constant('join'), idx: fc.array(fc.nat(31), { minLength: 2, maxLength: 6 }) }),
  },
  {
    weight: 1,
    arbitrary: fc.record({ kind: fc.constant('free'), text: fc.string({ unit: 'grapheme', minLength: 0, maxLength: 40 }) }),
  }
)

function materializePattern(pool, spec) {
  switch (spec.kind) {
    case 'word':
      return pick(pool, spec.i)
    case 'substring': {
      const w = pick(pool, spec.i)
      const lo = Math.min(spec.a, spec.b) % (w.length + 1)
      const hi = Math.max(spec.a, spec.b) % (w.length + 1)
      return w.slice(lo, hi)
    }
    case 'typo': {
      const w = pick(pool, spec.i)
      if (w.length === 0) return spec.ch
      const pos = spec.pos % w.length
      if (spec.op === 'sub') return w.slice(0, pos) + spec.ch + w.slice(pos + 1)
      if (spec.op === 'del') return w.slice(0, pos) + w.slice(pos + 1)
      if (spec.op === 'ins') return w.slice(0, pos) + spec.ch + w.slice(pos)
      const p2 = Math.min(pos + 1, w.length - 1)
      return w.slice(0, pos) + w[p2] + w[pos] + w.slice(p2 + 1)
    }
    case 'join':
      return spec.idx.map((i) => pick(pool, i)).join('')
    default:
      return spec.text
  }
}

/** Every supported option, each optional so defaults are exercised too. */
const optionsArb = fc.record(
  {
    threshold: fc.oneof(fc.constantFrom(0, 0.3, 0.6, 1), fc.double({ min: 0, max: 1, noNaN: true })),
    location: fc.oneof(fc.constant(0), fc.integer({ min: -5, max: 200 })),
    distance: fc.oneof(fc.constant(100), fc.constant(0), fc.nat(300)),
    minMatchCharLength: fc.integer({ min: 1, max: 6 }),
    includeMatches: fc.boolean(),
    ignoreLocation: fc.boolean(),
    findAllMatches: fc.boolean(),
    isCaseSensitive: fc.boolean(),
    shouldSort: fc.boolean(),
    ignoreFieldNorm: fc.boolean(),
    fieldNormWeight: fc.constantFrom(1, 0.5, 2),
  },
  { requiredKeys: [] }
)

const limitArb = fc.oneof(fc.constant(-1), fc.nat(5))

const stringScenarioArb = fc.record({
  pool: fc.array(wordArb, { minLength: 1, maxLength: 12 }),
  docs: fc.array(docSpecArb, { minLength: 0, maxLength: 30 }),
  patterns: fc.array(patternSpecArb, { minLength: 1, maxLength: 4 }),
  options: optionsArb,
  limit: limitArb,
})

// Object documents: string, nested-array, missing, blank and numeric values.
const KEY_NAMES = ['title', 'author', 'tags', 'meta.code', 'year']

const objectDocSpecArb = fc.record({
  title: fc.oneof({ weight: 6, arbitrary: docSpecArb }, { weight: 1, arbitrary: fc.constant(null) }),
  author: fc.nat(31),
  tags: fc.oneof(
    fc.constant(undefined),
    fc.array(fc.nat(31), { maxLength: 3 }),
    // nested arrays: the reference flattens them depth-first, in reverse
    fc.array(fc.oneof(fc.nat(31), fc.array(fc.nat(31), { maxLength: 2 })), { maxLength: 3 })
  ),
  code: docSpecArb,
  year: fc.oneof(fc.integer({ min: 1900, max: 2100 }), fc.constant(undefined), fc.constant(null)),
})

function materializeObject(pool, spec) {
  const words = (s) => (Array.isArray(s) ? s.map((x) => (Array.isArray(x) ? words(x) : pick(pool, x))) : s)
  const obj = {
    title: spec.title === null ? '   ' : materializeDoc(pool, spec.title),
    author: pick(pool, spec.author),
    meta: { code: materializeDoc(pool, spec.code) },
  }
  if (spec.tags !== undefined) obj.tags = words(spec.tags)
  if (spec.year !== undefined) obj.year = spec.year
  return obj
}

const keysArb = fc.uniqueArray(
  fc.record({ name: fc.constantFrom(...KEY_NAMES), weight: fc.constantFrom(undefined, 0.5, 1, 2) }),
  { minLength: 1, maxLength: 4, selector: (k) => k.name }
).map((keys) => keys.map((k) => (k.weight === undefined ? k.name : { name: k.name, weight: k.weight })))

const objectScenarioArb = fc.record({
  pool: fc.array(wordArb, { minLength: 1, maxLength: 12 }),
  docs: fc.array(objectDocSpecArb, { minLength: 0, maxLength: 20 }),
  keys: keysArb,
  patterns: fc.array(patternSpecArb, { minLength: 1, maxLength: 4 }),
  options: optionsArb,
  limit: limitArb,
})

// ------------------------------------------------------------- oracle ---

/**
 * Search every pattern on both implementations; return null on parity or a
 * diagnostic naming the first pattern that differs (the property throws it,
 * so fast-check shrinks the scenario and reports the minimal one).
 */
function parityProblem(docs, patterns, options, limit) {
  const opts = { ...options, includeScore: true }
  const mojo = new Fuse(docs, opts)
  try {
    if (mojo.backend !== BACKEND) {
      return `expected ${BACKEND} backend, got ${mojo.backend}`
    }
    const ref = new ReferenceFuse(docs, opts)
    for (const pattern of patterns) {
      const problem = compareResults(mojo.search(pattern, { limit }), ref.search(pattern, { limit }), opts)
      if (problem) {
        return `pattern ${JSON.stringify(pattern)} (limit ${limit}, options ${JSON.stringify(opts)}): ${problem}`
      }
    }
    return null
  } finally {
    mojo.destroy()
  }
}

function assertProperty(t, arbitrary, run) {
  t.diagnostic(`fast-check seed ${SEED} numRuns ${NUM_RUNS} backend ${BACKEND}`)
  fc.assert(
    fc.property(arbitrary, (scenario) => {
      const problem = run(scenario)
      if (problem) {
        throw new Error(problem)
      }
    }),
    RUN_PARAMS
  )
}

// --------------------------------------------------------------- tests ---

test(`property [${BACKEND}] string collections match Fuse.js 7.1.0`, (t) => {
  assertProperty(t, stringScenarioArb, ({ pool, docs, patterns, options, limit }) =>
    parityProblem(
      docs.map((d) => materializeDoc(pool, d)),
      patterns.map((p) => materializePattern(pool, p)),
      options,
      limit
    )
  )
})

test(`property [${BACKEND}] object collections with weighted keys match Fuse.js 7.1.0`, (t) => {
  assertProperty(t, objectScenarioArb, ({ pool, docs, keys, patterns, options, limit }) =>
    parityProblem(
      docs.map((d) => materializeObject(pool, d)),
      patterns.map((p) => materializePattern(pool, p)),
      { ...options, keys },
      limit
    )
  )
})

test(`property [${BACKEND}] setCollection re-index equals a fresh index`, (t) => {
  // Re-indexing an instance must be indistinguishable from constructing a
  // new one on the same backend (the native handle is rebuilt).
  assertProperty(
    t,
    fc.record({
      pool: fc.array(wordArb, { minLength: 1, maxLength: 8 }),
      first: fc.array(docSpecArb, { maxLength: 10 }),
      second: fc.array(docSpecArb, { maxLength: 10 }),
      pattern: patternSpecArb,
      options: optionsArb,
    }),
    ({ pool, first, second, pattern, options }) => {
      const opts = { ...options, includeScore: true }
      const docsA = first.map((d) => materializeDoc(pool, d))
      const docsB = second.map((d) => materializeDoc(pool, d))
      const query = materializePattern(pool, pattern)
      const reused = new Fuse(docsA, opts)
      const fresh = new Fuse(docsB, opts)
      try {
        reused.setCollection(docsB)
        assert.equal(reused.backend, BACKEND)
        return compareResults(reused.search(query), fresh.search(query), opts)
      } finally {
        reused.destroy()
        fresh.destroy()
      }
    }
  )
})
