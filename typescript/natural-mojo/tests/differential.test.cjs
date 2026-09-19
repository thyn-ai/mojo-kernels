'use strict'

/**
 * Differential suite: @natural-mojo/core (native backend, or forced fallback
 * when NATURAL_MOJO_DISABLE_NATIVE=1) vs the published natural 8.1.1
 * package, on deterministic seeded string pairs.
 *
 * Distances are asserted for EXACT equality: both backends and the oracle
 * compute IEEE-754 float64 in the same operation order, so integer and
 * fractional-cost results must agree bit-for-bit.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const naturalMojo = require('@natural-mojo/core')
const natural = require('natural')
const {
  generatePairs,
  generateTranspositionPairs,
  makeWord,
  rng,
  compareExact,
} = require('./helpers.cjs')

const BACKEND = process.env.NATURAL_MOJO_DISABLE_NATIVE === '1' ? 'fallback' : 'native'

test(`backend selection [${BACKEND}]`, () => {
  assert.equal(naturalMojo.nativeAvailable(), BACKEND === 'native')
})

function runCell(name, pairs, which, options) {
  test(`differential [${BACKEND}] ${name} (${pairs.length} pairs)`, () => {
    const mojoFn = which === 'lev' ? naturalMojo.LevenshteinDistance : naturalMojo.DamerauLevenshteinDistance
    const refFn = which === 'lev' ? natural.LevenshteinDistance : natural.DamerauLevenshteinDistance
    const failures = []
    for (const [s, t] of pairs) {
      const problem = compareExact(mojoFn(s, t, options), refFn(s, t, options), `${JSON.stringify(s)}/${JSON.stringify(t)} ${JSON.stringify(options)}`)
      if (problem) {
        failures.push(problem)
        if (failures.length >= 3) break
      }
    }
    assert.deepEqual(failures, [], failures.join('\n'))
  })
}

const pairs = generatePairs(1337, 200, { unicodeEvery: 11 })
const transpoPairs = generateTranspositionPairs(4242, 80)
const unicodePairs = generatePairs(777, 60, { unicodeEvery: 1 })

// --- Levenshtein option matrix ---
runCell('lev defaults', pairs, 'lev', undefined)
runCell('lev substitution_cost 2', pairs, 'lev', { substitution_cost: 2 })
runCell('lev substitution_cost 0', pairs, 'lev', { substitution_cost: 0 })
runCell('lev insertion/deletion costs', pairs, 'lev', { insertion_cost: 2, deletion_cost: 3 })
runCell('lev float costs', pairs, 'lev', {
  insertion_cost: 0.25,
  deletion_cost: 0.5,
  substitution_cost: 1.5,
})

// --- Damerau unrestricted (Lowrance-Wagner; the reference default) ---
runCell('dlu defaults', pairs, 'dam', undefined)
runCell('dlu substitution_cost 2', pairs, 'dam', { substitution_cost: 2 })
runCell('dlu transposition_cost 3', pairs, 'dam', { transposition_cost: 3 })
runCell('dlu all costs', pairs, 'dam', {
  insertion_cost: 2,
  deletion_cost: 2,
  substitution_cost: 4,
  transposition_cost: 1,
})
runCell('dlu float costs', pairs, 'dam', {
  insertion_cost: 0.5,
  deletion_cost: 0.25,
  substitution_cost: 2,
  transposition_cost: 0.75,
})
runCell('dlu explicit restricted false', pairs, 'dam', { restricted: false })

// --- Damerau restricted (OSA) ---
runCell('dlr defaults', pairs, 'dam', { restricted: true })
runCell('dlr substitution_cost 2', pairs, 'dam', { restricted: true, substitution_cost: 2 })
runCell('dlr transposition_cost 5', pairs, 'dam', { restricted: true, transposition_cost: 5 })
runCell('dlr float costs', pairs, 'dam', {
  restricted: true,
  insertion_cost: 0.5,
  deletion_cost: 1.5,
  substitution_cost: 0.25,
  transposition_cost: 2,
})

// --- transposition-heavy pairs (OSA vs unrestricted disagreement bait) ---
runCell('dlu transposition-heavy', transpoPairs, 'dam', undefined)
runCell('dlr transposition-heavy', transpoPairs, 'dam', { restricted: true })
runCell('dlu transposition-heavy tra 3', transpoPairs, 'dam', { transposition_cost: 3 })
runCell('dlr transposition-heavy tra 5', transpoPairs, 'dam', { restricted: true, transposition_cost: 5 })
runCell('dlu transposition-heavy sub 2', transpoPairs, 'dam', { substitution_cost: 2 })
runCell('dlr transposition-heavy sub 2', transpoPairs, 'dam', { restricted: true, substitution_cost: 2 })

// --- unicode-heavy pairs (diacritics as-is, emoji = 2 UTF-16 units, CJK) ---
runCell('lev unicode', unicodePairs, 'lev', undefined)
runCell('dlu unicode', unicodePairs, 'dam', undefined)
runCell('dlr unicode', unicodePairs, 'dam', { restricted: true })
runCell('dlu unicode sub 2', unicodePairs, 'dam', { substitution_cost: 2 })

// --- edge-string sweep: empty strings, single chars, digraph reversals,
// diacritics, astral chars, 64/65-char runs (all pairs, 5 option cells) ---
const EDGE_STRINGS = [
  '', 'a', 'b', 'ab', 'ba', 'aa', 'abc', 'acb', 'abcd', 'badc', 'ca',
  'café', 'cafe', 'naïve', 'a😀b', '😀ab', '😀', '日本', 'x'.repeat(64), 'x'.repeat(64) + 'y',
]
const edgePairs = []
for (const s of EDGE_STRINGS) {
  for (const t of EDGE_STRINGS) {
    edgePairs.push([s, t])
  }
}
runCell('lev edge sweep', edgePairs, 'lev', undefined)
runCell('lev edge sweep sub 2', edgePairs, 'lev', { substitution_cost: 2 })
runCell('dlu edge sweep', edgePairs, 'dam', undefined)
runCell('dlr edge sweep', edgePairs, 'dam', { restricted: true })
runCell('dlu edge sweep tra 2 sub 3', edgePairs, 'dam', { transposition_cost: 2, substitution_cost: 3 })

// --- long strings (full-matrix / long-row paths; ~300-1200 code units) ---
function longPairs(seed) {
  const rand = rng(seed)
  const sizes = [300, 301, 700, 1200]
  return sizes.map((size) => {
    let s = ''
    while (s.length < size) s += makeWord(rand, 3)
    s = s.slice(0, size)
    // ~1% seeded edits plus one block move
    let t = s
    const nEdits = Math.max(1, Math.floor(size / 100))
    for (let i = 0; i < nEdits; i += 1) t = t.slice(0, -1) // trim then regrow deterministically
    t = t + makeWord(rand, 1).slice(0, nEdits)
    const mid = Math.floor(t.length / 2)
    t = t.slice(0, 10) + t.slice(mid, mid + 10) + t.slice(10, mid) + t.slice(mid + 10)
    return [s, t]
  })
}
const bigPairs = longPairs(9001)
runCell('lev long strings', bigPairs, 'lev', undefined)
runCell('lev long strings sub 2', bigPairs, 'lev', { substitution_cost: 2 })
runCell('dlu long strings', bigPairs, 'dam', undefined)

// --- option-merge semantics: the exact reference behavior on tricky option
// objects, asserted per case on both backends ---
test(`differential [${BACKEND}] option-merge semantics`, () => {
  const cases = [
    // [source, target, which, options]
    ['ca', 'abc', 'dam', { transposition_cost: undefined }], // NaN disables transpositions
    ['CA', 'ABC', 'dam', { transposition_cost: undefined }],
    ['ca', 'abc', 'dam', { transposition_cost: NaN }],
    ['ca', 'abc', 'dam', { transposition_cost: null }], // null coerces to 0
    ['kitten', 'sitting', 'lev', { substitution_cost: undefined }],
    ['kitten', 'sitting', 'lev', { substitution_cost: NaN }],
    ['kitten', 'sitting', 'lev', { substitution_cost: null }], // 0
    ['kitten', 'sitting', 'lev', { insertion_cost: true }], // 1
    ['kitten', 'sitting', 'lev', { deletion_cost: false }], // 0
    ['abcd', 'badc', 'dam', { restricted: 1 }], // truthy → OSA
    ['abcd', 'badc', 'dam', { restricted: 0 }], // falsy → unrestricted
    ['abcd', 'badc', 'dam', { restricted: 'yes' }], // truthy → OSA
    ['abcd', 'badc', 'lev', { damerau: true }], // overridden: plain lev
    ['kitten', 'sitting', 'lev', { search: true }], // overridden: distance
    ['kitten', 'sitting', 'dam', { search: true }], // overridden: distance
    ['ca', 'abc', 'dam', { damerau: false }], // overridden: still damerau
    ['kitten', 'sitting', 'lev', 5], // non-object options → defaults
    ['kitten', 'sitting', 'dam', 5],
    ['kitten', 'sitting', 'lev', { foo: 1 }], // unknown keys ignored
    ['kitten', 'sitting', 'dam', { foo: 1 }],
    ['saturday', 'sunday', 'lev', { insertion_cost: 2, deletion_cost: 0.5, substitution_cost: 3 }],
  ]
  const failures = []
  for (const [s, t, which, options] of cases) {
    const mojoFn = which === 'lev' ? naturalMojo.LevenshteinDistance : naturalMojo.DamerauLevenshteinDistance
    const refFn = which === 'lev' ? natural.LevenshteinDistance : natural.DamerauLevenshteinDistance
    const problem = compareExact(mojoFn(s, t, options), refFn(s, t, options), `${JSON.stringify(s)}/${JSON.stringify(t)} ${which} ${JSON.stringify(options)}`)
    if (problem) failures.push(problem)
  }
  assert.deepEqual(failures, [], failures.join('\n'))
})
