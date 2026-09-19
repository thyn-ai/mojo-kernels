'use strict'

/**
 * Deterministic seeded string-pair generators for the differential suite.
 * Same seeds on every run (and every backend) produce identical pairs, so
 * any mismatch is a real behavioral difference.
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

const SYLLABLES = [
  'ka', 'zo', 'mi', 'ru', 'ta', 'ne', 'lo', 'vi', 'sa', 'dre', 'pin', 'gor',
  'ma', 'fel', 'ix', 'qu', 'an', 'bel', 'or', 'din', 'sta', 'ver', 'mo', 'chi',
]
const UNICODE_TOKENS = [
  'café', 'naïve', 'résumé', 'über', 'sœur', '日本語', '東京', 'Москва',
  'a😀b', '🚀launch', 'Æsop', 'ﬁsh', 'İstanbul', 'ßeta',
]
const ALPHA = 'abcdefghijklmnopqrstuvwxyz'

function makeWord(rand, maxSyllables = 4) {
  const n = 1 + Math.floor(rand() * maxSyllables)
  let w = ''
  for (let i = 0; i < n; i += 1) {
    w += SYLLABLES[Math.floor(rand() * SYLLABLES.length)]
  }
  return w
}

/** One seeded edit: substitution, deletion, insertion, or adjacent swap. */
function mutateToken(rand, token) {
  let out = token
  const nMut = 1 + Math.floor(rand() * 2)
  for (let m = 0; m < nMut && out.length > 0; m += 1) {
    const op = Math.floor(rand() * 4)
    const pos = Math.floor(rand() * out.length)
    if (op === 0) {
      out = out.slice(0, pos) + ALPHA[Math.floor(rand() * 26)] + out.slice(pos + 1)
    } else if (op === 1 && out.length > 1) {
      out = out.slice(0, pos) + out.slice(pos + 1)
    } else if (op === 2) {
      out = out.slice(0, pos) + ALPHA[Math.floor(rand() * 26)] + out.slice(pos)
    } else if (out.length > 1) {
      const p2 = Math.min(pos + 1, out.length - 1)
      out = out.slice(0, pos) + out[p2] + out[pos] + out.slice(p2 + 1)
    }
  }
  return out
}

/**
 * ~count pairs with a deterministic mix: identical words, 1-2-edit typos,
 * unrelated words, prefix/suffix relations, empty strings, unicode tokens,
 * and block moves (transposition bait).
 */
function generatePairs(seed, count, { unicodeEvery = 0 } = {}) {
  const rand = rng(seed)
  const pairs = []
  while (pairs.length < count) {
    const kind = pairs.length % 10
    const w = makeWord(rand)
    if (kind === 0) {
      pairs.push([w, w]) // identical
    } else if (kind === 1 || kind === 2 || kind === 3) {
      pairs.push([w, mutateToken(rand, w)]) // typo'd
    } else if (kind === 4) {
      pairs.push([w, makeWord(rand)]) // unrelated
    } else if (kind === 5) {
      pairs.push([w, w + makeWord(rand, 1)]) // prefix relation
    } else if (kind === 6) {
      pairs.push([w, w.slice(0, Math.max(1, Math.floor(w.length / 2)))]) // truncation
    } else if (kind === 7) {
      pairs.push([w, '']) // empty target
    } else if (kind === 8 && unicodeEvery && pairs.length % unicodeEvery === 0) {
      const u = UNICODE_TOKENS[Math.floor(rand() * UNICODE_TOKENS.length)]
      pairs.push([u, mutateToken(rand, u)]) // unicode typo
    } else {
      // block move: split in three, swap the outer blocks (damerau bait)
      const a = Math.floor(w.length / 3)
      const b = Math.floor((2 * w.length) / 3)
      pairs.push([w, w.slice(b) + w.slice(a, b) + w.slice(0, a)])
    }
  }
  return pairs
}

/**
 * Pairs engineered to make OSA and unrestricted Damerau disagree:
 * reversed digraphs, interleaved swaps, and swapped blocks at varying
 * distances (true Damerau fixes far-apart transpositions cheaply).
 */
function generateTranspositionPairs(seed, count) {
  const rand = rng(seed)
  const pairs = []
  while (pairs.length < count) {
    const w = makeWord(rand, 5)
    const kind = pairs.length % 4
    if (kind === 0 && w.length >= 2) {
      const i = Math.floor(rand() * (w.length - 1))
      pairs.push([w, w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2)]) // adjacent swap
    } else if (kind === 1 && w.length >= 4) {
      // two disjoint adjacent swaps
      const i = Math.floor(rand() * (w.length - 3))
      const j = i + 2 + Math.floor(rand() * (w.length - i - 3))
      const s = w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2)
      pairs.push([s, s.slice(0, j) + s[j + 1] + s[j] + s.slice(j + 2)])
    } else if (kind === 2 && w.length >= 3) {
      pairs.push([w, w[1] + w[0] + w.slice(2)]) // head swap (ca/abc family)
    } else {
      const mid = Math.floor(w.length / 2)
      pairs.push([w, w.slice(mid) + w.slice(0, mid)]) // half rotation
    }
  }
  return pairs
}

/**
 * Exact-equality comparison for two distances. Both backends and the oracle
 * compute IEEE-754 float64 in the same operation order, so integer and
 * fractional-cost results alike must agree bit-for-bit. NaN is compared with
 * Number.isNaN (unreachable for finite-cost inputs).
 */
function compareExact(mine, ref, ctx) {
  if (Number.isNaN(mine) && Number.isNaN(ref)) return null
  if (mine !== ref) {
    return `${ctx}: distance differs: mojo=${mine} vs natural=${ref}`
  }
  return null
}

module.exports = {
  rng,
  makeWord,
  mutateToken,
  generatePairs,
  generateTranspositionPairs,
  compareExact,
  UNICODE_TOKENS,
}
