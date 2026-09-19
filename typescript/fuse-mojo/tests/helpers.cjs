'use strict'

/**
 * Deterministic seeded corpus and pattern generators for the differential
 * suite. Same seeds on every run (and every backend) produce byte-identical
 * corpora and patterns, so any mismatch is a real behavioral difference.
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
  'la', 'ren', 'do', 'pas', 'te', 'gu', 'yar', 'es', 'in', 'um', 'ax', 'oth',
]
const UNICODE_TOKENS = [
  'café', 'naïve', 'résumé', 'über', 'sœur', '日本語', '東京', 'Москва',
  'a😀b', '🚀launch', 'Æsop', 'ﬁsh', 'İstanbul', 'ßeta',
]

function makeWord(rand) {
  const n = 2 + Math.floor(rand() * 3)
  let w = ''
  for (let i = 0; i < n; i += 1) {
    w += SYLLABLES[Math.floor(rand() * SYLLABLES.length)]
  }
  return w
}

/**
 * Generate `nDocs` documents of 5-15 word-like tokens; every `unicodeEvery`-th
 * document gets one unicode token injected at a deterministic position.
 */
function generateCorpus(seed, nDocs, { unicodeEvery = 0 } = {}) {
  const rand = rng(seed)
  const vocab = new Set()
  while (vocab.size < Math.min(3000, nDocs * 4)) {
    vocab.add(makeWord(rand))
  }
  const words = [...vocab]
  const docs = []
  for (let d = 0; d < nDocs; d += 1) {
    const nTok = 5 + Math.floor(rand() * 11)
    const tokens = []
    for (let t = 0; t < nTok; t += 1) {
      tokens.push(words[Math.floor(rand() * words.length)])
    }
    if (unicodeEvery && d % unicodeEvery === 0) {
      tokens[Math.floor(rand() * tokens.length)] =
        UNICODE_TOKENS[Math.floor(rand() * UNICODE_TOKENS.length)]
    }
    docs.push(tokens.join(' '))
  }
  return docs
}

function mutateToken(rand, token) {
  const ops = ['sub', 'del', 'ins', 'swap']
  const chars = 'abcdefghijklmnopqrstuvwxyz'
  let out = token
  const nMut = 1 + Math.floor(rand() * 2)
  for (let m = 0; m < nMut && out.length > 0; m += 1) {
    const op = ops[Math.floor(rand() * ops.length)]
    const pos = Math.floor(rand() * out.length)
    if (op === 'sub') {
      out = out.slice(0, pos) + chars[Math.floor(rand() * 26)] + out.slice(pos + 1)
    } else if (op === 'del' && out.length > 1) {
      out = out.slice(0, pos) + out.slice(pos + 1)
    } else if (op === 'ins') {
      out = out.slice(0, pos) + chars[Math.floor(rand() * 26)] + out.slice(pos)
    } else if (op === 'swap' && out.length > 1) {
      const p2 = Math.min(pos + 1, out.length - 1)
      out = out.slice(0, pos) + out[p2] + out[pos] + out.slice(p2 + 1)
    }
  }
  return out
}

/**
 * ~count patterns with a deterministic mix: exact corpus tokens, typo'd
 * tokens, substrings, multi-token spans, absent tokens, unicode, very short,
 * and a few longer than 32 code units (chunking path).
 */
function generatePatterns(seed, docs, count) {
  const rand = rng(seed ^ 0x9e3779b9)
  const patterns = []
  const tokenOf = (doc) => {
    const tokens = doc.split(' ')
    return tokens[Math.floor(rand() * tokens.length)]
  }
  const randDoc = () => docs[Math.floor(rand() * docs.length)]

  while (patterns.length < count) {
    const kind = patterns.length % 10
    if (kind === 0) {
      patterns.push(tokenOf(randDoc())) // exact token
    } else if (kind === 1 || kind === 2 || kind === 3) {
      patterns.push(mutateToken(rand, tokenOf(randDoc()))) // typo'd
    } else if (kind === 4) {
      const tok = tokenOf(randDoc())
      if (tok.length >= 4) {
        const start = Math.floor(rand() * (tok.length - 3))
        patterns.push(tok.slice(start, start + 3 + Math.floor(rand() * 4)))
      } else {
        patterns.push(tok)
      }
    } else if (kind === 5) {
      patterns.push(`${tokenOf(randDoc())} ${tokenOf(randDoc())}`) // two tokens
    } else if (kind === 6) {
      patterns.push(`${tokenOf(randDoc())}zzqx`) // absent
    } else if (kind === 7) {
      patterns.push(UNICODE_TOKENS[Math.floor(rand() * UNICODE_TOKENS.length)])
    } else if (kind === 8) {
      const tok = tokenOf(randDoc())
      patterns.push(tok.slice(0, 1 + Math.floor(rand() * 2))) // very short
    } else {
      patterns.push(
        `${tokenOf(randDoc())}${tokenOf(randDoc())}${tokenOf(randDoc())}${tokenOf(randDoc())}`
      ) // often > 32 code units
    }
  }
  return patterns
}

/** Object-list variant of a string corpus: {title, author, tags}. */
function toObjectDocs(seed, docs) {
  const rand = rng(seed ^ 0x51f15e)
  const first = ['ka', 'me', 'ra', 'zo', 'lu', 'vi']
  const last = ['sen', 'berg', 'man', 'son', 'ova', 'li']
  return docs.map((doc, i) => {
    const tokens = doc.split(' ')
    const obj = {
      title: tokens.slice(0, 3).join(' '),
      author:
        first[Math.floor(rand() * first.length)] + last[Math.floor(rand() * last.length)],
      tags: tokens.slice(3, 3 + Math.floor(rand() * 3)),
      meta: { code: `doc-${i} ${tokens[0]}` },
      year: 1950 + Math.floor(rand() * 75),
    }
    if (i % 17 === 0) {
      obj.tags = undefined // missing key value on some docs
    }
    if (i % 23 === 0) {
      obj.title = '   ' // blank value on some docs
    }
    return obj
  })
}

/**
 * Compare two result arrays element-wise: identical refIndex order, scores
 * within 1e-9, structurally identical matches. Returns null when equal, else
 * a diagnostic string.
 */
function compareResults(a, b, { includeScore, includeMatches }) {
  if (a.length !== b.length) {
    return `result count differs: ${a.length} vs ${b.length}`
  }
  let maxDiff = 0
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].refIndex !== b[i].refIndex) {
      return `refIndex order differs at rank ${i}: ${a[i].refIndex} vs ${b[i].refIndex}`
    }
    if (includeScore) {
      const diff = Math.abs(a[i].score - b[i].score)
      maxDiff = Math.max(maxDiff, diff)
      if (!(diff <= 1e-9)) {
        return `score differs at rank ${i} (refIndex ${a[i].refIndex}): ${a[i].score} vs ${b[i].score}`
      }
    }
    if (includeMatches) {
      const am = JSON.stringify(a[i].matches ?? null)
      const bm = JSON.stringify(b[i].matches ?? null)
      if (am !== bm) {
        return `matches differ at rank ${i} (refIndex ${a[i].refIndex}):\n  mojo: ${am}\n  fuse: ${bm}`
      }
    }
  }
  return null
}

module.exports = { rng, generateCorpus, generatePatterns, toObjectDocs, compareResults, UNICODE_TOKENS }
