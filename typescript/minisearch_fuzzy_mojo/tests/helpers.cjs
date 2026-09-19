'use strict'

/**
 * Deterministic seeded corpus and query generators for the differential
 * suite. Same seeds on every run (and every backend) produce byte-identical
 * corpora and queries, so any mismatch is a real behavioral difference.
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

/** Build a vocabulary of unique word-like tokens. */
function generateVocab(seed, size) {
  const rand = rng(seed)
  const vocab = new Set()
  while (vocab.size < size) {
    vocab.add(makeWord(rand))
  }
  return [...vocab]
}

/**
 * Generate `nDocs` documents {id, title, text, category} drawing tokens from
 * the given vocabulary; every `unicodeEvery`-th document gets one unicode
 * token injected, and every 17th document has a missing text field.
 */
function generateCorpus(seed, nDocs, vocab, { unicodeEvery = 0 } = {}) {
  const rand = rng(seed ^ 0x51f15e)
  const docs = []
  for (let d = 0; d < nDocs; d += 1) {
    const nTitle = 1 + Math.floor(rand() * 3)
    const nText = 3 + Math.floor(rand() * 12)
    const title = []
    const text = []
    for (let t = 0; t < nTitle; t += 1) {
      title.push(vocab[Math.floor(rand() * vocab.length)])
    }
    for (let t = 0; t < nText; t += 1) {
      text.push(vocab[Math.floor(rand() * vocab.length)])
    }
    if (unicodeEvery && d % unicodeEvery === 0) {
      text[Math.floor(rand() * text.length)] =
        UNICODE_TOKENS[Math.floor(rand() * UNICODE_TOKENS.length)]
    }
    const doc = {
      id: d + 1,
      title: title.join(' '),
      text: text.join(' '),
      category: d % 2 === 0 ? 'even' : 'odd',
    }
    if (d % 17 === 5) {
      delete doc.text
    }
    if (d % 23 === 7) {
      doc.title = ''
    }
    docs.push(doc)
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
 * Query mix: exact vocab tokens, typo'd tokens (fuzzy range), truncations
 * (prefix range), 2-token combos, repeated tokens, absent tokens, unicode,
 * punctuation-containing strings, and very short tokens.
 */
function generateQueries(seed, vocab, count) {
  const rand = rng(seed ^ 0x9e3779b9)
  const queries = []
  const randVocab = () => vocab[Math.floor(rand() * vocab.length)]

  while (queries.length < count) {
    const kind = queries.length % 10
    if (kind === 0) {
      queries.push(randVocab()) // exact token
    } else if (kind === 1 || kind === 2 || kind === 3) {
      queries.push(mutateToken(rand, randVocab())) // typo'd
    } else if (kind === 4) {
      const tok = randVocab()
      queries.push(tok.slice(0, 1 + Math.floor(rand() * Math.max(tok.length - 1, 1)))) // prefix
    } else if (kind === 5) {
      queries.push(`${randVocab()} ${randVocab()}`) // two tokens
    } else if (kind === 6) {
      queries.push(`${randVocab()}zzqx`) // likely absent
    } else if (kind === 7) {
      queries.push(UNICODE_TOKENS[Math.floor(rand() * UNICODE_TOKENS.length)])
    } else if (kind === 8) {
      const tok = randVocab()
      queries.push(`${tok} ${tok} ${randVocab()}`) // repeated token
    } else {
      queries.push(`${randVocab()}, ${randVocab()}.`) // punctuation in query
    }
  }
  return queries
}

/**
 * Compare two search result arrays: same document set, per-document scores
 * within 1e-9, identical terms/queryTerms/match/stored fields per document,
 * and consistent ranking. Ranking rule: whenever two results in the
 * reference list have scores differing by more than 1e-12, the higher-scored
 * document must also rank higher in the candidate list. Ties below that
 * epsilon are genuine float ties whose relative order depends on the
 * reference's internal operation order (its own scores for tied documents
 * routinely differ by 1-2 ulps from any reimplementation's), so their order
 * is not asserted — while every document's score must still agree within
 * the 1e-9 gate. Returns null when consistent, else a diagnostic string.
 */
function compareSearch(a, b) {
  if (a.length !== b.length) {
    return `result count differs: ${a.length} vs ${b.length}`
  }
  const TIE_EPS = 1e-12
  const byIdA = new Map(a.map((r) => [r.id, r]))
  const byIdB = new Map(b.map((r) => [r.id, r]))
  const rankA = new Map(a.map((r, i) => [r.id, i]))
  for (let i = 0; i < b.length; i += 1) {
    const ra = byIdA.get(b[i].id)
    if (ra === undefined) {
      return `id ${b[i].id} at rank ${i} missing from candidate results`
    }
    const diff = Math.abs(ra.score - b[i].score)
    if (!(diff <= 1e-9)) {
      return `score differs for id ${b[i].id}: ${ra.score} vs ${b[i].score}`
    }
    for (const key of ['terms', 'queryTerms']) {
      const aj = JSON.stringify(ra[key])
      const bj = JSON.stringify(b[i][key])
      if (aj !== bj) {
        return `${key} differs for id ${b[i].id}:\n  mojo: ${aj}\n  mini: ${bj}`
      }
    }
    const am = JSON.stringify(ra.match ?? null)
    const bm = JSON.stringify(b[i].match ?? null)
    if (am !== bm) {
      return `match differs for id ${b[i].id}:\n  mojo: ${am}\n  mini: ${bm}`
    }
    for (const key of Object.keys(ra)) {
      if (['id', 'score', 'terms', 'queryTerms', 'match'].includes(key)) continue
      const aj = JSON.stringify(ra[key])
      const bj = JSON.stringify(b[i][key])
      if (aj !== bj) {
        return `stored field ${key} differs for id ${b[i].id}: ${aj} vs ${bj}`
      }
    }
  }
  if (byIdA.size !== a.length || byIdB.size !== b.length) {
    return 'duplicate ids in results'
  }
  // Ranking consistency: every adjacent reference pair separated by more
  // than TIE_EPS must keep its relative order in the candidate list.
  for (let i = 0; i + 1 < b.length; i += 1) {
    if (b[i].score - b[i + 1].score > TIE_EPS && !(rankA.get(b[i].id) < rankA.get(b[i + 1].id))) {
      return `rank order inverted near rank ${i}: ref id ${b[i].id} (${b[i].score}) before id ${b[i + 1].id} (${b[i + 1].score}), candidate reverses them`
    }
  }
  return null
}

/**
 * Compare two autoSuggest result arrays: same suggestion set, identical
 * terms per suggestion, scores within 1e-9, and consistent ranking. As with
 * compareSearch, ranking is asserted for every adjacent reference pair whose
 * scores differ by more than 1e-12; nearer ties are genuine float ties (the
 * suggestion score is a mean of per-document scores, so 1-2 ulp of
 * accumulation-order noise can flip their order). Returns null when
 * consistent, else a diagnostic string.
 */
function compareSuggest(a, b) {
  if (a.length !== b.length) {
    return `suggestion count differs: ${a.length} vs ${b.length}\n  mojo: ${JSON.stringify(a.slice(0, 5))}\n  mini: ${JSON.stringify(b.slice(0, 5))}`
  }
  const TIE_EPS = 1e-12
  const byTextA = new Map(a.map((s) => [s.suggestion, s]))
  const rankA = new Map(a.map((s, i) => [s.suggestion, i]))
  for (let i = 0; i < b.length; i += 1) {
    const sa = byTextA.get(b[i].suggestion)
    if (sa === undefined) {
      return `suggestion missing from candidate at rank ${i}:\n  mini: ${JSON.stringify(b[i])}`
    }
    const aj = JSON.stringify(sa.terms)
    const bj = JSON.stringify(b[i].terms)
    if (aj !== bj) {
      return `suggestion terms differ for ${b[i].suggestion}:\n  mojo: ${aj}\n  mini: ${bj}`
    }
    const diff = Math.abs(sa.score - b[i].score)
    if (!(diff <= 1e-9)) {
      return `suggestion score differs for ${b[i].suggestion}: ${sa.score} vs ${b[i].score}`
    }
  }
  for (let i = 0; i + 1 < b.length; i += 1) {
    if (b[i].score - b[i + 1].score > TIE_EPS && !(rankA.get(b[i].suggestion) < rankA.get(b[i + 1].suggestion))) {
      return `suggestion rank inverted near rank ${i}: ${b[i].suggestion} (${b[i].score}) before ${b[i + 1].suggestion} (${b[i + 1].score}), candidate reverses them`
    }
  }
  return null
}

module.exports = {
  rng,
  generateVocab,
  generateCorpus,
  generateQueries,
  mutateToken,
  compareSearch,
  compareSuggest,
  UNICODE_TOKENS,
}
