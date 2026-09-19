'use strict'

/**
 * Index construction and search orchestration for the native backend.
 *
 * The engine extracts every searchable string from the document collection
 * (replicating the reference's index records, including the reversed order of
 * array-valued sub-records), lowercases them once at build time, and hands a
 * flat UTF-16 buffer to the native kernel. Each search call then maps to one
 * kernel call per pattern chunk and combines per-text results exactly like
 * the reference's BitapSearch/Fuse pipeline.
 */

const { fuseGet, isBlank, createNormGetter, defaultSortFn } = require('./options.cjs')

const MAX_BITS = 32
const INT32_MAX = 0x7fffffff

class EngineBuildError extends Error {
  constructor(message) {
    super(message)
    this.name = 'EngineBuildError'
    this.code = 'FUSE_MOJO_ENGINE_BUILD'
  }
}

/**
 * Split a (lowered) pattern into <=32-code-unit chunks, mirroring the
 * reference: full chunks of 32, with the final chunk of `remainder` units
 * taken from the tail so it always has length 32 when possible.
 */
function chunkPattern(pattern) {
  const len = pattern.length
  const chunks = []
  if (len > MAX_BITS) {
    let i = 0
    const remainder = len % MAX_BITS
    const end = len - remainder
    while (i < end) {
      chunks.push({ text: pattern.slice(i, i + MAX_BITS), startIndex: i })
      i += MAX_BITS
    }
    if (remainder) {
      const startIndex = len - MAX_BITS
      chunks.push({ text: pattern.slice(startIndex), startIndex })
    }
  } else {
    chunks.push({ text: pattern, startIndex: 0 })
  }
  return chunks
}

function toU16(str) {
  const out = new Uint16Array(str.length)
  for (let i = 0; i < str.length; i += 1) {
    out[i] = str.charCodeAt(i)
  }
  return out
}

/**
 * Build the searchable-text engine. Every entry corresponds to one string the
 * reference would run Bitap on, in the reference's iteration order:
 *   string list: one entry per non-blank document
 *   object list: per document, per configured key, one entry per value
 *                (array values contribute one entry per element, in the
 *                reversed order produced by the reference's stack DFS)
 */
function buildEngine(docs, keyStore, options) {
  const normOf = createNormGetter(options.fieldNormWeight, 3)
  const nDocs = docs && typeof docs.length === 'number' ? docs.length : 0
  const mode = typeof docs[0] === 'string' ? 'string' : 'object'
  const entries = []

  if (mode === 'string') {
    for (let i = 0; i < nDocs; i += 1) {
      const doc = docs[i]
      if (doc === undefined || doc === null) {
        continue
      }
      if (isBlank(doc)) {
        continue
      }
      entries.push({ docIdx: i, keyIdx: -1, arrIdx: -1, value: doc, norm: normOf(doc) })
    }
  } else {
    for (let i = 0; i < nDocs; i += 1) {
      const doc = docs[i]
      for (let k = 0; k < keyStore.length; k += 1) {
        const key = keyStore[k]
        const value = fuseGet(doc, key.path)
        if (value === undefined || value === null) {
          continue
        }
        if (Array.isArray(value)) {
          // Reference _addObject: stack-based DFS pops later elements first,
          // so sub-records come out reversed relative to document order.
          const stack = [{ nestedArrIndex: -1, value }]
          while (stack.length) {
            const { nestedArrIndex, value: v } = stack.pop()
            if (v === undefined || v === null) {
              continue
            }
            if (typeof v === 'string' && !isBlank(v)) {
              entries.push({ docIdx: i, keyIdx: k, arrIdx: nestedArrIndex, value: v, norm: normOf(v) })
            } else if (Array.isArray(v)) {
              for (let kk = 0; kk < v.length; kk += 1) {
                stack.push({ nestedArrIndex: kk, value: v[kk] })
              }
            }
            // Non-string leaves are silently ignored, as in the reference.
          }
        } else if (typeof value === 'string' && !isBlank(value)) {
          entries.push({ docIdx: i, keyIdx: k, arrIdx: -1, value, norm: normOf(value) })
        }
      }
    }
  }

  // Flatten the lowered texts into one UTF-16 buffer.
  const loweredStrings = new Array(entries.length)
  let total = 0
  for (let e = 0; e < entries.length; e += 1) {
    const s = options.isCaseSensitive ? entries[e].value : entries[e].value.toLowerCase()
    loweredStrings[e] = s
    total += s.length
  }
  if (total > INT32_MAX) {
    throw new EngineBuildError(
      `collection too large for the native kernel: ${total} UTF-16 code units (> 2^31)`
    )
  }
  const chars = new Uint16Array(total)
  const offsets = new Int32Array(entries.length + 1)
  let off = 0
  for (let e = 0; e < entries.length; e += 1) {
    const s = loweredStrings[e]
    for (let k = 0; k < s.length; k += 1) {
      chars[off + k] = s.charCodeAt(k)
    }
    off += s.length
    offsets[e + 1] = off
  }

  return { mode, entries, loweredStrings, chars, offsets, docs, keyStore }
}

/**
 * Run a search on the native backend and format results identically to the
 * reference `Fuse#search`.
 */
function nativeSearch(engine, nativeIndex, pattern, limit, options) {
  const { entries, loweredStrings, docs, keyStore } = engine
  const nEntries = entries.length

  const loweredPattern = options.isCaseSensitive ? pattern : pattern.toLowerCase()
  if (!loweredPattern.length) {
    return []
  }
  const chunks = chunkPattern(loweredPattern)
  const singleChunk = chunks.length === 1
  const includeMatches = options.includeMatches

  const totalScore = new Float64Array(nEntries)
  const hasMatch = new Uint8Array(nEntries)
  const indicesByEntry = includeMatches ? new Array(nEntries).fill(null) : null

  for (const chunk of chunks) {
    const chunkU16 = toU16(chunk.text)
    const { totalPairs, scores, isMatch, idxOffsets } = nativeIndex.searchChunk(
      chunkU16,
      chunk.startIndex,
      singleChunk
    )
    let pairs = null
    if (includeMatches && totalPairs > 0) {
      pairs = nativeIndex.copyIndices(totalPairs)
    }
    for (let e = 0; e < nEntries; e += 1) {
      // The reference folds every chunk's (clamped) score into the average,
      // even for chunks that did not match.
      totalScore[e] += scores[e]
      if (isMatch[e]) {
        hasMatch[e] = 1
        if (includeMatches) {
          const from = idxOffsets[e]
          const to = idxOffsets[e + 1]
          if (to > from) {
            let arr = indicesByEntry[e]
            if (arr === null) {
              arr = []
              indicesByEntry[e] = arr
            }
            for (let p = from; p < to; p += 1) {
              arr.push([pairs[p * 2], pairs[p * 2 + 1]])
            }
          }
        }
      }
    }
  }

  // Whole-pattern exact-equality fast path. For single-chunk patterns the
  // kernel already applied it; for multi-chunk patterns it is replicated here
  // (the reference checks pattern === text before running any chunk).
  if (!singleChunk) {
    for (let e = 0; e < nEntries; e += 1) {
      if (loweredStrings[e] === loweredPattern) {
        hasMatch[e] = 1
        totalScore[e] = 0
        if (includeMatches) {
          indicesByEntry[e] = [[0, loweredStrings[e].length - 1]]
        }
      }
    }
  }

  // Assemble per-document results with reference match-record shapes.
  const results = []
  if (engine.mode === 'string') {
    for (let e = 0; e < nEntries; e += 1) {
      if (!hasMatch[e]) continue
      const entry = entries[e]
      results.push({
        idx: entry.docIdx,
        matches: [
          {
            score: totalScore[e] / chunks.length,
            value: entry.value,
            norm: entry.norm,
            indices: includeMatches ? indicesByEntry[e] || [] : undefined,
          },
        ],
      })
    }
  } else {
    let current = null
    let currentDoc = -1
    for (let e = 0; e < nEntries; e += 1) {
      if (!hasMatch[e]) continue
      const entry = entries[e]
      const match = {
        score: totalScore[e] / chunks.length,
        key: keyStore[entry.keyIdx],
        value: entry.value,
        norm: entry.norm,
        indices: includeMatches ? indicesByEntry[e] || [] : undefined,
      }
      if (entry.arrIdx > -1) {
        match.idx = entry.arrIdx
      }
      if (entry.docIdx !== currentDoc) {
        current = { idx: entry.docIdx, matches: [] }
        results.push(current)
        currentDoc = entry.docIdx
      }
      current.matches.push(match)
    }
  }

  // Practical scoring: product over matches of score^(weight * norm).
  for (const result of results) {
    let score = 1
    for (const match of result.matches) {
      const weight = match.key ? match.key.weight : null
      score *= Math.pow(
        match.score === 0 && weight ? Number.EPSILON : match.score,
        (weight || 1) * (options.ignoreFieldNorm ? 1 : match.norm)
      )
    }
    result.score = score
  }

  if (options.shouldSort) {
    results.sort(defaultSortFn)
  }

  if (typeof limit === 'number' && limit > -1) {
    return formatResults(results.slice(0, limit), docs, options)
  }
  return formatResults(results, docs, options)
}

function formatResults(results, docs, options) {
  const { includeMatches, includeScore } = options
  return results.map((result) => {
    const data = {
      item: docs[result.idx],
      refIndex: result.idx,
    }
    if (includeMatches) {
      data.matches = []
      for (const match of result.matches) {
        if (!match.indices || !match.indices.length) {
          continue
        }
        const obj = { indices: match.indices, value: match.value }
        if (match.key) {
          obj.key = match.key.src
        }
        if (match.idx > -1) {
          obj.refIndex = match.idx
        }
        data.matches.push(obj)
      }
    }
    if (includeScore) {
      data.score = result.score
    }
    return data
  })
}

module.exports = { buildEngine, nativeSearch, chunkPattern, EngineBuildError }
