'use strict'

/**
 * Index construction and query orchestration for the native backend.
 *
 * The engine replicates the reference's observable indexing semantics:
 * per-field tokenization (Unicode space/punctuation split) and lowercasing,
 * per-(term, doc, field) term frequencies, per-(doc, field) unique-term
 * counts, document-level term document-frequency, per-field average lengths
 * (over documents that contributed terms), and a vocabulary in first
 * insertion order (documents in add order, fields in declaration order,
 * tokens in text order).
 *
 * Each query resolves every distinct query term into its variant set (exact,
 * then prefix matches in reverse-insertion order, then fuzzy matches in
 * insertion order — prefix taking precedence over fuzzy on overlap), hands
 * the variants to the native kernel for BM25 accumulation, and assembles
 * results with the reference's result shapes, score combination
 * (sum of weighted term scores times the number of matched distinct query
 * terms), and ordering (score descending, ties stable in insertion order).
 */

const {
  tokenizeRaw,
  tokenizeAndProcess,
  processTermDefault,
  resolveMaxEdits,
  resolvePrefix,
  fuzzyWeight,
  prefixWeight,
  popcount32,
  toU16,
  UnsupportedOptionError,
} = require('./options.cjs')
const { NativeIndex, NativeUnavailable } = require('./native.cjs')

const MAX_DISTINCT_QUERY_TERMS = 32 // match bitmask is Int32
const MAX_RAW_QUERY_TERMS = 64

class EngineBuildError extends Error {
  constructor(message) {
    super(message)
    this.name = 'EngineBuildError'
    this.code = 'MINISEARCH_MOJO_ENGINE_BUILD'
  }
}

/** Per-field idf, replicating the reference's observed BM25 idf exactly
 * (computed in JavaScript so Math.log matches the reference bit-for-bit). */
function idfOf(docCount, df) {
  return 1.5 * Math.log(1 + (docCount - df + 0.5) / (df + 0.5))
}

/**
 * Compressed radix tree replicating the reference vocabulary's observable
 * traversal order. Node maps mix edge entries with a special value entry
 * (inserted when the term ending at that node was added; a split deletes the
 * old edge and re-inserts the shortened one, moving it to the end of the
 * map). A reverse-map-order DFS of this tree, with the start node's value
 * pulled to the front, yields exactly the reference's prefix-variant order;
 * the same DFS in forward map order yields its fuzzy-variant order.
 */
const TREE_VALUE = Symbol('tree-value')

class RadixNode {
  constructor() {
    this.edges = new Map() // label string -> RadixNode, TREE_VALUE -> term string
  }

  get value() {
    return this.edges.get(TREE_VALUE)
  }

  set value(term) {
    this.edges.set(TREE_VALUE, term)
  }
}

function radixInsert(root, word) {
  let node = root
  let i = 0
  while (i < word.length) {
    let found = null
    let foundLabel = null
    for (const [label, child] of node.edges) {
      if (label !== TREE_VALUE && label[0] === word[i]) {
        found = child
        foundLabel = label
        break
      }
    }
    if (!found) {
      const child = new RadixNode()
      child.value = word
      node.edges.set(word.slice(i), child)
      return
    }
    let k = 0
    while (k < foundLabel.length && i + k < word.length && foundLabel[k] === word[i + k]) {
      k += 1
    }
    if (k === foundLabel.length) {
      node = found
      i += k
      if (i === word.length) {
        node.value = word
        return
      }
    } else {
      const mid = new RadixNode()
      node.edges.delete(foundLabel)
      node.edges.set(foundLabel.slice(0, k), mid)
      mid.edges.set(foundLabel.slice(k), found)
      if (i + k === word.length) {
        mid.value = word
      } else {
        const child = new RadixNode()
        child.value = word
        mid.edges.set(word.slice(i + k), child)
      }
      return
    }
  }
}

/** Terms of the subtree in reverse-map-order DFS sequence (prefix order). */
function reverseMapDfs(root) {
  const out = []
  const walk = (node) => {
    const entries = [...node.edges.entries()]
    for (let e = entries.length - 1; e >= 0; e -= 1) {
      const [label, child] = entries[e]
      if (label === TREE_VALUE) {
        out.push(child)
      } else {
        walk(child)
      }
    }
  }
  walk(root)
  return out
}

/**
 * Inverted index with reference-compatible statistics, plus a lazily built
 * native kernel handle (rebuilt whenever documents were added since the
 * last query).
 */
class Engine {
  constructor(options) {
    this.fields = options.fields
    this.storeFields = options.storeFields || []
    this.idField = options.idField || 'id'
    this.nFields = this.fields.length
    this.fieldIndex = new Map(this.fields.map((f, i) => [f, i]))

    this.vocab = [] // term string -> first-insertion order
    this.termId = new Map()
    // Per-(term, field) document frequency, flattened [nTerms * nFields + f]:
    // the reference's idf uses the term's document frequency in the MATCHED
    // field, not across the whole document.
    this.termFieldDf = []
    // Per-term posting lists, appended in (doc, field) sorted order.
    this.postDoc = []
    this.postField = []
    this.postTf = []
    // Per-doc structures.
    this.ids = []
    this.docIdToIdx = new Map()
    this.stored = []
    this.docfieldLen = [] // nDocs * nFields, unique-term counts
    // Per-field average-length state: every added document increments the
    // count; a document contributes its unique-term count when it produced
    // terms for the field, and the CURRENT running average otherwise (this
    // replicates the reference's observed average-length semantics exactly).
    this.fieldLenSum = new Array(this.nFields).fill(0)
    this.fieldDocCount = new Array(this.nFields).fill(0)

    this._nativeIndex = null
    this._dirty = false
  }

  get nDocs() {
    return this.ids.length
  }

  addAll(docs) {
    for (const doc of docs) {
      this.add(doc)
    }
  }

  add(doc) {
    if (doc === null || typeof doc !== 'object') {
      throw new EngineBuildError('document must be an object')
    }
    const id = doc[this.idField]
    if (id === undefined || id === null) {
      throw new EngineBuildError(`document is missing the id field "${this.idField}"`)
    }
    if (this.docIdToIdx.has(id)) {
      throw new EngineBuildError(`duplicate document id: ${String(id)}`)
    }
    const docIdx = this.nDocs
    this.docIdToIdx.set(id, docIdx)
    this.ids.push(id)

    const storedDoc = {}
    for (const field of this.storeFields) {
      storedDoc[field] = doc[field]
    }
    this.stored.push(storedDoc)

    for (let f = 0; f < this.nFields; f += 1) {
      const value = doc[this.fields[f]]
      // Present fields tokenize including edge-empty tokens: the reference's
      // field length counts the empty term produced by leading/trailing
      // separator runs (it has no postings and never matches queries, but it
      // inflates the document field length). Absent fields contribute the
      // running average to the field-length average instead.
      const rawTokens =
        value === undefined || value === null ? null : tokenizeRaw(String(value))
      const tokens = rawTokens === null ? null : rawTokens.map(processTermDefault)
      const tfMap = new Map()
      let uniqueWithEmpty = 0
      if (tokens !== null) {
        // The document field length counts unique RAW tokens (before
        // lowercasing — 'The' and 'the' are two distinct entries there),
        // including the empty token from leading/trailing separators.
        const uniqueSet = new Set(rawTokens)
        for (const term of tokens) {
          if (term.length === 0) {
            continue // the empty term affects field length only, never postings
          }
          let id2 = this.termId.get(term)
          if (id2 === undefined) {
            id2 = this.vocab.length
            this.vocab.push(term)
            this.termId.set(term, id2)
            this.termFieldDf.push(new Array(this.nFields).fill(0))
            this.postDoc.push([])
            this.postField.push([])
            this.postTf.push([])
          }
          tfMap.set(id2, (tfMap.get(id2) || 0) + 1)
        }
        uniqueWithEmpty = uniqueSet.size
      }
      this.docfieldLen.push(uniqueWithEmpty)
      for (const [termId2, tf] of tfMap) {
        this.postDoc[termId2].push(docIdx)
        this.postField[termId2].push(f)
        this.postTf[termId2].push(tf)
        this.termFieldDf[termId2][f] += 1
      }
      // Contribution to the field length average is read BEFORE this
      // document is counted (an absent field inherits the running average).
      const lenContribution = tokens !== null ? uniqueWithEmpty : this.avgdl(f)
      this.fieldDocCount[f] += 1
      this.fieldLenSum[f] += lenContribution
    }
    this._dirty = true
  }

  avgdl(fieldIdx) {
    const count = this.fieldDocCount[fieldIdx]
    return count === 0 ? 0 : this.fieldLenSum[fieldIdx] / count
  }

  /** Flatten the index into kernel buffers and (re)build the native handle.
   *  The kernel vocabulary is stored in reverse-map-order radix DFS sequence
   *  so that the kernel's scan orders reproduce the reference's variant
   *  enumeration orders exactly (prefix: forward scan; fuzzy: backward). */
  _buildNative() {
    if (!this._dirty && this._nativeIndex) {
      return this._nativeIndex
    }
    this._destroyNative()
    const nTerms = this.vocab.length

    // Radix tree over the insertion-ordered vocabulary -> kernel term order.
    const root = new RadixNode()
    for (const term of this.vocab) {
      radixInsert(root, term)
    }
    const kernelVocab = reverseMapDfs(root)
    const kernelId = new Map()
    for (let k = 0; k < kernelVocab.length; k += 1) {
      kernelId.set(kernelVocab[k], k)
    }
    this.kernelVocab = kernelVocab
    this.kernelId = kernelId

    let totalChars = 0
    for (const term of kernelVocab) {
      totalChars += term.length
    }
    if (totalChars > 0x7fffffff) {
      throw new EngineBuildError(
        `vocabulary too large for the native kernel: ${totalChars} UTF-16 code units (> 2^31)`
      )
    }
    const vocabChars = new Uint16Array(totalChars)
    const vocabOffsets = new Int32Array(nTerms + 1)
    let off = 0
    for (let t = 0; t < nTerms; t += 1) {
      const term = kernelVocab[t]
      for (let k = 0; k < term.length; k += 1) {
        vocabChars[off + k] = term.charCodeAt(k)
      }
      off += term.length
      vocabOffsets[t + 1] = off
    }

    let totalPostings = 0
    for (let t = 0; t < nTerms; t += 1) {
      totalPostings += this.postDoc[t].length
    }
    const postOffsets = new Int32Array(nTerms + 1)
    const postDoc = new Int32Array(totalPostings)
    const postField = new Int32Array(totalPostings)
    const postTf = new Int32Array(totalPostings)
    let p = 0
    for (let t = 0; t < nTerms; t += 1) {
      const insertionId = this.termId.get(kernelVocab[t])
      const docsT = this.postDoc[insertionId]
      for (let k = 0; k < docsT.length; k += 1) {
        postDoc[p] = docsT[k]
        postField[p] = this.postField[insertionId][k]
        postTf[p] = this.postTf[insertionId][k]
        p += 1
      }
      postOffsets[t + 1] = p
    }

    const avgdlArr = new Float64Array(this.nFields)
    for (let f = 0; f < this.nFields; f += 1) {
      avgdlArr[f] = this.avgdl(f)
    }

    this._nativeIndex = new NativeIndex({
      vocabChars,
      vocabOffsets,
      nTerms,
      postOffsets,
      postDoc,
      postField,
      postTf,
      docfieldLen: Int32Array.from(this.docfieldLen),
      avgdl: avgdlArr,
      nDocs: this.nDocs,
      nFields: this.nFields,
    })
    this._dirty = false
    return this._nativeIndex
  }

  _destroyNative() {
    if (this._nativeIndex) {
      this._nativeIndex.destroy()
      this._nativeIndex = null
    }
  }

  destroy() {
    this._destroyNative()
  }

  /**
   * Resolve one query into per-query-term variant lists and kernel scoring
   * plans. `suggestMode` applies autoSuggest's expansion rules: prefix only
   * on the last query term unless the caller passed an explicit prefix
   * option. Returns null when the query has no terms.
   */
  _planQuery(nativeIndex, queryTermsRaw, opts, suggestMode) {
    const nRaw = queryTermsRaw.length
    if (nRaw === 0) {
      return null
    }
    if (nRaw > MAX_RAW_QUERY_TERMS) {
      throw new UnsupportedOptionError(`more than ${MAX_RAW_QUERY_TERMS} query terms`)
    }
    const distinct = []
    const rawToDistinct = []
    const distinctIndex = new Map()
    for (const term of queryTermsRaw) {
      let di = distinctIndex.get(term)
      if (di === undefined) {
        di = distinct.length
        distinct.push(term)
        distinctIndex.set(term, di)
      }
      rawToDistinct.push(di)
    }
    const nDistinct = distinct.length
    if (nDistinct > MAX_DISTINCT_QUERY_TERMS) {
      throw new UnsupportedOptionError(
        `more than ${MAX_DISTINCT_QUERY_TERMS} distinct query terms`
      )
    }

    // Expansion flags per distinct query term.
    const lastTerm = queryTermsRaw[nRaw - 1]
    const explicitPrefix = opts.prefix !== undefined
    const perTerm = distinct.map((term) => {
      const maxEdits = resolveMaxEdits(opts.fuzzy, term)
      let prefixOn
      if (suggestMode && !explicitPrefix) {
        prefixOn = term === lastTerm // autoSuggest default: prefix the last term only
      } else {
        prefixOn = resolvePrefix(opts.prefix, term)
      }
      return { term, maxEdits, prefixOn }
    })

    // Resolve variants per distinct query term, in canonical order:
    // exact, prefix (reverse-map radix DFS), fuzzy-only (forward-map DFS).
    const variantLists = perTerm.map(({ term, maxEdits, prefixOn }) => {
      const variants = []
      const seen = new Set()
      const exactId = this.kernelId.get(term)
      if (exactId !== undefined) {
        variants.push({ id: exactId, weight: 1 })
        seen.add(exactId)
      }
      if (prefixOn) {
        const { count, ids } = nativeIndex.prefixScan(toU16(term))
        for (let k = 0; k < count; k += 1) {
          const id = ids[k]
          if (seen.has(id)) continue
          seen.add(id)
          variants.push({ id, weight: prefixWeight(term.length, this.kernelVocab[id].length) })
        }
      }
      if (maxEdits > 0) {
        const { count, ids, dist } = nativeIndex.fuzzyScan(toU16(term), maxEdits)
        for (let k = 0; k < count; k += 1) {
          const id = ids[k]
          const d = dist[k]
          if (d === 0 || seen.has(id)) continue
          seen.add(id)
          variants.push({ id, weight: fuzzyWeight(this.kernelVocab[id].length, d) })
        }
      }
      return variants
    })

    // Kernel scoring plan: one CSR slice per RAW query-term occurrence.
    const qvOffsets = new Int32Array(nRaw + 1)
    let totalVariants = 0
    for (let q = 0; q < nRaw; q += 1) {
      totalVariants += variantLists[rawToDistinct[q]].length
      qvOffsets[q + 1] = totalVariants
    }
    const qvIds = new Int32Array(totalVariants)
    const qvWeight = new Float64Array(totalVariants)
    const qvIdf = new Float64Array(totalVariants * this.nFields)
    const qvQt = new Int32Array(nRaw)
    let recordCap = 0
    {
      let v = 0
      for (let q = 0; q < nRaw; q += 1) {
        const di = rawToDistinct[q]
        qvQt[q] = di
        for (const variant of variantLists[di]) {
          qvIds[v] = variant.id
          qvWeight[v] = variant.weight
          const dfRow = this.termFieldDf[this.termId.get(this.kernelVocab[variant.id])]
          for (let f = 0; f < this.nFields; f += 1) {
            qvIdf[v * this.nFields + f] = idfOf(this.nDocs, dfRow[f])
          }
          recordCap += dfRow.reduce((a, b) => a + b, 0)
          v += 1
        }
      }
    }

    // Field boost vector + allowed-fields mask.
    const fieldBoost = new Float64Array(this.nFields).fill(1)
    let allowedFieldsMask = 0
    const boostSpec = opts.boost || {}
    const fieldsSpec = opts.fields
    for (let f = 0; f < this.nFields; f += 1) {
      const name = this.fields[f]
      if (Object.prototype.hasOwnProperty.call(boostSpec, name)) {
        fieldBoost[f] = boostSpec[name]
      }
      if (fieldsSpec === undefined || fieldsSpec.includes(name)) {
        allowedFieldsMask |= 1 << f
      }
    }

    return {
      distinct,
      nDistinct,
      rawToDistinct,
      variantLists,
      qvOffsets,
      qvIds,
      qvWeight,
      qvIdf,
      qvQt,
      nRawQ: nRaw,
      fieldBoost,
      allowedFieldsMask,
      recordCap,
    }
  }

  /**
   * Run the scoring kernel and finalize per-document results.
   * Returns an array of per-doc records in FIRST-ENCOUNTERED order (the
   * kernel emits records in (raw qt, variant, doc) order, which matches the
   * reference's result-collection order; a stable score-descending sort
   * later preserves it for ties):
   * { docIdx, score, matched: Map(distinctQt -> [{term, fieldMask}]) }.
   */
  _execute(nativeIndex, plan, combineWith) {
    const { scores, masks, records, nRecords } = nativeIndex.score(plan)
    const requireAll = combineWith === 'AND'

    // Group emitted records per doc (emission order is (raw qt, variant,
    // doc); within a distinct query term that is the canonical variant
    // order). First record per (qt, term) wins.
    const docMatches = new Map()
    for (let r = 0; r < nRecords; r += 1) {
      const base = r * 4
      const doc = records[base]
      const qt = records[base + 1]
      const term = records[base + 2]
      const fieldMask = records[base + 3]
      let perDoc = docMatches.get(doc)
      if (perDoc === undefined) {
        perDoc = new Map()
        docMatches.set(doc, perDoc)
      }
      let perQt = perDoc.get(qt)
      if (perQt === undefined) {
        perQt = []
        perDoc.set(qt, perQt)
      }
      let dup = false
      for (const rec of perQt) {
        if (rec.term === term) {
          dup = true
          break
        }
      }
      if (!dup) {
        perQt.push({ term, fieldMask })
      }
    }

    const out = []
    for (const [doc, perDoc] of docMatches) {
      const mask = masks[doc]
      const matchedCount = popcount32(mask)
      if (matchedCount === 0) continue
      if (requireAll && matchedCount !== plan.nDistinct) continue
      const score = scores[doc] * matchedCount
      out.push({ docIdx: doc, score, matched: perDoc, matchedMask: mask })
    }
    return out
  }

  /** Ordered (terms, match) pair for one scored document. */
  _termsAndMatch(plan, perDoc) {
    const terms = []
    const match = {}
    const seenTerms = new Set()
    for (let qt = 0; qt < plan.nDistinct; qt += 1) {
      const perQt = perDoc.get(qt)
      if (perQt === undefined) continue
      for (const { term, fieldMask } of perQt) {
        if (seenTerms.has(term)) continue
        seenTerms.add(term)
        const termStr = this.kernelVocab[term]
        terms.push(termStr)
        const fieldNames = []
        for (let f = 0; f < this.nFields; f += 1) {
          if (fieldMask & (1 << f)) {
            fieldNames.push(this.fields[f])
          }
        }
        match[termStr] = fieldNames
      }
    }
    return { terms, match }
  }

  search(nativeIndex, query, opts) {
    const queryTermsRaw = tokenizeAndProcess(query)
    const plan = this._planQuery(nativeIndex, queryTermsRaw, opts, false)
    if (plan === null) {
      return []
    }
    const combineWith = opts.combineWith || 'OR'
    const docs = this._execute(nativeIndex, plan, combineWith)
    const results = docs.map(({ docIdx, score, matched, matchedMask }) => {
      const { terms, match } = this._termsAndMatch(plan, matched)
      const queryTerms = plan.distinct.filter((_, qt) => matchedMask & (1 << qt))
      return {
        id: this.ids[docIdx],
        ...this.stored[docIdx],
        score,
        terms,
        queryTerms,
        match,
      }
    })
    // Score descending; Array.prototype.sort is stable, so ties keep
    // ascending document (insertion) order, exactly like the reference.
    results.sort((a, b) => b.score - a.score)
    return results
  }

  autoSuggest(nativeIndex, query, opts) {
    const queryTermsRaw = tokenizeAndProcess(query)
    const plan = this._planQuery(nativeIndex, queryTermsRaw, opts, true)
    if (plan === null) {
      return []
    }
    const combineWith = opts.combineWith || 'AND'
    const docs = this._execute(nativeIndex, plan, combineWith)
    // One suggestion per document: its matched terms in canonical order.
    // Documents generating the same suggestion string are grouped, and the
    // suggestion's score is the MEAN of the generating documents' query
    // scores (the reference's observed behavior).
    const byText = new Map()
    for (const { docIdx, score, matched } of docs) {
      const terms = []
      const seenTerms = new Set()
      for (let qt = 0; qt < plan.nDistinct; qt += 1) {
        const perQt = matched.get(qt)
        if (perQt === undefined) continue
        for (const { term } of perQt) {
          if (seenTerms.has(term)) continue
          seenTerms.add(term)
          terms.push(this.kernelVocab[term])
        }
      }
      if (terms.length === 0) continue
      const text = terms.join(' ')
      const existing = byText.get(text)
      if (existing === undefined) {
        byText.set(text, { suggestion: text, terms, scoreSum: score, count: 1 })
      } else {
        existing.scoreSum += score
        existing.count += 1
      }
    }
    const suggestions = [...byText.values()].map(({ suggestion, terms, scoreSum, count }) => ({
      suggestion,
      terms,
      score: scoreSum / count,
    }))
    // Score descending; Array.prototype.sort is stable, so ties keep
    // first-generation order, exactly like the reference.
    suggestions.sort((a, b) => b.score - a.score)
    return suggestions
  }
}

module.exports = { Engine, EngineBuildError, NativeUnavailable }
