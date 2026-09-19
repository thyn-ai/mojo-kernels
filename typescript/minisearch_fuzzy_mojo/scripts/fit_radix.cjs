'use strict'
// Fit a radix-tree construction + DFS model against the oracle's observed
// prefix traversal order on the real seeded vocabulary.
const MiniSearchRef = require('minisearch')
const { generateVocab, generateCorpus } = require('../tests/helpers.cjs')

const vocab = generateVocab(1337, 1200)
const corpus = generateCorpus(1337, 600, vocab, { unicodeEvery: 9 })

// True insertion sequence (my engine's vocab order).
const { Engine } = require('../packages/core/src/engine.cjs')
const e = new Engine({ fields: ['title', 'text'] })
e.addAll(corpus)
const insertionOrder = e.vocab.slice()
e.destroy()

// Oracle's prefix order for prefix 'a' and 'bel'.
function oraclePrefixOrder(prefix) {
  const ref = new MiniSearchRef({ fields: ['title', 'text'] })
  ref.addAll(corpus)
  const seen = []
  const seenSet = new Set()
  for (const r of ref.search(prefix, { prefix: true })) {
    for (const t of r.terms) {
      if (!seenSet.has(t)) {
        seenSet.add(t)
        seen.push(t)
      }
    }
  }
  return seen
}

// --- radix tree simulator ---
class Node {
  constructor() {
    this.value = undefined
    this.edges = new Map() // label -> Node, in insertion order
  }
}

let SPLIT_KEEP_POSITION = false
let DFS_REVERSE = true

function insert(root, word) {
  let node = root
  let i = 0
  while (i < word.length) {
    let found = null
    let foundLabel = null
    for (const [label, child] of node.edges) {
      if (label[0] === word[i]) {
        found = child
        foundLabel = label
        break
      }
    }
    if (!found) {
      const child = new Node()
      child.value = word
      node.edges.set(word.slice(i), child)
      return
    }
    // common prefix length of word[i:] and foundLabel
    let k = 0
    while (k < foundLabel.length && i + k < word.length && foundLabel[k] === word[i + k]) {
      k += 1
    }
    if (k === foundLabel.length) {
      // edge fully consumed; descend
      node = found
      i += k
      if (i === word.length) {
        node.value = word
        return
      }
    } else {
      // split: replace edge `foundLabel` with edge foundLabel[:k] to a new
      // internal node holding the old subtree (foundLabel[k:]) and the new
      // word's remainder.
      const mid = new Node()
      const rest = foundLabel.slice(k)
      if (SPLIT_KEEP_POSITION) {
        // rebuild the map preserving the old edge's position
        const next = new Map()
        for (const [label, child] of node.edges) {
          if (label === foundLabel) next.set(foundLabel.slice(0, k), mid)
          else next.set(label, child)
        }
        node.edges = next
      } else {
        node.edges.delete(foundLabel)
        node.edges.set(foundLabel.slice(0, k), mid)
      }
      mid.edges.set(rest, found)
      if (i + k === word.length) {
        mid.value = word
      } else {
        const child = new Node()
        child.value = word
        mid.edges.set(word.slice(i + k), child)
      }
      return
    }
  }
}

function dfs(node, prefix, out) {
  if (node.value !== undefined) {
    out.push(node.value)
  }
  const entries = [...node.edges.entries()]
  if (DFS_REVERSE) entries.reverse()
  for (const [label, child] of entries) {
    dfs(child, prefix + label, out)
  }
}

function simulate(words, queryPrefix) {
  const root = new Node()
  for (const w of words) insert(root, w)
  // descend to the prefix node
  let node = root
  let i = 0
  while (i < queryPrefix.length) {
    let found = null
    let foundLabel = null
    for (const [label, child] of node.edges) {
      if (label[0] === queryPrefix[i]) {
        found = child
        foundLabel = label
        break
      }
    }
    if (!found) return []
    let k = 0
    while (k < foundLabel.length && i + k < queryPrefix.length && foundLabel[k] === queryPrefix[i + k]) {
      k += 1
    }
    if (k < foundLabel.length && i + k < queryPrefix.length) {
      // prefix ends mid-edge: the subtree is `found` (terms under it match)
      node = found
      i = queryPrefix.length
      break
    }
    node = found
    i += k
  }
  const out = []
  dfs(node, queryPrefix, out)
  return out
}

const expectedA = oraclePrefixOrder('a').filter((t) => t.startsWith('a'))
const expectedBel = oraclePrefixOrder('bel').filter((t) => t.startsWith('bel'))
for (const reverse of [true, false]) {
  for (const keepPos of [true, false]) {
    DFS_REVERSE = reverse
    SPLIT_KEEP_POSITION = keepPos
    const gotA = simulate(insertionOrder, 'a')
    const gotBel = simulate(insertionOrder, 'bel')
    const mA = JSON.stringify(expectedA) === JSON.stringify(gotA)
    const mB = JSON.stringify(expectedBel) === JSON.stringify(gotBel)
    console.log(`reverse=${reverse} keepPos=${keepPos}: a=${mA} bel=${mB}`)
    if (!mA) {
      for (let i = 0; i < expectedA.length; i += 1) {
        if (expectedA[i] !== gotA[i]) {
          console.log(`  a diverges at ${i}: exp ${expectedA[i]} vs got ${gotA[i]}`)
          break
        }
      }
    }
    if (!mB) {
      for (let i = 0; i < expectedBel.length; i += 1) {
        if (expectedBel[i] !== gotBel[i]) {
          console.log(`  bel diverges at ${i}: exp ${expectedBel[i]} vs got ${gotBel[i]}`)
          break
        }
      }
    }
  }
}
