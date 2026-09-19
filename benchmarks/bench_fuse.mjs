#!/usr/bin/env node
/**
 * Reproducible benchmark: Fuse.js 7.1.0 vs fuse-mojo (native Mojo kernel).
 *
 * Corpora are generated locally from fixed seeds: 10k / 50k / 100k documents
 * of 5-15 word-like tokens from a ~3000-word vocabulary (plus a unicode token
 * in every 11th document). Patterns of length 3 / 8 / 16 code units are drawn
 * from corpus tokens, half with 1-2 seeded typos. Median of 5 runs after a
 * full warmup round. A correctness gate (identical refIndex order, scores
 * within 1e-9) runs before any timing.
 *
 * Deps resolve through the typescript/fuse-mojo workspace — run
 * `npm install` there first, then: `node benchmarks/bench_fuse.mjs`
 * (or `pixi run bench-fuse`).
 */
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import os from 'node:os'

const require = createRequire(new URL('../typescript/fuse-mojo/package.json', import.meta.url))
const ReferenceFuse = require('fuse.js')
const FuseMojo = require('./packages/core/index.cjs')
const { generateCorpus, rng } = require('./tests/helpers.cjs')

const CORPUS_SIZES = [10_000, 50_000, 100_000]
const PATTERN_LENGTHS = [3, 8, 16]
const PATTERNS_PER_LENGTH = 15
const RUNS = 5
const OPTIONS = { includeScore: true } // the common scoring path

function makePatterns(seed, docs, length, count) {
  const rand = rng(seed ^ (length * 0x9e37))
  const out = []
  while (out.length < count) {
    const doc = docs[Math.floor(rand() * docs.length)]
    const tokens = doc.split(' ')
    let pat = tokens[Math.floor(rand() * tokens.length)]
    while (pat.length < length) {
      pat += tokens[Math.floor(rand() * tokens.length)]
    }
    pat = pat.slice(0, length)
    if (out.length % 2 === 1) {
      // 1-2 seeded typos
      const chars = 'abcdefghijklmnopqrstuvwxyz'
      const nMut = 1 + Math.floor(rand() * 2)
      const arr = [...pat]
      for (let m = 0; m < nMut; m += 1) {
        const pos = Math.floor(rand() * arr.length)
        arr[pos] = chars[Math.floor(rand() * 26)]
      }
      pat = arr.join('')
    }
    out.push(pat)
  }
  return out
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function fmt(n, digits = 3) {
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

console.log('# fuse-mojo vs Fuse.js benchmark')
console.log(
  `env: ${os.cpus()[0]?.model ?? 'unknown CPU'} (${os.cpus().length} threads), ` +
    `${os.platform()} ${os.arch}, node ${process.version}, fuse.js ${ReferenceFuse.version}, ` +
    `fuse-mojo native: ${FuseMojo.nativeAvailable()} (threads: ${FuseMojo.backendInfo().threads}), ${new Date().toISOString().slice(0, 10)}`
)
if (!FuseMojo.nativeAvailable()) {
  console.error('ERROR: native kernel unavailable; refusing to benchmark the fallback.')
  process.exit(1)
}

const rows = []
const coldRows = []
const buildRows = []
let worstGate = { diff: 0, ctx: '' }

for (const nDocs of CORPUS_SIZES) {
  const docs = generateCorpus(9001, nDocs, { unicodeEvery: 11 })
  const patternSets = PATTERN_LENGTHS.map((len) => makePatterns(9001, docs, len, PATTERNS_PER_LENGTH))

  // --- correctness gate: identical refIndex order + scores within 1e-9 ---
  {
    const mojo = new FuseMojo(docs, OPTIONS)
    const ref = new ReferenceFuse(docs, OPTIONS)
    for (const patterns of patternSets) {
      for (const pattern of patterns) {
        const a = mojo.search(pattern)
        const b = ref.search(pattern)
        const aIdx = a.map((r) => r.refIndex).join(',')
        const bIdx = b.map((r) => r.refIndex).join(',')
        if (aIdx !== bIdx) {
          console.error(`GATE FAIL (order) at ${nDocs} docs, pattern ${pattern}`)
          process.exit(1)
        }
        for (let i = 0; i < a.length; i += 1) {
          const diff = Math.abs(a[i].score - b[i].score)
          if (diff > worstGate.diff) worstGate = { diff, ctx: `${nDocs}/${pattern}` }
          if (diff > 1e-9) {
            console.error(`GATE FAIL (score ${diff}) at ${nDocs} docs, pattern ${pattern}`)
            process.exit(1)
          }
        }
      }
    }
    mojo.destroy()
  }

  // --- index construction (median of 5 fresh builds) ---
  const fuseBuilds = []
  const mojoBuilds = []
  for (let r = 0; r < RUNS; r += 1) {
    let t0 = performance.now()
    const ref = new ReferenceFuse(docs, OPTIONS)
    fuseBuilds.push(performance.now() - t0)
    t0 = performance.now()
    const mojo = new FuseMojo(docs, OPTIONS)
    mojoBuilds.push(performance.now() - t0)
    mojo.destroy()
    void ref
  }
  buildRows.push({ nDocs, fuse: median(fuseBuilds), mojo: median(mojoBuilds) })

  for (let ps = 0; ps < PATTERN_LENGTHS.length; ps += 1) {
    const len = PATTERN_LENGTHS[ps]
    const patterns = patternSets[ps]

    // Cold (§11: first call incl. index build): fresh instance + first query,
    // median of 5 runs x 3 patterns. Process/JIT is already warm; the cold
    // axis here is the per-instance index + first-query state a user hits on
    // their very first search call.
    const fuseCold = []
    const mojoCold = []
    for (let r = 0; r < RUNS; r += 1) {
      for (const p of patterns.slice(0, 3)) {
        let t0 = performance.now()
        const coldRef = new ReferenceFuse(docs, OPTIONS)
        coldRef.search(p)
        fuseCold.push(performance.now() - t0)
        t0 = performance.now()
        const coldMojo = new FuseMojo(docs, OPTIONS)
        coldMojo.search(p)
        mojoCold.push(performance.now() - t0)
        coldMojo.destroy()
      }
    }
    coldRows.push({ nDocs, len, fuseMs: median(fuseCold), mojoMs: median(mojoCold) })

    const ref = new ReferenceFuse(docs, OPTIONS)
    const mojo = new FuseMojo(docs, OPTIONS)

    // Warmup: one full round for both engines (JIT + caches).
    for (const p of patterns) {
      ref.search(p)
      mojo.search(p)
    }

    const fuseTimes = []
    const mojoTimes = []
    for (let r = 0; r < RUNS; r += 1) {
      let t0 = performance.now()
      for (const p of patterns) ref.search(p)
      fuseTimes.push((performance.now() - t0) / patterns.length)
      t0 = performance.now()
      for (const p of patterns) mojo.search(p)
      mojoTimes.push((performance.now() - t0) / patterns.length)
    }
    mojo.destroy()
    const fuseMs = median(fuseTimes)
    const mojoMs = median(mojoTimes)
    rows.push({ nDocs, len, fuseMs, mojoMs, speedup: fuseMs / mojoMs })
    console.error(`done ${nDocs} docs / len ${len}`)
  }
}

console.log('\n## Search — warm steady state (default threshold 0.6, includeScore, median of 5 runs)\n')
console.log('| corpus | pattern len | Fuse.js ms/query | fuse-mojo ms/query | Fuse.js q/s | fuse-mojo q/s | speedup |')
console.log('|---:|---:|---:|---:|---:|---:|---:|')
for (const r of rows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${r.len} | ${fmt(r.fuseMs)} | ${fmt(r.mojoMs)} | ` +
      `${fmt(1000 / r.fuseMs, 1)} | ${fmt(1000 / r.mojoMs, 1)} | ${fmt(r.speedup, 1)}x |`
  )
}

console.log('\n## Search — cold first call (fresh index build + first query, median of 5 runs x 3 patterns)\n')
console.log('| corpus | pattern len | Fuse.js cold (ms) | fuse-mojo cold (ms) | speedup |')
console.log('|---:|---:|---:|---:|---:|')
for (const r of coldRows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${r.len} | ${fmt(r.fuseMs, 1)} | ${fmt(r.mojoMs, 1)} | ${fmt(r.fuseMs / r.mojoMs, 1)}x |`
  )
}
console.log('\n## Index construction (median of 5 builds)\n')
console.log('| corpus | Fuse.js build (ms) | fuse-mojo build (ms) | ratio |')
console.log('|---:|---:|---:|---:|')
for (const r of buildRows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${fmt(r.fuse, 1)} | ${fmt(r.mojo, 1)} | ${fmt(r.mojo / r.fuse, 2)}x |`
  )
}
console.log(`\ncorrectness gate: max |Δscore| = ${worstGate.diff.toExponential(2)} (at ${worstGate.ctx})`)
