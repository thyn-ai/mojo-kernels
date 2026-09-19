#!/usr/bin/env node
/**
 * Reproducible benchmark: MiniSearch 7.2.0 vs minisearch-mojo (native Mojo
 * kernel), on the operations the kernel accelerates: fuzzy/prefix term
 * resolution over the vocabulary plus BM25 accumulation.
 *
 * Corpora are generated locally from fixed seeds: 10k / 50k / 100k documents
 * (title + text fields) over a ~3000-token vocabulary. Query mixes per
 * corpus: single-term fuzzy (0.2), two-term fuzzy AND, single-term prefix,
 * autoSuggest with fuzzy, autoSuggest default (last-term prefix), and exact
 * search. Median of 5 runs after a full warmup round. A correctness gate
 * (scores within 1e-9, ranking consistent, identical suggestions) runs
 * before any timing.
 *
 * Deps resolve through the typescript/minisearch_fuzzy_mojo workspace — run
 * `npm install` there first, then:
 * `node benchmarks/bench_minisearch_fuzzy.mjs`
 */
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import os from 'node:os'

const require = createRequire(new URL('../typescript/minisearch_fuzzy_mojo/package.json', import.meta.url))
const MiniSearchRef = require('minisearch')
const MiniSearchMojo = require('@minisearch-mojo/core')
const { generateVocab, generateCorpus, generateQueries, compareSearch, compareSuggest } = require('./tests/helpers.cjs')

const CORPUS_SIZES = [10_000, 50_000, 100_000]
const QUERIES_PER_MIX = 40
const RUNS = 5
const INDEX_OPTIONS = { fields: ['title', 'text'], storeFields: ['title'] }

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function fmt(n, digits = 3) {
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

const MIXES = [
  { name: 'search exact (OR)', kind: 'search', opts: {}, pick: (qs) => qs.filter((_, i) => i % 10 === 0) },
  { name: 'search fuzzy 0.2', kind: 'search', opts: { fuzzy: 0.2 }, pick: (qs) => qs.filter((_, i) => i % 10 === 1 || i % 10 === 2 || i % 10 === 6) },
  { name: 'search fuzzy+prefix AND', kind: 'search', opts: { fuzzy: 0.2, prefix: true, combineWith: 'AND' }, pick: (qs) => qs.filter((_, i) => i % 10 === 5 || i % 10 === 8) },
  { name: 'search prefix', kind: 'search', opts: { prefix: true }, pick: (qs) => qs.filter((_, i) => i % 10 === 4) },
  { name: 'autoSuggest fuzzy 0.2', kind: 'suggest', opts: { fuzzy: 0.2 }, pick: (qs) => qs.filter((_, i) => i % 10 === 1 || i % 10 === 6) },
  { name: 'autoSuggest default (prefix last)', kind: 'suggest', opts: {}, pick: (qs) => qs.filter((_, i) => i % 10 === 4) },
]

const refVersion = JSON.parse(
  require('node:fs').readFileSync(
    new URL('../typescript/minisearch_fuzzy_mojo/node_modules/minisearch/package.json', import.meta.url),
    'utf8'
  )
).version

console.log('# minisearch-mojo vs MiniSearch benchmark')
console.log(
  `env: ${os.cpus()[0]?.model ?? 'unknown CPU'} (${os.cpus().length} threads), ` +
    `${os.platform()} ${os.arch}, node ${process.version}, minisearch ${refVersion}, ` +
    `minisearch-mojo native: ${MiniSearchMojo.nativeAvailable()}, ${new Date().toISOString().slice(0, 10)}`
)
if (!MiniSearchMojo.nativeAvailable()) {
  console.error('ERROR: native kernel unavailable; refusing to benchmark the fallback.')
  process.exit(1)
}

const rows = []
const coldRows = []
const buildRows = []
let worstGate = { diff: 0, ctx: '' }

for (const nDocs of CORPUS_SIZES) {
  const vocab = generateVocab(9001, 3000)
  const docs = generateCorpus(9001, nDocs, vocab, { unicodeEvery: 11 })
  const allQueries = generateQueries(9001, vocab, 400)

  // --- correctness gate ---
  {
    const mojo = new MiniSearchMojo(INDEX_OPTIONS)
    const ref = new MiniSearchRef(INDEX_OPTIONS)
    mojo.addAll(docs)
    ref.addAll(docs)
    for (const mix of MIXES) {
      for (const query of mix.pick(allQueries).slice(0, 10)) {
        if (mix.kind === 'search') {
          const problem = compareSearch(mojo.search(query, mix.opts), ref.search(query, mix.opts))
          if (problem) {
            console.error(`GATE FAIL at ${nDocs} docs, ${mix.name}, query ${query}: ${problem}`)
            process.exit(1)
          }
          const a = mojo.search(query, mix.opts)
          const b = ref.search(query, mix.opts)
          for (let i = 0; i < a.length; i += 1) {
            const diff = Math.abs(a[i].score - b[i].score)
            if (diff > worstGate.diff) worstGate = { diff, ctx: `${nDocs}/${mix.name}/${query}` }
          }
        } else {
          const problem = compareSuggest(mojo.autoSuggest(query, mix.opts), ref.autoSuggest(query, mix.opts))
          if (problem) {
            console.error(`GATE FAIL (suggest) at ${nDocs} docs, ${mix.name}, query ${query}: ${problem}`)
            process.exit(1)
          }
          const a = mojo.autoSuggest(query, mix.opts)
          const b = ref.autoSuggest(query, mix.opts)
          for (let i = 0; i < a.length; i += 1) {
            const diff = Math.abs(a[i].score - b[i].score)
            if (diff > worstGate.diff) worstGate = { diff, ctx: `${nDocs}/${mix.name}/${query}` }
          }
        }
      }
    }
    mojo.destroy()
  }

  // --- index construction (median of 5 fresh builds) ---
  const refBuilds = []
  const mojoBuilds = []
  for (let r = 0; r < RUNS; r += 1) {
    let t0 = performance.now()
    const ref = new MiniSearchRef(INDEX_OPTIONS)
    ref.addAll(docs)
    refBuilds.push(performance.now() - t0)
    t0 = performance.now()
    const mojo = new MiniSearchMojo(INDEX_OPTIONS)
    mojo.addAll(docs)
    mojoBuilds.push(performance.now() - t0)
    mojo.destroy()
    void ref
  }
  buildRows.push({ nDocs, ref: median(refBuilds), mojo: median(mojoBuilds) })

  for (const mix of MIXES) {
    const queries = mix.pick(allQueries).slice(0, QUERIES_PER_MIX)
    if (queries.length === 0) continue
    const run = (engine, q, opts, kind) => (kind === 'search' ? engine.search(q, opts) : engine.autoSuggest(q, opts))

    // Cold: fresh instance (addAll + native build) + first query.
    const refCold = []
    const mojoCold = []
    for (let r = 0; r < RUNS; r += 1) {
      for (const q of queries.slice(0, 3)) {
        let t0 = performance.now()
        const coldRef = new MiniSearchRef(INDEX_OPTIONS)
        coldRef.addAll(docs)
        run(coldRef, q, mix.opts, mix.kind)
        refCold.push(performance.now() - t0)
        t0 = performance.now()
        const coldMojo = new MiniSearchMojo(INDEX_OPTIONS)
        coldMojo.addAll(docs)
        run(coldMojo, q, mix.opts, mix.kind)
        mojoCold.push(performance.now() - t0)
        coldMojo.destroy()
      }
    }
    coldRows.push({ nDocs, mix: mix.name, refMs: median(refCold), mojoMs: median(mojoCold) })

    const ref = new MiniSearchRef(INDEX_OPTIONS)
    ref.addAll(docs)
    const mojo = new MiniSearchMojo(INDEX_OPTIONS)
    mojo.addAll(docs)

    // Warmup round.
    for (const q of queries) {
      run(ref, q, mix.opts, mix.kind)
      run(mojo, q, mix.opts, mix.kind)
    }

    const refTimes = []
    const mojoTimes = []
    for (let r = 0; r < RUNS; r += 1) {
      let t0 = performance.now()
      for (const q of queries) run(ref, q, mix.opts, mix.kind)
      refTimes.push((performance.now() - t0) / queries.length)
      t0 = performance.now()
      for (const q of queries) run(mojo, q, mix.opts, mix.kind)
      mojoTimes.push((performance.now() - t0) / queries.length)
    }
    mojo.destroy()
    const refMs = median(refTimes)
    const mojoMs = median(mojoTimes)
    rows.push({ nDocs, mix: mix.name, n: queries.length, refMs, mojoMs, speedup: refMs / mojoMs })
    console.error(`done ${nDocs} docs / ${mix.name}`)
  }
}

console.log('\n## Warm steady state (median of 5 runs, ms per query)\n')
console.log('| corpus | query mix | n queries | MiniSearch ms | mojo ms | MiniSearch q/s | mojo q/s | speedup |')
console.log('|---:|---|---:|---:|---:|---:|---:|---:|')
for (const r of rows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${r.mix} | ${r.n} | ${fmt(r.refMs)} | ${fmt(r.mojoMs)} | ` +
      `${fmt(1000 / r.refMs, 1)} | ${fmt(1000 / r.mojoMs, 1)} | ${fmt(r.speedup, 1)}x |`
  )
}

console.log('\n## Cold first call (fresh addAll + first query, median of 5 runs x 3 queries)\n')
console.log('| corpus | query mix | MiniSearch cold (ms) | mojo cold (ms) | speedup |')
console.log('|---:|---|---:|---:|---:|')
for (const r of coldRows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${r.mix} | ${fmt(r.refMs, 1)} | ${fmt(r.mojoMs, 1)} | ${fmt(r.refMs / r.mojoMs, 1)}x |`
  )
}

console.log('\n## Index construction (median of 5 builds)\n')
console.log('| corpus | MiniSearch build (ms) | mojo build (ms) | ratio |')
console.log('|---:|---:|---:|---:|')
for (const r of buildRows) {
  console.log(
    `| ${r.nDocs.toLocaleString('en-US')} | ${fmt(r.ref, 1)} | ${fmt(r.mojo, 1)} | ${fmt(r.mojo / r.ref, 2)}x |`
  )
}
console.log(`\ncorrectness gate: max |Δscore| = ${worstGate.diff.toExponential(2)} (at ${worstGate.ctx})`)
