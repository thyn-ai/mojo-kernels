#!/usr/bin/env node
/**
 * Reproducible benchmark: natural 8.1.1 vs natural-mojo (native Mojo kernel).
 *
 * String pairs are generated locally from fixed seeds. Cells: pair sizes of
 * 8 / 64 / 512 / 1024 UTF-16 code units x three variants — Levenshtein
 * (lev), unrestricted Damerau-Levenshtein (dlu, the reference default) and
 * restricted/OSA Damerau (dlr). Warm steady state: median of 5 runs of
 * per-pair calls. Cold: fresh subprocess with module load (require) and the
 * first call after load measured separately, median of 5 subprocesses. A
 * correctness gate (exact float64 equality across ~3000 pairs and all
 * variants) runs before any timing.
 *
 * Deps resolve through the typescript/natural-mojo workspace — run
 * `npm install` there first, then: `node benchmarks/bench_natural.mjs`.
 */
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import { execFileSync } from 'node:child_process'
import os from 'node:os'

const require = createRequire(new URL('../typescript/natural-mojo/package.json', import.meta.url))
const natural = require('natural')
const naturalMojo = require('./packages/core/index.cjs')
const { generatePairs, generateTranspositionPairs, rng, makeWord, mutateToken, UNICODE_TOKENS } = require('./tests/helpers.cjs')

const RUNS = 5

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function fmt(n, digits = 3) {
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

/** n pairs of exactly `size` code units, half with 1-2 seeded typos. */
function makeSizedPairs(seed, size, count) {
  const rand = rng(seed ^ (size * 0x9e37))
  const pairs = []
  while (pairs.length < count) {
    let s = ''
    while (s.length < size) s += makeWord(rand, 3)
    s = s.slice(0, size)
    let t = s
    if (pairs.length % 2 === 1) {
      const nMut = 1 + Math.floor(rand() * 2)
      for (let m = 0; m < nMut; m += 1) {
        const pos = Math.floor(rand() * t.length)
        t = t.slice(0, pos) + 'abcdefghijklmnopqrstuvwxyz'[Math.floor(rand() * 26)] + t.slice(pos + 1)
      }
    }
    pairs.push([s, t])
  }
  return pairs
}

const VARIANTS = [
  ['lev', (pkg, s, t) => pkg.LevenshteinDistance(s, t)],
  ['dlu', (pkg, s, t) => pkg.DamerauLevenshteinDistance(s, t)],
  ['dlr', (pkg, s, t) => pkg.DamerauLevenshteinDistance(s, t, { restricted: true })],
]

// size -> [pairs, reps per run]
const CELLS = [
  [8, [30, 100]],
  [64, [30, 10]],
  [512, [12, 1]],
  [1024, [6, 1]],
]

console.log('# natural-mojo vs natural benchmark')
console.log(
  `env: ${os.cpus()[0]?.model ?? 'unknown CPU'} (${os.cpus().length} threads), ` +
    `${os.platform()} ${os.arch}, node ${process.version}, natural ${require('natural/package.json').version} (npm), ` +
    `Mojo kernel native: ${naturalMojo.nativeAvailable()}, ${new Date().toISOString().slice(0, 10)}`
)
if (!naturalMojo.nativeAvailable()) {
  console.error('ERROR: native kernel unavailable; refusing to benchmark the fallback.')
  process.exit(1)
}

// --- correctness gate: exact float64 equality across all variants ---
{
  const gatePairs = [
    ...generatePairs(31337, 400, { unicodeEvery: 7 }),
    ...generateTranspositionPairs(2718, 200),
  ]
  // long-string spot checks
  const rand = rng(31415)
  for (const size of [200, 255, 256, 511]) {
    let s = ''
    while (s.length < size) s += makeWord(rand, 3)
    s = s.slice(0, size)
    gatePairs.push([s, mutateToken(rand, s)])
    gatePairs.push([s, UNICODE_TOKENS[rand() * UNICODE_TOKENS.length | 0] + s.slice(4)])
  }
  const optionSets = [
    undefined,
    { substitution_cost: 2 },
    { insertion_cost: 2, deletion_cost: 3, substitution_cost: 0.5 },
    { transposition_cost: 3, restricted: true },
    { transposition_cost: 0.75, insertion_cost: 0.5 },
  ]
  let checked = 0
  for (const [s, t] of gatePairs) {
    for (const options of optionSets) {
      const a1 = naturalMojo.LevenshteinDistance(s, t, options)
      const b1 = natural.LevenshteinDistance(s, t, options)
      const a2 = naturalMojo.DamerauLevenshteinDistance(s, t, options)
      const b2 = natural.DamerauLevenshteinDistance(s, t, options)
      if (a1 !== b1 || a2 !== b2) {
        console.error(`GATE FAIL at ${JSON.stringify(s.slice(0, 24))}/${JSON.stringify(t.slice(0, 24))} ${JSON.stringify(options)}: lev ${a1} vs ${b1}, dam ${a2} vs ${b2}`)
        process.exit(1)
      }
      checked += 2
    }
  }
  console.log(`correctness gate: ${checked} distance comparisons, exact equality (max |Δ| = 0)`)
}

// --- cold: fresh subprocess, module load and first call measured separately ---
// Module load is reported for context but is dominated by natural's large
// dependency tree (dotenv and friends), which performs environment-dependent
// work on this machine; the kernel-relevant cold cost is the first call
// after load (natural-mojo pays dlopen + ABI handshake + scratch init there).
console.error('cold measurements (fresh subprocesses)...')
const coldRows = []
for (const size of [8, 512]) {
  const [pairs] = makeSizedPairs(555, size, 1)
  const script = (pkgExpr, call, a, b) => `
    const { performance } = require('node:perf_hooks')
    const t0 = performance.now()
    const pkg = require(${pkgExpr})
    const t1 = performance.now()
    const d = pkg.${call}(${JSON.stringify(a)}, ${JSON.stringify(b)})
    const t2 = performance.now()
    process.stdout.write(JSON.stringify({ load: t1 - t0, first: t2 - t1 }))
    if (typeof d !== 'number') process.exit(1)
  `
  const cwd = new URL('../typescript/natural-mojo/', import.meta.url).pathname
  // Requiring natural prints dotenvx banners to stdout; the JSON timing is
  // the last {...} object written (emitted after the require returns).
  const runCold = (code) => {
    const out = execFileSync(process.execPath, ['-e', code], { encoding: 'utf8', cwd })
    const match = out.match(/\{[^{}]*"load"[^{}]*\}/g)
    if (!match) throw new Error(`cold subprocess produced no timing: ${out.slice(0, 200)}`)
    return JSON.parse(match[match.length - 1])
  }
  const natLoad = []
  const natFirst = []
  const mojoLoad = []
  const mojoFirst = []
  for (let r = 0; r < RUNS; r += 1) {
    const nat = runCold(script('"natural"', 'LevenshteinDistance', ...pairs))
    natLoad.push(nat.load)
    natFirst.push(nat.first)
    const mojo = runCold(script('"./packages/core/index.cjs"', 'LevenshteinDistance', ...pairs))
    mojoLoad.push(mojo.load)
    mojoFirst.push(mojo.first)
  }
  coldRows.push({
    size,
    naturalLoadMs: median(natLoad),
    naturalFirstMs: median(natFirst),
    mojoLoadMs: median(mojoLoad),
    mojoFirstMs: median(mojoFirst),
  })
}

// --- warm steady state ---
const rows = []
for (const [size, [nPairs, reps]] of CELLS) {
  const pairs = makeSizedPairs(9001, size, nPairs)
  for (const [vname, call] of VARIANTS) {
    // warmup round for both sides (JIT + caches)
    for (const [s, t] of pairs) {
      call(natural, s, t)
      call(naturalMojo, s, t)
    }
    const naturalTimes = []
    const mojoTimes = []
    for (let r = 0; r < RUNS; r += 1) {
      let t0 = performance.now()
      for (let rep = 0; rep < reps; rep += 1) for (const [s, t] of pairs) call(natural, s, t)
      naturalTimes.push((performance.now() - t0) / (nPairs * reps))
      t0 = performance.now()
      for (let rep = 0; rep < reps; rep += 1) for (const [s, t] of pairs) call(naturalMojo, s, t)
      mojoTimes.push((performance.now() - t0) / (nPairs * reps))
    }
    const naturalMs = median(naturalTimes)
    const mojoMs = median(mojoTimes)
    rows.push({ size, variant: vname, naturalMs, mojoMs, speedup: naturalMs / mojoMs })
    console.error(`done size ${size} / ${vname}`)
  }
}

console.log('\n## Distance — warm steady state (median of 5 runs)\n')
console.log('| pair size (code units) | variant | natural ms/call | natural-mojo ms/call | speedup |')
console.log('|---:|---|---:|---:|---:|')
for (const r of rows) {
  console.log(`| ${r.size.toLocaleString('en-US')} | ${r.variant} | ${fmt(r.naturalMs)} | ${fmt(r.mojoMs)} | ${fmt(r.speedup, 1)}x |`)
}

console.log('\n## Distance — cold (fresh subprocess, median of 5): module load vs first call after load\n')
console.log('| pair size (code units) | natural require (ms) | natural-mojo require (ms) | natural first call (ms) | natural-mojo first call (ms) |')
console.log('|---:|---:|---:|---:|---:|')
for (const r of coldRows) {
  console.log(`| ${r.size} | ${fmt(r.naturalLoadMs, 1)} | ${fmt(r.mojoLoadMs, 1)} | ${fmt(r.naturalFirstMs)} | ${fmt(r.mojoFirstMs)} |`)
}
console.log('\nrequire() of the full natural package is dominated by its dependency tree (environment-dependent); natural-mojo\'s first call includes dlopen + ABI handshake + kernel scratch init.')
