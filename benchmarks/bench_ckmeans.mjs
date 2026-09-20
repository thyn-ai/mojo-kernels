#!/usr/bin/env node
/**
 * Reproducible benchmark: simple-statistics 7.12.0 ckmeans vs @ckmeans-mojo/core
 * (native Mojo kernel).
 *
 * Datasets are generated locally from fixed seeds: mixture blobs (realistic
 * clustering) and small-integer grids (heavy DP ties), at n = 1k / 5k / 20k
 * with k = 3 / 10. Median of 5 timed calls after a 2-call warmup. A
 * correctness gate (exact cluster-assignment equality) runs before any
 * timing. **Cold** is a fresh node process loading the package and making
 * its first call (koffi dlopen + JIT included), median of 5 spawns.
 *
 * Deps resolve through the typescript/ckmeans-mojo workspace — run
 * `npm install` there first, then: `node benchmarks/bench_ckmeans.mjs`.
 */
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import { execFileSync } from 'node:child_process'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

const require = createRequire(new URL('../typescript/ckmeans-mojo/package.json', import.meta.url))
const { makeDataset } = require('./tests/helpers.cjs')

const CELLS = []
for (const kind of ['mixture', 'int-grid']) {
  for (const n of [1_000, 5_000, 20_000]) {
    for (const k of [3, 10]) {
      CELLS.push({ kind, n, k })
    }
  }
}
const RUNS = 5
const SEED = 20260919

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function fmt(n, digits = 3) {
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

// --- cold-child mode: load one package, time the first call, print ms ---
if (process.argv[2] === '--cold-child') {
  const [, , , which, kind, nStr, kStr] = process.argv
  const n = Number(nStr)
  const k = Number(kStr)
  const x = makeDataset(SEED, n, kind)
  const pkg = which === 'mojo' ? require('@ckmeans-mojo/core') : require('simple-statistics')
  const fn = which === 'mojo' ? pkg : pkg.ckmeans
  const t0 = performance.now()
  fn(x, k)
  process.stdout.write(String(performance.now() - t0))
  process.exit(0)
}

const reference = require('simple-statistics')
const ckmeans = require('@ckmeans-mojo/core')

console.log('# ckmeans-mojo vs simple-statistics benchmark')
console.log(
  `env: ${os.cpus()[0]?.model ?? 'unknown CPU'} (${os.cpus().length} threads), ` +
    `${os.platform()} ${os.arch}, node ${process.version}, simple-statistics ${require('simple-statistics/package.json').version}, ` +
    `ckmeans-mojo native: ${ckmeans.nativeAvailable()}, ${new Date().toISOString().slice(0, 10)}`
)
if (!ckmeans.nativeAvailable()) {
  console.error('ERROR: native kernel unavailable; refusing to benchmark the fallback.')
  process.exit(1)
}

function coldMs(which, cell) {
  const times = []
  for (let r = 0; r < RUNS; r += 1) {
    const out = execFileSync(
      process.execPath,
      [fileURLToPath(import.meta.url), '--cold-child', which, cell.kind, String(cell.n), String(cell.k)],
      { encoding: 'utf8' }
    )
    times.push(Number(out))
  }
  return median(times)
}

const warmRows = []
const coldRows = []

for (const cell of CELLS) {
  const x = makeDataset(SEED, cell.n, cell.kind)

  // --- correctness gate: exact cluster-assignment equality ---
  {
    const a = ckmeans(x, cell.k)
    const b = reference.ckmeans(x, cell.k)
    if (JSON.stringify(a) !== JSON.stringify(b)) {
      console.error(`GATE FAIL at ${cell.kind} n=${cell.n} k=${cell.k}`)
      process.exit(1)
    }
  }

  // --- cold: fresh process, first call (median of 5 spawns) ---
  const refCold = coldMs('ref', cell)
  const mojoCold = coldMs('mojo', cell)
  coldRows.push({ ...cell, refCold, mojoCold })

  // --- warm: 2 warmup calls, then median of 5 single timed calls ---
  for (let w = 0; w < 2; w += 1) {
    reference.ckmeans(x, cell.k)
    ckmeans(x, cell.k)
  }
  const refTimes = []
  const mojoTimes = []
  for (let r = 0; r < RUNS; r += 1) {
    let t0 = performance.now()
    reference.ckmeans(x, cell.k)
    refTimes.push(performance.now() - t0)
    t0 = performance.now()
    ckmeans(x, cell.k)
    mojoTimes.push(performance.now() - t0)
  }
  const refMs = median(refTimes)
  const mojoMs = median(mojoTimes)
  warmRows.push({ ...cell, refMs, mojoMs, speedup: refMs / mojoMs })
  console.error(`done ${cell.kind} n=${cell.n} k=${cell.k}`)
}

console.log('\n## ckmeans — warm steady state (median of 5 timed calls)\n')
console.log('| dataset | n | k | simple-statistics ms/call | ckmeans-mojo ms/call | simple-statistics calls/s | ckmeans-mojo calls/s | speedup |')
console.log('|---|---:|---:|---:|---:|---:|---:|---:|')
for (const r of warmRows) {
  console.log(
    `| ${r.kind} | ${r.n.toLocaleString('en-US')} | ${r.k} | ${fmt(r.refMs)} | ${fmt(r.mojoMs)} | ` +
      `${fmt(1000 / r.refMs, 1)} | ${fmt(1000 / r.mojoMs, 1)} | ${fmt(r.speedup, 1)}x |`
  )
}

console.log('\n## ckmeans — cold first call (fresh node process: module load + first call, median of 5 spawns)\n')
console.log('| dataset | n | k | simple-statistics cold (ms) | ckmeans-mojo cold (ms) | speedup |')
console.log('|---|---:|---:|---:|---:|---:|')
for (const r of coldRows) {
  console.log(
    `| ${r.kind} | ${r.n.toLocaleString('en-US')} | ${r.k} | ${fmt(r.refCold, 1)} | ${fmt(r.mojoCold, 1)} | ${fmt(r.refCold / r.mojoCold, 1)}x |`
  )
}
