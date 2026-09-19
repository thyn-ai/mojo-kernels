/**
 * End-user smoke test for @ckmeans-mojo/core: the simple-statistics ckmeans
 * README example, unmodified except for the import, plus a larger mixed
 * workload. Must produce identical results on the native kernel and on the
 * pure-JS fallback.
 *
 * Usage: node quickstart.mjs [--assert-native|--assert-fallback]
 */
import ckmeans, { backendInfo } from '@ckmeans-mojo/core'

// --- simple-statistics README ckmeans example ------------------------------
// https://github.com/simple-statistics/simple-statistics#ckmeans
const readme = ckmeans([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3)

// --- a slightly larger seeded workload (ties, floats, duplicates) ----------
let a = 42 >>> 0
const rand = () => {
  a |= 0
  a = (a + 0x6d2b79f5) | 0
  let t = Math.imul(a ^ (a >>> 15), 1 | a)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const data = []
for (let i = 0; i < 500; i += 1) {
  if (i % 3 === 0) data.push(Math.floor(rand() * 6)) // heavy ties
  else if (i % 3 === 1) data.push(Math.floor(rand() * 4) * 25 + rand() * 8) // blobs
  else data.push(rand() * 100) // floats
}
const clusters = ckmeans(data, 5)
const edge = [
  ckmeans([7, 7, 7], 2), // constant array collapses to one cluster
  ckmeans([5, 3, 1, 4, 2], 5), // k = n
  ckmeans([5, 3, 1, 4, 2], 1), // k = 1
]

const mode = process.argv[2] || ''
if (mode === '--assert-native' && ckmeans.backend !== 'native') {
  console.error(`FAIL: expected native backend, got ${ckmeans.backend}: ${JSON.stringify(backendInfo())}`)
  process.exit(1)
}
if (mode === '--assert-fallback' && ckmeans.backend !== 'fallback') {
  console.error(`FAIL: expected fallback backend, got ${ckmeans.backend}`)
  process.exit(1)
}
if (JSON.stringify(readme) !== JSON.stringify([[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]])) {
  console.error(`FAIL: unexpected README example result: ${JSON.stringify(readme)}`)
  process.exit(1)
}

console.log(JSON.stringify({ backend: ckmeans.backend, readme, clusters, edge }, null, 1))
