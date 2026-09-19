/**
 * End-user smoke test for @natural-mojo/core: the natural README distance
 * examples, unmodified except for the import. Must produce identical results
 * on the native kernel and on the pure-JS fallback.
 *
 * Usage: node quickstart.mjs [--assert-native|--assert-fallback]
 */
import natural, { nativeAvailable, backendInfo } from '@natural-mojo/core'

// --- natural README quick-start (https://github.com/NaturalNode/natural) ---
const results = {
  levenshtein_ones_one: natural.LevenshteinDistance('ones', 'one'),
  levenshtein_one_one: natural.LevenshteinDistance('one', 'one'),
  levenshtein_kitten_sitting: natural.LevenshteinDistance('kitten', 'sitting'),
  levenshtein_sub_cost_2: natural.LevenshteinDistance('kitten', 'sitting', { substitution_cost: 2 }),
  damerau_ca_abc: natural.DamerauLevenshteinDistance('ca', 'abc'),
  damerau_ca_abc_restricted: natural.DamerauLevenshteinDistance('ca', 'abc', { restricted: true }),
  damerau_kitten_sitting: natural.DamerauLevenshteinDistance('kitten', 'sitting'),
  diacritics_as_is: natural.LevenshteinDistance('café', 'cafe'),
  empty: natural.LevenshteinDistance('', ''),
}

const expected = {
  levenshtein_ones_one: 1,
  levenshtein_one_one: 0,
  levenshtein_kitten_sitting: 3,
  levenshtein_sub_cost_2: 5,
  damerau_ca_abc: 2,
  damerau_ca_abc_restricted: 3,
  damerau_kitten_sitting: 3,
  diacritics_as_is: 1,
  empty: 0,
}

const backend = nativeAvailable() ? 'native' : 'fallback'
const mode = process.argv[2] || ''
if (mode === '--assert-native' && backend !== 'native') {
  console.error(`FAIL: expected native backend, got ${backend}: ${JSON.stringify(backendInfo())}`)
  process.exit(1)
}
if (mode === '--assert-fallback' && backend !== 'fallback') {
  console.error(`FAIL: expected fallback backend, got ${backend}`)
  process.exit(1)
}
for (const [key, value] of Object.entries(expected)) {
  if (results[key] !== value) {
    console.error(`FAIL: ${key}: expected ${value}, got ${results[key]}`)
    process.exit(1)
  }
}

console.log(JSON.stringify({ backend, results }, null, 1))
