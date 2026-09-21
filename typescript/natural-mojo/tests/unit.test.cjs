'use strict'

/**
 * Unit tests for the wrapper itself: backend selection, input validation,
 * error surfaces, and API-shape guarantees (backend-independent).
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const naturalMojo = require('@natural-mojo/core')

const NATIVE_EXPECTED = process.env.NATURAL_MOJO_DISABLE_NATIVE !== '1'

test('backend selection follows the environment', () => {
  const info = naturalMojo.backendInfo()
  assert.equal(typeof info.native_available, 'boolean')
  assert.equal(info.abi_version_expected, 1)
  assert.equal(info.disabled_by_env, !NATIVE_EXPECTED)
  assert.equal(naturalMojo.nativeAvailable(), NATIVE_EXPECTED)
  if (NATIVE_EXPECTED) {
    assert.equal(info.abi_version_native, 1)
    assert.match(info.native_source, /libnaturalmojo/)
  } else {
    assert.equal(info.native_available, false)
  }
})

test('known reference values (both backends)', () => {
  // Textbook values every implementation must reproduce.
  assert.equal(naturalMojo.LevenshteinDistance('kitten', 'sitting'), 3)
  assert.equal(naturalMojo.LevenshteinDistance('saturday', 'sunday'), 3)
  assert.equal(naturalMojo.LevenshteinDistance('', ''), 0)
  assert.equal(naturalMojo.LevenshteinDistance('', 'abc'), 3)
  assert.equal(naturalMojo.LevenshteinDistance('abc', ''), 3)
  assert.equal(naturalMojo.LevenshteinDistance('abc', 'abc'), 0)
  // OSA (restricted) vs unrestricted on the classic CA/ABC divergence:
  // true Damerau transposes C,A across the inserted B; OSA cannot.
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ca', 'abc'), 2)
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ca', 'abc', { restricted: true }), 3)
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ab', 'ba'), 1)
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ab', 'ba', { restricted: true }), 1)
})

test('diacritics are compared as-is (no normalization)', () => {
  assert.equal(naturalMojo.LevenshteinDistance('café', 'cafe'), 1)
  assert.equal(naturalMojo.LevenshteinDistance('résumé', 'resume'), 2)
  // Astral characters count as two UTF-16 code units, exactly like the
  // reference: '😀' (2 units) vs 'x' (1 unit) is a 2-edit distance.
  assert.equal(naturalMojo.LevenshteinDistance('😀', 'x'), 2)
  assert.equal(naturalMojo.LevenshteinDistance('😀', '😀'), 0)
})

test('non-string inputs throw TypeError on both backends', () => {
  for (const [s, t] of [[null, 'a'], ['a', undefined], [11, 12], [{}, 'a'], [['a'], ['b']]]) {
    assert.throws(() => naturalMojo.LevenshteinDistance(s, t), TypeError)
    assert.throws(() => naturalMojo.DamerauLevenshteinDistance(s, t), TypeError)
  }
})

test('substring-search variants are out of scope and fail loudly', () => {
  for (const fn of [naturalMojo.LevenshteinDistanceSearch, naturalMojo.DamerauLevenshteinDistanceSearch]) {
    assert.throws(
      () => fn('a', 'abc'),
      (err) => {
        assert.equal(err.code, 'NATURAL_MOJO_UNSUPPORTED_OPTION')
        assert.match(err.message, /Supported functions are: LevenshteinDistance, DamerauLevenshteinDistance/)
        return true
      }
    )
  }
})

test('explicit undefined transposition_cost disables transpositions (reference semantics)', () => {
  // ca → abc is 2 with transpositions, 3 without.
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ca', 'abc'), 2)
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ca', 'abc', { transposition_cost: undefined }), 3)
  assert.equal(naturalMojo.DamerauLevenshteinDistance('ca', 'abc', { transposition_cost: NaN }), 3)
})

test('isNaN cost coercion matches the reference (null → 0, true → 1)', () => {
  // substitution_cost: null acts as 0 (free substitutions).
  assert.equal(naturalMojo.LevenshteinDistance('abc', 'xyz', { substitution_cost: null }), 0)
  // insertion_cost: true acts as 1.
  assert.equal(naturalMojo.LevenshteinDistance('', 'ab', { insertion_cost: true }), 2)
})

test('ESM and CJS surfaces agree', async () => {
  const esm = await import('@natural-mojo/core')
  assert.equal(esm.LevenshteinDistance, naturalMojo.LevenshteinDistance)
  assert.equal(esm.DamerauLevenshteinDistance, naturalMojo.DamerauLevenshteinDistance)
  assert.equal(esm.version, naturalMojo.version)
  assert.equal(esm.default.LevenshteinDistance, naturalMojo.LevenshteinDistance)
})

test('version is exposed', () => {
  assert.equal(naturalMojo.version, '0.1.3') // x-release-please-version
})

test('in-process forced fallback agrees with native on a tricky pair set', { skip: !NATIVE_EXPECTED }, () => {
  const pairs = [
    ['kitten', 'sitting'], ['ca', 'abc'], ['CA', 'ABC'], ['a cat', 'an act'],
    ['café', 'cafe'], ['a😀b', '😀ab'], ['', 'abc'], ['abcd', 'badc'],
  ]
  const optionSets = [
    undefined,
    { substitution_cost: 2 },
    { restricted: true },
    { transposition_cost: 3, substitution_cost: 2 },
    { insertion_cost: 0.5, deletion_cost: 0.25, substitution_cost: 2, transposition_cost: 0.75 },
  ]
  const vendor = require('../packages/core/vendor/natural_distance.cjs')
  for (const [s, t] of pairs) {
    for (const options of optionSets) {
      assert.equal(
        naturalMojo.LevenshteinDistance(s, t, options),
        vendor.LevenshteinDistance(s, t, options),
        `lev ${s}/${t} ${JSON.stringify(options)}`
      )
      assert.equal(
        naturalMojo.DamerauLevenshteinDistance(s, t, options),
        vendor.DamerauLevenshteinDistance(s, t, options),
        `dam ${s}/${t} ${JSON.stringify(options)}`
      )
    }
  }
})

test('a bogus NATURAL_MOJO_NATIVE_LIB path does not break resolution', { skip: !NATIVE_EXPECTED }, () => {
  // The env override is the first resolver candidate, not an exclusive one:
  // an unloadable path is skipped and the next candidate (platform package,
  // then the repo dev build) is used.
  const { spawnSync } = require('node:child_process')
  const script = `
    const nm = require('@natural-mojo/core')
    if (nm.nativeAvailable() !== true) { console.error('native should resolve via the next candidate'); process.exit(1) }
    if (nm.LevenshteinDistance('kitten', 'sitting') !== 3) { console.error('wrong distance'); process.exit(1) }
    console.log('child done')
  `
  const res = spawnSync(process.execPath, ['-e', script], {
    cwd: require('node:path').resolve(__dirname, '..'),
    encoding: 'utf8',
    env: { ...process.env, NATURAL_MOJO_NATIVE_LIB: '/nonexistent/libnaturalmojo.dylib' },
  })
  assert.equal(res.status, 0, res.stderr)
  assert.match(res.stdout, /child done/)
})

test('NATURAL_MOJO_DISABLE_NATIVE=1 forces the fallback in a child process', { skip: !NATIVE_EXPECTED }, () => {
  const { spawnSync } = require('node:child_process')
  const script = `
    const nm = require('@natural-mojo/core')
    if (nm.nativeAvailable() !== false) { console.error('native should be disabled'); process.exit(1) }
    if (nm.DamerauLevenshteinDistance('ca', 'abc') !== 2) { console.error('wrong fallback distance'); process.exit(1) }
    console.log('child done')
  `
  const res = spawnSync(process.execPath, ['-e', script], {
    cwd: require('node:path').resolve(__dirname, '..'),
    encoding: 'utf8',
    env: { ...process.env, NATURAL_MOJO_DISABLE_NATIVE: '1' },
  })
  assert.equal(res.status, 0, res.stderr)
  assert.match(res.stdout, /child done/)
})
