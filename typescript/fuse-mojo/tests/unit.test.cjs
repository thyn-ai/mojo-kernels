'use strict'

/**
 * Unit tests for the wrapper itself: backend selection, option validation,
 * error surfaces, and API-shape guarantees (backend-independent).
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const Fuse = require('@fuse-mojo/core')

const NATIVE_EXPECTED = process.env.FUSE_MOJO_DISABLE_NATIVE !== '1'

test('backend selection follows the environment', () => {
  const info = Fuse.backendInfo()
  assert.equal(typeof info.native_available, 'boolean')
  assert.equal(info.abi_version_expected, 1)
  assert.equal(info.disabled_by_env, !NATIVE_EXPECTED)
  assert.equal(Fuse.nativeAvailable(), NATIVE_EXPECTED)
  const fuse = new Fuse(['alpha', 'beta'], {})
  assert.equal(fuse.backend, NATIVE_EXPECTED ? 'native' : 'fallback')
})

test('unsupported options throw a clear error listing supported options', () => {
  const cases = [
    [{ useExtendedSearch: true }, 'useExtendedSearch'],
    [{ ignoreDiacritics: true }, 'ignoreDiacritics'],
    [{ getFn: () => 'x' }, 'getFn'],
    [{ sortFn: () => 0 }, 'sortFn'],
    [{ keys: [{ name: 'a', getFn: () => 'x' }] }, 'getFn'],
    [{ location: 1.5 }, 'location'],
    [{ minMatchCharLength: 2.5 }, 'minMatchCharLength'],
    [{ threshold: NaN }, 'threshold'],
  ]
  for (const [options, what] of cases) {
    assert.throws(
      () => new Fuse([{ a: 'x' }], options),
      (err) => {
        assert.equal(err.code, 'FUSE_MOJO_UNSUPPORTED_OPTION')
        assert.match(err.message, /Supported options are: keys, threshold, location, distance/)
        return true
      },
      `expected UnsupportedOptionError for ${what}`
    )
  }
})

test('extended-search syntax in the pattern is treated literally (basic path)', () => {
  // useExtendedSearch stays off: '^abc' is a literal 4-char fuzzy pattern.
  const fuse = new Fuse(['^abc def', 'abc'], { includeScore: true, threshold: 0.0 })
  const res = fuse.search('^abc')
  assert.equal(res.length, 1)
  assert.equal(res[0].refIndex, 0)
})

test('non-string queries throw (logical search unsupported)', () => {
  const fuse = new Fuse(['alpha'], {})
  assert.throws(() => fuse.search({ $and: [{ text: 'a' }] }), /non-string query/)
})

test('external indices are rejected', () => {
  assert.throws(() => new Fuse(['a'], {}, {}), /FUSE_MOJO_UNSUPPORTED_OPTION|external index/)
  assert.throws(() => Fuse.createIndex(['a'], ['b']), /unsupported/i)
  assert.throws(() => Fuse.parseIndex({}), /unsupported/i)
})

test('empty pattern matches nothing', () => {
  const fuse = new Fuse(['alpha', 'beta'], { includeScore: true })
  assert.deepEqual(fuse.search(''), [])
})

test('empty and blank-only collections are valid', () => {
  assert.deepEqual(new Fuse([], { includeScore: true }).search('a'), [])
  assert.deepEqual(new Fuse(['', '   '], { includeScore: true }).search('a'), [])
})

test('invalid key objects fail fast', () => {
  assert.throws(() => new Fuse([{ a: 'x' }], { keys: [{ weight: 1 }] }), /name/)
  assert.throws(() => new Fuse([{ a: 'x' }], { keys: [{ name: 'a', weight: 0 }] }), /weight/i)
})

test('result shape without includeScore/includeMatches is minimal', () => {
  const fuse = new Fuse(['alpha', 'beta'], {})
  const res = fuse.search('alpah')
  assert.equal(res.length, 1)
  assert.deepEqual(Object.keys(res[0]).sort(), ['item', 'refIndex'])
  assert.equal(res[0].item, 'alpha')
})

test('destroy() releases the native index and search keeps working after GC-safe reuse', () => {
  const fuse = new Fuse(['alpha', 'beta'], {})
  fuse.destroy()
  // After destroy the instance must not be used; constructing a new one works.
  const fuse2 = new Fuse(['alpha'], {})
  assert.equal(fuse2.search('alpha').length, 1)
  fuse2.destroy()
})

test('setCollection reindexes', () => {
  const fuse = new Fuse(['alpha'], {})
  fuse.setCollection(['gamma', 'delta'])
  const res = fuse.search('gamma')
  assert.equal(res.length, 1)
  assert.equal(res[0].item, 'gamma')
})

test('explicit destroy + GC does not double-free the native index', { skip: !NATIVE_EXPECTED }, () => {
  // Regression: FinalizationRegistry entries must be unregistered on explicit
  // destroy (a no-token register() can never be unregistered per spec).
  // A double free surfaces as a tcmalloc "Object was not in-use" complaint on
  // the child's stderr; assert clean exit and clean stderr.
  const { spawnSync } = require('node:child_process')
  const script = `
    const Fuse = require('@fuse-mojo/core')
    const docs = Array.from({ length: 20000 }, (_, i) => 'tok' + i + ' alpha beta gamma')
    for (let r = 0; r < 3; r++) {
      const f = new Fuse(docs, { includeScore: true })
      f.search('alpha')
      f.destroy()
    }
    gc()
    console.log('child done')
  `
  const res = spawnSync(process.execPath, ['--expose-gc', '-e', script], {
    cwd: require('node:path').resolve(__dirname, '..'),
    encoding: 'utf8',
  })
  assert.equal(res.status, 0, res.stderr)
  assert.match(res.stdout, /child done/)
  assert.doesNotMatch(res.stderr, /in-use|tcmalloc|double free/i)
})

test('config defaults are exposed and mutable like the reference', () => {
  assert.equal(Fuse.config.threshold, 0.6)
  assert.equal(Fuse.version, '0.1.0')
})

test('in-process forced fallback agrees with native on a tricky corpus', () => {
  if (!NATIVE_EXPECTED) {
    return // the whole-process fallback run covers this
  }
  const docs = ['hello world', 'a😀bc xy', 'café résumé', 'xxab', 'the quick brown fox']
  const options = { includeScore: true, includeMatches: true, threshold: 0.7, minMatchCharLength: 2 }
  const native = new Fuse(docs, options)
  assert.equal(native.backend, 'native')
  process.env.FUSE_MOJO_DISABLE_NATIVE = '1'
  let fallback
  try {
    fallback = new Fuse(docs, options)
  } finally {
    delete process.env.FUSE_MOJO_DISABLE_NATIVE
  }
  assert.equal(fallback.backend, 'fallback')
  for (const pattern of ['hello', '😀b', 'cafe', 'xx', 'quik', 'world', '']) {
    assert.deepEqual(native.search(pattern), fallback.search(pattern), `pattern ${pattern}`)
  }
})
