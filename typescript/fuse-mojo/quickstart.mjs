/**
 * End-user smoke test for @fuse-mojo/core: the Fuse.js README quick-start,
 * unmodified except for the import. Must produce identical results on the
 * native kernel and on the pure-JS fallback.
 *
 * Usage: node quickstart.mjs [--assert-native|--assert-fallback]
 */
import Fuse, { backendInfo } from '@fuse-mojo/core'

// --- Fuse.js README quick-start (https://fusejs.io) -------------------------
const books = [
  {
    title: "Old Man's War",
    author: { firstName: 'John', lastName: 'Scalzi' },
  },
  {
    title: 'The Lock Artist',
    author: { firstName: 'Steve', lastName: 'Hamilton' },
  },
  {
    title: 'HTML5',
    author: { firstName: 'Remy', lastName: 'Sharp' },
  },
  {
    title: 'Right Ho Jeeves',
    author: { firstName: 'P.D', lastName: 'Woodhouse' },
  },
  {
    title: 'The Code of the Wooster',
    author: { firstName: 'P.D', lastName: 'Woodhouse' },
  },
  {
    title: 'Thank You Jeeves',
    author: { firstName: 'P.D', lastName: 'Woodhouse' },
  },
  {
    title: 'The DaVinci Code',
    author: { firstName: 'Dan', lastName: 'Brown' },
  },
  {
    title: 'Angels & Demons',
    author: { firstName: 'Dan', lastName: 'Brown' },
  },
  {
    title: 'The Silmarillion',
    author: { firstName: 'J.R.R', lastName: 'Tolkien' },
  },
  {
    title: 'Syrup',
    author: { firstName: 'Max', lastName: 'Barry' },
  },
  {
    title: 'The Lost Symbol',
    author: { firstName: 'Dan', lastName: 'Brown' },
  },
  {
    title: 'The Book of Lies',
    author: { firstName: 'Brad', lastName: 'Meltzer' },
  },
  {
    title: 'Lamb',
    author: { firstName: 'Christopher', lastName: 'Moore' },
  },
  {
    title: 'Fool',
    author: { firstName: 'Christopher', lastName: 'Moore' },
  },
  {
    title: 'Incompetence',
    author: { firstName: 'Rob', lastName: 'Grant' },
  },
  {
    title: 'Fat',
    author: { firstName: 'Rob', lastName: 'Grant' },
  },
  {
    title: 'Colony',
    author: { firstName: 'Rob', lastName: 'Grant' },
  },
  {
    title: 'Backwards, Red Dwarf',
    author: { firstName: 'Rob', lastName: 'Grant' },
  },
  {
    title: 'The Grand Design',
    author: { firstName: 'Stephen', lastName: 'Hawking' },
  },
  {
    title: 'The Book of Samson',
    author: { firstName: 'David', lastName: 'Maine' },
  },
]

const options = {
  includeScore: true,
  includeMatches: true,
  keys: ['title', 'author.firstName'],
}

const fuse = new Fuse(books, options)
const results = fuse.search('old man')

const mode = process.argv[2] || ''
if (mode === '--assert-native' && fuse.backend !== 'native') {
  console.error(`FAIL: expected native backend, got ${fuse.backend}: ${JSON.stringify(backendInfo())}`)
  process.exit(1)
}
if (mode === '--assert-fallback' && fuse.backend !== 'fallback') {
  console.error(`FAIL: expected fallback backend, got ${fuse.backend}`)
  process.exit(1)
}
if (results.length === 0 || results[0].item.title !== "Old Man's War") {
  console.error(`FAIL: unexpected top hit: ${JSON.stringify(results[0])}`)
  process.exit(1)
}

console.log(JSON.stringify({ backend: fuse.backend, results }, null, 1))
