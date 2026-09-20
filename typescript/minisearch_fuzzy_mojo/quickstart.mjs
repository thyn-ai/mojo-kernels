/**
 * End-user quickstart for @minisearch-mojo/core, mirroring the MiniSearch
 * README example: build a small index, run a fuzzy search and an
 * auto-suggestion, print the results as JSON with the active backend.
 * Used by scripts/smoke.sh for the fresh-install end-user test.
 */
import MiniSearch from '@minisearch-mojo/core'

const documents = [
  { id: 1, title: 'Moby Dick', text: 'Call me Ishmael. Some years ago...', category: 'fiction' },
  { id: 2, title: 'Zen and the Art of Motorcycle Maintenance', text: 'I can see by my watch...', category: 'fiction' },
  { id: 3, title: 'Neuromancer', text: 'The sky above the port was...', category: 'fiction' },
  { id: 4, title: 'Zen in the Art of Archery', text: 'At first sight it must seem...', category: 'non-fiction' },
]

const miniSearch = new MiniSearch({
  fields: ['title', 'text'],
  storeFields: ['title', 'category'],
  searchOptions: { fuzzy: 0.2 },
})
miniSearch.addAll(documents)

const results = miniSearch.search('zen art motorcycle')
const suggestions = miniSearch.autoSuggest('neromancer', { fuzzy: 0.2 })

const out = {
  backend: miniSearch.backend,
  search: results.map((r) => ({ id: r.id, title: r.title, score: r.score, terms: r.terms })),
  autoSuggest: suggestions,
}
console.log(JSON.stringify(out, null, 1))

if (process.argv.includes('--assert-native') && out.backend !== 'native') {
  throw new Error(`expected native backend, got ${out.backend}`)
}
if (process.argv.includes('--assert-fallback') && out.backend !== 'fallback') {
  throw new Error(`expected fallback backend, got ${out.backend}`)
}
