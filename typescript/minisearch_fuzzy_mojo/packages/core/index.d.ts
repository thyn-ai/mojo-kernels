/**
 * @minisearch-mojo/core — drop-in faster replacement for MiniSearch v7
 * fuzzy/prefix search and auto-suggestions, powered by a Mojo kernel.
 */

export interface IndexOptions {
  /** Fields to index for full-text search (required, declaration order). */
  fields: string[]
  /** Fields to return with search results. */
  storeFields?: string[]
  /** Document id field name (default "id"). */
  idField?: string
  /** Default search options applied to every search/autoSuggest call. */
  searchOptions?: SearchOptions
}

export interface SearchOptions {
  /** true/false, or a function evaluated per query term. */
  prefix?: boolean | ((term: string) => boolean)
  /** Edit budget: < 1 means round(fuzzy * term.length), >= 1 is absolute,
   *  true means 0.2; or a function evaluated per query term. */
  fuzzy?: boolean | number | ((term: string) => boolean | number)
  /** Per-field boost multipliers. */
  boost?: Record<string, number>
  /** Restrict scoring to these fields. */
  fields?: string[]
  /** 'OR' (search default) or 'AND' (autoSuggest default). */
  combineWith?: 'AND' | 'OR'
}

export interface SearchResult {
  id: any
  score: number
  terms: string[]
  queryTerms: string[]
  match: Record<string, string[]>
  [storedField: string]: any
}

export interface Suggestion {
  suggestion: string
  terms: string[]
  score: number
}

export interface BackendInfo {
  native_available: boolean
  native_source: string | null
  abi_version_expected: number
  abi_version_native: number | null
  disabled_by_env: boolean
  platform: string
  arch: string
  error: string | null
}

export default class MiniSearchMojo {
  constructor(options: IndexOptions)
  add(doc: Record<string, any>): this
  addAll(docs: Array<Record<string, any>>): this
  search(query: string, options?: SearchOptions): SearchResult[]
  autoSuggest(query: string, options?: SearchOptions): Suggestion[]
  /** Which backend serves this instance: "native" or "fallback". */
  readonly backend: 'native' | 'fallback'
  /** Free the native index handle (also freed automatically on GC). */
  destroy(): void
  static nativeAvailable(): boolean
  static backendInfo(): BackendInfo
  static readonly version: string
}

export { MiniSearchMojo as MiniSearch }
export declare const nativeAvailable: typeof MiniSearchMojo.nativeAvailable
export declare const backendInfo: typeof MiniSearchMojo.backendInfo
export declare const version: string
export declare class UnsupportedOptionError extends Error {
  code: string
}
