// Type definitions for the supported drop-in surface of @fuse-mojo/core.
// Mirrors the Fuse.js v7 API shape for the supported option subset.

declare namespace FuseMojo {
  interface FuseKeyObject {
    name: string | string[]
    weight?: number
  }

  interface FuseOptions {
    /** List of properties to search (dotted paths supported). */
    keys?: Array<string | string[] | FuseKeyObject>
    /** Give up threshold in [0, 1]; 0.0 requires a perfect match. Default 0.6. */
    threshold?: number
    /** Approximately where in the text the pattern is expected. Default 0. */
    location?: number
    /** How close the match must be to `location`. Default 100. */
    distance?: number
    /** Minimum matched-run length for a result to count. Default 1. */
    minMatchCharLength?: number
    /** Include the score in results. Default false. */
    includeScore?: boolean
    /** Include matched character index ranges in results. Default false. */
    includeMatches?: boolean
    /** Sort results by score. Default true. */
    shouldSort?: boolean
    /** Ignore `location`/`distance` when scoring. Default false. */
    ignoreLocation?: boolean
    /** Case-sensitive matching. Default false. */
    isCaseSensitive?: boolean
    /** Keep searching after a perfect match. Default false. */
    findAllMatches?: boolean
    /** Ignore the field-length norm in scoring. Default false. */
    ignoreFieldNorm?: boolean
    /** Field-length norm weight. Default 1. */
    fieldNormWeight?: number
  }

  interface FuseSearchOptions {
    limit?: number
  }

  type FuseIndexRange = [number, number]

  interface FuseResultMatch {
    indices: FuseIndexRange[]
    value: string
    key?: string | string[]
    refIndex?: number
  }

  interface FuseResult<T> {
    item: T
    refIndex: number
    score?: number
    matches?: FuseResultMatch[]
  }

  interface BackendInfo {
    native_available: boolean
    native_source: string | null
    abi_version_expected: number
    abi_version_native: number | null
    disabled_by_env: boolean
    platform: string
    arch: string
    error: string | null
  }
}

declare class FuseMojo<T = any> {
  constructor(list: ReadonlyArray<T>, options?: FuseMojo.FuseOptions)
  options: FuseMojo.FuseOptions
  search(pattern: string, opts?: FuseMojo.FuseSearchOptions): Array<FuseMojo.FuseResult<T>>
  setCollection(docs: ReadonlyArray<T>): this
  readonly backend: 'native' | 'fallback'
  destroy(): void

  static config: FuseMojo.FuseOptions
  static version: string
  static nativeAvailable(): boolean
  static backendInfo(): FuseMojo.BackendInfo
  static createIndex(): never
  static parseIndex(): never
}

export = FuseMojo
