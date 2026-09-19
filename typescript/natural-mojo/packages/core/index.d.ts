// Type definitions for @natural-mojo/core.
// Mirrors natural v8.1.1's LevenshteinDistance / DamerauLevenshteinDistance
// option surface (the substring-search variants are out of scope and throw
// UnsupportedOptionError at runtime).

export interface LevenshteinOptions {
  /** Cost of inserting one target character. Default 1. */
  insertion_cost?: number
  /** Cost of deleting one source character. Default 1. */
  deletion_cost?: number
  /** Cost of substituting one character. Default 1. */
  substitution_cost?: number
}

export interface DamerauLevenshteinOptions extends LevenshteinOptions {
  /** Cost of one transposition. Default 1 (only when the key is absent). */
  transposition_cost?: number
  /**
   * true: OSA (Optimal String Alignment — adjacent transpositions, no
   * substring edited twice). false (default): unrestricted
   * Damerau-Levenshtein (Lowrance-Wagner).
   */
  restricted?: boolean
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

export class UnsupportedOptionError extends Error {
  code: 'NATURAL_MOJO_UNSUPPORTED_OPTION'
}

/** Levenshtein distance (insert/delete/substitute) between two strings. */
export function LevenshteinDistance(
  source: string,
  target: string,
  options?: LevenshteinOptions
): number

/** Damerau-Levenshtein distance (adds transpositions) between two strings. */
export function DamerauLevenshteinDistance(
  source: string,
  target: string,
  options?: DamerauLevenshteinOptions
): number

/** Out of scope: throws UnsupportedOptionError. */
export function LevenshteinDistanceSearch(): never
/** Out of scope: throws UnsupportedOptionError. */
export function DamerauLevenshteinDistanceSearch(): never

/** True when the native Mojo kernel can compute distances right now. */
export function nativeAvailable(): boolean
/** Diagnostics for the active backend. */
export function backendInfo(): BackendInfo

export const version: string

declare const naturalMojo: {
  LevenshteinDistance: typeof LevenshteinDistance
  DamerauLevenshteinDistance: typeof DamerauLevenshteinDistance
  LevenshteinDistanceSearch: typeof LevenshteinDistanceSearch
  DamerauLevenshteinDistanceSearch: typeof DamerauLevenshteinDistanceSearch
  nativeAvailable: typeof nativeAvailable
  backendInfo: typeof backendInfo
  UnsupportedOptionError: typeof UnsupportedOptionError
  version: string
}

export default naturalMojo
