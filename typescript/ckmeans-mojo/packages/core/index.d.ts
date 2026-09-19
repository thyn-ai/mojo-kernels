// Type definitions for @ckmeans-mojo/core.
// Mirrors the simple-statistics ckmeans(x, nClusters) API shape.

declare function ckmeans(x: number[], nClusters: number): number[][]

declare namespace ckmeans {
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

  /** True if the native kernel can cluster right now. Never throws. */
  function nativeAvailable(): boolean
  /** Diagnostics for the active backend. Never throws. */
  function backendInfo(): BackendInfo
  /** Which backend serves calls on this platform. */
  const backend: 'native' | 'fallback'
  const version: string
}

export = ckmeans
