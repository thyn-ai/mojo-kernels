# Security Policy

This repository contains Algenta's open-source Mojo kernels and the thin
wrapper packages that ship them: `bm25-mojo` and `cclib-mojo` (Python,
`ctypes`) and `fuse-mojo` (TypeScript/Node, `koffi`), each with a vendored
pure-language fallback. We take the security of these packages seriously and
appreciate responsible disclosure from the community.

## Supported versions

Each package is versioned and released independently (per-platform wheels on
PyPI, platform packages on npm). Security fixes land on `main` and in the
latest release of each affected package.

| Channel | Supported |
| --- | --- |
| Latest release of each package / `main` | :white_check_mark: |
| Older releases | Best-effort; please upgrade to the latest |

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for
security problems.** Public disclosure before a fix is available puts other
users at risk.

Report privately through either channel:

1. **GitHub Security Advisories** (preferred) — open a private report from
   this repository's **Security → Report a vulnerability** tab.
2. **Email** — `security@algenta.ai`.

Please include, where possible: a description of the issue and its impact,
the affected package (`bm25-mojo`, `cclib-mojo`, `fuse-mojo`) and component
(kernel, wrapper, fallback, or packaging), steps to reproduce or a proof of
concept, the package version and the platform (macOS arm64 / Linux x86_64 /
fallback) you tested.

## What to expect

- Acknowledgement within 3 business days.
- An initial assessment and severity triage within 7 business days.
- Regular updates as we work on a fix, and credit in the published advisory
  (unless you prefer to remain anonymous).
- Coordinated disclosure: we agree on a timeline with you and publish a
  GitHub Security Advisory once a fix is available.

## Scope

**In scope** — this repository's own code, including:

- The Mojo kernels under `kernels/` and their C ABI: memory-safety bugs
  (out-of-bounds reads or writes, integer overflow in index or length
  arithmetic, use-after-free across `*_create` / `*_destroy`), and the
  `kernels/fuse/src/shim.c` pthread shim
- The wrappers' native-library loading in `python/*/_native.py` and
  `typescript/fuse-mojo/packages/core/src/native.cjs`: how the shared
  library is located and loaded (search paths, environment overrides such as
  `*_DISABLE_NATIVE`, rpath handling), and how inputs are marshalled across
  the FFI boundary
- Divergence between a kernel and its fallback that a caller could exploit
  (the differential suites assert parity; a reproducible break is a bug)
- Packaging: what the wheel-repair (`delocate` / `auditwheel`) and npm
  pack (`pack-platform.sh`) steps vendor into the published artifacts, and
  the `hatch_build.py` build hooks
- Insecure defaults in any package in `python/` or `typescript/`

**Out of scope for this repository** (please report upstream, or privately
to `security@algenta.ai` for anything Algenta-owned):

- The reference libraries used as test and benchmark oracles — `rank_bm25`,
  cclib / PyQuante, and Fuse.js (also vendored as fuse-mojo's fallback
  backend, unmodified, under its own license)
- The Mojo compiler and MAX runtime from Modular, which the kernels are
  built with and whose shared libraries the wheels vendor
- The Algenta Engine and Algenta's hosted infrastructure (separate, private
  repositories)

**Also out of scope:** third-party dependencies (report those upstream; we
still want to hear how they affect these packages), and social-engineering,
physical, or denial-of-service testing against any hosted environment.

Thank you for helping keep Algenta and its users safe.
