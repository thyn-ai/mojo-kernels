## What does this change?

<!-- One or two sentences. -->

## Checklist

- [ ] Differential tests added or updated, and passing on **both** backends
      (native Mojo kernel and forced fallback via `<NAME>_MOJO_DISABLE_NATIVE=1`)
- [ ] Docs updated if needed (package README, kernel table in README.md,
      CHANGELOG.md `Unreleased`)
- [ ] Kernel ABI changes bump `<name>mojo_abi_version` and the matching
      wrapper handshake
- [ ] Benchmark numbers, if changed, were re-measured with the seeded
      `pixi run bench*` scripts and the correctness gate passed first
- [ ] No hardcoded credentials or secrets
- [ ] I agree my contribution is licensed under the project's Apache-2.0
      license (inbound=outbound, GitHub Terms of Service §D.6)
