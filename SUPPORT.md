# Support

## Where to get help

| What you need | Where to go |
| --- | --- |
| A bug in a kernel, wrapper package, or fallback | [GitHub Issues](https://github.com/thyn-ai/mojo-kernels/issues) with the `bug` label — include the package version, active backend (`backend_info()` / `Fuse.backendInfo()`), and a minimal reproduction |
| A new kernel, missing option, or other improvement | [GitHub Issues](https://github.com/thyn-ai/mojo-kernels/issues) with the `enhancement` label (see [CONTRIBUTING.md](./CONTRIBUTING.md) and the kernel factory section of the [README](./README.md)) |
| Usage questions, ideas, benchmark results | [GitHub Discussions](https://github.com/thyn-ai/mojo-kernels/discussions) |
| Real-time chat with the community | [Discord](https://discord.gg/w8NDsph9an) |
| Documentation | The [README](./README.md) and the per-package READMEs: [`bm25-mojo`](./python/bm25_mojo/README.md), [`cclib-mojo`](./python/cclib_mojo/README.md), [`fuse-mojo`](./typescript/fuse-mojo/README.md) |

Before filing an issue, please search existing issues and discussions — your
question may already have an answer. If a result differs from the reference
library, first check whether it reproduces with the fallback forced
(`BM25_MOJO_DISABLE_NATIVE=1`, `CCLIB_MOJO_DISABLE_NATIVE=1`,
`FUSE_MOJO_DISABLE_NATIVE=1`) and say so in the report.

## Security issues

**Never report a security vulnerability in a public issue, discussion, or
pull request.** Follow [SECURITY.md](./SECURITY.md): use GitHub Security
Advisories or email `security@algenta.ai`.

## Response expectations

Community support is provided on a best-effort basis by maintainers and other
community members. **There is no SLA for community support** — we triage
issues as time permits, prioritizing security reports (which have their own
response commitments in [SECURITY.md](./SECURITY.md)), correctness bugs where
a kernel diverges from its reference library, and regressions in the latest
release.

Enterprise customers with a commercial Algenta agreement should use their
contracted support channel or contact community@algenta.ai to be routed.
