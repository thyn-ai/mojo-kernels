#!/usr/bin/env bash
# Full nx-mojo differential suite: once against the native Mojo kernel,
# once with the pure-Python fallback forced on.
#
# Run from the repository root inside the pixi environment, with the wrapper
# on PYTHONPATH and the kernel already built:
#   ~/.pixi/bin/pixi run bash kernels/networkx-graph/build.sh
#   PYTHONPATH=python/nx_mojo ~/.pixi/bin/pixi run bash scripts/test_all_networkx_graph.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# The differential oracle is the published PyPI package networkx==3.5,
# provided by the repo pixi environment (pixi.toml [pypi-dependencies],
# pinned and lock-verified). The version check guards the parity claim
# against an env that drifted from the lock.
python - <<'EOF'
import networkx
assert networkx.__version__ == "3.5", f"oracle must be networkx==3.5, got {networkx.__version__}"
EOF

SUITE="tests/test_networkx_graph_differential.py tests/test_networkx_graph_loader.py"

echo "== differential suite: native backend =="
pytest $SUITE -q

echo
echo "== differential suite: forced fallback (NX_MOJO_DISABLE_NATIVE=1) =="
NX_MOJO_DISABLE_NATIVE=1 pytest $SUITE -q
