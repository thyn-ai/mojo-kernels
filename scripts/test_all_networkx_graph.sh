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

# Oracle provisioning: the differential oracle is the published PyPI package
# networkx==3.5. It is deliberately NOT in pixi.toml (this kernel's scope
# does not include that file), so install it into a local target dir and
# put it on PYTHONPATH. The dir also survives pixi environment re-syncs,
# which wipe pip installs from the pixi site-packages.
ORACLE_DIR="$PWD/build/nx-test-oracle"
export PYTHONPATH="$ORACLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -d "$ORACLE_DIR/networkx" ]; then
  if python -m pip --version >/dev/null 2>&1; then
    python -m pip install -q --target "$ORACLE_DIR" "networkx==3.5"
  elif command -v uv >/dev/null 2>&1; then
    # pixi re-syncs can wipe pip from the env; uv is host-managed.
    uv pip install -q --python "$(python -c 'import sys; print(sys.executable)')" \
      --target "$ORACLE_DIR" "networkx==3.5"
  else
    echo "error: need pip in the env or uv on PATH to provision the networkx oracle" >&2
    exit 1
  fi
fi
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
