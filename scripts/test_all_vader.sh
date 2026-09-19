#!/usr/bin/env bash
# Full vader-mojo differential suite: once against the native kernel, once
# with the pure-Python fallback forced on. Run from the repository root:
#
#   pixi run bash scripts/test_all_vader.sh
#
# The oracle (PyPI vaderSentiment 3.3.2) must be importable. If it is not, it
# is downloaded once into a cache dir OUTSIDE the pixi environment (pixi
# re-syncs .pixi and would wipe a pip-installed package) and put on PYTHONPATH.
# Override the location with VADER_ORACLE_DIR.
set -euo pipefail

cd "$(dirname "$0")/.."

VADER_SENTIMENT_VERSION="3.3.2"
VADER_WHEEL_SHA256="3bf1d243b98b1afad575b9f22bc2cb1e212b94ff89ca74f8a23a588d024ea311"
ORACLE_DIR="${VADER_ORACLE_DIR:-$HOME/.cache/vader-oracle/pkg}"

# Build the kernel if it is missing (idempotent).
if [ ! -f kernels/vader/build/libvadermojo.dylib ] && [ ! -f kernels/vader/build/libvadermojo.so ]; then
  bash kernels/vader/build.sh
fi

if ! python -c "import vaderSentiment" >/dev/null 2>&1; then
  if [ ! -d "$ORACLE_DIR/vaderSentiment" ]; then
    echo "== fetching oracle vaderSentiment==${VADER_SENTIMENT_VERSION} (sha256-pinned) =="
    VADER_SENTIMENT_VERSION="$VADER_SENTIMENT_VERSION" \
    VADER_WHEEL_SHA256="$VADER_WHEEL_SHA256" \
    ORACLE_DIR="$ORACLE_DIR" \
    python - <<'PYEOF'
import hashlib
import json
import os
import urllib.request
import zipfile

version = os.environ["VADER_SENTIMENT_VERSION"]
expected = os.environ["VADER_WHEEL_SHA256"]
dest = os.environ["ORACLE_DIR"]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/vaderSentiment/{version}/json"))
wheel = next(u for u in meta["urls"] if u["filename"].endswith(".whl"))
data = urllib.request.urlopen(wheel["url"]).read()
digest = hashlib.sha256(data).hexdigest()
if digest != expected:
    raise SystemExit(f"oracle wheel sha256 mismatch: {digest} != {expected}")
os.makedirs(dest, exist_ok=True)
zipfile.ZipFile(__import__("io").BytesIO(data)).extractall(dest)
print(f"oracle {version} extracted to {dest}")
PYEOF
  fi
  export PYTHONPATH="${ORACLE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
fi

export PYTHONPATH="python/vader_mojo${PYTHONPATH:+:$PYTHONPATH}"

SUITE="tests/test_vader_differential.py tests/test_vader_loader.py"

echo "== differential suite: native backend =="
python -m pytest $SUITE -q

echo
echo "== differential suite: forced fallback (VADER_MOJO_DISABLE_NATIVE=1) =="
VADER_MOJO_DISABLE_NATIVE=1 python -m pytest $SUITE -q
