#!/usr/bin/env python3
"""Reproducible benchmark: ASE's primitive_neighbor_list vs ase_mojo.

Workloads are random atomistic configurations at a typical molecular-
dynamics density (cell volume ~ 16 A^3/atom, cutoff 2.0 A): the exact
workload the ASE neighbor search runs in geometry optimizations and MD.
Three cases: 2k and 8k atoms in an orthorhombic box, 8k atoms in a
triclinic box.

Method (mirrors the repo's other kernel benchmarks):

* correctness is asserted before any timing — the native and fallback
  (i, j, S) sets must equal the oracle's on a separate validation case;
* every implementation is timed in its own fresh subprocess: call #1 is
  the COLD time (first call after imports: dlopen, ctypes binding,
  allocator warm-up), calls #2..6 give the WARM median of 5;
* inputs are built once per process before any timing, so data
  generation is excluded everywhere;
* the ASE oracle is imported from ASE_ORACLE_SITE (see
  scripts/test_all_ase_neighborlist.sh); the benchmark skips the oracle
  column with a warning when ASE is not importable.

Run from the repository root:

    PYTHONPATH=python/ase_mojo:$(pwd)/.oracle pixi run python \\
        benchmarks/bench_ase_neighborlist.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_WARM = 5
TRICLINIC_BASE = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]])

WORKLOADS = {
    "ortho-2k": {"n": 2000, "triclinic": False, "cutoff": 2.0, "seed": 11},
    "ortho-8k": {"n": 8000, "triclinic": False, "cutoff": 2.0, "seed": 13},
    "triclinic-8k": {"n": 8000, "triclinic": True, "cutoff": 2.0, "seed": 17},
}

# Subprocess worker: builds inputs, times cold + warm calls for one impl.
_WORKER = r"""
import json
import sys
import time

import numpy as np

spec = json.loads(sys.argv[1])
impl = spec["impl"]
rng = np.random.default_rng(spec["seed"])
n = spec["n"]
if spec["triclinic"]:
    base = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]])
    cell = base * (n / 6.0) ** (1 / 3)
else:
    side = (16.0 * n) ** (1 / 3)
    cell = np.diag([side, side, side])
pos = rng.random((n, 3)) @ cell
cutoff = spec["cutoff"]
pbc = [True, True, True]

if impl == "oracle":
    from ase.neighborlist import primitive_neighbor_list as fn
elif impl == "native":
    from ase_mojo import primitive_neighbor_list as fn
elif impl == "fallback":
    import os

    os.environ["ASE_MOJO_DISABLE_NATIVE"] = "1"
    from ase_mojo import primitive_neighbor_list as fn
else:
    raise SystemExit(f"unknown impl {impl}")

t0 = time.perf_counter()
i, j, S = fn("ijS", pbc, cell, pos, cutoff)
cold = time.perf_counter() - t0
warm = []
for _ in range(spec["n_warm"]):
    t0 = time.perf_counter()
    fn("ijS", pbc, cell, pos, cutoff)
    warm.append(time.perf_counter() - t0)
print(json.dumps({"cold": cold, "warm": warm, "npairs": int(len(i))}))
"""

# Validation case worker: dumps the pair set for correctness checks.
_VALIDATE_WORKER = r"""
import json
import sys

import numpy as np

spec = json.loads(sys.argv[1])
impl = spec["impl"]
rng = np.random.default_rng(4242)
cell = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]]) * 1.3
pos = rng.random((200, 3)) @ cell + rng.normal(0, 0.2, (200, 3))
cutoff = 1.9
pbc = [True, True, True]
if impl == "oracle":
    from ase.neighborlist import primitive_neighbor_list as fn
else:
    import os

    if impl == "fallback":
        os.environ["ASE_MOJO_DISABLE_NATIVE"] = "1"
    from ase_mojo import primitive_neighbor_list as fn
i, j, S = fn("ijS", pbc, cell, pos, cutoff)
print(json.dumps(sorted(zip(i.tolist(), j.tolist(), [tuple(s) for s in S.tolist()]))))
"""


def _env() -> dict:
    env = dict(os.environ)
    pythonpath = [os.path.join(REPO_ROOT, "python", "ase_mojo")]
    oracle_site = os.environ.get("ASE_ORACLE_SITE")
    if oracle_site:
        pythonpath.append(oracle_site)
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run_worker(script: str, spec: dict) -> dict | list:
    out = subprocess.run(
        [sys.executable, "-c", script, json.dumps(spec)],
        check=True,
        capture_output=True,
        text=True,
        env=_env(),
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _oracle_importable() -> bool:
    try:
        _run_worker(_VALIDATE_WORKER, {"impl": "oracle"})
        return True
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def main() -> None:
    print(f"machine: {platform.machine()} / {platform.system()} — {platform.platform()}")
    print(f"python:  {sys.version.split()[0]}")
    print(f"numpy:   {np.__version__}")
    try:
        import ase_mojo

        info = ase_mojo.backend_info()
        print(f"ase_mojo native: {info['native_available']} ({info['native_source']})")
    except Exception as exc:  # pragma: no cover
        print(f"ase_mojo import failed in parent (workers retry): {exc}")

    # --- correctness gate -------------------------------------------------
    print("\ncorrectness gate (200-atom triclinic validation case):")
    impls = ["native", "fallback"]
    have_oracle = _oracle_importable()
    if have_oracle:
        impls = ["oracle", "native", "fallback"]
    sets = {}
    for impl in impls:
        pairs = _run_worker(_VALIDATE_WORKER, {"impl": impl})
        sets[impl] = {
            (int(a), int(b), (int(s[0]), int(s[1]), int(s[2]))) for a, b, s in pairs
        }
        print(f"  {impl:9s}: {len(sets[impl])} pairs")
    reference = sets.get("oracle") or sets["native"]
    for impl in impls:
        assert sets[impl] == reference, f"{impl} pair set != reference pair set"
    print("  pair sets identical across implementations")

    # --- timing ------------------------------------------------------------
    timed_impls = ["native", "fallback"] + (["oracle"] if have_oracle else [])
    if not have_oracle:
        print("\nWARNING: ASE oracle not importable; oracle column skipped "
              "(set ASE_ORACLE_SITE)")
    rows = []
    for name, spec in WORKLOADS.items():
        row = {"workload": name}
        for impl in timed_impls:
            result = _run_worker(
                _WORKER, {**spec, "impl": impl, "n_warm": N_WARM}
            )
            row[impl] = result
        rows.append(row)

    print("\nresults (milliseconds; cold = first call in a fresh process, "
          f"warm = median of {N_WARM}):")
    header = f"{'workload':>13} | {'pairs':>8} |"
    for impl in timed_impls:
        header += f" {impl + ' cold':>12} {impl + ' warm':>12} |"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['workload']:>13} | {row['native']['npairs']:>8} |"
        for impl in timed_impls:
            cold_ms = row[impl]["cold"] * 1e3
            warm_ms = statistics.median(row[impl]["warm"]) * 1e3
            line += f" {cold_ms:12.3f} {warm_ms:12.3f} |"
        print(line)
    if have_oracle:
        print("\nspeedup (oracle warm / ase_mojo warm):")
        for row in rows:
            oracle_warm = statistics.median(row["oracle"]["warm"])
            native_warm = statistics.median(row["native"]["warm"])
            fallback_warm = statistics.median(row["fallback"]["warm"])
            print(
                f"  {row['workload']:>13}: native {oracle_warm / native_warm:8.1f}x "
                f"fallback {oracle_warm / fallback_warm:8.2f}x"
            )


if __name__ == "__main__":
    main()
