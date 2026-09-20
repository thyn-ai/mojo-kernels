#!/usr/bin/env python3
"""Reproducible benchmark: spatialmath-python (oracle) vs spatialmath_mojo.

Poses and points are generated locally from fixed seeds (no network, no
datasets): proper random SE(3)/SO(3) batches (Rodrigues rotations) of 100 /
1k / 10k / 100k poses and matching point clouds. Two measurements per
operation:

* cold first-call — median of 5 fresh-interpreter first calls at n=10k
  (imports happen before the timer; for spatialmath_mojo this includes
  dlopen + ABI handshake of the native kernel),
* warm steady-state — per-call latency, median of 5 batches of repeated
  calls at each size.

The oracle side constructs its SE3/SO3 pose objects once, outside the
timer; only the pose operation itself (`Ta * Tb`, `Ta.inv()`, `Ta * v`,
`T * P`) is timed — the per-pose numpy dispatch in
`spatialmath.baseposelist._binop` that this package accelerates. Our side
times the whole `spatialmath_mojo` call on plain numpy arrays, validation
included.

Correctness is asserted (element-wise vs the oracle within 1e-10) before
any timing happens, so the numbers below always come from a
verified-correct build.

Run from the repository root:

    PYTHONPATH="python/spatialmath_mojo:.oracle-spatialmath" \
        pixi run python benchmarks/bench_spatialmath.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import numpy as np

SIZES = [100, 1_000, 10_000, 100_000]
COLD_SIZE = 10_000
N_RUNS = 5  # median over this many batches / fresh processes
BATCH = 20  # calls per warm batch
SEED = 20260919
ATOL = 1e-10

# (label, kind): kind selects the oracle/ours call pair below.
OPERATIONS = [
    ("se3_compose", "se3_compose"),
    ("se3_inverse", "se3_inverse"),
    ("se3_transform_batch_vector", "se3_tf_bv"),
    ("se3_transform_single_points", "se3_tf_sp"),
    ("so3_compose", "so3_compose"),
    ("so3_inverse", "so3_inverse"),
    ("so3_transform_batch_vector", "so3_tf_bv"),
]


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(theta)
        + (1.0 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * k
    )


def make_inputs(seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    se3 = np.repeat(np.eye(4)[np.newaxis, :, :], n, axis=0)
    so3 = np.empty((n, 3, 3))
    for i in range(n):
        so3[i] = _rodrigues(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
    se3[:, :3, :3] = so3
    se3[:, :3, 3] = rng.normal(0.0, 5.0, size=(n, 3))
    so3_b = np.empty((n, 3, 3))
    se3_b = np.repeat(np.eye(4)[np.newaxis, :, :], n, axis=0)
    for i in range(n):
        so3_b[i] = _rodrigues(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
    se3_b[:, :3, :3] = so3_b
    se3_b[:, :3, 3] = rng.normal(0.0, 5.0, size=(n, 3))
    points = rng.normal(0.0, 10.0, size=(n, 3))
    vector = np.array([1.5, -2.0, 3.25])
    return {"se3": se3, "se3_b": se3_b, "so3": so3, "so3_b": so3_b,
            "points": points, "vector": vector}


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    lines.append(f"- python: {platform.python_version()}, numpy: {np.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


# --------------------------------------------------------------- oracle side


def oracle_call(sm_ops: dict, kind: str) -> None:
    if kind == "se3_compose":
        sm_ops["Ta"] * sm_ops["Tb"]
    elif kind == "se3_inverse":
        sm_ops["Ta"].inv()
    elif kind == "se3_tf_bv":
        sm_ops["Ta"] * sm_ops["vector"]
    elif kind == "se3_tf_sp":
        sm_ops["T0"] * sm_ops["points_T"]
    elif kind == "so3_compose":
        sm_ops["Ra"] * sm_ops["Rb"]
    elif kind == "so3_inverse":
        sm_ops["Ra"].inv()
    elif kind == "so3_tf_bv":
        sm_ops["Ra"] * sm_ops["vector"]


def make_oracle_operands(SE3, SO3, inputs: dict) -> dict:
    se3, se3_b = inputs["se3"], inputs["se3_b"]
    so3, so3_b = inputs["so3"], inputs["so3_b"]
    return {
        "Ta": SE3([se3[i] for i in range(se3.shape[0])]),
        "Tb": SE3([se3_b[i] for i in range(se3_b.shape[0])]),
        "T0": SE3(se3[0]),
        "Ra": SO3([so3[i] for i in range(so3.shape[0])]),
        "Rb": SO3([so3_b[i] for i in range(so3_b.shape[0])]),
        "vector": inputs["vector"],
        "points_T": np.ascontiguousarray(inputs["points"].T),
    }


# ---------------------------------------------------------------- our side


def ours_call(smm, inputs: dict, kind: str) -> None:
    if kind == "se3_compose":
        smm.compose(inputs["se3"], inputs["se3_b"])
    elif kind == "se3_inverse":
        smm.inverse(inputs["se3"])
    elif kind == "se3_tf_bv":
        smm.transform(inputs["se3"], inputs["vector"])
    elif kind == "se3_tf_sp":
        smm.transform(inputs["se3"][0], inputs["points"])
    elif kind == "so3_compose":
        smm.compose(inputs["so3"], inputs["so3_b"])
    elif kind == "so3_inverse":
        smm.inverse(inputs["so3"])
    elif kind == "so3_tf_bv":
        smm.transform(inputs["so3"], inputs["vector"])


def cold_first_call(kind: str, ours: bool) -> float:
    """Median first-call latency (ms) over N_RUNS fresh interpreter processes."""
    # Fixture math is duplicated in the child snippet: imports and fixture
    # construction happen before the timer; only the first call is timed.
    # Rotations come from normalized random quaternions (vectorized — a
    # `-c` snippet cannot define a function after a semicolon).
    fixture = (
        "import numpy as np;"
        "rng=np.random.default_rng(%d);n=%d;"
        "q=rng.normal(size=(n,4));q/=np.linalg.norm(q,axis=1,keepdims=True);"
        "w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3];"
        "so3=np.empty((n,3,3));"
        "so3[:,0,0]=1-2*(y*y+z*z);so3[:,0,1]=2*(x*y-w*z);so3[:,0,2]=2*(x*z+w*y);"
        "so3[:,1,0]=2*(x*y+w*z);so3[:,1,1]=1-2*(x*x+z*z);so3[:,1,2]=2*(y*z-w*x);"
        "so3[:,2,0]=2*(x*z-w*y);so3[:,2,1]=2*(y*z+w*x);so3[:,2,2]=1-2*(x*x+y*y);"
        "q2=rng.normal(size=(n,4));q2/=np.linalg.norm(q2,axis=1,keepdims=True);"
        "w,x,y,z=q2[:,0],q2[:,1],q2[:,2],q2[:,3];"
        "so3b=np.empty((n,3,3));"
        "so3b[:,0,0]=1-2*(y*y+z*z);so3b[:,0,1]=2*(x*y-w*z);so3b[:,0,2]=2*(x*z+w*y);"
        "so3b[:,1,0]=2*(x*y+w*z);so3b[:,1,1]=1-2*(x*x+z*z);so3b[:,1,2]=2*(y*z-w*x);"
        "so3b[:,2,0]=2*(x*z-w*y);so3b[:,2,1]=2*(y*z+w*x);so3b[:,2,2]=1-2*(x*x+y*y);"
        "se3=np.repeat(np.eye(4)[None],n,axis=0);se3[:,:3,:3]=so3;se3[:,:3,3]=rng.normal(0,5,(n,3));"
        "se3b=np.repeat(np.eye(4)[None],n,axis=0);se3b[:,:3,:3]=so3b;se3b[:,:3,3]=rng.normal(0,5,(n,3));"
        "pts=rng.normal(0,10,(n,3));vec=np.array([1.5,-2.,3.25]);"
    ) % (SEED, COLD_SIZE)
    calls = {
        "se3_compose": ("smm.compose(se3,se3b)", "Ta*Tb"),
        "se3_inverse": ("smm.inverse(se3)", "Ta.inv()"),
        "se3_tf_bv": ("smm.transform(se3,vec)", "Ta*vec"),
        "se3_tf_sp": ("smm.transform(se3[0],pts)", "T0*ptsT"),
        "so3_compose": ("smm.compose(so3,so3b)", "Ra*Rb"),
        "so3_inverse": ("smm.inverse(so3)", "Ra.inv()"),
        "so3_tf_bv": ("smm.transform(so3,vec)", "Ra*vec"),
    }
    ours_expr, oracle_expr = calls[kind]
    if ours:
        snippet = (
            "import time,sys;sys.path.insert(0,'python/spatialmath_mojo');"
            + fixture
            + "import spatialmath_mojo as smm;"
            + "t0=time.perf_counter();"
            + ours_expr
            + ";print(1e3*(time.perf_counter()-t0))"
        )
    else:
        snippet = (
            "import time;"
            + fixture
            + "from spatialmath import SE3,SO3;"
            + "Ta=SE3([se3[i] for i in range(n)]);Tb=SE3([se3b[i] for i in range(n)]);"
            + "Ra=SO3([so3[i] for i in range(n)]);Rb=SO3([so3b[i] for i in range(n)]);"
            + "T0=SE3(se3[0]);"
            + "ptsT=np.ascontiguousarray(pts.T);"
            + "t0=time.perf_counter();"
            + oracle_expr
            + ";print(1e3*(time.perf_counter()-t0))"
        )
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


def warm_latency(call, n_calls: int = BATCH) -> float:
    """Median per-call latency (ms) over N_RUNS batches of n_calls calls."""
    call()  # one warmup batch element outside the timer
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            call()
        samples.append((time.perf_counter() - t0) / n_calls)
    return 1e3 * statistics.median(samples)


def main() -> None:
    import importlib.metadata

    from spatialmath import SE3, SO3

    import spatialmath_mojo as smm

    info = smm.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- oracle: spatialmath-python {importlib.metadata.version('spatialmath-python')}")
    print(f"- spatialmath_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    print(f"- seed: {SEED}; warm: median of {N_RUNS} batches x {BATCH} calls; "
          f"cold: median of {N_RUNS} fresh-process first calls at n={COLD_SIZE:,}")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'spatialmath_mojo'"
        )

    print("\n== correctness gate (element-wise vs spatialmath-python) ==")
    gate = make_inputs(SEED + 1, 2_000)
    ops = make_oracle_operands(SE3, SO3, gate)
    checks = [
        ("se3_compose",
         smm.compose(gate["se3"], gate["se3_b"]),
         np.stack([x.A for x in ops["Ta"] * ops["Tb"]])),
        ("se3_inverse",
         smm.inverse(gate["se3"]),
         np.stack([x.A for x in ops["Ta"].inv()])),
        ("se3_tf_bv",
         smm.transform(gate["se3"], gate["vector"]),
         (ops["Ta"] * gate["vector"]).T),
        ("se3_tf_sp",
         smm.transform(gate["se3"][0], gate["points"]),
         (ops["T0"] * ops["points_T"]).T),
        ("so3_compose",
         smm.compose(gate["so3"], gate["so3_b"]),
         np.stack([x.A for x in ops["Ra"] * ops["Rb"]])),
        ("so3_inverse",
         smm.inverse(gate["so3"]),
         np.stack([x.A for x in ops["Ra"].inv()])),
        ("so3_tf_bv",
         smm.transform(gate["so3"], gate["vector"]),
         (ops["Ra"] * gate["vector"]).T),
    ]
    for name, ours_arr, ref_arr in checks:
        worst = float(np.max(np.abs(ours_arr - ref_arr)))
        status = "OK" if worst <= ATOL else "FAIL"
        print(f"  {name:>28}: max|diff| = {worst:.3e}  [{status}]")
        if worst > ATOL:
            sys.exit(f"correctness gate failed for {name}")

    print(f"\n== cold first-call latency (ms), n={COLD_SIZE:,}, median of {N_RUNS} fresh processes ==")
    print(f"{'operation':>28} | {'spatialmath-python':>18} | {'spatialmath_mojo':>16} | {'speedup':>8}")
    print(f"{'-' * 28}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    cold_rows = []
    for label, kind in OPERATIONS:
        c_ref = cold_first_call(kind, ours=False)
        c_ours = cold_first_call(kind, ours=True)
        cold_rows.append((label, c_ref, c_ours, c_ref / c_ours))
        print(f"{label:>28} | {c_ref:>18.3f} | {c_ours:>16.4f} | {c_ref / c_ours:>7.1f}x")

    print(f"\n== warm steady-state latency (ms/call), median of {N_RUNS} batches ==")
    print(f"{'operation':>28} | {'poses':>8} | {'spatialmath-python':>18} | "
          f"{'spatialmath_mojo':>16} | {'speedup':>8}")
    print(f"{'-' * 28}-+-{'-' * 8}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    warm_rows = []
    for label, kind in OPERATIONS:
        for n in SIZES:
            inputs = make_inputs(SEED + n, n)
            ops = make_oracle_operands(SE3, SO3, inputs)
            t_ref = warm_latency(lambda: oracle_call(ops, kind))
            t_ours = warm_latency(lambda: ours_call(smm, inputs, kind))
            warm_rows.append((label, n, t_ref, t_ours, t_ref / t_ours))
            print(f"{label:>28} | {n:>8,} | {t_ref:>18.3f} | {t_ours:>16.4f} "
                  f"| {t_ref / t_ours:>7.1f}x")

    print("\n== README paste block ==")
    print(f"Cold first call (n={COLD_SIZE:,} poses, median of {N_RUNS} fresh processes):")
    print()
    print("| operation | spatialmath-python (ms) | spatialmath-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|")
    for label, c_ref, c_ours, speedup in cold_rows:
        print(f"| {label} | {c_ref:.3f} | {c_ours:.4f} | {speedup:.1f}x |")
    print()
    print(f"Warm steady state (ms per call, median of {N_RUNS} batches of {BATCH} calls):")
    print()
    print("| operation | poses | spatialmath-python (ms) | spatialmath-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for label, n, t_ref, t_ours, speedup in warm_rows:
        print(f"| {label} | {n:,} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")


if __name__ == "__main__":
    main()
