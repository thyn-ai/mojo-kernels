"""Reproducible dynesty-mojo benchmark: dynesty-mojo vs the PyPI dynesty
sampler on cheap analytic likelihoods (median of 5, cold + warm).

Workloads (both with analytic ground truth, asserted before every timing
run — these are the same problems as the differential suite):

  * gauss10 — 10-D correlated Gaussian likelihood, uniform box prior
    (nlive=500, dlogz=0.25): the "cheap likelihood" regime of dynesty
    issue #432, where sampler-side Python overhead dominates.
  * shells2 — two Gaussian shells in 2-D, bimodal (nlive=250, dlogz=0.2).

Arms:

  * oracle      — dynesty 3.1.0 DynamicNestedSampler, documented defaults
                  (shells uses bootstrap=0: with the default bootstrap the
                  oracle's own UserWarning prescribes bootstrap=0 for
                  multi-modal bounds, and uniform sampling becomes ~20x
                  slower — measured, not tuned away).
  * mojo        — dynesty-mojo native kernel, scalar Python callbacks.
  * mojo-vec    — dynesty-mojo native kernel, vectorized callbacks.
  * fallback    — dynesty-mojo forced NumPy fallback, vectorized.

Protocol: warm = median of 5 in-process runs (fixed seeds); cold = median
of 5 fresh-subprocess first calls (same interpreter, fixed seed). The
correctness gate |logz - logz_true| <= 4*logz_err + 0.15 must pass on the
gate run before that arm is timed.

Run with an interpreter that has numpy + dynesty, e.g. the oracle venv:

    PYTHONPATH=python/dynesty_mojo /tmp/dynesty-oracle-venv/bin/python \
        benchmarks/bench_dynesty.py            # full run (all arms)
    .../python benchmarks/bench_dynesty.py --once gauss10 mojo   # one rep
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# Make the in-repo wrapper importable without installation (benchmark runs
# from the repository; CI installs the wheel instead).
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python", "dynesty_mojo")),
)

import dynesty_mojo  # noqa: E402
from dynesty_mojo import sample  # noqa: E402

REPS = 5
LOGZ_GATE_SIGMA = 4.0
LOGZ_GATE_SLACK = 0.15
SEED = 20260919

# ---------------------------------------------------------------------------
# Workload definitions (fresh, from the literature; see the test suite).
# ---------------------------------------------------------------------------


def make_gauss(ndim=10, nlive=500, dlogz=0.25):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(ndim, ndim))
    cov = (a @ a.T) * 0.005 + np.eye(ndim) * 0.002
    cov_inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    bound = 10.0
    logz_true = (
        0.5 * ndim * np.log(2.0 * np.pi) + 0.5 * logdet - ndim * np.log(2.0 * bound)
    )

    def loglike(v):
        return -0.5 * float(v @ cov_inv @ v)

    def prior_transform(u):
        return 2.0 * bound * u - bound

    def loglike_vec(v):
        return -0.5 * np.einsum("ij,jk,ik->i", v, cov_inv, v)

    def pt_vec(u):
        return 2.0 * bound * u - bound

    return {
        "ndim": ndim,
        "nlive": nlive,
        "dlogz": dlogz,
        "logz_true": logz_true,
        "scalar": (loglike, prior_transform),
        "vectorized": (loglike_vec, pt_vec),
        "oracle_bootstrap": None,
    }


def make_shells(ndim=2, nlive=250, dlogz=0.2):
    r, w, bound = 2.0, 0.1, 6.0
    c1 = np.array([-3.0, 0.0])
    c2 = np.array([3.0, 0.0])

    def loglike(v):
        a = -0.5 * ((np.linalg.norm(v - c1) - r) / w) ** 2
        b = -0.5 * ((np.linalg.norm(v - c2) - r) / w) ** 2
        return float(np.logaddexp(a, b))

    def prior_transform(u):
        return 2.0 * bound * u - bound

    def loglike_vec(v):
        a = -0.5 * ((np.linalg.norm(v - c1, axis=1) - r) / w) ** 2
        b = -0.5 * ((np.linalg.norm(v - c2, axis=1) - r) / w) ** 2
        return np.logaddexp(a, b)

    def pt_vec(u):
        return 2.0 * bound * u - bound

    rho = np.linspace(1e-12, r + 10.0 * w, 2_000_001)
    weight = rho * np.exp(-0.5 * ((rho - r) / w) ** 2)
    radial = np.trapezoid(weight, rho)
    shell_integral = (2.0 * np.pi) / (np.sqrt(2.0 * np.pi) * w) * radial
    z_true = 2.0 * (np.sqrt(2.0 * np.pi) * w) * shell_integral / (2.0 * bound) ** 2

    return {
        "ndim": ndim,
        "nlive": nlive,
        "dlogz": dlogz,
        "logz_true": float(np.log(z_true)),
        "scalar": (loglike, prior_transform),
        "vectorized": (loglike_vec, pt_vec),
        "oracle_bootstrap": 0,
    }


PROBLEMS = {"gauss10": make_gauss, "shells2": make_shells}
ARMS = ["oracle", "mojo", "mojo-vec", "fallback"]


# ---------------------------------------------------------------------------
# Arm runners (one full sampling run; returns (seconds, logz, logz_err, ncall)).
# ---------------------------------------------------------------------------


def run_arm(problem_name: str, arm: str, seed: int):
    prob = PROBLEMS[problem_name]()
    if arm == "oracle":
        from dynesty import DynamicNestedSampler

        kwargs = {"nlive": prob["nlive"], "rstate": np.random.default_rng(seed)}
        if prob["oracle_bootstrap"] is not None:
            kwargs["bootstrap"] = prob["oracle_bootstrap"]
        loglike, pt = prob["scalar"]

        t0 = time.perf_counter()
        sampler = DynamicNestedSampler(loglike, pt, prob["ndim"], **kwargs)
        sampler.run_nested(dlogz_init=prob["dlogz"], print_progress=False)
        dt = time.perf_counter() - t0
        res = sampler.results
        return dt, float(res.logz[-1]), float(res.logzerr[-1]), int(np.sum(res.ncall))

    if arm == "fallback":
        os.environ["DYNESTY_MOJO_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("DYNESTY_MOJO_DISABLE_NATIVE", None)
    vectorized = arm == "mojo-vec" or arm == "fallback"
    loglike, pt = prob["vectorized"] if vectorized else prob["scalar"]

    t0 = time.perf_counter()
    res = sample(
        loglike,
        pt,
        prob["ndim"],
        nlive=prob["nlive"],
        dlogz=prob["dlogz"],
        seed=seed,
        vectorized=vectorized,
    )
    dt = time.perf_counter() - t0
    return dt, res.logz, res.logz_err, res.ncall


def gate(problem_name: str, arm: str) -> None:
    """Correctness gate: the arm must land within 4 sigma (+slack) of the
    analytic logz on its gate run before any timing happens."""
    _, logz, logz_err, _ = run_arm(problem_name, arm, seed=SEED)
    true = PROBLEMS[problem_name]()["logz_true"]
    tol = LOGZ_GATE_SIGMA * logz_err + LOGZ_GATE_SLACK
    if abs(logz - true) > tol:
        raise AssertionError(
            f"gate failed for {problem_name}/{arm}: logz {logz:.4f}±{logz_err:.4f} "
            f"vs truth {true:.4f} (tol {tol:.4f})"
        )


def bench_warm(problem_name: str, arm: str):
    times, ncalls = [], []
    for rep in range(REPS):
        dt, _, _, ncall = run_arm(problem_name, arm, seed=SEED + rep)
        times.append(dt)
        ncalls.append(ncall)
    return float(np.median(times)), int(np.median(ncalls))


def bench_cold(problem_name: str, arm: str):
    """Fresh-subprocess first calls, same interpreter, median of REPS."""
    times = []
    for rep in range(REPS):
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cold-once", problem_name, arm],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.path.pathsep.join(sys.path[:1])},
        )
        payload = json.loads(out.stdout.strip().splitlines()[-1])
        times.append(payload["seconds"])
    return float(np.median(times))


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--cold-once":
        problem_name, arm = sys.argv[2], sys.argv[3]
        dt, logz, logz_err, ncall = run_arm(problem_name, arm, seed=SEED)
        print(json.dumps({"seconds": dt, "logz": logz, "logz_err": logz_err, "ncall": ncall}))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--once":
        problem_name, arm = sys.argv[2], sys.argv[3]
        dt, logz, logz_err, ncall = run_arm(problem_name, arm, seed=SEED)
        print(f"{problem_name}/{arm}: {dt:.3f}s logz {logz:.4f}±{logz_err:.4f} ncall {ncall}")
        return

    info = dynesty_mojo.backend_info()
    print(f"dynesty-mojo {dynesty_mojo.__version__} backend: {info['native_source']}")
    for problem_name in PROBLEMS:
        prob = PROBLEMS[problem_name]()
        print(f"\n## {problem_name} (nlive={prob['nlive']}, dlogz={prob['dlogz']}, "
              f"logz_true={prob['logz_true']:.4f})")
        print("| arm | cold (s) | warm (s) | ncall | speedup (warm) |")
        print("|---|---:|---:|---:|---:|")
        base = None
        for arm in ARMS:
            gate(problem_name, arm)
            warm, ncall = bench_warm(problem_name, arm)
            cold = bench_cold(problem_name, arm)
            if base is None:
                base = warm
            print(f"| {arm} | {cold:.3f} | {warm:.3f} | {ncall} | {base / warm:.2f}x |",
                  flush=True)


if __name__ == "__main__":
    main()
