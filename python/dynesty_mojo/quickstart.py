"""End-user quickstart for dynesty-mojo (also the wheel smoke test).

Samples a 3-D correlated Gaussian with a uniform box prior and checks the
evidence against its analytic value. Runs on either backend — native where
the wheel carries the Mojo kernel, pure NumPy everywhere else — with no
Mojo toolchain and no dynesty installation required.

    python quickstart.py
"""

import numpy as np

from dynesty_mojo import backend_info, sample

ndim = 3
# Correlated Gaussian likelihood: logL(v) = -0.5 v^T C^-1 v.
rng = np.random.default_rng(0)
a = rng.normal(size=(ndim, ndim))
cov = (a @ a.T) * 0.01 + np.eye(ndim) * 0.005
cov_inv = np.linalg.inv(cov)
_, logdet = np.linalg.slogdet(cov)

B = 10.0  # uniform prior on [-B, B]^ndim (covers the likelihood mass)


def loglike(v):
    return -0.5 * float(v @ cov_inv @ v)


def prior_transform(u):
    return 2.0 * B * u - B


def main() -> None:
    info = backend_info()
    res = sample(loglike, prior_transform, ndim, nlive=200, dlogz=0.25, seed=42)

    # Analytic evidence: Z = (2 pi)^(d/2) sqrt(det C) / (2 B)^d.
    logz_true = 0.5 * ndim * np.log(2.0 * np.pi) + 0.5 * logdet - ndim * np.log(2.0 * B)

    mean = res.weights @ res.samples
    print(f"backend:            {res.backend} (native_available={info['native_available']})")
    print(f"logz:               {res.logz:.4f} +/- {res.logz_err:.4f}")
    print(f"logz (analytic):    {logz_true:.4f}")
    print(f"posterior mean:     {np.array2string(mean, precision=4)} (truth: zeros)")
    print(f"likelihood calls:   {res.ncall} for {res.niter} iterations")
    print(f"evidence checksum:  {res.logz:.6f}")

    tol = 4.0 * res.logz_err + 0.15
    assert abs(res.logz - logz_true) <= tol, (
        f"logz {res.logz:.4f} is {abs(res.logz - logz_true):.4f} from the analytic "
        f"{logz_true:.4f}, beyond {tol:.4f}"
    )
    assert np.all(np.abs(mean) < 0.1), f"posterior mean off: {mean}"
    print("quickstart OK")


if __name__ == "__main__":
    main()
