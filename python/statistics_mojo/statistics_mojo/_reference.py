"""Pure-Python fallback backend: delegation to the CPython stdlib.

The stdlib ``statistics`` module is the reference implementation this
package accelerates, and — being part of the standard library — it is
always importable, so the fallback backend is correct by construction on
every platform (including Windows, where no native kernel is shipped).
The Mojo kernel accelerates the hot paths; everything it does not cover
routes here, on both backends.
"""

from __future__ import annotations

import statistics

__all__ = [
    "mean",
    "fmean",
    "median",
    "median_low",
    "median_high",
    "variance",
    "stdev",
    "pvariance",
    "pstdev",
    "quantiles",
]


def mean(data):
    return statistics.mean(data)


def fmean(data, weights=None):
    return statistics.fmean(data, weights)


def median(data):
    return statistics.median(data)


def median_low(data):
    return statistics.median_low(data)


def median_high(data):
    return statistics.median_high(data)


def variance(data, xbar=None):
    return statistics.variance(data, xbar)


def stdev(data, xbar=None):
    return statistics.stdev(data, xbar)


def pvariance(data, mu=None):
    return statistics.pvariance(data, mu)


def pstdev(data, mu=None):
    return statistics.pstdev(data, mu)


def quantiles(data, *, n=4, method="exclusive"):
    return statistics.quantiles(data, n=n, method=method)
