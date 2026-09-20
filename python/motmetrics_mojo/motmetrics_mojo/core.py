"""Public API: MOTAccumulator + compute — MOTChallenge metrics, drop-in-shaped.

`MOTAccumulator.update` accepts one frame of integer object/hypothesis ids and
their pairwise distance matrix (NaN or +/-inf = do-not-pair), exactly like the
reference `motmetrics.MOTAccumulator`. `compute(acc, metrics=[...])` then
returns the metric values the reference `motmetrics.metrics.compute` would
produce for the same event stream — powered by the Mojo kernel where the
platform supports it, transparently by the vendored pure-Python fallback
everywhere else. Both backends are covered by the same differential suite
against the py-motmetrics 1.4.0 oracle (counts exactly equal; distances /
MOTP within 1e-10).

Unlike the oracle this package does not materialize the per-frame event
DataFrame and does not depend on pandas; `compute` returns a plain mapping
(:class:`MetricSummary`) instead of a one-row pandas DataFrame.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator, Mapping

import numpy as np

from motmetrics_mojo import _native, _reference

#: Metrics computed by `compute(metrics=None)`, in the reference's
#: motchallenge_metrics order.
MOTCHALLENGE_METRICS: tuple[str, ...] = (
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
    "num_transfer",
    "num_ascend",
    "num_migrate",
)

#: Additional scalar metrics supported beyond the MOTChallenge set.
EXTRA_METRICS: tuple[str, ...] = (
    "num_frames",
    "num_objects",
    "num_predictions",
    "num_matches",
    "num_detections",
    "idtp",
    "idfp",
    "idfn",
)

#: Metrics the oracle registers but this package does not compute (they are
#: vector/matrix valued or need the materialized event DataFrame).
UNSUPPORTED_METRICS: tuple[str, ...] = (
    "track_ratios",
    "obj_frequencies",
    "pred_frequencies",
    "id_global_assignment",
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _quiet_divide(a, b) -> float:
    """IEEE float64 division that yields nan/inf instead of raising on 0/0,
    matching the oracle's quiet_divide (warnings suppressed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.true_divide(a, b))


def _metrics_from_counts(c: dict) -> dict:
    """Deterministic scalar metrics from the accumulated counters.

    Every count is an exact integer; ratios are single IEEE-754 float64
    operations — identical on both backends and to the oracle (only `motp`
    involves a floating-point sum and can differ from the oracle's pairwise
    summation by ~1 ulp).
    """
    detections = c["num_matches"] + c["num_switches"]
    return {
        "idf1": _quiet_divide(2 * c["idtp"], c["num_objects"] + c["num_predictions"]),
        "idp": _quiet_divide(c["idtp"], c["idtp"] + c["idfp"]),
        "idr": _quiet_divide(c["idtp"], c["idtp"] + c["idfn"]),
        "recall": _quiet_divide(detections, c["num_objects"]),
        "precision": _quiet_divide(detections, c["num_false_positives"] + detections),
        "num_unique_objects": c["num_unique_objects"],
        "mostly_tracked": c["mostly_tracked"],
        "partially_tracked": c["partially_tracked"],
        "mostly_lost": c["mostly_lost"],
        "num_false_positives": c["num_false_positives"],
        "num_misses": c["num_misses"],
        "num_switches": c["num_switches"],
        "num_fragmentations": c["num_fragmentations"],
        "mota": 1.0
        - _quiet_divide(
            c["num_misses"] + c["num_switches"] + c["num_false_positives"],
            c["num_objects"],
        ),
        "motp": _quiet_divide(c["dist_sum"], detections),
        "num_transfer": c["num_transfer"],
        "num_ascend": c["num_ascend"],
        "num_migrate": c["num_migrate"],
        "num_frames": c["num_frames"],
        "num_objects": c["num_objects"],
        "num_predictions": c["num_predictions"],
        "num_matches": c["num_matches"],
        "num_detections": detections,
        "idtp": c["idtp"],
        "idfp": c["idfp"],
        "idfn": c["idfn"],
    }


class MetricSummary(Mapping):
    """Ordered metric name -> value mapping with attribute access.

    The drop-in equivalent of the one-row pandas DataFrame the oracle's
    `compute` returns (this package has no pandas dependency). Values are
    Python ints (counts) and floats (ratios / MOTP).
    """

    __slots__ = ("_values", "name")

    def __init__(self, values: dict, name=None) -> None:
        self._values = dict(values)
        self.name = name

    def __getitem__(self, key: str):
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_dict(self) -> dict:
        """A plain dict copy of the summary."""
        return dict(self._values)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"MetricSummary({body})"


def _as_id_array(name: str, values) -> np.ndarray:
    """Coerce a 1-D array-like of integer ids to contiguous int64."""
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D array-like of integer ids: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape!r}")
    if arr.dtype.kind in "iu":
        out = arr.astype(np.int64, copy=False)
    else:
        # Accept Python/int-like objects (e.g. lists arrive as int arrays
        # already; object arrays of int are converted element-wise).
        try:
            out = np.array(
                [_as_int_id(f"{name}[{i}]", v) for i, v in enumerate(arr)],
                dtype=np.int64,
            )
        except ValueError as exc:
            raise ValueError(
                f"{name} must contain integer ids (MOTChallenge format); {exc}"
            ) from exc
    return np.ascontiguousarray(out)


def _as_int_id(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} is not an integer id: {value!r}")
    iv = int(value)
    if iv < _INT64_MIN or iv > _INT64_MAX:
        raise ValueError(f"{name} does not fit a signed 64-bit id: {value!r}")
    return iv


class MOTAccumulator:
    """Accumulate tracking events frame by frame (drop-in-shaped).

    Params mirror the reference: `auto_id` auto-increments frame ids starting
    at 0 (passing `frameid` to `update` is then an error, and vice versa);
    `max_switch_time` bounds the frame-id distance over which a re-appearing
    object may still generate SWITCH events (default: no bound).

    Ids must be integers that fit in signed 64 bits (the MOTChallenge format).
    Non-integer ids, which the reference accepts, are out of scope here.
    """

    def __init__(self, auto_id: bool = False, max_switch_time: float = float("inf")) -> None:
        mst = float(max_switch_time)
        if math.isnan(mst) or mst < 0.0:
            raise ValueError(
                f"max_switch_time must be a non-negative number or +inf, got "
                f"{max_switch_time!r}"
            )
        self.auto_id = bool(auto_id)
        self.max_switch_time = mst
        self._last_frameid: int | None = None
        self._impl = None
        self._make_impl()

    def _make_impl(self) -> None:
        try:
            self._impl = _native.NativeAccumulator(self.max_switch_time)
            self._native = True
        except _native.NativeUnavailable:
            self._impl = _reference.RefAccumulator(self.max_switch_time)
            self._native = False

    @property
    def native_backend(self) -> bool:
        """True when this accumulator runs on the native Mojo kernel."""
        return self._native

    def reset(self) -> None:
        """Reset the accumulator to empty state (like the reference)."""
        impl, self._impl = self._impl, None
        if hasattr(impl, "close"):
            impl.close()
        self._last_frameid = None
        self._make_impl()

    def update(self, oids, hids, dists, frameid=None) -> int:
        """Accumulate one frame of objects / hypotheses / pairwise distances.

        `dists` must have exactly ``len(oids) * len(hids)`` entries and is
        interpreted row-major; NaN and +/-inf entries signal do-not-pair
        constellations. Returns the frame id used, like the reference.
        """
        if frameid is None:
            if not self.auto_id:
                raise AssertionError("auto-id is not enabled")
            frameid = 0 if self._last_frameid is None else self._last_frameid + 1
        else:
            if self.auto_id:
                raise AssertionError("Cannot provide frame id when auto-id is enabled")
        frameid = _as_int_id("frameid", frameid)

        o = _as_id_array("oids", oids)
        h = _as_id_array("hids", hids)
        try:
            d = np.asarray(dists, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dists must be an array-like of floats with "
                f"{o.shape[0]} x {h.shape[0]} entries: {exc}"
            ) from exc
        if d.size != o.shape[0] * h.shape[0]:
            raise ValueError(
                f"dists must have exactly len(oids) * len(hids) "
                f"({o.shape[0]} x {h.shape[0]}) entries, got {d.size}"
            )
        d = np.ascontiguousarray(d.reshape(o.shape[0], h.shape[0]))

        if self._native:
            self._impl.update(o, h, d, frameid)
        else:
            self._impl.update(o.tolist(), h.tolist(), d.tolist(), frameid)
        self._last_frameid = frameid
        return frameid

    def counts(self) -> dict:
        """The raw accumulated counters (same for both backends)."""
        return self._impl.counts()


def compute(acc: MOTAccumulator, metrics=None, *, name=None) -> MetricSummary:
    """Compute metrics over an accumulated event stream (drop-in-shaped).

    `metrics` defaults to the MOTChallenge metric set (the reference's
    `motchallenge_metrics`); a single string is treated as a one-element
    list. Returns a :class:`MetricSummary` in the requested order.
    """
    if not isinstance(acc, MOTAccumulator):
        raise TypeError(f"compute expects a motmetrics_mojo.MOTAccumulator, got {type(acc)!r}")
    if metrics is None:
        metrics = list(MOTCHALLENGE_METRICS)
    elif isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)
    values = _metrics_from_counts(acc.counts())
    out = {}
    for mname in metrics:
        if mname in values:
            out[mname] = values[mname]
        elif mname in UNSUPPORTED_METRICS:
            raise NotImplementedError(
                f"metric {mname!r} needs the materialized per-frame event "
                "table, which this package does not build; it computes the "
                "scalar MOTChallenge metrics only (see the package README)"
            )
        else:
            raise ValueError(f"unknown metric {mname!r}")
    return MetricSummary(out, name=name)
