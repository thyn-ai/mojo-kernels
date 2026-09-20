"""Drop-in nuScenes 3D object detection evaluation, API-shaped after DetectionEval.

`evaluate(gt, predictions)` runs the detection metric on plain-dict corpora —
ground truth in an extracted-annotation form and predictions in the exact
nuScenes submission JSON shape — and returns a result dict with the same
schema as `nuscenes.eval.detection.data_classes.DetectionMetrics.serialize()`
(`label_aps`, `mean_dist_aps`, `mean_ap`, `label_tp_errors`, `tp_errors`,
`tp_scores`, `nd_score`, `eval_time`, `cfg`).

The x40 fan-out of greedy matching passes (10 classes x 4 distance
thresholds) runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back to
the vendored pure-Python implementation otherwise. Both backends share box
loading, filtering, grouping, and all metric-data post-processing in this
module, so results are identical either way; the differential test suite
asserts agreement with nuscenes-devkit on both paths.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from nuscenes_eval_mojo import _reference
from nuscenes_eval_mojo._native import NativeUnavailable, native_match

DETECTION_NAMES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)
ATTRIBUTE_NAMES = (
    "pedestrian.moving",
    "pedestrian.sitting_lying_down",
    "pedestrian.standing",
    "cycle.with_rider",
    "cycle.without_rider",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
)
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
NELEM = 101  # DetectionMetricData.nelem: 101 recall steps, 0% to 100%.

_CLASS_TO_ID = {name: idx for idx, name in enumerate(DETECTION_NAMES)}
_ATTR_TO_ID = {name: idx for idx, name in enumerate(ATTRIBUTE_NAMES)}
_BARRIER_ID = _CLASS_TO_ID["barrier"]

# Default label mapping from nuScenes categories to detection classes
# (movable_object / vehicle / human categories only; everything else,
# including static_object.bicycle_rack, maps to None and is not evaluated).
_CATEGORY_TO_DETECTION = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
_BIKE_RACK_CATEGORY = "static_object.bicycle_rack"

# The CVPR 2019 detection configuration (the devkit's default eval config).
_DEFAULT_CLASS_RANGE = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30,
}

# Classes x TP metrics that the reference scores as NaN (not evaluated).
_TP_NAN_RULES = {
    ("traffic_cone", "attr_err"),
    ("traffic_cone", "vel_err"),
    ("traffic_cone", "orient_err"),
    ("barrier", "attr_err"),
    ("barrier", "vel_err"),
}


class EvaluationError(ValueError):
    """Structured input/config validation error.

    Carries a stable snake_case `code` and a `details` dict so callers can
    handle failures programmatically.
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class DetectionConfig:
    """Evaluation settings; defaults are the devkit's detection_cvpr_2019 config."""

    class_range: dict
    dist_fcn: str = "center_distance"
    dist_ths: tuple = (0.5, 1.0, 2.0, 4.0)
    dist_th_tp: float = 2.0
    min_recall: float = 0.1
    min_precision: float = 0.1
    max_boxes_per_sample: int = 500
    mean_ap_weight: int = 5

    def __post_init__(self) -> None:
        if set(self.class_range.keys()) != set(DETECTION_NAMES):
            raise EvaluationError(
                "class_count_mismatch",
                "class_range must cover exactly the 10 detection classes",
                {"class_range_keys": sorted(self.class_range.keys())},
            )
        if any(not _is_real_number(v) or v <= 0 for v in self.class_range.values()):
            raise EvaluationError(
                "invalid_class_range", "class_range values must be positive numbers"
            )
        if self.dist_fcn != "center_distance":
            raise EvaluationError(
                "unknown_dist_fcn",
                f"unsupported distance function {self.dist_fcn!r}; only 'center_distance'",
            )
        if not self.dist_ths or any(not _is_real_number(t) or t <= 0 for t in self.dist_ths):
            raise EvaluationError(
                "invalid_dist_ths", "dist_ths must be a non-empty sequence of positive numbers"
            )
        if self.dist_th_tp not in self.dist_ths:
            raise EvaluationError(
                "dist_th_tp_not_in_dist_ths", "dist_th_tp must be in the set of dist_ths"
            )
        if not 0 <= self.min_recall <= 1:
            raise EvaluationError("invalid_min_recall", "min_recall must be in [0, 1]")
        if not 0 <= self.min_precision < 1:
            raise EvaluationError("invalid_min_precision", "min_precision must be in [0, 1)")
        if not isinstance(self.max_boxes_per_sample, int) or self.max_boxes_per_sample <= 0:
            raise EvaluationError(
                "invalid_max_boxes", "max_boxes_per_sample must be a positive int"
            )

    @classmethod
    def default(cls) -> "DetectionConfig":
        """The CVPR 2019 configuration used by the official detection eval."""
        return cls(class_range=dict(_DEFAULT_CLASS_RANGE))

    @property
    def class_names(self) -> tuple:
        return DETECTION_NAMES

    def serialize(self) -> dict:
        """Same shape as DetectionConfig.serialize() in the devkit."""
        return {
            "class_range": dict(self.class_range),
            "dist_fcn": self.dist_fcn,
            "dist_ths": list(self.dist_ths),
            "dist_th_tp": self.dist_th_tp,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }


def _is_real_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class _Box:
    """One parsed detection box (gt or prediction), pre-matching."""

    __slots__ = (
        "sample_token",
        "cls_id",
        "x",
        "y",
        "z",
        "w",
        "l",
        "h",
        "yaw",
        "vx",
        "vy",
        "attr_id",
        "score",
        "ego_x",
        "ego_y",
        "num_pts",
    )

    def numeric_record(self) -> tuple:
        """The kernel's 8-double box layout: x,y,w,l,h,yaw,vx,vy."""
        return (self.x, self.y, self.w, self.l, self.h, self.yaw, self.vx, self.vy)


def _yaw_from_quaternion(rotation) -> float:
    """Yaw of a box quaternion: atan2 of the xy projection of its x-axis.

    Mirrors the reference (pyquaternion): the quaternion is normalized first
    unless its norm is already 1 within 1e-14; a zero quaternion yields yaw 0.
    """
    w, x, y, z = (float(v) for v in rotation)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not abs(n - 1.0) < 1e-14 and n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n
    v0 = w * w + x * x - y * y - z * z
    v1 = 2.0 * (x * y + w * z)
    return math.atan2(v1, v0)


def _require_keys(record: dict, keys: tuple, ctx: str) -> None:
    for key in keys:
        if key not in record:
            raise EvaluationError("missing_field", f"{ctx}: missing required field {key!r}")


def _float_triplet(value, field: str, ctx: str, allow_nan: bool = False) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise EvaluationError(
            "invalid_field", f"{ctx}: {field} must have exactly 3 elements"
        )
    out = tuple(float(v) for v in value)
    if not allow_nan and any(math.isnan(v) for v in out):
        raise EvaluationError("nan_field", f"{ctx}: {field} may not be NaN")
    return out


def _float_pair(value, field: str, ctx: str) -> tuple:
    """2-element float field; NaN allowed (database velocities can be NaN)."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EvaluationError("invalid_field", f"{ctx}: {field} must have exactly 2 elements")
    return (float(value[0]), float(value[1]))


def _rotation(value, ctx: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EvaluationError("invalid_field", f"{ctx}: rotation must have exactly 4 elements")
    out = tuple(float(v) for v in value)
    if any(math.isnan(v) for v in out):
        raise EvaluationError("nan_field", f"{ctx}: rotation may not be NaN")
    return out


def _attribute_id(name: str, ctx: str) -> int:
    """Attribute name -> id, or -1 for 'no attribute' (the empty string)."""
    if name == "":
        return -1
    if name not in _ATTR_TO_ID:
        raise EvaluationError(
            "unknown_attribute", f"{ctx}: unknown attribute_name {name!r}",
            {"attribute_name": name},
        )
    return _ATTR_TO_ID[name]


def _parse_common_fields(record: dict, ctx: str) -> dict:
    _require_keys(record, ("translation", "size", "rotation", "velocity"), ctx)
    translation = _float_triplet(record["translation"], "translation", ctx)
    size = _float_triplet(record["size"], "size", ctx)
    if any(not math.isfinite(v) or v <= 0 for v in size):
        raise EvaluationError(
            "invalid_size", f"{ctx}: size elements must be positive and finite"
        )
    rotation = _rotation(record["rotation"], ctx)
    vx, vy = _float_pair(record["velocity"], "velocity", ctx)
    return {
        "x": translation[0],
        "y": translation[1],
        "z": translation[2],
        "w": size[0],
        "l": size[1],
        "h": size[2],
        "yaw": _yaw_from_quaternion(rotation),
        "vx": vx,
        "vy": vy,
    }


def _load_json_arg(value, what: str) -> dict:
    """Accept an in-memory dict or a filesystem path to a JSON document."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, os.PathLike)):
        with open(value, encoding="utf-8") as handle:
            return json.load(handle)
    raise EvaluationError(
        "invalid_input_type", f"{what} must be a dict or a path to a JSON file",
        {"type": type(value).__name__},
    )


def _load_gt(gt: dict) -> tuple[list, list, list]:
    """Parse the gt corpus: (sample records, boxes per sample, bike racks per sample).

    Schema::

        {"samples": [{"token": str,
                      "ego_translation": [x, y, z],
                      "annotations": [{"category_name", "translation", "size",
                                       "rotation", "velocity", "num_lidar_pts",
                                       "num_radar_pts", "attribute_name"?}, ...]},
                      ...]}

    This is the extracted form of what load_gt() pulls out of a NuScenes DB:
    category names are mapped to detection classes here (unmapped categories
    are dropped, bike racks are retained for bike-rack filtering).
    """
    samples = gt.get("samples")
    if not isinstance(samples, list) or len(samples) == 0:
        raise EvaluationError("empty_gt", "gt must contain a non-empty 'samples' list")
    seen_tokens: set = set()
    sample_records, boxes_per_sample, racks_per_sample = [], [], []
    for sample in samples:
        _require_keys(sample, ("token", "ego_translation"), "gt sample")
        token = sample["token"]
        if not isinstance(token, str):
            raise EvaluationError("invalid_sample_token", "gt sample token must be a string")
        if token in seen_tokens:
            raise EvaluationError(
                "duplicate_sample", f"duplicate gt sample token {token!r}", {"token": token}
            )
        seen_tokens.add(token)
        ego = _float_triplet(sample["ego_translation"], "ego_translation", f"gt sample {token}")
        annotations = sample.get("annotations", [])
        if not isinstance(annotations, list):
            raise EvaluationError(
                "invalid_annotations", f"gt sample {token}: annotations must be a list"
            )
        boxes, racks = [], []
        for idx, ann in enumerate(annotations):
            ctx = f"gt sample {token} annotation {idx}"
            _require_keys(ann, ("category_name",), ctx)
            category = ann["category_name"]
            fields = _parse_common_fields(ann, ctx)
            if category == _BIKE_RACK_CATEGORY:
                # Kept only for bike-rack filtering; never evaluated.
                _require_keys(ann, ("rotation",), ctx)
                racks.append((fields["x"], fields["y"], fields["z"],
                              fields["w"], fields["l"], fields["h"],
                              _rotation(ann["rotation"], ctx)))
                continue
            detection_name = _CATEGORY_TO_DETECTION.get(category)
            if detection_name is None:
                continue  # not a detection class: dropped like the reference does
            _require_keys(ann, ("num_lidar_pts", "num_radar_pts"), ctx)
            num_lidar_pts, num_radar_pts = ann["num_lidar_pts"], ann["num_radar_pts"]
            if not isinstance(num_lidar_pts, int) or not isinstance(num_radar_pts, int):
                raise EvaluationError("invalid_num_pts", f"{ctx}: point counts must be ints")
            box = _Box()
            box.sample_token = token
            box.cls_id = _CLASS_TO_ID[detection_name]
            box.attr_id = _attribute_id(ann.get("attribute_name", ""), ctx)
            box.score = -1.0  # GT samples do not have a score
            box.num_pts = num_lidar_pts + num_radar_pts
            for key, value in fields.items():
                setattr(box, key, value)
            # add_center_dist: box translation minus the sample's ego pose.
            box.ego_x = box.x - ego[0]
            box.ego_y = box.y - ego[1]
            boxes.append(box)
        sample_records.append({"token": token, "ego": ego})
        boxes_per_sample.append(boxes)
        racks_per_sample.append(racks)
    return sample_records, boxes_per_sample, racks_per_sample


def _load_predictions(predictions: dict, cfg: DetectionConfig) -> tuple[list, list]:
    """Parse the nuScenes submission format: (ordered tokens, boxes per token).

    Schema::

        {"meta": {...}?, "results": {sample_token: [{"sample_token", "translation",
            "size", "rotation", "velocity", "detection_name", "detection_score",
            "attribute_name", "ego_translation"?, "num_pts"?}, ...]}}
    """
    if "results" not in predictions:
        raise EvaluationError(
            "missing_results",
            "prediction file must contain a 'results' field (the nuScenes submission format)",
        )
    results = predictions["results"]
    if not isinstance(results, dict) or len(results) == 0:
        raise EvaluationError("empty_results", "'results' must be a non-empty object")
    tokens, boxes_per_sample = [], []
    for token, records in results.items():
        if len(records) > cfg.max_boxes_per_sample:
            raise EvaluationError(
                "too_many_boxes",
                f"sample {token}: only <= {cfg.max_boxes_per_sample} boxes per sample allowed",
                {"sample_token": token, "count": len(records)},
            )
        boxes = []
        for idx, record in enumerate(records):
            ctx = f"prediction sample {token} box {idx}"
            _require_keys(
                record, ("sample_token", "detection_name", "detection_score", "attribute_name"),
                ctx,
            )
            detection_name = record["detection_name"]
            if detection_name not in _CLASS_TO_ID:
                raise EvaluationError(
                    "unknown_detection_name", f"{ctx}: unknown detection_name {detection_name!r}",
                    {"detection_name": detection_name},
                )
            score = float(record["detection_score"])
            if math.isnan(score):
                raise EvaluationError("nan_score", f"{ctx}: detection_score may not be NaN")
            box = _Box()
            # The box's own token drives gt lookup during matching, exactly
            # like the reference (EvalBoxes groups by the outer key, but
            # accumulate() indexes ground truth by the box's token).
            box.sample_token = record["sample_token"]
            if not isinstance(box.sample_token, str):
                raise EvaluationError(
                    "invalid_sample_token", f"{ctx}: sample_token must be a string"
                )
            box.cls_id = _CLASS_TO_ID[detection_name]
            box.attr_id = _attribute_id(record["attribute_name"], ctx)
            box.score = score
            box.num_pts = int(record.get("num_pts", -1))
            for key, value in _parse_common_fields(record, ctx).items():
                setattr(box, key, value)
            boxes.append(box)
        tokens.append(token)
        boxes_per_sample.append(boxes)
    return tokens, boxes_per_sample


def _rotation_matrix(rotation) -> np.ndarray:
    """3x3 rotation matrix of a box quaternion (normalized like the reference)."""
    w, x, y, z = (float(v) for v in rotation)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not abs(n - 1.0) < 1e-14 and n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )


def _point_in_box(box, point) -> bool:
    """Whether a 3D point lies inside an oriented box (corners projection test)."""
    cx, cy, cz, w, l, h, rotation = box
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.dot(_rotation_matrix(rotation), np.vstack((x_corners, y_corners, z_corners)))
    corners[0, :] = corners[0, :] + cx
    corners[1, :] = corners[1, :] + cy
    corners[2, :] = corners[2, :] + cz
    p1, p_x, p_y, p_z = corners[:, 0], corners[:, 4], corners[:, 1], corners[:, 3]
    i, j, k = p_x - p1, p_y - p1, p_z - p1
    v = np.array(point) - p1
    iv, jv, kv = np.dot(i, v), np.dot(j, v), np.dot(k, v)
    return bool(0 <= iv <= np.dot(i, i) and 0 <= jv <= np.dot(j, j) and 0 <= kv <= np.dot(k, k))


def _filter_sample_boxes(boxes: list, racks: list, class_range: dict) -> list:
    """Distance, zero-points, and bike-rack filtering (reference filter_eval_boxes)."""
    out = []
    for box in boxes:
        ego_dist = math.sqrt(box.ego_x * box.ego_x + box.ego_y * box.ego_y)
        if not ego_dist < class_range[DETECTION_NAMES[box.cls_id]]:
            continue
        if box.num_pts == 0:
            continue
        if DETECTION_NAMES[box.cls_id] in ("bicycle", "motorcycle"):
            in_a_rack = any(_point_in_box(rack, (box.x, box.y, box.z)) for rack in racks)
            if in_a_rack:
                continue
        out.append(box)
    return out


def _cummean(x: np.ndarray) -> np.ndarray:
    """Cumulative mean, NaN-sensitive: all-NaN input yields an array of ones."""
    if sum(np.isnan(x)) == len(x):
        return np.ones(len(x))
    sum_vals = np.nancumsum(x.astype(float))
    count_vals = np.cumsum(~np.isnan(x))
    return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def _no_predictions_md() -> dict:
    """Metric data corresponding to no predictions (reference no_predictions())."""
    return {
        "recall": np.linspace(0, 1, NELEM),
        "precision": np.zeros(NELEM),
        "confidence": np.zeros(NELEM),
        "trans_err": np.ones(NELEM),
        "vel_err": np.ones(NELEM),
        "scale_err": np.ones(NELEM),
        "orient_err": np.ones(NELEM),
        "attr_err": np.ones(NELEM),
    }


def _metric_data_from_matches(npos: int, conf_sorted: np.ndarray, match_rows) -> dict:
    """Precision/recall interpolation and TP-error resampling for one (class, th).

    `match_rows` are the kernel's per-prediction rows (tp flag + 5 errors) in
    the reference's confidence-sorted prediction order.
    """
    tp_flags = match_rows[:, 0]
    matched = tp_flags == 1.0
    if not np.any(matched):
        return _no_predictions_md()

    tp = np.cumsum(tp_flags).astype(float)
    fp = np.cumsum(1.0 - tp_flags).astype(float)
    conf = conf_sorted.astype(float)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, NELEM)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    match_conf = conf_sorted[matched].astype(float)
    # Kernel output columns per prediction row:
    # [tp, trans_err, vel_err, scale_err, orient_err, attr_err].
    err_cols = {"trans_err": 1, "vel_err": 2, "scale_err": 3, "orient_err": 4, "attr_err": 5}
    errors = {}
    for key in TP_METRICS:
        errs = match_rows[matched, err_cols[key]]
        tmp = _cummean(np.array(errs))
        # Interpolate based on the confidences (reversed: interp needs increasing xp).
        errors[key] = np.interp(conf[::-1], match_conf[::-1], tmp[::-1])[::-1]

    return {
        "recall": rec,
        "precision": prec,
        "confidence": conf,
        "trans_err": errors["trans_err"],
        "vel_err": errors["vel_err"],
        "scale_err": errors["scale_err"],
        "orient_err": errors["orient_err"],
        "attr_err": errors["attr_err"],
    }


def _calc_ap(md: dict, min_recall: float, min_precision: float) -> float:
    """Average precision over recall thresholds above min_recall/min_precision."""
    prec = np.copy(md["precision"])
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 excludes the min bin.
    prec -= min_precision  # Clip low precision.
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def _max_recall_ind(md: dict) -> int:
    """Index of max recall achieved: last instance of confidence > 0."""
    non_zero = np.nonzero(md["confidence"])[0]
    if len(non_zero) == 0:
        return 0
    return int(non_zero[-1])


def _calc_tp(md: dict, min_recall: float, metric_name: str) -> float:
    """True-positive error: mean error between min recall and max achieved recall."""
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = _max_recall_ind(md)
    if last_ind < first_ind:
        return 1.0  # No prediction at min_recall: assign error 1 (score 0).
    return float(np.mean(md[metric_name][first_ind: last_ind + 1]))


def _run_matching(cfg, n_samples, arrays) -> tuple[np.ndarray, str]:
    """Run all (class, threshold) matching passes; native first, fallback second."""
    kwargs = dict(arrays, n_classes=len(DETECTION_NAMES), n_samples=n_samples)
    try:
        return native_match(**kwargs), "native"
    except NativeUnavailable:
        return _reference.reference_match(**kwargs), "fallback"


def evaluate(gt, predictions, config: DetectionConfig | None = None) -> dict:
    """Evaluate 3D detection predictions against ground truth. Drop-in-shaped.

    :param gt: Ground-truth corpus (dict, or path to a JSON file) — see
        :func:`_load_gt` for the schema.
    :param predictions: Predictions in the nuScenes submission JSON format
        (dict, or path to a JSON file).
    :param config: Evaluation settings; defaults to the CVPR 2019 detection
        configuration (the devkit's default).
    :return: A dict with the DetectionMetrics.serialize() schema.
    """
    start_time = time.time()
    cfg = config if config is not None else DetectionConfig.default()

    gt = _load_json_arg(gt, "gt")
    predictions = _load_json_arg(predictions, "predictions")

    # -----------------------------------
    # Load, validate, and filter boxes.
    # -----------------------------------
    sample_records, gt_per_sample, racks_per_sample = _load_gt(gt)
    sample_index = {rec["token"]: idx for idx, rec in enumerate(sample_records)}
    pred_tokens, pred_per_sample = _load_predictions(predictions, cfg)

    if set(pred_tokens) != set(sample_index):
        raise EvaluationError(
            "sample_token_mismatch",
            "samples in ground truth don't match samples in predictions",
            {
                "only_in_gt": sorted(set(sample_index) - set(pred_tokens))[:5],
                "only_in_predictions": sorted(set(pred_tokens) - set(sample_index))[:5],
            },
        )
    n_samples = len(sample_records)

    # add_center_dist for predictions uses the *grouping* sample's ego pose.
    for token, boxes in zip(pred_tokens, pred_per_sample):
        ego = sample_records[sample_index[token]]["ego"]
        for box in boxes:
            box.ego_x = box.x - ego[0]
            box.ego_y = box.y - ego[1]
            if box.sample_token not in sample_index:
                raise EvaluationError(
                    "unknown_sample_token",
                    f"prediction box sample_token {box.sample_token!r} is not a gt sample",
                    {"sample_token": box.sample_token},
                )

    gt_filtered = [
        _filter_sample_boxes(boxes, racks, cfg.class_range)
        for boxes, racks in zip(gt_per_sample, racks_per_sample)
    ]
    pred_filtered = [
        _filter_sample_boxes(boxes, racks_per_sample[sample_index[token]], cfg.class_range)
        for token, boxes in zip(pred_tokens, pred_per_sample)
    ]

    # -----------------------------------
    # Group per class; sort predictions by confidence (reference order).
    # -----------------------------------
    arrays, per_class = _build_kernel_arrays(
        sample_records, gt_filtered, pred_tokens, pred_filtered, cfg
    )

    # -----------------------------------
    # Match (x40 fan-out) and compute metric data.
    # -----------------------------------
    out, _backend = _run_matching(cfg, n_samples, arrays)
    dist_ths = [float(t) for t in cfg.dist_ths]

    md = {}
    for cls_id, class_name in enumerate(DETECTION_NAMES):
        npos = per_class["npos"][cls_id]
        conf_sorted = per_class["conf"][cls_id]
        p0, p1 = per_class["pred_class_offsets"][cls_id: cls_id + 2]
        for t, dist_th in enumerate(dist_ths):
            if npos == 0 or p1 == p0:
                # No positives, or none of this class predicted. Note: with no
                # predictions at all the reference accumulates zero matches and
                # lands on the same no-predictions metric data.
                md[(class_name, dist_th)] = _no_predictions_md()
                continue
            rows = out[t, p0:p1]
            md[(class_name, dist_th)] = _metric_data_from_matches(npos, conf_sorted, rows)

    # -----------------------------------
    # Calculate metrics from the data.
    # -----------------------------------
    metrics = _assemble_metrics(cfg, md, dist_ths)
    metrics["eval_time"] = time.time() - start_time
    return metrics


def _build_kernel_arrays(sample_records, gt_filtered, pred_tokens, pred_filtered, cfg):
    """Flatten grouped boxes into the kernel's numeric arrays.

    GT is grouped per class and per sample (CSR), preserving in-sample order.
    Predictions are grouped per class in the reference's confidence-sorted
    order: descending score, ties broken by descending insertion index (the
    insertion order follows the predictions' sample order).
    """
    n_samples = len(sample_records)
    sample_index = {rec["token"]: idx for idx, rec in enumerate(sample_records)}

    gt_class_offsets = [0]
    gt_sample_offsets: list[int] = []
    gt_vals: list[float] = []
    gt_attr: list[int] = []
    pred_class_offsets = [0]
    pred_vals: list[float] = []
    pred_attr: list[int] = []
    pred_sample: list[int] = []
    npos_per_class: list[int] = []
    conf_per_class: list[np.ndarray] = []

    for cls_id in range(len(DETECTION_NAMES)):
        # GT: per-sample CSR over the canonical (gt) sample order.
        running = len(gt_attr)
        for s in range(n_samples):
            gt_sample_offsets.append(running)
            for box in gt_filtered[s]:
                if box.cls_id != cls_id:
                    continue
                gt_vals.extend(box.numeric_record())
                gt_attr.append(box.attr_id)
                running += 1
        gt_sample_offsets.append(running)
        gt_class_offsets.append(running)
        npos_per_class.append(running - gt_class_offsets[-2])

        # Predictions of this class, in insertion order, then reference sort.
        class_preds = []
        for boxes in pred_filtered:
            for box in boxes:
                if box.cls_id == cls_id:
                    class_preds.append(box)
        confs = [box.score for box in class_preds]
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(confs))][::-1]
        for i in sortind:
            box = class_preds[i]
            pred_vals.extend(box.numeric_record())
            pred_attr.append(box.attr_id)
            pred_sample.append(sample_index[box.sample_token])
        pred_class_offsets.append(len(pred_attr))
        conf_per_class.append(np.array([class_preds[i].score for i in sortind]))

    periods = np.array(
        [math.pi if cls_id == _BARRIER_ID else 2 * math.pi for cls_id in range(len(DETECTION_NAMES))],
        dtype=np.float64,
    )
    arrays = {
        "gt_class_offsets": np.array(gt_class_offsets, dtype=np.int64),
        "gt_sample_offsets": np.array(gt_sample_offsets, dtype=np.int64),
        "gt_vals": np.array(gt_vals, dtype=np.float64),
        "gt_attr": np.array(gt_attr, dtype=np.int32),
        "pred_class_offsets": np.array(pred_class_offsets, dtype=np.int64),
        "pred_sample": np.array(pred_sample, dtype=np.int32),
        "pred_vals": np.array(pred_vals, dtype=np.float64),
        "pred_attr": np.array(pred_attr, dtype=np.int32),
        "periods": periods,
        "dist_ths": np.array([float(t) for t in cfg.dist_ths], dtype=np.float64),
    }
    per_class = {
        "npos": npos_per_class,
        "conf": conf_per_class,
        "pred_class_offsets": pred_class_offsets,
    }
    return arrays, per_class


def _assemble_metrics(cfg: DetectionConfig, md: dict, dist_ths: list) -> dict:
    """AP/TP aggregation with the DetectionMetrics.serialize() schema."""
    label_aps: dict = {}
    label_tp_errors: dict = {}
    for class_name in cfg.class_names:
        label_aps[class_name] = {
            dist_th: _calc_ap(md[(class_name, dist_th)], cfg.min_recall, cfg.min_precision)
            for dist_th in dist_ths
        }
        label_tp_errors[class_name] = {}
        for metric_name in TP_METRICS:
            if (class_name, metric_name) in _TP_NAN_RULES:
                tp = np.nan
            else:
                tp = _calc_tp(md[(class_name, cfg.dist_th_tp)], cfg.min_recall, metric_name)
            label_tp_errors[class_name][metric_name] = tp

    mean_dist_aps = {
        class_name: float(np.mean(list(aps.values())))
        for class_name, aps in label_aps.items()
    }
    mean_ap = float(np.mean(list(mean_dist_aps.values())))
    tp_errors = {
        metric_name: float(
            np.nanmean([label_tp_errors[class_name][metric_name] for class_name in cfg.class_names])
        )
        for metric_name in TP_METRICS
    }
    tp_scores = {
        metric_name: max(0.0, 1.0 - tp_errors[metric_name]) for metric_name in TP_METRICS
    }
    nd_score = float(
        cfg.mean_ap_weight * mean_ap + np.sum(list(tp_scores.values()))
    ) / float(cfg.mean_ap_weight + len(tp_scores))

    return {
        "label_aps": label_aps,
        "mean_dist_aps": mean_dist_aps,
        "mean_ap": mean_ap,
        "label_tp_errors": label_tp_errors,
        "tp_errors": tp_errors,
        "tp_scores": tp_scores,
        "nd_score": nd_score,
        "cfg": cfg.serialize(),
    }
