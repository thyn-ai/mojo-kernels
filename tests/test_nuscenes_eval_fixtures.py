"""Shared fixtures for the nuscenes-eval-mojo differential suite.

Deterministic seeded synthetic box corpora plus an oracle harness that drives
the real nuscenes-devkit reference code (DetectionBox/EvalBoxes construction,
the devkit's own add_center_dist / filter_eval_boxes / accumulate / calc_ap /
calc_tp, and DetectionMetrics assembly — the same steps DetectionEval runs
after DB loading) so the full metrics dicts can be compared end to end.

Everything is generated locally from explicit seeds — no network, no dataset
downloads — so the suite is bit-reproducible on any machine.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np

SCORE_ATOL = 1e-8  # documented tolerance; results are typically bit-identical

DETECTION_NAMES = (
    "car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
    "motorcycle", "bicycle", "traffic_cone", "barrier",
)

# One category per detection class for corpus generation (bus alternates
# between bendy/rigid, pedestrian cycles through the four human categories).
_CLASS_TO_CATEGORIES = {
    "car": ["vehicle.car"],
    "truck": ["vehicle.truck"],
    "bus": ["vehicle.bus.bendy", "vehicle.bus.rigid"],
    "trailer": ["vehicle.trailer"],
    "construction_vehicle": ["vehicle.construction"],
    "pedestrian": [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    ],
    "motorcycle": ["vehicle.motorcycle"],
    "bicycle": ["vehicle.bicycle"],
    "traffic_cone": ["movable_object.trafficcone"],
    "barrier": ["movable_object.barrier"],
}

_CLASS_ATTRIBUTES = {
    "pedestrian": ["pedestrian.moving", "pedestrian.sitting_lying_down", "pedestrian.standing"],
    "bicycle": ["cycle.with_rider", "cycle.without_rider"],
    "motorcycle": ["cycle.with_rider", "cycle.without_rider"],
    "car": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "bus": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "construction_vehicle": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "trailer": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "truck": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "barrier": [""],
    "traffic_cone": [""],
}

# Rough real-world size ranges (w, l, h) per class.
_CLASS_SIZE_RANGES = {
    "car": ((1.6, 2.1), (3.6, 5.2), (1.4, 1.9)),
    "truck": ((2.2, 2.8), (6.0, 11.0), (2.4, 3.8)),
    "bus": ((2.4, 2.8), (9.0, 13.0), (2.8, 3.6)),
    "trailer": ((2.3, 2.7), (7.0, 14.0), (2.6, 4.0)),
    "construction_vehicle": ((2.2, 3.2), (5.0, 9.0), (2.4, 3.9)),
    "pedestrian": ((0.5, 0.8), (0.5, 0.9), (1.5, 1.95)),
    "motorcycle": ((0.6, 0.9), (1.8, 2.4), (1.2, 1.7)),
    "bicycle": ((0.5, 0.7), (1.5, 2.0), (1.0, 1.4)),
    "traffic_cone": ((0.3, 0.5), (0.3, 0.5), (0.6, 1.0)),
    "barrier": ((0.3, 0.6), (1.5, 2.5), (0.8, 1.2)),
}


def _yaw_quaternion(rng: random.Random) -> list:
    """Unit quaternion for a random yaw with a small random roll/pitch."""
    yaw = rng.uniform(-math.pi, math.pi)
    roll = rng.uniform(-0.05, 0.05)
    pitch = rng.uniform(-0.05, 0.05)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # ZYX intrinsic composition.
    return [
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ]


def _velocity(rng: random.Random, allow_nan: bool) -> list:
    if allow_nan and rng.random() < 0.08:
        return [float("nan"), float("nan")]
    return [rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)]


def _score(rng: random.Random, ties: bool) -> float:
    if ties and rng.random() < 0.3:
        # Coarse grid -> frequent exact confidence ties (tie-break coverage).
        return round(rng.uniform(0.05, 0.99), 1)
    return rng.uniform(0.01, 0.99)


def make_corpus(
    seed: int,
    n_samples: int = 24,
    gt_per_sample: tuple = (4, 14),
    pred_per_gt: float = 1.35,
    fp_rate: float = 0.5,
    drop_classes: tuple = (),
    no_pred_classes: tuple = (),
    no_match_classes: tuple = (),
    ties: bool = False,
    bike_racks: bool = True,
    shuffle_pred_samples: bool = True,
) -> tuple:
    """Deterministic synthetic (gt, predictions) corpus.

    - gt: the extracted-annotation schema of nuscenes_eval_mojo.evaluate.
    - predictions: the nuScenes submission JSON shape.
    - drop_classes: classes with zero gt boxes (npos == 0 coverage).
    - no_pred_classes: classes with gt but zero predictions.
    - no_match_classes: classes whose predictions are all far away (npos > 0,
      predictions exist, zero matches -> no_predictions() coverage).
    """
    rng = random.Random(seed)
    active_classes = [c for c in DETECTION_NAMES if c not in drop_classes]
    samples = []
    gt_by_token = {}
    for s in range(n_samples):
        token = f"{rng.getrandbits(128):032x}"
        ego = [rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0]
        annotations = []
        n_gt = rng.randint(*gt_per_sample)
        for _ in range(n_gt):
            cls = rng.choice(active_classes)
            # Distances up to 60 m: some boxes fall outside the per-class eval
            # range and must be dropped by the distance filter.
            dist = rng.uniform(2.0, 60.0)
            angle = rng.uniform(-math.pi, math.pi)
            cx = ego[0] + dist * math.cos(angle)
            cy = ego[1] + dist * math.sin(angle)
            wr, lr, hr = _CLASS_SIZE_RANGES[cls]
            attrs = _CLASS_ATTRIBUTES[cls]
            annotations.append(
                {
                    "category_name": rng.choice(_CLASS_TO_CATEGORIES[cls]),
                    "translation": [cx, cy, ego[2] + rng.uniform(-0.5, 1.5)],
                    "size": [rng.uniform(*wr), rng.uniform(*lr), rng.uniform(*hr)],
                    "rotation": _yaw_quaternion(rng),
                    "velocity": _velocity(rng, allow_nan=True),
                    "num_lidar_pts": rng.randint(0, 25),
                    "num_radar_pts": rng.randint(0, 4),
                    "attribute_name": rng.choice(attrs) if rng.random() > 0.06 else "",
                }
            )
        # Bike racks with bicycles placed inside (filtered) and outside (kept).
        if bike_racks and s % 3 == 0:
            rack_c = [ego[0] + rng.uniform(5, 25), ego[1] + rng.uniform(5, 25), ego[2]]
            annotations.append(
                {
                    "category_name": "static_object.bicycle_rack",
                    "translation": rack_c,
                    "size": [2.0, 5.0, 1.5],
                    "rotation": _yaw_quaternion(rng),
                    "velocity": [0.0, 0.0],
                    "num_lidar_pts": 40,
                    "num_radar_pts": 0,
                    "attribute_name": "",
                }
            )
            for inside in (True, False):
                offset = rng.uniform(0.2, 0.9) if inside else rng.uniform(6.0, 10.0)
                annotations.append(
                    {
                        "category_name": "vehicle.bicycle",
                        "translation": [rack_c[0] + offset, rack_c[1] + offset, rack_c[2]],
                        "size": [0.6, 1.8, 1.2],
                        "rotation": _yaw_quaternion(rng),
                        "velocity": _velocity(rng, allow_nan=True),
                        "num_lidar_pts": rng.randint(1, 15),
                        "num_radar_pts": rng.randint(0, 2),
                        "attribute_name": rng.choice(_CLASS_ATTRIBUTES["bicycle"]),
                    }
                )
        # An ignored (non-detection) category, dropped like the reference does.
        if s % 5 == 1:
            annotations.append(
                {
                    "category_name": "animal.deer",
                    "translation": [ego[0] + 3, ego[1] + 3, ego[2]],
                    "size": [0.5, 1.0, 0.8],
                    "rotation": _yaw_quaternion(rng),
                    "velocity": [0.0, 0.0],
                    "num_lidar_pts": 5,
                    "num_radar_pts": 0,
                    "attribute_name": "",
                }
            )
        samples.append({"token": token, "ego_translation": ego, "annotations": annotations})
        gt_by_token[token] = annotations

    # Predictions: jittered copies of gt boxes (all 4 distance thresholds get
    # matches), duplicate detections, and pure false positives.
    results = {}
    for sample in samples:
        token = sample["token"]
        ego = sample["ego_translation"]
        preds = []
        for ann in gt_by_token[token]:
            category = ann["category_name"]
            cls = next(
                (c for c, cats in _CLASS_TO_CATEGORIES.items() if category in cats), None
            )
            if cls is None or cls in no_pred_classes:
                continue
            if rng.random() > pred_per_gt:
                continue  # missed detection
            jitter = 10.0 if cls in no_match_classes else rng.uniform(0.0, 3.8)
            jangle = rng.uniform(-math.pi, math.pi)
            for _ in range(2 if rng.random() < 0.15 else 1):  # occasional duplicate
                wr, lr, hr = _CLASS_SIZE_RANGES[cls]
                attrs = _CLASS_ATTRIBUTES[cls]
                box = {
                    "sample_token": token,
                    "translation": [
                        ann["translation"][0] + jitter * math.cos(jangle),
                        ann["translation"][1] + jitter * math.sin(jangle),
                        ann["translation"][2] + rng.uniform(-0.3, 0.3),
                    ],
                    "size": [
                        ann["size"][0] * rng.uniform(0.8, 1.2),
                        ann["size"][1] * rng.uniform(0.8, 1.2),
                        ann["size"][2] * rng.uniform(0.8, 1.2),
                    ],
                    "rotation": _yaw_quaternion(rng) if rng.random() < 0.4 else list(ann["rotation"]),
                    "velocity": _velocity(rng, allow_nan=True),
                    "detection_name": cls,
                    "detection_score": _score(rng, ties),
                    "attribute_name": rng.choice(attrs) if attrs != [""] else "",
                }
                if rng.random() < 0.05:
                    box["num_pts"] = 0  # must be dropped by the points filter
                if rng.random() < 0.05:
                    box["ego_translation"] = [1e9, 1e9, 1e9]  # must be overwritten
                preds.append(box)
        # Pure false positives.
        for _ in range(rng.randint(0, int(3 + 6 * fp_rate))):
            cls = rng.choice(active_classes)
            dist = rng.uniform(2.0, 58.0)
            angle = rng.uniform(-math.pi, math.pi)
            wr, lr, hr = _CLASS_SIZE_RANGES[cls]
            attrs = _CLASS_ATTRIBUTES[cls]
            preds.append(
                {
                    "sample_token": token,
                    "translation": [ego[0] + dist * math.cos(angle),
                                    ego[1] + dist * math.sin(angle),
                                    ego[2] + rng.uniform(-0.5, 1.5)],
                    "size": [rng.uniform(*wr), rng.uniform(*lr), rng.uniform(*hr)],
                    "rotation": _yaw_quaternion(rng),
                    "velocity": _velocity(rng, allow_nan=True),
                    "detection_name": cls,
                    "detection_score": _score(rng, ties),
                    "attribute_name": rng.choice(attrs) if attrs != [""] else "",
                }
            )
        results[token] = preds

    if shuffle_pred_samples:
        # The reference groups predictions in submission order; make it differ
        # from the gt sample order on purpose.
        items = list(results.items())
        rng.shuffle(items)
        results = dict(items)

    gt = {"samples": samples}
    predictions = {"meta": {"use_camera": False}, "results": results}
    return gt, predictions


# ---------------------------------------------------------------------------
# Oracle: drive the real nuscenes-devkit code on a synthetic corpus.
# ---------------------------------------------------------------------------


class StubNusc:
    """Minimal duck-typed stand-in for the NuScenes tables the devkit's
    add_center_dist / filter_eval_boxes read (sample, sample_data, ego_pose,
    sample_annotation). Everything else is intentionally absent."""

    def __init__(self, samples: list):
        self._samples = {s["token"]: s for s in samples}

    def get(self, table: str, token: str) -> dict:
        if table == "sample":
            anns = [f"ann_{token}_{i}" for i in range(len(self._samples[token]["annotations"]))]
            return {"token": token, "data": {"LIDAR_TOP": f"sd_{token}"}, "anns": anns}
        if table == "sample_data":
            return {"token": token, "ego_pose_token": f"pose_{token[3:]}"}
        if table == "ego_pose":
            sample = self._samples[token[5:]]
            return {"token": token, "translation": list(sample["ego_translation"])}
        if table == "sample_annotation":
            sample_token, idx = token[4:].rsplit("_", 1)
            return self._samples[sample_token]["annotations"][int(idx)]
        raise KeyError(table)


def reference_metrics(gt: dict, predictions: dict, cfg=None) -> dict:
    """The official evaluation chain (post-DB-load), run by the real devkit.

    Mirrors DetectionEval: build EvalBoxes (gt via the load_gt field mapping,
    predictions via EvalBoxes.deserialize of the submission), add_center_dist,
    filter_eval_boxes, accumulate() at every class x dist_th, calc_ap/calc_tp
    with the class-specific NaN rules, DetectionMetrics assembly.
    """
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetrics,
    )
    from nuscenes.eval.detection.utils import category_to_detection_name

    if cfg is None:
        cfg = DetectionConfig(
            class_range={
                "car": 50, "truck": 50, "bus": 50, "trailer": 50,
                "construction_vehicle": 50, "pedestrian": 40, "motorcycle": 40,
                "bicycle": 40, "traffic_cone": 30, "barrier": 30,
            },
            dist_fcn="center_distance",
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            max_boxes_per_sample=500,
            mean_ap_weight=5,
        )

    stub = StubNusc(gt["samples"])

    # GT boxes, constructed exactly like load_gt() does per annotation.
    gt_boxes = EvalBoxes()
    for sample in gt["samples"]:
        boxes = []
        for ann in sample["annotations"]:
            detection_name = category_to_detection_name(ann["category_name"])
            if detection_name is None:
                continue
            boxes.append(
                DetectionBox(
                    sample_token=sample["token"],
                    translation=ann["translation"],
                    size=ann["size"],
                    rotation=ann["rotation"],
                    velocity=ann["velocity"],
                    num_pts=ann["num_lidar_pts"] + ann["num_radar_pts"],
                    detection_name=detection_name,
                    detection_score=-1.0,
                    attribute_name=ann.get("attribute_name", ""),
                )
            )
        gt_boxes.add_boxes(sample["token"], boxes)

    # Predictions via the devkit's own submission deserializer.
    pred_boxes = EvalBoxes.deserialize(predictions["results"], DetectionBox)

    pred_boxes = add_center_dist(stub, pred_boxes)
    gt_boxes = add_center_dist(stub, gt_boxes)
    pred_boxes = filter_eval_boxes(stub, pred_boxes, cfg.class_range)
    gt_boxes = filter_eval_boxes(stub, gt_boxes, cfg.class_range)

    md_list = {}
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
            md_list[(class_name, dist_th)] = md

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metrics.add_label_ap(
                class_name, dist_th,
                calc_ap(md_list[(class_name, dist_th)], cfg.min_recall, cfg.min_precision),
            )
        for metric_name in TP_METRICS:
            md = md_list[(class_name, cfg.dist_th_tp)]
            if class_name in ["traffic_cone"] and metric_name in ["attr_err", "vel_err", "orient_err"]:
                tp = np.nan
            elif class_name in ["barrier"] and metric_name in ["attr_err", "vel_err"]:
                tp = np.nan
            else:
                tp = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    out = metrics.serialize()
    out.pop("eval_time", None)
    return out


# ---------------------------------------------------------------------------
# Metrics-dict comparison.
# ---------------------------------------------------------------------------


def compare_metrics(ours: dict, reference: dict, atol: float = SCORE_ATOL, _path: str = ""):
    """Recursively compare two metrics dicts (NaN-aware). Returns max |diff|.

    `eval_time` is skipped (it is wall-clock by definition). Raises
    AssertionError with the offending path on any structural or numeric
    disagreement beyond atol.
    """
    if isinstance(reference, dict):
        assert isinstance(ours, dict), f"{_path}: expected dict, got {type(ours)}"
        ours = {k: v for k, v in ours.items() if str(k) != "eval_time"}
        reference = {k: v for k, v in reference.items() if str(k) != "eval_time"}
        ours_keys = {str(k) for k in ours}
        ref_keys = {str(k) for k in reference}
        assert ours_keys == ref_keys, (
            f"{_path}: key mismatch: only ours {sorted(ours_keys - ref_keys)}, "
            f"only reference {sorted(ref_keys - ours_keys)}"
        )
        worst = 0.0
        for key in reference:
            hit = next(k for k in ours if str(k) == str(key))
            worst = max(
                worst, compare_metrics(ours[hit], reference[key], atol, f"{_path}.{key}")
            )
        return worst
    if isinstance(reference, (int, float, np.floating)) or reference is None:
        if reference is None:
            assert ours is None, f"{_path}: expected None, got {ours!r}"
            return 0.0
        ours_v, ref_v = float(ours), float(reference)
        if math.isnan(ref_v):
            assert math.isnan(ours_v), f"{_path}: expected NaN, got {ours_v!r}"
            return 0.0
        diff = abs(ours_v - ref_v)
        assert diff <= atol, f"{_path}: ours {ours_v!r} vs reference {ref_v!r} (|diff| {diff:.3e})"
        return diff
    assert ours == reference, f"{_path}: ours {ours!r} != reference {reference!r}"
    return 0.0


def corpus_json_roundtrip(gt: dict, predictions: dict) -> tuple:
    """Round-trip a corpus through JSON (path-input and serialization coverage)."""
    return json.loads(json.dumps(gt)), json.loads(json.dumps(predictions))
