"""Differential tests: nuscenes_eval_mojo must match nuscenes-devkit's DetectionEval.

Run twice by `scripts/test_all_nuscenes_eval.sh`: once against the native Mojo
kernel and once with NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the reference within 1e-8 on the full
metrics dict (in practice the agreement is ulp-level, ~1e-15).

The oracle is the published PyPI package nuscenes-devkit, exercised through
its own EvalBoxes/DetectionBox construction, add_center_dist,
filter_eval_boxes, accumulate(), calc_ap()/calc_tp(), and DetectionMetrics —
the exact steps DetectionEval.evaluate() runs after DB loading (see
tests/test_nuscenes_eval_fixtures.py).
"""

from __future__ import annotations

import os

import pytest

import nuscenes_eval_mojo as ne
from nuscenes_eval_mojo import DetectionConfig

from test_nuscenes_eval_fixtures import (
    SCORE_ATOL,
    compare_metrics,
    corpus_json_roundtrip,
    make_corpus,
    reference_metrics,
)

CORPORA = [
    pytest.param(dict(seed=101, n_samples=16), id="base"),
    pytest.param(dict(seed=202, n_samples=24, ties=True), id="confidence-ties"),
    pytest.param(
        dict(seed=303, n_samples=16, drop_classes=("trailer", "construction_vehicle")),
        id="npos-zero-classes",
    ),
    pytest.param(
        dict(seed=404, n_samples=16, no_pred_classes=("barrier", "bus")),
        id="no-pred-classes",
    ),
    pytest.param(
        dict(seed=505, n_samples=16, no_match_classes=("pedestrian", "car")),
        id="zero-match-classes",
    ),
    pytest.param(dict(seed=606, n_samples=20, bike_racks=True, fp_rate=1.5), id="bike-racks-heavy-fp"),
    pytest.param(dict(seed=707, n_samples=10, bike_racks=False, gt_per_sample=(10, 22)), id="dense"),
    pytest.param(dict(seed=808, n_samples=6, gt_per_sample=(1, 3), pred_per_gt=0.4), id="sparse-low-recall"),
]


def _expected_backend() -> str:
    # scripts/test_all_nuscenes_eval.sh runs the suite once per backend.
    return "fallback" if os.environ.get("NUSCENES_EVAL_MOJO_DISABLE_NATIVE") == "1" else "native"


@pytest.mark.parametrize("corpus_kwargs", CORPORA)
def test_full_metrics_dict_parity(corpus_kwargs):
    gt, predictions = make_corpus(**corpus_kwargs)
    ours = ne.evaluate(gt, predictions)
    reference = reference_metrics(gt, predictions)
    worst = compare_metrics(ours, reference, atol=SCORE_ATOL)
    assert worst <= SCORE_ATOL


@pytest.mark.parametrize("corpus_kwargs", CORPORA[:4])
def test_json_roundtrip_parity(corpus_kwargs):
    gt, predictions = make_corpus(**corpus_kwargs)
    gt, predictions = corpus_json_roundtrip(gt, predictions)
    ours = ne.evaluate(gt, predictions)
    reference = reference_metrics(gt, predictions)
    compare_metrics(ours, reference, atol=SCORE_ATOL)


def test_path_inputs(tmp_path):
    gt, predictions = make_corpus(seed=909, n_samples=10)
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "submission.json"
    import json

    gt_path.write_text(json.dumps(gt))
    pred_path.write_text(json.dumps(predictions))
    ours = ne.evaluate(str(gt_path), str(pred_path))
    reference = reference_metrics(gt, predictions)
    compare_metrics(ours, reference, atol=SCORE_ATOL)


def test_custom_config_parity():
    gt, predictions = make_corpus(seed=111, n_samples=16, ties=True)
    cfg = DetectionConfig(
        class_range={
            "car": 40, "truck": 40, "bus": 45, "trailer": 45,
            "construction_vehicle": 35, "pedestrian": 30, "motorcycle": 30,
            "bicycle": 25, "traffic_cone": 20, "barrier": 25,
        },
        dist_ths=(0.25, 1.5, 3.0),
        dist_th_tp=1.5,
        min_recall=0.2,
        min_precision=0.05,
        max_boxes_per_sample=500,
        mean_ap_weight=5,
    )
    ours = ne.evaluate(gt, predictions, config=cfg)

    from nuscenes.eval.detection.data_classes import DetectionConfig as RefConfig

    ref_cfg = RefConfig(
        class_range=dict(cfg.class_range),
        dist_fcn=cfg.dist_fcn,
        dist_ths=list(cfg.dist_ths),
        dist_th_tp=cfg.dist_th_tp,
        min_recall=cfg.min_recall,
        min_precision=cfg.min_precision,
        max_boxes_per_sample=cfg.max_boxes_per_sample,
        mean_ap_weight=cfg.mean_ap_weight,
    )
    reference = reference_metrics(gt, predictions, cfg=ref_cfg)
    compare_metrics(ours, reference, atol=SCORE_ATOL)


def test_native_fallback_agreement():
    """The two backends must agree to ~1 ulp: compiled code (like the oracle's
    own Cython/BLAS paths) may contract a*b+c into a fused multiply-add, which
    CPython bytecode cannot."""
    if os.environ.get("NUSCENES_EVAL_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback-only run")
    gt, predictions = make_corpus(seed=212, n_samples=14, ties=True)
    ours_native = ne.evaluate(gt, predictions)
    os.environ["NUSCENES_EVAL_MOJO_DISABLE_NATIVE"] = "1"
    try:
        ours_fallback = ne.evaluate(gt, predictions)
    finally:
        del os.environ["NUSCENES_EVAL_MOJO_DISABLE_NATIVE"]
    compare_metrics(ours_native, ours_fallback, atol=1e-12)


def test_backend_is_the_expected_one():
    info = ne.backend_info()
    if _expected_backend() == "native":
        assert info["native_available"] is True
    else:
        assert info["native_available"] is False


def test_large_corpus_parity():
    """A bigger corpus exercises the x40 fan-out at a realistic scale."""
    gt, predictions = make_corpus(seed=313, n_samples=64, gt_per_sample=(8, 20), ties=True)
    ours = ne.evaluate(gt, predictions)
    reference = reference_metrics(gt, predictions)
    compare_metrics(ours, reference, atol=SCORE_ATOL)
