"""nuscenes-eval-mojo loader/backend behaviour and input validation tests."""

from __future__ import annotations

import os

import pytest

import nuscenes_eval_mojo as ne
from nuscenes_eval_mojo import DetectionConfig, EvaluationError
from nuscenes_eval_mojo import _native
from nuscenes_eval_mojo._native import NativeUnavailable

from test_nuscenes_eval_fixtures import make_corpus

GT, PRED = make_corpus(seed=5150, n_samples=6, gt_per_sample=(2, 5))

RESULT_KEYS = {
    "label_aps", "mean_dist_aps", "mean_ap", "label_tp_errors",
    "tp_errors", "tp_scores", "nd_score", "eval_time", "cfg",
}


def test_result_schema_shape():
    metrics = ne.evaluate(GT, PRED)
    assert set(metrics) == RESULT_KEYS
    assert set(metrics["cfg"]) == {
        "class_range", "dist_fcn", "dist_ths", "dist_th_tp",
        "min_recall", "min_precision", "max_boxes_per_sample", "mean_ap_weight",
    }
    assert set(metrics["label_aps"]) == set(ne.DETECTION_NAMES)
    for aps in metrics["label_aps"].values():
        assert set(aps) == {0.5, 1.0, 2.0, 4.0}
    assert set(metrics["label_tp_errors"]["car"]) == set(ne.TP_METRICS)
    assert isinstance(metrics["eval_time"], float)


def test_backend_info_shape():
    info = ne.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("NUSCENES_EVAL_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/nuscenes-eval/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("NUSCENES_EVAL_MOJO_DISABLE_NATIVE", "1")
    metrics = ne.evaluate(GT, PRED)
    assert set(metrics) == RESULT_KEYS
    assert ne.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("NUSCENES_EVAL_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("NUSCENES_EVAL_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        metrics = ne.evaluate(GT, PRED)
        assert set(metrics) == RESULT_KEYS
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("NUSCENES_EVAL_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        metrics = ne.evaluate(GT, PRED)
        assert set(metrics) == RESULT_KEYS
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def nuscenesevalmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("NUSCENES_EVAL_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libnuscenesevalmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def _with_pred_box(transform) -> dict:
    import copy

    pred = copy.deepcopy(PRED)
    token = next(iter(pred["results"]))
    transform(pred["results"][token][0])
    return pred


def test_sample_token_mismatch_rejected():
    import copy

    pred = copy.deepcopy(PRED)
    pred["results"]["deadbeef" * 4] = []
    with pytest.raises(EvaluationError, match="sample_token_mismatch"):
        ne.evaluate(GT, pred)


def test_too_many_boxes_rejected():
    import copy

    pred = copy.deepcopy(PRED)
    token = next(iter(pred["results"]))
    pred["results"][token] = pred["results"][token][:1] * 501
    with pytest.raises(EvaluationError, match="too_many_boxes"):
        ne.evaluate(GT, pred)


def test_unknown_detection_name_rejected():
    pred = _with_pred_box(lambda box: box.update(detection_name="hovercraft"))
    with pytest.raises(EvaluationError, match="unknown_detection_name"):
        ne.evaluate(GT, pred)


def test_unknown_attribute_rejected():
    pred = _with_pred_box(lambda box: box.update(attribute_name="vehicle.flying"))
    with pytest.raises(EvaluationError, match="unknown_attribute"):
        ne.evaluate(GT, pred)


def test_nan_score_rejected():
    pred = _with_pred_box(lambda box: box.update(detection_score=float("nan")))
    with pytest.raises(EvaluationError, match="nan_score"):
        ne.evaluate(GT, pred)


def test_non_positive_size_rejected():
    pred = _with_pred_box(lambda box: box.update(size=[1.0, 0.0, 1.0]))
    with pytest.raises(EvaluationError, match="invalid_size"):
        ne.evaluate(GT, pred)


def test_nan_translation_rejected():
    pred = _with_pred_box(lambda box: box.update(translation=[float("nan"), 0.0, 0.0]))
    with pytest.raises(EvaluationError, match="nan_field"):
        ne.evaluate(GT, pred)


def test_missing_results_rejected():
    with pytest.raises(EvaluationError, match="missing_results"):
        ne.evaluate(GT, {"meta": {}})


def test_empty_gt_rejected():
    with pytest.raises(EvaluationError, match="empty_gt"):
        ne.evaluate({"samples": []}, PRED)


def test_unknown_inner_sample_token_rejected():
    pred = _with_pred_box(lambda box: box.update(sample_token="0" * 32))
    with pytest.raises(EvaluationError, match="unknown_sample_token"):
        ne.evaluate(GT, pred)


def test_invalid_config_rejected():
    with pytest.raises(EvaluationError, match="class_count_mismatch"):
        DetectionConfig(class_range={"car": 50})
    with pytest.raises(EvaluationError, match="unknown_dist_fcn"):
        DetectionConfig(class_range=dict.fromkeys(ne.DETECTION_NAMES, 50), dist_fcn="iou")
    with pytest.raises(EvaluationError, match="dist_th_tp_not_in_dist_ths"):
        DetectionConfig(
            class_range=dict.fromkeys(ne.DETECTION_NAMES, 50),
            dist_ths=(0.5, 1.0),
            dist_th_tp=2.0,
        )
    with pytest.raises(EvaluationError, match="invalid_min_precision"):
        DetectionConfig(class_range=dict.fromkeys(ne.DETECTION_NAMES, 50), min_precision=1.0)


def test_invalid_input_type_rejected():
    with pytest.raises(EvaluationError, match="invalid_input_type"):
        ne.evaluate(42, PRED)
