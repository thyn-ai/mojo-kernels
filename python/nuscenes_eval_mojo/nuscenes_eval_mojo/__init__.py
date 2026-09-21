"""nuscenes-eval-mojo: a drop-in faster replacement for nuscenes-devkit's DetectionEval.

Same metrics schema, same values — powered by a Mojo kernel where the
platform supports it (macOS arm64, Linux x86_64), with a vendored pure-Python
fallback everywhere else (including Windows).

    import nuscenes_eval_mojo

    metrics = nuscenes_eval_mojo.evaluate(gt_corpus, submission_json)
    metrics["mean_ap"], metrics["nd_score"], metrics["tp_errors"]

Set NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from nuscenes_eval_mojo._native import backend_info, native_available
from nuscenes_eval_mojo.core import (
    ATTRIBUTE_NAMES,
    DETECTION_NAMES,
    TP_METRICS,
    DetectionConfig,
    EvaluationError,
    evaluate,
)

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "ATTRIBUTE_NAMES",
    "DETECTION_NAMES",
    "TP_METRICS",
    "DetectionConfig",
    "EvaluationError",
    "backend_info",
    "evaluate",
    "native_available",
    "__version__",
]
