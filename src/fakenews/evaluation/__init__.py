"""Metrics, significance testing and figures."""

from fakenews.evaluation.metrics import ClassificationResult, evaluate
from fakenews.evaluation.significance import (
    EffectSize,
    McNemarResult,
    bootstrap_accuracy_ci,
    class_conditional_effect,
    mcnemar_test,
    paired_difference_ci,
)

__all__ = [
    "ClassificationResult",
    "EffectSize",
    "McNemarResult",
    "bootstrap_accuracy_ci",
    "class_conditional_effect",
    "evaluate",
    "mcnemar_test",
    "paired_difference_ci",
]
