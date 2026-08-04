"""Classification metrics with explicit per-class reporting.

Because the WELFake label documentation is internally inconsistent, every metric
that depends on which class is "positive" is reported per class as well as in
aggregate. Aggregate accuracy, macro-F1 and ROC-AUC are invariant to the naming
of class 0; precision, recall and the false positive rate are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fakenews.config import CLASS_NAMES, POSITIVE_CLASS


@dataclass
class ClassificationResult:
    """Metrics for one model on one split."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    macro_f1: float
    confusion: list[list[int]]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def false_positive_rate(self) -> float:
        """Fraction of legitimate articles incorrectly flagged as fake.

        This is the quantity the ethics discussion needs, and it is not
        ``1 - recall``: recall is computed on the fake class.
        """
        (true_negative, false_positive), _ = self.confusion
        denominator = true_negative + false_positive
        return false_positive / denominator if denominator else 0.0

    @property
    def false_positives(self) -> int:
        """Count of legitimate articles incorrectly flagged as fake."""
        return int(self.confusion[0][1])

    @property
    def false_negatives(self) -> int:
        """Count of fake articles missed."""
        return int(self.confusion[1][0])

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "precision_fake": self.precision,
            "recall_fake": self.recall,
            "f1_fake": self.f1,
            "macro_f1": self.macro_f1,
            "roc_auc": self.roc_auc,
            "false_positive_rate": self.false_positive_rate,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "confusion": self.confusion,
            "per_class": self.per_class,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    model_name: str = "model",
) -> ClassificationResult:
    """Compute the full metric set for one model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class: dict[str, dict[str, float]] = {}
    for label, class_name in enumerate(CLASS_NAMES):
        per_class[class_name] = {
            "precision": float(precision_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "support": int((y_true == label).sum()),
        }

    return ClassificationResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
        confusion=matrix.tolist(),
        per_class=per_class,
        n_samples=int(len(y_true)),
    )
