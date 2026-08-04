import numpy as np

from fakenews.config import CLASS_NAMES
from fakenews.evaluation.metrics import evaluate


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    result = evaluate(y, y, y.astype(float))

    assert result.accuracy == 1.0
    assert result.f1 == 1.0
    assert result.false_positives == 0
    assert result.false_positive_rate == 0.0


def test_false_positive_rate_is_not_one_minus_recall():
    """Recall is measured on the fake class; the FPR is measured on the legit class.

    Conflating the two is the arithmetic error this project exists to avoid, so
    the distinction is pinned by a test.
    """
    # 10 legit (2 wrongly flagged fake), 10 fake (1 missed).
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 8 + [1] * 2 + [0] * 1 + [1] * 9)

    result = evaluate(y_true, y_pred)

    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.false_positive_rate == 2 / 10
    assert result.recall == 9 / 10
    assert result.false_positive_rate != 1 - result.recall


def test_aggregate_metrics_are_invariant_to_label_naming():
    """Accuracy and macro-F1 do not change when the two classes swap names.

    This is why the WELFake label inconsistency cannot invalidate published
    aggregate results.
    """
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_pred = np.where(rng.random(500) < 0.9, y_true, 1 - y_true)

    original = evaluate(y_true, y_pred)
    flipped = evaluate(1 - y_true, 1 - y_pred)

    assert original.accuracy == flipped.accuracy
    assert original.macro_f1 == flipped.macro_f1
    # Per-class precision does swap, which is exactly what is affected.
    assert original.precision != flipped.precision


def test_per_class_entries_cover_both_classes():
    y_true = np.array([0, 1, 0, 1])
    result = evaluate(y_true, y_true)

    assert set(result.per_class) == set(CLASS_NAMES)
    assert result.per_class["Legit"]["support"] == 2
    assert result.per_class["Fake"]["support"] == 2


def test_roc_auc_is_omitted_without_probabilities():
    y = np.array([0, 1, 0, 1])
    assert evaluate(y, y).roc_auc is None
    assert evaluate(y, y, y.astype(float)).roc_auc == 1.0


def test_serialised_keys_are_stable():
    """Pins the JSON contract the experiment scripts read.

    These names are consumed by ``run_seeds.py``, ``run_leakage.py`` and
    ``run_efficiency.py`` via saved JSON, where a rename fails only at runtime
    and only after the training that produced the file.
    """
    y = np.array([0, 1, 0, 1])
    payload = evaluate(y, y, y.astype(float)).to_dict()

    assert set(payload) == {
        "model",
        "n_samples",
        "accuracy",
        "precision_fake",
        "recall_fake",
        "f1_fake",
        "macro_f1",
        "roc_auc",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
        "confusion",
        "per_class",
    }
