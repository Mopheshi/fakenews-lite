import numpy as np
import pytest

from fakenews.evaluation.significance import (
    bootstrap_accuracy_ci,
    class_conditional_effect,
    mcnemar_test,
    paired_difference_ci,
)


def test_identical_predictions_give_no_discordant_pairs():
    y_true = np.array([0, 1, 0, 1, 1])
    result = mcnemar_test(y_true, y_true, y_true)

    assert result.n_discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_uses_the_exact_test_when_discordant_pairs_are_few():
    y_true = np.zeros(100, dtype=int)
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    pred_b[:5] = 1  # A right, B wrong on 5 items.

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert result.exact
    assert result.n_a_only == 5
    assert result.n_b_only == 0


def test_mcnemar_switches_to_chi_square_when_discordant_pairs_are_many():
    y_true = np.zeros(400, dtype=int)
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    pred_a[:40] = 1
    pred_b[40:120] = 1

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert not result.exact
    assert result.n_discordant == 120
    assert result.p_value < 0.05


def test_a_symmetric_disagreement_is_not_significant():
    y_true = np.zeros(400, dtype=int)
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    pred_a[:50] = 1
    pred_b[50:100] = 1  # Same number of errors each, in different places.

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert result.p_value > 0.05


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=1000)
    y_pred = np.where(rng.random(1000) < 0.9, y_true, 1 - y_true)

    point, low, high = bootstrap_accuracy_ci(y_true, y_pred, n_resamples=500, seed=0)

    assert low < point < high
    assert 0.85 < point < 0.95


def test_paired_difference_interval_covers_zero_for_identical_models():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=500)
    y_pred = np.where(rng.random(500) < 0.9, y_true, 1 - y_true)

    difference, low, high = paired_difference_ci(y_true, y_pred, y_pred, n_resamples=200, seed=0)

    assert difference == 0.0
    assert low == high == 0.0


def test_cohens_d_sign_follows_the_fake_class():
    rng = np.random.default_rng(4)
    labels = np.array([0] * 200 + [1] * 200)
    values = np.concatenate([rng.normal(0.10, 0.02, 200), rng.normal(0.14, 0.02, 200)])

    effect = class_conditional_effect(values, labels, feature="pos_adjectives")

    assert effect.mean_fake > effect.mean_legit
    assert effect.cohens_d > 1.0  # A two-SD separation is a large effect.
    assert effect.p_value < 0.01


@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
def test_cohens_d_is_zero_when_neither_class_varies():
    """Degenerate input must not produce a division by zero.

    SciPy warns about catastrophic cancellation on zero-variance groups, which is
    precisely the case under test.
    """
    labels = np.array([0] * 10 + [1] * 10)
    values = np.concatenate([np.zeros(10), np.ones(10)])

    effect = class_conditional_effect(values, labels, feature="pos_modals")

    assert effect.cohens_d == 0.0


def test_cohens_d_is_zero_without_a_class_difference():
    rng = np.random.default_rng(3)
    labels = np.array([0] * 500 + [1] * 500)
    values = rng.normal(size=1000)

    effect = class_conditional_effect(values, labels, feature="pos_nouns")

    assert abs(effect.cohens_d) < 0.2
