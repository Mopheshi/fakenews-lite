import numpy as np
import pytest

from fakenews.models.base import BaseClassifier
from fakenews.models.ensemble import EnsembleModel


class ConstantModel(BaseClassifier):
    """A stand-in that returns fixed probabilities, so ensembling can be tested alone."""

    def __init__(self, probabilities, name="constant"):
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.name = name

    def predict_proba(self, X):
        return self._probabilities

    def save(self, directory):  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def parameter_counts(self):
        return {"trainable": 0}


@pytest.fixture
def models():
    return [
        ConstantModel([0.9, 0.1, 0.8, 0.2], name="A"),
        ConstantModel([0.7, 0.3, 0.2, 0.4], name="B"),
    ]


def test_soft_voting_averages_probabilities(models):
    ensemble = EnsembleModel(models, method="soft_voting")
    probabilities = ensemble.predict_proba([None, None])

    np.testing.assert_allclose(probabilities, [0.8, 0.2, 0.5, 0.3])
    np.testing.assert_array_equal(ensemble.predict([None, None]), [1, 0, 0, 0])


def test_weighted_voting_normalises_its_weights(models):
    equal = EnsembleModel(models, method="weighted_voting", weights=[2.0, 2.0])
    soft = EnsembleModel(models, method="soft_voting")

    np.testing.assert_allclose(
        equal.predict_proba([None, None]), soft.predict_proba([None, None])
    )
    np.testing.assert_allclose(equal.weights, [0.5, 0.5])


def test_weighted_voting_leans_toward_the_heavier_model(models):
    ensemble = EnsembleModel(models, method="weighted_voting", weights=[0.9, 0.1])
    probabilities = ensemble.predict_proba([None, None])

    assert probabilities[0] == pytest.approx(0.9 * 0.9 + 0.1 * 0.7)


def test_weighted_voting_requires_weights(models):
    with pytest.raises(ValueError, match="requires weights"):
        EnsembleModel(models, method="weighted_voting")


def test_weight_count_must_match_model_count(models):
    with pytest.raises(ValueError, match="must match"):
        EnsembleModel(models, method="weighted_voting", weights=[1.0])


def test_stacking_needs_a_fitted_meta_learner(models):
    ensemble = EnsembleModel(models, method="stacking")
    with pytest.raises(RuntimeError, match="not been fitted"):
        ensemble.predict_proba([None, None])


def test_stacking_learns_from_validation_probabilities(models):
    ensemble = EnsembleModel(models, method="stacking")

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    base = np.column_stack([y * 0.8 + 0.1, y * 0.6 + 0.2])

    ensemble.fit_meta_model(base, y)
    assert ensemble.meta_model is not None

    probabilities = ensemble.predict_proba([None, None])
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_feature_matrix_count_must_match_model_count(models):
    ensemble = EnsembleModel(models, method="soft_voting")
    with pytest.raises(ValueError, match="Expected 2 feature matrices"):
        ensemble.predict_proba([None])


def test_at_least_one_model_is_required():
    with pytest.raises(ValueError, match="At least one"):
        EnsembleModel([])
