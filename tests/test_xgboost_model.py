import numpy as np
import pytest

from fakenews.config import XGBoostConfig
from fakenews.models.xgboost_model import XGBoostModel


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.3, size=300) > 0).astype(int)
    return X, y


@pytest.fixture
def fitted(dataset):
    X, y = dataset
    config = XGBoostConfig(n_estimators=20, max_depth=3)
    return XGBoostModel(config, random_state=0).fit(X, y), X, y


def test_save_and_load_round_trip(fitted, tmp_path):
    """Guards a real scikit-learn 1.8 / XGBoost incompatibility.

    scikit-learn 1.8 removed ``ClassifierMixin._estimator_type``, which XGBoost's
    ``save_model``/``load_model`` still require. Without the wrapper's fix, both
    raise ``TypeError`` -- and it surfaces only after training completes.
    """
    model, X, _ = fitted

    model.save(tmp_path)
    assert (tmp_path / "xgboost.json").exists()

    reloaded = XGBoostModel.load(tmp_path)
    np.testing.assert_allclose(reloaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)


def test_predict_proba_is_bounded_and_one_dimensional(fitted):
    model, X, _ = fitted
    proba = model.predict_proba(X)

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_thresholds_at_a_half(fitted):
    model, X, _ = fitted
    np.testing.assert_array_equal(model.predict(X), (model.predict_proba(X) > 0.5).astype(int))


def test_unfitted_model_refuses_to_predict_or_save(tmp_path):
    model = XGBoostModel()

    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.save(tmp_path)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.parameter_counts


def test_parameter_counts_are_internally_consistent(fitted):
    model, _, _ = fitted
    counts = model.parameter_counts

    assert counts["trees"] == 20
    assert counts["nodes_split"] + counts["nodes_leaf"] == counts["nodes_total"]
    assert counts["stored_values"] == counts["nodes_split"] * 2 + counts["nodes_leaf"]


def test_feature_importances_span_every_feature(fitted):
    model, X, _ = fitted
    gains = model.feature_importances("gain")

    assert gains.shape == (X.shape[1],)
    assert gains.sum() > 0
