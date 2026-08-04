"""Gradient-boosted trees over SVD-reduced TF-IDF plus POS ratio features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from fakenews.config import XGBoostConfig
from fakenews.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class XGBoostModel(BaseClassifier):
    """Thin wrapper over ``xgboost.XGBClassifier``."""

    name = "XGBoost"

    def __init__(self, config: XGBoostConfig | None = None, *, random_state: int = 0) -> None:
        self.config = config or XGBoostConfig()
        self.random_state = random_state
        self.model: xgb.XGBClassifier | None = None
        self.best_params: dict | None = None

    @staticmethod
    def _tag_as_classifier(estimator: xgb.XGBClassifier) -> xgb.XGBClassifier:
        """Restore the ``_estimator_type`` attribute XGBoost's serialiser needs.

        scikit-learn 1.8 removed ``ClassifierMixin._estimator_type`` in favour of
        ``__sklearn_tags__``, but ``XGBClassifier.save_model`` and ``load_model``
        still read it and raise ``TypeError`` when it is missing. Setting it here
        is accurate -- this estimator is a classifier -- and keeps both paths
        working across the two library versions.
        """
        if not hasattr(estimator, "_estimator_type"):
            estimator._estimator_type = "classifier"
        return estimator

    def _make_estimator(self, **overrides) -> xgb.XGBClassifier:
        params = {
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "min_child_weight": self.config.min_child_weight,
            "gamma": self.config.gamma,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "n_jobs": self.config.n_jobs,
            "random_state": self.random_state,
        }
        params.update(overrides)
        return self._tag_as_classifier(xgb.XGBClassifier(**params))

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> XGBoostModel:
        """Fit with the configured (untuned) hyperparameters."""
        self.model = self._make_estimator()

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

        logger.info("XGBoost fitted: %d trees, depth %d", self.config.n_estimators, self.config.max_depth)
        return self

    def tune(self, X_train, y_train, param_grid: dict, *, cv: int = 3, scoring: str = "f1") -> dict:
        """Grid-search hyperparameters and keep the best estimator.

        Used by the tuning experiment to quantify how much the untuned
        configuration gives away, so it can be described honestly as a lower bound.
        """
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(
            self._make_estimator(), param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_
        logger.info("Best params %s (CV %s = %.4f)", search.best_params_, scoring, search.best_score_)
        return search.best_params_

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self, importance_type: str = "gain") -> np.ndarray:
        """Per-feature importance from the fitted booster."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted")

        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        n_features = self.model.n_features_in_
        out = np.zeros(n_features, dtype=np.float64)
        for key, value in scores.items():
            out[int(key.lstrip("f"))] = value
        return out

    def save(self, directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        directory.mkdir(parents=True, exist_ok=True)
        self._tag_as_classifier(self.model)
        self.model.save_model(directory / "xgboost.json")

    @classmethod
    def load(cls, directory: Path) -> XGBoostModel:
        instance = cls()
        instance.model = cls._tag_as_classifier(xgb.XGBClassifier())
        instance.model.load_model(Path(directory) / "xgboost.json")
        return instance

    @property
    def parameter_counts(self) -> dict[str, int]:
        """Tree-ensemble size.

        A boosted forest has no weight matrix, so "parameters" is reported as the
        structures that actually occupy memory: split nodes and leaf values.
        Quoting a single number against a transformer's weight count would not be
        a like-for-like comparison.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted")

        trees = self.model.get_booster().trees_to_dataframe()
        is_leaf = trees["Feature"] == "Leaf"
        return {
            "trees": int(trees["Tree"].nunique()),
            "nodes_total": int(len(trees)),
            "nodes_split": int((~is_leaf).sum()),
            "nodes_leaf": int(is_leaf.sum()),
            # Each split stores a feature index and a threshold; each leaf stores
            # one value. This is the closest honest analogue to a weight count.
            "stored_values": int((~is_leaf).sum() * 2 + is_leaf.sum()),
        }
