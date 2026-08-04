"""Combining base-model probabilities by voting or stacking."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from fakenews.models.base import BaseClassifier

logger = logging.getLogger(__name__)

EnsembleMethod = Literal["soft_voting", "weighted_voting", "stacking"]


class EnsembleModel:
    """Combine several fitted base classifiers.

    Each base model consumes its own feature representation, so ``X_list`` is a
    list of feature matrices ordered to match ``models``.
    """

    def __init__(
        self,
        models: Sequence[BaseClassifier],
        *,
        method: EnsembleMethod = "soft_voting",
        weights: Sequence[float] | None = None,
        model_names: Sequence[str] | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one base model is required")

        self.models = list(models)
        self.method = method
        self.model_names = list(model_names) if model_names else [m.name for m in models]
        self.meta_model: LogisticRegression | None = None

        if method == "weighted_voting":
            if weights is None:
                raise ValueError("weighted_voting requires weights")
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = np.asarray(weights, dtype=float) / np.sum(weights)
        else:
            self.weights = None

    @property
    def name(self) -> str:
        return f"Ensemble ({self.method.replace('_', ' ')})"

    def base_probabilities(self, X_list: Sequence) -> np.ndarray:
        """Return an ``(n_samples, n_models)`` matrix of base-model P(fake)."""
        if len(X_list) != len(self.models):
            raise ValueError(
                f"Expected {len(self.models)} feature matrices, got {len(X_list)}"
            )
        return np.column_stack([model.predict_proba(X) for model, X in zip(self.models, X_list, strict=True)])

    def fit_meta_model(self, base_probabilities: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> None:
        """Fit the stacking meta-learner on held-out base-model probabilities.

        ``base_probabilities`` must come from a split the base models were not
        trained on -- the validation split -- otherwise the meta-learner is fitted
        on overfitted inputs.
        """
        if self.method != "stacking":
            raise ValueError("fit_meta_model applies to the stacking method only")

        self.meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.meta_model.fit(base_probabilities, y)
        logger.info(
            "Meta-learner coefficients: %s",
            dict(zip(self.model_names, self.meta_model.coef_[0].round(4), strict=True)),
        )

    def predict_proba(self, X_list: Sequence) -> np.ndarray:
        probabilities = self.base_probabilities(X_list)

        if self.method == "soft_voting":
            return probabilities.mean(axis=1)
        if self.method == "weighted_voting":
            return np.average(probabilities, axis=1, weights=self.weights)
        if self.method == "stacking":
            if self.meta_model is None:
                raise RuntimeError("Meta-learner has not been fitted")
            return self.meta_model.predict_proba(probabilities)[:, 1]

        raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict(self, X_list: Sequence, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_list) > threshold).astype(int)

    def save(self, directory: Path) -> None:
        """Persist ensemble configuration. Base models are saved separately."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        payload: dict = {"method": self.method, "model_names": self.model_names}
        if self.weights is not None:
            payload["weights"] = self.weights.tolist()
        if self.meta_model is not None:
            payload["meta_coef"] = self.meta_model.coef_.tolist()
            payload["meta_intercept"] = self.meta_model.intercept_.tolist()

        (directory / "ensemble.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
