"""Shared interface for the base classifiers.

The ensemble only needs ``predict_proba`` from its members, so anything
implementing this protocol -- including models this project does not ship -- can
be dropped into :class:`~fakenews.models.ensemble.EnsembleModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseClassifier(ABC):
    """A fitted binary classifier over a single feature representation."""

    #: Human-readable name used in results tables and logs.
    name: str = "model"

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return P(fake) for each row, as a 1-D array."""

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions, 1 = fake."""
        return (self.predict_proba(X) > threshold).astype(int)

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the fitted model into ``directory``."""

    @property
    @abstractmethod
    def parameter_counts(self) -> dict[str, int]:
        """Parameter accounting used by the efficiency benchmark.

        Keys should distinguish learned parameters from frozen ones so the
        reported totals can be interpreted unambiguously.
        """
