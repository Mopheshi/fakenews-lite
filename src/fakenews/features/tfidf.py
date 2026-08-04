"""TF-IDF vectorisation followed by truncated SVD.

XGBoost handles a 10,000-column sparse matrix poorly, so the TF-IDF space is
projected onto a few hundred dense parts before the POS features are
appended.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from fakenews.config import TfidfConfig

logger = logging.getLogger(__name__)


class TfidfSvdEncoder:
    """Fit TF-IDF + SVD on the training split and apply it to any split."""

    def __init__(self, config: TfidfConfig | None = None, *, random_state: int = 0) -> None:
        self.config = config or TfidfConfig()
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    @property
    def explained_variance(self) -> float:
        """Fraction of TF-IDF variance retained by the SVD projection."""
        if self.svd is None:
            raise RuntimeError("Encoder has not been fitted")
        return float(self.svd.explained_variance_ratio_.sum())

    @property
    def n_components(self) -> int:
        return self.config.svd_components

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF matrix %s (vocabulary %d)", tfidf.shape, len(self.vectorizer.vocabulary_)
        )

        self.svd = TruncatedSVD(
            n_components=self.config.svd_components, random_state=self.random_state
        )
        reduced = self.svd.fit_transform(tfidf)
        logger.info(
            "SVD -> %s, explained variance %.4f", reduced.shape, self.explained_variance
        )
        return reduced

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("Encoder has not been fitted")
        return self.svd.transform(self.vectorizer.transform(texts))


def combine_features(svd_features: np.ndarray, pos_features: np.ndarray | None) -> np.ndarray:
    """Append POS ratio columns to the SVD components.

    Passing ``None`` returns the SVD block unchanged, which is what the
    no-POS arm of the ablation uses.
    """
    if pos_features is None:
        return svd_features
    return np.hstack([svd_features, np.asarray(pos_features, dtype=np.float64)])
