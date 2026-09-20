"""Single-article inference over a saved model directory.

This is the object the web application and the ``fakenews predict`` command use.
It owns the whole path from raw title/text to a labelled probability, so the
served model cannot drift away from the trained one.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fakenews.data.preprocessing import BLOCKLIST_PRESETS, TextPreprocessor
from fakenews.features.pos import (
    POS_CATEGORIES,
    extract_pos_features,
    extract_pos_features_batch,
)
from fakenews.models.ensemble import EnsembleModel
from fakenews.models.text_cnn import TextCNNModel
from fakenews.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class FakeNewsDetector:
    """Classify a single article as fake or legitimate."""

    def __init__(
        self,
        *,
        xgboost_model: XGBoostModel | None = None,
        textcnn_model: TextCNNModel | None = None,
        preprocessor: TextPreprocessor,
        tfidf_encoder: Any = None,
        sequence_encoder: Any = None,
        ensemble_model: EnsembleModel | None = None,
        use_pos_features: bool = True,
    ) -> None:
        if xgboost_model is None and textcnn_model is None:
            raise ValueError("At least one base model is required")

        self.xgboost_model = xgboost_model
        self.textcnn_model = textcnn_model
        self.preprocessor = preprocessor
        self.tfidf_encoder = tfidf_encoder
        self.sequence_encoder = sequence_encoder
        self.ensemble_model = ensemble_model
        self.use_pos_features = use_pos_features

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, directory: Path | str) -> FakeNewsDetector:
        """Reconstruct a detector from a directory written by the pipeline."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

        with (directory / "encoders.pkl").open("rb") as handle:
            encoders = pickle.load(handle)

        config = manifest["config"]
        preset = manifest.get("blocklist_preset", "none")

        xgboost_model = XGBoostModel.load(directory) if (directory / "xgboost.json").exists() else None
        textcnn_model = (
            TextCNNModel.load(
                directory, max_len=manifest["max_len"], vocab_size=manifest["vocab_size"]
            )
            if (directory / "text_cnn.keras").exists()
            else None
        )

        ensemble = None
        ensemble_path = directory / "ensemble.json"
        if ensemble_path.exists() and xgboost_model and textcnn_model:
            payload = json.loads(ensemble_path.read_text(encoding="utf-8"))
            ensemble = EnsembleModel(
                [xgboost_model, textcnn_model],
                method=payload["method"],
                weights=payload.get("weights"),
            )

        return cls(
            xgboost_model=xgboost_model,
            textcnn_model=textcnn_model,
            preprocessor=TextPreprocessor(blocklist=BLOCKLIST_PRESETS[preset]),
            tfidf_encoder=encoders["tfidf"],
            sequence_encoder=encoders["sequences"],
            ensemble_model=ensemble,
            use_pos_features=config.get("use_pos_features", True),
        )

    # --------------------------------------------------------------- features
    def _clean(self, title: object, text: object) -> str:
        """Title and body, cleaned and joined exactly as training joined them."""
        return f"{self.preprocessor.preprocess(title)} {self.preprocessor.preprocess(text)}".strip()

    def _xgboost_features(self, cleaned: str, original: str) -> np.ndarray:
        X = self.tfidf_encoder.transform([cleaned])
        if self.use_pos_features:
            ratios = extract_pos_features(original)
            X = np.hstack([X, np.array([[ratios[category] for category in POS_CATEGORIES]])])
        return X

    def _textcnn_features(self, cleaned: str) -> np.ndarray:
        return self.sequence_encoder.transform([cleaned])

    # -------------------------------------------------------- batch inference
    def predict_proba_many(
        self,
        titles: Sequence[object],
        texts: Sequence[object],
        *,
        pos_n_jobs: int = -1,
    ) -> dict[str, np.ndarray]:
        """Fake-class probability for a corpus, one array per model held.

        Deliberately not the implementation behind :meth:`predict_article`.
        That method is the measured single-article latency path, and routing it
        through the parallel POS tagger would add process startup to a figure
        the paper reports. The two share cleaning and nothing else.

        Args:
            titles: Raw headlines, aligned with ``texts``.
            texts: Raw bodies.
            pos_n_jobs: Workers for POS tagging; ``1`` runs inline.

        Returns:
            Arrays keyed as the pipeline keys them (``xgboost``, ``textcnn``,
            ``ensemble_soft_voting``), so a cross-dataset result can be read
            against a training-corpus result without renaming anything.
        """
        if len(titles) != len(texts):
            raise ValueError(f"Got {len(titles)} titles and {len(texts)} texts")

        cleaned = [self._clean(title, text) for title, text in zip(titles, texts, strict=True)]
        originals = [f"{title} {text}" for title, text in zip(titles, texts, strict=True)]

        probabilities: dict[str, np.ndarray] = {}
        feature_matrices = []
        models = []

        if self.xgboost_model is not None:
            X = self.tfidf_encoder.transform(cleaned)
            if self.use_pos_features:
                X = np.hstack([X, extract_pos_features_batch(originals, n_jobs=pos_n_jobs)])
            feature_matrices.append(X)
            models.append(self.xgboost_model)
            probabilities["xgboost"] = np.asarray(self.xgboost_model.predict_proba(X))

        if self.textcnn_model is not None:
            X = self.sequence_encoder.transform(cleaned)
            feature_matrices.append(X)
            models.append(self.textcnn_model)
            probabilities["textcnn"] = np.asarray(self.textcnn_model.predict_proba(X))

        if self.ensemble_model is not None and len(models) > 1:
            key = f"ensemble_{self.ensemble_model.method}"
            probabilities[key] = np.asarray(self.ensemble_model.predict_proba(feature_matrices))

        return probabilities

    # -------------------------------------------------------------- inference
    def predict_article(self, title: str, text: str, *, verbose: bool = False) -> dict:
        """Classify one article.

        Returns:
            A dict with ``prediction`` (``"FAKE"``/``"LEGIT"``), ``confidence``
            (probability of the predicted class) and both class probabilities.
        """
        cleaned = self._clean(title, text)
        original = f"{title} {text}"

        per_model: dict[str, dict[str, float]] = {}
        feature_matrices = []
        models = []

        if self.xgboost_model is not None:
            X = self._xgboost_features(cleaned, original)
            feature_matrices.append(X)
            models.append(self.xgboost_model)
            per_model["XGBoost"] = {"fake_probability": float(self.xgboost_model.predict_proba(X)[0])}

        if self.textcnn_model is not None:
            X = self._textcnn_features(cleaned)
            feature_matrices.append(X)
            models.append(self.textcnn_model)
            per_model["TextCNN"] = {"fake_probability": float(self.textcnn_model.predict_proba(X)[0])}

        if self.ensemble_model is not None and len(models) > 1:
            fake_probability = float(self.ensemble_model.predict_proba(feature_matrices)[0])
            source = self.ensemble_model.name
        else:
            fake_probability = float(next(iter(per_model.values()))["fake_probability"])
            source = next(iter(per_model))

        is_fake = fake_probability > 0.5
        result = {
            "prediction": "FAKE" if is_fake else "LEGIT",
            "confidence": fake_probability if is_fake else 1.0 - fake_probability,
            "fake_probability": fake_probability,
            "legit_probability": 1.0 - fake_probability,
            "decided_by": source,
        }
        if verbose:
            result["model_predictions"] = per_model
        return result
