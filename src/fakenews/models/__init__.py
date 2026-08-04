"""Model wrappers exposing a common ``predict`` / ``predict_proba`` interface."""

from fakenews.models.base import BaseClassifier
from fakenews.models.ensemble import EnsembleModel
from fakenews.models.text_cnn import TextCNNModel
from fakenews.models.xgboost_model import XGBoostModel

__all__ = ["BaseClassifier", "EnsembleModel", "TextCNNModel", "XGBoostModel"]
