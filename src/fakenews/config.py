"""Central configuration: filesystem layout and experiment hyperparameters.

Every path can be overridden with an environment variable so the package works
unchanged in a container, on a cluster, or from a checkout in any location.
Hyperparameters live in frozen dataclasses so an experiment run can record
exactly what produced its numbers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _path_from_env("FAKENEWS_ROOT", PACKAGE_ROOT.parent.parent)

DATA_DIR = _path_from_env("FAKENEWS_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CACHE_DIR = _path_from_env("FAKENEWS_CACHE_DIR", DATA_DIR / "cache")

ARTIFACTS_DIR = _path_from_env("FAKENEWS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

WELFAKE_CSV = _path_from_env("FAKENEWS_WELFAKE_CSV", RAW_DATA_DIR / "WELFake_Dataset.csv")
GLOVE_TXT = _path_from_env("FAKENEWS_GLOVE_TXT", EMBEDDINGS_DIR / "glove.6B.100d.txt")


def ensure_directories() -> None:
    """Create every writable directory the pipeline needs."""
    for directory in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Label semantics
# --------------------------------------------------------------------------- #

# The WELFake record is internally inconsistent: it states "Label 0 = fake and
# 1 = real" while also reporting 35,028 real and 37,106 fake articles. The
# distribution in the CSV is 35,028 zeros and 37,106 ones, so the counts fix the
# mapping as 0 = real, 1 = fake. See docs/label-semantics.md.
LABEL_NAMES = {0: "Legit", 1: "Fake"}
CLASS_NAMES = [LABEL_NAMES[0], LABEL_NAMES[1]]
POSITIVE_CLASS = 1  # "Fake" is the positive class for precision/recall/F1.


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #

#: Seed used for every headline number reported in the paper.
DEFAULT_SEED = 5

#: Seeds used for the mean +/- SD stability analysis.
STABILITY_SEEDS = (5, 17, 42, 123, 2024)


@dataclass(frozen=True)
class SplitConfig:
    """Stratified train/validation/test partitioning."""

    test_size: float = 0.2
    # 0.125 of the remaining 80% yields a 70/10/20 overall split.
    val_size_of_remainder: float = 0.125
    stratify: bool = True


@dataclass(frozen=True)
class TfidfConfig:
    """TF-IDF vectorisation followed by truncated SVD."""

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True
    svd_components: int = 300


@dataclass(frozen=True)
class SequenceConfig:
    """Integer-sequence encoding for the TextCNN."""

    max_words: int = 50_000
    max_len: int = 500
    embedding_dim: int = 100
    oov_token: str = "<OOV>"


@dataclass(frozen=True)
class XGBoostConfig:
    """Gradient-boosted tree hyperparameters.

    These are fixed rather than tuned; ``experiments/run_xgb_tuning.py`` reports
    what tuning buys, so the untuned model can be described honestly as a lower
    bound.
    """

    max_depth: int = 7
    learning_rate: float = 0.1
    n_estimators: int = 200
    min_child_weight: int = 3
    gamma: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    n_jobs: int = -1


@dataclass(frozen=True)
class TextCNNConfig:
    """Kim-style CNN over frozen GloVe embeddings."""

    filter_sizes: tuple[int, ...] = (3, 4, 5)
    num_filters: int = 128
    dense_units: int = 128
    dropout_rate: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 10
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    freeze_embeddings: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Full specification of a single pipeline run."""

    seed: int = DEFAULT_SEED
    use_pos_features: bool = True
    strip_source_markers: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    text_cnn: TextCNNConfig = field(default_factory=TextCNNConfig)

    def to_dict(self) -> dict:
        return asdict(self)
