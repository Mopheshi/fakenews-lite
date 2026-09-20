"""End-to-end training and evaluation.

One call to :func:`run_pipeline` reproduces every headline number for a given
configuration and returns the raw test-set predictions, so downstream
significance testing works on exactly the arrays that produced the metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from fakenews.config import (
    RESULTS_DIR,
    ExperimentConfig,
    ensure_directories,
)
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.metrics import ClassificationResult, evaluate
from fakenews.features.pos import POS_FEATURE_COLUMNS
from fakenews.features.sequences import SequenceEncoder, load_glove_embeddings
from fakenews.features.tfidf import TfidfSvdEncoder, combine_features
from fakenews.models.ensemble import EnsembleModel
from fakenews.models.text_cnn import TextCNNModel
from fakenews.models.xgboost_model import XGBoostModel
from fakenews.seeding import set_global_seed

logger = logging.getLogger(__name__)

#: Ensemble variants evaluated on every run.
ENSEMBLE_METHODS = ("soft_voting", "weighted_voting", "stacking")


@dataclass
class SplitData:
    """A stratified train/validation/test partition of the corpus."""

    train_texts: np.ndarray
    val_texts: np.ndarray
    test_texts: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    pos_train: np.ndarray | None
    pos_val: np.ndarray | None
    pos_test: np.ndarray | None

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "validation": len(self.y_val),
            "test": len(self.y_test),
            "total": len(self.y_train) + len(self.y_val) + len(self.y_test),
        }


@dataclass
class PipelineResult:
    """Everything one pipeline run produced."""

    config: ExperimentConfig
    blocklist_preset: str
    split_sizes: dict[str, int]
    results: dict[str, ClassificationResult]
    predictions: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    probabilities: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    y_test: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    timings: dict[str, float] = field(default_factory=dict)
    feature_info: dict[str, float | int] = field(default_factory=dict)
    parameter_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Fixed-width metric table for terminal output."""
        header = f"{'Model':<28}{'Acc':>9}{'Prec':>9}{'Rec':>9}{'F1':>9}{'MacroF1':>10}{'ROC-AUC':>10}"
        lines = [header, "-" * len(header)]
        for result in self.results.values():
            roc = f"{result.roc_auc:.4f}" if result.roc_auc is not None else "n/a"
            lines.append(
                f"{result.model_name:<28}{result.accuracy:>9.4f}{result.precision:>9.4f}"
                f"{result.recall:>9.4f}{result.f1:>9.4f}{result.macro_f1:>10.4f}{roc:>10}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "blocklist_preset": self.blocklist_preset,
            "split_sizes": self.split_sizes,
            "feature_info": self.feature_info,
            "timings_seconds": self.timings,
            "parameter_counts": self.parameter_counts,
            "results": {name: result.to_dict() for name, result in self.results.items()},
        }

    def save(self, path: Path, *, save_predictions: bool = True) -> Path:
        """Write metrics as JSON, and the raw prediction arrays alongside."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

        if save_predictions:
            np.savez_compressed(
                path.with_suffix(".predictions.npz"),
                y_test=self.y_test,
                **{f"pred_{name}": array for name, array in self.predictions.items()},
                **{f"proba_{name}": array for name, array in self.probabilities.items()},
            )
        return path


def split_indices(
    labels: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices of the stratified 70/10/20 partition.

    Exposed separately from :func:`make_splits` so an analysis can join saved
    predictions back to the articles that produced them. Recomputing the split
    by hand elsewhere would be one edit away from silently disagreeing with the
    split the models were actually trained on.

    Returns:
        ``(train_idx, val_idx, test_idx)``.
    """
    split = config.split
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=split.test_size,
        random_state=config.seed,
        stratify=labels if split.stratify else None,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=split.val_size_of_remainder,
        random_state=config.seed,
        stratify=labels[train_idx] if split.stratify else None,
    )
    return train_idx, val_idx, test_idx


def make_splits(
    texts: np.ndarray,
    labels: np.ndarray,
    pos_features: np.ndarray | None,
    config: ExperimentConfig,
) -> SplitData:
    """Stratified 70/10/20 partition, reproducible from ``config.seed``."""
    train_idx, val_idx, test_idx = split_indices(labels, config)

    def take_pos(index: np.ndarray) -> np.ndarray | None:
        return pos_features[index] if pos_features is not None else None

    return SplitData(
        train_texts=texts[train_idx],
        val_texts=texts[val_idx],
        test_texts=texts[test_idx],
        y_train=labels[train_idx],
        y_val=labels[val_idx],
        y_test=labels[test_idx],
        pos_train=take_pos(train_idx),
        pos_val=take_pos(val_idx),
        pos_test=take_pos(test_idx),
    )


def run_pipeline(
    config: ExperimentConfig | None = None,
    *,
    blocklist_preset: str = "none",
    splits: SplitData | None = None,
    save_models_to: Path | str | None = None,
    run_name: str | None = None,
    save_results: bool = True,
    verbose: int = 1,
) -> PipelineResult:
    """Train XGBoost, TextCNN and the three ensembles, then evaluate on the test split.

    Args:
        splits: A pre-built partition. Supplying one bypasses the stratified
            random split, which is what the publisher-holdout probe needs: its
            test set is defined by article provenance rather than by sampling.
    """
    config = config or ExperimentConfig()
    ensure_directories()
    set_global_seed(config.seed)

    logger.info(
        "Run: seed=%d pos=%s blocklist=%s", config.seed, config.use_pos_features, blocklist_preset
    )

    if splits is not None:
        data = splits
    else:
        texts, labels, pos_frame = load_cached_corpus(blocklist_preset=blocklist_preset)
        pos_features = (
            pos_frame[list(POS_FEATURE_COLUMNS)].to_numpy() if config.use_pos_features else None
        )
        data = make_splits(texts.to_numpy(), labels, pos_features, config)
    logger.info("Split sizes: %s", data.sizes)

    timings: dict[str, float] = {}
    feature_info: dict[str, float | int] = {}

    # ---------------------------------------------------------------- XGBoost
    started = time.perf_counter()
    tfidf = TfidfSvdEncoder(config.tfidf, random_state=config.seed)
    X_train_svd = tfidf.fit_transform(data.train_texts)
    X_val_svd = tfidf.transform(data.val_texts)
    X_test_svd = tfidf.transform(data.test_texts)

    X_train_xgb = combine_features(X_train_svd, data.pos_train)
    X_val_xgb = combine_features(X_val_svd, data.pos_val)
    X_test_xgb = combine_features(X_test_svd, data.pos_test)
    timings["tfidf_svd_seconds"] = time.perf_counter() - started

    feature_info["svd_components"] = tfidf.n_components
    feature_info["svd_explained_variance"] = tfidf.explained_variance
    feature_info["xgb_feature_dim"] = int(X_train_xgb.shape[1])
    feature_info["n_pos_features"] = 0 if data.pos_train is None else len(POS_FEATURE_COLUMNS)

    started = time.perf_counter()
    xgb_model = XGBoostModel(config.xgboost, random_state=config.seed)
    xgb_model.fit(X_train_xgb, data.y_train, X_val_xgb, data.y_val)
    timings["xgboost_train_seconds"] = time.perf_counter() - started

    # ---------------------------------------------------------------- TextCNN
    started = time.perf_counter()
    encoder = SequenceEncoder(config.sequence)
    X_train_seq = encoder.fit_transform(data.train_texts)
    X_val_seq = encoder.transform(data.val_texts)
    X_test_seq = encoder.transform(data.test_texts)
    timings["sequence_encoding_seconds"] = time.perf_counter() - started

    embedding_matrix, coverage = load_glove_embeddings(
        encoder.tokenizer,
        encoder.vocab_size,
        embedding_dim=config.sequence.embedding_dim,
    )
    feature_info["vocab_size"] = int(encoder.vocab_size)
    feature_info["glove_coverage"] = coverage

    cnn = TextCNNModel(
        vocab_size=encoder.vocab_size,
        max_len=config.sequence.max_len,
        embedding_dim=config.sequence.embedding_dim,
        embedding_matrix=embedding_matrix,
        config=config.text_cnn,
    )
    cnn.build()
    cnn.fit(X_train_seq, data.y_train, X_val_seq, data.y_val, verbose=verbose)
    timings["textcnn_train_seconds"] = cnn.train_seconds or 0.0
    timings["textcnn_best_epoch"] = float(cnn.best_epoch or 0)

    # ------------------------------------------------------------- Evaluation
    results: dict[str, ClassificationResult] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}

    for key, model, X_test in (("xgboost", xgb_model, X_test_xgb), ("textcnn", cnn, X_test_seq)):
        proba = model.predict_proba(X_test)
        pred = (proba > 0.5).astype(int)
        results[key] = evaluate(data.y_test, pred, proba, model_name=model.name)
        predictions[key] = pred
        probabilities[key] = proba

    X_test_list = [X_test_xgb, X_test_seq]
    base_models = [xgb_model, cnn]

    # The stacking meta-learner must see probabilities from a split the base
    # models were not trained on.
    val_probabilities = np.column_stack(
        [xgb_model.predict_proba(X_val_xgb), cnn.predict_proba(X_val_seq)]
    )

    for method in ENSEMBLE_METHODS:
        weights = None
        if method == "weighted_voting":
            weights = [results["xgboost"].f1, results["textcnn"].f1]

        ensemble = EnsembleModel(base_models, method=method, weights=weights)
        if method == "stacking":
            ensemble.fit_meta_model(val_probabilities, data.y_val, random_state=config.seed)

        proba = ensemble.predict_proba(X_test_list)
        pred = (proba > 0.5).astype(int)
        key = f"ensemble_{method}"
        results[key] = evaluate(data.y_test, pred, proba, model_name=ensemble.name)
        predictions[key] = pred
        probabilities[key] = proba

    parameter_counts = {
        "xgboost": xgb_model.parameter_counts,
        "textcnn": cnn.parameter_counts,
    }

    result = PipelineResult(
        config=config,
        blocklist_preset=blocklist_preset,
        split_sizes=data.sizes,
        results=results,
        predictions=predictions,
        probabilities=probabilities,
        y_test=data.y_test,
        timings=timings,
        feature_info=feature_info,
        parameter_counts=parameter_counts,
    )

    # Metrics are written before model artifacts: serialisation is the most
    # fragile step here, and a failure in it must not discard the run's results.
    if save_results:
        name = run_name or f"run_seed{config.seed}_pos{int(config.use_pos_features)}_{blocklist_preset}"
        result.save(RESULTS_DIR / f"{name}.json")

    if save_models_to is not None:
        directory = Path(save_models_to)
        _save_run_artifacts(directory, xgb_model, cnn, tfidf, encoder, config, blocklist_preset)
        # The served model is the soft-voting ensemble.
        EnsembleModel(base_models, method="soft_voting").save(directory)

    return result


def _save_run_artifacts(
    directory: Path,
    xgb_model: XGBoostModel,
    cnn: TextCNNModel,
    tfidf: TfidfSvdEncoder,
    encoder: SequenceEncoder,
    config: ExperimentConfig,
    blocklist_preset: str,
) -> None:
    """Persist fitted models and the encoders needed to reproduce their inputs."""
    import pickle

    directory.mkdir(parents=True, exist_ok=True)
    xgb_model.save(directory)
    cnn.save(directory)

    with (directory / "encoders.pkl").open("wb") as handle:
        pickle.dump({"tfidf": tfidf, "sequences": encoder}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "blocklist_preset": blocklist_preset,
                "vocab_size": encoder.vocab_size,
                "max_len": config.sequence.max_len,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved model artifacts to %s", directory)
