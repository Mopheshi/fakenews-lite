"""Why 300 SVD components, when 300 retains only a quarter of the variance?

Retained variance is a poor guide for a TF-IDF matrix: the spectrum is long and
flat, so a low percentage is expected and says little about task performance.
This sweeps the component count and reports both quantities, so the choice can
be justified by downstream accuracy instead of by a variance figure that sounds
alarming out of context.

Only XGBoost is trained -- the TextCNN never sees the SVD block.

Scoring is on the VALIDATION split: the sweep selects a hyperparameter, so the
test split must play no part in it.

    python experiments/run_svd_sweep.py
"""

from __future__ import annotations

import argparse
import logging
import time

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, ExperimentConfig, TfidfConfig
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.metrics import evaluate
from fakenews.evaluation.plots import plot_svd_sweep
from fakenews.features.pos import POS_FEATURE_COLUMNS
from fakenews.features.tfidf import TfidfSvdEncoder, combine_features
from fakenews.models.xgboost_model import XGBoostModel
from fakenews.pipeline import make_splits
from fakenews.seeding import set_global_seed

logger = logging.getLogger(__name__)

DEFAULT_COMPONENTS = (50, 100, 200, 300, 500, 800)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--components", type=int, nargs="+", default=list(DEFAULT_COMPONENTS))
    args = parser.parse_args()

    configure_logging()
    base_config = ExperimentConfig(seed=args.seed)
    set_global_seed(base_config.seed)

    texts, labels, pos_frame = load_cached_corpus(blocklist_preset="none")
    pos_matrix = pos_frame[list(POS_FEATURE_COLUMNS)].to_numpy()
    data = make_splits(texts.to_numpy(), labels, pos_matrix, base_config)

    records = []
    for k in args.components:
        logger.info("=== SVD components: %d ===", k)
        started = time.perf_counter()

        tfidf_config = TfidfConfig(**{**vars(base_config.tfidf), "svd_components": k})
        encoder = TfidfSvdEncoder(tfidf_config, random_state=args.seed)
        X_train = combine_features(encoder.fit_transform(data.train_texts), data.pos_train)
        X_val = combine_features(encoder.transform(data.val_texts), data.pos_val)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        model = XGBoostModel(base_config.xgboost, random_state=args.seed)
        model.fit(X_train, data.y_train, X_val, data.y_val)
        train_seconds = time.perf_counter() - started

        # Model selection is scored on the VALIDATION split. The test split must
        # not inform the choice of k, or every downstream number in the paper
        # becomes test-set-informed model selection.
        val_proba = model.predict_proba(X_val)
        val_metrics = evaluate(
            data.y_val, (val_proba > 0.5).astype(int), val_proba, model_name=f"XGBoost (k={k}, val)"
        )

        records.append(
            {
                "components": k,
                "explained_variance": encoder.explained_variance,
                "val_accuracy": val_metrics.accuracy,
                "val_f1": val_metrics.f1,
                "val_roc_auc": val_metrics.roc_auc,
                "feature_fit_seconds": round(fit_seconds, 1),
                "train_seconds": round(train_seconds, 1),
            }
        )
        logger.info(
            "k=%d: variance %.4f, VALIDATION accuracy %.4f",
            k,
            encoder.explained_variance,
            val_metrics.accuracy,
        )

    rows = [
        {
            "Components k": record["components"],
            "Explained variance": record["explained_variance"],
            "Validation accuracy": record["val_accuracy"],
            "Validation F1 (fake)": record["val_f1"],
            "Validation ROC-AUC": record["val_roc_auc"],
            "Feature fit (s)": record["feature_fit_seconds"],
            "XGBoost fit (s)": record["train_seconds"],
        }
        for record in records
    ]

    best = max(records, key=lambda record: record["val_accuracy"])
    logger.info(
        "Best k on validation: %d (accuracy %.4f)", best["components"], best["val_accuracy"]
    )

    figure_paths = plot_svd_sweep(
        [record["components"] for record in records],
        [record["val_accuracy"] for record in records],
        [record["explained_variance"] for record in records],
        chosen=base_config.tfidf.svd_components,
    )

    write_report(
        "svd_sweep",
        payload={
            "seed": args.seed,
            "scored_on": "validation",
            "records": records,
            "best_components_on_validation": best["components"],
            "best_validation_accuracy": best["val_accuracy"],
            "chosen_components": base_config.tfidf.svd_components,
            "figures": {kind: str(path) for kind, path in figure_paths.items()},
        },
        tables=[("SVD component sweep", markdown_table(rows))],
    )

    print()
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
