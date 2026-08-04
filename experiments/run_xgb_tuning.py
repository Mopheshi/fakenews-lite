"""How much does the untuned XGBoost configuration give away?

The pipeline uses fixed hyperparameters and no search. Comparing a tuned
ensemble against an untuned baseline would flatter the ensemble, so this
quantifies the gap: if tuning buys little, the fixed configuration is a fair
baseline; if it buys a lot, the baseline must be described as a lower bound.

A randomised search is used rather than the full grid -- the exhaustive grid in
the original notebook is 2,187 combinations, which is not runnable on CPU.

    python experiments/run_xgb_tuning.py --n-candidates 20
"""

from __future__ import annotations

import argparse
import logging
import time

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, ExperimentConfig, XGBoostConfig
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.metrics import evaluate
from fakenews.evaluation.significance import mcnemar_test
from fakenews.features.pos import POS_FEATURE_COLUMNS
from fakenews.features.tfidf import TfidfSvdEncoder, combine_features
from fakenews.models.xgboost_model import XGBoostModel
from fakenews.pipeline import make_splits
from fakenews.seeding import set_global_seed

logger = logging.getLogger(__name__)

SEARCH_SPACE = {
    "max_depth": [4, 5, 6, 7, 9, 11],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": [100, 200, 300, 500],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.4],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig(seed=args.seed)
    set_global_seed(config.seed)

    texts, labels, pos_frame = load_cached_corpus(blocklist_preset="none")
    data = make_splits(
        texts.to_numpy(), labels, pos_frame[list(POS_FEATURE_COLUMNS)].to_numpy(), config
    )

    encoder = TfidfSvdEncoder(config.tfidf, random_state=config.seed)
    X_train = combine_features(encoder.fit_transform(data.train_texts), data.pos_train)
    X_val = combine_features(encoder.transform(data.val_texts), data.pos_val)
    X_test = combine_features(encoder.transform(data.test_texts), data.pos_test)

    # ------------------------------------------------------------- fixed baseline
    started = time.perf_counter()
    baseline = XGBoostModel(config.xgboost, random_state=config.seed)
    baseline.fit(X_train, data.y_train, X_val, data.y_val)
    baseline_seconds = time.perf_counter() - started

    baseline_proba = baseline.predict_proba(X_test)
    baseline_pred = (baseline_proba > 0.5).astype(int)
    baseline_metrics = evaluate(
        data.y_test, baseline_pred, baseline_proba, model_name="XGBoost (fixed configuration)"
    )

    # --------------------------------------------------------------- random search
    from sklearn.model_selection import RandomizedSearchCV

    logger.info("Randomised search: %d candidates x %d folds", args.n_candidates, args.cv)
    started = time.perf_counter()
    search = RandomizedSearchCV(
        XGBoostModel(config.xgboost, random_state=config.seed)._make_estimator(),
        SEARCH_SPACE,
        n_iter=args.n_candidates,
        cv=args.cv,
        scoring="f1",
        n_jobs=1,  # XGBoost already parallelises across cores.
        random_state=args.seed,
        verbose=2,
    )
    search.fit(X_train, data.y_train)
    search_seconds = time.perf_counter() - started

    tuned = XGBoostModel(config.xgboost, random_state=config.seed)
    tuned.model = search.best_estimator_
    tuned.best_params = search.best_params_

    tuned_proba = tuned.predict_proba(X_test)
    tuned_pred = (tuned_proba > 0.5).astype(int)
    tuned_metrics = evaluate(data.y_test, tuned_pred, tuned_proba, model_name="XGBoost (tuned)")

    test = mcnemar_test(
        data.y_test, tuned_pred, baseline_pred, model_a="XGBoost tuned", model_b="XGBoost fixed"
    )

    rows = [
        {
            "Configuration": "Fixed (as used in the paper)",
            "Accuracy": baseline_metrics.accuracy,
            "F1 (fake)": baseline_metrics.f1,
            "ROC-AUC": baseline_metrics.roc_auc,
            "Fit seconds": round(baseline_seconds, 1),
        },
        {
            "Configuration": f"Tuned ({args.n_candidates} candidates, {args.cv}-fold)",
            "Accuracy": tuned_metrics.accuracy,
            "F1 (fake)": tuned_metrics.f1,
            "ROC-AUC": tuned_metrics.roc_auc,
            "Fit seconds": round(search_seconds, 1),
        },
        {
            "Configuration": "Difference (tuned - fixed)",
            "Accuracy": tuned_metrics.accuracy - baseline_metrics.accuracy,
            "F1 (fake)": tuned_metrics.f1 - baseline_metrics.f1,
            "ROC-AUC": (tuned_metrics.roc_auc or 0) - (baseline_metrics.roc_auc or 0),
            "Fit seconds": "",
        },
    ]

    parameter_rows = [
        {
            "Hyperparameter": key,
            "Fixed": getattr(XGBoostConfig(), key, ""),
            "Tuned": value,
        }
        for key, value in sorted(search.best_params_.items())
    ]

    write_report(
        "xgboost_tuning",
        payload={
            "seed": args.seed,
            "n_candidates": args.n_candidates,
            "cv_folds": args.cv,
            "search_space": SEARCH_SPACE,
            "best_params": search.best_params_,
            "best_cv_f1": float(search.best_score_),
            "baseline": baseline_metrics.to_dict(),
            "tuned": tuned_metrics.to_dict(),
            "mcnemar": test.to_dict(),
            "search_seconds": round(search_seconds, 1),
        },
        tables=[
            ("Fixed versus tuned XGBoost", markdown_table(rows)),
            ("Selected hyperparameters", markdown_table(parameter_rows)),
        ],
    )

    print()
    print(markdown_table(rows))
    print(f"\nMcNemar (tuned vs fixed): p = {test.p_value:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
