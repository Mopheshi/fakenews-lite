"""Does the POS feature block actually do anything?

The paper hypothesises that fake news carries more adjectives and adverbs and
fewer nouns. This experiment tests that claim three ways:

1. Class-conditional POS ratios with Cohen's d and Welch t-tests, so the
   hypothesis is measured rather than asserted.
2. XGBoost trained with and without the six POS columns on an identical split,
   compared with a paired McNemar test.
3. The share of total XGBoost gain the POS columns attract, which says whether
   the trees use them at all next to 300 SVD components.

Only XGBoost is retrained: the POS block never reaches the TextCNN. The
ensemble delta is computed by recombining each XGBoost variant with the TextCNN
probabilities saved by ``run_main.py``.

    python experiments/run_pos_ablation.py
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, RESULTS_DIR, ExperimentConfig
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.metrics import evaluate
from fakenews.evaluation.plots import plot_pos_effect_sizes
from fakenews.evaluation.significance import class_conditional_effect, mcnemar_test
from fakenews.features.pos import POS_FEATURE_COLUMNS
from fakenews.features.tfidf import TfidfSvdEncoder, combine_features
from fakenews.models.xgboost_model import XGBoostModel
from fakenews.pipeline import make_splits
from fakenews.seeding import set_global_seed

logger = logging.getLogger(__name__)


def _interpret(d: float) -> str:
    """Cohen's conventional bands, so the reader is not left to guess."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig(seed=args.seed)
    set_global_seed(config.seed)

    texts, labels, pos_frame = load_cached_corpus(blocklist_preset="none")
    pos_matrix = pos_frame[list(POS_FEATURE_COLUMNS)].to_numpy()

    # ------------------------------------------------ 1. class-conditional stats
    effects = [
        class_conditional_effect(pos_matrix[:, index], labels, feature=column)
        for index, column in enumerate(POS_FEATURE_COLUMNS)
    ]
    effect_rows = [
        {
            "POS category": effect.feature.replace("pos_", "").title(),
            "Mean (legit)": effect.mean_legit,
            "Mean (fake)": effect.mean_fake,
            "Difference": effect.mean_fake - effect.mean_legit,
            "Cohen's d": effect.cohens_d,
            "Magnitude": _interpret(effect.cohens_d),
            "p": f"{effect.p_value:.2e}",
        }
        for effect in effects
    ]

    figure_paths = plot_pos_effect_sizes([effect.to_dict() for effect in effects])

    # -------------------------------------------------------- 2. XGBoost ablation
    data = make_splits(texts.to_numpy(), labels, pos_matrix, config)

    tfidf = TfidfSvdEncoder(config.tfidf, random_state=config.seed)
    X_train_svd = tfidf.fit_transform(data.train_texts)
    X_val_svd = tfidf.transform(data.val_texts)
    X_test_svd = tfidf.transform(data.test_texts)

    arms: dict[str, dict] = {}
    for arm, train_pos, val_pos, test_pos in (
        ("with_pos", data.pos_train, data.pos_val, data.pos_test),
        ("without_pos", None, None, None),
    ):
        X_train = combine_features(X_train_svd, train_pos)
        X_val = combine_features(X_val_svd, val_pos)
        X_test = combine_features(X_test_svd, test_pos)

        started = time.perf_counter()
        model = XGBoostModel(config.xgboost, random_state=config.seed)
        model.fit(X_train, data.y_train, X_val, data.y_val)
        elapsed = time.perf_counter() - started

        proba = model.predict_proba(X_test)
        pred = (proba > 0.5).astype(int)
        arms[arm] = {
            "model": model,
            "metrics": evaluate(data.y_test, pred, proba, model_name=f"XGBoost ({arm})"),
            "pred": pred,
            "proba": proba,
            "n_features": int(X_train.shape[1]),
            "train_seconds": elapsed,
        }
        logger.info("%s: accuracy %.4f", arm, arms[arm]["metrics"].accuracy)

    ablation = mcnemar_test(
        data.y_test,
        arms["with_pos"]["pred"],
        arms["without_pos"]["pred"],
        model_a="XGBoost with POS",
        model_b="XGBoost without POS",
    )

    ablation_rows = [
        {
            "Configuration": f"SVD only ({arms['without_pos']['n_features']} features)",
            "Accuracy": arms["without_pos"]["metrics"].accuracy,
            "F1 (fake)": arms["without_pos"]["metrics"].f1,
            "Macro-F1": arms["without_pos"]["metrics"].macro_f1,
            "ROC-AUC": arms["without_pos"]["metrics"].roc_auc,
        },
        {
            "Configuration": f"SVD + POS ({arms['with_pos']['n_features']} features)",
            "Accuracy": arms["with_pos"]["metrics"].accuracy,
            "F1 (fake)": arms["with_pos"]["metrics"].f1,
            "Macro-F1": arms["with_pos"]["metrics"].macro_f1,
            "ROC-AUC": arms["with_pos"]["metrics"].roc_auc,
        },
    ]
    delta_accuracy = arms["with_pos"]["metrics"].accuracy - arms["without_pos"]["metrics"].accuracy
    ablation_rows.append(
        {
            "Configuration": "Difference (POS - no POS)",
            "Accuracy": delta_accuracy,
            "F1 (fake)": arms["with_pos"]["metrics"].f1 - arms["without_pos"]["metrics"].f1,
            "Macro-F1": arms["with_pos"]["metrics"].macro_f1 - arms["without_pos"]["metrics"].macro_f1,
            "ROC-AUC": (arms["with_pos"]["metrics"].roc_auc or 0)
            - (arms["without_pos"]["metrics"].roc_auc or 0),
        }
    )

    # --------------------------------------------------- 3. gain importance share
    gains = arms["with_pos"]["model"].feature_importances("gain")
    n_svd = config.tfidf.svd_components
    pos_gains = gains[n_svd:]
    total_gain = float(gains.sum())
    ranks = (-gains).argsort().argsort() + 1  # 1 = most important

    importance_rows = [
        {
            "POS category": column.replace("pos_", "").title(),
            "Gain": float(pos_gains[index]),
            "Share of total gain": float(pos_gains[index] / total_gain) if total_gain else 0.0,
            f"Rank (of {len(gains)})": int(ranks[n_svd + index]),
        }
        for index, column in enumerate(POS_FEATURE_COLUMNS)
    ]
    pos_gain_share = float(pos_gains.sum() / total_gain) if total_gain else 0.0

    # ---------------------------------------------- 4. ensemble-level consequence
    ensemble_rows: list[dict] = []
    ensemble_payload: dict = {}
    predictions_path = RESULTS_DIR / f"run_seed{args.seed}_pos1_none.predictions.npz"
    if predictions_path.exists():
        saved = np.load(predictions_path)
        if np.array_equal(saved["y_test"], data.y_test):
            cnn_proba = saved["proba_textcnn"]
            for arm in ("with_pos", "without_pos"):
                blended = (arms[arm]["proba"] + cnn_proba) / 2.0
                metrics = evaluate(
                    data.y_test,
                    (blended > 0.5).astype(int),
                    blended,
                    model_name=f"Ensemble ({arm})",
                )
                ensemble_payload[arm] = metrics.to_dict()
                ensemble_rows.append(
                    {
                        "Configuration": "Ensemble with POS" if arm == "with_pos" else "Ensemble without POS",
                        "Accuracy": metrics.accuracy,
                        "F1 (fake)": metrics.f1,
                        "ROC-AUC": metrics.roc_auc,
                    }
                )
        else:
            logger.warning("Saved test split does not match; skipping the ensemble arm")
    else:
        logger.warning("%s not found; run run_main.py first for the ensemble arm", predictions_path.name)

    tables = [
        ("Class-conditional POS ratios", markdown_table(effect_rows)),
        ("XGBoost ablation", markdown_table(ablation_rows)),
        ("POS feature importance (gain)", markdown_table(importance_rows)),
    ]
    if ensemble_rows:
        tables.append(("Ensemble-level effect", markdown_table(ensemble_rows)))

    write_report(
        "pos_ablation",
        payload={
            "seed": args.seed,
            "n_documents": int(len(labels)),
            "class_conditional_effects": [effect.to_dict() for effect in effects],
            "ablation": {
                arm: {
                    "metrics": arms[arm]["metrics"].to_dict(),
                    "n_features": arms[arm]["n_features"],
                    "train_seconds": arms[arm]["train_seconds"],
                }
                for arm in arms
            },
            "delta_accuracy": delta_accuracy,
            "mcnemar": ablation.to_dict(),
            "pos_gain_share": pos_gain_share,
            "pos_importance": importance_rows,
            "ensemble": ensemble_payload,
            "figures": {kind: str(path) for kind, path in figure_paths.items()},
        },
        tables=tables,
    )

    print()
    print(markdown_table(effect_rows))
    print()
    print(markdown_table(ablation_rows))
    print()
    print(
        f"McNemar (with vs without POS): discordant {ablation.n_discordant} "
        f"({ablation.n_a_only} / {ablation.n_b_only}), p = {ablation.p_value:.4g}"
    )
    print(f"POS share of total XGBoost gain: {pos_gain_share:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
