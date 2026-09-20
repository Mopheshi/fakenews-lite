"""Does the model still work on a publisher it has never been trained on?

The source-marker experiment in ``run_leakage.py`` answers a narrower question:
it deletes the token ``reuters`` and shows accuracy barely moves. A reviewer can
fairly reply that deleting a token does not remove house style, beat coverage or
the events a wire service happened to file on, all of which a random split
scatters across training and test alike.

This probe removes the publisher instead of the token. Every article carrying a
``(Reuters)`` dateline is withheld from training and validation entirely, and the
model is then asked to classify 3,000 of them. The comparison that matters is
against 3,000 legitimate articles from the publishers the model *did* train on,
drawn from the same corpus and scored by the same model at the same threshold.
If the false positive rate on the unseen publisher is close to the rate on the
seen ones, publisher identity is not what the model is keying on.

Two things this does not do, stated up front. Withholding the dateline articles
takes most of the legitimate class with them, so the training set is rebalanced
by subsampling the fake class and ends up at roughly 19,000 articles against the
50,487 of the main run. Absolute accuracy is therefore lower than Table 1 and
should not be read against it; both conditions are scored by the same model, so
the difference between them is what carries the meaning. And "Reuters" is one
publisher, not a held-out corpus, so this bounds the concern rather than
settling it.

    python experiments/run_source_holdout.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from _corpus import load_raw_aligned
from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, ExperimentConfig
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.metrics import evaluate
from fakenews.evaluation.significance import two_proportion_test
from fakenews.features.pos import POS_FEATURE_COLUMNS
from fakenews.pipeline import SplitData, run_pipeline

logger = logging.getLogger(__name__)

#: Articles per test group. Both legitimate groups are this size so the two
#: false positive rates are estimated with equal precision.
DEFAULT_GROUP_SIZE = 3_000

REPORTED_MODELS = ("xgboost", "textcnn", "ensemble_soft_voting")

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "textcnn": "TextCNN",
    "ensemble_soft_voting": "Ensemble (soft voting)",
}


def _build_splits(
    texts: np.ndarray,
    labels: np.ndarray,
    pos_matrix: np.ndarray,
    has_dateline: np.ndarray,
    *,
    group_size: int,
    seed: int,
    val_fraction: float,
    balance_training: bool,
) -> tuple[SplitData, dict[str, int]]:
    """Partition the corpus by publisher provenance rather than at random.

    The test block is laid out as ``[seen legit | fake | unseen legit]`` so the
    three groups can be recovered from a single prediction array by slicing.

    Args:
        balance_training: Subsample the fake class so training sees the two
            classes equally. Withholding the dateline articles removes 60% of
            the legitimate class, and a 76%-fake training set would raise both
            false positive rates enough to obscure the comparison they exist to
            support.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))

    unseen_legit_pool = indices[has_dateline & (labels == 0)]
    seen_legit_pool = indices[~has_dateline & (labels == 0)]
    seen_fake_pool = indices[~has_dateline & (labels == 1)]

    for name, pool in (
        ("unseen legit", unseen_legit_pool),
        ("seen legit", seen_legit_pool),
        ("fake", seen_fake_pool),
    ):
        if len(pool) < group_size:
            raise ValueError(f"Only {len(pool)} {name} articles available, need {group_size}")

    test_seen_legit = rng.choice(seen_legit_pool, group_size, replace=False)
    test_fake = rng.choice(seen_fake_pool, group_size, replace=False)
    test_unseen_legit = rng.choice(unseen_legit_pool, group_size, replace=False)
    test_idx = np.concatenate([test_seen_legit, test_fake, test_unseen_legit])

    # Training draws only on articles without a dateline, minus the held-in test
    # block. No dateline article reaches the model at any point.
    held_in = np.concatenate([test_seen_legit, test_fake])
    train_pool = np.setdiff1d(indices[~has_dateline], held_in, assume_unique=False)

    if balance_training:
        pool_legit = train_pool[labels[train_pool] == 0]
        pool_fake = train_pool[labels[train_pool] == 1]
        keep = min(len(pool_legit), len(pool_fake))
        train_pool = np.concatenate(
            [
                rng.choice(pool_legit, keep, replace=False),
                rng.choice(pool_fake, keep, replace=False),
            ]
        )

    train_idx, val_idx = train_test_split(
        train_pool,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels[train_pool],
    )

    splits = SplitData(
        train_texts=texts[train_idx],
        val_texts=texts[val_idx],
        test_texts=texts[test_idx],
        y_train=labels[train_idx],
        y_val=labels[val_idx],
        y_test=labels[test_idx],
        pos_train=pos_matrix[train_idx],
        pos_val=pos_matrix[val_idx],
        pos_test=pos_matrix[test_idx],
    )

    composition = {
        "train_total": len(train_idx),
        "train_legit": int((labels[train_idx] == 0).sum()),
        "train_fake": int((labels[train_idx] == 1).sum()),
        "validation_total": len(val_idx),
        "test_seen_legit": group_size,
        "test_fake": group_size,
        "test_unseen_legit": group_size,
        "dateline_articles_excluded": int(has_dateline.sum()),
        "training_balanced": int(balance_training),
    }
    return splits, composition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--no-balance-training",
        action="store_true",
        help="Keep the natural 76%% fake training prior instead of subsampling.",
    )
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig(seed=args.seed)

    texts, labels, pos_frame = load_cached_corpus(blocklist_preset="none")
    pos_matrix = pos_frame[list(POS_FEATURE_COLUMNS)].to_numpy()
    raw = load_raw_aligned(labels)
    has_dateline = raw["has_dateline"].to_numpy()

    splits, composition = _build_splits(
        texts.to_numpy(),
        labels,
        pos_matrix,
        has_dateline,
        group_size=args.group_size,
        seed=args.seed,
        val_fraction=config.split.val_size_of_remainder,
        balance_training=not args.no_balance_training,
    )
    logger.info("Split composition: %s", composition)

    result = run_pipeline(config, splits=splits, save_results=False, save_models_to=None)

    size = args.group_size
    rows = []
    comparisons = {}
    per_model = {}

    for key in REPORTED_MODELS:
        proba = result.probabilities[key]
        seen_legit, fake, unseen_legit = proba[:size], proba[size : 2 * size], proba[2 * size :]

        seen_false_positives = int((seen_legit > 0.5).sum())
        unseen_false_positives = int((unseen_legit > 0.5).sum())

        # Accuracy on each half uses the same 3,000 fake articles, so the two
        # figures differ only in which legitimate articles they contain.
        fake_correct = int((fake > 0.5).sum())
        seen_accuracy = (size - seen_false_positives + fake_correct) / (2 * size)
        unseen_accuracy = (size - unseen_false_positives + fake_correct) / (2 * size)

        comparison = two_proportion_test(
            unseen_false_positives,
            size,
            seen_false_positives,
            size,
            label_a="unseen publisher",
            label_b="seen publishers",
        )
        comparisons[key] = comparison.to_dict()

        per_model[key] = {
            "model": MODEL_LABELS[key],
            "seen": evaluate(
                np.concatenate([np.zeros(size, int), np.ones(size, int)]),
                (np.concatenate([seen_legit, fake]) > 0.5).astype(int),
                np.concatenate([seen_legit, fake]),
                model_name=f"{MODEL_LABELS[key]} (seen publishers)",
            ).to_dict(),
            "unseen": evaluate(
                np.concatenate([np.zeros(size, int), np.ones(size, int)]),
                (np.concatenate([unseen_legit, fake]) > 0.5).astype(int),
                np.concatenate([unseen_legit, fake]),
                model_name=f"{MODEL_LABELS[key]} (unseen publisher)",
            ).to_dict(),
        }

        rows.append(
            {
                "Model": MODEL_LABELS[key],
                "FPR, seen publishers": comparison.rate_b,
                "FPR, unseen publisher": comparison.rate_a,
                "Difference": comparison.difference,
                "95% CI": f"[{comparison.ci_lower:+.4f}, {comparison.ci_upper:+.4f}]",
                "p": f"{comparison.p_value:.3g}",
                "Acc. seen": seen_accuracy,
                "Acc. unseen": unseen_accuracy,
            }
        )

    composition_rows = [{"Quantity": key, "Articles": value} for key, value in composition.items()]

    write_report(
        "source_holdout",
        payload={
            "seed": args.seed,
            "group_size": args.group_size,
            "design": (
                "Every article carrying a (Reuters) dateline is excluded from training and "
                "validation. The false positive rate on 3,000 of those articles is compared "
                "against the rate on 3,000 legitimate articles from publishers present in "
                "training, scored by the same model at the same threshold."
            ),
            "composition": composition,
            "results": per_model,
            "comparisons": comparisons,
            "timings_seconds": result.timings,
        },
        tables=[
            ("False positive rate by publisher provenance", markdown_table(rows)),
            ("Split composition", markdown_table(composition_rows)),
        ],
    )

    print()
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
