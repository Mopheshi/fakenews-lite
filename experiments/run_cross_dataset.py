"""Does the model survive a corpus it was never trained on?

Every number in the paper comes from a random split of WELFake. Reviewer 1 asked
the obvious follow-up: a corpus is a sample of one collection process, and a
model can learn the collection rather than the phenomenon. This scores the
saved reference model, unchanged and untuned, on 980 articles from Perez-Rosas
et al. (COLING 2018) -- a corpus built years apart by different people with a
different definition of "fake".

The two sub-collections are reported separately on purpose, because they ask
different questions:

* **Celebrity** takes its fake class from tabloid and gossip sites. That is the
  kind of text WELFake contains, so this is a transfer test within roughly the
  same phenomenon.
* **FakeNewsAMT** pairs each genuine article with a crowdworker's fabricated
  version of it. The fake class was written to imitate legitimate reporting and
  carries none of the stylistic tells of a content farm, so this is closer to a
  worst case, and it should be read as one.

Two properties of the target corpus have to be stated before any number is read.
It is balanced at 50/50, where WELFake's test split is 51.6% fake, so the 0.5
threshold inherited from training is not the threshold this corpus would have
chosen. And FakeNewsAMT articles average 122 words against WELFake's several
hundred, which matters for a model whose features are TF-IDF over trigrams.
Accuracy at the deployed threshold is therefore reported alongside ROC-AUC,
which is threshold-free, and alongside an explicitly labelled oracle threshold.
The gap between the first and the third is the part a practitioner could
recover by recalibrating; the AUC is the part that no threshold can fix.

No model is trained here. The corpus is read only at test time.

    python experiments/run_cross_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _report import configure_logging, load_result, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, MODELS_DIR
from fakenews.data.corpora import PerezRosasProvider
from fakenews.detector import FakeNewsDetector
from fakenews.evaluation.metrics import evaluate
from fakenews.evaluation.plots import plot_cross_dataset
from fakenews.evaluation.significance import proportion_interval

logger = logging.getLogger(__name__)

REPORTED_MODELS = ("xgboost", "textcnn", "ensemble_soft_voting")

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "textcnn": "TextCNN",
    "ensemble_soft_voting": "Ensemble (soft voting)",
}

#: Label for the WELFake test split in tables and figures.
REFERENCE_GROUP = "WELFake test (reference)"


def _oracle_threshold(labels: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Best accuracy any single threshold could reach, and where it sits.

    This is an oracle: it reads the labels it is scored against, so it is an
    upper bound on what recalibration could buy and never a reportable
    operating point. It is here to separate "the decision boundary moved" from
    "the classes are not separable", which accuracy at 0.5 alone cannot.
    """
    candidates = np.unique(np.concatenate([proba, [0.5]]))
    accuracies = [float(((proba > threshold).astype(int) == labels).mean()) for threshold in candidates]
    best = int(np.argmax(accuracies))
    return float(candidates[best]), accuracies[best]


def _score_group(
    name: str,
    labels: np.ndarray,
    proba: np.ndarray,
    model_label: str,
) -> dict:
    """Every figure reported for one (model, article group) pair."""
    predictions = (proba > 0.5).astype(int)
    metrics = evaluate(labels, predictions, proba, model_name=f"{model_label} ({name})")

    correct = int((predictions == labels).sum())
    lower, upper = proportion_interval(correct, len(labels))
    threshold, oracle_accuracy = _oracle_threshold(labels, proba)

    return {
        "group": name,
        "n": int(len(labels)),
        "n_fake": int(labels.sum()),
        "metrics": metrics.to_dict(),
        "accuracy_ci": [lower, upper],
        "mean_fake_probability": float(proba.mean()),
        "mean_fake_probability_by_class": {
            "legit": float(proba[labels == 0].mean()) if (labels == 0).any() else None,
            "fake": float(proba[labels == 1].mean()) if (labels == 1).any() else None,
        },
        "oracle_threshold": threshold,
        "oracle_accuracy": oracle_accuracy,
    }


def _reference_scores() -> dict[str, dict[str, float]]:
    """WELFake test-split metrics for the same models, from the reference run.

    Read from the saved record rather than recomputed: these are the numbers
    printed in the paper, and a cross-dataset table that quietly disagreed with
    Table 1 would be worse than no table.
    """
    try:
        record = load_result("main_results")
    except FileNotFoundError:
        logger.warning("main_results.json not found; the reference column will be omitted")
        return {}

    return {
        key: record["results"][key]
        for key in REPORTED_MODELS
        if key in record.get("results", {})
    }


def _group_frames(frame: pd.DataFrame, *, by_domain: bool) -> list[tuple[str, np.ndarray]]:
    """Row masks for each reported breakdown, coarse first."""
    groups: list[tuple[str, np.ndarray]] = [("All Perez-Rosas", np.ones(len(frame), dtype=bool))]

    for subset in sorted(frame["subset"].unique()):
        groups.append((subset, (frame["subset"] == subset).to_numpy()))

    if by_domain:
        domains = frame.loc[frame["subset"] == "FakeNewsAMT", "domain"].unique()
        for domain in sorted(domains):
            mask = ((frame["subset"] == "FakeNewsAMT") & (frame["domain"] == domain)).to_numpy()
            groups.append((f"FakeNewsAMT / {domain}", mask))

    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=MODELS_DIR / f"seed{DEFAULT_SEED}",
        help="Saved model directory to score with.",
    )
    parser.add_argument(
        "--no-domain-breakdown",
        action="store_true",
        help="Report only the two sub-collections, not the six FakeNewsAMT domains.",
    )
    parser.add_argument("--pos-jobs", type=int, default=-1, help="Workers for POS tagging.")
    args = parser.parse_args()

    configure_logging()

    if not args.models.exists():
        raise SystemExit(
            f"No saved model at {args.models}. Run "
            f"`python experiments/run_main.py --save-models` first."
        )

    provider = PerezRosasProvider()
    frame = provider.load_validated()
    labels = frame["label"].to_numpy()

    logger.info("Loading the reference model from %s", args.models)
    detector = FakeNewsDetector.load(args.models)

    started = time.perf_counter()
    probabilities = detector.predict_proba_many(
        frame["title"].tolist(), frame["text"].tolist(), pos_n_jobs=args.pos_jobs
    )
    score_seconds = time.perf_counter() - started
    logger.info("Scored %d articles in %.1f s", len(frame), score_seconds)

    reference = _reference_scores()
    groups = _group_frames(frame, by_domain=not args.no_domain_breakdown)

    per_model: dict[str, dict] = {}
    headline_rows: list[dict] = []
    breakdown_rows: list[dict] = []
    figure_scores: dict[str, dict[str, float]] = {}

    for key in REPORTED_MODELS:
        if key not in probabilities:
            logger.warning("Saved model directory holds no %s; skipping", key)
            continue

        proba = probabilities[key]
        label = MODEL_LABELS[key]
        scored = {name: _score_group(name, labels[mask], proba[mask], label) for name, mask in groups}
        per_model[key] = scored

        reference_accuracy = reference.get(key, {}).get("accuracy")
        # Reference first so it plots as the top row: the figure is read as a
        # fall from the training corpus, not a rise towards it.
        figure_scores[label] = {}
        if reference_accuracy is not None:
            figure_scores[label][REFERENCE_GROUP] = reference_accuracy
        figure_scores[label].update(
            {
                group: scored[group]["metrics"]["accuracy"]
                for group in ("Celebrity", "FakeNewsAMT")
                if group in scored
            }
        )

        for group in ("All Perez-Rosas", "Celebrity", "FakeNewsAMT"):
            record = scored[group]
            metrics = record["metrics"]
            row = {
                "Model": label,
                "Corpus": group,
                "n": record["n"],
                "Accuracy": metrics["accuracy"],
                "95% CI": f"[{record['accuracy_ci'][0]:.3f}, {record['accuracy_ci'][1]:.3f}]",
                "F1 (fake)": metrics["f1_fake"],
                "ROC-AUC": metrics["roc_auc"],
                "FPR": metrics["false_positive_rate"],
                "Mean P(fake)": record["mean_fake_probability"],
                "Oracle threshold": record["oracle_threshold"],
                "Oracle accuracy": record["oracle_accuracy"],
            }
            if reference_accuracy is not None:
                row["WELFake accuracy"] = reference_accuracy
                row["Drop"] = reference_accuracy - metrics["accuracy"]
            headline_rows.append(row)

        for group, record in scored.items():
            if not group.startswith("FakeNewsAMT / "):
                continue
            breakdown_rows.append(
                {
                    "Model": label,
                    "Domain": group.removeprefix("FakeNewsAMT / "),
                    "n": record["n"],
                    "Accuracy": record["metrics"]["accuracy"],
                    "ROC-AUC": record["metrics"]["roc_auc"],
                    "FPR": record["metrics"]["false_positive_rate"],
                    "Mean P(fake)": record["mean_fake_probability"],
                }
            )

    figure_paths = plot_cross_dataset(figure_scores) if figure_scores else {}

    tables = [("Transfer to an unseen corpus", markdown_table(headline_rows))]
    if breakdown_rows:
        tables.append(("FakeNewsAMT by topical domain", markdown_table(breakdown_rows)))

    write_report(
        "cross_dataset",
        payload={
            "model_directory": str(args.models),
            "corpus": {
                "name": provider.metadata.name,
                "description": provider.metadata.description,
                "source_url": provider.metadata.source_url,
                "citation": provider.metadata.citation,
                "n_articles": int(len(frame)),
                "class_balance": frame["label"].value_counts().sort_index().to_dict(),
                "mean_words": float((frame["title"] + " " + frame["text"]).str.split().str.len().mean()),
            },
            "design": (
                "The saved seed-5 model is scored unchanged on the Perez-Rosas corpus. "
                "No retraining, no threshold fitting, no vocabulary refresh. The oracle "
                "threshold is an upper bound on recalibration and is not an operating point."
            ),
            "reference_welfake_test": reference,
            "results": per_model,
            "score_seconds": score_seconds,
            "figures": {kind: str(path) for kind, path in figure_paths.items()},
        },
        tables=tables,
    )

    print()
    print(markdown_table(headline_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
