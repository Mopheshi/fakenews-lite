"""Headline run: train and evaluate every model on the reference seed.

Produces the numbers behind the paper's Tables I-IV and Figure 1, saves the
fitted models for the efficiency benchmark, and saves the raw test predictions
so the significance tests operate on exactly these arrays.

    python experiments/run_main.py
"""

from __future__ import annotations

import argparse

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, MODELS_DIR, ExperimentConfig
from fakenews.evaluation.plots import plot_model_comparison
from fakenews.pipeline import run_pipeline

#: Models shown in the paper's comparison figure, in reading order.
FIGURE_MODELS = ("xgboost", "textcnn", "ensemble_soft_voting")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    configure_logging()

    config = ExperimentConfig(seed=args.seed)
    result = run_pipeline(
        config,
        save_models_to=None if args.no_save_models else MODELS_DIR / f"seed{args.seed}",
        run_name=f"run_seed{args.seed}_pos1_none",
    )

    print()
    print(result.summary_table())

    rows = []
    for metrics in result.results.values():
        rows.append(
            {
                "Model": metrics.model_name,
                "Accuracy": metrics.accuracy,
                "Precision (fake)": metrics.precision,
                "Recall (fake)": metrics.recall,
                "F1 (fake)": metrics.f1,
                "Macro-F1": metrics.macro_f1,
                "ROC-AUC": metrics.roc_auc,
            }
        )

    per_class_rows = []
    for metrics in result.results.values():
        for class_name, scores in metrics.per_class.items():
            per_class_rows.append(
                {
                    "Model": metrics.model_name,
                    "Class": class_name,
                    "Precision": scores["precision"],
                    "Recall": scores["recall"],
                    "F1": scores["f1"],
                    "Support": scores["support"],
                }
            )

    error_rows = [
        {
            "Model": metrics.model_name,
            "False positives (legit flagged fake)": metrics.false_positives,
            "False negatives (fake missed)": metrics.false_negatives,
            "False positive rate": metrics.false_positive_rate,
        }
        for metrics in result.results.values()
    ]

    figure_scores = {
        result.results[key].model_name: {
            "Accuracy": result.results[key].accuracy,
            "Precision": result.results[key].precision,
            "Recall": result.results[key].recall,
            "F1": result.results[key].f1,
            "ROC-AUC": result.results[key].roc_auc,
        }
        for key in FIGURE_MODELS
        if key in result.results
    }
    figure_paths = plot_model_comparison(figure_scores, stem="model_comparison")

    write_report(
        "main_results",
        payload={
            **result.to_dict(),
            "figures": {kind: str(path) for kind, path in figure_paths.items()},
        },
        tables=[
            ("Overall performance", markdown_table(rows)),
            ("Per-class performance", markdown_table(per_class_rows)),
            ("Error breakdown", markdown_table(error_rows)),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
