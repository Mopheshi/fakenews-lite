"""Is the ensemble's margin real, or is it noise?

The ensemble beats TextCNN by a fraction of a point on one split with one seed.
Two things are needed before that can be called a result:

* a paired McNemar test on the reference split, which asks whether the two
  models disagree asymmetrically rather than just scoring differently, and
* mean +/- SD across several seeds, which shows how much the split itself moves
  the number.

Runs already present in ``artifacts/results`` are reused, so this can be
interrupted and resumed.

    python experiments/run_seeds.py
"""

from __future__ import annotations

import argparse
import itertools
import logging

import numpy as np

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, RESULTS_DIR, STABILITY_SEEDS, ExperimentConfig
from fakenews.evaluation.significance import (
    bootstrap_accuracy_ci,
    mcnemar_test,
    paired_difference_ci,
)
from fakenews.pipeline import run_pipeline

logger = logging.getLogger(__name__)

REPORTED_MODELS = (
    "xgboost",
    "textcnn",
    "ensemble_soft_voting",
    "ensemble_weighted_voting",
    "ensemble_stacking",
)
#: Keys as emitted by ``ClassificationResult.to_dict`` -- note ``f1_fake``,
#: which names the class it is measured on because the positive class in this
#: dataset is ambiguous.
METRICS = ("accuracy", "f1_fake", "macro_f1", "roc_auc")


def _run_name(seed: int) -> str:
    return f"run_seed{seed}_pos1_none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STABILITY_SEEDS))
    parser.add_argument("--reference-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Re-run seeds even if results exist.")
    args = parser.parse_args()

    configure_logging()

    # ------------------------------------------------------------ run the seeds
    per_seed: dict[int, dict] = {}
    for seed in args.seeds:
        path = RESULTS_DIR / f"{_run_name(seed)}.json"
        if path.exists() and not args.force:
            import json

            logger.info("Reusing existing result for seed %d", seed)
            per_seed[seed] = json.loads(path.read_text(encoding="utf-8"))["results"]
            continue

        logger.info("=== Seed %d ===", seed)
        result = run_pipeline(ExperimentConfig(seed=seed), run_name=_run_name(seed))
        per_seed[seed] = {name: metrics.to_dict() for name, metrics in result.results.items()}

    # ------------------------------------------------------- stability across seeds
    stability_rows = []
    stability_payload: dict[str, dict[str, dict[str, float]]] = {}
    for key in REPORTED_MODELS:
        if not all(key in per_seed[seed] for seed in args.seeds):
            continue

        row: dict = {"Model": per_seed[args.seeds[0]][key]["model"]}
        stability_payload[key] = {}
        for metric in METRICS:
            values = np.array(
                [per_seed[seed][key][metric] for seed in args.seeds], dtype=float
            )
            row[metric.replace("_", "-")] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
            stability_payload[key][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        row["n seeds"] = len(args.seeds)
        stability_rows.append(row)

    # ------------------------------------- paired tests on the reference split
    predictions_path = RESULTS_DIR / f"{_run_name(args.reference_seed)}.predictions.npz"
    significance_rows: list[dict] = []
    significance_payload: list[dict] = []
    interval_rows: list[dict] = []

    if predictions_path.exists():
        saved = np.load(predictions_path)
        y_test = saved["y_test"]
        available = [key for key in REPORTED_MODELS if f"pred_{key}" in saved]

        for key in available:
            point, low, high = bootstrap_accuracy_ci(y_test, saved[f"pred_{key}"], seed=args.reference_seed)
            interval_rows.append(
                {
                    "Model": per_seed[args.reference_seed][key]["model"],
                    "Accuracy": point,
                    "95% CI": f"[{low:.4f}, {high:.4f}]",
                }
            )

        for key_a, key_b in itertools.combinations(available, 2):
            name_a = per_seed[args.reference_seed][key_a]["model"]
            name_b = per_seed[args.reference_seed][key_b]["model"]
            test = mcnemar_test(
                y_test, saved[f"pred_{key_a}"], saved[f"pred_{key_b}"], model_a=name_a, model_b=name_b
            )
            difference, low, high = paired_difference_ci(
                y_test, saved[f"pred_{key_a}"], saved[f"pred_{key_b}"], seed=args.reference_seed
            )
            significance_payload.append({**test.to_dict(), "accuracy_difference": difference})
            significance_rows.append(
                {
                    "Comparison": f"{name_a} vs {name_b}",
                    "A right / B wrong": test.n_a_only,
                    "B right / A wrong": test.n_b_only,
                    "Accuracy difference": difference,
                    "95% CI": f"[{low:+.4f}, {high:+.4f}]",
                    "p (McNemar)": f"{test.p_value:.4g}",
                    "Significant at 0.05": "yes" if test.p_value < 0.05 else "no",
                }
            )
    else:
        logger.warning("%s not found; skipping paired tests", predictions_path.name)

    write_report(
        "seed_stability",
        payload={
            "seeds": args.seeds,
            "reference_seed": args.reference_seed,
            "stability": stability_payload,
            "significance": significance_payload,
            "per_seed": per_seed,
        },
        tables=[
            ("Stability across seeds (mean ± SD)", markdown_table(stability_rows)),
            ("Accuracy with bootstrap confidence intervals", markdown_table(interval_rows)),
            ("Paired McNemar tests on the reference split", markdown_table(significance_rows)),
        ],
    )

    print()
    print(markdown_table(stability_rows))
    print()
    print(markdown_table(significance_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
