"""How much of the score is publisher memorisation?

WELFake's real-news portion inherits "(Reuters)" datelines from its ISOT source.
Cleaning strips the parentheses but leaves the bare token `reuters`, which is
close to a perfect class indicator inside a 10,000-feature TF-IDF space. Any
reviewer who knows the dataset will assume some of the reported accuracy is
publisher recognition rather than linguistic signal.

This runs the pipeline under three conditions -- unmodified, with wire-service
markers removed, and with outlet names removed as well -- and reports the deltas.
Both models are retrained for each condition because the vocabulary changes.

The unmodified condition is reused from ``run_main.py`` when its results are
already on disk, since it is the identical configuration.

    python experiments/run_leakage.py
"""

from __future__ import annotations

import argparse
import logging

from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, RESULTS_DIR, ExperimentConfig
from fakenews.data.preprocessing import (
    ATTRIBUTION_PHRASES,
    BLOCKLIST_PRESETS,
    PUBLISHER_MARKERS,
    WIRE_SERVICE_MARKERS,
)
from fakenews.evaluation.plots import plot_leakage_deltas
from fakenews.pipeline import run_pipeline

logger = logging.getLogger(__name__)

CONDITION_LABELS = {
    "none": "Unmodified",
    "wire": "Wire markers removed",
    "extended": "Wire + outlets removed",
}

REPORTED_MODELS = ("xgboost", "textcnn", "ensemble_soft_voting")


def _run_or_load(preset: str, seed: int) -> dict:
    """Return ``{model_key: metrics_dict}`` for one blocklist condition.

    Reuses a saved run when one exists for this exact configuration; the
    pipeline is deterministic given the seed, so re-running it would only spend
    twenty minutes reproducing the same numbers.
    """
    name = f"run_seed{seed}_pos1_{preset}"
    path = RESULTS_DIR / f"{name}.json"

    if path.exists():
        import json

        logger.info("Reusing saved run for condition '%s'", preset)
        return json.loads(path.read_text(encoding="utf-8"))["results"]

    logger.info("=== Blocklist condition: %s ===", preset)
    config = ExperimentConfig(seed=seed, strip_source_markers=preset != "none")
    result = run_pipeline(config, blocklist_preset=preset, run_name=name)
    return {key: metrics.to_dict() for key, metrics in result.results.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITION_LABELS),
        choices=list(BLOCKLIST_PRESETS),
    )
    args = parser.parse_args()

    configure_logging()

    runs = {preset: _run_or_load(preset, args.seed) for preset in args.conditions}
    baseline = runs.get("none")

    rows = []
    for preset, results in runs.items():
        for key in REPORTED_MODELS:
            if key not in results:
                continue
            metrics = results[key]
            row = {
                "Condition": CONDITION_LABELS[preset],
                "Model": metrics["model"],
                "Accuracy": metrics["accuracy"],
                "F1 (fake)": metrics["f1_fake"],
                "ROC-AUC": metrics["roc_auc"],
            }
            if baseline is not None and key in baseline:
                row["Delta accuracy"] = metrics["accuracy"] - baseline[key]["accuracy"]
            rows.append(row)

    figure_paths = {}
    if len(runs) > 1:
        scores = {
            CONDITION_LABELS[preset]: {
                results[key]["model"]: results[key]["accuracy"]
                for key in REPORTED_MODELS
                if key in results
            }
            for preset, results in runs.items()
        }
        figure_paths = plot_leakage_deltas(scores)

    blocklist_rows = [
        {
            "Preset": preset,
            "Tokens removed": len(tokens),
            "Examples": ", ".join(sorted(tokens)[:8]) or "(none)",
        }
        for preset, tokens in BLOCKLIST_PRESETS.items()
    ]

    write_report(
        "source_leakage",
        payload={
            "seed": args.seed,
            "blocklists": {
                "wire_service_markers": sorted(WIRE_SERVICE_MARKERS),
                "publisher_markers": sorted(PUBLISHER_MARKERS),
                "attribution_phrases": list(ATTRIBUTION_PHRASES),
            },
            "runs": runs,
            "figures": {kind: str(path) for kind, path in figure_paths.items()},
        },
        tables=[
            ("Performance under source-marker removal", markdown_table(rows)),
            ("Blocklist composition", markdown_table(blocklist_rows)),
        ],
    )

    print()
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
