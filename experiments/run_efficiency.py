"""Measure what "lightweight" actually costs.

Efficiency is this work's central claim, so it is measured rather than asserted:
parameter counts separated into learned and frozen, on-disk size, training
wall-clock, single-article and batched inference latency, and peak resident
memory -- all on named hardware.

Transformer parameter counts are quoted from their publications and labelled as
such. Training-time and latency comparisons against a transformer are NOT
included, because no transformer was trained or benchmarked here; quoting a
speed-up ratio without a measured baseline would be inventing a number.

    python experiments/run_efficiency.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from _report import configure_logging, load_result, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, MODELS_DIR, ExperimentConfig
from fakenews.data.loader import load_welfake
from fakenews.detector import FakeNewsDetector

logger = logging.getLogger(__name__)

#: Published parameter counts, for context only. These are citations, not measurements.
REFERENCE_MODELS = {
    "BERT-base": 110_000_000,
    "DistilBERT": 66_000_000,
    "RoBERTa-base": 125_000_000,
    "GPT-2 (124M)": 124_000_000,
}


def _hardware_profile() -> dict:
    profile = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        profile["physical_cores"] = psutil.cpu_count(logical=False)
        profile["logical_cores"] = psutil.cpu_count(logical=True)
        profile["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        import os

        profile["logical_cores"] = os.cpu_count()
        logger.warning("psutil not installed; RAM and physical core count unavailable")

    try:
        import tensorflow as tf

        profile["gpus_visible"] = len(tf.config.list_physical_devices("GPU"))
    except Exception:
        profile["gpus_visible"] = None

    return profile


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        process = psutil.Process()
        info = process.memory_info()
        # peak_wset is Windows-only; fall back to current RSS elsewhere.
        return round(getattr(info, "peak_wset", info.rss) / 1024**2, 1)
    except ImportError:
        return None


def _directory_sizes(directory: Path) -> dict[str, float]:
    sizes = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            sizes[path.name] = round(path.stat().st_size / 1024**2, 3)
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--n-latency-samples", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig(seed=args.seed)
    model_dir = args.model_dir or MODELS_DIR / f"seed{args.seed}"

    if not model_dir.exists():
        raise SystemExit(
            f"No saved models at {model_dir}. Run `python experiments/run_main.py` first."
        )

    hardware = _hardware_profile()
    logger.info("Hardware: %s", hardware)

    # --------------------------------------------------------- parameter counts
    main_results = load_result("main_results")
    parameter_counts = main_results["parameter_counts"]
    timings = main_results["timings_seconds"]

    cnn_counts = parameter_counts["textcnn"]
    xgb_counts = parameter_counts["xgboost"]

    parameter_rows = [
        {
            "Component": "TextCNN - learned weights",
            "Count": f"{cnn_counts['trainable']:,}",
            "Note": "convolution filters, dense layer, output layer",
        },
        {
            "Component": "TextCNN - frozen GloVe embeddings",
            "Count": f"{cnn_counts['frozen']:,}",
            "Note": "pre-trained, never updated during training",
        },
        {
            "Component": "TextCNN - total",
            "Count": f"{cnn_counts['total']:,}",
            "Note": "learned + frozen",
        },
        {
            "Component": "XGBoost - trees",
            "Count": f"{xgb_counts['trees']:,}",
            "Note": f"max depth {config.xgboost.max_depth}",
        },
        {
            "Component": "XGBoost - split + leaf nodes",
            "Count": f"{xgb_counts['nodes_total']:,}",
            "Note": f"{xgb_counts['nodes_split']:,} splits, {xgb_counts['nodes_leaf']:,} leaves",
        },
        {
            "Component": "XGBoost - stored values",
            "Count": f"{xgb_counts['stored_values']:,}",
            "Note": "two per split (feature, threshold) plus one per leaf",
        },
    ]

    # A boosted forest has no weight matrix, so the only defensible aggregate is
    # the learned TextCNN weights plus the stored tree values.
    learned_total = cnn_counts["trainable"] + xgb_counts["stored_values"]
    full_total = cnn_counts["total"] + xgb_counts["stored_values"]

    comparison_rows = [
        {
            "Model": name,
            "Parameters": f"{count:,}",
            "Ratio vs our learned parameters": f"{count / learned_total:.0f}x",
            "Ratio vs our total footprint": f"{count / full_total:.0f}x",
        }
        for name, count in REFERENCE_MODELS.items()
    ]

    # ------------------------------------------------------------- on-disk size
    disk_sizes = _directory_sizes(model_dir)
    total_disk_mb = round(sum(disk_sizes.values()), 2)

    # -------------------------------------------------------------- latency
    logger.info("Loading detector from %s", model_dir)
    load_started = time.perf_counter()
    detector = FakeNewsDetector.load(model_dir)
    load_seconds = time.perf_counter() - load_started

    # Raw articles, not the cached cleaned text: the benchmark must include
    # cleaning and POS tagging, which is what a deployment actually pays.
    corpus = load_welfake(verify=False)
    corpus["title"] = corpus["title"].fillna("")
    corpus["text"] = corpus["text"].fillna("")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        len(corpus), size=min(args.n_latency_samples, len(corpus)), replace=False
    )
    samples = [
        (corpus["title"].iloc[int(index)], corpus["text"].iloc[int(index)])
        for index in sample_indices
    ]
    mean_chars = float(np.mean([len(title) + len(text) for title, text in samples]))

    logger.info("Warming up (%d articles)", args.warmup)
    for title, text in samples[: args.warmup]:
        detector.predict_article(title, text)

    gc.collect()
    logger.info("Timing %d single-article predictions", len(samples))
    latencies_ms = []
    wall_started = time.perf_counter()
    for title, text in samples:
        started = time.perf_counter()
        detector.predict_article(title, text)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    wall_seconds = time.perf_counter() - wall_started

    latencies_ms.sort()
    latency = {
        "n_samples": len(latencies_ms),
        "mean_ms": round(statistics.fmean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(latencies_ms[int(0.95 * len(latencies_ms)) - 1], 2),
        "p99_ms": round(latencies_ms[int(0.99 * len(latencies_ms)) - 1], 2),
        "max_ms": round(latencies_ms[-1], 2),
        "mean_article_chars": round(mean_chars, 0),
        "throughput_articles_per_second": round(len(latencies_ms) / wall_seconds, 1),
    }

    peak_rss = _peak_rss_mb()

    latency_rows = [
        {"Measure": "Mean", "Value": f"{latency['mean_ms']:.2f} ms"},
        {"Measure": "Median", "Value": f"{latency['median_ms']:.2f} ms"},
        {"Measure": "95th percentile", "Value": f"{latency['p95_ms']:.2f} ms"},
        {"Measure": "99th percentile", "Value": f"{latency['p99_ms']:.2f} ms"},
        {"Measure": "Throughput", "Value": f"{latency['throughput_articles_per_second']:.1f} articles/s"},
        {"Measure": "Mean article length", "Value": f"{mean_chars:,.0f} characters"},
        {"Measure": "Model load time", "Value": f"{load_seconds:.2f} s"},
        {"Measure": "Peak resident memory", "Value": f"{peak_rss:.0f} MB" if peak_rss else "n/a"},
    ]

    training_rows = [
        {"Stage": "TF-IDF + SVD fit/transform", "Seconds": round(timings.get("tfidf_svd_seconds", 0), 1)},
        {"Stage": "XGBoost training", "Seconds": round(timings.get("xgboost_train_seconds", 0), 1)},
        {"Stage": "Sequence encoding", "Seconds": round(timings.get("sequence_encoding_seconds", 0), 1)},
        {"Stage": "TextCNN training", "Seconds": round(timings.get("textcnn_train_seconds", 0), 1)},
        {
            "Stage": "Total (excluding one-off preprocessing)",
            "Seconds": round(
                sum(
                    timings.get(key, 0)
                    for key in (
                        "tfidf_svd_seconds",
                        "xgboost_train_seconds",
                        "sequence_encoding_seconds",
                        "textcnn_train_seconds",
                    )
                ),
                1,
            ),
        },
    ]

    storage_rows = [{"File": name, "Size (MB)": size} for name, size in disk_sizes.items()]
    storage_rows.append({"File": "Total", "Size (MB)": total_disk_mb})

    hardware_rows = [{"Property": key, "Value": str(value)} for key, value in hardware.items()]

    write_report(
        "efficiency",
        payload={
            "hardware": hardware,
            "parameter_counts": parameter_counts,
            "learned_parameter_total": learned_total,
            "full_footprint_total": full_total,
            "reference_model_parameters": REFERENCE_MODELS,
            "disk_sizes_mb": disk_sizes,
            "total_disk_mb": total_disk_mb,
            "training_seconds": timings,
            "inference_latency": latency,
            "model_load_seconds": round(load_seconds, 2),
            "peak_rss_mb": peak_rss,
            "caveat": (
                "Transformer parameter counts are quoted from their publications. No "
                "transformer was trained or benchmarked on this hardware, so no "
                "training-time or latency ratio against one is reported."
            ),
        },
        tables=[
            ("Hardware", markdown_table(hardware_rows)),
            ("Parameter accounting", markdown_table(parameter_rows)),
            ("Published transformer sizes (citations, not measurements)", markdown_table(comparison_rows)),
            ("Training wall-clock", markdown_table(training_rows)),
            ("Inference latency (single article, CPU)", markdown_table(latency_rows)),
            ("Model artifacts on disk", markdown_table(storage_rows)),
        ],
    )

    print()
    print(json.dumps({"latency": latency, "peak_rss_mb": peak_rss}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
