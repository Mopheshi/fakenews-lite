"""Preprocessing cache.

POS tagging the full corpus is the single most expensive step in the pipeline,
and it is identical across every experiment. This module runs cleaning and
tagging once and stores the result as Parquet, so ablations, seed sweeps and
leakage tests all start from the same materialised corpus.

Text variants for each blocklist preset are materialised together, because the
blocklist changes the text but not the POS ratios (which are computed from the
untouched original text).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.config import CACHE_DIR, ensure_directories
from fakenews.data.loader import load_welfake
from fakenews.data.preprocessing import BLOCKLIST_PRESETS, TextPreprocessor
from fakenews.features.pos import POS_FEATURE_COLUMNS, extract_pos_features_batch

logger = logging.getLogger(__name__)

CACHE_FILENAME = "welfake_preprocessed.parquet"

#: Column holding the cleaned text for each blocklist preset.
TEXT_COLUMNS = {name: f"combined_{name}" for name in BLOCKLIST_PRESETS}


def cache_path(directory: Path | None = None) -> Path:
    return (directory or CACHE_DIR) / CACHE_FILENAME


def build_cache(
    *,
    csv_path: Path | None = None,
    output_path: Path | None = None,
    n_jobs: int = -1,
    overwrite: bool = False,
) -> Path:
    """Clean, tag and materialise the corpus.

    Returns:
        Path to the written Parquet file.
    """
    ensure_directories()
    destination = output_path or cache_path()

    if destination.exists() and not overwrite:
        logger.info("Cache already present at %s; pass overwrite=True to rebuild", destination)
        return destination

    started = time.perf_counter()
    df = load_welfake(csv_path)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # POS ratios come from the original text, so they are computed once and are
    # shared by every blocklist variant.
    original = (df["title"] + " " + df["text"]).tolist()
    logger.info("Extracting POS features for %d documents", len(original))
    pos_started = time.perf_counter()
    pos_matrix = extract_pos_features_batch(original, n_jobs=n_jobs)
    pos_seconds = time.perf_counter() - pos_started
    logger.info("POS extraction finished in %.1f s", pos_seconds)

    out = pd.DataFrame({"label": df["label"].astype(np.int8)})
    for column, values in zip(POS_FEATURE_COLUMNS, pos_matrix.T, strict=True):
        out[column] = values.astype(np.float32)

    for preset, blocklist in BLOCKLIST_PRESETS.items():
        logger.info("Cleaning text variant '%s'", preset)
        preprocessor = TextPreprocessor(blocklist=blocklist)
        titles = df["title"].map(preprocessor.preprocess)
        texts = df["text"].map(preprocessor.preprocess)
        out[TEXT_COLUMNS[preset]] = (titles + " " + texts).str.strip()

    # A document that cleans to nothing carries no signal for either model. Rows
    # are dropped on the unblocklisted variant so every variant keeps the same
    # rows and stays directly comparable.
    baseline_column = TEXT_COLUMNS["none"]
    empty_mask = out[baseline_column].str.len() == 0
    n_dropped = int(empty_mask.sum())
    if n_dropped:
        logger.info("Dropping %d documents that clean to an empty string", n_dropped)
    out = out.loc[~empty_mask].reset_index(drop=True)

    out.attrs["n_dropped_empty"] = n_dropped
    out.to_parquet(destination, index=False, compression="zstd")

    metadata = pd.DataFrame(
        [
            {
                "n_source_rows": len(df),
                "n_cached_rows": len(out),
                "n_dropped_empty": n_dropped,
                "pos_seconds": round(pos_seconds, 2),
                "total_seconds": round(time.perf_counter() - started, 2),
            }
        ]
    )
    metadata.to_json(destination.with_suffix(".meta.json"), orient="records", indent=2)

    logger.info(
        "Cache written to %s (%d rows, %d dropped) in %.1f s",
        destination,
        len(out),
        n_dropped,
        time.perf_counter() - started,
    )
    return destination


def load_cached_corpus(
    *,
    blocklist_preset: str = "none",
    path: Path | None = None,
) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Load the materialised corpus for one blocklist preset.

    Returns:
        ``(texts, labels, pos_features)`` where ``pos_features`` is a frame with
        the six :data:`~fakenews.features.pos.POS_FEATURE_COLUMNS` columns.
    """
    if blocklist_preset not in TEXT_COLUMNS:
        raise ValueError(
            f"Unknown blocklist preset {blocklist_preset!r}; "
            f"expected one of {sorted(TEXT_COLUMNS)}"
        )

    source = path or cache_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Preprocessing cache not found at {source}. Build it with "
            "`python -m fakenews.cli build-cache`."
        )

    column = TEXT_COLUMNS[blocklist_preset]
    frame = pd.read_parquet(source, columns=[column, "label", *POS_FEATURE_COLUMNS])

    return (
        frame[column],
        frame["label"].to_numpy(),
        frame[list(POS_FEATURE_COLUMNS)],
    )
