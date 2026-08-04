"""Loading the WELFake corpus and reporting its label distribution."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fakenews.config import WELFAKE_CSV

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("title", "text", "label")

#: Label counts published alongside the dataset, used as an integrity check.
EXPECTED_LABEL_COUNTS = {0: 35_028, 1: 37_106}


def load_welfake(path: Path | str | None = None, *, verify: bool = True) -> pd.DataFrame:
    """Load the WELFake CSV.

    Args:
        path: Location of ``WELFake_Dataset.csv``. Defaults to the configured path.
        verify: When True, warn if the label distribution differs from the
            published counts, which would indicate a modified or truncated copy.

    Returns:
        A frame with at least ``title``, ``text`` and ``label`` columns.
    """
    csv_path = Path(path) if path is not None else WELFAKE_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"WELFake dataset not found at {csv_path}. Download it from "
            "https://zenodo.org/records/4561253 and place it there, or set "
            "FAKENEWS_WELFAKE_CSV."
        )

    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    counts = df["label"].value_counts().to_dict()
    logger.info("Loaded %d articles from %s", len(df), csv_path)
    logger.info(
        "Label distribution: %d legit (0), %d fake (1)",
        counts.get(0, 0),
        counts.get(1, 0),
    )

    if verify and counts != EXPECTED_LABEL_COUNTS:
        logger.warning(
            "Label distribution %s differs from the published counts %s; results "
            "will not be comparable with the reported figures.",
            counts,
            EXPECTED_LABEL_COUNTS,
        )

    n_missing_titles = int(df["title"].isna().sum())
    n_missing_texts = int(df["text"].isna().sum())
    if n_missing_titles or n_missing_texts:
        logger.info(
            "Missing values: %d titles, %d texts (filled with empty strings)",
            n_missing_titles,
            n_missing_texts,
        )

    return df
