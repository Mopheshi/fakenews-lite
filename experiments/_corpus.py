"""Aligning the raw WELFake CSV to the rows held in the preprocessing cache.

Two experiments need the original article text rather than the cleaned text the
cache stores: the error analysis has to be readable by a human, and the
publisher-holdout probe has to find "(Reuters)" datelines that cleaning
destroys (parentheses are stripped, so the dateline and an ordinary mention of
the agency become indistinguishable).

The cache drops the rows that clean to an empty string and then reindexes, so
raw row ``i`` and cached row ``i`` are not the same article. This module
reproduces that drop and checks the alignment against the cached labels before
handing anything back, because a silent off-by-one here would corrupt every
number downstream.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from fakenews.data.loader import load_welfake
from fakenews.data.preprocessing import RETAINED_STOPWORDS, clean_text

logger = logging.getLogger(__name__)

#: A wire dateline as it appears in the raw text, e.g. "WASHINGTON (Reuters) -".
#: Matching the parenthesised form rather than the bare token is the whole point:
#: a dateline marks the article's origin, while "reuters" in running prose is
#: usually just a citation.
DATELINE_RE = re.compile(r"\(reuters\)", flags=re.IGNORECASE)


def _surviving_mask(titles: pd.Series, texts: pd.Series) -> np.ndarray:
    """Rows the cache keeps: those with at least one non-stopword token.

    Mirrors :func:`fakenews.data.cache.build_cache`, which drops a row when
    ``preprocess(title) + " " + preprocess(text)`` is empty. Lemmatisation never
    empties a token, so cleaning and stopword removal alone decide the outcome
    and the expensive WordNet pass can be skipped.
    """
    stop_words = set(stopwords.words("english")) - RETAINED_STOPWORDS

    def has_content(title: object, text: object) -> bool:
        combined = f"{clean_text(title)} {clean_text(text)}"
        return any(token not in stop_words for token in combined.split())

    return np.fromiter(
        (has_content(title, text) for title, text in zip(titles, texts, strict=True)),
        dtype=bool,
        count=len(titles),
    )


def load_raw_aligned(cached_labels: np.ndarray) -> pd.DataFrame:
    """Return the raw corpus reduced and reindexed to match the cache.

    Args:
        cached_labels: Label array from
            :func:`fakenews.data.cache.load_cached_corpus`, used to verify the
            alignment.

    Returns:
        A frame with ``title``, ``text``, ``label`` and ``has_dateline``, whose
        row order matches the cached corpus exactly.

    Raises:
        RuntimeError: If the reconstructed row set does not reproduce the cached
            labels, which means the cache was built from different source data.
    """
    raw = load_welfake()
    raw["title"] = raw["title"].fillna("")
    raw["text"] = raw["text"].fillna("")

    keep = _surviving_mask(raw["title"], raw["text"])
    aligned = raw.loc[keep, ["title", "text", "label"]].reset_index(drop=True)

    if len(aligned) != len(cached_labels) or not np.array_equal(
        aligned["label"].to_numpy(), cached_labels
    ):
        raise RuntimeError(
            f"Raw corpus does not align with the cache: reconstructed {len(aligned)} rows "
            f"against {len(cached_labels)} cached. Rebuild the cache with "
            "`python -m fakenews.cli build-cache --overwrite`."
        )

    aligned["has_dateline"] = aligned["text"].str.contains(DATELINE_RE, na=False)
    logger.info(
        "Aligned %d raw articles to the cache (%d dropped, %d carry a wire dateline)",
        len(aligned),
        len(raw) - len(aligned),
        int(aligned["has_dateline"].sum()),
    )
    return aligned
