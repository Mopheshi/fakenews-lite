"""Part-of-speech ratio features.

Six coarse POS categories are counted per document and divided by the token
count, giving scale-free ratios. Tagging runs on the *original* text rather than
the cleaned text: stopword removal and lemmatisation destroy exactly the
syntactic structure the tagger needs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from nltk import pos_tag_sents, word_tokenize

logger = logging.getLogger(__name__)

#: Penn Treebank tags grouped into the six categories used as features.
POS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "nouns": ("NN", "NNS", "NNP", "NNPS"),
    "verbs": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adjectives": ("JJ", "JJR", "JJS"),
    "adverbs": ("RB", "RBR", "RBS"),
    "pronouns": ("PRP", "PRP$", "WP", "WP$"),
    "modals": ("MD",),
}

POS_FEATURE_COLUMNS: tuple[str, ...] = tuple(f"pos_{name}" for name in POS_CATEGORIES)

#: Reverse index from tag to category, built once.
_TAG_TO_CATEGORY: dict[str, str] = {
    tag: category for category, tags in POS_CATEGORIES.items() for tag in tags
}

#: Safety cap on tokens per document. Set far above the corpus maximum so it
#: never binds in practice; it exists only to bound the cost of a pathological
#: input. Documents are otherwise tagged in full.
MAX_TOKENS = 20_000


def _ratios_from_tags(tags: Sequence[tuple[str, str]]) -> dict[str, float]:
    total = len(tags)
    if total == 0:
        return {category: 0.0 for category in POS_CATEGORIES}

    counts = dict.fromkeys(POS_CATEGORIES, 0)
    for _, tag in tags:
        category = _TAG_TO_CATEGORY.get(tag)
        if category is not None:
            counts[category] += 1

    return {category: count / total for category, count in counts.items()}


def extract_pos_features(text: str) -> dict[str, float]:
    """Return the six POS ratios for one document."""
    if not text or not text.strip():
        return {category: 0.0 for category in POS_CATEGORIES}

    try:
        tokens = word_tokenize(text)[:MAX_TOKENS]
        if not tokens:
            return {category: 0.0 for category in POS_CATEGORIES}
        return _ratios_from_tags(pos_tag_sents([tokens])[0])
    except Exception:  # pragma: no cover - tagger failures must not abort a run
        logger.warning("POS tagging failed for a document; returning zero ratios")
        return {category: 0.0 for category in POS_CATEGORIES}


def _tag_chunk(texts: Sequence[str]) -> list[dict[str, float]]:
    """Tag a chunk of documents in one call so the tagger loads once per worker."""
    tokenised = []
    for text in texts:
        if not text or not text.strip():
            tokenised.append([])
            continue
        try:
            tokenised.append(word_tokenize(text)[:MAX_TOKENS])
        except Exception:
            tokenised.append([])

    tagged = pos_tag_sents(tokenised)
    return [_ratios_from_tags(tags) for tags in tagged]


def extract_pos_features_batch(
    texts: Sequence[str],
    *,
    n_jobs: int = -1,
    chunk_size: int = 500,
) -> np.ndarray:
    """Extract POS ratios for a corpus.

    Args:
        texts: Raw (uncleaned) document strings.
        n_jobs: Worker processes; ``-1`` uses every core, ``1`` runs inline.
        chunk_size: Documents per task, amortising tagger load across a chunk.

    Returns:
        Array of shape ``(len(texts), 6)`` with columns ordered as
        :data:`POS_FEATURE_COLUMNS`.
    """
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    logger.info("Tagging %d documents in %d chunks (n_jobs=%s)", len(texts), len(chunks), n_jobs)

    if n_jobs == 1:
        results: list[dict[str, float]] = []
        for index, chunk in enumerate(chunks, start=1):
            results.extend(_tag_chunk(chunk))
            if index % 20 == 0:
                logger.info("  tagged %d/%d documents", index * chunk_size, len(texts))
    else:
        from joblib import Parallel, delayed

        chunk_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_tag_chunk)(chunk) for chunk in chunks
        )
        results = [item for chunk_result in chunk_results for item in chunk_result]

    return np.asarray(
        [[row[category] for category in POS_CATEGORIES] for row in results],
        dtype=np.float64,
    )
