"""Feature extraction: POS ratios, TF-IDF + SVD, and padded integer sequences.

``sequences`` pulls in TensorFlow, which costs several seconds to import, so the
submodules are resolved lazily. POS-only workflows such as the cache build never
touch TensorFlow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fakenews.features.pos import (
    POS_CATEGORIES,
    POS_FEATURE_COLUMNS,
    extract_pos_features,
    extract_pos_features_batch,
)

if TYPE_CHECKING:  # pragma: no cover
    from fakenews.features.sequences import SequenceEncoder, load_glove_embeddings
    from fakenews.features.tfidf import TfidfSvdEncoder, combine_features

_LAZY_EXPORTS = {
    "SequenceEncoder": "fakenews.features.sequences",
    "load_glove_embeddings": "fakenews.features.sequences",
    "TfidfSvdEncoder": "fakenews.features.tfidf",
    "combine_features": "fakenews.features.tfidf",
}

__all__ = [
    "POS_CATEGORIES",
    "POS_FEATURE_COLUMNS",
    "SequenceEncoder",
    "TfidfSvdEncoder",
    "combine_features",
    "extract_pos_features",
    "extract_pos_features_batch",
    "load_glove_embeddings",
]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(module_path), name)
