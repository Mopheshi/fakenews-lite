"""Integer-sequence encoding and GloVe embedding matrix construction."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from fakenews.config import GLOVE_TXT, SequenceConfig

logger = logging.getLogger(__name__)


class SequenceEncoder:
    """Word-index tokenisation with fixed-length padding for the TextCNN."""

    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()
        self.tokenizer: Tokenizer | None = None
        self.vocab_size: int | None = None

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.tokenizer = Tokenizer(
            num_words=self.config.max_words, oov_token=self.config.oov_token
        )
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, self.config.max_words)
        logger.info("Sequence vocabulary: %d", self.vocab_size)
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Encoder has not been fitted")
        return pad_sequences(
            self.tokenizer.texts_to_sequences(texts),
            maxlen=self.config.max_len,
            padding="post",
            truncating="post",
        )


@lru_cache(maxsize=1)
def _load_glove_vectors(glove_path: Path) -> dict[str, np.ndarray]:
    """Parse a GloVe text file into a word -> vector mapping.

    Cached because a seed sweep re-reads the same 350 MB file for every run;
    call :func:`clear_glove_cache` to release the ~250 MB it holds.
    """
    vectors: dict[str, np.ndarray] = {}
    with glove_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            vectors[parts[0]] = np.asarray(parts[1:], dtype="float32")
    logger.info("Loaded %d GloVe vectors from %s", len(vectors), glove_path.name)
    return vectors


def clear_glove_cache() -> None:
    """Drop the parsed GloVe vectors held by :func:`_load_glove_vectors`."""
    _load_glove_vectors.cache_clear()


def load_glove_embeddings(
    tokenizer: Tokenizer,
    vocab_size: int,
    *,
    path: Path | None = None,
    embedding_dim: int = 100,
) -> tuple[np.ndarray, float]:
    """Build an embedding matrix for ``tokenizer``'s vocabulary from a GloVe file.

    Returns:
        ``(embedding_matrix, coverage)`` where coverage is the fraction of the
        vocabulary found in GloVe. Words not found keep a zero vector.
    """
    glove_path = Path(path) if path is not None else GLOVE_TXT
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe embeddings not found at {glove_path}. Download glove.6B.zip from "
            "https://nlp.stanford.edu/projects/glove/ or set FAKENEWS_GLOVE_TXT."
        )

    vectors = _load_glove_vectors(glove_path)

    matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    hits = 0
    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue
        vector = vectors.get(word)
        if vector is not None:
            matrix[index] = vector
            hits += 1

    coverage = hits / vocab_size
    logger.info("Embedding matrix %s, vocabulary coverage %.2f%%", matrix.shape, coverage * 100)
    return matrix, coverage
