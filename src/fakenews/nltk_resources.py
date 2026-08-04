"""One-shot download of the NLTK corpora the pipeline depends on."""

from __future__ import annotations

import logging

import nltk

logger = logging.getLogger(__name__)

#: Resource identifier -> the path ``nltk.data.find`` uses to test for it.
REQUIRED_RESOURCES: dict[str, str] = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def ensure_nltk_resources(*, quiet: bool = True) -> None:
    """Download any missing NLTK resource. Safe to call repeatedly."""
    for resource, lookup in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(lookup)
        except LookupError:
            logger.info("Downloading NLTK resource %s", resource)
            nltk.download(resource, quiet=quiet)
