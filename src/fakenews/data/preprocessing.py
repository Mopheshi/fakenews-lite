"""Text cleaning, normalisation and the optional source-marker blocklist."""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Iterable, Sequence

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
_HTML_RE = re.compile(r"<.*?>")
_EMAIL_RE = re.compile(r"\S+@\S+")
_HANDLE_RE = re.compile(r"@\w+|#\w+")
_DIGIT_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

#: Negations carry sentiment and are deliberately kept despite being stopwords.
RETAINED_STOPWORDS = frozenset({"no", "not", "nor", "neither", "never", "none"})


# --------------------------------------------------------------------------- #
# Source-marker blocklists (used by the leakage experiment)
# --------------------------------------------------------------------------- #

#: Wire services and attribution artefacts embedded in article bodies. WELFake's
#: real-news portion inherits "(Reuters)" datelines from its ISOT source, which
#: makes the bare token `reuters` a near-perfect giveaway after punctuation is
#: stripped. Removing these tests how much of the score is publisher memorisation
#: rather than linguistic signal.
WIRE_SERVICE_MARKERS = frozenset(
    {
        "reuters", "ap", "afp", "upi", "ansa", "pti", "ians", "epa", "tass",
        "interfax", "getty", "apimages", "reuter", "wire", "newswire",
    }
)

#: Attribution phrases that survive cleaning as multi-word sequences.
ATTRIBUTION_PHRASES = (
    "featured image",
    "image via",
    "photo credit",
    "screen grab",
    "screengrab",
    "image credit",
    "photo by",
    "additional reporting by",
    "reporting by",
    "editing by",
)

#: Outlet names. Broader and blunter than the wire-service list: these also occur
#: as legitimate topical mentions, so removing them is a deliberately aggressive
#: stress test rather than a claim that they are pure leakage.
PUBLISHER_MARKERS = frozenset(
    {
        "cnn", "msnbc", "foxnews", "fox", "bbc", "npr", "nytimes", "nyt",
        "washingtonpost", "wapo", "wsj", "bloomberg", "cnbc", "cbs", "abc",
        "nbc", "guardian", "telegraph", "independent", "dailymail", "newsweek",
        "politico", "thehill", "huffpost", "huffingtonpost", "buzzfeed",
        "breitbart", "infowars", "salon", "vox", "vice", "slate", "usatoday",
        "sputnik", "rt", "activistpost", "zerohedge", "dailycaller",
        "dailywire", "newsmax", "oann", "gateway", "pundit",
    }
)

#: Named blocklist presets selectable from the command line.
BLOCKLIST_PRESETS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "wire": WIRE_SERVICE_MARKERS,
    "extended": WIRE_SERVICE_MARKERS | PUBLISHER_MARKERS,
}


def clean_text(text: object) -> str:
    """Lowercase, strip markup/URLs/emails/handles/digits/punctuation, collapse space."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _HANDLE_RE.sub("", text)
    text = _DIGIT_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def strip_source_markers(text: str, blocklist: Iterable[str]) -> str:
    """Remove publisher/wire-service tokens and attribution phrases from cleaned text.

    Operates on already-cleaned text, where punctuation is gone and "(Reuters)"
    has become the bare token ``reuters``.
    """
    blocked = frozenset(blocklist)
    if not blocked:
        return text

    for phrase in ATTRIBUTION_PHRASES:
        text = text.replace(phrase, " ")

    kept = [token for token in text.split() if token not in blocked]
    return " ".join(kept)


class TextPreprocessor:
    """Cleaning, stopword removal and lemmatisation for the WELFake corpus."""

    def __init__(self, *, blocklist: Iterable[str] | None = None) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english")) - RETAINED_STOPWORDS
        self.blocklist = frozenset(blocklist or ())
        # A corpus has orders of magnitude more tokens than distinct words, and
        # each WordNet lookup is comparatively slow, so memoising by word turns
        # the dominant cost of preprocessing into a dictionary hit.
        self._lemma_cache: dict[str, str] = {}

    def remove_stopwords(self, text: str) -> str:
        return " ".join(word for word in text.split() if word not in self.stop_words)

    def lemmatize_text(self, text: str) -> str:
        cache = self._lemma_cache
        lemmatize = self.lemmatizer.lemmatize

        lemmas = []
        for word in text.split():
            lemma = cache.get(word)
            if lemma is None:
                lemma = cache[word] = lemmatize(word)
            lemmas.append(lemma)
        return " ".join(lemmas)

    def preprocess(self, text: object) -> str:
        """Run the full cleaning pipeline on a single document."""
        cleaned = clean_text(text)
        if self.blocklist:
            cleaned = strip_source_markers(cleaned, self.blocklist)
        cleaned = self.remove_stopwords(cleaned)
        return self.lemmatize_text(cleaned)

    def clear_cache(self) -> None:
        """Release the memoised lemmas."""
        self._lemma_cache.clear()

    def preprocess_many(self, texts: Sequence[object]) -> list[str]:
        return [self.preprocess(text) for text in texts]
