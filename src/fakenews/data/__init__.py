"""Dataset loading, text preprocessing and the preprocessing cache."""

from fakenews.data.cache import build_cache, load_cached_corpus
from fakenews.data.loader import load_welfake
from fakenews.data.preprocessing import TextPreprocessor, clean_text

__all__ = [
    "TextPreprocessor",
    "build_cache",
    "clean_text",
    "load_cached_corpus",
    "load_welfake",
]
