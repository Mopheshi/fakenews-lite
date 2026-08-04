import pytest

from fakenews.data.preprocessing import (
    BLOCKLIST_PRESETS,
    RETAINED_STOPWORDS,
    TextPreprocessor,
    clean_text,
    strip_source_markers,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Visit https://example.com now", "visit now"),
        ("<p>Hello <b>world</b></p>", "hello world"),
        ("Contact me@example.com today", "contact today"),
        ("@handle and #tag remain?", "and remain"),
        ("In 2024 there were 300 cases", "in there were cases"),
        ("Mixed   CASE   spacing", "mixed case spacing"),
    ],
)
def test_clean_text_removes_noise(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_strip_source_markers_removes_wire_tokens():
    cleaned = clean_text("WASHINGTON (Reuters) - The senate voted today")
    assert "reuters" in cleaned.split()

    stripped = strip_source_markers(cleaned, BLOCKLIST_PRESETS["wire"])
    assert "reuters" not in stripped.split()
    assert "senate" in stripped.split()


def test_strip_source_markers_removes_attribution_phrases():
    stripped = strip_source_markers("the claim featured image via twitter", {"via"})
    assert "featured" not in stripped
    assert "image" not in stripped
    assert "claim" in stripped


def test_strip_source_markers_is_a_noop_for_empty_blocklist():
    assert strip_source_markers("reuters said so", []) == "reuters said so"


def test_negations_survive_stopword_removal():
    preprocessor = TextPreprocessor()
    result = preprocessor.remove_stopwords("this is not the case and never was")
    for negation in ("not", "never"):
        assert negation in result.split()
    assert "the" not in result.split()


def test_retained_stopwords_are_excluded_from_the_stopword_set():
    preprocessor = TextPreprocessor()
    assert not (RETAINED_STOPWORDS & preprocessor.stop_words)


def test_lemma_cache_matches_uncached_result():
    preprocessor = TextPreprocessor()
    text = "cars running dogs cars dogs"

    first = preprocessor.lemmatize_text(text)
    second = preprocessor.lemmatize_text(text)

    assert first == second
    assert preprocessor._lemma_cache["cars"] == "car"

    preprocessor.clear_cache()
    assert preprocessor.lemmatize_text(text) == first


def test_preprocess_applies_blocklist_end_to_end():
    with_markers = TextPreprocessor()
    without_markers = TextPreprocessor(blocklist=BLOCKLIST_PRESETS["wire"])

    raw = "LONDON (Reuters) - Officials confirmed the report."
    assert "reuters" in with_markers.preprocess(raw).split()
    assert "reuters" not in without_markers.preprocess(raw).split()


def test_blocklist_presets_are_nested():
    assert BLOCKLIST_PRESETS["none"] < BLOCKLIST_PRESETS["wire"]
    assert BLOCKLIST_PRESETS["wire"] < BLOCKLIST_PRESETS["extended"]
