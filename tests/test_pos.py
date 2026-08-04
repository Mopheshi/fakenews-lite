import numpy as np

from fakenews.features.pos import (
    POS_CATEGORIES,
    POS_FEATURE_COLUMNS,
    extract_pos_features,
    extract_pos_features_batch,
)


def test_feature_columns_match_the_categories():
    assert tuple(f"pos_{name}" for name in POS_CATEGORIES) == POS_FEATURE_COLUMNS
    assert len(POS_FEATURE_COLUMNS) == 6


def test_ratios_are_bounded_and_sum_to_at_most_one():
    ratios = extract_pos_features("The quick brown fox quickly jumped over the very lazy dog.")

    assert set(ratios) == set(POS_CATEGORIES)
    assert all(0.0 <= value <= 1.0 for value in ratios.values())
    # The six categories do not cover every tag (determiners, prepositions...).
    assert sum(ratios.values()) <= 1.0


def test_empty_input_yields_zero_ratios():
    for text in ("", "   ", None or ""):
        ratios = extract_pos_features(text)
        assert all(value == 0.0 for value in ratios.values())


def test_adjective_heavy_text_scores_above_noun_heavy_text():
    adjectives = extract_pos_features("shocking outrageous unbelievable terrible disgraceful")
    nouns = extract_pos_features("committee report budget senate policy")

    assert adjectives["adjectives"] > nouns["adjectives"]
    assert nouns["nouns"] > adjectives["nouns"]


def test_batch_matches_single_document_extraction():
    texts = [
        "The senate approved the budget on Tuesday.",
        "SHOCKING! You will never believe what happened next.",
        "",
    ]

    batch = extract_pos_features_batch(texts, n_jobs=1, chunk_size=2)
    single = np.array(
        [[extract_pos_features(text)[category] for category in POS_CATEGORIES] for text in texts]
    )

    assert batch.shape == (3, 6)
    np.testing.assert_allclose(batch, single, rtol=1e-9)
