from fakenews.config import (
    CLASS_NAMES,
    LABEL_NAMES,
    POSITIVE_CLASS,
    ExperimentConfig,
    SplitConfig,
)


def test_label_mapping_is_count_consistent():
    """0 = legit, 1 = fake -- the reading the published counts force.

    See docs/label-semantics.md.
    """
    assert LABEL_NAMES == {0: "Legit", 1: "Fake"}
    assert CLASS_NAMES == ["Legit", "Fake"]
    assert POSITIVE_CLASS == 1


def test_split_fractions_yield_seventy_ten_twenty():
    split = SplitConfig()
    test = split.test_size
    validation = (1 - test) * split.val_size_of_remainder
    train = 1 - test - validation

    assert round(train, 3) == 0.7
    assert round(validation, 3) == 0.1
    assert round(test, 3) == 0.2


def test_config_serialises_nested_dataclasses():
    payload = ExperimentConfig().to_dict()

    assert payload["seed"] == 5
    assert payload["tfidf"]["svd_components"] == 300
    assert payload["text_cnn"]["filter_sizes"] == (3, 4, 5)


def test_configs_are_immutable():
    import dataclasses

    import pytest

    config = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 99
