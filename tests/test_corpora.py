import pandas as pd
import pytest

from fakenews.data.corpora import CorpusMetadata, CorpusProvider, PerezRosasProvider


def write_article(root, subset, folder, name, content):
    directory = root / subset / folder
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(content, encoding="utf-8")


def test_headline_split_uses_the_blank_line_when_there_is_one():
    title, body = PerezRosasProvider._split_headline("A Short Headline\n\nThe body text.")

    assert title == "A Short Headline"
    assert body == "The body text."


def test_headline_split_falls_back_to_a_single_newline():
    title, body = PerezRosasProvider._split_headline("A Short Headline\nThe body text.")

    assert title == "A Short Headline"
    assert body == "The body text."


def test_a_long_leading_paragraph_is_body_not_a_headline():
    paragraph = "word " * 60  # Comfortably over MAX_HEADLINE_CHARS.
    content = f"{paragraph}\n\nMore body."

    title, body = PerezRosasProvider._split_headline(content)

    assert title == ""
    assert body == content


def test_an_unbroken_article_keeps_all_of_its_text():
    content = "word " * 80

    title, body = PerezRosasProvider._split_headline(content)

    assert title == ""
    assert body == content


def test_domain_comes_from_the_filename_prefix():
    provider = PerezRosasProvider()

    assert provider._domain_from_filename("biz09.fake.txt", subset="FakeNewsAMT") == "business"
    assert provider._domain_from_filename("entmt01.legit.txt", subset="FakeNewsAMT") == "entertainment"
    assert provider._domain_from_filename("001fake.txt", subset="Celebrity") == "celebrity"


def test_an_unrecognised_prefix_is_reported_rather_than_guessed():
    provider = PerezRosasProvider()

    assert provider._domain_from_filename("weather01.fake.txt", subset="FakeNewsAMT") == "unknown"


def test_loading_a_directory_tree_produces_the_shared_schema(tmp_path):
    write_article(tmp_path, "fakeNewsDataset", "fake", "biz01.fake.txt", "Headline\n\nBody.")
    write_article(tmp_path, "fakeNewsDataset", "legit", "biz01.legit.txt", "Other\n\nBody.")
    write_article(tmp_path, "celebrityDataset", "fake", "001fake.txt", "Gossip\n\nBody.")
    write_article(tmp_path, "celebrityDataset", "legit", "001legit.txt", "Real\n\nBody.")

    frame = PerezRosasProvider(tmp_path).load_validated()

    assert len(frame) == 4
    assert set(frame["label"]) == {0, 1}
    assert set(frame["subset"]) == {"FakeNewsAMT", "Celebrity"}
    assert {"title", "text", "label"} <= set(frame.columns)


def test_a_missing_dataset_directory_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download"):
        PerezRosasProvider(tmp_path / "absent").load()


def test_an_empty_dataset_directory_is_an_error_not_an_empty_frame(tmp_path):
    (tmp_path / "fakeNewsDataset" / "fake").mkdir(parents=True)

    with pytest.raises(ValueError, match="No article files"):
        PerezRosasProvider(tmp_path).load()


class StubProvider(CorpusProvider):
    """Minimal provider used to exercise the schema guarantees."""

    def __init__(self, frame):
        self._frame = frame

    @property
    def metadata(self):
        return CorpusMetadata(name="Stub", description="", source_url="", citation="")

    def load(self):
        return self._frame


def test_validation_rejects_a_missing_column():
    provider = StubProvider(pd.DataFrame({"title": ["a"], "label": [0]}))

    with pytest.raises(ValueError, match="missing required columns"):
        provider.load_validated()


def test_validation_rejects_labels_outside_the_binary_convention():
    provider = StubProvider(pd.DataFrame({"title": ["a"], "text": ["b"], "label": [2]}))

    with pytest.raises(ValueError, match="labels outside"):
        provider.load_validated()


def test_validation_fills_missing_text_rather_than_dropping_the_article():
    provider = StubProvider(pd.DataFrame({"title": ["a"], "text": [None], "label": [1]}))

    frame = provider.load_validated()

    assert frame.loc[0, "text"] == ""
    assert len(frame) == 1
