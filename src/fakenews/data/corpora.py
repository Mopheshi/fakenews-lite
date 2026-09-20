"""Corpora behind one interface.

Until now the codebase had exactly one dataset and loading it was a function
call. The cross-dataset probe adds a second corpus with a completely different
storage layout -- one text file per article, label encoded in the directory name
-- and evaluation code should not have to know that. A provider hands back the
same three columns whichever corpus is behind it, so an experiment can iterate
over corpora instead of special-casing them.

Extra columns are allowed and carry whatever the corpus knows about its own
structure (topical domain, sub-collection). ``group_columns`` names them so a
report can break results down without hard-coding the corpus it is reading.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fakenews.config import PEREZ_ROSAS_DIR
from fakenews.data.loader import load_welfake

logger = logging.getLogger(__name__)

#: Schema every provider guarantees. ``label`` follows the WELFake convention
#: fixed in config.py: 0 = legitimate, 1 = fake.
CORPUS_COLUMNS = ("title", "text", "label")


@dataclass(frozen=True)
class CorpusMetadata:
    """Provenance for a corpus, recorded alongside any result computed from it."""

    name: str
    description: str
    source_url: str
    citation: str


class CorpusProvider(ABC):
    """A named source of labelled articles."""

    @property
    @abstractmethod
    def metadata(self) -> CorpusMetadata:
        """Provenance, written into experiment records."""

    @property
    def group_columns(self) -> tuple[str, ...]:
        """Extra columns worth breaking results down by. May be empty."""
        return ()

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the corpus as ``title``, ``text``, ``label`` and any extras."""

    def load_validated(self) -> pd.DataFrame:
        """``load`` plus a schema check, so a malformed corpus fails here.

        Raises:
            ValueError: A required column is missing, the frame is empty, or a
                label falls outside ``{0, 1}``.
        """
        frame = self.load()

        missing = [column for column in CORPUS_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.metadata.name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{self.metadata.name} loaded zero articles")

        labels = set(frame["label"].unique().tolist())
        if not labels <= {0, 1}:
            raise ValueError(f"{self.metadata.name} has labels outside {{0, 1}}: {sorted(labels)}")

        frame = frame.copy()
        frame["title"] = frame["title"].fillna("").astype(str)
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["label"] = frame["label"].astype(int)
        return frame


class WelfakeProvider(CorpusProvider):
    """The corpus every headline number in the paper is trained and tested on."""

    def __init__(self, path: Path | str | None = None, *, verify: bool = True) -> None:
        self._path = path
        self._verify = verify

    @property
    def metadata(self) -> CorpusMetadata:
        return CorpusMetadata(
            name="WELFake",
            description=(
                "72,134 articles merged from Kaggle, McIntire, Reuters and BuzzFeed "
                "Political, predominantly US political news from 2016-2017."
            ),
            source_url="https://zenodo.org/records/4561253",
            citation="Verma et al. (2021), IEEE Transactions on Computational Social Systems",
        )

    def load(self) -> pd.DataFrame:
        frame = load_welfake(self._path, verify=self._verify)
        return frame[list(CORPUS_COLUMNS)]


class PerezRosasProvider(CorpusProvider):
    """The COLING 2018 fake news datasets, used here as unseen test data only.

    Two sub-collections ship together and they are not the same kind of thing.
    FakeNewsAMT pairs a genuine article with a crowdworker's fabricated version
    of it, so the fake class is deliberately written to imitate real reporting.
    Celebrity takes its fake class from tabloid and gossip sites, which is closer
    to what WELFake contains. Reporting them separately matters: a model trained
    on WELFake should be expected to do better on the second than the first, and
    a single pooled number would hide that.

    Articles are stored one per file, title on the first line, body after a blank
    line. The label is in the directory name.
    """

    #: Filename prefixes to the domain they denote, per the dataset README.
    DOMAIN_PREFIXES = {
        "biz": "business",
        "edu": "education",
        "entmt": "entertainment",
        "polit": "politics",
        "sports": "sports",
        "tech": "technology",
    }

    #: Directory name to the label used throughout this project.
    LABEL_DIRECTORIES = {"legit": 0, "fake": 1}

    #: Sub-collection directory to the name used in reports.
    SUBSETS = {"fakeNewsDataset": "FakeNewsAMT", "celebrityDataset": "Celebrity"}

    #: A leading segment longer than this is a paragraph, not a headline. The
    #: 75th percentile headline is 76 characters, so the threshold is not close
    #: to anything it has to decide.
    MAX_HEADLINE_CHARS = 200

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else PEREZ_ROSAS_DIR

    @property
    def metadata(self) -> CorpusMetadata:
        return CorpusMetadata(
            name="Perez-Rosas",
            description=(
                "980 articles across seven domains: 480 FakeNewsAMT (crowdsourced "
                "fabrications paired with mainstream originals) and 500 Celebrity "
                "(tabloid and gossip reporting verified against GossipCop)."
            ),
            source_url="https://web.eecs.umich.edu/~mihalcea/downloads/fakeNewsDatasets.zip",
            citation="Perez-Rosas et al. (2018), COLING",
        )

    @property
    def group_columns(self) -> tuple[str, ...]:
        return ("subset", "domain")

    def load(self) -> pd.DataFrame:
        if not self._root.exists():
            raise FileNotFoundError(
                f"Perez-Rosas dataset not found at {self._root}. Download "
                f"{self.metadata.source_url} and unzip it into "
                f"{self._root.parent}, or set FAKENEWS_PEREZ_ROSAS_DIR."
            )

        records = [
            self._read_article(path, subset=subset, label=label)
            for directory, subset in self.SUBSETS.items()
            for folder, label in self.LABEL_DIRECTORIES.items()
            for path in sorted((self._root / directory / folder).glob("*.txt"))
        ]
        if not records:
            raise ValueError(f"No article files found under {self._root}")

        frame = pd.DataFrame.from_records(records)
        logger.info(
            "Loaded %d Perez-Rosas articles: %s",
            len(frame),
            frame.groupby("subset")["label"].value_counts().to_dict(),
        )
        return frame

    def _read_article(self, path: Path, *, subset: str, label: int) -> dict:
        """Split one file into title and body."""
        # A few files contain characters outside the declared encoding; replacing
        # them loses a glyph, whereas failing loses the article.
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        title, body = self._split_headline(content)

        return {
            "title": title.strip(),
            "text": body.strip(),
            "label": label,
            "subset": subset,
            "domain": self._domain_from_filename(path.name, subset=subset),
            "article_id": path.stem,
        }

    @classmethod
    def _split_headline(cls, content: str) -> tuple[str, str]:
        """Separate the headline from the body.

        The README describes one layout, the files use three: 929 articles
        separate headline and body with a blank line, 20 with a single newline,
        and 31 are a single unbroken paragraph. Some files in the first two
        groups open with a body paragraph rather than a headline, so the
        separator alone does not settle it and the leading segment is accepted
        as a headline only if it is short enough to be one.

        Nothing downstream is affected either way -- every consumer concatenates
        the two fields -- but a frame with a "title" column should hold titles.

        Returns:
            ``(title, body)``, either of which may be empty.
        """
        for separator in ("\n\n", "\n"):
            head, found, tail = content.partition(separator)
            if not found:
                continue
            if len(head) <= cls.MAX_HEADLINE_CHARS:
                return head.strip(), tail.strip()
            break

        return "", content

    def _domain_from_filename(self, filename: str, *, subset: str) -> str:
        """Recover the topical domain from the filename prefix.

        Only FakeNewsAMT encodes a domain; Celebrity is a single domain by
        construction.
        """
        if subset != "FakeNewsAMT":
            return "celebrity"

        prefix = filename.split(".")[0].rstrip("0123456789")
        return self.DOMAIN_PREFIXES.get(prefix, "unknown")
