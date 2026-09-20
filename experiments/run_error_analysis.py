"""What do the mistakes actually look like?

A false positive rate answers "how often" and says nothing about "on what". This
splits the reference run's errors into named groups and, for each group, asks
whether it is over-represented among the errors relative to the articles the
model gets right. That control is the point: roughly a tenth of all legitimate
articles are war reporting, so finding war reporting among the false positives
means nothing until it is compared against that base rate.

Two passes are reported. The automated pass applies documented surface rules to
every error, which gives coverage and a significance test. The manual pass
covers a deterministic sample read by hand, which gives the judgments no rule
captures -- satire, quoted misinformation and probable label errors in
particular. Manual labels live in ``experiments/error_categories.json`` so the
table can be regenerated without anyone re-reading the sample.

No model is trained here. The splits are reproducible from the seed and the
predictions are the ones saved by ``run_main.py``.

    python experiments/run_error_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from _corpus import DATELINE_RE, load_raw_aligned
from _report import configure_logging, markdown_table, write_report
from fakenews.config import DEFAULT_SEED, RESULTS_DIR, ExperimentConfig
from fakenews.data.cache import load_cached_corpus
from fakenews.evaluation.significance import two_proportion_test
from fakenews.pipeline import split_indices

logger = logging.getLogger(__name__)

#: Manual labels for the read sample, keyed by article index.
CATEGORIES_PATH = Path(__file__).with_name("error_categories.json")

#: Articles below this many characters carry too little text for either model to
#: work with; the TextCNN pads them to 500 tokens out of almost nothing.
SHORT_ARTICLE_CHARS = 1_000

#: Armed-conflict vocabulary. Wire copy about a war shares much of its lexicon
#: with the atrocity-focused fabrications in the fake class, which is the
#: mechanism this group is testing for.
CONFLICT_TERMS = frozenset(
    {
        "airstrike", "airstrikes", "insurgent", "insurgents", "militant", "militants",
        "militia", "jihadist", "jihadists", "massacre", "mortar", "offensive",
        "rebel", "rebels", "shelling", "troops", "warplanes", "ceasefire",
        "casualties", "battalion", "gunmen", "bombardment",
    }
)

#: Outlets and terms that signal deliberate satire rather than deception.
SATIRE_TERMS = frozenset(
    {"satire", "satirical", "onion", "borowitz", "duffelblog", "clickhole", "waterfordwhispers"}
)

#: First and second person pronouns. Dense use of these marks commentary and
#: blog-style writing rather than the third-person register of wire copy.
PERSONAL_PRONOUNS = frozenset({"i", "me", "my", "we", "our", "us", "you", "your"})

_WORD_RE = re.compile(r"[a-z']+")
_QUOTE_RE = re.compile(r"[\"“”]")


def _signals(title: str, text: str) -> dict[str, bool]:
    """Surface properties of one article, each either present or absent.

    Deliberately crude and deliberately documented: every rule here is one line
    a reader can disagree with, which is preferable to a learned classifier
    whose groupings cannot be argued with.
    """
    combined = f"{title} {text}"
    words = _WORD_RE.findall(combined.lower())
    n_words = len(words) or 1
    counts = Counter(words)

    letters = [character for character in title if character.isalpha()]
    upper_ratio = sum(character.isupper() for character in letters) / (len(letters) or 1)

    return {
        "very_short": len(combined.strip()) < SHORT_ARTICLE_CHARS,
        "wire_dateline": bool(DATELINE_RE.search(text)),
        "conflict_reporting": sum(counts[term] for term in CONFLICT_TERMS) >= 3,
        "satire_marker": any(counts[term] for term in SATIRE_TERMS),
        "shouting_title": len(letters) >= 10 and upper_ratio >= 0.5,
        "exclamation_heavy": combined.count("!") >= 3,
        "quotation_heavy": len(_QUOTE_RE.findall(combined)) >= 10,
        "personal_voice": sum(counts[word] for word in PERSONAL_PRONOUNS) / n_words >= 0.02,
    }


SIGNAL_LABELS = {
    "very_short": "Under 1,000 characters",
    "wire_dateline": "Carries a wire dateline",
    "conflict_reporting": "Armed-conflict vocabulary",
    "satire_marker": "Satire marker",
    "shouting_title": "Title mostly upper case",
    "exclamation_heavy": "Three or more exclamation marks",
    "quotation_heavy": "Ten or more quotation marks",
    "personal_voice": "Dense first/second person",
}


def _signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`_signals` row-wise and return the boolean matrix."""
    records = [
        _signals(title, text)
        for title, text in zip(frame["title"], frame["text"], strict=True)
    ]
    return pd.DataFrame(records, index=frame.index)


def _compare(
    errors: pd.DataFrame, controls: pd.DataFrame, *, error_label: str
) -> tuple[list[dict], dict]:
    """Rate of each signal among errors against its rate among correct predictions."""
    rows, payload = [], {}
    for signal, label in SIGNAL_LABELS.items():
        comparison = two_proportion_test(
            int(errors[signal].sum()),
            len(errors),
            int(controls[signal].sum()),
            len(controls),
            label_a=f"{error_label} errors",
            label_b=f"{error_label} correct",
        )
        payload[signal] = comparison.to_dict()
        rows.append(
            {
                "Property": label,
                "Among errors": f"{comparison.count_a} ({comparison.rate_a:.1%})",
                "Among correct": f"{comparison.count_b} ({comparison.rate_b:.1%})",
                "Difference": f"{comparison.difference:+.1%}",
                "95% CI": f"[{comparison.ci_lower:+.1%}, {comparison.ci_upper:+.1%}]",
                "p": f"{comparison.p_value:.3g}",
            }
        )
    return rows, payload


def _select_sample(
    false_positives: np.ndarray, false_negatives: np.ndarray, *, size: int, seed: int
) -> dict[str, np.ndarray]:
    """Draw a reproducible sample, allocated in proportion to the two error types."""
    rng = np.random.default_rng(seed)
    total = len(false_positives) + len(false_negatives)
    n_positive = min(len(false_positives), round(size * len(false_positives) / total))
    n_negative = min(len(false_negatives), size - n_positive)

    return {
        "false_positive": np.sort(rng.choice(false_positives, n_positive, replace=False)),
        "false_negative": np.sort(rng.choice(false_negatives, n_negative, replace=False)),
    }


def _write_sample_for_reading(
    path: Path, sample: dict[str, np.ndarray], raw: pd.DataFrame, proba: pd.Series
) -> None:
    """Dump the sampled articles in full so they can be read and categorised."""
    lines = [
        "# Error sample for manual categorisation",
        "",
        "Generated by `experiments/run_error_analysis.py`. Record one category per",
        "index in `experiments/error_categories.json`, then re-run the script.",
        "",
    ]
    for kind, indices in sample.items():
        lines.append(f"## {kind.replace('_', ' ').title()} ({len(indices)} articles)")
        lines.append("")
        for index in indices:
            row = raw.loc[index]
            lines.extend(
                [
                    f"### index {index} | true={row['label']} | P(fake)={proba[index]:.3f}",
                    "",
                    f"**Title:** {row['title'] or '(none)'}",
                    "",
                    "```",
                    str(row["text"])[:2_500].strip(),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s for manual reading", path.name)


def _manual_table(sample: dict[str, np.ndarray], categories: dict[str, str]) -> list[dict]:
    """Tabulate the hand-assigned categories, or report that none exist yet."""
    rows = []
    for kind, indices in sample.items():
        assigned = Counter(categories.get(str(index), "unlabelled") for index in indices)
        for category, count in assigned.most_common():
            rows.append(
                {
                    "Error type": kind.replace("_", " "),
                    "Category": category,
                    "Articles": count,
                    "Share of sample": count / len(indices) if len(indices) else 0.0,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default="ensemble_soft_voting")
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig(seed=args.seed)

    _, labels, _ = load_cached_corpus(blocklist_preset="none")
    raw = load_raw_aligned(labels)
    _, _, test_idx = split_indices(labels, config)

    predictions_path = RESULTS_DIR / f"run_seed{args.seed}_pos1_none.predictions.npz"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Expected {predictions_path}; run experiments/run_main.py first.")
    saved = np.load(predictions_path)

    # If this fails the split no longer matches the one the models were trained
    # on, and every article-to-prediction join below would be meaningless.
    if not np.array_equal(saved["y_test"], labels[test_idx]):
        raise RuntimeError(
            "Saved predictions do not correspond to the reconstructed split. Re-run "
            "experiments/run_main.py, or check that the seed matches."
        )

    predicted = saved[f"pred_{args.model}"]
    probability = saved[f"proba_{args.model}"]

    test = raw.loc[test_idx].copy()
    test["predicted"] = predicted
    test["probability"] = probability
    signals = _signal_frame(test)
    test = test.join(signals)

    is_legit = test["label"] == 0
    is_fake = test["label"] == 1
    flagged = test["predicted"] == 1

    false_positives = test[is_legit & flagged]
    legit_correct = test[is_legit & ~flagged]
    false_negatives = test[is_fake & ~flagged]
    fake_correct = test[is_fake & flagged]

    logger.info(
        "%s: %d false positives of %d legitimate, %d false negatives of %d fake",
        args.model,
        len(false_positives),
        int(is_legit.sum()),
        len(false_negatives),
        int(is_fake.sum()),
    )

    fp_rows, fp_payload = _compare(false_positives, legit_correct, error_label="legitimate")
    fn_rows, fn_payload = _compare(false_negatives, fake_correct, error_label="fake")

    sample = _select_sample(
        false_positives.index.to_numpy(),
        false_negatives.index.to_numpy(),
        size=args.sample_size,
        seed=args.seed,
    )
    _write_sample_for_reading(
        RESULTS_DIR / "error_sample.md", sample, raw, test["probability"]
    )

    categories = (
        json.loads(CATEGORIES_PATH.read_text(encoding="utf-8")) if CATEGORIES_PATH.exists() else {}
    )
    manual_rows = _manual_table(sample, categories)

    # The dateline signal is far stronger than any other, and a share of errors
    # is the wrong way to show it: 61% of legitimate articles carry a dateline,
    # so the interesting quantity is the error rate within each group.
    provenance_rows = []
    for name, group, error_mask in (
        ("Legitimate, wire dateline", test[is_legit & test["wire_dateline"]], flagged),
        ("Legitimate, no dateline", test[is_legit & ~test["wire_dateline"]], flagged),
        ("Fake, no dateline", test[is_fake & ~test["wire_dateline"]], ~flagged),
    ):
        errors = int(error_mask.loc[group.index].sum())
        provenance_rows.append(
            {
                "Group": name,
                "Articles": len(group),
                "Errors": errors,
                "Error rate": errors / len(group) if len(group) else 0.0,
            }
        )

    dateline_legit = test[is_legit & test["wire_dateline"]]
    plain_legit = test[is_legit & ~test["wire_dateline"]]
    provenance_test = two_proportion_test(
        int(flagged.loc[plain_legit.index].sum()),
        len(plain_legit),
        int(flagged.loc[dateline_legit.index].sum()),
        len(dateline_legit),
        label_a="legitimate without a dateline",
        label_b="legitimate with a dateline",
    )

    confidence_rows = [
        {
            "Group": name,
            "Articles": len(group),
            "Mean P(fake)": float(group["probability"].mean()),
            "Median P(fake)": float(group["probability"].median()),
            "Within 0.1 of the threshold": float(
                (group["probability"].sub(0.5).abs() < 0.1).mean()
            ),
        }
        for name, group in (
            ("False positives", false_positives),
            ("Correct legitimate", legit_correct),
            ("False negatives", false_negatives),
            ("Correct fake", fake_correct),
        )
    ]

    write_report(
        "error_analysis",
        payload={
            "seed": args.seed,
            "model": args.model,
            "counts": {
                "false_positives": len(false_positives),
                "legitimate_total": int(is_legit.sum()),
                "false_negatives": len(false_negatives),
                "fake_total": int(is_fake.sum()),
            },
            "signal_definitions": {
                "very_short": f"fewer than {SHORT_ARTICLE_CHARS} characters of title plus text",
                "wire_dateline": "raw text matches (Reuters)",
                "conflict_reporting": "three or more armed-conflict terms",
                "satire_marker": "any satire outlet or term",
                "shouting_title": "at least half the title's letters upper case",
                "exclamation_heavy": "three or more exclamation marks",
                "quotation_heavy": "ten or more quotation marks",
                "personal_voice": "first or second person pronouns at 2% of tokens or more",
            },
            "false_positive_signals": fp_payload,
            "false_negative_signals": fn_payload,
            "confidence": confidence_rows,
            "provenance": provenance_rows,
            "provenance_test": provenance_test.to_dict(),
            "sample": {kind: indices.tolist() for kind, indices in sample.items()},
            "manual_categories": categories,
            "n_manually_labelled": sum(
                1 for indices in sample.values() for index in indices if str(index) in categories
            ),
        },
        tables=[
            ("False positives against correctly classified legitimate articles", markdown_table(fp_rows)),
            ("False negatives against correctly classified fake articles", markdown_table(fn_rows)),
            ("Manual categorisation of the read sample", markdown_table(manual_rows)),
            ("Error rate by article provenance", markdown_table(provenance_rows)),
            ("Model score by group", markdown_table(confidence_rows)),
        ],
    )

    print()
    print(markdown_table(fp_rows))
    print()
    print(markdown_table(provenance_rows))
    print()
    print(
        f"False positive rate, no dateline vs dateline: "
        f"{provenance_test.rate_a:.2%} vs {provenance_test.rate_b:.2%} "
        f"(difference {provenance_test.difference:+.2%}, p = {provenance_test.p_value:.3g})"
    )
    print()
    print(markdown_table(manual_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
