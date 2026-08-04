# WELFake label semantics

## The inconsistency

The WELFake record documents its `label` column as:

> Label 0 = fake and 1 = real

and, in the same description, reports the corpus as containing **35,028 real**
and **37,106 fake** news articles.

The CSV contains **35,028 rows with `label == 0`** and **37,106 rows with
`label == 1`**. Those two statements cannot both be true. Matching the counts to
the classes fixes the mapping:

| Label | Count | Class implied by the counts |
| --- | --- | --- |
| 0 | 35,028 | Real / legitimate |
| 1 | 37,106 | Fake |

This project therefore treats **0 as legitimate and 1 as fake**, which is the
count-consistent reading. `fakenews.config.LABEL_NAMES` is the single place that
encodes it.

No manual inspection of article text is needed to reach this conclusion — the
documentation contradicts itself arithmetically, and the counts settle it.

## What this does and does not affect

It is worth being precise about the consequences, because it is easy to overstate
them.

**Unaffected.** Accuracy, macro-F1, ROC-AUC and the confusion matrix as a whole
are invariant under renaming the classes. Every study trains and evaluates
against the same column, so a study that adopted the documented mapping obtains
*identical* aggregate numbers. Swapping the names cannot change how many
articles a model classifies correctly.

**Affected.** Anything reported per class:

- which class "precision" and "recall" belong to when a single positive class is
  quoted;
- the false positive rate, and therefore any discussion of legitimate articles
  being wrongly flagged;
- per-class rows of a classification report.

So the practical consequence is a **reporting** concern, not a validity concern.
Prior results are not invalidated by this; their per-class labels may simply be
transposed.

## Verification

```python
import pandas as pd

df = pd.read_csv("data/raw/WELFake_Dataset.csv")
print(df["label"].value_counts().sort_index())
# 0    35028
# 1    37106
```

`fakenews.data.loader.load_welfake` checks this distribution on load and warns if
it differs, which catches a modified or truncated copy of the dataset before it
reaches a model.
