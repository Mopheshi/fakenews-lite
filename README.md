# fakenews-lite

Fake news detection with an **XGBoost + TextCNN soft-voting ensemble** over
TF-IDF, part-of-speech ratios and frozen GloVe embeddings — trained and served
on a CPU.

This is the reference implementation for *Lightweight Ensemble Fake News
Detection on Commodity CPUs: A Measured Comparison of Cost and Accuracy*.

The question the paper asks is how much accuracy is lost when a GPU is not
available — and the point of this repository is that the answer comes with the
evidence that usually goes missing: an ablation showing whether the POS features
earn their place, a leakage test showing how much of the score is publisher
memorisation, a measured cost profile on named hardware instead of a quoted
ratio, and paired significance tests instead of a decimal-place comparison.

## What is here

```
src/fakenews/          Library: data, features, models, evaluation, pipeline, CLI
experiments/           One script per table and figure in the paper
app/                   Flask demo (paste an article or a URL)
tests/                 Unit tests
docs/                  Label semantics, reproduction guide, figure conventions
notebooks/             Guided walkthrough of the library
```

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[app,bench,dev]"
```

Python 3.11–3.13. TensorFlow publishes no 3.14 wheel, so 3.14 will not work.

## Data

Neither file is vendored; both are freely downloadable.

| File | Put it in | Source |
| --- | --- | --- |
| `WELFake_Dataset.csv` | `data/raw/` | <https://zenodo.org/records/4561253> |
| `glove.6B.100d.txt` | `data/embeddings/` | <https://nlp.stanford.edu/projects/glove/> |

Override with `FAKENEWS_WELFAKE_CSV` and `FAKENEWS_GLOVE_TXT`.

## Quick start

```bash
fakenews build-cache              # clean + POS-tag once (~30 min, cached thereafter)
python experiments/run_main.py    # train, evaluate, save models (~16 min)
python app/app.py                 # serve the demo at http://127.0.0.1:5000
```

Classify a single article without the web UI:

```bash
fakenews predict --model artifacts/models/seed5 --title "..." --text "..."
```

## Results

Mean ± SD over five seeds on the WELFake test split (14,425 articles), CPU only:

| Model | Accuracy | F1 (fake) | ROC-AUC |
| --- | --- | --- | --- |
| XGBoost | 0.9582 ± 0.0006 | 0.9596 ± 0.0006 | 0.9924 ± 0.0002 |
| TextCNN | 0.9746 ± 0.0031 | 0.9754 ± 0.0031 | 0.9968 ± 0.0008 |
| **Ensemble (soft voting)** | **0.9781 ± 0.0020** | 0.9788 ± 0.0020 | 0.9971 ± 0.0005 |
| Ensemble (stacking) | 0.9785 ± 0.0018 | 0.9791 ± 0.0017 | 0.9972 ± 0.0005 |

The ensemble's margin over the TextCNN is small but real: +0.21 points, 95% CI
[0.03, 0.40], McNemar p = 0.022. The three ensemble variants are statistically
indistinguishable from one another (p ≥ 0.80), so soft voting is used for
simplicity.

**Cost**, on a 4-core / 8-thread CPU with no GPU: 14.9 minutes to train
end-to-end, 256 ms mean latency per article (p95 307 ms), 1.2 GB peak RSS,
62 MB on disk. 256,432 learned parameters — 429× fewer than BERT-base, or 21×
smaller counting the frozen GloVe table.

That places the ensemble **1.06 points below the strongest reported BERT result**
on this dataset (98.87%) and 1.91 below GPT-2 (99.72%), at roughly one
four-hundredth of the learned parameters. This is a cost–accuracy trade, not a
win: whether a percentage point is worth a GPU depends on the deployment. The
contribution is making the trade measurable, not deciding it.

No speed-up ratio against a transformer is claimed anywhere here. None was
retrained under this protocol, and published inference times for this dataset are
GPU forward passes over truncated input, which are not comparable to an
end-to-end CPU measurement over full articles.

**Two findings you should know before you build on this:**

- **The POS features do not improve accuracy.** Adverb and pronoun ratios do
  separate the classes (Cohen's *d* ≈ 0.5) and XGBoost ranks adverbs 7th of its
  306 features, but adding all six to the SVD block is worth +0.15 points, which
  McNemar cannot distinguish from chance (p = 0.19), and nothing at all at
  ensemble level. The commonly repeated claim that fake news uses *more
  adjectives and fewer nouns* is not supported here — both effects run the other
  way and are negligible.
- **The score is not publisher memorisation.** Removing 15 wire-service markers
  (`reuters`, `ap`, `getty`, …) costs 0.51 points; removing 43 outlet names as
  well costs 0.82. Accuracy stays above 0.97 with every publisher name stripped.

## A note on the WELFake labels

The dataset's own description says `Label 0 = fake and 1 = real`, and in the same
paragraph reports 35,028 real and 37,106 fake articles. The CSV holds 35,028
zeros and 37,106 ones, so the counts settle it: **0 is legitimate, 1 is fake**.

This is worth stating precisely, because it is easy to overstate. Accuracy,
macro-F1 and ROC-AUC are invariant to what class 0 is called, so no published
aggregate result is invalidated by it. What it changes is which class a quoted
precision, recall or false positive rate belongs to. Full argument:
[`docs/label-semantics.md`](docs/label-semantics.md).

`fakenews.config.LABEL_NAMES` is the single place this is encoded, and
`load_welfake` warns if the distribution it sees disagrees.

## Reproducing every result

See [`docs/reproducing.md`](docs/reproducing.md). In dependency order:

| Script | Question it answers |
| --- | --- |
| `run_main.py` | How do the models and the ensemble score? |
| `run_pos_ablation.py` | Do the six POS features actually contribute anything? |
| `run_leakage.py` | How much of the score survives removing publisher names? |
| `run_seeds.py` | Is the ensemble's margin distinguishable from noise? |
| `run_efficiency.py` | What does "lightweight" cost, measured on named hardware? |
| `run_svd_sweep.py` | Why 300 SVD components? |
| `run_xgb_tuning.py` | How much does the untuned baseline give away? |
| `run_source_holdout.py` | Does it still work on a publisher held out of training? |
| `run_error_analysis.py` | What do the mistakes actually look like? |
| `run_cross_dataset.py` | Does it transfer to a corpus collected by other people? |

Each writes `artifacts/results/<name>.json` and a Markdown table beside it;
figures go to `artifacts/figures/` as 300-DPI PNG and vector PDF.

## Design notes

**Preprocessing is cached.** POS-tagging 72,134 articles dominates the runtime
and is identical across every experiment, so it runs once into a Parquet file
that all the experiments share. The blocklist text variants are materialised in
the same pass.

**Seeds flow from one place.** `ExperimentConfig.seed` reaches the split, the
SVD, XGBoost and every Keras initialiser. TensorFlow's default CPU kernels are
still not deterministic across thread schedules, so single-run accuracy varies in
the third decimal place (TextCNN SD 0.0031). That is why results are reported as
mean ± SD over five seeds rather than as a single figure.

**Metrics are reported per class.** Because the positive class is ambiguous in
this dataset, `ClassificationResult` carries per-class precision/recall/F1 and a
`false_positive_rate` computed from the confusion matrix — not as `1 - recall`,
which measures a different class.

**Parameter counts are not collapsed into one number.** A boosted forest has no
weight matrix, and the TextCNN's embedding table is frozen GloVe that nobody
trained here. The efficiency report separates learned weights, frozen weights
and stored tree values, so a comparison against a transformer's parameter count
can be read for what it is.

## Testing

```bash
pytest          # unit tests, no dataset required
ruff check .
```

To run the whole CI pipeline locally — install, NLTK resources, lint, tests —
exactly as `.github/workflows/ci.yml` does:

```bash
./scripts/ci-local.sh                  # macOS / Linux / Git Bash
pwsh ./scripts/ci-local.ps1            # PowerShell
```

Add `--skip-install` (or `-SkipInstall`) when the environment is already set up.
This exercises one interpreter; CI covers 3.11, 3.12 and 3.13.

## Citing

If you use this code, cite the software via [`CITATION.cff`](CITATION.cff) —
GitHub renders a "Cite this repository" button from it.

The accompanying paper, *Lightweight Ensemble Fake News Detection on Commodity
CPUs: A Measured Comparison of Cost and Accuracy*, is under review. This section
will be updated with the publication reference on acceptance.

## Licence

MIT — see [LICENSE](LICENSE).

The WELFake dataset and GloVe embeddings carry their own licences and are not
redistributed here.
