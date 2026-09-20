# Reproducing the results

Every number in the paper comes from a script in `experiments/`. This document
lists them in dependency order, with the wall-clock cost measured on the
reference machine (an 8-thread Intel CPU, no GPU).

## 0. Environment

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[app,bench,dev]"
```

Python 3.11–3.13. TensorFlow does not publish wheels for 3.14, so 3.14 will not
work.

## 1. Data

Two files are downloaded, not vendored:

| File | Destination | Source |
| --- | --- | --- |
| `WELFake_Dataset.csv` | `data/raw/` | <https://zenodo.org/records/4561253> |
| `glove.6B.100d.txt` | `data/embeddings/` | <https://nlp.stanford.edu/projects/glove/> (`glove.6B.zip`) |
| `fakeNewsDatasets/` | `data/raw/` | <https://web.eecs.umich.edu/~mihalcea/downloads/fakeNewsDatasets.zip> |

The first two are required. The third is the Perez-Rosas (COLING 2018) corpus and
is needed only by `run_cross_dataset.py`; unzip it so that
`data/raw/fakeNewsDatasets/celebrityDataset/` exists. It is never trained on.

Override any location with `FAKENEWS_WELFAKE_CSV`, `FAKENEWS_GLOVE_TXT` or
`FAKENEWS_PEREZ_ROSAS_DIR`.

## 2. Build the preprocessing cache (once, ~30 min)

```bash
fakenews build-cache --jobs -1
```

Cleans the corpus, POS-tags it, and materialises every blocklist variant to
`data/cache/welfake_preprocessed.parquet`. Tagging is the expensive step and is
shared by every experiment below, so it is done once.

## 3. Experiments

Run `run_main.py` first: it saves the fitted models and the test-set prediction
arrays that three of the other scripts consume.

| Script | Produces | Measured cost |
| --- | --- | --- |
| `run_main.py` | Table 1 single-run columns, error counts, saved models | 16.1 min |
| `run_pos_ablation.py` | Table 3, Figure 1, gain shares | 4.9 min |
| `run_efficiency.py` | Table 4, hardware and threading record | 4.6 min |
| `run_svd_sweep.py` | Figure 3, §4.5 | 20.7 min |
| `run_leakage.py` | Figure 2, §4.4 | 30.7 min |
| `run_seeds.py` | Table 1 five-seed columns, Table 2 | ~65 min |
| `run_xgb_tuning.py` | Quantifies the untuned-baseline gap (§3.3, §5.4) | 50.8 min |
| `run_source_holdout.py` | Publisher-held-out generalisation probe | 11.3 min |
| `run_error_analysis.py` | Error categorisation, `error_sample.md` for manual reading | 1.3 min |
| `run_cross_dataset.py` | Transfer to the Perez-Rosas corpus, `cross_dataset` figure | 0.6 min |

The last three answer review points rather than producing an original table.
`run_source_holdout.py` and `run_cross_dataset.py` both test generalisation
beyond a random split of WELFake, and both report a large drop; read their
module docstrings before reading their numbers, because each has a design
caveat that changes what the number means. `run_cross_dataset.py` scores the
models saved by `run_main.py` and trains nothing.

`run_error_analysis.py` writes `artifacts/results/error_sample.md` for a human
to read. Record one category per article index in
`experiments/error_categories.json` and re-run it to fill in the manual table;
without that file the automated signal tables still populate.

Table 5 compares against published figures and is assembled by hand from the
cited papers; no script produces it, because no transformer was retrained here.
Figure 4 is a screenshot of the Flask app in `app/`.

`run_main.py` also writes `model_comparison.{pdf,png}`, a dot plot of all five
models. The paper presents that comparison as Table 1 instead, so the figure is
unused there — it is kept because it is useful when iterating.

```bash
python experiments/run_main.py
python experiments/run_pos_ablation.py
python experiments/run_svd_sweep.py
python experiments/run_leakage.py
python experiments/run_seeds.py
python experiments/run_efficiency.py
python experiments/run_source_holdout.py
python experiments/run_error_analysis.py
python experiments/run_cross_dataset.py
```

Each writes `artifacts/results/<name>.json` (machine-readable) and
`artifacts/results/<name>.md` (tables ready to paste into a paper). Figures land
in `artifacts/figures/` as both 300-DPI PNG and vector PDF.

`run_seeds.py` reuses any per-seed run already present in
`artifacts/results/`, so it can be interrupted and resumed. Pass `--force` to
re-run regardless.

## Determinism

`fakenews.seeding.set_global_seed` seeds Python, NumPy and TensorFlow, and every
stochastic component (splits, SVD, XGBoost, Keras initialisers) takes its seed
from `ExperimentConfig.seed`.

Two caveats apply to exact bitwise reproduction:

- TensorFlow's default CPU kernels are not deterministic across thread
  schedules. Call `fakenews.seeding.enable_deterministic_ops()` for bitwise
  reproducibility, at a measurable cost in speed.
- `PYTHONHASHSEED` must be fixed *before* the interpreter starts to be fully
  effective. `fakenews.seeding.hashseed_is_fixed()` reports whether it was.

In practice, run-to-run accuracy varies in the fourth decimal place. This is
exactly why `run_seeds.py` reports mean ± SD rather than a single figure.

## Hardware and threading

`run_efficiency.py` records the machine itself: exact CPU model and rated clock,
core and thread counts, RAM, OS build, every `*_NUM_THREADS` variable (including
the unset ones), the loaded BLAS and its thread count, and TensorFlow's op
parallelism. It lands in `efficiency.json` under `hardware` and `runtime`, so
nobody has to take the paper's word for what the numbers were measured on.

The reference machine is an 11th Gen Intel Core i7-1165G7 @ 2.80 GHz, 4 cores /
8 threads, 15.8 GB RAM, no GPU, on Windows 11, with OpenBLAS 0.3.30 (pthreads)
and no thread variables set.

One thing to be careful about when quoting latency. The default run measures
articles one at a time but leaves every core available to BLAS, OpenMP and
TensorFlow, so it is *sequential throughput*, not single-threaded execution. For
a genuine single-thread figure:

```bash
python experiments/run_efficiency.py --threads 1 --report-name efficiency_single_thread
```

That pins BLAS, OpenMP (which is what XGBoost's `n_jobs=-1` actually obeys) and
TensorFlow to one thread, and records the thread counts that applied while the
timer ran rather than the ones requested.

Measured on the reference machine, 1,000 articles per run:

| Configuration | Mean | Median | p95 |
| --- | --- | --- | --- |
| Unconstrained (cooled machine) | 364.79 ms | 261.32 ms | 717.66 ms |
| 1 thread, run 1 | 259.55 ms | 207.81 ms | 611.69 ms |
| 1 thread, run 2 | 233.64 ms | 204.55 ms | 346.24 ms |

One thread is *faster*. Four physical cores running eight BLAS threads plus
eight OpenMP threads is oversubscription, and the per-article work is far too
small to pay for the dispatch: a single SVD transform, 200 tree traversals and
one small forward pass. Mean sits well above median in every run because POS
tagging scales with article length and the corpus has a long tail.

These figures are laptop measurements and move with thermal state. An
unconstrained run taken immediately after another benchmark gave 393.02 ms
against the 364.79 ms above, so leave the machine idle for a few minutes before
timing anything that will be quoted.

## Package versions

The reference results were produced with:

| Package | Version |
| --- | --- |
| Python | 3.13.5 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.16.3 |
| tensorflow | 2.20.0 |
| xgboost | 3.1.2 |
| nltk | 3.9.2 |

`requirements-lock.txt` pins the full transitive set.
