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

Override either location with `FAKENEWS_WELFAKE_CSV` / `FAKENEWS_GLOVE_TXT`.

## 2. Build the preprocessing cache (once, ~20 min)

```bash
fakenews build-cache --jobs -1
```

Cleans the corpus, POS-tags it, and materialises every blocklist variant to
`data/cache/welfake_preprocessed.parquet`. Tagging is the expensive step and is
shared by every experiment below, so it is done once.

## 3. Experiments

Run `run_main.py` first: it saves the fitted models and the test-set prediction
arrays that three of the other scripts consume.

| Script | Produces | Approx. cost |
| --- | --- | --- |
| `run_main.py` | Tables I–IV, Figure 1, saved models | ~20 min |
| `run_pos_ablation.py` | POS effect sizes, ablation, gain shares, Figure 2 | ~8 min |
| `run_svd_sweep.py` | Component sweep, Figure 3 | ~25 min |
| `run_leakage.py` | Source-marker removal deltas, Figure 4 | ~45 min |
| `run_seeds.py` | Mean ± SD over 5 seeds, McNemar tests | ~80 min |
| `run_efficiency.py` | Parameter accounting, latency, memory, disk | ~5 min |
| `run_xgb_tuning.py` | Fixed-versus-tuned XGBoost gap | ~60 min |

```bash
python experiments/run_main.py
python experiments/run_pos_ablation.py
python experiments/run_svd_sweep.py
python experiments/run_leakage.py
python experiments/run_seeds.py
python experiments/run_efficiency.py
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
