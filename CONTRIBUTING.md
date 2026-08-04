# Contributing

## Setting up

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[app,bench,dev]"
python -c "from fakenews.nltk_resources import ensure_nltk_resources; ensure_nltk_resources()"
```

Python 3.11–3.13. TensorFlow publishes no 3.14 wheel.

## Before opening a pull request

```bash
ruff check .
pytest
```

The unit tests deliberately require neither the dataset nor the GloVe file, so
they run in seconds and in CI. Anything that needs the corpus belongs in
`experiments/`, not in `tests/`.

## Where things go

| Adding... | Put it in |
| --- | --- |
| A reusable component | `src/fakenews/` |
| A result reported somewhere | `experiments/run_*.py` |
| A hyperparameter or path | `src/fakenews/config.py` |
| A metric or statistical test | `src/fakenews/evaluation/` |

## Conventions

**Experiments write both formats.** Use `experiments/_report.py::write_report`
so every run leaves a JSON record and a Markdown table. Numbers that appear in
prose must be traceable to a file in `artifacts/results/`.

**Randomness flows from `ExperimentConfig.seed`.** No bare `random_state=42` and
no unseeded `np.random` calls — a new stochastic component takes its seed from
the config like everything else.

**Metrics are reported per class.** The positive class in this dataset is
ambiguous enough that aggregate-only reporting has already produced one
published error. If you add a metric that depends on which class is positive,
report it per class.

**Claims need measurements.** A performance or efficiency claim in the README,
the docs or a docstring should point at an experiment that produces it. If it
cannot be measured here, attribute it to its source and say it was not measured
here.

## Figures

`src/fakenews/evaluation/plots.py` holds the shared style and writes every
figure as both a 300-DPI PNG and a vector PDF. New figures should go through
`save_figure` rather than calling `savefig` directly, and should stay legible in
greyscale — series identity carries a marker shape, never colour alone.
