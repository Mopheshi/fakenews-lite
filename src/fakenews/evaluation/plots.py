"""Publication figures.

Every figure is written twice: a 300-DPI raster for drafts and a vector PDF for
camera-ready submission. Figures are sized in inches for a two-column A4 layout,
so text lands at its intended point size rather than being scaled by the
typesetter.

Design constraints, all of which the paper's print context forces:

* Scores cluster between 0.95 and 1.00, so magnitudes are drawn as dots on a
  zoomed axis rather than bars from zero -- a truncated bar baseline would
  misrepresent the ratios.
* Series identity never rests on hue alone: each series carries a distinct
  marker shape, so the figures survive greyscale printing and colour-vision
  deficiency.
* Effect sizes have a meaningful zero, so those *are* drawn as bars from zero,
  with a diverging blue/red pair encoding the direction of the difference.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from fakenews.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Palette (validated for the white page surface; see docs/figures.md)
# --------------------------------------------------------------------------- #

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
SERIES_MARKERS = ("o", "s", "^")

DIVERGING_POSITIVE = "#2a78d6"  # higher in fake news
DIVERGING_NEGATIVE = "#d03b3b"  # higher in legitimate news

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"

#: Single-column and full-width figure widths for two-column A4.
COLUMN_WIDTH_IN = 3.45
FULL_WIDTH_IN = 7.16


def use_paper_style() -> None:
    """Apply the shared rcParams. Idempotent; safe to call per figure."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": AXIS_RULE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "text.color": INK_PRIMARY,
            # Keep text as text in the PDF so the typesetter can search it.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str, *, directory: Path | None = None) -> dict[str, Path]:
    """Write ``stem.pdf`` (vector) and ``stem.png`` (300 DPI)."""
    target = directory or FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)

    paths = {}
    for suffix in ("pdf", "png"):
        path = target / f"{stem}.{suffix}"
        fig.savefig(path)
        paths[suffix] = path

    plt.close(fig)
    logger.info("Wrote %s.pdf and %s.png to %s", stem, stem, target)
    return paths


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_model_comparison(
    scores: Mapping[str, Mapping[str, float]],
    *,
    metrics: Sequence[str] = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
    stem: str = "model_comparison",
    directory: Path | None = None,
    xlim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Dot plot of every metric for each model.

    Args:
        scores: ``{model_name: {metric_name: value}}``, at most three models.
        metrics: Metric order, top to bottom.
    """
    use_paper_style()

    model_names = list(scores)
    if len(model_names) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(model_names)} models exceeds the {len(SERIES_COLORS)}-slot validated palette; "
            "facet into small multiples instead of adding hues"
        )

    values = np.array([[scores[model][metric] for metric in metrics] for model in model_names])
    if xlim is None:
        low = float(values.min())
        xlim = (max(0.0, low - 0.012), 1.0)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN * 0.62, 2.55))
    y_positions = np.arange(len(metrics))[::-1]

    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major")
    ax.grid(False, axis="y")

    for index, model in enumerate(model_names):
        ax.scatter(
            values[index],
            y_positions,
            s=42,
            marker=SERIES_MARKERS[index],
            color=SERIES_COLORS[index],
            edgecolors="white",
            linewidths=0.8,
            label=model,
            zorder=3,
        )

    # Direct-label the best model only; the axis and the paper's table carry the rest.
    best_index = int(np.argmax(values.mean(axis=1)))
    for y, value in zip(y_positions, values[best_index], strict=True):
        ax.annotate(
            f"{value:.4f}",
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK_SECONDARY,
        )

    ax.set_yticks(y_positions, list(metrics))
    # Headroom on the right so the direct labels are not clipped.
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.10)
    ax.set_xlabel("Score")
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.spines["left"].set_visible(False)
    # Above the axes, not inside them: an in-plot legend collides with the
    # bottom row, where all three models sit within a hair of each other.
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.1),
        mode="expand",
        ncols=len(model_names),
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    return save_figure(fig, stem, directory=directory)


def plot_pos_effect_sizes(
    effects: Iterable[Mapping[str, float | str]],
    *,
    stem: str = "pos_effect_sizes",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Diverging bars of Cohen's d per POS category.

    Positive d means the category is more frequent in fake news. Effect size has
    a true zero, so bars from zero are the honest encoding here.
    """
    use_paper_style()

    records = sorted(effects, key=lambda record: float(record["cohens_d"]))
    labels = [str(record["feature"]).replace("pos_", "").title() for record in records]
    values = [float(record["cohens_d"]) for record in records]
    colors = [DIVERGING_POSITIVE if value > 0 else DIVERGING_NEGATIVE for value in values]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    y_positions = np.arange(len(labels))

    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major")
    ax.barh(y_positions, values, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color=AXIS_RULE, linewidth=0.8, zorder=4)

    for y, value in zip(y_positions, values, strict=True):
        offset = 4 if value >= 0 else -4
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, y),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
            color=INK_SECONDARY,
        )

    span = max(abs(min(values)), abs(max(values))) * 1.45
    ax.set_xlim(-span, span)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Cohen's $d$  (positive = higher in fake news)")
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, stem, directory=directory)


def plot_svd_sweep(
    components: Sequence[int],
    accuracies: Sequence[float],
    explained_variance: Sequence[float],
    *,
    chosen: int | None = None,
    stem: str = "svd_sweep",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Accuracy and retained variance against SVD component count.

    Two stacked panels sharing one x-axis rather than a dual-axis plot: the two
    quantities have unrelated scales, and overlaying them would invent a
    relationship the data does not contain.
    """
    use_paper_style()

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 3.0), sharex=True, height_ratios=[1.0, 0.75]
    )

    for axis in (top, bottom):
        axis.set_axisbelow(True)
        axis.grid(axis="y", which="major")

    top.plot(
        components,
        accuracies,
        color=SERIES_COLORS[0],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    top.set_ylabel("XGBoost accuracy")

    bottom.plot(
        components,
        explained_variance,
        color=SERIES_COLORS[1],
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    bottom.set_ylabel("Explained variance")
    bottom.set_xlabel("SVD components $k$")

    # Headroom so the annotation at the chosen k does not touch the panel edge.
    top.margins(y=0.16)
    bottom.margins(y=0.16)

    if chosen is not None and chosen in components:
        index = list(components).index(chosen)
        for axis, series in ((top, accuracies), (bottom, explained_variance)):
            axis.axvline(chosen, color=AXIS_RULE, linewidth=0.8, zorder=2)
            axis.annotate(
                f"{series[index]:.4f}" if axis is top else f"{series[index]:.3f}",
                xy=(chosen, series[index]),
                xytext=(7, -9),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
            )
        top.annotate(
            f"$k$ = {chosen}",
            xy=(chosen, top.get_ylim()[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=INK_MUTED,
        )

    fig.tight_layout()
    return save_figure(fig, stem, directory=directory)


def plot_leakage_deltas(
    scores: Mapping[str, Mapping[str, float]],
    *,
    stem: str = "source_leakage",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Dot plot of accuracy under each source-marker blocklist.

    Args:
        scores: ``{condition_name: {model_name: accuracy}}``, at most three conditions.
    """
    use_paper_style()

    conditions = list(scores)
    models = list(next(iter(scores.values())))
    values = np.array([[scores[condition][model] for model in models] for condition in conditions])

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN * 0.78, 2.25))
    y_positions = np.arange(len(models))[::-1]

    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major")

    for index, condition in enumerate(conditions):
        ax.scatter(
            values[index],
            y_positions,
            s=42,
            marker=SERIES_MARKERS[index],
            color=SERIES_COLORS[index],
            edgecolors="white",
            linewidths=0.8,
            label=condition,
            zorder=3,
        )

    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.35, 0.004)
    ax.set_xlim(low - margin, high + margin)
    ax.set_yticks(y_positions, models)
    ax.set_xlabel("Test accuracy")
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.spines["left"].set_visible(False)
    # Above the axes; an in-plot legend lands on the bottom row of markers.
    # Centred rather than expanded: these labels are long enough that stretching
    # them across the full axis width runs them into each other.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=len(conditions),
        handletextpad=0.3,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    return save_figure(fig, stem, directory=directory)
