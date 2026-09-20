"""Statistical tests for comparing classifiers on a shared test set.

A 0.2-point accuracy gap on one split is not evidence on its own. These
functions supply the paired test and interval estimates that turn "the ensemble
scores higher" into a claim that can be defended or withdrawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class McNemarResult:
    """Outcome of a paired McNemar test between two classifiers."""

    model_a: str
    model_b: str
    #: A correct, B wrong.
    n_a_only: int
    #: B correct, A wrong.
    n_b_only: int
    statistic: float
    p_value: float
    exact: bool

    @property
    def n_discordant(self) -> int:
        return self.n_a_only + self.n_b_only

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "a_correct_b_wrong": self.n_a_only,
            "b_correct_a_wrong": self.n_b_only,
            "n_discordant": self.n_discordant,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "test": "exact binomial" if self.exact else "chi-square with continuity correction",
        }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    model_a: str = "A",
    model_b: str = "B",
    exact_threshold: int = 25,
) -> McNemarResult:
    """Paired test of whether two classifiers differ on the same test set.

    Only the discordant pairs carry information. The exact binomial form is used
    when they are few enough that the chi-square approximation is unreliable.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true

    n_a_only = int(np.sum(correct_a & ~correct_b))
    n_b_only = int(np.sum(~correct_a & correct_b))
    n_discordant = n_a_only + n_b_only

    if n_discordant == 0:
        return McNemarResult(model_a, model_b, 0, 0, 0.0, 1.0, exact=True)

    if n_discordant < exact_threshold:
        p_value = float(stats.binomtest(n_a_only, n_discordant, 0.5).pvalue)
        statistic = float(min(n_a_only, n_b_only))
        return McNemarResult(model_a, model_b, n_a_only, n_b_only, statistic, p_value, exact=True)

    # Edwards' continuity correction.
    statistic = (abs(n_a_only - n_b_only) - 1) ** 2 / n_discordant
    p_value = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(model_a, model_b, n_a_only, n_b_only, float(statistic), p_value, exact=False)


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for accuracy.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    y_true = np.asarray(y_true)
    correct = (np.asarray(y_pred) == y_true).astype(np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
    accuracies = correct[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(accuracies, [alpha, 1.0 - alpha])
    return float(correct.mean()), float(lower), float(upper)


def paired_difference_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the accuracy difference (A - B) on paired predictions."""
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(y_pred_a) == y_true).astype(np.float64)
    correct_b = (np.asarray(y_pred_b) == y_true).astype(np.float64)
    difference = correct_a - correct_b

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(n_resamples, len(difference)))
    differences = difference[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    return float(difference.mean()), float(lower), float(upper)


@dataclass
class EffectSize:
    """Class-conditional comparison of a single continuous feature."""

    feature: str
    mean_legit: float
    mean_fake: float
    std_legit: float
    std_fake: float
    cohens_d: float
    t_statistic: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "mean_legit": self.mean_legit,
            "mean_fake": self.mean_fake,
            "std_legit": self.std_legit,
            "std_fake": self.std_fake,
            "cohens_d": self.cohens_d,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
        }


def class_conditional_effect(values: np.ndarray, labels: np.ndarray, *, feature: str) -> EffectSize:
    """Welch t-test and Cohen's d for one feature between the two classes.

    With ~72,000 documents almost any difference reaches significance, so the
    effect size is the number that matters; the p-value is reported alongside it
    for completeness rather than as the headline.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)

    legit = values[labels == 0]
    fake = values[labels == 1]

    t_statistic, p_value = stats.ttest_ind(fake, legit, equal_var=False)

    # Pooled SD for Cohen's d.
    n_legit, n_fake = len(legit), len(fake)
    pooled_variance = (
        (n_legit - 1) * legit.var(ddof=1) + (n_fake - 1) * fake.var(ddof=1)
    ) / (n_legit + n_fake - 2)
    pooled_sd = float(np.sqrt(pooled_variance))
    cohens_d = float((fake.mean() - legit.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    return EffectSize(
        feature=feature,
        mean_legit=float(legit.mean()),
        mean_fake=float(fake.mean()),
        std_legit=float(legit.std(ddof=1)),
        std_fake=float(fake.std(ddof=1)),
        cohens_d=cohens_d,
        t_statistic=float(t_statistic),
        p_value=float(p_value),
    )


@dataclass
class ProportionComparison:
    """Unpaired comparison of one rate measured on two independent samples."""

    label_a: str
    label_b: str
    count_a: int
    total_a: int
    count_b: int
    total_b: int
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float

    @property
    def rate_a(self) -> float:
        return self.count_a / self.total_a if self.total_a else 0.0

    @property
    def rate_b(self) -> float:
        return self.count_b / self.total_b if self.total_b else 0.0

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "count_a": self.count_a,
            "total_a": self.total_a,
            "rate_a": self.rate_a,
            "count_b": self.count_b,
            "total_b": self.total_b,
            "rate_b": self.rate_b,
            "difference": self.difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "test": "two-sided Fisher exact, Newcombe interval on the difference",
        }


def two_proportion_test(
    count_a: int,
    total_a: int,
    count_b: int,
    total_b: int,
    *,
    label_a: str = "A",
    label_b: str = "B",
    confidence: float = 0.95,
) -> ProportionComparison:
    """Compare a rate across two *independent* samples.

    McNemar does not apply here: the publisher-holdout probe measures the same
    model's false positive rate on two disjoint sets of articles, so there are no
    matched pairs. The interval is Newcombe's hybrid score method, which stays
    well behaved when a rate approaches zero -- the normal approximation does
    not, and these rates are a few percent.

    Returns:
        A :class:`ProportionComparison` for ``A - B``.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError("Both samples must be non-empty")

    table = [[count_a, total_a - count_a], [count_b, total_b - count_b]]
    p_value = float(stats.fisher_exact(table, alternative="two-sided")[1])

    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    lower_a, upper_a = _wilson_interval(count_a, total_a, z)
    lower_b, upper_b = _wilson_interval(count_b, total_b, z)

    rate_a, rate_b = count_a / total_a, count_b / total_b
    difference = rate_a - rate_b

    # Newcombe: propagate each Wilson bound into the bound of the difference
    # that it can actually move.
    ci_lower = difference - np.sqrt((rate_a - lower_a) ** 2 + (upper_b - rate_b) ** 2)
    ci_upper = difference + np.sqrt((upper_a - rate_a) ** 2 + (rate_b - lower_b) ** 2)

    return ProportionComparison(
        label_a=label_a,
        label_b=label_b,
        count_a=count_a,
        total_a=total_a,
        count_b=count_b,
        total_b=total_b,
        difference=difference,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
    )


def proportion_interval(
    count: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a single rate.

    Used where a number is a count over a denominator -- accuracy on a small
    corpus, a false positive rate -- and the normal approximation would run off
    the end of [0, 1].

    Returns:
        ``(lower, upper)``.
    """
    if total <= 0:
        raise ValueError("Sample must be non-empty")

    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    return _wilson_interval(count, total, z)


def _wilson_interval(count: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    rate = count / total
    denominator = 1.0 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))
