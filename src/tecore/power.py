"""
Power and sample size utilities for online experiments.

Focus:
- binary metrics (conversion rate, retention flags, churn flags);
- continuous metrics (ARPU, revenue per user, etc.).

The formulas are simple normal-approximation formulas for
two-sample tests with equal group sizes. They are good enough
for most practical B2C experiments at typical sample sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm


TailType = Literal["two-sided", "one-sided"]


@dataclass
class SampleSizeResult:
    """
    Container for sample size calculations.

    Attributes
    ----------
    n_per_group : int
        Required number of observations in each group.
    n_total : int
        Total number of observations across both groups.
    alpha : float
        Significance level used in the calculation.
    power : float
        Target statistical power of the test.
    effect : float
        Minimal detectable effect (absolute difference).
    metric_type : str
        "proportion" or "mean".
    """
    n_per_group: int
    n_total: int
    alpha: float
    power: float
    effect: float
    metric_type: str


def _z_alpha(alpha: float, tail: TailType) -> float:
    """Return critical z-value for a given alpha and tail type."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if tail == "two-sided":
        return norm.ppf(1 - alpha / 2.0)
    elif tail == "one-sided":
        return norm.ppf(1 - alpha)
    else:
        raise ValueError("tail must be 'two-sided' or 'one-sided'.")


def _z_beta(power: float) -> float:
    """Return z-value corresponding to the desired power (1 - beta)."""
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")
    return norm.ppf(power)

# Proportions (conversion rate, retention, churn, etc.)

def sample_size_proportions(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
) -> SampleSizeResult:
    """
    Compute required sample size per group for a two-sample test on proportions.

    This uses a normal approximation for the difference of two proportions
    with **equal group sizes**.

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate in the control group (0 < p < 1).
    mde : float
        Minimal detectable effect as an **absolute difference**
        in conversion rate (e.g. 0.02 = +2 percentage points).
        The treatment rate is assumed to be p_baseline + mde.
    alpha : float, optional
        Significance level of the test, by default 0.05.
    power : float, optional
        Desired power (1 - beta), by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    SampleSizeResult
        Required sample size per group and related parameters.
    """
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1).")
    if mde == 0:
        raise ValueError("mde must be non-zero.")
    p_treatment = p_baseline + mde

    if not 0 < p_treatment < 1:
        raise ValueError(
            "p_baseline + mde must be in (0, 1). "
            "Check that your minimal detectable effect is realistic."
        )

    delta = abs(mde)
    z_a = _z_alpha(alpha, tail)
    z_b = _z_beta(power)

    # Pooled proportion under alternative (simple approximation).
    p_bar = (p_baseline + p_treatment) / 2.0
    var_per_group = p_bar * (1 - p_bar)

    # For two-sample difference in proportions with equal n:
    # n_per_group = ( (z_a + z_b)^2 * 2 * var ) / delta^2
    numerator = (z_a + z_b) ** 2 * 2.0 * var_per_group
    n_per_group = math.ceil(numerator / (delta ** 2))
    n_total = 2 * n_per_group

    return SampleSizeResult(
        n_per_group=n_per_group,
        n_total=n_total,
        alpha=alpha,
        power=power,
        effect=delta,
        metric_type="proportion",
    )


def power_proportions(
    p_baseline: float,
    mde: float,
    n_per_group: int,
    alpha: float = 0.05,
    tail: TailType = "two-sided",
) -> float:
    """
    Compute achieved power for a two-sample test on proportions.

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate in the control group (0 < p < 1).
    mde : float
        Effect size as an **absolute difference** in conversion rate.
        Treatment rate is p_baseline + mde.
    n_per_group : int
        Sample size per group.
    alpha : float, optional
        Significance level, by default 0.05.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    float
        Approximate statistical power (probability of rejecting H0
        when the true difference equals `mde`).
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1).")
    if mde == 0:
        raise ValueError("mde must be non-zero.")

    p_treatment = p_baseline + mde
    if not 0 < p_treatment < 1:
        raise ValueError("p_baseline + mde must be in (0, 1).")

    delta = abs(mde)
    z_a = _z_alpha(alpha, tail)

    # Pooled proportion under alternative.
    p_bar = (p_baseline + p_treatment) / 2.0
    se = math.sqrt(2.0 * p_bar * (1 - p_bar) / n_per_group)

    # Non-centrality parameter under the alternative.
    if se == 0:
        return 0.0
    z_effect = delta / se

    if tail == "two-sided":
        # Power = P(|Z| > z_a | Z ~ N(z_effect, 1))
        # = P(Z > z_a - z_effect) + P(Z < -z_a - z_effect)
        upper = 1 - norm.cdf(z_a - z_effect)
        lower = norm.cdf(-z_a - z_effect)
        return upper + lower
    else:  # one-sided
        # Power = P(Z > z_a | Z ~ N(z_effect, 1))
        return 1 - norm.cdf(z_a - z_effect)

# Means (ARPU, revenue per user, etc.)

def sample_size_means(
    std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
) -> SampleSizeResult:
    """
    Compute required sample size per group for a two-sample test on means.

    This assumes:
    - equal group sizes,
    - common standard deviation in both groups,
    - normal approximation (or large samples).

    Parameters
    ----------
    std : float
        Estimated standard deviation of the metric (same units as the metric).
    mde : float
        Minimal detectable effect as an **absolute difference** in means
        (e.g. ARPU increase of 0.15 units).
    alpha : float, optional
        Significance level of the test, by default 0.05.
    power : float, optional
        Desired power (1 - beta), by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    SampleSizeResult
        Required sample size per group and related parameters.
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    if mde == 0:
        raise ValueError("mde must be non-zero.")

    delta = abs(mde)
    z_a = _z_alpha(alpha, tail)
    z_b = _z_beta(power)

    # For two-sample difference in means with equal n and common std:
    # n_per_group = (2 * std^2 * (z_a + z_b)^2) / delta^2
    numerator = 2.0 * (std ** 2) * (z_a + z_b) ** 2
    n_per_group = math.ceil(numerator / (delta ** 2))
    n_total = 2 * n_per_group

    return SampleSizeResult(
        n_per_group=n_per_group,
        n_total=n_total,
        alpha=alpha,
        power=power,
        effect=delta,
        metric_type="mean",
    )


def power_means(
    std: float,
    mde: float,
    n_per_group: int,
    alpha: float = 0.05,
    tail: TailType = "two-sided",
) -> float:
    """
    Compute achieved power for a two-sample test on means.

    Parameters
    ----------
    std : float
        Common standard deviation of the metric in both groups.
    mde : float
        Effect size as an absolute difference in means.
    n_per_group : int
        Sample size per group.
    alpha : float, optional
        Significance level, by default 0.05.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    float
        Approximate statistical power of the test.
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")
    if mde == 0:
        raise ValueError("mde must be non-zero.")

    delta = abs(mde)
    z_a = _z_alpha(alpha, tail)

    # Standard error of the difference in means with equal n:
    # se = sqrt(2 * std^2 / n)
    se = math.sqrt(2.0 * (std ** 2) / n_per_group)
    if se == 0:
        return 0.0

    z_effect = delta / se

    if tail == "two-sided":
        upper = 1 - norm.cdf(z_a - z_effect)
        lower = norm.cdf(-z_a - z_effect)
        return upper + lower
    else:
        return 1 - norm.cdf(z_a - z_effect)
