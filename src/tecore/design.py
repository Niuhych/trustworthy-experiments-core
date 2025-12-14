"""
Utilities for high-level experiment design.

This module builds on top of `tecore.power` and provides:

- simple helpers to design experiments for:
    * binary metrics (conversion rate, retention, churn flags),
    * continuous metrics (ARPU, revenue per user, etc.);
- functions to compute:
    * required sample size per group for a given MDE,
    * minimal detectable effect (MDE) for a fixed sample size.

The goal is to give analysts an easy way to reason about
trade-offs between sample size, power and effect size,
without going deep into statistical formulas every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .power import (
    TailType,
    SampleSizeResult,
    sample_size_proportions,
    sample_size_means,
    power_proportions,
    power_means,
)


MetricType = Literal["proportion", "mean"]


@dataclass
class ExperimentDesign:
    """
    High-level description of an experiment design.

    Attributes
    ----------
    metric_type : {"proportion", "mean"}
        Type of the primary metric.
    baseline : float
        Baseline value of the metric:
        - for proportions: baseline conversion rate (0..1),
        - for means: estimated standard deviation of the metric.
    mde : float
        Minimal detectable effect (absolute difference in metric units).
    alpha : float
        Significance level of the test.
    power : float
        Target statistical power (1 - beta).
    tail : {"two-sided", "one-sided"}
        Type of the statistical test.
    n_per_group : int
        Required or used sample size per group.
    n_total : int
        Total sample size across both groups.
    """

    metric_type: MetricType
    baseline: float
    mde: float
    alpha: float
    power: float
    tail: TailType
    n_per_group: int
    n_total: int

    def as_dict(self) -> dict:
        """Return the design as a plain dictionary (convenient for logging/JSON)."""
        return {
            "metric_type": self.metric_type,
            "baseline": self.baseline,
            "mde": self.mde,
            "alpha": self.alpha,
            "power": self.power,
            "tail": self.tail,
            "n_per_group": self.n_per_group,
            "n_total": self.n_total,
        }


# Forward design: choose MDE -> get required sample size

def design_proportion_experiment(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
) -> ExperimentDesign:
    """
    Design an experiment for a binary metric (e.g. conversion rate).

    This is a thin wrapper around `sample_size_proportions` that returns
    a high-level `ExperimentDesign` object.

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate in the control group (0 < p < 1).
    mde : float
        Minimal detectable effect as an absolute difference in rate
        (e.g. 0.01 = +1 percentage point).
    alpha : float, optional
        Significance level, by default 0.05.
    power : float, optional
        Desired power (1 - beta), by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of statistical test, by default "two-sided".

    Returns
    -------
    ExperimentDesign
        High-level experiment design object.
    """
    result: SampleSizeResult = sample_size_proportions(
        p_baseline=p_baseline,
        mde=mde,
        alpha=alpha,
        power=power,
        tail=tail,
    )

    return ExperimentDesign(
        metric_type="proportion",
        baseline=p_baseline,
        mde=result.effect,
        alpha=alpha,
        power=power,
        tail=tail,
        n_per_group=result.n_per_group,
        n_total=result.n_total,
    )


def design_mean_experiment(
    std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
) -> ExperimentDesign:
    """
    Design an experiment for a continuous metric (e.g. ARPU).

    Wrapper around `sample_size_means` that returns a high-level
    `ExperimentDesign` object.

    Parameters
    ----------
    std : float
        Estimated standard deviation of the metric (same units as the metric).
    mde : float
        Minimal detectable effect as an absolute difference in means.
    alpha : float, optional
        Significance level, by default 0.05.
    power : float, optional
        Desired power (1 - beta), by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    ExperimentDesign
        High-level experiment design object.
    """
    result: SampleSizeResult = sample_size_means(
        std=std,
        mde=mde,
        alpha=alpha,
        power=power,
        tail=tail,
    )

    return ExperimentDesign(
        metric_type="mean",
        baseline=std,
        mde=result.effect,
        alpha=alpha,
        power=power,
        tail=tail,
        n_per_group=result.n_per_group,
        n_total=result.n_total,
    )

# Reverse design: fixed n -> what MDE can we detect?

def mde_proportions_from_n(
    p_baseline: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
    mde_min: float = 0.0005,
    mde_max: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Compute the minimal detectable effect (MDE) for a binary metric
    given a fixed sample size per group.

    This uses a simple binary search on the absolute difference in
    conversion rate `mde` so that the achieved power is close to the
    desired target. In other words, we solve:

        power_proportions(p_baseline, mde, n_per_group, alpha) ~= power

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate in the control group (0 < p < 1).
    n_per_group : int
        Fixed sample size per group.
    alpha : float, optional
        Significance level, by default 0.05.
    power : float, optional
        Desired power, by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".
    mde_min : float, optional
        Lower bound for the search over MDE (absolute difference),
        by default 0.0005 (0.05 percentage points).
    mde_max : float, optional
        Upper bound for the search over MDE, by default 0.5.
    tol : float, optional
        Tolerance for the binary search; the algorithm stops when
        the interval length is below this value, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations for the search, by default 100.

    Returns
    -------
    float
        Approximate minimal detectable effect (absolute difference in rate).

    Raises
    ------
    RuntimeError
        If the desired power cannot be bracketed within [mde_min, mde_max].
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1).")

    # Power at the lower and upper bounds
    power_low = power_proportions(
        p_baseline=p_baseline,
        mde=mde_min,
        n_per_group=n_per_group,
        alpha=alpha,
        tail=tail,
    )
    power_high = power_proportions(
        p_baseline=p_baseline,
        mde=mde_max,
        n_per_group=n_per_group,
        alpha=alpha,
        tail=tail,
    )

    if power_low > power:
        # Even the tiniest effect in the range is detectable with > target power.
        # MDE is smaller than mde_min; we return the lower bound.
        return mde_min

    if power_high < power:
        # Even a large effect is not detectable with the given sample size.
        raise RuntimeError(
            "Target power cannot be achieved within the provided MDE range. "
            "Consider increasing n_per_group or extending [mde_min, mde_max]."
        )

    # Binary search over MDE
    lo, hi = mde_min, mde_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pwr = power_proportions(
            p_baseline=p_baseline,
            mde=mid,
            n_per_group=n_per_group,
            alpha=alpha,
            tail=tail,
        )
        if pwr < power:
            # We need a larger effect to reach the target power
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def mde_means_from_n(
    std: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: TailType = "two-sided",
) -> float:
    """
    Compute the minimal detectable effect (MDE) for a continuous metric
    given a fixed sample size per group.

    For the two-sample difference in means with equal group sizes and
    common standard deviation `std`, the required sample size formula is:

        n_per_group = (2 * std^2 * (z_alpha + z_beta)^2) / mde^2

    Solving for `mde` yields:

        mde = sqrt(2 * std^2 * (z_alpha + z_beta)^2 / n_per_group)

    We reuse this relationship in reverse.

    Parameters
    ----------
    std : float
        Estimated standard deviation of the metric.
    n_per_group : int
        Fixed sample size per group.
    alpha : float, optional
        Significance level, by default 0.05.
    power : float, optional
        Desired power, by default 0.8.
    tail : {"two-sided", "one-sided"}, optional
        Type of test, by default "two-sided".

    Returns
    -------
    float
        Minimal detectable effect (absolute difference in means).
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")

    # We can simply reuse sample_size_means through power_means by
    # binary search, but using the closed-form relationship is simpler:
    # we call sample_size_means with a dummy MDE = 1.0 to obtain the
    # constant in front, then scale.
    # However, it's even easier to directly compute via power_means:

    # We find mde such that power_means(...) ~= target power using
    # a simple scaling relationship. For clarity and robustness we
    # can also do a small binary search, but here we prefer the
    # analytical formula derived above using SampleSizeResult.

    # Create a temporary design with MDE = 1 to extract the scaling factor
    tmp = sample_size_means(
        std=std,
        mde=1.0,
        alpha=alpha,
        power=power,
        tail=tail,
    )

    # From the formula n_prop = C / mde^2 we have:
    # C = n_per_group_temp * (mde_temp)^2 = tmp.n_per_group * 1^2
    C = tmp.n_per_group

    # For a given n_per_group: mde = sqrt(C / n_per_group)
    mde = (C / float(n_per_group)) ** 0.5
    return mde
