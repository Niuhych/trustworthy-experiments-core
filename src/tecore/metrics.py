"""
Helpers for metric design in online experiments.

Focus:
- simple summaries for customer-level metrics (means, variance, etc.);
- ratio metrics (e.g. ARPU = revenue / users, revenue per session, etc.);
- linearization for ratio metrics, which is often used to apply
  standard t-tests to ratio KPIs.

The goal is not to cover every possible case, but to provide
clear, readable building blocks for B2C experimentation notebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert input to a 1D NumPy array of floats."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    return arr


# Basic summaries for customer-level metrics

@dataclass
class MetricSummary:
    """
    Simple summary statistics for a 1D metric.

    Attributes
    ----------
    n : int
        Number of observations.
    mean : float
        Sample mean of the metric.
    var : float
        Sample variance (unbiased, ddof=1; 0 if n <= 1).
    std : float
        Sample standard deviation (sqrt(var)).
    """

    n: int
    mean: float
    var: float
    std: float


def summarize_metric(values: ArrayLike) -> MetricSummary:
    """
    Compute basic summary statistics for a 1D metric.

    Parameters
    ----------
    values : array-like of float
        Customer-level values of the metric
        (e.g. customer revenue over a period, number of sessions, etc.).

    Returns
    -------
    MetricSummary
        Summary containing n, mean, variance and standard deviation.
    """
    arr = _to_numpy(values)
    n = arr.size
    if n == 0:
        raise ValueError("Cannot summarize an empty array.")

    mean = float(np.mean(arr))
    # Unbiased sample variance (ddof=1), falling back to 0 if n == 1
    var = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    std = math.sqrt(var)

    return MetricSummary(n=n, mean=mean, var=var, std=std)


def difference_in_means(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the difference in sample means between two groups (B - A).

    Parameters
    ----------
    a : array-like
        Metric values for the control group.
    b : array-like
        Metric values for the treatment group.

    Returns
    -------
    float
        Sample mean of group B minus sample mean of group A.
    """
    a_arr = _to_numpy(a)
    b_arr = _to_numpy(b)
    return float(np.mean(b_arr) - np.mean(a_arr))

# Ratio metrics and linearization

@dataclass
class RatioSummary:
    """
    Summary for a ratio metric of the form R = sum(numerator) / sum(denominator).

    Attributes
    ----------
    numerator_sum : float
        Sum of numerator values.
    denominator_sum : float
        Sum of denominator values.
    ratio : float
        Ratio R = numerator_sum / denominator_sum.
    """

    numerator_sum: float
    denominator_sum: float
    ratio: float


def ratio_metric(numerator: ArrayLike, denominator: ArrayLike) -> RatioSummary:
    """
    Compute a ratio metric for a group.

    Typical examples:
    - ARPU = total_revenue / number_of_users;
    - revenue per session = total_revenue / number_of_sessions.

    Parameters
    ----------
    numerator : array-like of float
        Numerator values per unit (e.g. revenue per user, revenue per session).
    denominator : array-like of float
        Denominator values per unit (e.g. user count, session count).
        For per-user ratios, this is often simply an array of ones.

    Returns
    -------
    RatioSummary
        Summary with numerator_sum, denominator_sum and ratio.
    """
    num = _to_numpy(numerator)
    den = _to_numpy(denominator)

    if num.size != den.size:
        raise ValueError("numerator and denominator must have the same length.")

    numerator_sum = float(np.sum(num))
    denominator_sum = float(np.sum(den))

    if denominator_sum == 0:
        raise ZeroDivisionError("denominator_sum is zero; cannot compute ratio metric.")

    ratio_value = numerator_sum / denominator_sum

    return RatioSummary(
        numerator_sum=numerator_sum,
        denominator_sum=denominator_sum,
        ratio=ratio_value,
    )


def linearize_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    baseline_ratio: float | None = None,
) -> np.ndarray:
    """
    Compute linearized values for a ratio metric.

    This is a common technique for handling ratio KPIs in experiments:
    - let R be a baseline ratio (e.g. global ratio or control-group ratio),
    - for each observation (x_i, y_i) compute z_i = x_i - R * y_i.

    Under mild assumptions, the difference in means of z between
    treatment and control behaves like a difference in *non-ratio* means,
    which allows using standard t-tests.

    Parameters
    ----------
    numerator : array-like of float
        Numerator values x_i (e.g. revenue per user).
    denominator : array-like of float
        Denominator values y_i (e.g. 1 for per-user, sessions, etc.).
    baseline_ratio : float, optional
        Baseline ratio R. If None, it is estimated as:
        R = sum(numerator) / sum(denominator) over the provided data.

    Returns
    -------
    np.ndarray
        1D array of linearized values z_i = x_i - R * y_i.
    """
    num = _to_numpy(numerator)
    den = _to_numpy(denominator)

    if num.size != den.size:
        raise ValueError("numerator and denominator must have the same length.")

    if baseline_ratio is None:
        summary = ratio_metric(num, den)
        baseline_ratio = summary.ratio

    # z_i = x_i - R * y_i
    linearized = num - baseline_ratio * den
    return linearized


def make_per_user_arrays(
    revenues: ArrayLike,
    user_counts: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper for ARPU-style metrics.

    In many B2C settings we work with arrays where:
    - each row corresponds to a user,
    - `revenues[i]` is the total revenue from user i over a period.

    Sometimes we also want to explicitly pass the denominator as
    "number of users" (which is usually an array of ones).

    This function returns two aligned arrays:

    - numerator: revenue per user;
    - denominator: user count per user (all ones if not provided).

    Parameters
    ----------
    revenues : array-like of float
        Revenue per user (or any other per-user monetary metric).
    user_counts : array-like of float, optional
        Number of users per row (usually ones). If None, an array
        of ones with the same length as `revenues` is created.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple (numerator, denominator) ready for ratio/linearization functions.
    """
    rev = _to_numpy(revenues)

    if user_counts is None:
        den = np.ones_like(rev)
    else:
        den = _to_numpy(user_counts)
        if den.size != rev.size:
            raise ValueError("revenues and user_counts must have the same length.")

    return rev, den
