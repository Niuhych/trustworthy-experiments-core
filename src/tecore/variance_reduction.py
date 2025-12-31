"""
Variance reduction utilities for online experiments.

Currently implemented:
- CUPED (Controlled Experiments Using Pre-Experiment Data)

CUPED reduces variance by leveraging a pre-period covariate X that is
correlated with the post-period metric Y.

Core idea:
    Y_cuped = Y - theta * (X - mean(X))
where theta is estimated as:
    theta = Cov(Y, X) / Var(X)

Notes:
- CUPED is unbiased for A/B tests when X is unaffected by treatment and is measured pre-experiment.
- You can apply CUPED to:
  * continuous metrics (ARPU, revenue per user, time spent, etc.)
  * transformed metrics (e.g., linearized ratio metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CupedResult:
    """
    Container for CUPED adjustment output.

    Attributes
    ----------
    theta : float
        Estimated CUPED coefficient.
    y_adj : np.ndarray
        Adjusted outcomes (same shape as y).
    var_reduction : float
        Estimated relative variance reduction: 1 - Var(y_adj)/Var(y).
    """
    theta: float
    y_adj: np.ndarray
    var_reduction: float


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Estimate CUPED coefficient theta = Cov(Y, X) / Var(X)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    vx = np.var(x, ddof=1)
    if vx <= 0:
        return 0.0

    cov = np.cov(y, x, ddof=1)[0, 1]
    return float(cov / vx)


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> CupedResult:
    """Apply CUPED adjustment."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    th = cuped_theta(y, x) if theta is None else float(theta)
    x_centered = x - np.mean(x)
    y_adj = y - th * x_centered

    vy = np.var(y, ddof=1)
    vy_adj = np.var(y_adj, ddof=1)

    vr = 0.0 if vy <= 0 else float(1.0 - (vy_adj / vy))
    return CupedResult(theta=th, y_adj=y_adj, var_reduction=vr)


def cuped_split_adjust(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray
) -> Tuple[CupedResult, CupedResult]:
    """Apply CUPED using a theta estimated on control only."""
    res_c_tmp = cuped_adjust(y_c, x_c, theta=None)
    theta = res_c_tmp.theta
    res_c = cuped_adjust(y_c, x_c, theta=theta)
    res_t = cuped_adjust(y_t, x_t, theta=theta)
    return res_c, res_t
