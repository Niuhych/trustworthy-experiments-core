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


def cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
    theta: float | None = None,
    x_mean: float | None = None,
) -> CupedResult:
    """Apply CUPED adjustment to a single sample.

    Parameters
    ----------
    theta : float | None
        If None, theta is estimated from (y, x). Otherwise fixed theta is used.
    x_mean : float | None
        Centering constant for X. If None, uses mean(x) of the provided sample.
        For A/B CUPED, you typically want pooled mean across groups.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    th = cuped_theta(y, x) if theta is None else float(theta)
    mu = float(np.mean(x)) if x_mean is None else float(x_mean)

    y_adj = y - th * (x - mu)

    vy = np.var(y, ddof=1)
    vy_adj = np.var(y_adj, ddof=1)
    vr = 0.0 if vy <= 0 else float(1.0 - (vy_adj / vy))

    return CupedResult(theta=th, y_adj=y_adj, var_reduction=vr)


def cuped_ab_adjust(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, theta: float | None = None
) -> Tuple[CupedResult, CupedResult]:
    """A/B CUPED with pooled theta (optional) and pooled centering of X.

    This is the recommended default for mean-like metrics.
    """
    y_c = np.asarray(y_c, float); x_c = np.asarray(x_c, float)
    y_t = np.asarray(y_t, float); x_t = np.asarray(x_t, float)

    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])

    mu = float(np.mean(x_all))
    th = cuped_theta(y_all, x_all) if theta is None else float(theta)

    res_c = cuped_adjust(y_c, x_c, theta=th, x_mean=mu)
    res_t = cuped_adjust(y_t, x_t, theta=th, x_mean=mu)
    return res_c, res_t


def cuped_ab_crossfit_adjust(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, seed: int = 0
) -> Tuple[CupedResult, CupedResult]:
    """2-fold cross-fitted A/B CUPED.

    Use this for heavy-tailed metrics and transformed metrics (e.g., linearized ratio),
    where estimating theta on the same sample can inflate Type I error.
    """
    rng = np.random.default_rng(seed)

    y_c = np.asarray(y_c, float); x_c = np.asarray(x_c, float)
    y_t = np.asarray(y_t, float); x_t = np.asarray(x_t, float)

    idx_c = rng.permutation(len(y_c))
    idx_t = rng.permutation(len(y_t))

    c_half = len(y_c) // 2
    t_half = len(y_t) // 2

    folds_c = [idx_c[:c_half], idx_c[c_half:]]
    folds_t = [idx_t[:t_half], idx_t[t_half:]]

    y_adj_c = np.empty_like(y_c)
    y_adj_t = np.empty_like(y_t)
    thetas = []

    for k in [0, 1]:
        tr_c = folds_c[1 - k]; te_c = folds_c[k]
        tr_t = folds_t[1 - k]; te_t = folds_t[k]

        y_tr = np.concatenate([y_c[tr_c], y_t[tr_t]])
        x_tr = np.concatenate([x_c[tr_c], x_t[tr_t]])

        mu = float(np.mean(x_tr))
        th = cuped_theta(y_tr, x_tr)
        thetas.append(th)

        y_adj_c[te_c] = y_c[te_c] - th * (x_c[te_c] - mu)
        y_adj_t[te_t] = y_t[te_t] - th * (x_t[te_t] - mu)

    # variance reduction estimate (pooled)
    y_all = np.concatenate([y_c, y_t])
    y_adj_all = np.concatenate([y_adj_c, y_adj_t])
    vy = np.var(y_all, ddof=1)
    vy_adj = np.var(y_adj_all, ddof=1)
    vr = 0.0 if vy <= 0 else float(1.0 - (vy_adj / vy))

    th_avg = float(np.mean(thetas))

    return (
        CupedResult(theta=th_avg, y_adj=y_adj_c, var_reduction=vr),
        CupedResult(theta=th_avg, y_adj=y_adj_t, var_reduction=vr),
    )


def cuped_split_adjust(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray
) -> Tuple[CupedResult, CupedResult]:
    """Deprecated: theta estimated on control only.

    Kept for backward compatibility; prefer:
    - cuped_ab_adjust for mean metrics
    - cuped_ab_crossfit_adjust for linearized ratio / heavy-tail metrics
    """
    res_c_tmp = cuped_adjust(y_c, x_c, theta=None, x_mean=None)
    theta = res_c_tmp.theta
    res_c = cuped_adjust(y_c, x_c, theta=theta, x_mean=float(np.mean(x_c)))
    res_t = cuped_adjust(y_t, x_t, theta=theta, x_mean=float(np.mean(x_t)))
    return res_c, res_t
