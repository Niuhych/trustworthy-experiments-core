from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² without sklearn; returns NaN if variance is zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def residual_autocorr(residuals: np.ndarray, max_lag: int = 7) -> Dict[int, float]:
    """
    Simple residual autocorrelation by lag (Pearson correlation).
    This is not a formal whiteness test, but is a pragmatic diagnostic.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]

    out: Dict[int, float] = {}
    if len(r) < max_lag + 3:
        for k in range(1, max_lag + 1):
            out[k] = float("nan")
        return out

    r0 = r - float(np.mean(r))
    for k in range(1, max_lag + 1):
        a = r0[:-k]
        b = r0[k:]
        denom = float(np.std(a) * np.std(b))
        out[k] = float(np.corrcoef(a, b)[0, 1]) if denom > 0 else float("nan")

    return out


@dataclass(frozen=True)
class QualityThresholds:
    """Default trustworthiness gates; tune as needed."""
    min_pre_r2: float = 0.20
    max_abs_autocorr_lag1: float = 0.50


def quality_warnings(
    pre_r2: float,
    pre_rmse: float,
    acf: Dict[int, float],
    thresholds: QualityThresholds = QualityThresholds(),
) -> List[str]:
    warnings: List[str] = []

    if not np.isfinite(pre_r2) or pre_r2 < thresholds.min_pre_r2:
        warnings.append(
            f"Pre-fit quality is weak (R2={pre_r2:.3f}). Counterfactual may be unreliable."
        )

    lag1 = acf.get(1, float("nan"))
    if np.isfinite(lag1) and abs(float(lag1)) > thresholds.max_abs_autocorr_lag1:
        warnings.append(
            f"Residual autocorrelation is high at lag 1 (acf1={float(lag1):.3f}). CI may be optimistic."
        )

    # RMSE is context-dependent (scale of y), so we do not hard-gate by default.
    _ = pre_rmse
    return warnings


def compute_pre_fit_diagnostics(
    y_pre: np.ndarray, yhat_pre: np.ndarray, max_lag: int = 7
) -> Dict[str, object]:
    """
    Returns:
      - pre_r2: float
      - pre_rmse: float
      - resid_autocorr: Dict[int, float]
    """
    y_pre = np.asarray(y_pre, dtype=float)
    yhat_pre = np.asarray(yhat_pre, dtype=float)
    res = y_pre - yhat_pre

    return {
        "pre_r2": r2_score_safe(y_pre, yhat_pre),
        "pre_rmse": rmse(y_pre, yhat_pre),
        "resid_autocorr": residual_autocorr(res, max_lag=max_lag),
    }
