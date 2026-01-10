from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 without sklearn; returns NaN if variance is zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# Backward-compatible name expected by impact.py (and other callers)
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score_safe(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def residual_autocorr(residuals: np.ndarray, max_lag: int = 7) -> Dict[int, float]:
    """
    Simple residual autocorrelation by lag (Pearson correlation).
    Pragmatic diagnostic, not a formal whiteness test.
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


def residual_autocorr_summary(residuals: np.ndarray, max_lag: int = 7) -> Dict[str, float]:
    """
    Flat summary used by impact.py.

    Keys:
      - acf_lag1 .. acf_lag{max_lag}
      - max_abs_acf_1_to_{max_lag}
    """
    acf = residual_autocorr(residuals, max_lag=max_lag)

    out: Dict[str, float] = {}
    vals: List[float] = []
    for k in range(1, max_lag + 1):
        v = float(acf.get(k, float("nan")))
        out[f"acf_lag{k}"] = v
        if np.isfinite(v):
            vals.append(abs(v))

    out[f"max_abs_acf_1_to_{max_lag}"] = float(max(vals)) if vals else float("nan")
    return out


@dataclass(frozen=True)
class QualityThresholds:
    """Default trustworthiness gates; tune as needed."""
    min_pre_r2: float = 0.20
    max_abs_autocorr_lag1: float = 0.50


def quality_warnings(
    pre_r2: float,
    pre_rmse: float | None = None,
    acf: Dict[int, float] | None = None,
    thresholds: QualityThresholds | float | None = QualityThresholds(),
    **kwargs,
) -> List[str]:
    """
    Produce human-readable warnings for trustworthiness.

    Backward-compatible with multiple historical calling patterns:
      1) quality_warnings(pre_r2, pre_rmse, acf, thresholds=QualityThresholds(...))
      2) quality_warnings(pre_r2, acf_dict, r2_min=..., residual_acf_abs_max=...)
      3) quality_warnings(pre_r2, pre_rmse, acf_dict, r2_min_float, ...)
         where 4th positional arg is actually r2_min (float), not QualityThresholds.
    """

    # Legacy pattern: second positional arg is acf dict (pre_rmse omitted)
    if acf is None and isinstance(pre_rmse, dict):
        acf = pre_rmse  # type: ignore[assignment]
        pre_rmse = None

    if acf is None:
        acf = {}

    # Legacy pattern: thresholds passed as a float (meaning r2_min)
    if isinstance(thresholds, (int, float, np.floating)):
        kwargs.setdefault("r2_min", float(thresholds))
        thresholds_obj = QualityThresholds()
    elif thresholds is None:
        thresholds_obj = QualityThresholds()
    else:
        thresholds_obj = thresholds

    warnings: List[str] = []

    # Back-compat overrides (if provided by callers)
    r2_min = float(kwargs.get("r2_min", thresholds_obj.min_pre_r2))
    acf_abs_max = float(
        kwargs.get(
            "residual_acf_abs_max",
            kwargs.get("acf_lag1_abs_max", thresholds_obj.max_abs_autocorr_lag1),
        )
    )

    if not np.isfinite(pre_r2) or pre_r2 < r2_min:
        warnings.append(
            f"Pre-fit quality is weak (R2={pre_r2:.3f} < {r2_min:.3f}). Counterfactual may be unreliable."
        )

    lag1 = float(acf.get(1, float("nan")))
    if np.isfinite(lag1) and abs(lag1) > acf_abs_max:
        warnings.append(
            f"Residual autocorrelation is high at lag 1 (acf1={lag1:.3f} > {acf_abs_max:.3f}). CI may be optimistic."
        )

    _ = pre_rmse  # optional; kept for reporting
    return warnings


def compute_pre_fit_diagnostics(
    y_pre: np.ndarray, yhat_pre: np.ndarray, max_lag: int = 7
) -> Dict[str, object]:
    """
    Returns:
      - pre_r2: float
      - pre_rmse: float
      - resid_autocorr: Dict[int, float]
      - resid_autocorr_summary: Dict[str, float]
    """
    y_pre = np.asarray(y_pre, dtype=float)
    yhat_pre = np.asarray(yhat_pre, dtype=float)
    res = y_pre - yhat_pre

    acf = residual_autocorr(res, max_lag=max_lag)

    return {
        "pre_r2": r2_score_safe(y_pre, yhat_pre),
        "pre_rmse": rmse(y_pre, yhat_pre),
        "resid_autocorr": acf,
        "resid_autocorr_summary": residual_autocorr_summary(res, max_lag=max_lag),
    }


__all__ = [
    "r2_score_safe",
    "r2_score",
    "rmse",
    "residual_autocorr",
    "residual_autocorr_summary",
    "QualityThresholds",
    "quality_warnings",
    "compute_pre_fit_diagnostics",
]
