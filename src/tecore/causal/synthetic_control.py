from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .diagnostics import r2_score, rmse, residual_autocorr_summary, quality_warnings


@dataclass(frozen=True)
class SynthControlFit:
    model: Ridge
    y_hat: np.ndarray
    residuals_pre: np.ndarray
    diagnostics: Dict[str, float]
    warnings: list[str]


def fit_synthetic_control_ridge(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    X_all: np.ndarray,
    ridge_alpha: float,
    pre_r2_min: float,
    residual_acf_abs_max: float,
) -> SynthControlFit:
    model = Ridge(alpha=ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X_pre, y_pre)

    y_hat_all = model.predict(X_all)
    y_hat_pre = model.predict(X_pre)
    resid_pre = y_pre - y_hat_pre

    pre_r2 = r2_score(y_pre, y_hat_pre)
    pre_rmse = rmse(y_pre, y_hat_pre)
    acf = residual_autocorr_summary(resid_pre, max_lag=7)

    diags: Dict[str, float] = {"r2_pre": pre_r2, "rmse_pre": pre_rmse, **acf}
    warns = quality_warnings(pre_r2=pre_r2, acf=acf, r2_min=pre_r2_min, acf_abs_max=residual_acf_abs_max)

    return SynthControlFit(
        model=model,
        y_hat=y_hat_all,
        residuals_pre=resid_pre,
        diagnostics=diags,
        warnings=warns,
    )
