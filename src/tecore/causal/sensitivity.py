from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .schema import DataSpec, ImpactConfig
from .preprocess import prepare_timeseries


@dataclass(frozen=True)
class SensitivityRow:
    scenario: str
    cum_effect: float
    rel_effect: float


def _estimate_cum_effect_with_feature_subset(
    df: pd.DataFrame,
    spec: DataSpec,
    cfg: ImpactConfig,
    keep_features: List[str],
) -> SensitivityRow:
    intervention_date = cfg.intervention_ts()
    prepared = prepare_timeseries(df, spec, intervention_date=intervention_date)

    dff = prepared.df
    pre_mask = prepared.pre_mask.values
    post_mask = prepared.post_mask.values

    y = dff[spec.y_col].astype(float).values
    X = dff[keep_features].astype(float).values

    X_pre = X[pre_mask]
    y_pre = y[pre_mask]
    X_post = X[post_mask]
    y_post = y[post_mask]

    model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X_pre, y_pre)
    y_hat_post = model.predict(X_post)

    eff = y_post - y_hat_post
    cum = float(np.sum(eff))
    denom = float(np.sum(y_hat_post))
    rel = float(cum / denom) if denom != 0 else float("nan")
    return SensitivityRow(scenario="custom", cum_effect=cum, rel_effect=rel)


def leave_one_covariate_out_test(
    df: pd.DataFrame,
    spec: DataSpec,
    cfg: ImpactConfig,
) -> pd.DataFrame:
    """
    Stress test: drop one covariate at a time (keeping engineered features like _t / dow_*)
    and see how cumulative effect changes.

    Returns a table with scenario and cum/rel effects.
    """
    intervention_date = cfg.intervention_ts()
    prepared = prepare_timeseries(df, spec, intervention_date=intervention_date)
    full_features = list(prepared.feature_cols)

    base = _estimate_cum_effect_with_feature_subset(df, spec, cfg, keep_features=full_features)
    rows = [SensitivityRow(scenario="all_features", cum_effect=base.cum_effect, rel_effect=base.rel_effect)]

    # Only drop raw covariates (not engineered features)
    for x in spec.x_cols:
        keep = [f for f in full_features if f != x]
        r = _estimate_cum_effect_with_feature_subset(df, spec, cfg, keep_features=keep)
        rows.append(SensitivityRow(scenario=f"drop:{x}", cum_effect=r.cum_effect, rel_effect=r.rel_effect))

    return pd.DataFrame([r.__dict__ for r in rows])


def drop_last_k_days_of_pre_test(
    df: pd.DataFrame,
    spec: DataSpec,
    cfg: ImpactConfig,
    k_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Stability test: reduce the training pre-period by dropping last k days of pre,
    and re-estimate cumulative effect.

    k_list: e.g., [7, 14, 21]
    """
    if k_list is None:
        k_list = [7, 14, 21]

    intervention_date = cfg.intervention_ts()
    prepared = prepare_timeseries(df, spec, intervention_date=intervention_date)
    dff = prepared.df
    pre_idx = np.where(prepared.pre_mask.values)[0]
    post_mask = prepared.post_mask.values

    y = dff[spec.y_col].astype(float).values
    X = dff[prepared.feature_cols].astype(float).values

    rows: List[Dict[str, float]] = []

    # baseline
    rows.append({"scenario": "baseline", "cum_effect": float("nan"), "rel_effect": float("nan")})

    for k in k_list:
        if len(pre_idx) - k < cfg.pre_period_min_points:
            rows.append({"scenario": f"drop_last_pre:{k}", "cum_effect": float("nan"), "rel_effect": float("nan")})
            continue

        cut = pre_idx[-k]  # index of first dropped element
        custom_pre_mask = np.zeros(len(dff), dtype=bool)
        custom_pre_mask[pre_idx[: len(pre_idx) - k]] = True

        X_pre = X[custom_pre_mask]
        y_pre = y[custom_pre_mask]
        X_post = X[post_mask]
        y_post = y[post_mask]

        model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
        model.fit(X_pre, y_pre)
        y_hat_post = model.predict(X_post)

        eff = y_post - y_hat_post
        cum = float(np.sum(eff))
        denom = float(np.sum(y_hat_post))
        rel = float(cum / denom) if denom != 0 else float("nan")

        rows.append({"scenario": f"drop_last_pre:{k}", "cum_effect": cum, "rel_effect": rel})

    # fill baseline using full pre (for convenience)
    full_model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
    full_model.fit(X[prepared.pre_mask.values], y[prepared.pre_mask.values])
    y_hat_post_full = full_model.predict(X[post_mask])
    eff_full = y[post_mask] - y_hat_post_full
    cum_full = float(np.sum(eff_full))
    denom_full = float(np.sum(y_hat_post_full))
    rel_full = float(cum_full / denom_full) if denom_full != 0 else float("nan")

    rows[0]["cum_effect"] = cum_full
    rows[0]["rel_effect"] = rel_full

    return pd.DataFrame(rows)
