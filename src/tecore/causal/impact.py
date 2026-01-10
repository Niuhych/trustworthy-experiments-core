from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .schema import DataSpec, ImpactConfig, ImpactMethod, ImpactResult
from .preprocess import prepare_timeseries
from .diagnostics import r2_score_safe as r2_score, rmse, residual_autocorr_summary, quality_warnings
from .placebo import placebo_in_time_test, two_sided_p_value
from .synthetic_control import fit_synthetic_control_ridge
from .did import did_time_series
from .economics import compute_economics, sum_spend_over_post


def _block_bootstrap_residuals(residuals: np.ndarray, n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    m = len(residuals)
    if m == 0:
        return np.zeros(n, dtype=float)
    if block_size <= 1:
        return residuals[rng.integers(0, m, size=n)]

    starts = rng.integers(0, m, size=int(np.ceil(n / block_size)))
    out = []
    for s in starts:
        out.extend(list(residuals[s:min(s + block_size, m)]))
        if len(out) >= n:
            break
    if len(out) < n:
        out.extend(list(residuals[: (n - len(out))]))
    return np.asarray(out[:n], dtype=float)


def _fit_ridge_counterfactual(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    X_all: np.ndarray,
    ridge_alpha: float,
) -> Tuple[Ridge, np.ndarray, np.ndarray]:
    model = Ridge(alpha=ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X_pre, y_pre)
    y_hat_all = model.predict(X_all)
    y_hat_pre = model.predict(X_pre)
    resid_pre = y_pre - y_hat_pre
    return model, y_hat_all, resid_pre


def _compute_intervals_from_bootstrap(
    y_post: np.ndarray,
    y_hat_post: np.ndarray,
    resid_pre: np.ndarray,
    alpha: float,
    bootstrap_iters: int,
    block_size: int,
    random_state: int,
) -> Dict[str, Any]:
    """
    Produce uncertainty for counterfactual and effect using block-bootstrap over pre residuals.

    We approximate counterfactual distribution:
      y_cf_post = y_hat_post + boot_resid
    Then:
      effect_post = y_post - y_cf_post
    """
    rng = np.random.default_rng(random_state)

    n_post = len(y_post)
    if n_post <= 0:
        raise ValueError("Empty post period.")

    cf_samples = np.zeros((bootstrap_iters, n_post), dtype=float)
    eff_samples = np.zeros((bootstrap_iters, n_post), dtype=float)

    for b in range(int(bootstrap_iters)):
        boot_resid = _block_bootstrap_residuals(resid_pre, n_post, block_size, rng)
        y_cf = y_hat_post + boot_resid
        cf_samples[b, :] = y_cf
        eff_samples[b, :] = y_post - y_cf

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    y_cf_lo = np.quantile(cf_samples, lo_q, axis=0)
    y_cf_hi = np.quantile(cf_samples, hi_q, axis=0)

    eff_lo = np.quantile(eff_samples, lo_q, axis=0)
    eff_hi = np.quantile(eff_samples, hi_q, axis=0)

    # cumulative samples
    cum_samples = np.sum(eff_samples, axis=1)
    cum_lo = float(np.quantile(cum_samples, lo_q))
    cum_hi = float(np.quantile(cum_samples, hi_q))

    # point effect = average effect in post
    point_samples = np.mean(eff_samples, axis=1)
    point_lo = float(np.quantile(point_samples, lo_q))
    point_hi = float(np.quantile(point_samples, hi_q))

    return {
        "y_cf_lo": y_cf_lo,
        "y_cf_hi": y_cf_hi,
        "eff_lo": eff_lo,
        "eff_hi": eff_hi,
        "cum_ci": (cum_lo, cum_hi),
        "point_ci": (point_lo, point_hi),
        "cum_samples": cum_samples,
        "point_samples": point_samples,
    }


def run_impact(
    df: pd.DataFrame,
    spec: DataSpec,
    cfg: ImpactConfig,
) -> ImpactResult:
    """
    Unified entry point (AB-like):
      - returns effect + CI + p-value (placebo) + diagnostics
    """
    intervention_date = cfg.intervention_ts()

    prepared = prepare_timeseries(df, spec, intervention_date=intervention_date)
    dff = prepared.df
    pre_mask = prepared.pre_mask.values
    post_mask = prepared.post_mask.values

    if pre_mask.sum() < cfg.pre_period_min_points:
        raise ValueError(
            f"Pre-period too short: n_pre={int(pre_mask.sum())} < {cfg.pre_period_min_points}. "
            "Increase pre window or aggregate weekly."
        )

    y = dff[spec.y_col].astype(float).values
    X = dff[prepared.feature_cols].astype(float).values

    # Method dispatch
    if cfg.method == ImpactMethod.DID:
        if len(spec.x_cols) < 1:
            raise ValueError("DID requires at least one control series in x_cols (first one is used).")
        control = dff[spec.x_cols[0]].astype(float).values
        did_res = did_time_series(
            treated=y,
            control=control,
            pre_mask=pre_mask,
            post_mask=post_mask,
            alpha=cfg.alpha,
            bootstrap_iters=cfg.bootstrap_iters,
            block_size=cfg.block_size,
            random_state=cfg.random_state,
        )

        # effect_series for DID: use diff baseline
        dates = pd.to_datetime(dff[spec.date_col])
        diff = y - control
        baseline_pre = float(np.mean(diff[pre_mask]))
        y_hat = np.full_like(y, fill_value=np.nan, dtype=float)
        y_hat[post_mask] = control[post_mask] + baseline_pre  # treated counterfactual approx

        effect = np.full_like(y, fill_value=np.nan, dtype=float)
        effect[post_mask] = y[post_mask] - y_hat[post_mask]

        eff_df = pd.DataFrame({
            "date": dates.astype("datetime64[ns]"),
            "y": y,
            "y_hat": y_hat,
            "effect": effect,
        })

        # cum series only defined post
        eff_df["cum_effect"] = np.nan
        eff_df.loc[post_mask, "cum_effect"] = np.cumsum(eff_df.loc[post_mask, "effect"].values)

        rel = float(did_res.cum_effect / np.sum(y_hat[post_mask])) if np.isfinite(np.sum(y_hat[post_mask])) and np.sum(y_hat[post_mask]) != 0 else float("nan")
        rel_ci = (float("nan"), float("nan"))

        return ImpactResult(
            method=cfg.method,
            intervention_date=intervention_date,
            alpha=cfg.alpha,
            point_effect=did_res.point_effect,
            point_ci=did_res.point_ci,
            cum_effect=did_res.cum_effect,
            cum_ci=did_res.cum_ci,
            rel_effect=rel,
            rel_ci=rel_ci,
            p_value=None,
            diagnostics=did_res.diagnostics,
            warnings=[],
            effect_series=eff_df,
            placebo_results=None,
            economics=None,
        )

    # causal_impact_like and synthetic_control use same backbone in v1 (ridge on pre)
    X_pre = X[pre_mask]
    y_pre = y[pre_mask]
    X_post = X[post_mask]
    y_post = y[post_mask]

    # Fit
    if cfg.method in {ImpactMethod.CAUSAL_IMPACT_LIKE, ImpactMethod.SYNTHETIC_CONTROL}:
        fit = fit_synthetic_control_ridge(
            X_pre=X_pre,
            y_pre=y_pre,
            X_all=X,
            ridge_alpha=cfg.ridge_alpha,
            pre_r2_min=cfg.pre_r2_min,
            residual_acf_abs_max=cfg.residual_autocorr_abs_max,
        )
        y_hat = fit.y_hat
        resid_pre = fit.residuals_pre
        diagnostics = dict(fit.diagnostics)
        warnings = list(fit.warnings)
    else:
        raise ValueError(f"Unsupported method: {cfg.method}")

    y_hat_post = y_hat[post_mask]

    # Uncertainty via block bootstrap
    intervals = _compute_intervals_from_bootstrap(
        y_post=y_post,
        y_hat_post=y_hat_post,
        resid_pre=resid_pre,
        alpha=cfg.alpha,
        bootstrap_iters=cfg.bootstrap_iters,
        block_size=cfg.block_size,
        random_state=cfg.random_state,
    )

    effect_post = y_post - y_hat_post
    point_effect = float(np.mean(effect_post))
    cum_effect = float(np.sum(effect_post))

    denom = float(np.sum(y_hat_post))
    rel_effect = float(cum_effect / denom) if denom != 0 else float("nan")

    # rel CI from cum CI approximation
    cum_ci = intervals["cum_ci"]
    if denom != 0:
        rel_ci = (float(cum_ci[0] / denom), float(cum_ci[1] / denom))
    else:
        rel_ci = (float("nan"), float("nan"))

    # build effect series
    dates = pd.to_datetime(dff[spec.date_col]).astype("datetime64[ns]")
    eff_df = pd.DataFrame({
        "date": dates,
        "y": y,
        "y_hat": y_hat,
        "effect": (y - y_hat),
    })

    # attach post-only intervals; pre keep NaN (so plots are clean)
    eff_df["y_hat_lower"] = np.nan
    eff_df["y_hat_upper"] = np.nan
    eff_df.loc[post_mask, "y_hat_lower"] = intervals["y_cf_lo"]
    eff_df.loc[post_mask, "y_hat_upper"] = intervals["y_cf_hi"]

    eff_df["effect_lower"] = np.nan
    eff_df["effect_upper"] = np.nan
    eff_df.loc[post_mask, "effect_lower"] = intervals["eff_lo"]
    eff_df.loc[post_mask, "effect_upper"] = intervals["eff_hi"]

    eff_df["cum_effect"] = np.nan
    eff_df.loc[post_mask, "cum_effect"] = np.cumsum(eff_df.loc[post_mask, "effect"].values)

    # cum CI as a running band (approx by using pointwise quantiles cumulatively; conservative alternative is to keep only final)
    eff_df["cum_lower"] = np.nan
    eff_df["cum_upper"] = np.nan
    eff_df.loc[post_mask, "cum_lower"] = np.cumsum(eff_df.loc[post_mask, "effect_lower"].values)
    eff_df.loc[post_mask, "cum_upper"] = np.cumsum(eff_df.loc[post_mask, "effect_upper"].values)

    # Placebo p-value
    placebo_df = None
    p_value = None
    if cfg.run_placebo:
        post_len = int(post_mask.sum())

        def _estimate_placebo(placebo_date: pd.Timestamp, post_window_len: int) -> float:
            # Define pseudo pre/post inside original data:
            #   pre': dates < placebo_date
            #   post': [placebo_date, placebo_date + post_window_len)
            m_pre = dates < placebo_date
            m_post = (dates >= placebo_date) & (dates < (placebo_date + pd.Timedelta(days=post_window_len)))

            if m_pre.sum() < cfg.placebo_min_pre_points:
                raise ValueError("placebo pre too short")
            if m_post.sum() < post_window_len:
                raise ValueError("placebo window truncated")

            X_pre_p = X[m_pre]
            y_pre_p = y[m_pre]
            X_post_p = X[m_post]
            y_post_p = y[m_post]

            # fit ridge on placebo pre
            _, y_hat_all_p, resid_pre_p = _fit_ridge_counterfactual(
                X_pre=X_pre_p, y_pre=y_pre_p, X_all=X, ridge_alpha=cfg.ridge_alpha
            )
            y_hat_post_p = y_hat_all_p[m_post]

            # bootstrap only for stability (optional); for speed in placebo, use point estimate cumulative
            eff_p = y_post_p - y_hat_post_p
            return float(np.sum(eff_p))

        placebo_df, _ = placebo_in_time_test(
            df=dff,
            date_col=spec.date_col,
            intervention_date=intervention_date,
            post_window_len=int(post_mask.sum()),
            n_placebos=cfg.n_placebos,
            min_pre_points=cfg.placebo_min_pre_points,
            random_state=cfg.random_state,
            estimate_fn=_estimate_placebo,
        )
        if placebo_df is not None and len(placebo_df) > 0:
            p_value = two_sided_p_value(placebo_df["cum_effect"].values, cum_effect)

    economics = None
    if cfg.margin_rate is not None:
        spend = None
        if cfg.spend_col is not None:
            spend = sum_spend_over_post(dff, cfg.spend_col, pd.Series(post_mask))
        economics = compute_economics(
            cum_effect=cum_effect,
            cum_ci=cum_ci,
            margin_rate=cfg.margin_rate,
            campaign_spend=spend,
            incremental_cost=cfg.incremental_cost,
        )

    # headline intervals
    point_ci = intervals["point_ci"]

    # extra diagnostics (pre fit)
    y_hat_pre = y_hat[pre_mask]
    pre_r2 = r2_score(y_pre, y_hat_pre)
    pre_rmse = rmse(y_pre, y_hat_pre)
    acf = residual_autocorr_summary((y_pre - y_hat_pre), max_lag=7)
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    diagnostics.update({"r2_pre": pre_r2, "rmse_pre": pre_rmse, **acf})

    warnings = warnings + quality_warnings(pre_r2, acf, cfg.pre_r2_min, cfg.residual_autocorr_abs_max)

    return ImpactResult(
        method=cfg.method,
        intervention_date=intervention_date,
        alpha=cfg.alpha,
        point_effect=point_effect,
        point_ci=point_ci,
        cum_effect=cum_effect,
        cum_ci=cum_ci,
        rel_effect=rel_effect,
        rel_ci=rel_ci,
        p_value=p_value,
        diagnostics=diagnostics,
        warnings=warnings,
        effect_series=eff_df,
        placebo_results=placebo_df,
        economics=economics,
    )
