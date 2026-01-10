from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTSConfig:
    n_days: int = 200
    start_date: str = "2025-01-01"
    intervention_day: int = 120  # index (0-based), converted to date

    # baseline dynamics
    trend: float = 0.5
    season_amp: float = 10.0
    noise_sd: float = 5.0

    # covariates
    cov_noise_sd: float = 3.0
    cov_corr: float = 0.7  # correlation strength with y

    # effect types
    level_shift: float = 0.0
    slope_change: float = 0.0
    temp_effect_amp: float = 0.0
    temp_effect_decay: float = 0.15  # larger => faster decay

    # confounding: covariate also shifts at intervention
    confounding: bool = False
    confound_shift: float = 0.0

    random_state: int = 42


def generate_synthetic_time_series(cfg: SyntheticTSConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generates a synthetic daily time series:
      y(t) = baseline(trend+weekly season) + beta' X(t) + noise + treatment_effect(t)

    Returns:
      df with columns: date, y, sessions, active_users, marketing_spend, external_index
      meta dict with true cumulative effect over post-period
    """
    rng = np.random.default_rng(cfg.random_state)

    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    t = np.arange(cfg.n_days, dtype=float)

    weekly = np.sin(2 * np.pi * t / 7.0)
    baseline = 100.0 + cfg.trend * t + cfg.season_amp * weekly

    # base latent driver that will correlate covariates and y
    driver = rng.normal(size=cfg.n_days)

    # covariates
    def mk_cov(scale: float) -> np.ndarray:
        return (cfg.cov_corr * driver + np.sqrt(max(0.0, 1 - cfg.cov_corr**2)) * rng.normal(size=cfg.n_days)) * scale

    sessions = 1000 + 50 * mk_cov(scale=1.0) + rng.normal(0, cfg.cov_noise_sd, size=cfg.n_days)
    active_users = 200 + 10 * mk_cov(scale=1.0) + rng.normal(0, cfg.cov_noise_sd, size=cfg.n_days)
    marketing_spend = 300 + 30 * mk_cov(scale=1.0) + rng.normal(0, cfg.cov_noise_sd, size=cfg.n_days)
    external_index = 50 + 5 * mk_cov(scale=1.0) + rng.normal(0, cfg.cov_noise_sd, size=cfg.n_days)

    intervention_idx = int(cfg.intervention_day)
    intervention_idx = max(1, min(cfg.n_days - 2, intervention_idx))
    post_mask = t >= intervention_idx

    # treatment effect components
    effect = np.zeros(cfg.n_days, dtype=float)

    if cfg.level_shift != 0.0:
        effect[post_mask] += cfg.level_shift

    if cfg.slope_change != 0.0:
        effect[post_mask] += cfg.slope_change * (t[post_mask] - intervention_idx)

    if cfg.temp_effect_amp != 0.0:
        dt = (t[post_mask] - intervention_idx)
        effect[post_mask] += cfg.temp_effect_amp * np.exp(-cfg.temp_effect_decay * dt)

    # confounding: covariate shifts too, creating spurious identification risk
    if cfg.confounding:
        sessions[post_mask] += cfg.confound_shift
        marketing_spend[post_mask] += 0.5 * cfg.confound_shift

    # Create y with covariate contributions
    beta = np.array([0.03, 0.2, 0.05, 0.3])  # sessions, dau, spend, external
    X = np.vstack([
        (sessions - np.mean(sessions)) / np.std(sessions),
        (active_users - np.mean(active_users)) / np.std(active_users),
        (marketing_spend - np.mean(marketing_spend)) / np.std(marketing_spend),
        (external_index - np.mean(external_index)) / np.std(external_index),
    ]).T

    y = baseline + X @ beta * 20.0 + rng.normal(0, cfg.noise_sd, size=cfg.n_days) + effect

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "y": y.astype(float),
        "sessions": sessions.astype(float),
        "active_users": active_users.astype(float),
        "marketing_spend": marketing_spend.astype(float),
        "external_index": external_index.astype(float),
    })

    meta = {
        "intervention_date": str(pd.Timestamp(dates[intervention_idx]).date()),
        "true_cum_effect": float(np.sum(effect[post_mask])),
        "true_avg_effect": float(np.mean(effect[post_mask])),
    }
    return df, meta
