"""
Synthetic B2C-like data generator for experimentation notebooks.

Design goals:
- Neutral domain (no product-specific vocabulary).
- Aggregated per-user data for a fixed analysis window (e.g., 14 days).
- Realistic properties:
  * Heterogeneous activity (sessions) with over-dispersion.
  * Heavy-tailed monetary outcomes (revenue), including a "high spenders" segment.
  * Correlated pre-period and post-period behavior (useful for CUPED demos).
- Multiple effect mechanisms:
  * activity: treatment changes activity (sessions).
  * monetization: treatment changes revenue per session.
  * mixed: treatment changes both.

Output schema (minimal):
- user_id
- group: "control" / "test"
- sessions_pre, revenue_pre
- sessions, revenue

Optional extensions:
- conversions, converted (binary metric demo)

Notes:
- This is a pedagogical generator. It is not intended to exactly reproduce
  any specific business domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


Group = Literal["control", "test"]
EffectType = Literal["none", "activity", "monetization", "mixed"]


@dataclass(frozen=True)
class SyntheticB2CConfig:
    """Configuration for synthetic B2C-like data generation."""

    n_users: int = 50_000
    test_share: float = 0.5
    seed: int = 42

    # Activity (sessions) model: Negative Binomial via mean + dispersion.
    # Larger dispersion -> heavier tail and higher variance at fixed mean.
    sessions_mean_pre: float = 3.0
    sessions_mean_post: float = 3.2
    sessions_dispersion: float = 1.5  # > 0

    # Monetization model: revenue per session is lognormal mixture.
    # Mixture adds a small fraction of high spenders with higher mean.
    high_spender_share: float = 0.03
    revps_lognorm_mu: float = -0.2     # mean parameter in log-space
    revps_lognorm_sigma: float = 1.0   # std parameter in log-space
    high_revps_multiplier: float = 6.0 # high spenders have higher rev/session

    # Correlation between pre and post behavior:
    # We use user-level latent multipliers for activity and monetization.
    # Higher values -> stronger correlation between pre and post.
    activity_latent_sigma: float = 0.6
    monetization_latent_sigma: float = 0.7

    # Treatment effect specification (multiplicative lifts).
    effect_type: EffectType = "none"
    lift_sessions: float = 0.00        # e.g. 0.05 means +5% sessions in test
    lift_rev_per_session: float = 0.00 # e.g. 0.03 means +3% rev/session in test

    # Optional binary metric demo (conversion per session)
    include_binary: bool = False
    conversion_rate_pre: float = 0.06
    conversion_rate_post: float = 0.06
    lift_conversion_rate: float = 0.00 # additive lift on conversion rate (abs points)


def _rng(seed: int) -> np.random.Generator:
    """Create a dedicated RNG to make experiments reproducible."""
    return np.random.default_rng(seed)


def _negative_binomial_from_mean_dispersion(
    rng: np.random.Generator,
    mean: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """
    Draw counts from a Negative Binomial given mean and dispersion.

    Parameterization:
    - Var = mean + mean^2 / dispersion
    - dispersion > 0

    We map to NumPy NB parameters:
    - n = dispersion
    - p = n / (n + mean)
    """
    if dispersion <= 0:
        raise ValueError("sessions_dispersion must be > 0.")

    n = dispersion
    p = n / (n + mean)
    return rng.negative_binomial(n=n, p=p).astype(int)


def _lognormal_mixture_rev_per_session(
    rng: np.random.Generator,
    n_users: int,
    mu: float,
    sigma: float,
    high_share: float,
    high_multiplier: float,
    monetization_latent: np.ndarray,
) -> np.ndarray:
    """
    Generate revenue per session as a lognormal mixture with user heterogeneity.

    Steps:
    - Sample base rev/session from lognormal(mu, sigma).
    - Apply user latent monetization multiplier to create pre/post correlation.
    - With probability high_share, multiply by high_multiplier (high spenders).
    """
    if not (0 <= high_share < 1):
        raise ValueError("high_spender_share must be in [0, 1).")
    if sigma <= 0:
        raise ValueError("revps_lognorm_sigma must be > 0.")
    if high_multiplier <= 1:
        raise ValueError("high_revps_multiplier must be > 1.")

    base = rng.lognormal(mean=mu, sigma=sigma, size=n_users)
    is_high = rng.random(n_users) < high_share
    mix = base * np.where(is_high, high_multiplier, 1.0)
    return mix * monetization_latent


def _make_latent_multipliers(
    rng: np.random.Generator,
    n_users: int,
    activity_sigma: float,
    monetization_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create user-level multiplicative latent factors for activity and monetization.

    We use lognormal multipliers with mean ~ 1.
    """
    if activity_sigma < 0 or monetization_sigma < 0:
        raise ValueError("Latent sigmas must be >= 0.")

    act_mu = -0.5 * activity_sigma**2
    mon_mu = -0.5 * monetization_sigma**2

    activity_latent = rng.lognormal(mean=act_mu, sigma=activity_sigma, size=n_users)
    monetization_latent = rng.lognormal(mean=mon_mu, sigma=monetization_sigma, size=n_users)
    return activity_latent, monetization_latent


def generate_user_level_data(cfg: SyntheticB2CConfig) -> pd.DataFrame:
    """
    Generate a single synthetic dataset for an A/B experiment.

    Returns a DataFrame with per-user aggregates for pre and post periods.
    """
    if cfg.n_users <= 0:
        raise ValueError("n_users must be positive.")
    if not (0 < cfg.test_share < 1):
        raise ValueError("test_share must be in (0, 1).")

    rng = _rng(cfg.seed)
    n = cfg.n_users

    group = np.where(rng.random(n) < cfg.test_share, "test", "control")
    is_test = (group == "test")

    activity_latent, monetization_latent = _make_latent_multipliers(
        rng=rng,
        n_users=n,
        activity_sigma=cfg.activity_latent_sigma,
        monetization_sigma=cfg.monetization_latent_sigma,
    )

    mean_pre = cfg.sessions_mean_pre * activity_latent
    mean_post = cfg.sessions_mean_post * activity_latent

    sessions_pre = _negative_binomial_from_mean_dispersion(
        rng=rng, mean=mean_pre, dispersion=cfg.sessions_dispersion
    )
    sessions = _negative_binomial_from_mean_dispersion(
        rng=rng, mean=mean_post, dispersion=cfg.sessions_dispersion
    )

    revps = _lognormal_mixture_rev_per_session(
        rng=rng,
        n_users=n,
        mu=cfg.revps_lognorm_mu,
        sigma=cfg.revps_lognorm_sigma,
        high_share=cfg.high_spender_share,
        high_multiplier=cfg.high_revps_multiplier,
        monetization_latent=monetization_latent,
    )

    sessions_lift = 0.0
    revps_lift = 0.0

    if cfg.effect_type == "none":
        sessions_lift = 0.0
        revps_lift = 0.0
    elif cfg.effect_type == "activity":
        sessions_lift = cfg.lift_sessions
        revps_lift = 0.0
    elif cfg.effect_type == "monetization":
        sessions_lift = 0.0
        revps_lift = cfg.lift_rev_per_session
    elif cfg.effect_type == "mixed":
        sessions_lift = cfg.lift_sessions
        revps_lift = cfg.lift_rev_per_session
    else:
        raise ValueError("effect_type must be one of: none, activity, monetization, mixed.")

    sessions = (sessions.astype(float) * np.where(is_test, 1.0 + sessions_lift, 1.0)).round().astype(int)

    # Post-period revenue
    jitter = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    revps_post = revps * np.where(is_test, 1.0 + revps_lift, 1.0) * jitter
    revenue = sessions * revps_post

    # Pre-period revenue
    jitter_pre = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    revenue_pre = sessions_pre * (revps * jitter_pre)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1, dtype=int),
            "group": group.astype(str),
            "sessions_pre": sessions_pre.astype(int),
            "revenue_pre": revenue_pre.astype(float),
            "sessions": sessions.astype(int),
            "revenue": revenue.astype(float),
        }
    )

    if cfg.include_binary:
        if not (0 < cfg.conversion_rate_pre < 1):
            raise ValueError("conversion_rate_pre must be in (0, 1).")
        if not (0 < cfg.conversion_rate_post < 1):
            raise ValueError("conversion_rate_post must be in (0, 1).")

        conv_rate_post = cfg.conversion_rate_post + np.where(is_test, cfg.lift_conversion_rate, 0.0)
        conv_rate_post = np.clip(conv_rate_post, 1e-6, 1 - 1e-6)

        conversions_pre = rng.binomial(n=sessions_pre, p=cfg.conversion_rate_pre)
        conversions = rng.binomial(n=sessions, p=conv_rate_post)

        df["conversions_pre"] = conversions_pre.astype(int)
        df["conversions"] = conversions.astype(int)
        df["converted"] = (conversions > 0).astype(int)

    return df


def save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with a stable, UTF-8 encoding."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def make_default_scenarios(base_cfg: Optional[SyntheticB2CConfig] = None) -> dict[str, SyntheticB2CConfig]:
    """
    Create three neutral scenarios:
    - monetization: treatment affects revenue per session
    - activity: treatment affects sessions
    - mixed: treatment affects both slightly
    """
    base = base_cfg or SyntheticB2CConfig()
    base_dict = dict(base.__dict__)

    return {
        "scenario_monetization": SyntheticB2CConfig(**{**base_dict, "effect_type": "monetization", "lift_rev_per_session": 0.05, "lift_sessions": 0.00}),
        "scenario_activity": SyntheticB2CConfig(**{**base_dict, "effect_type": "activity", "lift_sessions": 0.07, "lift_rev_per_session": 0.00}),
        "scenario_mixed": SyntheticB2CConfig(**{**base_dict, "effect_type": "mixed", "lift_sessions": 0.03, "lift_rev_per_session": 0.03}),
    }
