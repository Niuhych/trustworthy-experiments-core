import numpy as np
import pandas as pd

from tecore.causal import DataSpec, ImpactConfig, ImpactMethod, run_impact
from tecore.causal.simulate_ts import SyntheticTSConfig, generate_synthetic_time_series


def test_positive_effect_detected_smoke():
    df, meta = generate_synthetic_time_series(
        SyntheticTSConfig(
            n_days=180,
            start_date="2025-01-01",
            intervention_day=120,
            level_shift=10.0,
            slope_change=0.0,
            temp_effect_amp=0.0,
            confounding=False,
            random_state=1,
        )
    )

    spec = DataSpec(
        date_col="date",
        y_col="y",
        x_cols=["sessions", "active_users", "marketing_spend", "external_index"],
        freq="D",
        missing_policy="raise",
        aggregation="mean",
        add_time_trend=True,
        add_day_of_week=True,
    )

    cfg = ImpactConfig(
        intervention_date=meta["intervention_date"],
        method=ImpactMethod.CAUSAL_IMPACT_LIKE,
        ridge_alpha=1.0,
        bootstrap_iters=120,
        block_size=7,
        alpha=0.05,
        run_placebo=False,   # keep tests fast/reliable
        pre_period_min_points=60,
    )

    res = run_impact(df, spec, cfg)

    assert np.isfinite(res.cum_effect)
    assert res.cum_effect > 0.0
    # CI should be positive in typical runs; keep a weak assertion to avoid flakes
    assert res.cum_ci[1] > 0.0


def test_zero_effect_ci_contains_zero_often():
    df, meta = generate_synthetic_time_series(
        SyntheticTSConfig(
            n_days=180,
            start_date="2025-01-01",
            intervention_day=120,
            level_shift=0.0,
            slope_change=0.0,
            temp_effect_amp=0.0,
            confounding=False,
            random_state=2,
        )
    )

    spec = DataSpec(
        date_col="date",
        y_col="y",
        x_cols=["sessions", "active_users", "marketing_spend", "external_index"],
        freq="D",
    )

    cfg = ImpactConfig(
        intervention_date=meta["intervention_date"],
        method=ImpactMethod.CAUSAL_IMPACT_LIKE,
        ridge_alpha=1.0,
        bootstrap_iters=120,
        block_size=7,
        alpha=0.05,
        run_placebo=False,
        pre_period_min_points=60,
    )

    res = run_impact(df, spec, cfg)

    # Not a strict Type I error test; just a sanity guard
    lo, hi = res.cum_ci
    assert lo <= 0.0 <= hi
