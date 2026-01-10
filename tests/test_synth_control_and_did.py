import numpy as np
import pandas as pd

from tecore.causal import DataSpec, ImpactConfig, ImpactMethod, run_impact


def test_synthetic_control_runs():
    # Simple constructed dataset: y follows x closely, then intervention adds +5
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    x = np.linspace(100, 130, 120) + 3 * np.sin(2 * np.pi * np.arange(120) / 7)
    y = x.copy()
    intervention_date = "2025-03-15"
    post = dates >= pd.Timestamp(intervention_date)
    y[post] += 5.0

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "y": y,
        "x1": x,
    })

    spec = DataSpec(date_col="date", y_col="y", x_cols=["x1"], freq="D")
    cfg = ImpactConfig(
        intervention_date=intervention_date,
        method=ImpactMethod.SYNTHETIC_CONTROL,
        ridge_alpha=1.0,
        bootstrap_iters=100,
        run_placebo=False,
        pre_period_min_points=30,
    )

    res = run_impact(df, spec, cfg)
    assert res.cum_effect > 0.0


def test_did_runs():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    base = 100 + 0.1 * np.arange(100)
    control = base + np.sin(2 * np.pi * np.arange(100) / 7)
    treated = control.copy()

    intervention_date = "2025-03-10"
    post = dates >= pd.Timestamp(intervention_date)
    treated[post] += 3.0

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "y": treated,
        "control_series": control,
    })

    spec = DataSpec(date_col="date", y_col="y", x_cols=["control_series"], freq="D")
    cfg = ImpactConfig(
        intervention_date=intervention_date,
        method=ImpactMethod.DID,
        bootstrap_iters=80,
        block_size=7,
        run_placebo=False,
        pre_period_min_points=30,
    )

    res = run_impact(df, spec, cfg)
    assert res.cum_effect > 0.0
