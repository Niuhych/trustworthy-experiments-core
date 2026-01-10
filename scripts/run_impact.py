from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tecore.causal import DataSpec, ImpactConfig, ImpactMethod, run_impact
from tecore.causal.plotting import (
    plot_observed_vs_counterfactual,
    plot_point_effect,
    plot_cumulative_effect,
    plot_placebo_hist,
)
from tecore.causal.sensitivity import (
    leave_one_covariate_out_test,
    drop_last_k_days_of_pre_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run causal impact / synthetic control / DiD on a time series CSV.")
    p.add_argument("--input", required=True, type=str, help="Path to input CSV")
    p.add_argument("--out", required=True, type=str, help="Output directory")

    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--y", required=True, type=str, help="Target column name")
    p.add_argument("--x", type=str, default="", help="Comma-separated covariate columns")

    p.add_argument("--intervention", required=True, type=str, help="Intervention date YYYY-MM-DD")
    p.add_argument("--freq", type=str, default="D", help="Frequency D or W")

    p.add_argument("--method", type=str, default="causal_impact_like",
                   choices=["causal_impact_like", "synthetic_control", "did"])

    p.add_argument("--missing-policy", type=str, default="raise",
                   choices=["raise", "ffill", "bfill", "zero"])
    p.add_argument("--aggregation", type=str, default="mean", choices=["mean", "sum"])

    # model config
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--block-size", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.05)

    # placebo
    p.add_argument("--run-placebo", action="store_true", help="Run placebo-in-time tests")
    p.add_argument("--n-placebos", type=int, default=20)

    # economics
    p.add_argument("--margin-rate", type=float, default=None)
    p.add_argument("--spend-col", type=str, default=None)
    p.add_argument("--incremental-cost", type=float, default=None)

    # sensitivity
    p.add_argument("--run-sensitivity", action="store_true", help="Run basic sensitivity checks")
    return p.parse_args()


def _split_cols(s: str) -> List[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def main() -> None:
    a = parse_args()
    inp = Path(a.input)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    x_cols = _split_cols(a.x)

    spec = DataSpec(
        date_col=a.date_col,
        y_col=a.y,
        x_cols=x_cols,
        freq=a.freq,
        missing_policy=a.missing_policy,
        aggregation=a.aggregation,
        add_time_trend=True,
        add_day_of_week=True,
        winsorize_p=None,
    )

    cfg = ImpactConfig(
        intervention_date=a.intervention,
        method=ImpactMethod(a.method),
        alpha=a.alpha,
        ridge_alpha=a.ridge_alpha,
        bootstrap_iters=a.bootstrap,
        block_size=a.block_size,
        run_placebo=bool(a.run_placebo),
        n_placebos=a.n_placebos,
        margin_rate=a.margin_rate,
        spend_col=a.spend_col,
        incremental_cost=a.incremental_cost,
    )

    res = run_impact(df, spec, cfg)

    # save json summary
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(res.summary(), f, indent=2, ensure_ascii=False)

    # save effect series
    eff = res.effect_series.copy()
    eff["date"] = pd.to_datetime(eff["date"]).dt.strftime("%Y-%m-%d")
    eff.to_csv(out / "effect_series.csv", index=False)

    # placebo
    if res.placebo_results is not None:
        pl = res.placebo_results.copy()
        pl["placebo_date"] = pd.to_datetime(pl["placebo_date"]).dt.strftime("%Y-%m-%d")
        pl.to_csv(out / "placebo_results.csv", index=False)

    # economics
    if res.economics is not None:
        with open(out / "economics.json", "w", encoding="utf-8") as f:
            json.dump(res.economics, f, indent=2, ensure_ascii=False)

    # plots
    plot_observed_vs_counterfactual(res.effect_series, out / "plots" / "observed_vs_counterfactual.png")
    plot_point_effect(res.effect_series, out / "plots" / "point_effect.png")
    plot_cumulative_effect(res.effect_series, out / "plots" / "cumulative_effect.png")
    if res.placebo_results is not None:
        plot_placebo_hist(res.placebo_results, res.cum_effect, out / "plots" / "placebo_hist.png")

    # sensitivity checks
    if a.run_sensitivity and cfg.method == ImpactMethod.CAUSAL_IMPACT_LIKE and len(x_cols) > 0:
        loo = leave_one_covariate_out_test(df, spec, cfg)
        loo.to_csv(out / "sensitivity_leave_one_out.csv", index=False)

        stab = drop_last_k_days_of_pre_test(df, spec, cfg, k_list=[7, 14, 21])
        stab.to_csv(out / "sensitivity_drop_last_pre.csv", index=False)

    # console
    print("=== Impact summary ===")
    print(json.dumps(res.summary(), indent=2, ensure_ascii=False))
    if res.warnings:
        print("\nWarnings:")
        for w in res.warnings:
            print("-", w)
    print(f"\nWrote outputs to: {out}")


if __name__ == "__main__":
    main()
