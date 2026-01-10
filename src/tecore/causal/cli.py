from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import DataSpec, ImpactConfig, ImpactMethod
from .impact import run_impact
from .plotting import (
    plot_observed_vs_counterfactual,
    plot_point_effect,
    plot_cumulative_effect,
    plot_placebo_hist,
)


def add_causal_commands(app) -> None:
    """
    Register causal commands into an existing Typer app:
      tecore causal-impact ...
      tecore causal-impact-placebo ...

    If your repo doesn't use Typer, ignore this file and use scripts/.
    """
    try:
        import typer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Typer is required for CLI integration via tecore.causal.cli") from e

    @app.command("causal-impact")
    def causal_impact(
        data: Path = typer.Option(..., "--data", help="Path to CSV with time series"),
        y: str = typer.Option("y", "--y", help="Target column"),
        x: str = typer.Option("", "--x", help="Comma-separated covariate columns"),
        intervention: str = typer.Option(..., "--intervention", help="Intervention date YYYY-MM-DD"),
        freq: str = typer.Option("D", "--freq", help="Frequency D/W"),
        out: Path = typer.Option(Path("results/ci_run"), "--out", help="Output directory"),
        ridge_alpha: float = typer.Option(1.0, "--ridge-alpha", help="Ridge regularization"),
        bootstrap: int = typer.Option(200, "--bootstrap", help="Bootstrap iterations"),
        block_size: int = typer.Option(7, "--block-size", help="Block size (days)"),
        run_placebo: bool = typer.Option(True, "--run-placebo/--no-placebo", help="Run placebo-in-time"),
        n_placebos: int = typer.Option(20, "--n-placebos", help="Number of placebo dates"),
    ) -> None:
        df = pd.read_csv(data)
        x_cols = [c.strip() for c in x.split(",") if c.strip()]

        spec = DataSpec(date_col="date", y_col=y, x_cols=x_cols, freq=freq)
        cfg = ImpactConfig(
            intervention_date=intervention,
            method=ImpactMethod.CAUSAL_IMPACT_LIKE,
            ridge_alpha=ridge_alpha,
            bootstrap_iters=bootstrap,
            block_size=block_size,
            run_placebo=run_placebo,
            n_placebos=n_placebos,
        )

        res = run_impact(df, spec, cfg)

        out.mkdir(parents=True, exist_ok=True)
        (out / "plots").mkdir(parents=True, exist_ok=True)

        # save outputs
        pd.DataFrame(res.to_dict()).to_json(out / "results.json", indent=2)

        eff = res.effect_series.copy()
        eff["date"] = pd.to_datetime(eff["date"]).dt.strftime("%Y-%m-%d")
        eff.to_csv(out / "effect_series.csv", index=False)

        if res.placebo_results is not None:
            pl = res.placebo_results.copy()
            pl["placebo_date"] = pd.to_datetime(pl["placebo_date"]).dt.strftime("%Y-%m-%d")
            pl.to_csv(out / "placebo_results.csv", index=False)

        # plots
        plot_observed_vs_counterfactual(res.effect_series, out / "plots" / "observed_vs_counterfactual.png")
        plot_point_effect(res.effect_series, out / "plots" / "point_effect.png")
        plot_cumulative_effect(res.effect_series, out / "plots" / "cumulative_effect.png")
        if res.placebo_results is not None:
            plot_placebo_hist(res.placebo_results, res.cum_effect, out / "plots" / "placebo_hist.png")

    @app.command("causal-impact-placebo")
    def causal_impact_placebo(
        data: Path = typer.Option(..., "--data", help="Path to CSV"),
        y: str = typer.Option("y", "--y", help="Target column"),
        x: str = typer.Option("", "--x", help="Comma-separated covariates"),
        intervention: str = typer.Option(..., "--intervention", help="Intervention date"),
        freq: str = typer.Option("D", "--freq"),
        n_placebos: int = typer.Option(30, "--n-placebos"),
        out: Path = typer.Option(Path("results/ci_placebo"), "--out"),
    ) -> None:
        df = pd.read_csv(data)
        x_cols = [c.strip() for c in x.split(",") if c.strip()]

        spec = DataSpec(date_col="date", y_col=y, x_cols=x_cols, freq=freq)
        cfg = ImpactConfig(
            intervention_date=intervention,
            method=ImpactMethod.CAUSAL_IMPACT_LIKE,
            run_placebo=True,
            n_placebos=n_placebos,
        )
        res = run_impact(df, spec, cfg)
        out.mkdir(parents=True, exist_ok=True)
        if res.placebo_results is None:
            raise RuntimeError("Placebo results are empty.")
        pl = res.placebo_results.copy()
        pl["placebo_date"] = pd.to_datetime(pl["placebo_date"]).dt.strftime("%Y-%m-%d")
        pl.to_csv(out / "placebo_results.csv", index=False)
