from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tecore.cli.audit_api import write_audit_bundle
from tecore.cli.bundle import prepare_out_dir, save_plot, write_report_md, write_results_json, write_run_meta, write_table


def _warn(msg: str) -> None:
    print(f"[tecore][warn] {msg}", file=sys.stderr)


def _parse_x_list(x: str | None) -> list[str]:
    if x is None:
        return []
    x = str(x).strip()
    if not x:
        return []
    # allow: "a,b,c" or "a, b, c"
    return [p.strip() for p in x.split(",") if p.strip()]


def _to_payload(obj: Any) -> Any:
    # best-effort serializable
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_payload(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _to_payload(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _to_payload(vars(obj))
    return str(obj)


def _plot_observed_vs_cf(effect_df: pd.DataFrame, title: str = "Observed vs Counterfactual"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["y"], label="observed")
    plt.plot(effect_df["date"], effect_df["y_cf"], label="counterfactual")
    plt.axvline(effect_df["date"].iloc[effect_df["is_post"].to_numpy().argmax()], linestyle="--", label="intervention")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("y")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    return fig


def _plot_point_effect(effect_df: pd.DataFrame, title: str = "Point effect"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["point_effect"])
    plt.axhline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("y - y_cf")
    plt.xticks(rotation=30, ha="right")
    return fig


def _plot_cum_effect(effect_df: pd.DataFrame, title: str = "Cumulative effect"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["cum_effect"])
    plt.axhline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("cumulative (y - y_cf)")
    plt.xticks(rotation=30, ha="right")
    return fig


def cmd_causal_impact(args) -> int:
    """
    MVP causal-impact CLI.

    Requires internal causal module: tecore.causal.impact.run_impact
    Expected interface (MVP):
      res = run_impact(df, spec=..., cfg=...)
      res.effect_df -> DataFrame with columns: date, y, y_cf, point_effect, cum_effect, is_post
      res.summary -> dict-like (or dataclass) with key estimates
    """
    # Backward-compat note: no legacy out-json/out-md for causal (new command)
    out_dir = prepare_out_dir(getattr(args, "out", None), command="causal-impact")

    df = pd.read_csv(args.input)

    date_col = str(getattr(args, "date_col", "date"))
    y_col = str(getattr(args, "y", "y"))
    x_cols = _parse_x_list(getattr(args, "x", None))

    required = [date_col, y_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError(f"Failed to parse some dates in column `{date_col}`.")
    df = df.sort_values(date_col)

    intervention = str(getattr(args, "intervention", "")).strip()
    if not intervention:
        raise ValueError("`--intervention` is required (YYYY-MM-DD).")
    intervention_dt = pd.to_datetime(intervention)

    if getattr(args, "audit", False):
        write_audit_bundle(out_dir, df=df, schema="timeseries_causal_impact", parent_command="causal-impact")

    try:
        from tecore.causal.impact import ImpactConfig, DataSpec, run_impact  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Causal module not available. Expected tecore.causal.impact with ImpactConfig/DataSpec/run_impact. "
            f"Import error: {e}"
        )

    spec = DataSpec(
        date_col=date_col,
        y=y_col,
        x=x_cols,
        intervention=intervention_dt,
    )

    cfg = ImpactConfig(
        alpha=float(getattr(args, "alpha", 0.05)),
        bootstrap_iters=int(getattr(args, "bootstrap_iters", 200)),
        n_placebos=int(getattr(args, "n_placebos", 0)),
        seed=int(getattr(args, "seed", 42)),
    )

    res = run_impact(df, spec=spec, cfg=cfg)

    effect_df = getattr(res, "effect_df", None)
    if effect_df is None or not isinstance(effect_df, pd.DataFrame):
        raise RuntimeError("run_impact did not return `effect_df` as a pandas DataFrame.")

    ren = {}
    if "ds" in effect_df.columns and "date" not in effect_df.columns:
        ren["ds"] = "date"
    effect_df = effect_df.rename(columns=ren)

    for c in ["date", "y", "y_cf", "point_effect", "cum_effect", "is_post"]:
        if c not in effect_df.columns:
            raise RuntimeError(f"effect_df missing required column `{c}`.")

    effect_export = effect_df.copy()
    effect_export["date"] = pd.to_datetime(effect_export["date"]).dt.strftime("%Y-%m-%d")
    effect_rel = write_table(out_dir, "effect_series", effect_export)

    artifacts: dict[str, Any] = {"report_md": "report.md", "plots": [], "tables": [effect_rel]}
    fig1 = _plot_observed_vs_cf(effect_df)
    artifacts["plots"].append(save_plot(out_dir, "observed_vs_counterfactual", fig=fig1))
    fig2 = _plot_point_effect(effect_df)
    artifacts["plots"].append(save_plot(out_dir, "point_effect", fig=fig2))
    fig3 = _plot_cum_effect(effect_df)
    artifacts["plots"].append(save_plot(out_dir, "cumulative_effect", fig=fig3))

    placebo_df = getattr(res, "placebo_df", None)
    if isinstance(placebo_df, pd.DataFrame) and "placebo_p_value" in placebo_df.columns:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.hist(placebo_df["placebo_p_value"].dropna().to_numpy(), bins=20, density=False)
        plt.title("Placebo p-values")
        plt.xlabel("p_value")
        plt.ylabel("count")
        artifacts["plots"].append(save_plot(out_dir, "placebo_hist", fig=fig))

    summary = getattr(res, "summary", None)
    summary_payload = _to_payload(summary)

    write_run_meta(out_dir, vars(args), extra={"command": "causal-impact"})

    payload: dict[str, Any] = {
        "command": "causal-impact",
        "inputs": {
            "input": args.input,
            "schema": str(getattr(args, "schema", "timeseries_causal_impact")),
            "date_col": date_col,
            "y": y_col,
            "x": x_cols,
            "intervention": str(intervention_dt.date()),
            "alpha": float(getattr(args, "alpha", 0.05)),
            "bootstrap_iters": int(getattr(args, "bootstrap_iters", 200)),
            "n_placebos": int(getattr(args, "n_placebos", 0)),
            "seed": int(getattr(args, "seed", 42)),
        },
        "estimates": {"summary": summary_payload},
        "diagnostics": {},
        "warnings": [],
        "artifacts": artifacts,
    }
    write_results_json(out_dir, payload)

    report = f"""# tecore causal-impact report

## Inputs
- input: `{args.input}`
- date_col: `{date_col}`
- y: `{y_col}`
- x: `{", ".join(x_cols) if x_cols else "(none)" }`
- intervention: `{str(intervention_dt.date())}`
- alpha: `{float(getattr(args, "alpha", 0.05))}`
- bootstrap_iters: `{int(getattr(args, "bootstrap_iters", 200))}`
- n_placebos: `{int(getattr(args, "n_placebos", 0))}`
- seed: `{int(getattr(args, "seed", 42))}`

## Outputs
- tables/effect_series.csv
- plots/observed_vs_counterfactual.png
- plots/point_effect.png
- plots/cumulative_effect.png
"""

    write_report_md(out_dir, report)

    return 0
