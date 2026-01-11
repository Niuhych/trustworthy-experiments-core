from __future__ import annotations

import inspect
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tecore.cli.audit_api import write_audit_bundle
from tecore.cli.bundle import (
    prepare_out_dir,
    save_plot,
    write_report_md,
    write_results_json,
    write_run_meta,
    write_table,
)


def _warn(msg: str) -> None:
    print(f"[tecore][warn] {msg}", file=sys.stderr)


def _parse_x_list(x: str | None) -> list[str]:
    if x is None:
        return []
    x = str(x).strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def _to_payload(obj: Any) -> Any:
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


def _plot_observed_vs_cf(effect_df: pd.DataFrame, intervention_date: str, title: str = "Observed vs Counterfactual"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["y"], label="observed")
    plt.plot(effect_df["date"], effect_df["y_cf"], label="counterfactual")
    plt.axvline(intervention_date, linestyle="--", label="intervention")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("y")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    return fig


def _plot_point_effect(effect_df: pd.DataFrame, intervention_date: str, title: str = "Point effect"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["point_effect"])
    plt.axvline(intervention_date, linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("y - y_cf")
    plt.xticks(rotation=30, ha="right")
    return fig


def _plot_cum_effect(effect_df: pd.DataFrame, intervention_date: str, title: str = "Cumulative effect"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(effect_df["date"], effect_df["cum_effect"])
    plt.axvline(intervention_date, linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("cumulative (y - y_cf)")
    plt.xticks(rotation=30, ha="right")
    return fig


def _accepted_params(cls) -> set[str]:
    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys())
    params.discard("self")
    return params


def _build_dataspec(DataSpec, *, date_col: str, y_col: str, x_cols: list[str], intervention_dt: pd.Timestamp, df: pd.DataFrame):
    """
    Construct DataSpec using introspection + a mapping of common parameter names.
    This avoids breakage if internal tecore.causal.impact.DataSpec evolves.
    """
    p = _accepted_params(DataSpec)

    pre_start = pd.to_datetime(df[date_col].min())
    pre_end = pd.to_datetime(intervention_dt) - pd.Timedelta(days=1)
    post_start = pd.to_datetime(intervention_dt)
    post_end = pd.to_datetime(df[date_col].max())

    kwargs: dict[str, Any] = {}

    for name in ["date_col", "date", "ds", "time_col", "t_col"]:
        if name in p:
            kwargs[name] = date_col
            break

    for name in ["y_col", "y", "outcome", "target", "metric", "outcome_col"]:
        if name in p:
            kwargs[name] = y_col
            break

    for name in ["x", "x_cols", "covariates", "features", "controls", "regressors", "x_col_list"]:
        if name in p:
            kwargs[name] = x_cols
            break

    for name in ["intervention", "intervention_date", "t0", "post_start", "start_post", "cutoff", "treatment_start"]:
        if name in p:
            kwargs[name] = intervention_dt
            break

    if "pre_period" in p:
        kwargs["pre_period"] = (pre_start, pre_end)
    if "post_period" in p:
        kwargs["post_period"] = (post_start, post_end)

    if "pre_start" in p:
        kwargs["pre_start"] = pre_start
    if "pre_end" in p:
        kwargs["pre_end"] = pre_end
    if "post_start" in p and "post_start" not in kwargs:
        kwargs["post_start"] = post_start
    if "post_end" in p:
        kwargs["post_end"] = post_end

    return DataSpec(**kwargs)


def _build_config(ImpactConfig, *, intervention_dt: pd.Timestamp, alpha: float, bootstrap_iters: int, n_placebos: int, seed: int,):
    """
    Construct ImpactConfig using introspection + mapping for common names.
    """
    p = _accepted_params(ImpactConfig)
    kwargs: dict[str, Any] = {}

    for name in ["intervention_date", "intervention", "t0", "cutoff", "treatment_start", "post_start"]:
        if name in p:
            kwargs[name] = intervention_dt
            break

    for name in ["alpha", "significance", "p_alpha"]:
        if name in p:
            kwargs[name] = float(alpha)
            break

    for name in ["bootstrap_iters", "n_bootstrap", "num_bootstrap", "boot_iters", "n_boot_iters"]:
        if name in p:
            kwargs[name] = int(bootstrap_iters)
            break

    for name in ["n_placebos", "num_placebos", "placebo_iters", "placebos"]:
        if name in p:
            kwargs[name] = int(n_placebos)
            break

    for name in ["seed", "random_state", "rng_seed"]:
        if name in p:
            kwargs[name] = int(seed)
            break

    return ImpactConfig(**kwargs)


def _normalize_effect_df(effect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a variety of column naming conventions from internal run_impact
    and normalize to:
      date, y, y_cf, point_effect, cum_effect, is_post
    """
    df = effect_df.copy()

    if "date" not in df.columns:
        for c in ["ds", "time", "t", "timestamp"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "y" not in df.columns:
        for c in ["observed", "actual", "y_obs"]:
            if c in df.columns:
                df = df.rename(columns={c: "y"})
                break

    if "y_cf" not in df.columns:
        for c in ["counterfactual", "y0", "y_hat", "pred", "prediction", "y_pred"]:
            if c in df.columns:
                df = df.rename(columns={c: "y_cf"})
                break

    if "point_effect" not in df.columns:
        for c in ["effect", "tau", "point_tau", "diff", "y_minus_ycf"]:
            if c in df.columns:
                df = df.rename(columns={c: "point_effect"})
                break

    if "cum_effect" not in df.columns:
        for c in ["cumulative_effect", "cum_tau", "cumsum_effect", "cum_diff"]:
            if c in df.columns:
                df = df.rename(columns={c: "cum_effect"})
                break

    if "is_post" not in df.columns:
        for c in ["post", "in_post", "is_post_period"]:
            if c in df.columns:
                df = df.rename(columns={c: "is_post"})
                break

    if "is_post" not in df.columns and "date" in df.columns:
        if "y_cf" in df.columns:
            df["is_post"] = df["y_cf"].notna()
        else:
            df["is_post"] = False

    needed = ["date", "y", "y_cf", "point_effect", "cum_effect", "is_post"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise RuntimeError(f"effect_df missing required columns after normalization: {miss}")

    return df


def cmd_causal_impact(args) -> int:
    out_dir = prepare_out_dir(getattr(args, "out", None), command="causal-impact")

    df = pd.read_csv(args.input)

    date_col = str(getattr(args, "date_col", "date"))
    y_col = str(getattr(args, "y", "y"))
    x_cols = _parse_x_list(getattr(args, "x", None))

    missing = [c for c in [date_col, y_col] if c not in df.columns]
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
        from tecore.causal.impact import ImpactConfig, DataSpec, run_impact 
    except Exception as e:
        raise RuntimeError(
            "Causal module import failed. Expected tecore.causal.impact with ImpactConfig/DataSpec/run_impact. "
            f"Import error: {e}"
        )

    spec = _build_dataspec(
        DataSpec,
        date_col=date_col,
        y_col=y_col,
        x_cols=x_cols,
        intervention_dt=intervention_dt,
        df=df,
    )

    cfg = _build_config(
        ImpactConfig,
        intervention_dt=intervention_dt,
        alpha=float(getattr(args, "alpha", 0.05)),
        bootstrap_iters=int(getattr(args, "bootstrap_iters", 200)),
        n_placebos=int(getattr(args, "n_placebos", 0)),
        seed=int(getattr(args, "seed", 42)),
    )

    import inspect as _inspect
    
    call_sig = _inspect.signature(run_impact)
    call_params = set(call_sig.parameters.keys())
    
    kwargs = {}
    if "spec" in call_params:
        kwargs["spec"] = spec
    elif "data_spec" in call_params:
        kwargs["data_spec"] = spec
    
    if "cfg" in call_params:
        kwargs["cfg"] = cfg
    elif "config" in call_params:
        kwargs["config"] = cfg
    
    res = run_impact(df, **kwargs)

    effect_df = getattr(res, "effect_df", None)
    if effect_df is None:
        # allow dict-like return
        if isinstance(res, dict) and "effect_df" in res:
            effect_df = res["effect_df"]
    if not isinstance(effect_df, pd.DataFrame):
        raise RuntimeError("run_impact did not return `effect_df` as a pandas DataFrame (field `effect_df`).")

    effect_df = _normalize_effect_df(effect_df)

    effect_export = effect_df.copy()
    effect_export["date"] = pd.to_datetime(effect_export["date"]).dt.strftime("%Y-%m-%d")
    effect_rel = write_table(out_dir, "effect_series", effect_export)

    artifacts: dict[str, Any] = {"report_md": "report.md", "plots": [], "tables": [effect_rel]}

    intervention_str = pd.to_datetime(intervention_dt).strftime("%Y-%m-%d")

    fig1 = _plot_observed_vs_cf(effect_df, intervention_str)
    artifacts["plots"].append(save_plot(out_dir, "observed_vs_counterfactual", fig=fig1))

    fig2 = _plot_point_effect(effect_df, intervention_str)
    artifacts["plots"].append(save_plot(out_dir, "point_effect", fig=fig2))

    fig3 = _plot_cum_effect(effect_df, intervention_str)
    artifacts["plots"].append(save_plot(out_dir, "cumulative_effect", fig=fig3))

    summary = getattr(res, "summary", None)
    if summary is None and isinstance(res, dict):
        summary = res.get("summary")
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
            "intervention": intervention_str,
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
- intervention: `{intervention_str}`
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
