from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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


def _apply_transform(s: pd.Series, transform: str, winsor_q: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if transform == "raw":
        return s
    if transform == "winsor":
        q = float(winsor_q)
        if not (0.5 < q < 1.0):
            raise ValueError("--winsor-q must be in (0.5, 1.0)")
        cap = s.quantile(q)
        return s.clip(upper=cap)
    if transform == "log1p":
        return np.log1p(s.clip(lower=0))
    raise ValueError(f"Unknown transform: {transform}")


def _welch_mean_diff(a: np.ndarray, b: np.ndarray, alpha: float) -> dict[str, Any]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Not enough observations per group for inference (need >=2).")

    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = np.sqrt(v1 / n1 + v2 / n2)

    if se == 0:
        t_stat, p = 0.0, 1.0
        ci_low, ci_high = (m2 - m1), (m2 - m1)
        df = float(n1 + n2 - 2)
    else:
        # Welch–Satterthwaite df
        df_num = (v1 / n1 + v2 / n2) ** 2
        df_den = (v1**2) / (n1**2 * (n1 - 1)) + (v2**2) / (n2**2 * (n2 - 1))
        df = float(df_num / df_den) if df_den > 0 else float(n1 + n2 - 2)

        diff = (m2 - m1)
        t_stat = float(diff / se)
        p = float(2 * stats.t.sf(np.abs(t_stat), df=df))
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=df))
        ci_low = float(diff - tcrit * se)
        ci_high = float(diff + tcrit * se)

    return {
        "n_control": int(n1),
        "n_test": int(n2),
        "mean_control": m1,
        "mean_test": m2,
        "diff": float(m2 - m1),
        "p_value": p,
        "t_stat": t_stat,
        "df": df,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    y = y.astype(float)
    x = x.astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    y2, x2 = y[mask], x[mask]
    if len(y2) < 3:
        return y, 0.0

    vx = float(np.var(x2, ddof=1))
    if vx <= 0:
        return y, 0.0

    cov = float(np.cov(y2, x2, ddof=1)[0, 1])
    theta = cov / vx
    x_mean = float(np.mean(x2))
    y_adj = y - theta * (x - x_mean)
    return y_adj, float(theta)


def _plot_hist_by_group(df: pd.DataFrame, group_col: str, control: str, test: str, col: str, title: str):
    import matplotlib.pyplot as plt

    a = df.loc[df[group_col] == control, col].dropna().to_numpy()
    b = df.loc[df[group_col] == test, col].dropna().to_numpy()

    fig = plt.figure()
    plt.hist(a, bins=40, density=True, alpha=0.6, label=control)
    plt.hist(b, bins=40, density=True, alpha=0.6, label=test)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("density")
    plt.legend()
    return fig


def _plot_scatter(df: pd.DataFrame, group_col: str, control: str, test: str, x: str, y: str, title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    d0 = df[df[group_col] == control]
    d1 = df[df[group_col] == test]
    plt.scatter(d0[x], d0[y], s=10, alpha=0.6, label=control)
    plt.scatter(d1[x], d1[y], s=10, alpha=0.6, label=test)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    return fig


def cmd_cuped(args) -> int:
    # Backward-compat: if --out is set, ignore --out-json/--out-md
    if getattr(args, "out", None):
        if getattr(args, "out_json", None) or getattr(args, "out_md", None):
            _warn("`--out` is set; ignoring `--out-json` / `--out-md` (backward-compat).")

    df = pd.read_csv(args.input)

    required = {args.group_col, args.y, args.x}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Transform y/x
    df = df.copy()
    df[args.y] = _apply_transform(df[args.y], args.transform, args.winsor_q)
    df[args.x] = _apply_transform(df[args.x], "raw", args.winsor_q)  # covariate: keep raw by default

    d_control = df[df[args.group_col] == args.control]
    d_test = df[df[args.group_col] == args.test]
    if len(d_control) == 0 or len(d_test) == 0:
        raise ValueError("Control/test group is empty. Check --group-col/--control/--test values.")

    y_c = d_control[args.y].to_numpy(dtype=float)
    y_t = d_test[args.y].to_numpy(dtype=float)

    base = _welch_mean_diff(y_c, y_t, alpha=float(args.alpha))

    # CUPED adjusted
    y_all = df[args.y].to_numpy(dtype=float)
    x_all = df[args.x].to_numpy(dtype=float)
    y_adj_all, theta = _cuped_adjust(y_all, x_all)
    df["_y_adj"] = y_adj_all

    y_adj_c = df.loc[df[args.group_col] == args.control, "_y_adj"].to_numpy(dtype=float)
    y_adj_t = df.loc[df[args.group_col] == args.test, "_y_adj"].to_numpy(dtype=float)
    cuped = _welch_mean_diff(y_adj_c, y_adj_t, alpha=float(args.alpha))

    var_raw = float(np.nanvar(y_all, ddof=1))
    var_adj = float(np.nanvar(y_adj_all, ddof=1))
    vr = (var_adj / var_raw) if (var_raw > 0) else None

    warnings: list[str] = []
    corr = float(pd.Series(df[args.y]).corr(pd.Series(df[args.x])))
    if np.isfinite(corr) and abs(corr) < 0.05:
        warnings.append("Low corr(x,y). CUPED may give little variance reduction on this dataset.")

    warnings_block = ("- " + "\n- ".join(warnings)) if warnings else "(none)"

    out_dir = prepare_out_dir(getattr(args, "out", None), command="cuped")
    artifacts: dict[str, Any] = {"report_md": None, "plots": [], "tables": []}

    summary = pd.DataFrame(
        [
            {
                "method": "base",
                **{k: base[k] for k in ["n_control", "n_test", "mean_control", "mean_test", "diff", "p_value", "ci_low", "ci_high"]},
            },
            {
                "method": "cuped",
                **{k: cuped[k] for k in ["n_control", "n_test", "mean_control", "mean_test", "diff", "p_value", "ci_low", "ci_high"]},
            },
        ]
    )

    report = f"""# tecore cuped report

## Inputs
- input: `{args.input}`
- group_col: `{args.group_col}`
- control: `{args.control}`
- test: `{args.test}`
- y (post): `{args.y}`
- x (pre): `{args.x}`
- alpha: `{args.alpha}`
- transform(y): `{args.transform}`
- winsor_q: `{args.winsor_q}`

## Results (base vs CUPED)
| method | n_control | n_test | mean_control | mean_test | diff (test-control) | p_value | ci_low | ci_high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | {base["n_control"]} | {base["n_test"]} | {base["mean_control"]:.6g} | {base["mean_test"]:.6g} | {base["diff"]:.6g} | {base["p_value"]:.4g} | {base["ci_low"]:.6g} | {base["ci_high"]:.6g} |
| cuped | {cuped["n_control"]} | {cuped["n_test"]} | {cuped["mean_control"]:.6g} | {cuped["mean_test"]:.6g} | {cuped["diff"]:.6g} | {cuped["p_value"]:.4g} | {cuped["ci_low"]:.6g} | {cuped["ci_high"]:.6g} |

## Diagnostics
- theta: {theta:.6g}
- corr(x,y): {corr:.6g}
- var_ratio (adj/raw): {vr if vr is not None else "NA"}

## Warnings
{warnings_block}
"""

    payload: dict[str, Any] = {
        "command": "cuped",
        "inputs": {
            "input": args.input,
            "group_col": args.group_col,
            "control": args.control,
            "test": args.test,
            "y": args.y,
            "x": args.x,
            "alpha": float(args.alpha),
            "transform": args.transform,
            "winsor_q": float(args.winsor_q),
        },
        "estimates": {"base": base, "cuped": cuped},
        "diagnostics": {"theta": theta, "corr_xy": corr, "var_ratio_adj_over_raw": vr},
        "warnings": warnings,
        "artifacts": artifacts,
    }

    if out_dir is not None:
        write_run_meta(out_dir, vars(args), extra={"command": "cuped"})
        artifacts["tables"].append(write_table(out_dir, "summary", summary))

        fig1 = _plot_hist_by_group(
            df, args.group_col, args.control, args.test, args.y, title="Post metric distribution by group"
        )
        artifacts["plots"].append(save_plot(out_dir, "y_post_by_group", fig=fig1))

        fig2 = _plot_scatter(
            df, args.group_col, args.control, args.test, x=args.x, y=args.y, title="Pre covariate vs post metric (sanity for CUPED)"
        )
        artifacts["plots"].append(save_plot(out_dir, "x_vs_y_scatter", fig=fig2))

        artifacts["report_md"] = "report.md"
        payload["artifacts"] = artifacts
        write_results_json(out_dir, payload)
        write_report_md(out_dir, report)
        return 0

    if getattr(args, "out_json", None):
        import json

        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if getattr(args, "out_md", None):
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(report, encoding="utf-8")

    return 0
