from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    return num / den.replace(0, np.nan)


def _linearize(num: pd.Series, den: pd.Series, r0: float) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return num - r0 * den


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


def cmd_cuped_ratio(args) -> int:
    # Backward-compat: if --out is set, ignore --out-json/--out-md
    if getattr(args, "out", None):
        if getattr(args, "out_json", None) or getattr(args, "out_md", None):
            _warn("`--out` is set; ignoring `--out-json` / `--out-md` (backward-compat).")

    df = pd.read_csv(args.input)

    required = {args.group_col, args.num, args.den, args.num_pre, args.den_pre}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute out_dir ONCE (critical for future default-out behavior)
    out_dir = prepare_out_dir(getattr(args, "out", None), command="cuped-ratio")

    # Optional audit on RAW input before transformations
    if getattr(args, "audit", False):
        if out_dir is not None:
            write_audit_bundle(out_dir, df=df, schema="b2c_ratio", parent_command="cuped-ratio")
        else:
            _warn("`--audit` is set but `--out` is not provided; skipping audit (nowhere to write).")

    df = df.copy()

    # Per-user ratio (for plot only)
    df["_ratio_post"] = _ratio(df[args.num], df[args.den])

    d_control = df[df[args.group_col] == args.control]
    d_test = df[df[args.group_col] == args.test]
    if len(d_control) == 0 or len(d_test) == 0:
        raise ValueError("Control/test group is empty. Check --group-col/--control/--test values.")

    # Control group ratio baselines
    den_sum_post_c = d_control[args.den].replace(0, np.nan).sum()
    den_sum_pre_c = d_control[args.den_pre].replace(0, np.nan).sum()
    if not np.isfinite(den_sum_post_c) or den_sum_post_c == 0:
        raise ValueError("Control denominator sum (post) is zero/NaN. Cannot compute r0_post.")
    if not np.isfinite(den_sum_pre_c) or den_sum_pre_c == 0:
        raise ValueError("Control denominator sum (pre) is zero/NaN. Cannot compute r0_pre.")

    r0_post = float(d_control[args.num].sum() / den_sum_post_c)
    r0_pre = float(d_control[args.num_pre].sum() / den_sum_pre_c)

    # Linearization
    df["_lin_post"] = _linearize(df[args.num], df[args.den], r0=r0_post)
    df["_lin_pre"] = _linearize(df[args.num_pre], df[args.den_pre], r0=r0_pre)

    lin_c = df.loc[df[args.group_col] == args.control, "_lin_post"].to_numpy(dtype=float)
    lin_t = df.loc[df[args.group_col] == args.test, "_lin_post"].to_numpy(dtype=float)
    base_lin = _welch_mean_diff(lin_c, lin_t, alpha=float(args.alpha))

    # Convert linearized diff -> ratio diff (approx): diff_lin / mean_den_control
    mean_den_c = float(d_control[args.den].replace(0, np.nan).mean())
    ratio_diff_base = float(base_lin["diff"] / mean_den_c) if (mean_den_c and np.isfinite(mean_den_c)) else np.nan

    # CUPED on linearized outcome with pre linearized covariate
    y_all = df["_lin_post"].to_numpy(dtype=float)
    x_all = df["_lin_pre"].to_numpy(dtype=float)
    y_adj_all, theta = _cuped_adjust(y_all, x_all)
    df["_lin_post_adj"] = y_adj_all

    lin_adj_c = df.loc[df[args.group_col] == args.control, "_lin_post_adj"].to_numpy(dtype=float)
    lin_adj_t = df.loc[df[args.group_col] == args.test, "_lin_post_adj"].to_numpy(dtype=float)
    cuped_lin = _welch_mean_diff(lin_adj_c, lin_adj_t, alpha=float(args.alpha))

    ratio_diff_cuped = float(cuped_lin["diff"] / mean_den_c) if (mean_den_c and np.isfinite(mean_den_c)) else np.nan

    control_ratio_post = float(d_control[args.num].sum() / den_sum_post_c)
    rel_lift_base = float(ratio_diff_base / control_ratio_post) if control_ratio_post != 0 else np.nan
    rel_lift_cuped = float(ratio_diff_cuped / control_ratio_post) if control_ratio_post != 0 else np.nan

    warnings: list[str] = []
    zero_den_share = float((pd.to_numeric(df[args.den], errors="coerce") == 0).mean())
    if zero_den_share > 0:
        warnings.append(
            f"Denominator has zeros (share={zero_den_share:.3g}). "
            "Per-user ratio uses NaN for zero denominators; inference is on linearized variable."
        )

    corr = float(pd.Series(df["_lin_post"]).corr(pd.Series(df["_lin_pre"])))
    if np.isfinite(corr) and abs(corr) < 0.05:
        warnings.append("Low corr(pre,post) on linearized variable. CUPED may give little variance reduction.")

    warnings_block = ("- " + "\n- ".join(warnings)) if warnings else "(none)"

    summary = pd.DataFrame(
        [
            {
                "method": "base_linearized",
                "diff_linearized": base_lin["diff"],
                "p_value": base_lin["p_value"],
                "ratio_diff_approx": ratio_diff_base,
                "rel_lift_approx": rel_lift_base,
            },
            {
                "method": "cuped_linearized",
                "diff_linearized": cuped_lin["diff"],
                "p_value": cuped_lin["p_value"],
                "ratio_diff_approx": ratio_diff_cuped,
                "rel_lift_approx": rel_lift_cuped,
            },
        ]
    )

    report = f"""# tecore cuped-ratio report

## Inputs
- input: `{args.input}`
- group_col: `{args.group_col}`
- control: `{args.control}`
- test: `{args.test}`
- num/den (post): `{args.num}` / `{args.den}`
- num/den (pre): `{args.num_pre}` / `{args.den_pre}`
- alpha: `{args.alpha}`

## Results (linearization, base vs CUPED)
- control_ratio_post (sum(num)/sum(den), control): {control_ratio_post:.6g}
- mean_den_control (per-unit mean den, control): {mean_den_c:.6g}

| method | diff_linearized | p_value | ratio_diff_approx | rel_lift_approx |
|---|---:|---:|---:|---:|
| base | {base_lin["diff"]:.6g} | {base_lin["p_value"]:.4g} | {ratio_diff_base:.6g} | {rel_lift_base:.6g} |
| cuped | {cuped_lin["diff"]:.6g} | {cuped_lin["p_value"]:.4g} | {ratio_diff_cuped:.6g} | {rel_lift_cuped:.6g} |

## Diagnostics
- r0_post (control): {r0_post:.6g}
- r0_pre (control): {r0_pre:.6g}
- theta: {theta:.6g}
- corr(lin_pre, lin_post): {corr:.6g}
- zero_den_share: {zero_den_share:.6g}

## Warnings
{warnings_block}
"""

    artifacts: dict[str, Any] = {"report_md": None, "plots": [], "tables": []}
    payload: dict[str, Any] = {
        "command": "cuped-ratio",
        "inputs": {
            "input": args.input,
            "group_col": args.group_col,
            "control": args.control,
            "test": args.test,
            "num": args.num,
            "den": args.den,
            "num_pre": args.num_pre,
            "den_pre": args.den_pre,
            "alpha": float(args.alpha),
        },
        "estimates": {
            "base": {
                "diff_linearized": base_lin["diff"],
                "p_value": base_lin["p_value"],
                "ratio_diff_approx": ratio_diff_base,
                "rel_lift_approx": rel_lift_base,
            },
            "cuped": {
                "diff_linearized": cuped_lin["diff"],
                "p_value": cuped_lin["p_value"],
                "ratio_diff_approx": ratio_diff_cuped,
                "rel_lift_approx": rel_lift_cuped,
            },
        },
        "diagnostics": {
            "r0_post_control": r0_post,
            "r0_pre_control": r0_pre,
            "theta": theta,
            "corr_pre_post_linearized": corr,
            "zero_den_share": zero_den_share,
            "mean_den_control": mean_den_c,
        },
        "warnings": warnings,
        "artifacts": artifacts,
    }

    if out_dir is not None:
        write_run_meta(out_dir, vars(args), extra={"command": "cuped-ratio"})
        artifacts["tables"].append(write_table(out_dir, "summary", summary))

        fig1 = _plot_hist_by_group(
            df, args.group_col, args.control, args.test, "_ratio_post", title="Post ratio distribution by group (per-unit)"
        )
        artifacts["plots"].append(save_plot(out_dir, "ratio_post_by_group", fig=fig1))

        fig2 = _plot_hist_by_group(
            df, args.group_col, args.control, args.test, "_lin_post", title="Linearized post variable by group"
        )
        artifacts["plots"].append(save_plot(out_dir, "linearized_by_group", fig=fig2))

        artifacts["report_md"] = "report.md"
        payload["artifacts"] = artifacts
        write_results_json(out_dir, payload)
        write_report_md(out_dir, report)
        return 0

    # Legacy outputs (no --out)
    if getattr(args, "out_json", None):
        import json

        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if getattr(args, "out_md", None):
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(report, encoding="utf-8")

    return 0
