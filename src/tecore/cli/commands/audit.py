from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tecore.cli.bundle import (
    ensure_subdirs,
    prepare_out_dir,
    write_audit_json,
    write_audit_md,
    write_results_json,
    write_run_meta,
    write_report_md,
    write_table,
)


def _warn(msg: str) -> None:
    print(f"[tecore][warn] {msg}", file=sys.stderr)


def _infer_group_col(df: pd.DataFrame, preferred: str = "group") -> str | None:
    if preferred in df.columns:
        return preferred
    # very light heuristic
    for c in ["variant", "arm", "bucket", "treatment", "ab_group", "grp"]:
        if c in df.columns:
            return c
    return None


def _basic_numeric_cols(df: pd.DataFrame) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude internal cols if any
    return [c for c in num if not c.startswith("_")]


def _make_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        dtype = str(s.dtype)
        nunique = int(s.nunique(dropna=True))
        example = None
        try:
            example = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else None
        except Exception:
            example = None

        rows.append(
            {
                "column": c,
                "dtype": dtype,
                "n_rows": n,
                "n_missing": miss,
                "missing_share": float(miss / n) if n > 0 else np.nan,
                "n_unique": nunique,
                "example": str(example) if example is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _group_balance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    vc = df[group_col].value_counts(dropna=False).rename_axis(group_col).reset_index(name="n")
    total = int(vc["n"].sum()) if len(vc) else 0
    vc["share"] = vc["n"] / total if total > 0 else np.nan
    return vc


def _detect_user_id_col(df: pd.DataFrame) -> str | None:
    for c in ["user_id", "uid", "customer_id", "client_id", "id"]:
        if c in df.columns:
            return c
    return None


def _denominator_like_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["session", "den", "denom", "impression", "click", "visit"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def cmd_audit(args) -> int:
    df = pd.read_csv(args.input)

    schema = str(getattr(args, "schema", "b2c_user_level"))

    out_dir = prepare_out_dir(getattr(args, "out", None), command="audit")
    if out_dir is None:
        raise ValueError("`tecore audit` requires --out (bundle directory).")

    ensure_subdirs(out_dir, ["tables", "plots"])

    schema = str(getattr(args, "schema", "b2c_user_level"))
    out_dir = prepare_out_dir(getattr(args, "out", None), command="audit")
    if out_dir is None:
        raise ValueError("`tecore audit` requires --out (bundle directory).")

    # ---- Core checks (MVP) ----
    warnings: list[str] = []
    errors: list[str] = []

    group_col = _infer_group_col(df, preferred="group")
    if group_col is None:
        warnings.append("No obvious group column found (expected `group`). Group balance checks skipped.")
    else:
        # sanity: at least 2 groups
        if df[group_col].nunique(dropna=True) < 2:
            warnings.append(f"Group column `{group_col}` has <2 unique values. Experiment comparison may be impossible.")

    # schema-specific light checks (MVP)
    if schema not in {"b2c_user_level", "b2c_ratio", "timeseries_causal_impact"}:
        warnings.append(f"Unknown schema `{schema}`. Running generic audit only.")

    # user_id checks (optional)
    user_id_col = _detect_user_id_col(df)
    if user_id_col is None:
        warnings.append("No user identifier column found (`user_id`/`uid`/etc). Duplicate-user checks skipped.")
    else:
        miss_uid = int(df[user_id_col].isna().sum())
        if miss_uid > 0:
            warnings.append(f"`{user_id_col}` has missing values: n={miss_uid}.")
        dup_uid = int(df[user_id_col].duplicated().sum())
        if dup_uid > 0:
            warnings.append(f"`{user_id_col}` has duplicates: n={dup_uid}. Data may be not user-level.")

    # denominator-like checks for ratio-ish datasets
    den_cols = _denominator_like_cols(df) if schema in {"b2c_ratio", "b2c_user_level"} else []
    zero_den: dict[str, float] = {}
    for c in den_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        z = float((s == 0).mean())
        if z > 0:
            zero_den[c] = z
    if zero_den:
        warnings.append(f"Detected zeros in denominator-like columns: {zero_den}")

    # missing summary
    col_profile = _make_column_profile(df)

    # numeric summary (light)
    num_cols = _basic_numeric_cols(df)
    numeric_summary = None
    if num_cols:
        numeric_summary = df[num_cols].describe(percentiles=[0.01, 0.5, 0.99]).T.reset_index().rename(columns={"index": "column"})
    else:
        warnings.append("No numeric columns detected. Numeric summary skipped.")

    # group balance table
    group_balance = None
    if group_col is not None:
        group_balance = _group_balance(df, group_col)

        # imbalance warning
        if len(group_balance) >= 2:
            nmax = float(group_balance["n"].max())
            nmin = float(group_balance["n"].min())
            if nmin > 0 and (nmax / nmin) > 3.0:
                warnings.append(f"Strong group size imbalance detected: max/min={nmax/nmin:.3g} (>3).")

    # ---- Write bundle artifacts ----
    write_run_meta(out_dir, vars(args), extra={"command": "audit"})

    artifacts = {"tables": [], "plots": []}  # MVP: no plots yet

    artifacts["tables"].append(write_table(out_dir, "column_profile", col_profile))
    if numeric_summary is not None:
        artifacts["tables"].append(write_table(out_dir, "numeric_summary", numeric_summary))
    if group_balance is not None:
        artifacts["tables"].append(write_table(out_dir, "group_balance", group_balance))

    audit_payload: dict[str, Any] = {
        "command": "audit",
        "inputs": {
            "input": args.input,
            "schema": schema,
        },
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "detected": {
            "group_col": group_col,
            "user_id_col": user_id_col,
            "denominator_like_cols": den_cols,
        },
        "checks": {
            "zero_denominator_share": zero_den,
        },
        "warnings": warnings,
        "errors": errors,
        "artifacts": artifacts,
    }

    # audit.json + audit.md
    write_audit_json(out_dir, audit_payload)

    warn_block = ("- " + "\n- ".join(warnings)) if warnings else "(none)"
    err_block = ("- " + "\n- ".join(errors)) if errors else "(none)"

    audit_md = f"""# tecore audit report

## Inputs
- input: `{args.input}`
- schema: `{schema}`

## Dataset
- shape: {df.shape[0]} rows × {df.shape[1]} cols
- detected group_col: `{group_col}`
- detected user_id_col: `{user_id_col}`

## Warnings
{warn_block}

## Errors
{err_block}

## Tables
- tables/column_profile.csv
- tables/numeric_summary.csv (if numeric columns exist)
- tables/group_balance.csv (if group column detected)
"""
    write_audit_md(out_dir, audit_md)

    # For bundle consistency with other commands: also create results.json + report.md
    # (results.json is the unified envelope; report.md is a human-readable entry point)
    results_payload: dict[str, Any] = {
        "command": "audit",
        "inputs": {"input": args.input, "schema": schema},
        "estimates": {},
        "diagnostics": {"shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}},
        "warnings": warnings,
        "artifacts": {
            "report_md": "report.md",
            "plots": [],
            "tables": artifacts["tables"],
        },
    }
    write_results_json(out_dir, results_payload)
    write_report_md(out_dir, audit_md)

    return 0
