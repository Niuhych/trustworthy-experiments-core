from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tecore.cli.bundle import ensure_subdirs, write_audit_json, write_audit_md, write_table


def _infer_group_col(df: pd.DataFrame, preferred: str = "group") -> str | None:
    if preferred in df.columns:
        return preferred
    for c in ["variant", "arm", "bucket", "treatment", "ab_group", "grp"]:
        if c in df.columns:
            return c
    return None


def _basic_numeric_cols(df: pd.DataFrame) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if not c.startswith("_")]


def _make_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        dtype = str(s.dtype)
        nunique = int(s.nunique(dropna=True))

        example = ""
        try:
            if s.dropna().shape[0] > 0:
                example = str(s.dropna().iloc[0])
        except Exception:
            example = ""

        rows.append(
            {
                "column": c,
                "dtype": dtype,
                "n_rows": n,
                "n_missing": miss,
                "missing_share": float(miss / n) if n > 0 else np.nan,
                "n_unique": nunique,
                "example": example,
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
    cols: list[str] = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["session", "den", "denom", "impression", "click", "visit"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def build_audit(df: pd.DataFrame, schema: str, parent_command: str | None = None) -> tuple[dict[str, Any], str, dict[str, pd.DataFrame]]:
    warnings: list[str] = []
    errors: list[str] = []

    group_col = _infer_group_col(df, preferred="group")
    if group_col is None:
        warnings.append("No obvious group column found (expected `group`). Group balance checks skipped.")
    else:
        if df[group_col].nunique(dropna=True) < 2:
            warnings.append(f"Group column `{group_col}` has <2 unique values. Experiment comparison may be impossible.")

    if schema not in {"b2c_user_level", "b2c_ratio", "timeseries_causal_impact", "sequential_mean", "sequential_ratio"}:
        warnings.append(f"Unknown schema `{schema}`. Running generic audit only.")


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

    den_cols = _denominator_like_cols(df) if schema in {"b2c_ratio", "b2c_user_level", "sequential_ratio"} else []
    zero_den: dict[str, float] = {}
    for c in den_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        z = float((s == 0).mean())
        if z > 0:
            zero_den[c] = z
    if zero_den:
        warnings.append(f"Detected zeros in denominator-like columns: {zero_den}")

    col_profile = _make_column_profile(df)

    num_cols = _basic_numeric_cols(df)
    numeric_summary = None
    if num_cols:
        numeric_summary = (
            df[num_cols]
            .describe(percentiles=[0.01, 0.5, 0.99])
            .T.reset_index()
            .rename(columns={"index": "column"})
        )
    else:
        warnings.append("No numeric columns detected. Numeric summary skipped.")

    group_balance = None
    if group_col is not None:
        group_balance = _group_balance(df, group_col)
        if len(group_balance) >= 2:
            nmax = float(group_balance["n"].max())
            nmin = float(group_balance["n"].min())
            if nmin > 0 and (nmax / nmin) > 3.0:
                warnings.append(f"Strong group size imbalance detected: max/min={nmax/nmin:.3g} (>3).")

    tables: dict[str, pd.DataFrame] = {
        "audit_column_profile": col_profile,
    }
    if numeric_summary is not None:
        tables["audit_numeric_summary"] = numeric_summary
    if group_balance is not None:
        tables["audit_group_balance"] = group_balance

    artifacts = {"tables": [], "plots": []}

    audit_payload: dict[str, Any] = {
        "command": "audit",
        "parent_command": parent_command,
        "inputs": {"schema": schema},
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "detected": {
            "group_col": group_col,
            "user_id_col": user_id_col,
            "denominator_like_cols": den_cols,
        },
        "checks": {"zero_denominator_share": zero_den},
        "warnings": warnings,
        "errors": errors,
        "artifacts": artifacts,
    }

    warn_block = ("- " + "\n- ".join(warnings)) if warnings else "(none)"
    err_block = ("- " + "\n- ".join(errors)) if errors else "(none)"

    audit_md = f"""# tecore audit report

## Inputs
- schema: `{schema}`
- parent_command: `{parent_command or ""}`

## Dataset
- shape: {df.shape[0]} rows × {df.shape[1]} cols
- detected group_col: `{group_col}`
- detected user_id_col: `{user_id_col}`

## Warnings
{warn_block}

## Errors
{err_block}

## Tables
- tables/audit_column_profile.csv
- tables/audit_numeric_summary.csv (if numeric columns exist)
- tables/audit_group_balance.csv (if group column detected)
"""

    return audit_payload, audit_md, tables


def write_audit_bundle(out_dir, df: pd.DataFrame, schema: str, parent_command: str | None = None) -> dict[str, Any]:
    ensure_subdirs(out_dir, ["tables", "plots"])

    audit_payload, audit_md, tables = build_audit(df=df, schema=schema, parent_command=parent_command)

    artifacts = {"tables": [], "plots": []}
    for name, tdf in tables.items():
        artifacts["tables"].append(write_table(out_dir, name, tdf))

    audit_payload["artifacts"] = artifacts

    write_audit_json(out_dir, audit_payload)
    write_audit_md(out_dir, audit_md)

    return audit_payload
