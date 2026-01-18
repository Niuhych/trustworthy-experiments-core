from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from tecore.cli.bundle import ensure_subdirs, write_audit_json, write_audit_md, write_table
from tecore.sequential.schema import LookSchedule, sort_for_sequential


def _as_int_list(x: Any) -> Optional[list[int]]:
    """Best-effort parsing for a look schedule representation.

    Accepts list/tuple/set of ints, or a CSV string like "200,400,600".
    """
    if x is None:
        return None

    if isinstance(x, list):
        out: list[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out or None

    if isinstance(x, (tuple, set)):
        return _as_int_list(list(x))

    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",") if p.strip()]
        out: list[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out or None

    try:
        return [int(x)]
    except Exception:
        return None


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


def _sequential_schedule_table(
    df: pd.DataFrame,
    *,
    group_col: str,
    control_label: str,
    test_label: str,
    looks: list[int],
    timestamp_col: Optional[str],
    min_n_per_group: Optional[int],
) -> pd.DataFrame:
    ordered = sort_for_sequential(df, timestamp_col)
    g = ordered[group_col]
    rows: list[dict[str, Any]] = []
    for n in looks:
        prefix = g.iloc[: int(n)]
        nc = int((prefix == control_label).sum())
        nt = int((prefix == test_label).sum())
        ok = None
        if min_n_per_group is not None:
            ok = (nc >= min_n_per_group) and (nt >= min_n_per_group)
        rows.append(
            {
                "look_n": int(n),
                "n_control_prefix": nc,
                "n_test_prefix": nt,
                "min_n_per_group": int(min_n_per_group) if min_n_per_group is not None else None,
                "ok_min_n": ok,
            }
        )
    return pd.DataFrame(rows)


def build_audit(
    df: pd.DataFrame,
    schema: str,
    parent_command: str | None = None,
    context: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str, dict[str, pd.DataFrame]]:
    """Build audit payload + markdown + tables.

    `context` is an optional dict passed from CLI commands. For sequential
    commands it enables schedule sanity checks (looks <= N, min_n_per_group
    achievable on early looks, etc.).
    """
    context = context or {}
    warnings: list[str] = []
    errors: list[str] = []

    # Prefer explicit group_col from context for sequential pipelines.
    group_col = str(context.get("group_col")) if context.get("group_col") in df.columns else None
    if group_col is None:
        group_col = _infer_group_col(df, preferred="group")

    if group_col is None:
        warnings.append("No obvious group column found (expected `group`). Group balance checks skipped.")
    else:
        if df[group_col].nunique(dropna=True) < 2:
            warnings.append(f"Group column `{group_col}` has <2 unique values. Experiment comparison may be impossible.")

    known_schemas = {"b2c_user_level", "b2c_ratio", "timeseries_causal_impact", "sequential_mean", "sequential_ratio"}
    if schema not in known_schemas:
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

    # Basic schema-specific column existence checks.
    if schema == "sequential_mean":
        y_col = context.get("y_col")
        if y_col and y_col not in df.columns:
            errors.append(f"Expected y column `{y_col}` not found.")
        if y_col and y_col in df.columns:
            s = pd.to_numeric(df[y_col], errors="coerce")
            miss = float(s.isna().mean())
            if miss > 0.0:
                warnings.append(f"`{y_col}` has missing/non-numeric values: share={miss:.4f}.")

    if schema == "sequential_ratio":
        num_col = context.get("num_col")
        den_col = context.get("den_col")
        if num_col and num_col not in df.columns:
            errors.append(f"Expected num column `{num_col}` not found.")
        if den_col and den_col not in df.columns:
            errors.append(f"Expected den column `{den_col}` not found.")
        if den_col and den_col in df.columns:
            den = pd.to_numeric(df[den_col], errors="coerce")
            z0 = float((den == 0).mean())
            zneg = float((den < 0).mean())
            zbad = float((~np.isfinite(den.to_numpy(dtype=float))).mean())
            if z0 > 0:
                warnings.append(f"`{den_col}` has zeros: share={z0:.4f}. Ratio/linearization may be unstable.")
            if zneg > 0:
                warnings.append(f"`{den_col}` has negative values: share={zneg:.4f}. Check metric definition.")
            if zbad > 0:
                warnings.append(f"`{den_col}` has missing/non-numeric values: share={zbad:.4f}.")

    # Denominator-like detection (kept for legacy ratio audits).
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

    # Sequential schedule sanity (if context provided).
    schedule_table = None
    resolved_looks: Optional[list[int]] = None
    if schema in {"sequential_mean", "sequential_ratio"} and group_col is not None:
        looks = _as_int_list(context.get("looks"))
        n_looks = context.get("n_looks")
        max_n = context.get("max_n")

        try:
            sched = LookSchedule(
                looks=looks,
                n_looks=int(n_looks) if n_looks is not None else None,
                max_n=int(max_n) if max_n is not None else None,
            )
            resolved_looks = sched.resolve(total_n=int(len(df)))
        except Exception as e:
            warnings.append(f"Failed to resolve look schedule: {e}")

        if looks is not None and any(int(x) > len(df) for x in looks):
            warnings.append(f"Some requested looks exceed N rows: max_look={max(looks)}, N={len(df)}. They will be dropped.")

        if resolved_looks:
            control_label = str(context.get("control_label", "control"))
            test_label = str(context.get("test_label", "test"))
            timestamp_col = context.get("timestamp_col")
            min_n_per_group = context.get("min_n_per_group")
            try:
                min_n_per_group_i = int(min_n_per_group) if min_n_per_group is not None else None
            except Exception:
                min_n_per_group_i = None

            schedule_table = _sequential_schedule_table(
                df,
                group_col=group_col,
                control_label=control_label,
                test_label=test_label,
                looks=resolved_looks,
                timestamp_col=str(timestamp_col) if timestamp_col else None,
                min_n_per_group=min_n_per_group_i,
            )
            if min_n_per_group_i is not None:
                if not bool(schedule_table["ok_min_n"].all()):
                    warnings.append(
                        "min_n_per_group is not satisfied on some looks (given current ordering). "
                        "Early looks may be skipped or produce warnings."
                    )

    tables: dict[str, pd.DataFrame] = {
        "audit_column_profile": col_profile,
    }
    if numeric_summary is not None:
        tables["audit_numeric_summary"] = numeric_summary
    if group_balance is not None:
        tables["audit_group_balance"] = group_balance
    if schedule_table is not None:
        tables["audit_schedule_sanity"] = schedule_table

    artifacts = {"tables": [], "plots": []}

    audit_payload: dict[str, Any] = {
        "command": "audit",
        "parent_command": parent_command,
        "inputs": {
            "schema": schema,
            "context": context,
        },
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "detected": {
            "group_col": group_col,
            "user_id_col": user_id_col,
            "denominator_like_cols": den_cols,
            "resolved_looks": resolved_looks,
        },
        "checks": {"zero_denominator_share": zero_den},
        "warnings": warnings,
        "errors": errors,
        "artifacts": artifacts,
    }

    warn_block = ("- " + "\n- ".join(warnings)) if warnings else "(none)"
    err_block = ("- " + "\n- ".join(errors)) if errors else "(none)"

    extra_tables = ""
    if schedule_table is not None:
        extra_tables += "- tables/audit_schedule_sanity.csv\n"

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
{extra_tables}"""

    return audit_payload, audit_md, tables


def write_audit_bundle(
    out_dir,
    df: pd.DataFrame,
    schema: str,
    parent_command: str | None = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    ensure_subdirs(out_dir, ["tables", "plots"])

    audit_payload, audit_md, tables = build_audit(df=df, schema=schema, parent_command=parent_command, context=context)

    artifacts = {"tables": [], "plots": []}
    for name, tdf in tables.items():
        artifacts["tables"].append(write_table(out_dir, name, tdf))

    audit_payload["artifacts"] = artifacts

    write_audit_json(out_dir, audit_payload)
    write_audit_md(out_dir, audit_md)

    return audit_payload
