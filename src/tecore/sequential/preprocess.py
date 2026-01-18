from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tecore.sequential.schema import LookSchedule, SequentialConfig, SequentialSpec, sort_for_sequential


def build_looks_from_rows(
    df: pd.DataFrame,
    schedule: LookSchedule,
    *,
    timestamp_col: Optional[str] = None,
) -> List[pd.DataFrame]:
    """Split a user-level dataframe into cumulative looks.

    v1: looks are defined by cumulative number of rows (after ordering).
    """
    ordered = sort_for_sequential(df, timestamp_col)
    looks = schedule.resolve(total_n=len(ordered))
    return [ordered.iloc[:n].copy() for n in looks]


def compute_two_sample_stats(
    df_look: pd.DataFrame,
    *,
    group_col: str,
    control_label: str,
    test_label: str,
    y_col: str,
    cfg: SequentialConfig,
) -> Dict[str, Any]:
    """Compute two-sample (Welch) stats for a single look."""
    y = pd.to_numeric(df_look[y_col], errors="coerce")
    g = df_look[group_col]

    yc = y[g == control_label].to_numpy(dtype=float)
    yt = y[g == test_label].to_numpy(dtype=float)
    yc = yc[np.isfinite(yc)]
    yt = yt[np.isfinite(yt)]

    n_c = int(len(yc))
    n_t = int(len(yt))

    mean_c = float(np.mean(yc)) if n_c > 0 else np.nan
    mean_t = float(np.mean(yt)) if n_t > 0 else np.nan

    var_c = float(np.var(yc, ddof=1)) if n_c > 1 else 0.0
    var_t = float(np.var(yt, ddof=1)) if n_t > 1 else 0.0

    var_c = float(max(var_c, cfg.var_floor))
    var_t = float(max(var_t, cfg.var_floor))

    diff = float(mean_t - mean_c) if (np.isfinite(mean_t) and np.isfinite(mean_c)) else np.nan

    se = float(np.sqrt(var_c / max(n_c, 1) + var_t / max(n_t, 1)))
    if not np.isfinite(se) or se <= 0:
        z = np.nan
        info = 0.0
    else:
        z = float(diff / se) if np.isfinite(diff) else np.nan
        info = float(1.0 / (se**2))

    return {
        "n_control": n_c,
        "n_test": n_t,
        "mean_control": mean_c,
        "mean_test": mean_t,
        "var_control": var_c,
        "var_test": var_t,
        "diff": diff,
        "se": se,
        "z": z,
        "info": info,
    }


def build_look_table_mean(
    df: pd.DataFrame,
    spec: SequentialSpec,
    schedule: LookSchedule,
    cfg: SequentialConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a per-look table for a mean metric."""
    warnings: list[str] = []
    if spec.y_col is None:
        raise ValueError("SequentialSpec.y_col is required")

    ordered = sort_for_sequential(df, spec.timestamp_col)
    looks = schedule.resolve(total_n=len(ordered))
    rows: list[dict[str, Any]] = []

    for n in looks:
        d = compute_two_sample_stats(
            ordered.iloc[:n],
            group_col=spec.group_col,
            control_label=spec.control_label,
            test_label=spec.test_label,
            y_col=spec.y_col,
            cfg=cfg,
        )
        d["look_n"] = int(n)

        if d["n_control"] < cfg.min_n_per_group or d["n_test"] < cfg.min_n_per_group:
            warnings.append(
                f"look_n={n}: n_control={d['n_control']}, n_test={d['n_test']} < min_n_per_group={cfg.min_n_per_group}."
            )
        rows.append(d)

    return pd.DataFrame(rows), warnings
