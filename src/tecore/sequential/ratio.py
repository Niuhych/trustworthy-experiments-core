from __future__ import annotations

import inspect

from typing import Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from tecore.metrics.ratio import ratio_point
from tecore.sequential.schema import LookSchedule, SequentialSpec, sort_for_sequential, validate_spec_for_ratio


def linearize_ratio_frame(
    df: pd.DataFrame,
    spec: SequentialSpec,
    schedule: LookSchedule,
    *,
    baseline_mode: str = "first_look",
    output_col: str = "_y_lin",
) -> tuple[pd.DataFrame, float, List[str]]:
    """Linearize a ratio metric in a sequential setting.

    v1: avoid moving target baseline.

    baseline_mode:
      - "first_look": estimate r0 from CONTROL rows in the first look and keep it fixed.
      - "pre": alias for "first_look" (reserved for future pre-period support).
    """
    validate_spec_for_ratio(df, spec)
    warnings: List[str] = []

    ordered = sort_for_sequential(df, spec.timestamp_col)
    looks = schedule.resolve(total_n=len(ordered))
    if not looks:
        raise ValueError("LookSchedule resolved to empty looks")

    first_n = int(looks[0])
    baseline = ordered.iloc[:first_n]
    if baseline_mode not in {"first_look", "pre"}:
        warnings.append(f"Unknown baseline_mode={baseline_mode!r}; falling back to 'first_look'.")

    num = pd.to_numeric(baseline[spec.num_col], errors="coerce")
    den = pd.to_numeric(baseline[spec.den_col], errors="coerce")
    g = baseline[spec.group_col]

    num_c = num[g == spec.control_label].to_numpy(dtype=float)
    den_c = den[g == spec.control_label].to_numpy(dtype=float)
    num_c = num_c[np.isfinite(num_c)]
    den_c = den_c[np.isfinite(den_c)]

    if float(np.sum(den_c)) <= 0:
        raise ValueError("Baseline control denominator sum must be > 0 to linearize ratio.")
    r0 = ratio_point(num_c, den_c)

    den_all = pd.to_numeric(ordered[spec.den_col], errors="coerce")
    zshare = float((den_all == 0).mean())
    if zshare > 0:
        warnings.append(f"Denominator has zeros: share={zshare:.4f}. Linearization may be unstable.")

    num_all = pd.to_numeric(ordered[spec.num_col], errors="coerce").to_numpy(dtype=float)
    den_all_np = den_all.to_numpy(dtype=float)
    ordered = ordered.copy()
    ordered[output_col] = num_all - float(r0) * den_all_np

    return ordered, float(r0), warnings

def linearize_ratio(
    df: pd.DataFrame,
    num_col: str,
    den_col: str,
    group_col: str,
    control_label: str,
    baseline_mode: str = "first_look",
    first_look_n: int | None = None,
    y_lin_col: str = "y_lin",
    test_label: str = "test",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, float]:
    """
    Public v1 wrapper (stable notebook/CLI API).

    - Builds SequentialSpec + LookSchedule internally
    - Calls linearize_ratio_frame(df, spec, schedule, baseline_mode=..., output_col=y_lin_col)
    - Returns (df_out, baseline_ratio)
    - Preserves warnings in df_out.attrs['tecore_warnings'] (non-breaking)
    """
    if baseline_mode == "first_look":
        if first_look_n is None:
            raise ValueError("first_look_n must be provided when baseline_mode='first_look'")
        schedule = LookSchedule(looks=[int(first_look_n)])
    else:
        schedule = LookSchedule(looks=[int(len(df))])

    spec = SequentialSpec(
        group_col=group_col,
        control_label=control_label,
        test_label=test_label,
        num_col=num_col,
        den_col=den_col,
        timestamp_col=timestamp_col,
    )

    df_out, r0, warnings = linearize_ratio_frame(
        df=df,
        spec=spec,
        schedule=schedule,
        baseline_mode=baseline_mode,
        output_col=y_lin_col,  
    )

    if warnings:
        try:
            existing = list(df_out.attrs.get("tecore_warnings", []))
            df_out.attrs["tecore_warnings"] = existing + list(warnings)
        except Exception:
            pass

    return df_out, float(r0)

