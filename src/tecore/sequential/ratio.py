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
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, float]:
    """
    v1 public wrapper.

    Returns:
      (df_out, baseline_ratio)

    Behavior:
      - linearize ratio around a fixed baseline ratio r0 (estimated from control)
      - output column name: y_lin_col (default: 'y_lin')
      - baseline_mode:
          * 'first_look' (requires first_look_n)
          * 'full_control' (optional, if supported by underlying implementation)

    This wrapper forwards to the internal implementation `linearize_ratio_frame`
    while staying robust to small signature differences.
    """
    try:
        fn = linearize_ratio_frame  
    except NameError as e:
        raise ImportError("linearize_ratio_frame not found in tecore.sequential.ratio") from e

    sig = inspect.signature(fn)
    param_names = set(sig.parameters.keys())

    candidates: dict[str, Any] = {
        "df": df,
        "frame": df,
        "data": df,

        # columns
        "num_col": num_col,
        "numerator_col": num_col,
        "num": num_col,

        "den_col": den_col,
        "denominator_col": den_col,
        "den": den_col,

        "group_col": group_col,
        "variant_col": group_col,
        "arm_col": group_col,

        "control_label": control_label,
        "control": control_label,
        "control_value": control_label,
        "control_group": control_label,

        "baseline_mode": baseline_mode,
        "baseline": baseline_mode,

        "first_look_n": first_look_n,
        "look_n": first_look_n,
        "initial_look_n": first_look_n,

        "y_lin_col": y_lin_col,
        "out_col": y_lin_col,
        "y_col": y_lin_col,
        "output_col": y_lin_col,

        "eps": eps,
        "epsilon": eps,
    }

    call_kwargs: dict[str, Any] = {}
    for p in param_names:
        if p in candidates:
            call_kwargs[p] = candidates[p]

    if baseline_mode == "first_look" and first_look_n is None:
        if any(k in param_names for k in ["first_look_n", "look_n", "initial_look_n"]):
            raise ValueError("first_look_n must be provided when baseline_mode='first_look'")

    out = fn(**call_kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        return out[0], float(out[1])

    if isinstance(out, pd.DataFrame):
        raise TypeError(
            "linearize_ratio_frame returned a DataFrame only; expected (df, baseline_ratio). "
            "Please update linearize_ratio_frame to also return baseline_ratio."
        )

    raise TypeError(
        f"Unexpected return type from linearize_ratio_frame: {type(out)}. "
        "Expected (df_out, baseline_ratio)."
    )

