from __future__ import annotations

from typing import List

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
