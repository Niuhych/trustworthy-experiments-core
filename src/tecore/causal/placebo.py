from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def choose_placebo_dates(
    dates: pd.Series,
    intervention_date: pd.Timestamp,
    n_placebos: int,
    min_pre_points: int,
    post_window_len: int,
    random_state: int,
) -> pd.DatetimeIndex:
    """
    Choose placebo intervention dates within pre-period such that:
      - at least min_pre_points before placebo date
      - placebo post window fits entirely within pre-period (i.e., placebo_date + post_window_len <= intervention_date)
    """
    all_dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    pre_dates = all_dates[all_dates < intervention_date]

    if len(pre_dates) < (min_pre_points + post_window_len + 1):
        return pd.DatetimeIndex([])

    earliest_idx = min_pre_points
    latest_idx = len(pre_dates) - post_window_len - 1
    if latest_idx <= earliest_idx:
        return pd.DatetimeIndex([])

    candidate = pre_dates.iloc[earliest_idx:latest_idx + 1]
    rng = np.random.default_rng(random_state)

    if len(candidate) <= n_placebos:
        return pd.DatetimeIndex(candidate.values)

    chosen = rng.choice(candidate.values, size=n_placebos, replace=False)
    return pd.DatetimeIndex(sorted(pd.to_datetime(chosen)))


def placebo_in_time_test(
    df: pd.DataFrame,
    date_col: str,
    intervention_date: pd.Timestamp,
    post_window_len: int,
    n_placebos: int,
    min_pre_points: int,
    random_state: int,
    estimate_fn: Callable[[pd.Timestamp, int], float],
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Generic placebo-in-time:
      - pick placebo dates
      - for each placebo date: compute a cumulative effect over a window of length post_window_len
      - p-value: two-sided relative to |true_effect|
        (returned separately by the caller if desired)

    estimate_fn(placebo_date, post_window_len) -> cum_effect_placebo
    """
    dates = pd.to_datetime(df[date_col])
    placebo_dates = choose_placebo_dates(
        dates=dates,
        intervention_date=intervention_date,
        n_placebos=n_placebos,
        min_pre_points=min_pre_points,
        post_window_len=post_window_len,
        random_state=random_state,
    )

    rows = []
    for d in placebo_dates:
        try:
            ce = float(estimate_fn(pd.Timestamp(d), post_window_len))
            rows.append({"placebo_date": pd.Timestamp(d), "cum_effect": ce})
        except Exception:
            rows.append({"placebo_date": pd.Timestamp(d), "cum_effect": np.nan})

    placebo_df = pd.DataFrame(rows).sort_values("placebo_date")
    return placebo_df, None


def two_sided_p_value(placebo_cum_effects: np.ndarray, true_cum_effect: float) -> Optional[float]:
    x = np.asarray(placebo_cum_effects, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return float(np.mean(np.abs(x) >= abs(true_cum_effect)))
