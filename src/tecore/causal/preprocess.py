from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import DataSpec, validate_timeseries_df


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]
    pre_mask: pd.Series
    post_mask: pd.Series


def _winsorize_series(s: pd.Series, p: float) -> pd.Series:
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _make_complete_calendar(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    start = df[date_col].min()
    end = df[date_col].max()
    full_idx = pd.date_range(start=start, end=end, freq=freq)
    out = df.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()
    return out


def _apply_missing_policy(df: pd.DataFrame, cols: List[str], policy: str) -> pd.DataFrame:
    out = df.copy()
    if policy == "raise":
        if out[cols].isna().any().any():
            missing_counts = out[cols].isna().sum().to_dict()
            raise ValueError(
                "Missing values after calendar alignment. "
                f"missing_policy='raise' triggered. Missing counts: {missing_counts}"
            )
        return out

    if policy == "ffill":
        out[cols] = out[cols].ffill()
    elif policy == "bfill":
        out[cols] = out[cols].bfill()
    elif policy == "zero":
        out[cols] = out[cols].fillna(0.0)
    else:
        raise ValueError(f"Unsupported missing_policy: {policy}")

    if out[cols].isna().any().any():
        # handle edge effects for ffill/bfill
        out[cols] = out[cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    return out


def add_features(
    df: pd.DataFrame,
    spec: DataSpec,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create modeling features:
      - covariates x_cols
      - optional time trend t
      - optional day-of-week dummies (daily freq only)
    """
    out = df.copy()
    feature_cols: List[str] = []

    # Base covariates
    for c in spec.x_cols:
        feature_cols.append(c)

    if spec.add_time_trend:
        out["_t"] = np.arange(len(out), dtype=float)
        feature_cols.append("_t")

    if spec.add_day_of_week and spec.freq.upper().startswith("D"):
        dow = out[spec.date_col].dt.dayofweek.astype(int)  # 0=Mon
        dummies = pd.get_dummies(dow, prefix="dow", drop_first=True)
        out = pd.concat([out, dummies], axis=1)
        feature_cols.extend(list(dummies.columns))

    return out, feature_cols


def prepare_timeseries(
    df: pd.DataFrame,
    spec: DataSpec,
    intervention_date: pd.Timestamp,
) -> PreparedData:
    """
    Validate, align to full calendar, handle missing, optional winsorize, create features.
    """
    clean = validate_timeseries_df(df, spec)

    # calendar alignment
    clean = _make_complete_calendar(clean, spec.date_col, spec.freq)

    # missing policy applies to y and x
    value_cols = [spec.y_col] + list(spec.x_cols)
    clean = _apply_missing_policy(clean, value_cols, spec.missing_policy)

    # winsorize numeric
    if spec.winsorize_p is not None:
        p = float(spec.winsorize_p)
        if not (0.0 < p < 0.5):
            raise ValueError("winsorize_p must be in (0, 0.5)")
        for c in value_cols:
            clean[c] = _winsorize_series(clean[c], p)

    # add features
    with_features, feature_cols = add_features(clean, spec)

    # split
    pre_mask = with_features[spec.date_col] < intervention_date
    post_mask = with_features[spec.date_col] >= intervention_date

    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        raise ValueError(
            f"Empty pre or post period. Check intervention_date={intervention_date.date()} "
            f"vs data range {with_features[spec.date_col].min().date()}..{with_features[spec.date_col].max().date()}"
        )

    return PreparedData(
        df=with_features,
        feature_cols=feature_cols,
        pre_mask=pre_mask,
        post_mask=post_mask,
    )
