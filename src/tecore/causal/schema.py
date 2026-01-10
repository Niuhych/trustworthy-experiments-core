from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ImpactMethod(str, Enum):
    CAUSAL_IMPACT_LIKE = "causal_impact_like"
    SYNTHETIC_CONTROL = "synthetic_control"
    DID = "did"


@dataclass(frozen=True)
class DataSpec:
    """
    Input specification for single-unit time-series causal inference.

    Minimal v1 assumptions:
      - One treated unit: y(t)
      - Intervention happens at intervention_date
      - Covariates X(t) explain y(t) in pre-period
    """
    date_col: str = "date"
    y_col: str = "y"
    x_cols: List[str] = field(default_factory=list)

    # Frequency: "D" or "W"
    freq: str = "D"

    # Handling missing dates after reindexing to a complete calendar
    # - "raise": error if any missing
    # - "ffill": forward fill numeric columns
    # - "bfill": backward fill numeric columns
    # - "zero": fill 0 for numeric columns
    missing_policy: str = "raise"

    aggregation: str = "mean"

    # Feature engineering
    add_time_trend: bool = True
    add_day_of_week: bool = True

    # Optional winsorization for numeric columns (0 < p < 0.5)
    winsorize_p: Optional[float] = None


@dataclass(frozen=True)
class ImpactConfig:
    intervention_date: Union[str, pd.Timestamp]

    method: ImpactMethod = ImpactMethod.CAUSAL_IMPACT_LIKE

    # uncertainty level
    alpha: float = 0.05

    # ridge regularization strength
    ridge_alpha: float = 1.0

    # bootstrap settings
    bootstrap_iters: int = 200
    block_size: int = 7
    random_state: int = 42

    # quality gates
    pre_period_min_points: int = 30
    pre_r2_min: float = 0.25
    residual_autocorr_abs_max: float = 0.6 

    # placebo (in-time permutation) settings
    run_placebo: bool = True
    n_placebos: int = 20
    placebo_min_pre_points: int = 30

    margin_rate: Optional[float] = None 
    spend_col: Optional[str] = None      
    incremental_cost: Optional[float] = None 

    verbose: bool = False

    def intervention_ts(self) -> pd.Timestamp:
        ts = pd.to_datetime(self.intervention_date)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts


@dataclass
class ImpactResult:
    method: ImpactMethod
    intervention_date: pd.Timestamp
    alpha: float

    # headline metrics
    point_effect: float
    point_ci: Tuple[float, float]

    cum_effect: float
    cum_ci: Tuple[float, float]

    rel_effect: float
    rel_ci: Tuple[float, float]

    p_value: Optional[float] = None

    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    effect_series: Optional[pd.DataFrame] = None
    placebo_results: Optional[pd.DataFrame] = None
    economics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intervention_date"] = self.intervention_date.strftime("%Y-%m-%d")
        if self.effect_series is not None:
            d["effect_series"] = self.effect_series.to_dict(orient="list")
        if self.placebo_results is not None:
            d["placebo_results"] = self.placebo_results.to_dict(orient="list")
        return d

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "intervention_date": self.intervention_date.strftime("%Y-%m-%d"),
            "point_effect": self.point_effect,
            "point_ci": self.point_ci,
            "cum_effect": self.cum_effect,
            "cum_ci": self.cum_ci,
            "rel_effect": self.rel_effect,
            "rel_ci": self.rel_ci,
            "p_value": self.p_value,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "economics": self.economics,
        }


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    non_numeric = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            non_numeric.append(c)
    if non_numeric:
        raise TypeError(f"Columns must be numeric: {non_numeric}")


def validate_timeseries_df(df: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
    """
    Basic, strict validation:
      - required columns exist
      - date parsable
      - sorted by date, unique (or aggregatable)
      - numeric y and x
    """
    cols = [spec.date_col, spec.y_col] + list(spec.x_cols)
    ensure_columns(df, cols)

    out = df.copy()
    out[spec.date_col] = pd.to_datetime(out[spec.date_col])
    if out[spec.date_col].isna().any():
        raise ValueError("Some dates could not be parsed. Check date_col values.")

    out = out.sort_values(spec.date_col)

    if out[spec.date_col].duplicated().any():
        if spec.aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggregation: {spec.aggregation}. Use 'mean' or 'sum'.")
        agg_fn = "mean" if spec.aggregation == "mean" else "sum"
        num_cols = [spec.y_col] + list(spec.x_cols)
        out = (
            out.groupby(spec.date_col, as_index=False)[num_cols]
            .agg(agg_fn)
            .sort_values(spec.date_col)
        )

    ensure_numeric(out, [spec.y_col] + list(spec.x_cols))
    return out
