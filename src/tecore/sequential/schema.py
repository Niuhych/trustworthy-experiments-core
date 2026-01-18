from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SequentialMode(str, Enum):
    GROUP_SEQUENTIAL = "group_sequential"
    CONFIDENCE_SEQUENCE = "confidence_sequence"


class SpendingFunction(str, Enum):
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"


class EffectDirection(str, Enum):
    TWO_SIDED = "two_sided"
    INCREASE = "increase"  # test > control
    DECREASE = "decrease"  # test < control


@dataclass(frozen=True)
class SequentialSpec:
    """Input specification for sequential A/B analysis."""

    group_col: str = "group"
    control_label: str = "control"
    test_label: str = "test"

    # Mean metric
    y_col: Optional[str] = None

    # Ratio metric
    num_col: Optional[str] = None
    den_col: Optional[str] = None

    # Optional ordering
    timestamp_col: Optional[str] = None
    unit_col: Optional[str] = None


@dataclass(frozen=True)
class LookSchedule:
    """Defines when we compute interim analyses (looks)."""

    looks: Optional[List[int]] = None
    n_looks: Optional[int] = None
    max_n: Optional[int] = None

    def resolve(self, total_n: int) -> List[int]:
        if total_n <= 0:
            return []

        if self.looks is not None:
            raw = [int(x) for x in self.looks]
        else:
            if self.n_looks is None:
                raise ValueError("LookSchedule: provide either looks or n_looks.")
            if self.max_n is None:
                raise ValueError("LookSchedule: max_n is required when using n_looks.")
            if self.n_looks <= 0:
                raise ValueError("LookSchedule: n_looks must be > 0.")
            if self.max_n <= 0:
                raise ValueError("LookSchedule: max_n must be > 0.")
            raw = [int(round((k + 1) * self.max_n / self.n_looks)) for k in range(self.n_looks)]

        out: List[int] = []
        for n in raw:
            if n <= 0:
                continue
            if n > total_n:
                continue
            out.append(int(n))
        out = sorted(set(out))
        if not out:
            out = [min(total_n, int(self.max_n) if self.max_n else total_n)]
        out = [n for n in out if n <= total_n]
        if out and out[-1] != min(total_n, out[-1]):
            out[-1] = min(total_n, out[-1])
        return out


@dataclass(frozen=True)
class SequentialConfig:
    mode: SequentialMode = SequentialMode.GROUP_SEQUENTIAL
    alpha: float = 0.05

    two_sided: bool = True
    spending: SpendingFunction = SpendingFunction.OBRIEN_FLEMING
    effect_direction: EffectDirection = EffectDirection.TWO_SIDED

    min_n_per_group: int = 50
    var_floor: float = 1e-12

    cs_tau: float = 1.0

    seed: int = 42


@dataclass
class SequentialResult:
    stopped: bool
    stop_look: Optional[int]
    decision: str 

    final_p_value: Optional[float]
    final_ci: Optional[Tuple[float, float]]
    cs: Optional[Tuple[float, float]]

    look_table: pd.DataFrame
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["look_table"] = self.look_table.to_dict(orient="list") if self.look_table is not None else None
        return d


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")


def validate_spec_for_mean(df: pd.DataFrame, spec: SequentialSpec) -> None:
    if not spec.y_col:
        raise ValueError("SequentialSpec.y_col is required for mean analysis")
    ensure_columns(df, [spec.group_col, spec.y_col])


def validate_spec_for_ratio(df: pd.DataFrame, spec: SequentialSpec) -> None:
    if not spec.num_col or not spec.den_col:
        raise ValueError("SequentialSpec.num_col and den_col are required for ratio analysis")
    ensure_columns(df, [spec.group_col, spec.num_col, spec.den_col])


def sort_for_sequential(df: pd.DataFrame, timestamp_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if timestamp_col and timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        out = out.sort_values(timestamp_col, kind="mergesort")
    return out.reset_index(drop=True)
