from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # "int", "float", "str"


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    required: List[ColumnSpec]
    group_col: str = "group"
    allowed_groups: tuple[str, str] = ("control", "test")


def get_schema(name: str) -> DatasetSchema:
    name = name.strip().lower()

    if name in ("b2c_user_level", "user_level"):
        return DatasetSchema(
            name="b2c_user_level",
            required=[
                ColumnSpec("group", "str"),
                ColumnSpec("revenue", "float"),
                ColumnSpec("revenue_pre", "float"),
            ],
        )

    if name in ("b2c_ratio", "ratio"):
        return DatasetSchema(
            name="b2c_ratio",
            required=[
                ColumnSpec("group", "str"),
                ColumnSpec("revenue", "float"),
                ColumnSpec("sessions", "float"),
                ColumnSpec("revenue_pre", "float"),
                ColumnSpec("sessions_pre", "float"),
            ],
        )

    raise ValueError(f"Unknown schema: {name}")
