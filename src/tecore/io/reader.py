from __future__ import annotations

import pandas as pd

from tecore.io.schema import DatasetSchema


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_df(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    errors: list[str] = []

    cols = set(df.columns)
    for c in schema.required:
        if c.name not in cols:
            errors.append(f"Missing required column: {c.name}")

    if errors:
        return errors

    g = schema.group_col
    allowed = set(schema.allowed_groups)
    bad = sorted(set(df[g].dropna().astype(str)) - allowed)
    if bad:
        errors.append(f"Unexpected group labels in '{g}': {bad}. Allowed: {sorted(allowed)}")

    return errors
