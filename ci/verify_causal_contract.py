from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "results.json",
    "report.md",
    "run_meta.json",
    "audit.json",
    "audit.md",
    "tables/effect_series.csv",
    "plots/observed_vs_counterfactual.png",
    "plots/point_effect.png",
    "plots/cumulative_effect.png",
]

REQUIRED_EFFECT_COLS = ["date", "y", "y_cf", "point_effect", "cum_effect", "is_post"]


def _die(msg: str) -> None:
    raise SystemExit(f"[causal-contract][FAIL] {msg}")


def _load_nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify causal-impact bundle contract.")
    ap.add_argument("bundle", help="Path to bundle directory (e.g., out/yaml_ci_001)")
    args = ap.parse_args()

    bundle = Path(args.bundle)

    if not bundle.exists() or not bundle.is_dir():
        _die(f"bundle path does not exist or is not a directory: {bundle}")

    missing = []
    empty = []
    for rel in REQUIRED_FILES:
        p = bundle / rel
        if not p.exists():
            missing.append(rel)
        elif p.is_file() and p.stat().st_size == 0:
            empty.append(rel)

    if missing:
        _die(f"missing required files: {missing}")
    if empty:
        _die(f"empty required files: {empty}")

    effect_path = bundle / "tables/effect_series.csv"
    df = pd.read_csv(effect_path)

    miss_cols = [c for c in REQUIRED_EFFECT_COLS if c not in df.columns]
    if miss_cols:
        _die(f"effect_series.csv missing required columns: {miss_cols}. got={list(df.columns)}")

    if len(df) < 10:
        _die(f"effect_series.csv too short: n={len(df)}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        bad = int(dates.isna().sum())
        _die(f"effect_series.csv has invalid dates: {bad} NaT values")
    if not dates.is_monotonic_increasing:
        _die("effect_series.csv date column is not monotonic increasing")

    is_post = df["is_post"]
    if not (
        pd.api.types.is_bool_dtype(is_post)
        or pd.api.types.is_integer_dtype(is_post)
        or pd.api.types.is_object_dtype(is_post)
    ):
        _die(f"is_post has unexpected dtype: {is_post.dtype}")

    is_post_bool = is_post.astype(str).str.lower().isin(["true", "1", "yes"])
    if is_post_bool.all() or (~is_post_bool).all():
        _die("effect_series.csv has no pre or no post rows (is_post is all one value)")

    for col in ["y", "y_cf", "point_effect", "cum_effect"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            bad = int(vals.isna().sum())
            _die(f"effect_series.csv column {col} has non-numeric values: {bad} NaN after coercion")

    results = json.loads((bundle / "results.json").read_text(encoding="utf-8"))

    point = _load_nested(results, "estimates.summary.point_effect")
    cum = _load_nested(results, "estimates.summary.cum_effect")

    if point is None:
        _die("results.json missing estimates.summary.point_effect")
    if cum is None:
        _die("results.json missing estimates.summary.cum_effect")

    try:
        float(point)
        float(cum)
    except Exception:
        _die(f"summary effects not numeric: point_effect={point!r}, cum_effect={cum!r}")

    for rel in [
        "plots/observed_vs_counterfactual.png",
        "plots/point_effect.png",
        "plots/cumulative_effect.png",
    ]:
        p = bundle / rel
        if p.stat().st_size < 2_000:
            _die(f"plot too small (likely broken): {rel} size={p.stat().st_size} bytes")

    print("[causal-contract][OK] bundle contract verified:", bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
