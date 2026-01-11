import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _die(msg: str) -> None:
    print(f"[golden][FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[golden] {msg}")


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"Failed to read json: {p} ({e})")


def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _as_float(x: Any) -> float:
    if isinstance(x, (np.integer, np.floating)):
        return float(x)
    return float(x)


def _compare_numbers(path: str, actual: Any, expected: Any, rtol: float, atol: float) -> list[str]:
    if not _is_number(actual) or not _is_number(expected):
        return [f"{path}: expected/actual are not both numeric (expected={expected!r}, actual={actual!r})"]
    a = _as_float(actual)
    e = _as_float(expected)
    if not np.isfinite(a) or not np.isfinite(e):
        # for NaN/inf be strict
        if (np.isnan(a) and np.isnan(e)) or (a == e):
            return []
        return [f"{path}: non-finite mismatch (expected={e!r}, actual={a!r})"]
    if not np.isclose(a, e, rtol=rtol, atol=atol):
        return [f"{path}: mismatch (expected={e:.12g}, actual={a:.12g}, rtol={rtol}, atol={atol})"]
    return []


def _compare_csv(actual_path: Path, expected_path: Path, rtol: float, atol: float, key_cols: list[str] | None) -> list[str]:
    if not expected_path.exists():
        return [f"missing expected csv: {expected_path}"]
    if not actual_path.exists():
        return [f"missing actual csv: {actual_path}"]

    a = pd.read_csv(actual_path)
    e = pd.read_csv(expected_path)

    if key_cols:
        for k in key_cols:
            if k not in a.columns or k not in e.columns:
                return [f"key col {k!r} missing in csv (actual_cols={list(a.columns)}, expected_cols={list(e.columns)})"]
        a = a.sort_values(key_cols).reset_index(drop=True)
        e = e.sort_values(key_cols).reset_index(drop=True)

    if list(a.columns) != list(e.columns):
        return [f"csv columns mismatch: {actual_path} (actual={list(a.columns)} expected={list(e.columns)})"]

    if a.shape != e.shape:
        return [f"csv shape mismatch: {actual_path} (actual={a.shape} expected={e.shape})"]

    errs: list[str] = []
    for col in a.columns:
        sa = a[col]
        se = e[col]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(se):
            da = pd.to_numeric(sa, errors="coerce").to_numpy(dtype=float)
            de = pd.to_numeric(se, errors="coerce").to_numpy(dtype=float)
            ok = np.isclose(da, de, rtol=rtol, atol=atol) | (np.isnan(da) & np.isnan(de))
            if not bool(np.all(ok)):
                i = int(np.where(~ok)[0][0])
                errs.append(
                    f"{actual_path.name}:{col}[{i}] mismatch (expected={de[i]!r}, actual={da[i]!r}, rtol={rtol}, atol={atol})"
                )
        else:
            if not sa.astype(str).equals(se.astype(str)):
                i = int(np.where(sa.astype(str).to_numpy() != se.astype(str).to_numpy())[0][0])
                errs.append(f"{actual_path.name}:{col}[{i}] mismatch (expected={se.iloc[i]!r}, actual={sa.iloc[i]!r})")

    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Path to ci/golden")
    ap.add_argument("out_dirs", nargs="+", help="Bundle directories under out/")
    args = ap.parse_args()

    golden_root = Path(args.golden)
    if not golden_root.exists():
        _die(f"golden root does not exist: {golden_root}")

    failed = False

    for out_dir_str in args.out_dirs:
        out_dir = Path(out_dir_str)
        if not out_dir.exists():
            _die(f"out dir does not exist: {out_dir}")

        name = out_dir.name
        expected_dir = golden_root / name
        expected_json = expected_dir / "expected.json"
        if not expected_json.exists():
            _die(f"missing golden spec: {expected_json}")

        spec = _read_json(expected_json)

        for rel in spec.get("required_files", []):
            p = out_dir / rel
            if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
                _die(f"{name}: missing/empty required file: {p}")

        results_path = out_dir / "results.json"
        results = _read_json(results_path)

        errs: list[str] = []
        for path, rule in (spec.get("results_checks", {}) or {}).items():
            try:
                actual_val = _get_by_path(results, path)
            except KeyError:
                errs.append(f"{name}: results.json missing path: {path}")
                continue
            expected_val = rule.get("value")
            rtol = float(rule.get("rtol", 1e-6))
            atol = float(rule.get("atol", 1e-8))
            errs.extend(_compare_numbers(f"{name}:{path}", actual_val, expected_val, rtol=rtol, atol=atol))

        for item in spec.get("csv_checks", []) or []:
            rel = item["rel"]
            rtol = float(item.get("rtol", 1e-6))
            atol = float(item.get("atol", 1e-8))
            key_cols = item.get("key_cols")
            actual_csv = out_dir / rel
            expected_csv = expected_dir / rel
            errs.extend(_compare_csv(actual_csv, expected_csv, rtol=rtol, atol=atol, key_cols=key_cols))

        for rel in spec.get("plot_files", []) or []:
            p = out_dir / rel
            if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
                errs.append(f"{name}: missing/empty plot file: {p}")

        if errs:
            failed = True
            for e in errs[:50]:
                print(f"[golden][DIFF] {e}", file=sys.stderr)
            if len(errs) > 50:
                print(f"[golden][DIFF] ... and {len(errs) - 50} more", file=sys.stderr)
        else:
            _ok(f"{out_dir}: OK")

    if failed:
        _die("golden verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
