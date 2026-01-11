import json
import os
from pathlib import Path
import sys


def _die(msg: str) -> None:
    print(f"[verify_bundle][FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[verify_bundle] {msg}")


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"Failed to read json: {p} ({e})")


def _require_file(p: Path) -> None:
    if not p.exists() or not p.is_file():
        _die(f"Missing file: {p}")
    if p.stat().st_size <= 0:
        _die(f"Empty file: {p}")


def _require_dir(p: Path) -> None:
    if not p.exists() or not p.is_dir():
        _die(f"Missing dir: {p}")


def _require_any(paths: list[Path]) -> None:
    if not any(p.exists() for p in paths):
        _die(f"None of expected paths exist: {', '.join(map(str, paths))}")


def _is_list_of_str(x) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def verify_bundle_common(out_dir: Path) -> dict:
    """
    Structural + integrity checks:
      - required top-level files exist and non-empty
      - results.json schema basics
      - artifacts paths exist and non-empty
    """
    _require_dir(out_dir)

    results = out_dir / "results.json"
    report = out_dir / "report.md"
    meta = out_dir / "run_meta.json"

    _require_file(results)
    _require_file(report)
    _require_file(meta)

    payload = _read_json(results)

    if "command" not in payload:
        _die(f"{out_dir}: results.json missing 'command'")
    if "inputs" not in payload:
        _die(f"{out_dir}: results.json missing 'inputs'")
    if "artifacts" not in payload:
        _die(f"{out_dir}: results.json missing 'artifacts'")

    artifacts = payload.get("artifacts", {})
    tables = artifacts.get("tables", [])
    plots = artifacts.get("plots", [])
    report_md = artifacts.get("report_md", "report.md")

    if report_md:
        _require_file(out_dir / str(report_md))

    if not _is_list_of_str(tables):
        _die(f"{out_dir}: artifacts.tables must be list[str]")
    if not _is_list_of_str(plots):
        _die(f"{out_dir}: artifacts.plots must be list[str]")

    for rel in tables + plots:
        rel = str(rel).replace("\\", "/")
        _require_file(out_dir / rel)

    return payload


def verify_validate(out_dir: Path) -> None:
    _require_file(out_dir / "tables" / "schema_expected.csv")
    _require_file(out_dir / "tables" / "schema_missing.csv")


def verify_audit(out_dir: Path) -> None:
    _require_file(out_dir / "audit.json")
    _require_file(out_dir / "audit.md")

    audit = _read_json(out_dir / "audit.json")
    artifacts = audit.get("artifacts", {}) if isinstance(audit, dict) else {}
    tables = artifacts.get("tables", [])
    plots = artifacts.get("plots", [])

    if tables is not None and not _is_list_of_str(tables):
        _die(f"{out_dir}: audit.json artifacts.tables must be list[str]")
    if plots is not None and not _is_list_of_str(plots):
        _die(f"{out_dir}: audit.json artifacts.plots must be list[str]")

    for rel in (tables or []) + (plots or []):
        rel = str(rel).replace("\\", "/")
        _require_file(out_dir / rel)

    _require_any([
        out_dir / "tables" / "audit_column_profile.csv",
    ])

def verify_cuped(out_dir: Path) -> None:
    _require_file(out_dir / "tables" / "summary.csv")
    _require_file(out_dir / "plots" / "y_post_by_group.png")
    _require_file(out_dir / "plots" / "x_vs_y_scatter.png")

    if (out_dir / "audit.json").exists():
        verify_audit(out_dir)


def verify_cuped_ratio(out_dir: Path) -> None:
    _require_file(out_dir / "tables" / "summary.csv")
    _require_file(out_dir / "plots" / "ratio_post_by_group.png")
    _require_file(out_dir / "plots" / "linearized_by_group.png")

    if (out_dir / "audit.json").exists():
        verify_audit(out_dir)


def verify_causal_impact(out_dir: Path) -> None:
    _require_file(out_dir / "tables" / "effect_series.csv")
    _require_file(out_dir / "plots" / "observed_vs_counterfactual.png")
    _require_file(out_dir / "plots" / "point_effect.png")
    _require_file(out_dir / "plots" / "cumulative_effect.png")

    if (out_dir / "audit.json").exists():
        verify_audit(out_dir)


def verify_one(out_dir: Path) -> None:
    payload = verify_bundle_common(out_dir)
    cmd = str(payload.get("command", "")).strip()

    if cmd == "validate":
        verify_validate(out_dir)
    elif cmd == "audit":
        verify_audit(out_dir)
    elif cmd == "cuped":
        verify_cuped(out_dir)
    elif cmd == "cuped-ratio":
        verify_cuped_ratio(out_dir)
    elif cmd == "causal-impact":
        verify_causal_impact(out_dir)
    else:
        _ok(f"{out_dir}: unknown command '{cmd}', only common checks applied")

    _ok(f"{out_dir}: OK")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python ci/verify_bundle.py <out_dir> [<out_dir> ...]")
        return 2

    out_dirs = [Path(a) for a in argv[1:]]
    for d in out_dirs:
        verify_one(d)

    _ok(f"OK ({len(out_dirs)} bundles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
