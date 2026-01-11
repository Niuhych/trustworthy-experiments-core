from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_TOP_LEVEL = ["command", "inputs", "estimates", "diagnostics", "warnings", "artifacts"]
REQUIRED_ARTIFACTS = ["report_md", "plots", "tables"]


def fail(msg: str) -> int:
    print(f"[verify_bundle][error] {msg}", file=sys.stderr)
    return 2


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} ({e})") from e


def verify_one(out_dir: Path) -> None:
    if not out_dir.exists():
        raise RuntimeError(f"Missing out dir: {out_dir}")

    for fname in ["run_meta.json", "results.json", "report.md"]:
        p = out_dir / fname
        if not p.exists():
            raise RuntimeError(f"Missing required file: {p}")

    results = load_json(out_dir / "results.json")

    for k in REQUIRED_TOP_LEVEL:
        if k not in results:
            raise RuntimeError(f"{out_dir}/results.json missing key: {k}")

    artifacts = results["artifacts"]
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"{out_dir}/results.json artifacts must be dict")

    for k in REQUIRED_ARTIFACTS:
        if k not in artifacts:
            raise RuntimeError(f"{out_dir}/results.json artifacts missing key: {k}")

    report_rel = artifacts["report_md"]
    if not isinstance(report_rel, str) or not report_rel:
        raise RuntimeError(f"{out_dir}/results.json artifacts.report_md must be non-empty string")
    if not (out_dir / report_rel).exists():
        raise RuntimeError(f"Missing report file referenced by results.json: {out_dir / report_rel}")

    if not isinstance(artifacts["plots"], list):
        raise RuntimeError(f"{out_dir}/results.json artifacts.plots must be list")
    if not isinstance(artifacts["tables"], list):
        raise RuntimeError(f"{out_dir}/results.json artifacts.tables must be list")

    for rel in artifacts["plots"]:
        if not (out_dir / rel).exists():
            raise RuntimeError(f"Missing plot referenced by results.json: {out_dir / rel}")
    for rel in artifacts["tables"]:
        if not (out_dir / rel).exists():
            raise RuntimeError(f"Missing table referenced by results.json: {out_dir / rel}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return fail("Usage: python ci/verify_bundle.py <out_dir1> [<out_dir2> ...]")

    out_dirs = [Path(x) for x in argv[1:]]
    try:
        for d in out_dirs:
            verify_one(d)
    except Exception as e:
        return fail(str(e))

    print(f"[verify_bundle] OK ({len(out_dirs)} bundles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
