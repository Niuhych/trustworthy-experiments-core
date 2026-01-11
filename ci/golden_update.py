#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


VOLATILE_FILES = {
    "run_meta.json",  
}

TEXT_EXT = {".json", ".csv", ".md", ".txt", ".yaml", ".yml"}


DEFAULT_BUNDLES = {
    "validate_user_level": "out/validate_user_level",
    "validate_ratio": "out/validate_ratio",
    "audit_user_level": "out/audit_user_level",
    "audit_ratio": "out/audit_ratio",
    "run_mean": "out/run_mean",
    "run_ratio": "out/run_ratio",
    "yaml_run_mean": "out/yaml_run_mean",
    "yaml_run_ratio": "out/yaml_run_ratio",
    "yaml_ci_001": "out/yaml_ci_001",
}


def _die(msg: str) -> None:
    print(f"[golden_update][FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[golden_update] {msg}")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_hash_for_file(p: Path) -> str:
    """
    Normalized content hash for text-ish artifacts:
      - .json: parse + dump sorted keys
      - others: normalize newlines
    For binary artifacts: raw sha256
    """
    if p.name in VOLATILE_FILES:
        return ""

    ext = p.suffix.lower()
    if ext == ".json":
        try:
            obj = json.loads(_read_text(p))
        except Exception:
            return _sha256_bytes(_read_text(p).encode("utf-8"))
        stable = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return _sha256_bytes(stable.encode("utf-8"))

    if ext in TEXT_EXT:
        return _sha256_bytes(_read_text(p).encode("utf-8"))

    return _sha256_file(p)


def _collect_manifest(bundle_dir: Path) -> dict[str, Any]:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        _die(f"Bundle dir missing: {bundle_dir}")

    files = []
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(bundle_dir).as_posix()
            files.append(
                {
                    "path": rel,
                    "size": int(p.stat().st_size),
                    "volatile": (p.name in VOLATILE_FILES),
                    "hash": _norm_hash_for_file(p),
                }
            )

    return {"files": files}


def _copy_bundle(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for p in src.rglob("*"):
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def main() -> int:
    ap = argparse.ArgumentParser(prog="golden_update", description="Update golden snapshots from out bundles.")
    ap.add_argument("--golden-dir", default="ci/golden", help="Directory to store golden snapshots.")
    ap.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Override/add bundles: name=path (e.g. run_mean=out/run_mean). Can be repeated.",
    )
    ap.add_argument("--use-defaults", action="store_true", help="Use default bundle mapping.")
    args = ap.parse_args()

    mapping: dict[str, str] = {}
    if args.use_defaults:
        mapping.update(DEFAULT_BUNDLES)

    for item in args.bundle:
        if "=" not in item:
            _die(f"Bad --bundle format (expected name=path): {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            _die(f"Bad --bundle value: {item}")
        mapping[name] = path

    if not mapping:
        _die("No bundles specified. Use --use-defaults or --bundle name=path.")

    golden_root = Path(args.golden_dir)
    golden_root.mkdir(parents=True, exist_ok=True)

    updated = 0
    for name, path in mapping.items():
        src = Path(path)
        if not src.exists():
            _die(f"Bundle not found: {name} -> {src}")

        dest_dir = golden_root / name
        dest_bundle = dest_dir / "bundle"

        _ok(f"Updating {name} from {src}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        _copy_bundle(src, dest_bundle)

        manifest = _collect_manifest(dest_bundle)
        (dest_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        updated += 1

    _ok(f"Updated {updated} golden bundle(s) under {golden_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
