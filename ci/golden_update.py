from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BUNDLES = ["run_mean", "run_ratio", "yaml_ci_001"]


def _copy_bundle(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"[golden_update][FAIL] missing bundle dir: {src}")

    if dst.exists():
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Golden root dir, e.g. ci/golden")
    ap.add_argument("--out", required=True, help="Out root dir, e.g. out")
    args = ap.parse_args()

    golden_root = Path(args.golden)
    out_root = Path(args.out)

    golden_root.mkdir(parents=True, exist_ok=True)

    for b in BUNDLES:
        src = out_root / b
        dst = golden_root / b
        _copy_bundle(src, dst)
        print(f"[golden_update] updated: {b}")

    print("[golden_update] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
