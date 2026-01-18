from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Build out/ bundles (same as CI).")
    p.add_argument("--out", default="out", help="Output directory (default: out)")
    p.add_argument("--clean", action="store_true", help="Remove out/ before building")
    p.add_argument("--print-tree", action="store_true", help="Print out/ file tree after build")
    args = p.parse_args()

    out = Path(args.out)

    if args.clean and out.exists():
        shutil.rmtree(out)

    run(["tecore", "--help"])
    run(["tecore", "version"])

    run(["tecore", "validate", "--input", "examples/example_user_level.csv", "--schema", "b2c_user_level", "--out", str(out / "validate_user_level")])
    run(["tecore", "validate", "--input", "examples/example_ratio.csv", "--schema", "b2c_ratio", "--out", str(out / "validate_ratio")])

    run(["tecore", "audit", "--input", "examples/example_user_level.csv", "--schema", "b2c_user_level", "--out", str(out / "audit_user_level")])
    run(["tecore", "audit", "--input", "examples/example_ratio.csv", "--schema", "b2c_ratio", "--out", str(out / "audit_ratio")])

    run([
        "tecore", "cuped",
        "--input", "examples/example_user_level.csv",
        "--y", "revenue", "--x", "revenue_pre",
        "--out", str(out / "run_mean"),
        "--audit",
    ])

    run([
        "tecore", "cuped-ratio",
        "--input", "examples/example_ratio.csv",
        "--num", "revenue", "--den", "sessions",
        "--num-pre", "revenue_pre", "--den-pre", "sessions_pre",
        "--out", str(out / "run_ratio"),
        "--audit",
    ])

    run([
        "tecore", "sequential-simulate",
        "--n", "12000",
        "--effect", "0.05",
        "--noise-sd", "1.0",
        "--seed", "123",
        "--looks", "2000,4000,6000,8000,10000,12000",
        "--mode", "group_sequential",
        "--spending", "obrien_fleming",
        "--out", str(out / "sequential_simulate_mean"),
    ])

    run([
        "tecore", "sequential-mean",
        "--input", str(out / "sequential_simulate_mean" / "data.csv"),
        "--y", "y",
        "--looks", "2000,4000,6000,8000,10000,12000",
        "--mode", "group_sequential",
        "--spending", "obrien_fleming",
        "--alpha", "0.05",
        "--min-n-per-group", "50",
        "--out", str(out / "run_sequential_mean"),
    ])

    run([
        "tecore", "sequential-simulate",
        "--n", "20000",
        "--effect", "0.03",
        "--noise-sd", "1.0",
        "--seed", "41",
        "--ratio",
        "--looks", "2000,4000,6000,8000,12000,16000,20000",
        "--mode", "group_sequential",
        "--spending", "obrien_fleming",
        "--out", str(out / "sequential_simulate_ratio"),
    ])

    run([
        "tecore", "sequential-ratio",
        "--input", str(out / "sequential_simulate_ratio" / "data.csv"),
        "--num", "num",
        "--den", "den",
        "--baseline-mode", "first_look",
        "--looks", "2000,4000,6000,8000,12000,16000,20000",
        "--mode", "group_sequential",
        "--spending", "obrien_fleming",
        "--alpha", "0.05",
        "--min-n-per-group", "50",
        "--out", str(out / "run_sequential_ratio"),
    ])
    
    run(["tecore", "run", "--config", "configs/templates/cuped_example.yaml"])
    run(["tecore", "run", "--config", "configs/templates/cuped_ratio_example.yaml"])
    run(["tecore", "run", "--config", "configs/templates/causal_impact_example.yaml"])
    run(["tecore", "run", "--config", "configs/templates/sequential_mean_example.yaml"])
    run(["tecore", "run", "--config", "configs/templates/sequential_ratio_example.yaml"])
    run(["tecore", "run", "--config", "configs/templates/sequential_simulate_then_run.yaml"])

    if args.print_tree:
        print("=== out tree (files) ===")
        for f in sorted(out.rglob("*")):
            if f.is_file():
                try:
                    rel = f.relative_to(Path.cwd())
                except Exception:
                    rel = f
                print(rel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
