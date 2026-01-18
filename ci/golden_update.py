import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


BUNDLES = {
    "run_mean": {
        "required_files": ["results.json", "report.md", "run_meta.json", "audit.json", "audit.md"],
        "results_checks": {
            "estimates.base.diff": {"rtol": 1e-6, "atol": 1e-6},
            "estimates.cuped.diff": {"rtol": 1e-6, "atol": 1e-6},
            "diagnostics.theta": {"rtol": 1e-6, "atol": 1e-6},
        },
        "csv_checks": [
            {"rel": "tables/summary.csv", "rtol": 1e-6, "atol": 1e-8, "key_cols": ["method"]},
        ],
        "plot_files": [
            "plots/y_post_by_group.png",
            "plots/x_vs_y_scatter.png",
        ],
    },
    "run_ratio": {
        "required_files": [
            "results.json",
            "report.md",
            "run_meta.json",
            "tables/summary.csv",
            "audit.json",
            "audit.md",
        ],
        "results_checks": {
            "estimates.base.diff_linearized": {"rtol": 1e-6, "atol": 1e-6},
            "estimates.cuped.diff_linearized": {"rtol": 1e-6, "atol": 1e-6},
            "diagnostics.theta": {"rtol": 1e-6, "atol": 1e-6},
        },
        # Формат summary.csv уже менялся → не фиксируем CSV-гольденом, держим контракт на results.json.
        "csv_checks": [],
        "plot_files": [
            "plots/ratio_post_by_group.png",
            "plots/linearized_by_group.png",
        ],
    },
    "run_sequential_mean": {
        "required_files": [
            "results.json",
            "report.md",
            "run_meta.json",
            "tables/look_table.csv",
        ],
        "results_checks": {
            "estimates.stop_look": {"rtol": 0.0, "atol": 0.0},
            "estimates.final_p_value": {"rtol": 1e-6, "atol": 1e-8},
        },
        "csv_checks": [],
        "plot_files": [
            "plots/z_trajectory.png",
            "plots/effect_trajectory.png",
        ],
    },
    "run_sequential_ratio": {
        "required_files": [
            "results.json",
            "report.md",
            "run_meta.json",
            "tables/look_table.csv",
        ],
        "results_checks": {
            "estimates.stop_look": {"rtol": 0.0, "atol": 0.0},
            "inputs.r0": {"rtol": 1e-6, "atol": 1e-8},
            "estimates.final_p_value": {"rtol": 1e-6, "atol": 1e-8},
        },
        "csv_checks": [],
        "plot_files": [
            "plots/z_trajectory.png",
            "plots/effect_trajectory.png",
        ],
    },
    "yaml_ci_001": {
        "required_files": [
            "results.json",
            "report.md",
            "run_meta.json",
            "tables/effect_series.csv",
            "audit.json",
            "audit.md",
        ],
        "results_checks": {
            "estimates.summary.point_effect": {"rtol": 1e-5, "atol": 1e-5},
            "estimates.summary.cum_effect": {"rtol": 1e-5, "atol": 1e-5},
        },
        # effect_series.csv потенциально “дрожит” → existence-only + проверяем стабильные summary числа.
        "csv_checks": [],
        "plot_files": [
            "plots/observed_vs_counterfactual.png",
            "plots/point_effect.png",
            "plots/cumulative_effect.png",
        ],
    },
}


def _copy_tree(src: Path, dst: Path, rels: list[str]) -> None:
    for rel in rels:
        s = src / rel
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if not s.exists():
            raise FileNotFoundError(f"missing in out bundle: {s}")
        shutil.copy2(s, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="Path to out/ directory")
    ap.add_argument("--golden", default="ci/golden", help="Path to ci/golden")
    ap.add_argument("--bundles", nargs="*", default=["run_mean", "run_ratio", "run_sequential_mean", "run_sequential_ratio", "yaml_ci_001"])
    args = ap.parse_args()

    out_root = Path(args.out)
    golden_root = Path(args.golden)
    golden_root.mkdir(parents=True, exist_ok=True)

    for name in args.bundles:
        if name not in BUNDLES:
            raise SystemExit(f"Unknown bundle name: {name}")

        spec_cfg = BUNDLES[name]
        out_dir = out_root / name
        if not out_dir.exists():
            raise SystemExit(f"Missing out bundle: {out_dir}")

        results = _read_json(out_dir / "results.json")

        expected_dir = golden_root / name
        if expected_dir.exists():
            shutil.rmtree(expected_dir)
        expected_dir.mkdir(parents=True, exist_ok=True)

        rels = []
        rels += list(spec_cfg.get("required_files", []))
        rels += [x["rel"] for x in spec_cfg.get("csv_checks", [])]
        rels += list(spec_cfg.get("plot_files", []))
        _copy_tree(out_dir, expected_dir, rels)

        out_spec = {
            "required_files": spec_cfg.get("required_files", []),
            "results_checks": {},
            "csv_checks": spec_cfg.get("csv_checks", []),
            "plot_files": spec_cfg.get("plot_files", []),
        }
        for path, rule in (spec_cfg.get("results_checks") or {}).items():
            val = _get_by_path(results, path)
            out_spec["results_checks"][path] = {"value": val, **rule}

        (expected_dir / "expected.json").write_text(
            json.dumps(out_spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"[golden_update] updated {expected_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
