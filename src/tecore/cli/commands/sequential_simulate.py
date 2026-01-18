from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tecore.cli.bundle import (
    prepare_out_dir,
    save_plot,
    write_report_md,
    write_results_json,
    write_run_meta,
    write_table,
)
from tecore.sequential import (
    EffectDirection,
    LookSchedule,
    SequentialConfig,
    SequentialMode,
    SequentialSpec,
    SpendingFunction,
)
from tecore.sequential.confidence_sequences import run_confidence_sequence
from tecore.sequential.group_sequential import run_group_sequential
from tecore.sequential.preprocess import build_look_table_mean
from tecore.sequential.ratio import linearize_ratio_frame
from tecore.sequential.reporting import make_sequential_plots, render_sequential_md
from tecore.sequential.simulate import SequentialSimConfig, simulate_ab_stream


def _parse_looks_csv(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def cmd_sequential_simulate(args) -> int:
    sim_cfg = SequentialSimConfig(
        n=int(getattr(args, "n", 20000)),
        effect=float(getattr(args, "effect", 0.0)),
        noise_sd=float(getattr(args, "noise_sd", 1.0)),
        heavy_tail=bool(getattr(args, "heavy_tail", False)),
        drift=bool(getattr(args, "drift", False)),
        seed=int(getattr(args, "seed", 42)),
        ratio=bool(getattr(args, "ratio", False)),
    )
    df = simulate_ab_stream(sim_cfg)

    mode = str(getattr(args, "mode", "group_sequential"))
    spending = str(getattr(args, "spending", "obrien_fleming"))
    effect_dir = str(getattr(args, "effect_direction", "two_sided"))

    looks: Optional[list[int]] = None
    n_looks: Optional[int] = None
    max_n: Optional[int] = None
    if getattr(args, "looks", None):
        looks = _parse_looks_csv(args.looks)
    else:
        n_looks = int(getattr(args, "n_looks"))
        max_n = int(getattr(args, "max_n"))

    schedule = LookSchedule(looks=looks, n_looks=n_looks, max_n=max_n)

    cfg = SequentialConfig(
        mode=SequentialMode(mode),
        alpha=float(getattr(args, "alpha", 0.05)),
        two_sided=bool(getattr(args, "two_sided", True)),
        spending=SpendingFunction(spending),
        effect_direction=EffectDirection(effect_dir),
        min_n_per_group=int(getattr(args, "min_n_per_group", 50)),
        var_floor=float(getattr(args, "var_floor", 1e-12)),
        cs_tau=float(getattr(args, "cs_tau", 1.0)),
        seed=int(getattr(args, "seed", 42)),
    )

    out_dir = prepare_out_dir(getattr(args, "out", None), command="sequential-simulate")
    write_run_meta(out_dir, vars(args), extra={"command": "sequential-simulate"})

    df.to_csv(out_dir / "data.csv", index=False)

    if not sim_cfg.ratio:
        spec = SequentialSpec(group_col="group", control_label="control", test_label="test", y_col="y", timestamp_col="timestamp")
        look_table, warn_prep = build_look_table_mean(df, spec, schedule, cfg)
        res = run_group_sequential(look_table, cfg) if cfg.mode == SequentialMode.GROUP_SEQUENTIAL else run_confidence_sequence(look_table, cfg)
        res.warnings = list(res.warnings) + list(warn_prep)
    else:
        spec_ratio = SequentialSpec(group_col="group", control_label="control", test_label="test", num_col="num", den_col="den", timestamp_col="timestamp")
        df_lin, r0, warn_lin = linearize_ratio_frame(df, spec_ratio, schedule)
        spec_mean = SequentialSpec(group_col="group", control_label="control", test_label="test", y_col="_y_lin", timestamp_col="timestamp")
        look_table, warn_prep = build_look_table_mean(df_lin, spec_mean, schedule, cfg)
        res = run_group_sequential(look_table, cfg) if cfg.mode == SequentialMode.GROUP_SEQUENTIAL else run_confidence_sequence(look_table, cfg)
        res.warnings = list(res.warnings) + list(warn_prep) + list(warn_lin)
        res.diagnostics = dict(res.diagnostics) | {"ratio_baseline_r0": float(r0)}
        spec = spec_ratio

    artifacts: dict[str, Any] = {"report_md": "report.md", "plots": [], "tables": [], "data": "data.csv"}
    artifacts["tables"].append(write_table(out_dir, "look_table", res.look_table))

    plots = make_sequential_plots(res, cfg)
    for name, fig in plots.items():
        artifacts["plots"].append(save_plot(out_dir, name, fig))

    report = render_sequential_md(res, cfg, spec)

    payload: dict[str, Any] = {
        "command": "sequential-simulate",
        "inputs": {"sim_config": sim_cfg.__dict__, "mode": cfg.mode.value, "alpha": cfg.alpha, "spending": cfg.spending.value},
        "estimates": {
            "decision": res.decision,
            "stopped": res.stopped,
            "stop_look": res.stop_look,
            "final_p_value": res.final_p_value,
            "final_ci": res.final_ci,
            "cs": res.cs,
        },
        "diagnostics": res.diagnostics,
        "warnings": res.warnings,
        "artifacts": artifacts,
    }

    write_results_json(out_dir, payload)
    write_report_md(out_dir, report)
    return 0
