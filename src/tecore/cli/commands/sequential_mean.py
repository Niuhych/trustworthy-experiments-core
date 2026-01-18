from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tecore.cli.audit_api import write_audit_bundle
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
from tecore.sequential.reporting import make_sequential_plots, render_sequential_md


def _parse_looks_csv(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    return out


def cmd_sequential_mean(args) -> int:
    df = pd.read_csv(args.input)

    spec = SequentialSpec(
        group_col=str(getattr(args, "group_col", "group")),
        control_label=str(getattr(args, "control", "control")),
        test_label=str(getattr(args, "test", "test")),
        y_col=str(getattr(args, "y")),
        timestamp_col=str(getattr(args, "timestamp_col")) if getattr(args, "timestamp_col", None) else None,
        unit_col=str(getattr(args, "unit_col")) if getattr(args, "unit_col", None) else None,
    )

    mode = str(getattr(args, "mode", "group_sequential"))
    if mode not in {"group_sequential", "confidence_sequence"}:
        raise ValueError("--mode must be group_sequential or confidence_sequence")

    spending = str(getattr(args, "spending", "obrien_fleming"))
    if spending not in {"obrien_fleming", "pocock"}:
        raise ValueError("--spending must be obrien_fleming or pocock")

    effect_dir = str(getattr(args, "effect_direction", "two_sided"))
    if effect_dir not in {"two_sided", "increase", "decrease"}:
        raise ValueError("--effect-direction must be two_sided|increase|decrease")

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

    look_table, warn_prep = build_look_table_mean(df, spec, schedule, cfg)

    if cfg.mode == SequentialMode.GROUP_SEQUENTIAL:
        res = run_group_sequential(look_table, cfg)
    else:
        res = run_confidence_sequence(look_table, cfg)
    res.warnings = list(res.warnings) + list(warn_prep)

    out_dir = prepare_out_dir(getattr(args, "out", None), command="sequential-mean")
    write_run_meta(out_dir, vars(args), extra={"command": "sequential-mean"})

    artifacts: dict[str, Any] = {"report_md": "report.md", "plots": [], "tables": []}
    artifacts["tables"].append(write_table(out_dir, "look_table", res.look_table))

    plots = make_sequential_plots(res, cfg)
    for name, fig in plots.items():
        artifacts["plots"].append(save_plot(out_dir, name, fig))

    if bool(getattr(args, "audit", False)):
        write_audit_bundle(out_dir, df, schema="sequential_mean", parent_command="sequential-mean")
        artifacts["audit"] = {"audit_json": "audit.json", "audit_md": "audit.md"}

    report = render_sequential_md(res, cfg, spec)

    payload: dict[str, Any] = {
        "command": "sequential-mean",
        "inputs": {
            "input": args.input,
            "group_col": spec.group_col,
            "control": spec.control_label,
            "test": spec.test_label,
            "y": spec.y_col,
            "mode": cfg.mode.value,
            "alpha": cfg.alpha,
            "spending": cfg.spending.value,
            "effect_direction": cfg.effect_direction.value,
        },
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
