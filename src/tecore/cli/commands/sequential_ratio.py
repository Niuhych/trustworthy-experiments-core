from __future__ import annotations

import sys
from typing import Any

import numpy as np
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

from tecore.sequential.schema import (
    EffectDirection,
    LookSchedule,
    SequentialConfig,
    SequentialMode,
    SequentialSpec,
    SpendingFunction,
)
from tecore.sequential.preprocess import build_look_table_mean
from tecore.sequential.group_sequential import run_group_sequential
from tecore.sequential.confidence_sequences import run_confidence_sequence
from tecore.sequential.ratio import linearize_ratio_frame


def _warn(msg: str) -> None:
    print(f"[tecore][warn] {msg}", file=sys.stderr)


def _parse_looks(looks: str | None, n_looks: int | None, max_n: int | None) -> list[int]:
    if looks and str(looks).strip():
        parts = [p.strip() for p in str(looks).split(",") if p.strip()]
        out = [int(x) for x in parts]
        out = [x for x in out if x > 0]
        if not out:
            raise ValueError("--looks parsed to empty list")
        return sorted(list(dict.fromkeys(out)))

    if n_looks is not None and max_n is not None:
        K = int(n_looks)
        N = int(max_n)
        if K <= 0 or N <= 0:
            raise ValueError("--n-looks and --max-n must be > 0")
        grid = np.linspace(N / K, N, K)
        out = [int(round(x)) for x in grid]
        out = [x for x in out if x > 0]
        return sorted(list(dict.fromkeys(out)))

    raise ValueError("Provide either --looks OR (--n-looks and --max-n).")


def _plot_z_trajectory(tab: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt

    x = pd.to_numeric(tab["look_n"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(tab["z"], errors="coerce").to_numpy(dtype=float)

    bz = None
    if "boundary_z" in tab.columns:
        bz = pd.to_numeric(tab["boundary_z"], errors="coerce").to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)

    ax.plot(x, z, marker="o", label="z (diff / se)")
    if bz is not None and np.isfinite(bz).any():
        ax.plot(x, bz, linestyle="--", label="boundary (+)")
        ax.plot(x, -bz, linestyle="--", label="boundary (-)")

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("look_n")
    ax.set_ylabel("z")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    return fig


def _plot_effect_trajectory(tab: pd.DataFrame, title: str, use_cs: bool):
    import matplotlib.pyplot as plt

    x = pd.to_numeric(tab["look_n"], errors="coerce").to_numpy(dtype=float)
    diff = pd.to_numeric(tab["diff"], errors="coerce").to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)

    ax.plot(x, diff, marker="o", label="effect (linearized diff)")

    if use_cs and ("cs_low" in tab.columns) and ("cs_high" in tab.columns):
        lo = pd.to_numeric(tab["cs_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(tab["cs_high"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.2, label="CS band (time-uniform)")
    else:
        if "se" in tab.columns:
            se = pd.to_numeric(tab["se"], errors="coerce").to_numpy(dtype=float)
            lo = diff - 1.96 * se
            hi = diff + 1.96 * se
            ax.fill_between(x, lo, hi, alpha=0.2, label="~95% CI (fixed/naive)")

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("look_n")
    ax.set_ylabel("effect")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    return fig


def cmd_sequential_ratio(args) -> int:
    df = pd.read_csv(args.input)

    req = [args.group_col, args.num, args.den]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    looks = _parse_looks(getattr(args, "looks", None), getattr(args, "n_looks", None), getattr(args, "max_n", None))

    out_dir = prepare_out_dir(getattr(args, "out", None), command="sequential-ratio")
    write_run_meta(out_dir, vars(args), extra={"command": "sequential-ratio"})

    if getattr(args, "audit", False):
        write_audit_bundle(out_dir, df=df, schema="b2c_ratio", parent_command="sequential-ratio")

    spec_ratio = SequentialSpec(
        group_col=args.group_col,
        control_label=args.control,
        test_label=args.test,
        num_col=args.num,
        den_col=args.den,
        timestamp_col=getattr(args, "timestamp_col", "timestamp"),
        unit_col=getattr(args, "unit_col", None),
    )
    schedule = LookSchedule(looks=looks)

    baseline_mode = str(getattr(args, "baseline_mode", "first_look")).strip()
    df_lin, baseline_ratio, lin_warnings = linearize_ratio_frame(
        df=df,
        spec=spec_ratio,
        schedule=schedule,
        baseline_mode=baseline_mode,
        output_col="y_lin",
    )

    spec_mean = SequentialSpec(
        group_col=args.group_col,
        control_label=args.control,
        test_label=args.test,
        y_col="y_lin",
        timestamp_col=getattr(args, "timestamp_col", "timestamp"),
        unit_col=getattr(args, "unit_col", None),
    )

    mode = str(getattr(args, "mode", "group_sequential")).strip()
    if mode in {"group_sequential", "gs"}:
        seq_mode = SequentialMode.GROUP_SEQUENTIAL
    elif mode in {"confidence_sequence", "cs", "anytime"}:
        seq_mode = SequentialMode.CONFIDENCE_SEQUENCE
    else:
        raise ValueError(f"Unknown --mode: {mode}")

    spending = str(getattr(args, "spending", "obrien_fleming")).strip()
    if spending in {"obrien_fleming", "obf"}:
        spend = SpendingFunction.OBRIEN_FLEMING
    elif spending in {"pocock"}:
        spend = SpendingFunction.POCOCK
    else:
        raise ValueError(f"Unknown --spending: {spending}")

    ed = str(getattr(args, "effect_direction", "two_sided")).strip()
    if ed in {"two_sided", "two-sided"}:
        eff_dir = EffectDirection.TWO_SIDED
    elif ed in {"increase", "up"}:
        eff_dir = EffectDirection.INCREASE
    elif ed in {"decrease", "down"}:
        eff_dir = EffectDirection.DECREASE
    else:
        raise ValueError(f"Unknown --effect-direction: {ed}")

    cfg = SequentialConfig(
        mode=seq_mode,
        alpha=float(getattr(args, "alpha", 0.05)),
        two_sided=bool(getattr(args, "two_sided", True)),
        spending=spend,
        effect_direction=eff_dir,
        min_n_per_group=int(getattr(args, "min_n_per_group", 50)),
        var_floor=float(getattr(args, "var_floor", 1e-12)),
        seed=int(getattr(args, "seed", 0)) if getattr(args, "seed", None) is not None else 0,
    )

    look_table, warnings = build_look_table_mean(df_lin, spec_mean, schedule, cfg)

    if seq_mode == SequentialMode.GROUP_SEQUENTIAL:
        res = run_group_sequential(look_table, cfg)
    else:
        res = run_confidence_sequence(look_table, cfg)

    tab = res.look_table.copy()

    artifacts: dict[str, Any] = {"report_md": "report.md", "tables": [], "plots": []}
    artifacts["tables"].append(write_table(out_dir, "look_table", tab))

    fig_z = _plot_z_trajectory(tab, title="Sequential monitoring (ratio via linearization): z-trajectory + boundary")
    artifacts["plots"].append(save_plot(out_dir, "z_trajectory", fig_z))

    use_cs = (seq_mode == SequentialMode.CONFIDENCE_SEQUENCE)
    fig_eff = _plot_effect_trajectory(tab, title="Effect trajectory (linearized ratio diff)", use_cs=use_cs)
    artifacts["plots"].append(save_plot(out_dir, "effect_trajectory", fig_eff))

    if use_cs and ("cs_low" in tab.columns) and ("cs_high" in tab.columns):
        import matplotlib.pyplot as plt

        x = pd.to_numeric(tab["look_n"], errors="coerce").to_numpy(dtype=float)
        diff = pd.to_numeric(tab["diff"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(tab["cs_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(tab["cs_high"], errors="coerce").to_numpy(dtype=float)

        fig = plt.figure(figsize=(10, 5.5))
        ax = fig.add_subplot(111)
        ax.plot(x, diff, marker="o", label="effect")
        ax.fill_between(x, lo, hi, alpha=0.2, label="CS band (time-uniform)")
        ax.axhline(0, linewidth=1)
        ax.set_title("Confidence sequence band (ratio via linearization)")
        ax.set_xlabel("look_n")
        ax.set_ylabel("effect")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        fig.tight_layout()
        artifacts["plots"].append(save_plot(out_dir, "cs_band", fig))

    warn_lines = (res.warnings or []) + (warnings or []) + (lin_warnings or [])
    warn_lines = [str(w) for w in warn_lines if str(w).strip()]
    warn_block = ("- " + "\n- ".join(warn_lines)) if warn_lines else "(none)"

    report = f"""# tecore sequential-ratio report

## Inputs
- input: `{args.input}`
- group_col: `{args.group_col}`
- control: `{args.control}`
- test: `{args.test}`
- num: `{args.num}`
- den: `{args.den}`
- baseline_mode: `{baseline_mode}`
- baseline_ratio (control, first look): `{baseline_ratio}`
- looks: `{looks}`
- mode: `{mode}`
- alpha: `{cfg.alpha}`
- spending: `{getattr(args, "spending", "obrien_fleming")}`
- effect_direction: `{getattr(args, "effect_direction", "two_sided")}`
- min_n_per_group: `{cfg.min_n_per_group}`

## Decision
- stopped: `{bool(res.stopped)}`
- stop_look: `{res.stop_look}`
- decision: `{res.decision}`
- final_p_value: `{res.final_p_value}`

## Artifacts
- tables/look_table.csv
- plots/z_trajectory.png
- plots/effect_trajectory.png
{("- plots/cs_band.png" if use_cs else "")}

## Warnings
{warn_block}
"""
    write_report_md(out_dir, report)

    results_payload: dict[str, Any] = {
        "command": "sequential-ratio",
        "inputs": {
            "input": args.input,
            "group_col": args.group_col,
            "control": args.control,
            "test": args.test,
            "num": args.num,
            "den": args.den,
            "baseline_mode": baseline_mode,
            "looks": looks,
        },
        "estimates": {
            "baseline_ratio": float(baseline_ratio),
            "stopped": bool(res.stopped),
            "stop_look": res.stop_look,
            "decision": res.decision,
            "final_p_value": res.final_p_value,
        },
        "diagnostics": {},
        "warnings": warn_lines,
        "artifacts": artifacts,
    }
    write_results_json(out_dir, results_payload)

    if getattr(args, "strict", False) and warn_lines:
        _warn("Strict mode: warnings present => exiting with code 2.")
        return 2

    return 0
