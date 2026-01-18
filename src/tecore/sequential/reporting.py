from __future__ import annotations

from typing import Dict

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from tecore.sequential.schema import SequentialConfig, SequentialResult, SequentialSpec


def render_sequential_md(result: SequentialResult, cfg: SequentialConfig, spec: SequentialSpec) -> str:
    lt = result.look_table
    stop_str = f"Yes (look_n={result.stop_look})" if result.stopped else "No"
    p = result.final_p_value
    p_str = f"{p:.6g}" if (p is not None and np.isfinite(p)) else "(n/a)"
    ci_str = "(n/a)"
    if result.final_ci is not None:
        ci_str = f"[{result.final_ci[0]:.6g}, {result.final_ci[1]:.6g}]"
    cs_str = "(n/a)"
    if result.cs is not None:
        cs_str = f"[{result.cs[0]:.6g}, {result.cs[1]:.6g}]"

    warn_block = ("- " + "\n- ".join(result.warnings)) if result.warnings else "(none)"

    notes: list[str] = []
    if str(cfg.mode.value) == "confidence_sequence":
        notes.append(
            "This run uses a conservative anytime-valid confidence sequence based on a normal-mixture boundary "
            "with information time derived from the current standard error. It relies on a CLT-style approximation "
            "and independence/stability assumptions."
        )
    else:
        notes.append(
            "This run uses group sequential monitoring (K looks) with z-boundaries. The decision rule is based "
            "on boundary crossing; the displayed p-value is a naive z-test p-value for interpretability."
        )

    note_block = "\n".join([f"- {x}" for x in notes])

    return f"""# tecore sequential report

## Inputs
- group_col: `{spec.group_col}` (control=`{spec.control_label}`, test=`{spec.test_label}`)
- metric: `{spec.y_col or (spec.num_col + '/' + spec.den_col)}`
- mode: `{cfg.mode.value}`
- alpha: `{cfg.alpha}`

## Decision
- stopped early: **{stop_str}**
- decision: **{result.decision}**
- final p-value (mode-specific): `{p_str}`
- fixed (non-sequential) CI: `{ci_str}`
- confidence sequence (time-uniform) CI: `{cs_str}`

## Notes
{note_block}

## Warnings
{warn_block}

## Artifacts
- tables/look_table.csv
- plots/z_trajectory.png
- plots/effect_trajectory.png
"""


def make_z_trajectory_plot(result: SequentialResult) -> Figure:
    lt = result.look_table
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    x = lt["look_n"] if "look_n" in lt.columns else np.arange(1, len(lt) + 1)
    ax.plot(x, lt["z"], label="z")
    if "boundary_z" in lt.columns:
        ax.plot(x, np.abs(lt["boundary_z"]), linestyle="--", label="boundary")
        ax.plot(x, -np.abs(lt["boundary_z"]), linestyle="--")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Sequential z-trajectory")
    ax.set_xlabel("look_n")
    ax.set_ylabel("z")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_effect_trajectory_plot(result: SequentialResult, cfg: SequentialConfig) -> Figure:
    lt = result.look_table
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    x = lt["look_n"] if "look_n" in lt.columns else np.arange(1, len(lt) + 1)
    ax.plot(x, lt["diff"], label="estimate")
    if "cs_low" in lt.columns and "cs_high" in lt.columns:
        ax.fill_between(x, lt["cs_low"], lt["cs_high"], alpha=0.2, label="CS band")
    else:
        zcrit = 1.96 if cfg.two_sided else 1.645
        lo = lt["diff"] - zcrit * lt["se"]
        hi = lt["diff"] + zcrit * lt["se"]
        ax.fill_between(x, lo, hi, alpha=0.2, label="fixed CI")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Effect estimate trajectory")
    ax.set_xlabel("look_n")
    ax.set_ylabel("effect (test - control)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_sequential_plots(result: SequentialResult, cfg: SequentialConfig) -> Dict[str, Figure]:
    return {
        "z_trajectory": make_z_trajectory_plot(result),
        "effect_trajectory": make_effect_trajectory_plot(result, cfg),
    }
