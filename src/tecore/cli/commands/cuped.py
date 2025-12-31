from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from tecore.io.reader import read_csv
from tecore.variance_reduction import cuped_split_adjust
from tecore.report.render import render_cuped_report


def _winsorize_upper(a: np.ndarray, q: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    cap = float(np.quantile(a, q))
    return np.minimum(a, cap)


def _transform(vec: np.ndarray, mode: str, winsor_q: float) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    if mode == "raw":
        return v
    if mode == "winsor":
        return _winsorize_upper(v, winsor_q)
    if mode == "log1p":
        return np.log1p(v)
    raise ValueError(f"Unknown transform: {mode}")


def _welch_pvalue(y_c: np.ndarray, y_t: np.ndarray) -> float:
    _, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    return float(p)


def cmd_cuped(args) -> int:
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be in (0, 1).")
    
    df = read_csv(args.input)

    g = df[args.group_col].astype(str).to_numpy()
    mask_c = g == args.control
    mask_t = g == args.test

    if mask_c.sum() == 0 or mask_t.sum() == 0:
        raise ValueError("Empty control/test group. Check --group-col/--control/--test.")

    y = _transform(df[args.y].to_numpy(), args.transform, args.winsor_q)
    x = _transform(df[args.x].to_numpy(), args.transform, args.winsor_q)

    y_c, y_t = y[mask_c], y[mask_t]
    x_c, x_t = x[mask_c], x[mask_t]

    p_base = _welch_pvalue(y_c, y_t)

    cup_c, cup_t = cuped_split_adjust(y_c, x_c, y_t, x_t)
    p_cuped = _welch_pvalue(cup_c.y_adj, cup_t.y_adj)

    if not np.isfinite(p_base):
        raise ValueError("Base p-value is not finite (check that Y has variance in both groups).")
    if not np.isfinite(p_cuped):
        raise ValueError("CUPED p-value is not finite (check variance after adjustment).")

    reject_base = p_base < args.alpha
    reject_cuped = p_cuped < args.alpha

    out = {
        "input": args.input,
        "group_col": args.group_col,
        "control": args.control,
        "test": args.test,
        "y": args.y,
        "x": args.x,
        "transform": args.transform,
        "winsor_q": args.winsor_q if args.transform == "winsor" else None,
        "alpha": args.alpha,
        "n_control": int(mask_c.sum()),
        "n_test": int(mask_t.sum()),
        "p_value_base": float(p_base),
        "p_value_cuped": float(p_cuped),
        "theta": float(cup_c.theta),
        "var_reduction_control": float(cup_c.var_reduction),
        "var_reduction_test": float(cup_t.var_reduction),
        "reject_base": bool(reject_base),
        "reject_cuped": bool(reject_cuped),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_cuped_report(out), encoding="utf-8")

    return 0
