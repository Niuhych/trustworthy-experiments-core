from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from tecore.io.reader import read_csv
from tecore.metrics.ratio import ratio_point, linearize_ratio
from tecore.variance_reduction import cuped_split_adjust
from tecore.report.render import render_cuped_report


def _welch_pvalue(y_c: np.ndarray, y_t: np.ndarray) -> float:
    _, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    return float(p)


def cmd_cuped_ratio(args) -> int:
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be in (0, 1).")

    df = read_csv(args.input)

    g = df[args.group_col].astype(str).to_numpy()
    mask_c = g == args.control
    mask_t = g == args.test

    if mask_c.sum() == 0 or mask_t.sum() == 0:
        raise ValueError("Empty control/test group. Check --group-col/--control/--test.")

    num = df[args.num].to_numpy(float)
    den = df[args.den].to_numpy(float)
    num_pre = df[args.num_pre].to_numpy(float)
    den_pre = df[args.den_pre].to_numpy(float)

    # Baselines from control only (standard approach)
    r0_post = ratio_point(num[mask_c], den[mask_c])
    r0_pre = ratio_point(num_pre[mask_c], den_pre[mask_c])

    z = linearize_ratio(num, den, r0_post)
    z_pre = linearize_ratio(num_pre, den_pre, r0_pre)

    z_c, z_t = z[mask_c], z[mask_t]
    zpre_c, zpre_t = z_pre[mask_c], z_pre[mask_t]

    p_base = _welch_pvalue(z_c, z_t)

    cup_c, cup_t = cuped_split_adjust(z_c, zpre_c, z_t, zpre_t)
    p_cuped = _welch_pvalue(cup_c.y_adj, cup_t.y_adj)

    if not np.isfinite(p_base):
        raise ValueError("Base p-value is not finite (check variance of linearized ratio).")
    if not np.isfinite(p_cuped):
        raise ValueError("CUPED p-value is not finite (check variance after adjustment).")

    reject_base = p_base < args.alpha
    reject_cuped = p_cuped < args.alpha

    out = {
        "input": args.input,
        "group_col": args.group_col,
        "control": args.control,
        "test": args.test,
        "metric": "ratio_linearized",
        "num": args.num,
        "den": args.den,
        "num_pre": args.num_pre,
        "den_pre": args.den_pre,
        "alpha": args.alpha,
        "n_control": int(mask_c.sum()),
        "n_test": int(mask_t.sum()),
        "ratio0_post_control": float(r0_post),
        "ratio0_pre_control": float(r0_pre),
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
