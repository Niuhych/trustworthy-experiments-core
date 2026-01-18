from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from tecore.sequential.schema import EffectDirection, SequentialConfig, SequentialResult


def _mixture_martingale(z: float, info: float, tau: float) -> float:
    """Normal-mixture nonnegative martingale for Brownian motion."""
    if not np.isfinite(z) or not np.isfinite(info) or info <= 0:
        return 1.0
    t = float(info)
    tt = float(max(1e-12, tau))
    denom = 1.0 + (tt**2) * t
    b2 = float((z**2) * t)
    return float((denom ** (-0.5)) * np.exp(((tt**2) * b2) / (2.0 * denom)))


def cs_boundary(info: float, alpha: float, *, tau: float = 1.0, two_sided: bool = True) -> float:
    """Anytime-valid critical value for |Z| at information time."""
    if not np.isfinite(info) or info <= 0:
        return np.nan
    a = float(alpha)
    if not two_sided:
        a = float(alpha)

    t = float(info)
    tt = float(max(1e-12, tau))
    denom = 1.0 + (tt**2) * t
    target = np.log((1.0 / a) * np.sqrt(denom))
    if target <= 0:
        return 0.0
    z2 = (2.0 * denom * target) / ((tt**2) * t)
    return float(np.sqrt(max(0.0, z2)))


def _crosses(z: float, zcrit: float, direction: EffectDirection) -> bool:
    if not np.isfinite(z) or not np.isfinite(zcrit):
        return False
    if direction == EffectDirection.TWO_SIDED:
        return abs(z) >= zcrit
    if direction == EffectDirection.INCREASE:
        return z >= zcrit
    if direction == EffectDirection.DECREASE:
        return z <= -zcrit
    return abs(z) >= zcrit


def run_confidence_sequence(look_table: pd.DataFrame, cfg: SequentialConfig) -> SequentialResult:
    """Run conservative anytime-valid monitoring (confidence sequence)."""
    warnings: List[str] = []
    if len(look_table) == 0:
        raise ValueError("look_table is empty")

    lt = look_table.copy()
    if "info" not in lt.columns or "z" not in lt.columns:
        raise ValueError("look_table must contain columns: info, z")

    direction = cfg.effect_direction
    two_sided = bool(cfg.two_sided)
    if direction != EffectDirection.TWO_SIDED:
        two_sided = False

    infos = pd.to_numeric(lt["info"], errors="coerce").to_numpy(dtype=float)
    zs = pd.to_numeric(lt["z"], errors="coerce").to_numpy(dtype=float)

    bounds = [
        cs_boundary(info=float(infos[i]), alpha=float(cfg.alpha), tau=float(cfg.cs_tau), two_sided=two_sided)
        for i in range(len(lt))
    ]
    lt["boundary_z"] = bounds

    diffs = pd.to_numeric(lt["diff"], errors="coerce").to_numpy(dtype=float)
    ses = pd.to_numeric(lt["se"], errors="coerce").to_numpy(dtype=float)
    lt["cs_low"] = diffs - np.asarray(bounds, dtype=float) * ses
    lt["cs_high"] = diffs + np.asarray(bounds, dtype=float) * ses

    Ms = np.array(
        [_mixture_martingale(z=float(zs[i]), info=float(infos[i]), tau=float(cfg.cs_tau)) for i in range(len(lt))],
        dtype=float,
    )
    Ms = np.maximum(Ms, 1.0)
    running_max = np.maximum.accumulate(Ms)
    p_any = np.minimum(1.0, 1.0 / running_max)
    lt["m_value"] = Ms
    lt["p_anytime"] = p_any
    lt["p_value"] = p_any

    stopped = False
    stop_look: Optional[int] = None
    stop_idx = len(lt) - 1
    for i in range(len(lt)):
        if _crosses(float(zs[i]), float(bounds[i]), direction=direction):
            stopped = True
            stop_idx = i
            stop_look = int(lt.iloc[i]["look_n"]) if "look_n" in lt.columns else int(i + 1)
            break

    decision = "reject" if stopped else "continue"
    final_row = lt.iloc[stop_idx]
    final_p = float(final_row.get("p_anytime", np.nan)) if np.isfinite(final_row.get("p_anytime", np.nan)) else None

    diff = float(final_row.get("diff", np.nan))
    se = float(final_row.get("se", np.nan))
    if np.isfinite(diff) and np.isfinite(se) and se > 0:
        zcrit_fixed = float(norm.ppf(1 - cfg.alpha / 2)) if cfg.two_sided else float(norm.ppf(1 - cfg.alpha))
        final_ci = (float(diff - zcrit_fixed * se), float(diff + zcrit_fixed * se))
        cs = (float(final_row.get("cs_low")), float(final_row.get("cs_high")))
    else:
        final_ci = None
        cs = None

    lt["crossed"] = [_crosses(float(zs[i]), float(bounds[i]), direction=direction) for i in range(len(lt))]

    return SequentialResult(
        stopped=stopped,
        stop_look=stop_look,
        decision=decision,
        final_p_value=final_p,
        final_ci=final_ci,
        cs=cs,
        look_table=lt,
        diagnostics={
            "mode": "confidence_sequence",
            "cs_tau": float(cfg.cs_tau),
            "two_sided": bool(cfg.two_sided),
            "effect_direction": str(cfg.effect_direction.value),
        },
        warnings=warnings,
    )
