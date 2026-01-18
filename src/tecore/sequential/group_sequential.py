from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import multivariate_normal, norm

from tecore.sequential.schema import EffectDirection, SequentialConfig, SequentialResult, SpendingFunction


def information_fraction(info: float, max_info: float) -> float:
    if not np.isfinite(info) or info <= 0 or not np.isfinite(max_info) or max_info <= 0:
        return np.nan
    return float(min(1.0, max(0.0, info / max_info)))


def boundary_obrien_fleming(alpha: float, t: float, two_sided: bool = True) -> float:
    """Classic O'Brien-Fleming z-boundary (approx)."""
    if not np.isfinite(t) or t <= 0:
        return np.nan
    a = float(alpha)
    if two_sided:
        z_alpha = float(norm.ppf(1 - a / 2))
    else:
        z_alpha = float(norm.ppf(1 - a))
    return float(z_alpha / np.sqrt(t))


def _corr_from_information(infos: List[float]) -> np.ndarray:
    """Canonical joint correlation for Z at information times (Brownian motion)."""
    t = np.asarray(infos, dtype=float)
    t = np.maximum(t, 1e-12)
    K = len(t)
    C = np.empty((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            ti, tj = (t[i], t[j])
            if ti <= tj:
                C[i, j] = np.sqrt(ti / tj)
            else:
                C[i, j] = np.sqrt(tj / ti)
    np.fill_diagonal(C, 1.0)
    return C


def _pocock_alpha_equation(c: float, cov: np.ndarray, alpha: float, two_sided: bool) -> float:
    K = cov.shape[0]
    if K == 1:
        a = alpha / 2 if two_sided else alpha
        return c - float(norm.ppf(1 - a))

    mvn = multivariate_normal(mean=np.zeros(K), cov=cov, allow_singular=False)
    if two_sided:
        try:
            p_inside = float(mvn.cdf(np.full(K, c), lower_limit=np.full(K, -c)))
        except TypeError:
            p_inside = float((norm.cdf(c) - norm.cdf(-c)) ** K)
        return (1.0 - p_inside) - float(alpha)

    try:
        p_inside_1s = float(mvn.cdf(np.full(K, c), lower_limit=np.full(K, -np.inf)))
    except TypeError:
        p_inside_1s = float((norm.cdf(c)) ** K)
    return (1.0 - p_inside_1s) - float(alpha)


def boundary_pocock(alpha: float, info_fracs: List[float], two_sided: bool = True) -> float:
    """Compute (approx) Pocock constant z boundary for K looks."""
    K = len(info_fracs)
    if K <= 0:
        return np.nan
    if K == 1:
        a = alpha / 2 if two_sided else alpha
        return float(norm.ppf(1 - a))

    cov = _corr_from_information(info_fracs)

    a = alpha / 2 if two_sided else alpha
    lo = float(norm.ppf(1 - a))
    hi = 6.0

    f = lambda c: _pocock_alpha_equation(float(c), cov=cov, alpha=float(alpha), two_sided=two_sided)

    try:
        root = optimize.brentq(f, lo, hi, maxiter=200)
        return float(root)
    except Exception:
        a_bonf = alpha / (2 * K) if two_sided else alpha / K
        return float(norm.ppf(1 - a_bonf))


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


def run_group_sequential(look_table: pd.DataFrame, cfg: SequentialConfig) -> SequentialResult:
    """Run group sequential monitoring on a precomputed look table."""
    warnings: List[str] = []
    if len(look_table) == 0:
        raise ValueError("look_table is empty")

    lt = look_table.copy()
    if "info" not in lt.columns or "z" not in lt.columns:
        raise ValueError("look_table must contain columns: info, z")

    max_info = float(np.nanmax(pd.to_numeric(lt["info"], errors="coerce").to_numpy(dtype=float)))
    info_fracs = [information_fraction(float(x), max_info) for x in lt["info"].to_numpy(dtype=float)]
    lt["t"] = info_fracs

    two_sided = bool(cfg.two_sided)
    direction = cfg.effect_direction
    if direction != EffectDirection.TWO_SIDED:
        two_sided = False

    if cfg.spending == SpendingFunction.OBRIEN_FLEMING:
        lt["boundary_z"] = [boundary_obrien_fleming(cfg.alpha, t, two_sided=two_sided) for t in info_fracs]
    elif cfg.spending == SpendingFunction.POCOCK:
        c = boundary_pocock(cfg.alpha, info_fracs=info_fracs, two_sided=two_sided)
        lt["boundary_z"] = float(c)
    else:
        warnings.append(f"Unknown spending '{cfg.spending}'. Falling back to OBF.")
        lt["boundary_z"] = [boundary_obrien_fleming(cfg.alpha, t, two_sided=two_sided) for t in info_fracs]

    z = pd.to_numeric(lt["z"], errors="coerce").to_numpy(dtype=float)
    if two_sided:
        lt["p_value"] = 2 * norm.sf(np.abs(z))
    else:
        if direction == EffectDirection.INCREASE:
            lt["p_value"] = norm.sf(z)
        elif direction == EffectDirection.DECREASE:
            lt["p_value"] = norm.sf(-z)
        else:
            lt["p_value"] = norm.sf(np.abs(z))

    stopped = False
    stop_look: Optional[int] = None

    for i in range(len(lt)):
        zi = float(lt.iloc[i]["z"])
        zc = float(lt.iloc[i]["boundary_z"])
        if _crosses(zi, zc, direction=direction):
            stopped = True
            stop_look = int(lt.iloc[i]["look_n"]) if "look_n" in lt.columns else int(i + 1)
            break

    decision = "reject" if stopped else "continue"
    final_row = lt.iloc[i] if stopped else lt.iloc[-1]
    final_p = float(final_row.get("p_value", np.nan)) if np.isfinite(final_row.get("p_value", np.nan)) else None

    diff = float(final_row.get("diff", np.nan))
    se = float(final_row.get("se", np.nan))

    if np.isfinite(diff) and np.isfinite(se) and se > 0:
        zcrit_fixed = float(norm.ppf(1 - cfg.alpha / 2)) if cfg.two_sided else float(norm.ppf(1 - cfg.alpha))
        final_ci = (float(diff - zcrit_fixed * se), float(diff + zcrit_fixed * se))
    else:
        final_ci = None

    lt["crossed"] = [
        _crosses(float(lt.iloc[j]["z"]), float(lt.iloc[j]["boundary_z"]), direction=direction) for j in range(len(lt))
    ]

    return SequentialResult(
        stopped=stopped,
        stop_look=stop_look,
        decision=decision,
        final_p_value=final_p,
        final_ci=final_ci,
        cs=None,
        look_table=lt,
        diagnostics={
            "mode": "group_sequential",
            "spending": str(cfg.spending.value),
            "two_sided": bool(cfg.two_sided),
            "effect_direction": str(cfg.effect_direction.value),
        },
        warnings=warnings,
    )
