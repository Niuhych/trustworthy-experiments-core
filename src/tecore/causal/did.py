from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DidResult:
    point_effect: float
    cum_effect: float
    point_ci: Tuple[float, float]
    cum_ci: Tuple[float, float]
    diagnostics: Dict[str, float]


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if block_size <= 1:
        return rng.integers(0, n, size=n)

    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    idx = []
    for s in starts:
        idx.extend(list(range(s, min(s + block_size, n))))
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def did_time_series(
    treated: np.ndarray,
    control: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    alpha: float,
    bootstrap_iters: int,
    block_size: int,
    random_state: int,
) -> DidResult:
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)

    diff = treated - control

    pre = diff[pre_mask]
    post = diff[post_mask]

    # standard DID on aggregated series: change in diff
    point = float(np.mean(post) - np.mean(pre))
    cum = float(np.sum(post - np.mean(pre)))

    rng = np.random.default_rng(random_state)

    # bootstrap CI over time indices, separately in pre and post blocks
    pre_ci_samples = []
    cum_ci_samples = []

    pre_idx_base = np.arange(len(pre))
    post_idx_base = np.arange(len(post))

    for _ in range(int(bootstrap_iters)):
        bi_pre = _block_bootstrap_indices(len(pre), block_size, rng)
        bi_post = _block_bootstrap_indices(len(post), block_size, rng)

        pre_b = pre[bi_pre]
        post_b = post[bi_post]

        point_b = float(np.mean(post_b) - np.mean(pre_b))
        cum_b = float(np.sum(post_b - np.mean(pre_b)))

        pre_ci_samples.append(point_b)
        cum_ci_samples.append(cum_b)

    lo = alpha / 2
    hi = 1 - alpha / 2

    point_ci = (float(np.quantile(pre_ci_samples, lo)), float(np.quantile(pre_ci_samples, hi)))
    cum_ci = (float(np.quantile(cum_ci_samples, lo)), float(np.quantile(cum_ci_samples, hi)))

    diags = {
        "mean_diff_pre": float(np.mean(pre)),
        "mean_diff_post": float(np.mean(post)),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
    }

    return DidResult(
        point_effect=point,
        cum_effect=cum,
        point_ci=point_ci,
        cum_ci=cum_ci,
        diagnostics=diags,
    )
