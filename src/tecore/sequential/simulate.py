from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequentialSimConfig:
    n: int = 20000
    effect: float = 0.0
    noise_sd: float = 1.0
    heavy_tail: bool = False
    drift: bool = False
    seed: int = 42
    ratio: bool = False


def simulate_ab_stream(cfg: SequentialSimConfig) -> pd.DataFrame:
    """Generate a streaming-like user-level A/B dataset."""
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.n)

    unit_id = np.arange(n)
    group = np.where(rng.random(n) < 0.5, "control", "test")
    t = np.arange(n)

    if cfg.heavy_tail:
        noise = rng.standard_t(df=3, size=n) * float(cfg.noise_sd)
    else:
        noise = rng.normal(0.0, float(cfg.noise_sd), size=n)

    drift_term = (t / max(1, n - 1)) * (0.5 * float(cfg.noise_sd)) if cfg.drift else 0.0

    if not cfg.ratio:
        y = noise + drift_term
        y = y + (group == "test") * float(cfg.effect)
        df = pd.DataFrame({"unit_id": unit_id, "group": group, "y": y, "timestamp": t})
        return df

    den = rng.poisson(lam=3.0, size=n).astype(float) + 1.0
    base_rate = 1.0 + drift_term
    rate = base_rate + (group == "test") * float(cfg.effect)
    num = rate * den + noise
    num = np.maximum(num, 0.0)
    df = pd.DataFrame({"unit_id": unit_id, "group": group, "num": num, "den": den, "timestamp": t})
    return df
