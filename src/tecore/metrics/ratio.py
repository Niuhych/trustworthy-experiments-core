from __future__ import annotations

import numpy as np


def ratio_point(num: np.ndarray, den: np.ndarray) -> float:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    sden = float(np.sum(den))
    if sden <= 0:
        raise ValueError("Denominator sum must be > 0.")
    return float(np.sum(num) / sden)


def linearize_ratio(num: np.ndarray, den: np.ndarray, r0: float) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return num - r0 * den
