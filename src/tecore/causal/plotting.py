from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_observed_vs_counterfactual(effect_df: pd.DataFrame, outpath: Path, title: str = "Observed vs Counterfactual") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(effect_df["date"])
    y = effect_df["y"].astype(float).values
    y_hat = effect_df["y_hat"].astype(float).values

    plt.figure()
    plt.plot(x, y, label="observed")
    plt.plot(x, y_hat, label="counterfactual")
    if "y_hat_lower" in effect_df.columns and "y_hat_upper" in effect_df.columns:
        lo = effect_df["y_hat_lower"].astype(float).values
        hi = effect_df["y_hat_upper"].astype(float).values
        plt.fill_between(x, lo, hi, alpha=0.2, label="counterfactual CI")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_point_effect(effect_df: pd.DataFrame, outpath: Path, title: str = "Point Effect (Observed - Counterfactual)") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(effect_df["date"])
    e = effect_df["effect"].astype(float).values

    plt.figure()
    plt.plot(x, e, label="point effect")
    if "effect_lower" in effect_df.columns and "effect_upper" in effect_df.columns:
        lo = effect_df["effect_lower"].astype(float).values
        hi = effect_df["effect_upper"].astype(float).values
        plt.fill_between(x, lo, hi, alpha=0.2, label="effect CI")
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_cumulative_effect(effect_df: pd.DataFrame, outpath: Path, title: str = "Cumulative Effect") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(effect_df["date"])
    ce = effect_df["cum_effect"].astype(float).values

    plt.figure()
    plt.plot(x, ce, label="cumulative effect")
    if "cum_lower" in effect_df.columns and "cum_upper" in effect_df.columns:
        lo = effect_df["cum_lower"].astype(float).values
        hi = effect_df["cum_upper"].astype(float).values
        plt.fill_between(x, lo, hi, alpha=0.2, label="cum CI")
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_placebo_hist(placebo_df: pd.DataFrame, true_cum_effect: float, outpath: Path, title: str = "Placebo-in-time: cumulative effect") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = placebo_df["cum_effect"].astype(float).values
    x = x[np.isfinite(x)]

    plt.figure()
    plt.hist(x, bins=15, alpha=0.7, label="placebo")
    plt.axvline(true_cum_effect, linewidth=2, label="true")
    plt.axvline(-true_cum_effect, linewidth=1, linestyle="--")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
