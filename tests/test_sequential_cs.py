import numpy as np
import pandas as pd

from tecore.sequential.schema import (
    SequentialSpec,
    LookSchedule,
    SequentialConfig,
    SequentialMode,
    EffectDirection,
)
from tecore.sequential.preprocess import build_look_table_mean
from tecore.sequential.confidence_sequences import cs_boundary, run_confidence_sequence
from tecore.sequential.simulate import SequentialSimConfig, simulate_ab_stream


def test_cs_boundary_monotone_in_time():
    alpha = 0.05
    b1 = cs_boundary(t=0.2, alpha=alpha, two_sided=True)
    b2 = cs_boundary(t=0.8, alpha=alpha, two_sided=True)
    # Typically decreases with time (harder early)
    assert np.isfinite(b1) and np.isfinite(b2)
    assert b1 >= b2


def test_confidence_sequence_outputs_ci_and_p_anytime():
    N = 8000
    looks = [800, 1600, 2400, 3200, 4000, 6000, 8000]
    alpha = 0.05

    df = simulate_ab_stream(
        SequentialSimConfig(n=N, effect=0.0, noise_sd=1.0, heavy_tail=False, drift=False, seed=321, ratio=False)
    )

    spec = SequentialSpec(
        group_col="group",
        control_label="control",
        test_label="test",
        y_col="y",
        timestamp_col="timestamp",
    )
    schedule = LookSchedule(looks=looks)
    cfg = SequentialConfig(
        mode=SequentialMode.CONFIDENCE_SEQUENCE,
        alpha=alpha,
        two_sided=True,
        effect_direction=EffectDirection.TWO_SIDED,
        min_n_per_group=50,
        var_floor=1e-12,
        seed=0,
    )

    look_table, _ = build_look_table_mean(df, spec, schedule, cfg)
    res = run_confidence_sequence(look_table, cfg)

    tab = res.look_table
    assert isinstance(tab, pd.DataFrame)
    assert "p_anytime" in tab.columns
    assert "cs_low" in tab.columns and "cs_high" in tab.columns

    p = pd.to_numeric(tab["p_anytime"], errors="coerce").to_numpy(dtype=float)
    assert np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0
