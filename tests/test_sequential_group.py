import numpy as np
import pandas as pd

from tecore.sequential.schema import (
    SequentialSpec,
    LookSchedule,
    SequentialConfig,
    SequentialMode,
    SpendingFunction,
    EffectDirection,
)
from tecore.sequential.preprocess import build_look_table_mean
from tecore.sequential.group_sequential import (
    boundary_obrien_fleming,
    boundary_pocock,
    information_fraction,
    run_group_sequential,
)
from tecore.sequential.simulate import SequentialSimConfig, simulate_ab_stream


def test_information_fraction_bounds():
    assert information_fraction(10, 100) > 0
    assert information_fraction(100, 100) == 1.0
    assert information_fraction(200, 100) == 1.0


def test_boundaries_are_finite():
    alpha = 0.05
    z1 = boundary_obrien_fleming(alpha=alpha, t=0.2, two_sided=True)
    z2 = boundary_obrien_fleming(alpha=alpha, t=0.8, two_sided=True)
    assert np.isfinite(z1) and np.isfinite(z2)
    assert z1 > z2

    zp1 = boundary_pocock(alpha, 5, True)
    zp2 = boundary_pocock(alpha, 5, True)
    assert np.isfinite(zp1) and np.isfinite(zp2)
    assert abs(zp1 - zp2) < 1e-12


def test_group_sequential_runs_and_has_look_table():
    N = 4000
    looks = [400, 800, 1200, 1600, 2000, 3000, 4000]
    alpha = 0.05

    df = simulate_ab_stream(
        SequentialSimConfig(n=N, effect=0.0, noise_sd=1.0, heavy_tail=False, drift=False, seed=123, ratio=False)
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
        mode=SequentialMode.GROUP_SEQUENTIAL,
        alpha=alpha,
        two_sided=True,
        spending=SpendingFunction.OBRIEN_FLEMING,
        effect_direction=EffectDirection.TWO_SIDED,
        min_n_per_group=10,
        var_floor=1e-12,
        seed=0,
    )

    look_table, warnings = build_look_table_mean(df, spec, schedule, cfg)
    assert isinstance(look_table, pd.DataFrame)
    assert len(look_table) == len(looks)
    assert "z" in look_table.columns
    assert isinstance(warnings, list)
    
    res = run_group_sequential(look_table, cfg)
    assert res.look_table is not None
    assert "boundary_z" in res.look_table.columns
    assert res.decision in {"reject", "continue", "accept"}
