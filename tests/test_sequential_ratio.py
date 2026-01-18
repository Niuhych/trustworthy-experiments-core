import numpy as np
import pandas as pd

from tecore.sequential.schema import LookSchedule, SequentialSpec
from tecore.sequential.ratio import linearize_ratio, linearize_ratio_frame
from tecore.sequential.simulate import SequentialSimConfig, simulate_ab_stream


def test_linearize_ratio_public_wrapper_returns_expected_columns():
    N = 20000
    looks = [2000, 4000, 6000]
    df = simulate_ab_stream(
        SequentialSimConfig(n=N, effect=0.03, noise_sd=1.0, heavy_tail=False, drift=False, seed=41, ratio=True)
    )

    df_lin, r0 = linearize_ratio(
        df,
        num_col="num",
        den_col="den",
        group_col="group",
        control_label="control",
        baseline_mode="first_look",
        first_look_n=looks[0],
        y_lin_col="y_lin",
    )

    assert np.isfinite(r0)
    assert "y_lin" in df_lin.columns
    assert len(df_lin) == len(df)


def test_linearize_ratio_frame_returns_warnings_list():
    N = 5000
    looks = [1000, 5000]
    df = simulate_ab_stream(
        SequentialSimConfig(n=N, effect=0.0, noise_sd=1.0, heavy_tail=False, drift=False, seed=7, ratio=True)
    )

    spec = SequentialSpec(
        group_col="group",
        control_label="control",
        test_label="test",
        num_col="num",
        den_col="den",
        timestamp_col="timestamp",
    )
    schedule = LookSchedule(looks=looks)

    df_lin, r0, warnings = linearize_ratio_frame(
        df=df, spec=spec, schedule=schedule, baseline_mode="first_look", output_col="_y_lin"
    )
    assert np.isfinite(r0)
    assert isinstance(warnings, list)
    assert "_y_lin" in df_lin.columns
