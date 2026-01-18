# tecore causal-impact report

## Inputs
- input: `examples/example_ts_long.csv`
- date_col: `date`
- y: `y`
- x: `sessions, active_users`
- intervention: `2025-02-15`
- alpha: `0.05`
- bootstrap_iters: `200`
- n_placebos: `0`
- seed: `42`

## Outputs
- tables/effect_series.csv
- plots/observed_vs_counterfactual.png
- plots/point_effect.png
- plots/cumulative_effect.png
