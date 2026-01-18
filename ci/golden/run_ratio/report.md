# tecore cuped-ratio report

## Inputs
- input: `examples/example_ratio.csv`
- group_col: `group`
- control: `control`
- test: `test`
- num/den (post): `revenue` / `sessions`
- num/den (pre): `revenue_pre` / `sessions_pre`
- alpha: `0.05`

## Results (linearization, base vs CUPED)
- control_ratio_post (sum(num)/sum(den), control): 0.371429
- mean_den_control (per-unit mean den, control): 11.6667

| method | diff_linearized | p_value | ratio_diff_approx | rel_lift_approx |
|---|---:|---:|---:|---:|
| base | 0.504762 | 0.8318 | 0.0432653 | 0.116484 |
| cuped | 0.064817 | 0.8115 | 0.00555574 | 0.0149578 |

## Diagnostics
- r0_post (control): 0.371429
- r0_pre (control): 0.322581
- theta: 1.27859
- corr(lin_pre, lin_post): 0.993626
- zero_den_share: 0

## Warnings
(none)
