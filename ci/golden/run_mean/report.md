# tecore cuped report

## Inputs
- input: `examples/example_user_level.csv`
- group_col: `group`
- control: `control`
- test: `test`
- y (post): `revenue`
- x (pre): `revenue_pre`
- alpha: `0.05`
- transform(y): `raw`
- winsor_q: `0.99`

## Results (base vs CUPED)
| method | n_control | n_test | mean_control | mean_test | diff (test-control) | p_value | ci_low | ci_high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 5 | 5 | 0.96 | 1.22 | 0.26 | 0.7906 | -1.9432 | 2.4632 |
| cuped | 5 | 5 | 0.906747 | 1.27325 | 0.366506 | 0.07785 | -0.0520243 | 0.785036 |

## Diagnostics
- theta: 1.33132
- corr(x,y): 0.972288
- var_ratio (adj/raw): 0.05465669610312613

## Warnings
(none)
