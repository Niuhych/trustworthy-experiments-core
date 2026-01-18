# tecore sequential report

## Inputs
- group_col: `group` (control=`control`, test=`test`)
- metric: `num/den`
- mode: `group_sequential`
- alpha: `0.05`

## Decision
- stopped early: **Yes (look_n=8000)**
- decision: **reject**
- final p-value (mode-specific): `0.00038332`
- fixed (non-sequential) CI: `[0.0356132, 0.123338]`
- confidence sequence (time-uniform) CI: `(n/a)`

## Notes
- This run uses group sequential monitoring (K looks) with z-boundaries. The decision rule is based on boundary crossing; the displayed p-value is a naive z-test p-value for interpretability.

## Warnings
(none)

## Artifacts
- tables/look_table.csv
- plots/z_trajectory.png
- plots/effect_trajectory.png
