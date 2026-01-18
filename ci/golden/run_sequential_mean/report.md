# tecore sequential report

## Inputs
- group_col: `group` (control=`control`, test=`test`)
- metric: `y`
- mode: `group_sequential`
- alpha: `0.05`

## Decision
- stopped early: **Yes (look_n=10000)**
- decision: **reject**
- final p-value (mode-specific): `0.00114024`
- fixed (non-sequential) CI: `[0.0258393, 0.104148]`
- confidence sequence (time-uniform) CI: `(n/a)`

## Notes
- This run uses group sequential monitoring (K looks) with z-boundaries. The decision rule is based on boundary crossing; the displayed p-value is a naive z-test p-value for interpretability.

## Warnings
(none)

## Artifacts
- tables/look_table.csv
- plots/z_trajectory.png
- plots/effect_trajectory.png
