# tecore audit report

## Inputs
- schema: `timeseries_causal_impact`
- parent_command: `causal-impact`

## Dataset
- shape: 80 rows × 4 cols
- detected group_col: `None`
- detected user_id_col: `None`

## Warnings
- No obvious group column found (expected `group`). Group balance checks skipped.
- No user identifier column found (`user_id`/`uid`/etc). Duplicate-user checks skipped.

## Errors
(none)

## Tables
- tables/audit_column_profile.csv
- tables/audit_numeric_summary.csv (if numeric columns exist)
- tables/audit_group_balance.csv (if group column detected)
