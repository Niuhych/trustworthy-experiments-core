# tecore audit report

## Inputs
- schema: `b2c_ratio`
- parent_command: `cuped-ratio`

## Dataset
- shape: 6 rows × 5 cols
- detected group_col: `group`
- detected user_id_col: `None`

## Warnings
- No user identifier column found (`user_id`/`uid`/etc). Duplicate-user checks skipped.

## Errors
(none)

## Tables
- tables/audit_column_profile.csv
- tables/audit_numeric_summary.csv (if numeric columns exist)
- tables/audit_group_balance.csv (if group column detected)
