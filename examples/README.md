# Examples

This folder contains small example datasets so you can run `tecore validate` and `tecore cuped` without using proprietary data.

- `example_user_level.csv` - minimal user-level schema for CUPED on a mean metric.
- `example_ratio.csv` - minimal ratio schema (revenue per session) for base vs CUPED via linearization.
- `example_timeseries.csv` - minimal daily time series schema for causal impact / synthetic-control style evaluation.

## Quickstart (run on your data)

Install:

```bash
pip install "git+https://github.com/Niuhych/trustworthy-experiments-core.git"
```

## Validate input schema (user-level):
```bash
tecore validate --input data.csv --schema b2c_user_level
```
## Run base vs CUPED (mean metric)
```bash
tecore cuped --input data.csv --y revenue --x revenue_pre --out-md report.md --out-json result.json
```

## Run base vs CUPED (ratio via linearization)
```bash
tecore cuped-ratio \
  --input data.csv \
  --num revenue --den sessions \
  --num-pre revenue_pre --den-pre sessions_pre \
  --out-md report_ratio.md --out-json result_ratio.json
```
## Run causal impact on a time series (no randomization)
```bash
PYTHONPATH=src python scripts/run_impact.py \
  --input examples/example_timeseries.csv \
  --out out/causal_impact_demo \
  --y y \
  --x sessions,active_users,marketing_spend,external_index \
  --intervention 2025-05-01 \
  --freq D \
  --run-placebo
```

## Input format (b2c_user_level)
Required columns:

- `group`: "control" or "test"
- `revenue`: post-period metric per user
- `revenue_pre`: pre-period covariate per user

## Input format (b2c_ratio)
Required columns:

- `group`: `"control"` or `"test"`
- `revenue`: post-period numerator (example)
- `sessions`: post-period denominator (example)
- `revenue_pre`: pre-period numerator
- `sessions_pre`: pre-period denominator

## Input format (timeseries_causal_impact)
Required columns:
- `date`: YYYY-MM-DD (daily)
- `y`: target metric (e.g., revenue/orders/margin)

Recommended covariates (optional, passed via `--x`):
- `sessions`
- `active_users`
- `marketing_spend`
- `external_index`

Intervention date is provided via CLI: `--intervention YYYY-MM-DD`.

## Roadmap
This repository is designed to grow beyond CUPED:
- tecore sequential ... (planned),
- causal impact / synthetic control (time series) ... (available via `scripts/run_impact.py`, CLI command planned).
  
The tecore.io and tecore.report modules are intended to be reused across these components.
