# Example time series dataset (daily)

This folder contains a synthetic daily dataset you can use to run the causal module "out of the box".

## Files
- `example_daily.csv` — daily time series with a known intervention effect.
- You can also generate your own synthetic datasets via:
  ```bash
  PYTHONPATH=src python scripts/generate_synthetic_ts.py --out-dir data/synthetic_ts
  ```

### Expected columns (minimal)

date (YYYY-MM-DD)
y (target metric)

## Quick run
```
PYTHONPATH=src python scripts/run_impact.py \
  --input data/example_ts/example_daily.csv \
  --out results/ci_demo \
  --y y \
  --x sessions,active_users,marketing_spend,external_index \
  --intervention 2025-05-01 \
  --freq D \
  --run-placebo
```

### Outputs:
results.json
effect_series.csv
placebo_results.csv (if enabled)
plots/*.png
