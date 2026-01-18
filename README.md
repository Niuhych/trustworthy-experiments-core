# Trustworthy Experiments Core

A small, opinionated toolkit for designing and analyzing online experiments in data-driven B2C businesses.

The goal of this project is to provide **practical building blocks** for trustworthy experimentation:
- power and sample size analysis for typical B2C metrics,
- sensible defaults for metric design (ARPU, CR, LTV, retention, churn),
- helpers for segment-aware experiment planning (e.g. VIP vs non-VIP customers).

This repository is part of a broader effort to develop a framework for **trustworthy experimentation** in B2C products and services: e-commerce, subscription platforms, gaming, financial products for consumers, and other digital businesses.

## What's inside

- [`src/tecore/power.py`](src/tecore/power.py) – functions for power analysis and sample size calculation for common experiment setups (binary metrics like conversion rate, etc.).
- [`src/tecore/metrics.py`](src/tecore/metrics.py) – metric helpers (ratio metrics, linearization, etc.).
- [`src/tecore/design.py`](src/tecore/design.py) – utilities for high-level experiment design (MDE, sample size, duration).
- [`src/tecore/simulate.py`](src/tecore/simulate.py) - synthetic B2C-like data generator with pre/post aggregates and heavy-tailed revenue (useful for CUPED and ratio metrics).
- [`src/tecore/variance_reduction.py`](src/tecore/variance_reduction.py) - variance reduction utilities (currently: CUPED).
- [`src/tecore/sequential/`](src/tecore/sequential/) - sequential analysis module (group sequential boundaries + anytime-valid confidence sequences; mean-difference + ratio via linearization).
- [`src/tecore/causal/`](src/tecore/causal/) - causal inference module for time series interventions (impact estimation without randomization).
- [`schemas/timeseries_causal_impact.yaml`](schemas/timeseries_causal_impact.yaml) - schema for single-unit time series causal impact runs.
- [`scripts/run_impact.py`](scripts/run_impact.py) - one-command runner (no notebooks) for causal impact / synthetic control / DiD.

Part 1 - Fundamentals:
- [`notebooks/01_power_analysis_arpu_cr.ipynb`](notebooks/01_power_analysis_arpu_cr.ipynb) - step-by-step power analysis for ARPU and conversion rate.
- [`notebooks/02_metric_design_b2c.ipynb`](notebooks/02_metric_design_b2c.ipynb) - metric design examples for B2C, including ARPU and ratio metrics.
- [`notebooks/03_experiment_design_tradeoffs.ipynb`](notebooks/03_experiment_design_tradeoffs.ipynb) - experiment design trade-offs (MDE, sample size, duration).

Part 2 - Synthetic data and benchmarks:
- [`notebooks/04_synthetic_b2c_data_generator.ipynb`](notebooks/04_synthetic_b2c_data_generator.ipynb) - generate synthetic user-level data with pre/post periods and export scenario CSVs.
- [`notebooks/05_benchmark_arpu_vs_ratio_linearization.ipynb`](notebooks/05_benchmark_arpu_vs_ratio_linearization.ipynb) - benchmark ARPU vs ratio metrics (bootstrap and linearization) across multiple scenarios; includes empirical power/type I error estimates.
- [`notebooks/06_cuped_variance_reduction.ipynb`](notebooks/06_cuped_variance_reduction.ipynb) - CUPED for ARPU and ratio metrics (via linearization); variance reduction and empirical sensitivity.
- [`notebooks/07_cuped_on_arpu_with_outliers.ipynb`](notebooks/07_cuped_on_arpu_with_outliers.ipynb) - heavy tails and outliers: raw ARPU vs winsorization/log/trim + CUPED; CI width and empirical power comparisons.

Part 3 - Causal inference (time series):
- [`notebooks/08_prepare_time_series_for_causal.ipynb`](notebooks/08_prepare_time_series_for_causal.ipynb) - how to prepare a daily time series dataset for causal impact evaluation (schema, missing dates, features).
- [`notebooks/09_causal_impact_like_time_series.ipynb`](notebooks/09_causal_impact_like_time_series.ipynb) - causal impact (counterfactual prediction) with uncertainty, diagnostics, and placebo-in-time.
- [`notebooks/10_synthetic_control_donor_mode.ipynb`](notebooks/10_synthetic_control_donor_mode.ipynb) - synthetic control style evaluation in donor-series mode.
- [`notebooks/11_diff_in_diff_panel_minimal.ipynb`](notebooks/11_diff_in_diff_panel_minimal.ipynb) - minimal difference-in-differences example for panel-like setups.
- [`notebooks/12_causal_impact_failure_modes.ipynb`](notebooks/12_causal_impact_failure_modes.ipynb) - practical failure modes and diagnostics for time series impact evaluation.
- [`notebooks/13_placebo_and_sensitivity.ipynb`](notebooks/13_placebo_and_sensitivity.ipynb) - placebo tests and sensitivity checks.

Part 4 - Sequential analysis (safe peeking, early stopping, anytime-valid inference):
- [`notebooks/14_sequential_intro_peeking_vs_valid.ipynb`](notebooks/14_sequential_intro_peeking_vs_valid.ipynb) - why naïve peeking inflates type I error; group sequential as a pragmatic fix.
- [`notebooks/15_group_sequential_boundaries_pocock_obf.ipynb`](notebooks/15_group_sequential_boundaries_pocock_obf.ipynb) - Pocock vs O'Brien-Fleming: boundaries intuition and trade-offs.
- [`notebooks/16_confidence_sequences_anytime_valid.ipynb`](notebooks/16_confidence_sequences_anytime_valid.ipynb) - anytime-valid confidence sequences and anytime p-values (normal approximation).
- [`notebooks/17_sequential_ratio_linearized.ipynb`](notebooks/17_sequential_ratio_linearized.ipynb) - sequential monitoring for ratio metrics via linearization; stability caveats.
- [`notebooks/18_sequential_failure_modes_and_guardrails.ipynb`](notebooks/18_sequential_failure_modes_and_guardrails.ipynb) - drift, heavy tails, early looks: when not to trust sequential outputs.

### Data outputs (generated by notebooks)

Notebook 04 generates reusable scenario datasets (CSV) under:
- data/synthetic/

Typical files:
- data/synthetic/scenario_monetization.csv
- data/synthetic/scenario_activity.csv
- data/synthetic/scenario_mixed.csv

Notebooks 05–07 can:
- load those CSVs if present, or
- fall back to generating synthetic data on the fly.

## Who is this for?

- Product analysts and data analysts working in B2C companies,
- Growth and marketing teams that run experiments but don’t have a dedicated research group,
- Anyone who wants to move from “quick A/B tests” to more reliable, statistically sound decisions.

The focus is on **clarity and practicality**, not on covering every possible edge case. The code is meant to be read, modified, and adapted to real-world pipelines.

## Quickstart
The CLI produces a machine-readable JSON and a human-readable Markdown report (--out-json, --out-md).

### Install (pinned version)

```bash
pip install "git+https://github.com/Niuhych/trustworthy-experiments-core.git@v0.1.0"
```

### Install with pipx (recommended for CLI)
```
pipx install "git+https://github.com/Niuhych/trustworthy-experiments-core.git@v0.1.0"
```

### Sanity check (run on included examples)

## Validate example user-level dataset:
```bash
tecore validate --input examples/example_user_level.csv --schema b2c_user_level
```

## Run base vs CUPED on a mean metric:
```bash
tecore cuped \
  --input examples/example_user_level.csv \
  --y revenue --x revenue_pre \
  --out-md out/report_mean.md --out-json out/result_mean.json
```

## Validate example ratio dataset:
```bash
tecore validate --input examples/example_ratio.csv --schema b2c_ratio
```

## Run base vs CUPED on a ratio metric via linearization:
```bash
tecore cuped-ratio \
  --input examples/example_ratio.csv \
  --num revenue --den sessions \
  --num-pre revenue_pre --den-pre sessions_pre \
  --out-md out/report_ratio.md --out-json out/result_ratio.json
```

## Run base vs CUPED on a ratio metric via linearization:
```bash
tecore cuped-ratio \
  --input examples/example_ratio.csv \
  --num revenue --den sessions \
  --num-pre revenue_pre --den-pre sessions_pre \
  --out-md out/report_ratio.md --out-json out/result_ratio.json
```

## Sequential monitoring (safe peeking) on a mean metric:
```
tecore sequential-mean \
  --input examples/example_user_level.csv \
  --group-col group --control control --test test \
  --y revenue \
  --mode group_sequential \
  --spending obrien_fleming \
  --looks 4,8,10 \
  --min-n-per-group 2 \
  --out out/sequential_mean_demo
```

## Sequential monitoring (safe peeking) on a ratio metric via linearization:
```
tecore sequential-ratio \
  --input examples/example_ratio.csv \
  --group-col group --control control --test test \
  --num revenue --den sessions \
  --baseline-mode first_look \
  --mode confidence_sequence \
  --n-looks 5 --max-n 10 \
  --min-n-per-group 1 \
  --out out/sequential_ratio_demo
```

## Simulate streaming A/B data and run sequential analysis (demo pipeline):
```
tecore sequential-simulate \
  --out out/sequential_simulate_demo \
  --n 20000 --effect 0.1 --noise-sd 1.0 \
  --mode confidence_sequence \
  --n-looks 10 --max-n 20000
```

## Causal impact on a time series (no randomization)
```
PYTHONPATH=src python scripts/run_impact.py \
  --input data/example_ts/example_daily.csv \
  --out out/causal_impact_demo \
  --y y \
  --x sessions,active_users,marketing_spend,external_index \
  --intervention 2025-05-01 \
  --freq D \
  --run-placebo
```

### Run on your data

User-level mean metric:
```bash
tecore validate --input data.csv --schema b2c_user_level
tecore cuped --input data.csv --y revenue --x revenue_pre --out-md report.md --out-json result.json
```

## Ratio metric (linearized):
```bash
tecore validate --input data.csv --schema b2c_ratio
tecore cuped-ratio --input data.csv --num revenue --den sessions --num-pre revenue_pre --den-pre sessions_pre --out-md report_ratio.md --out-json result_ratio.json
```

## What to send back (pilot feedback)

To evaluate the framework without sharing proprietary data, please send back:

1) The exact command line you used (copy/paste).
2) The generated JSON output (`--out-json ...`).
3) The generated Markdown report (`--out-md ...`).

Optional (helps interpretation, still privacy-friendly):
- what the numerator/denominator represent in your domain (e.g., what `revenue` / `sessions` mean),
- unit of analysis (user / account / session) and experiment duration (days),
- confirmation that the pre-period covariate is measured strictly before exposure (yes/no),
- notable data properties: heavy tails, many zeros, strong outliers, bot traffic, etc.

If anything fails, please also include:
- the full error message / stack trace,
- output of `python --version`,
- your OS (e.g., Windows/macOS/Linux).

## Status

This is an early-stage project. The initial goal is to provide a clear, working baseline for:
- power analysis in B2C experiments,
- basic metric design helpers,
- reproducible examples and benchmarks (synthetic data + notebooks).

Future plans include: small refinements to CLI ergonomics and audit/guardrails as needed.

Contributions, issues and suggestions are very welcome.
