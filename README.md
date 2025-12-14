# Trustworthy Experiments Core

A small, opinionated toolkit for designing and analyzing online experiments in data-driven B2C businesses.

The goal of this project is to provide **practical building blocks** for trustworthy experimentation:
- power and sample size analysis for typical B2C metrics,
- sensible defaults for metric design (ARPU, CR, LTV, retention, churn),
- helpers for segment-aware experiment planning (e.g. VIP vs non-VIP customers).

This repository is part of a broader effort to develop a framework for **trustworthy experimentation** in B2C products and services: e-commerce, subscription platforms, gaming, financial products for consumers, and other digital businesses.

## What's inside

- [`src/tecore/power.py`](src/tecore/power.py) – functions for power analysis and sample size calculation for common experiment setups (binary metrics like conversion rate, etc.).
- [`src/tecore/metrics.py`](src/tecore/metrics.py) – helpers for metric design, ratio metrics, and linearization.
- [`src/tecore/design.py`](src/tecore/design.py) – utilities for high-level experiment design (MDE, sample size, duration).

- [`notebooks/01_power_analysis_arpu_cr.ipynb`](notebooks/01_power_analysis_arpu_cr.ipynb) – step-by-step example of power analysis for ARPU and conversion rate experiments.
- [`notebooks/02_metric_design_b2c.ipynb`](notebooks/02_metric_design_b2c.ipynb) – examples of metric design for B2C experiments, including ARPU and ratio metrics.
- [`notebooks/03_experiment_design_tradeoffs.ipynb`](notebooks/03_experiment_design_tradeoffs.ipynb) – experiment design trade-offs (MDE, sample size, duration).

## Who is this for?

- Product analysts and data analysts working in B2C companies,
- Growth and marketing teams that run experiments but don’t have a dedicated research group,
- Anyone who wants to move from “quick A/B tests” to more reliable, statistically sound decisions.

The focus is on **clarity and practicality**, not on covering every possible edge case. The code is meant to be read, modified, and adapted to real-world pipelines.

## Status

This is an early-stage project. The initial goal is to provide a clear, working baseline for:
- power analysis in B2C experiments,
- basic metric design helpers.

Future plans include:
- more advanced designs (e.g. uneven allocation, stratification),
- additional examples and notebooks.

Contributions, issues and suggestions are very welcome.
