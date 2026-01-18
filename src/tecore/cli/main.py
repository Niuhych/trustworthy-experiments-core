from __future__ import annotations

import argparse

from tecore.cli.commands.audit import cmd_audit
from tecore.cli.commands.cuped import cmd_cuped
from tecore.cli.commands.cuped_ratio import cmd_cuped_ratio
from tecore.cli.commands.sequential_mean import cmd_sequential_mean
from tecore.cli.commands.sequential_ratio import cmd_sequential_ratio
from tecore.cli.commands.sequential_simulate import cmd_sequential_simulate
from tecore.cli.commands.validate import cmd_validate
from tecore.cli.commands.version import cmd_version
from tecore.cli.commands.run_config import cmd_run_config
from tecore.cli.commands.causal_impact import cmd_causal_impact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tecore", description="Trustworthy Experiments Core CLI.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("version", help="Print installed package version.")
    sp.set_defaults(func=cmd_version)

    sp = sub.add_parser("validate", help="Validate an input dataset schema.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--schema", default="b2c_user_level", help="b2c_user_level | b2c_ratio")
    sp.add_argument("--out", default=None, help="Output directory for report bundle (e.g. results/run_001).")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("audit", help="Audit/guardrails for an input dataset (writes bundle).")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--schema", default="b2c_user_level", help="b2c_user_level | b2c_ratio | timeseries_causal_impact")
    sp.add_argument("--out", required=True, help="Output directory for audit bundle (e.g. out/audit_001).")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("cuped", help="Base vs CUPED for a mean-difference metric.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--group-col", default="group")
    sp.add_argument("--control", default="control")
    sp.add_argument("--test", default="test")
    sp.add_argument("--y", required=True, help="Post-period metric column (e.g., revenue).")
    sp.add_argument("--x", required=True, help="Pre-period covariate column (e.g., revenue_pre).")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--transform", choices=["raw", "winsor", "log1p"], default="raw")
    sp.add_argument("--winsor-q", type=float, default=0.99)

    sp.add_argument("--out-json", dest="out_json", default=None, help="(Deprecated) Write results JSON to a file.")
    sp.add_argument("--out-md", dest="out_md", default=None, help="(Deprecated) Write markdown report to a file.")

    sp.add_argument("--out", default=None, help="Output directory for report bundle (e.g. results/run_001).")
    sp.add_argument(
        "--audit",
        action="store_true",
        help="Run audit/guardrails and include in bundle (MVP: ignored for now).",
    )
    sp.set_defaults(func=cmd_cuped)

    sp = sub.add_parser("cuped-ratio", help="Base vs CUPED for a ratio metric via linearization.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--group-col", default="group")
    sp.add_argument("--control", default="control")
    sp.add_argument("--test", default="test")
    sp.add_argument("--num", required=True, help="Post-period numerator column (e.g., revenue).")
    sp.add_argument("--den", required=True, help="Post-period denominator column (e.g., sessions).")
    sp.add_argument("--num-pre", required=True, help="Pre-period numerator column (e.g., revenue_pre).")
    sp.add_argument("--den-pre", required=True, help="Pre-period denominator column (e.g., sessions_pre).")
    sp.add_argument("--alpha", type=float, default=0.05)

    sp.add_argument("--out-json", dest="out_json", default=None, help="(Deprecated) Write results JSON to a file.")
    sp.add_argument("--out-md", dest="out_md", default=None, help="(Deprecated) Write markdown report to a file.")

    sp.add_argument("--out", default=None, help="Output directory for report bundle (e.g. results/run_001).")
    sp.add_argument(
        "--audit",
        action="store_true",
        help="Run audit/guardrails and include in bundle (MVP: ignored for now).",
    )
    sp.set_defaults(func=cmd_cuped_ratio)

    sp = sub.add_parser("sequential-mean", help="Sequential monitoring for a mean-difference metric.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--group-col", default="group")
    sp.add_argument("--control", default="control")
    sp.add_argument("--test", default="test")
    sp.add_argument("--y", required=True, help="User-level metric column.")
    sp.add_argument("--timestamp-col", dest="timestamp_col", default=None, help="Optional ordering column.")
    sp.add_argument("--unit-col", dest="unit_col", default=None, help="Optional unit identifier column.")

    sp.add_argument("--mode", choices=["group_sequential", "confidence_sequence"], default="group_sequential")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--two-sided", dest="two_sided", action="store_true", default=True)
    sp.add_argument("--spending", choices=["obrien_fleming", "pocock"], default="obrien_fleming")
    sp.add_argument("--effect-direction", dest="effect_direction", choices=["two_sided", "increase", "decrease"], default="two_sided")
    sp.add_argument("--min-n-per-group", dest="min_n_per_group", type=int, default=50)
    sp.add_argument("--var-floor", dest="var_floor", type=float, default=1e-12)
    sp.add_argument("--cs-tau", dest="cs_tau", type=float, default=1.0, help="Confidence sequence tuning parameter.")
    sp.add_argument("--seed", type=int, default=42)

    sp.add_argument("--looks", default=None, help="Comma-separated look sizes, e.g. 2000,4000,6000")
    sp.add_argument("--n-looks", dest="n_looks", type=int, default=5, help="Number of evenly spaced looks.")
    sp.add_argument("--max-n", dest="max_n", type=int, default=10000, help="Max N for evenly spaced looks.")

    sp.add_argument("--out", default=None, help="Output directory for report bundle.")
    sp.add_argument("--audit", action="store_true", help="Run audit/guardrails and include in bundle.")
    sp.set_defaults(func=cmd_sequential_mean)

    sp = sub.add_parser("sequential-ratio", help="Sequential monitoring for a ratio metric via linearization.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--group-col", default="group")
    sp.add_argument("--control", default="control")
    sp.add_argument("--test", default="test")
    sp.add_argument("--num", required=True, help="Numerator column.")
    sp.add_argument("--den", required=True, help="Denominator column.")
    sp.add_argument("--timestamp-col", dest="timestamp_col", default=None)
    sp.add_argument("--unit-col", dest="unit_col", default=None)
    sp.add_argument("--baseline-mode", dest="baseline_mode", choices=["first_look", "pre"], default="first_look")

    sp.add_argument("--mode", choices=["group_sequential", "confidence_sequence"], default="group_sequential")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--two-sided", dest="two_sided", action="store_true", default=True)
    sp.add_argument("--spending", choices=["obrien_fleming", "pocock"], default="obrien_fleming")
    sp.add_argument("--effect-direction", dest="effect_direction", choices=["two_sided", "increase", "decrease"], default="two_sided")
    sp.add_argument("--min-n-per-group", dest="min_n_per_group", type=int, default=50)
    sp.add_argument("--var-floor", dest="var_floor", type=float, default=1e-12)
    sp.add_argument("--cs-tau", dest="cs_tau", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=42)

    sp.add_argument("--looks", default=None)
    sp.add_argument("--n-looks", dest="n_looks", type=int, default=5)
    sp.add_argument("--max-n", dest="max_n", type=int, default=10000)

    sp.add_argument("--out", default=None, help="Output directory for report bundle.")
    sp.add_argument("--audit", action="store_true", help="Run audit/guardrails and include in bundle.")
    sp.set_defaults(func=cmd_sequential_ratio)

    sp = sub.add_parser("sequential-simulate", help="Simulate streaming A/B data and run sequential analysis.")
    sp.add_argument("--out", default=None, help="Output directory for report bundle.")
    sp.add_argument("--n", type=int, default=20000)
    sp.add_argument("--effect", type=float, default=0.0)
    sp.add_argument("--noise-sd", dest="noise_sd", type=float, default=1.0)
    sp.add_argument("--heavy-tail", dest="heavy_tail", action="store_true")
    sp.add_argument("--drift", action="store_true")
    sp.add_argument("--ratio", action="store_true", help="Simulate ratio metric (num/den) instead of mean.")

    sp.add_argument("--mode", choices=["group_sequential", "confidence_sequence"], default="group_sequential")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--two-sided", dest="two_sided", action="store_true", default=True)
    sp.add_argument("--spending", choices=["obrien_fleming", "pocock"], default="obrien_fleming")
    sp.add_argument("--effect-direction", dest="effect_direction", choices=["two_sided", "increase", "decrease"], default="two_sided")
    sp.add_argument("--min-n-per-group", dest="min_n_per_group", type=int, default=50)
    sp.add_argument("--var-floor", dest="var_floor", type=float, default=1e-12)
    sp.add_argument("--cs-tau", dest="cs_tau", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=42)

    sp.add_argument("--looks", default=None)
    sp.add_argument("--n-looks", dest="n_looks", type=int, default=5)
    sp.add_argument("--max-n", dest="max_n", type=int, default=10000)
    sp.set_defaults(func=cmd_sequential_simulate)


    sp = sub.add_parser("run", help="Run a command from a YAML config.")
    sp.add_argument("--config", required=True, help="Path to YAML config file.")
    sp.set_defaults(func=cmd_run_config)

    sp = sub.add_parser("causal-impact", help="Causal impact analysis for time series (bundle output).")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--schema", default="timeseries_causal_impact", help="timeseries_causal_impact")
    sp.add_argument("--date-col", dest="date_col", default="date")
    sp.add_argument("--y", required=True, help="Outcome column.")
    sp.add_argument("--x", default="", help="Comma-separated covariates (e.g. sessions,active_users).")
    sp.add_argument("--intervention", required=True, help="Intervention date YYYY-MM-DD.")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--bootstrap-iters", dest="bootstrap_iters", type=int, default=200)
    sp.add_argument("--n-placebos", dest="n_placebos", type=int, default=0)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument(
        "--pre-period-min-points",
        dest="pre_period_min_points",
        type=int,
        default=30,
        help="Minimum number of pre-period points required (quality gate).",
    )
    sp.add_argument("--out", default=None, help="Output directory for report bundle.")
    sp.add_argument("--audit", action="store_true", help="Run audit/guardrails and include in bundle.")
    sp.set_defaults(func=cmd_causal_impact)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
