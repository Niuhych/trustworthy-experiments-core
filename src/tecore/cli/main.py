from __future__ import annotations

import argparse

from tecore.cli.commands.version import cmd_version
from tecore.cli.commands.validate import cmd_validate
from tecore.cli.commands.cuped import cmd_cuped
from tecore.cli.commands.cuped_ratio import cmd_cuped_ratio


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tecore", description="Trustworthy Experiments Core CLI.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("version", help="Print installed package version.")
    sp.set_defaults(func=cmd_version)

    sp = sub.add_parser("validate", help="Validate an input dataset schema.")
    sp.add_argument("--input", required=True, help="Path to CSV file.")
    sp.add_argument("--schema", default="b2c_user_level", help="b2c_user_level | b2c_ratio")
    sp.set_defaults(func=cmd_validate)

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
    sp.add_argument("--out-json", default=None)
    sp.add_argument("--out-md", default=None)
    sp.add_argument("--out", default=None, help="Output directory for report bundle (e.g. results/run_001).")
    sp.add_argument("--audit", action="store_true", help="Run audit/guardrails and include in bundle (MVP: ignored for now).")
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
    sp.add_argument("--out-json", default=None)
    sp.add_argument("--out-md", default=None)
    sp.add_argument("--out", default=None, help="Output directory for report bundle (e.g. results/run_001).")
    sp.add_argument("--audit", action="store_true", help="Run audit/guardrails and include in bundle (MVP: ignored for now).")
    sp.set_defaults(func=cmd_cuped_ratio)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
