from __future__ import annotations

import argparse

from tecore.cli.commands.version import cmd_version


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tecore", description="Trustworthy Experiments Core CLI.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("version", help="Print installed package version.")
    sp.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
