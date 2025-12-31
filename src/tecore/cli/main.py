import argparse

def main() -> None:
    parser = argparse.ArgumentParser(prog="tecore")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print package version")

    args = parser.parse_args()
    if args.cmd == "version":
        print("trustworthy-experiments-core 0.1.0")
