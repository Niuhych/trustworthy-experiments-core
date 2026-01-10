from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """
    Convenience wrapper:
      python scripts/run_synthetic_control.py --input ... --out ... --intervention ... --y ... --x ...
    """
    argv = sys.argv[1:]
    cmd = [sys.executable, "scripts/run_impact.py", "--method", "synthetic_control"] + argv
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
