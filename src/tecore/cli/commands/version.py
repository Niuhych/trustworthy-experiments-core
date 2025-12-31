from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError


def cmd_version(args) -> int:
    try:
        v = version("trustworthy-experiments-core")
    except PackageNotFoundError:
        v = "unknown"
    print(v)
    return 0
