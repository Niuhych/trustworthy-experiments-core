from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover
    pkg_version = None  # type: ignore


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_json(obj: Any) -> Any:
    """
    Make an object JSON-serializable (best-effort).
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _safe_json(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _safe_json(vars(obj))
    return str(obj)


def ensure_subdirs(out_dir: Path, names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (out_dir / n).mkdir(parents=True, exist_ok=True)


def prepare_out_dir(out: str | None, command: str) -> Path:
    from datetime import datetime
    from pathlib import Path

    if out is None or str(out).strip() == "":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / command / ts
    else:
        out_dir = Path(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_subdirs(out_dir, ["tables", "plots"])
    return out_dir


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_run_meta(out_dir: Path, args: Any, extra: dict[str, Any] | None = None) -> None:
    # Normalize args into a dict for clean serialization
    if isinstance(args, dict):
        args_dict = dict(args)
    else:
        # argparse.Namespace / SimpleNamespace / arbitrary object
        try:
            args_dict = dict(vars(args))
        except Exception:
            args_dict = {"args": str(args)}

    # Remove argparse dispatch function pointer if present
    args_dict.pop("func", None)

    pkg_ver = None
    if pkg_version is not None:
        try:
            pkg_ver = pkg_version("trustworthy-experiments-core")
        except Exception:
            pkg_ver = None

    meta: dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "tecore_version": pkg_ver,
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "cwd": os.getcwd(),
        "args": _safe_json(args_dict),
    }
    if extra:
        meta["extra"] = _safe_json(extra)

    write_json(out_dir / "run_meta.json", meta)


def write_results_json(out_dir: Path, payload: dict[str, Any]) -> None:
    write_json(out_dir / "results.json", payload)


def write_report_md(out_dir: Path, text: str) -> None:
    write_text(out_dir / "report.md", text)


def write_audit_json(out_dir: Path, payload: dict[str, Any]) -> None:
    write_json(out_dir / "audit.json", payload)


def write_audit_md(out_dir: Path, text: str) -> None:
    write_text(out_dir / "audit.md", text)


def write_table(out_dir: Path, name: str, df) -> str:
    """
    Writes tables/<name>.csv and returns relative path for artifacts registry.
    """
    rel = Path("tables") / f"{name}.csv"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(rel).replace("\\", "/")


def save_plot(out_dir: Path, name: str, fig=None, dpi: int = 140) -> str:
    """
    Saves plots/<name>.png and returns relative path for artifacts registry.
    If fig is None, saves current matplotlib figure.
    """
    rel = Path("plots") / f"{name}.png"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # local import to keep import-time light

    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(rel).replace("\\", "/")
