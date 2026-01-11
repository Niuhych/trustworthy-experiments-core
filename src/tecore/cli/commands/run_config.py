from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tecore.cli.commands.audit import cmd_audit
from tecore.cli.commands.cuped import cmd_cuped
from tecore.cli.commands.cuped_ratio import cmd_cuped_ratio
from tecore.cli.commands.validate import cmd_validate


def _fail(msg: str) -> int:
    print(f"[tecore][error] {msg}", file=sys.stderr)
    return 2


def _as_args(d: dict[str, Any]) -> SimpleNamespace:
    # cmd_* functions expect attribute access (args.foo)
    return SimpleNamespace(**d)


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (YAML dict).")
    return data


def cmd_run_config(args) -> int:
    cfg_path = str(args.config)
    cfg = _load_yaml(cfg_path)

    command = str(cfg.get("command", "")).strip()
    if not command:
        return _fail("Missing required field: command")

    input_path = cfg.get("input", None)
    out_dir = cfg.get("out", None)

    params = cfg.get("params", {}) or {}
    if not isinstance(params, dict):
        return _fail("Field `params` must be a mapping (YAML dict).")

    audit_flag = bool(cfg.get("audit", False))

    # Build args-like dict for underlying command
    base_args: dict[str, Any] = {"out": out_dir}

    if command == "cuped":
        if input_path is None:
            return _fail("cuped requires `input`")
        merged = {
            **base_args,
            "input": input_path,
            "group_col": params.get("group_col", "group"),
            "control": params.get("control", "control"),
            "test": params.get("test", "test"),
            "y": params.get("y"),
            "x": params.get("x"),
            "alpha": float(params.get("alpha", 0.05)),
            "transform": params.get("transform", "raw"),
            "winsor_q": float(params.get("winsor_q", 0.99)),
            "audit": audit_flag,
            # legacy (explicitly None so handlers don't break)
            "out_json": None,
            "out_md": None,
        }
        if not merged["y"] or not merged["x"]:
            return _fail("cuped requires params.y and params.x")
        return int(cmd_cuped(_as_args(merged)))

    if command in {"cuped-ratio", "cuped_ratio"}:
        if input_path is None:
            return _fail("cuped-ratio requires `input`")
        merged = {
            **base_args,
            "input": input_path,
            "group_col": params.get("group_col", "group"),
            "control": params.get("control", "control"),
            "test": params.get("test", "test"),
            "num": params.get("num"),
            "den": params.get("den"),
            "num_pre": params.get("num_pre"),
            "den_pre": params.get("den_pre"),
            "alpha": float(params.get("alpha", 0.05)),
            "audit": audit_flag,
            "out_json": None,
            "out_md": None,
        }
        req = ["num", "den", "num_pre", "den_pre"]
        miss = [k for k in req if not merged.get(k)]
        if miss:
            return _fail(f"cuped-ratio requires params: {miss}")
        return int(cmd_cuped_ratio(_as_args(merged)))

    if command == "validate":
        if input_path is None:
            return _fail("validate requires `input`")
        merged = {
            **base_args,
            "input": input_path,
            "schema": params.get("schema", cfg.get("schema", "b2c_user_level")),
        }
        return int(cmd_validate(_as_args(merged)))

    if command == "audit":
        if input_path is None:
            return _fail("audit requires `input`")
        merged = {
            **base_args,
            "input": input_path,
            "schema": params.get("schema", cfg.get("schema", "b2c_user_level")),
        }
        if merged.get("out") is None:
            return _fail("audit requires `out` (bundle dir)")
        return int(cmd_audit(_as_args(merged)))

    return _fail(f"Unknown command: {command}")
