from __future__ import annotations

from typing import Any

import pandas as pd

from tecore.cli.bundle import (
    prepare_out_dir,
    write_report_md,
    write_results_json,
    write_run_meta,
    write_table,
)


# Minimal schema contracts (MVP)
_SCHEMAS: dict[str, list[str]] = {
    "b2c_user_level": ["group", "revenue", "revenue_pre"],
    "b2c_ratio": ["group", "revenue", "sessions", "revenue_pre", "sessions_pre"],
    # Placeholder for future causal; keep here for CLI UX consistency
    "timeseries_causal_impact": ["date", "y"],
}


def _expected_cols(schema: str) -> list[str]:
    if schema in _SCHEMAS:
        return _SCHEMAS[schema]
    # Unknown schema: no hard requirements (generic validation only)
    return []


def cmd_validate(args) -> int:
    schema = str(getattr(args, "schema", "b2c_user_level"))
    df = pd.read_csv(args.input)

    expected = _expected_cols(schema)
    missing = [c for c in expected if c not in df.columns]

    warnings: list[str] = []
    if schema not in _SCHEMAS:
        warnings.append(f"Unknown schema `{schema}`. Only generic CSV readability checks were applied.")

    ok = len(missing) == 0

    # Bundle path (optional)
    out_dir = prepare_out_dir(getattr(args, "out", None), command="validate")

    # If bundle requested -> write unified artifacts
    if out_dir is not None:
        write_run_meta(out_dir, vars(args), extra={"command": "validate"})

        expected_df = pd.DataFrame({"expected_column": expected}) if expected else pd.DataFrame({"expected_column": []})
        missing_df = pd.DataFrame({"missing_column": missing}) if missing else pd.DataFrame({"missing_column": []})

        artifacts: dict[str, Any] = {"report_md": "report.md", "plots": [], "tables": []}
        artifacts["tables"].append(write_table(out_dir, "schema_expected", expected_df))
        artifacts["tables"].append(write_table(out_dir, "schema_missing", missing_df))

        report_lines = [
            "# tecore validate report",
            "",
            "## Inputs",
            f"- input: `{args.input}`",
            f"- schema: `{schema}`",
            "",
            "## Result",
            f"- ok: `{ok}`",
            f"- n_rows: `{int(df.shape[0])}`",
            f"- n_cols: `{int(df.shape[1])}`",
            "",
            "## Missing required columns",
        ]
        if missing:
            report_lines.extend([f"- `{c}`" for c in missing])
        else:
            report_lines.append("(none)")

        report_lines.extend(["", "## Warnings"])
        if warnings:
            report_lines.extend([f"- {w}" for w in warnings])
        else:
            report_lines.append("(none)")

        report = "\n".join(report_lines) + "\n"

        payload: dict[str, Any] = {
            "command": "validate",
            "inputs": {"input": args.input, "schema": schema},
            "estimates": {"ok": ok},
            "diagnostics": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "missing_required": missing},
            "warnings": warnings,
            "artifacts": artifacts,
        }

        write_results_json(out_dir, payload)
        write_report_md(out_dir, report)

        # Exit code semantics: keep it simple for MVP
        # 0 if ok else 2 (invalid input/schema)
        return 0 if ok else 2

    # Legacy behavior (no --out): print OK / raise for CI visibility
    if ok:
        print("OK")
        return 0
    raise ValueError(f"Validation failed: missing required columns for schema `{schema}`: {missing}")
