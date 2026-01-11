from __future__ import annotations

from typing import Any

import pandas as pd

from tecore.cli.audit_api import write_audit_bundle
from tecore.cli.bundle import prepare_out_dir, write_report_md, write_results_json, write_run_meta


def cmd_audit(args) -> int:
    df = pd.read_csv(args.input)
    schema = str(getattr(args, "schema", "b2c_user_level"))

    out_dir = prepare_out_dir(getattr(args, "out", None), command="audit")
    if out_dir is None:
        raise ValueError("`tecore audit` requires --out (bundle directory).")

    # ---- Write bundle artifacts ----
    write_run_meta(out_dir, vars(args), extra={"command": "audit"})

    audit_payload = write_audit_bundle(out_dir, df=df, schema=schema, parent_command="audit")

    # For bundle consistency with other commands: also create results.json + report.md
    results_payload: dict[str, Any] = {
        "command": "audit",
        "inputs": {"input": args.input, "schema": schema},
        "estimates": {},
        "diagnostics": {"shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}},
        "warnings": audit_payload.get("warnings", []),
        "artifacts": {
            "report_md": "report.md",
            "plots": audit_payload.get("artifacts", {}).get("plots", []),
            "tables": audit_payload.get("artifacts", {}).get("tables", []),
        },
    }
    write_results_json(out_dir, results_payload)

    # Make report.md a human-friendly entry point: reuse audit.md
    audit_md_path = out_dir / "audit.md"
    if audit_md_path.exists():
        write_report_md(out_dir, audit_md_path.read_text(encoding="utf-8"))
    else:
        # Should never happen, but keep bundle valid
        write_report_md(out_dir, "# tecore audit report\n\n(audit.md was not created)\n")

    return 0
