from __future__ import annotations


def render_cuped_report(res: dict) -> str:
    lines: list[str] = []
    lines.append("# CUPED report")
    lines.append("")

    lines.append("## Input")
    lines.append(f"- file: {res.get('input')}")
    lines.append(f"- groups: {res.get('control')} vs {res.get('test')} (col: {res.get('group_col')})")
    lines.append(f"- Y (post): {res.get('y')}")
    lines.append(f"- X (pre): {res.get('x')}")

    transform = res.get("transform")
    winsor_q = res.get("winsor_q")
    if transform == "winsor" and winsor_q is not None:
        lines.append(f"- transform: winsor (q={winsor_q})")
    else:
        lines.append(f"- transform: {transform}")

    lines.append("")
    lines.append("## Sample sizes")
    lines.append(f"- n_control: {res.get('n_control')}")
    lines.append(f"- n_test: {res.get('n_test')}")

    lines.append("")
    lines.append("## Results")
    lines.append(f"- p-value (base): {res.get('p_value_base'):.6f}")
    lines.append(f"- p-value (CUPED): {res.get('p_value_cuped'):.6f}")
    lines.append(f"- theta: {res.get('theta'):.6f}")
    lines.append(f"- variance reduction (control): {res.get('var_reduction_control'):.3f}")
    lines.append(f"- variance reduction (test): {res.get('var_reduction_test'):.3f}")

    alpha = res.get("alpha")
    reject_base = res.get("reject_base")
    reject_cuped = res.get("reject_cuped")
    if alpha is not None and reject_base is not None and reject_cuped is not None:
        lines.append(f"- reject H0 (base, alpha={alpha}): {reject_base}")
        lines.append(f"- reject H0 (CUPED, alpha={alpha}): {reject_cuped}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- CUPED is valid when the covariate X is measured pre-experiment and is not affected by treatment (no leakage).")
    lines.append("- Under heavy tails or after nonlinear transforms, consider cross-fitting CUPED to reduce sensitivity to overfitting (planned).")
    lines.append("")

    return "\n".join(lines)
