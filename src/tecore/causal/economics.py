from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_economics(
    cum_effect: float,
    cum_ci: Tuple[float, float],
    margin_rate: Optional[float] = None,
    campaign_spend: Optional[float] = None,
    incremental_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Translate causal lift to economic metrics.
    Assumptions:
      - cum_effect is incremental revenue (or another monetary KPI) over post-period.
      - margin_rate converts revenue -> profit.
      - campaign_spend and incremental_cost are costs to subtract.
    """
    out: Dict[str, Any] = {}

    out["incremental_revenue"] = float(cum_effect)
    out["incremental_revenue_ci"] = (float(cum_ci[0]), float(cum_ci[1]))

    if margin_rate is not None:
        mr = float(margin_rate)
        out["margin_rate"] = mr
        incr_profit = cum_effect * mr
        out["incremental_profit_gross"] = float(incr_profit)
        out["incremental_profit_gross_ci"] = (float(cum_ci[0] * mr), float(cum_ci[1] * mr))

        costs = 0.0
        if campaign_spend is not None:
            out["campaign_spend"] = float(campaign_spend)
            costs += float(campaign_spend)
        if incremental_cost is not None:
            out["incremental_cost"] = float(incremental_cost)
            costs += float(incremental_cost)

        out["total_costs"] = float(costs)
        out["incremental_profit_net"] = float(incr_profit - costs)
        out["incremental_profit_net_ci"] = (float(cum_ci[0] * mr - costs), float(cum_ci[1] * mr - costs))

        if costs > 0:
            out["roi_net"] = float((incr_profit - costs) / costs)
            out["roi_net_ci"] = (float((cum_ci[0] * mr - costs) / costs), float((cum_ci[1] * mr - costs) / costs))
        else:
            out["roi_net"] = None
            out["roi_net_ci"] = None

    return out


def sum_spend_over_post(
    df: pd.DataFrame,
    spend_col: str,
    post_mask: pd.Series,
) -> Optional[float]:
    if spend_col not in df.columns:
        return None
    s = df.loc[post_mask, spend_col]
    if s.isna().all():
        return None
    return float(np.nansum(s.values))
