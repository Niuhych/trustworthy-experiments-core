"""
Backward-compatible import path.

Preferred v3 path:
  from tecore.causal.simulate_ts import generate_synthetic_time_series, SyntheticTSConfig

Legacy-friendly:
  from tecore.simulate_ts import generate_synthetic_time_series
"""
from tecore.causal.simulate_ts import SyntheticTSConfig, generate_synthetic_time_series  # noqa: F401
