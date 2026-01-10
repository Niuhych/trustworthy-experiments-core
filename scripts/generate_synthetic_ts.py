from __future__ import annotations

import argparse
import json
from pathlib import Path

from tecore.causal.simulate_ts import SyntheticTSConfig, generate_synthetic_time_series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic time-series dataset for causal impact demos.")
    p.add_argument("--out-dir", type=str, default="data/synthetic_ts", help="Output directory")
    p.add_argument("--n-days", type=int, default=200)
    p.add_argument("--start-date", type=str, default="2025-01-01")
    p.add_argument("--intervention-day", type=int, default=120)

    # effects
    p.add_argument("--level-shift", type=float, default=12.0)
    p.add_argument("--slope-change", type=float, default=0.0)
    p.add_argument("--temp-effect-amp", type=float, default=8.0)
    p.add_argument("--temp-effect-decay", type=float, default=0.12)

    # confounding
    p.add_argument("--confounding", action="store_true")
    p.add_argument("--confound-shift", type=float, default=80.0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SyntheticTSConfig(
        n_days=a.n_days,
        start_date=a.start_date,
        intervention_day=a.intervention_day,
        level_shift=a.level_shift,
        slope_change=a.slope_change,
        temp_effect_amp=a.temp_effect_amp,
        temp_effect_decay=a.temp_effect_decay,
        confounding=bool(a.confounding),
        confound_shift=a.confound_shift,
        random_state=a.seed,
    )

    df, meta = generate_synthetic_time_series(cfg)
    df.to_csv(out_dir / "synthetic_daily.csv", index=False)

    meta_out = {"config": cfg.__dict__, "meta": meta}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Wrote: {out_dir / 'synthetic_daily.csv'}")
    print(f"Wrote: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
