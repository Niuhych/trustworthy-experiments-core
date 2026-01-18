"""Sequential analysis utilities.

This package provides a pragmatic v1 implementation of sequential monitoring:

* Two-group A/B (control vs test)
* Mean-difference (user-level)
* Ratio metrics via linearization
* Two monitoring modes:
    - Group sequential with K looks (Pocock / O'Brien-Fleming style boundaries)
    - Anytime-valid monitoring via a conservative normal-mixture confidence sequence

The anytime-valid mode is based on a normal approximation + information-time
calculus. It is intentionally conservative and should be presented with
appropriate caveats (CLT, independence, stable variance, etc.).
"""

from tecore.sequential.schema import (
    EffectDirection,
    LookSchedule,
    SequentialConfig,
    SequentialMode,
    SequentialResult,
    SequentialSpec,
    SpendingFunction,
)
from tecore.sequential.group_sequential import run_group_sequential  
from tecore.sequential.confidence_sequences import run_confidence_sequence  
from tecore.sequential.ratio import linearize_ratio_frame  
