"""Runtime setup overrides shared by setup and solve workflows."""

import os
from typing import Any, Dict


def augment_setup_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply repository-level environment overrides before AMG construction."""
    out = dict(params)
    overrides = (
        ("SETUP_RELAX_TYPE", "relax_type", int),
        ("SETUP_NUM_SWEEPS", "num_sweeps", int),
        ("SETUP_CYCLE_TYPE", "cycle_type", int),
        ("SETUP_MAX_LEVELS", "max_levels", int),
    )
    for environment_name, parameter_name, converter in overrides:
        raw_value = os.environ.get(environment_name, "").strip()
        if raw_value:
            out[parameter_name] = converter(raw_value)
    return out
