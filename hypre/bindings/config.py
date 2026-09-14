"""Runtime setup overrides shared by setup and solve workflows."""

import os
from typing import Any, Dict


SMOOTHER_PROFILES = (
    "hypre_default",
    "legacy_l1_jacobi",
    "l1_jacobi_direct_coarse",
)


def configure_smoother_profile(profile: str) -> None:
    """Choose native defaults or reproduce the historical experiment default.

    Native mode deliberately leaves all HYPRE smoother setters untouched.
    Explicit parameter dictionaries can still request a different smoother.
    """
    if profile not in SMOOTHER_PROFILES:
        raise ValueError(f"smoother_profile must be one of {SMOOTHER_PROFILES}")
    os.environ.pop("AMG_COARSE_RELAX_TYPE", None)
    if profile == "hypre_default":
        os.environ.pop("AMG_RELAX_TYPE", None)
        os.environ.pop("SETUP_RELAX_TYPE", None)
    elif profile == "l1_jacobi_direct_coarse":
        # Explicit experiment profile: change only the legacy coarsest solver.
        os.environ["AMG_RELAX_TYPE"] = "18"
        os.environ["AMG_COARSE_RELAX_TYPE"] = "9"
        os.environ.pop("SETUP_RELAX_TYPE", None)
    else:
        # Older JSON configs retain their old default; explicit env overrides
        # remain supported for historical experiment reproduction.
        os.environ.setdefault("AMG_RELAX_TYPE", "18")


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
