"""Setup-phase LinUCB construction and parameter spaces for the SISC studies."""

from .registry import (
    COMPOSABLE_SETUP_KINDS,
    ONLINE_SETUP_KINDS,
    SetupLearnerBuildSpec,
    build_online_setup_learner,
    make_setup_learner_spec,
    normalize_setup_kind,
    setup_kind_registration,
)

__all__ = [
    "COMPOSABLE_SETUP_KINDS",
    "ONLINE_SETUP_KINDS",
    "SetupLearnerBuildSpec",
    "build_online_setup_learner",
    "make_setup_learner_spec",
    "normalize_setup_kind",
    "setup_kind_registration",
]
