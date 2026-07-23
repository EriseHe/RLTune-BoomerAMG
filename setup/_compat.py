"""Install import aliases for checkpoints created before the package rename."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Mapping
import sys


_LEARNER_SUBMODULES = {
    "registry": "setup.registry",
    "Bayesianbandits_AMG_v1": (
        "setup.learners.bayesian.Bayesianbandits_AMG_v1"
    ),
    "Bayesianbandits_AMG_v2": (
        "setup.learners.bayesian.Bayesianbandits_AMG_v2"
    ),
    "LinUCB_AMG": "setup.learners.linucb.LinUCB_AMG",
    "LinUCB_AMG_v2": "setup.learners.linucb.LinUCB_AMG_v2",
    "SharedLinUCB_AMG": "setup.learners.linucb.SharedLinUCB_AMG",
    "SharedLinUCB_AMG_v2": "setup.learners.linucb.SharedLinUCB_AMG_v2",
    "SharedLinUCB_AMG_v3": "setup.learners.linucb.SharedLinUCB_AMG_v3",
    "SharedLinUCB_AMG_v4": "setup.learners.linucb.SharedLinUCB_AMG_v4",
    "RFF_TS_AMG": "setup.learners.thompson.RFF_TS_AMG",
    "SharedBootstrapTS_AMG": (
        "setup.learners.thompson.SharedBootstrapTS_AMG"
    ),
    "SharedLinTS_AMG": "setup.learners.thompson.SharedLinTS_AMG",
    "SharedLinTS_AMG_v2": "setup.learners.thompson.SharedLinTS_AMG_v2",
    "TsallisINF_AMG": "setup.learners.tsallis.TsallisINF_AMG",
    "_amg_action_features": "setup.learners.common.action_features",
    "_aot_candidates": "setup.learners.common.aot_candidates",
    "_candidate_subset": "setup.learners.common.candidate_subset",
    "bayesian": "setup.learners.bayesian",
    "bayesian.Bayesianbandits_AMG_v1": (
        "setup.learners.bayesian.Bayesianbandits_AMG_v1"
    ),
    "bayesian.Bayesianbandits_AMG_v2": (
        "setup.learners.bayesian.Bayesianbandits_AMG_v2"
    ),
    "common": "setup.learners.common",
    "common.action_features": "setup.learners.common.action_features",
    "common.aot_candidates": "setup.learners.common.aot_candidates",
    "common.candidate_subset": "setup.learners.common.candidate_subset",
    "common.config": "setup.learners.common.config",
    "linucb": "setup.learners.linucb",
    "linucb.LinUCB_AMG": "setup.learners.linucb.LinUCB_AMG",
    "linucb.LinUCB_AMG_v2": "setup.learners.linucb.LinUCB_AMG_v2",
    "linucb.SharedLinUCB_AMG": (
        "setup.learners.linucb.SharedLinUCB_AMG"
    ),
    "linucb.SharedLinUCB_AMG_v2": (
        "setup.learners.linucb.SharedLinUCB_AMG_v2"
    ),
    "linucb.SharedLinUCB_AMG_v3": (
        "setup.learners.linucb.SharedLinUCB_AMG_v3"
    ),
    "linucb.SharedLinUCB_AMG_v4": (
        "setup.learners.linucb.SharedLinUCB_AMG_v4"
    ),
    "linucb.setup_reselection": "setup.learners.linucb.setup_reselection",
    "thompson": "setup.learners.thompson",
    "thompson.RFF_TS_AMG": "setup.learners.thompson.RFF_TS_AMG",
    "thompson.SharedBootstrapTS_AMG": (
        "setup.learners.thompson.SharedBootstrapTS_AMG"
    ),
    "thompson.SharedLinTS_AMG": (
        "setup.learners.thompson.SharedLinTS_AMG"
    ),
    "thompson.SharedLinTS_AMG_v2": (
        "setup.learners.thompson.SharedLinTS_AMG_v2"
    ),
    "tsallis": "setup.learners.tsallis",
    "tsallis.TsallisINF_AMG": "setup.learners.tsallis.TsallisINF_AMG",
}

_UTILITY_SUBMODULES = {
    "paths": "setup.utils.paths",
    "plotting_amg": "setup.utils.plotting_amg",
    "setup_amg": "setup.utils.setup_amg",
}


def _install_module_tree(
    alias_root: str,
    canonical_root: str,
    submodules: Mapping[str, str],
) -> ModuleType:
    canonical = import_module(canonical_root)
    sys.modules[alias_root] = canonical
    for suffix, canonical_name in submodules.items():
        sys.modules[f"{alias_root}.{suffix}"] = import_module(canonical_name)
    return canonical


def install_legacy_learner_aliases(alias_root: str) -> ModuleType:
    """Make one historical learner root resolve to canonical modules."""

    return _install_module_tree(
        alias_root,
        "setup.learners",
        _LEARNER_SUBMODULES,
    )


def install_legacy_utility_aliases(alias_root: str) -> ModuleType:
    """Make one historical utility root resolve to canonical modules."""

    return _install_module_tree(
        alias_root,
        "setup.utils",
        _UTILITY_SUBMODULES,
    )


def install_checkpoint_aliases() -> None:
    """Expose module names embedded in historical setup learner pickles."""

    install_legacy_learner_aliases("learners")


__all__ = [
    "install_checkpoint_aliases",
    "install_legacy_learner_aliases",
    "install_legacy_utility_aliases",
]
