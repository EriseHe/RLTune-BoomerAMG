"""Locked legacy study rosters and validation.

These names remain importable through ``run_joint_online_sarsa_4k`` for old
diagnostics, but they are intentionally kept out of the composable runner's
controller construction and typed configuration layers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from joint_controller_build import (
    build_lsvi_controller_bundle,
    build_recalibrated_lsvi_controller_bundle,
    build_recursive_lstdq_controller_bundle,
    build_recursive_lstdq_v2_controller_bundle,
    build_recursive_mc_controller_bundle,
    build_structured_model_based_controller_bundle,
    make_encoder,
    parse_csv_values,
)
from solve.controllers.common import ControllerBundle
from solve.controllers.sarsa import (
    BehaviorPolicySarsaController,
    ExpectedSarsaLambdaConfig,
    SarsaBehaviorSpec,
    SolveStateEncoder,
)


REFERENCE_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    "bandit_ppo",
)
DEFAULT_SETUP_METHOD = "default_setup"
LSVI_METHOD = "bandit_stagewise_lsvi_lcb"
LSVI_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    LSVI_METHOD,
)
RECURSIVE_MC_METHOD = "bandit_recursive_mc_lcb"
RECURSIVE_LSTDQ_METHOD = "bandit_recursive_lstdq_lcb"
RECURSIVE_LSTDQ_V2_METHOD = "bandit_recursive_lstdq_v2_lcb"
STRUCTURED_MODEL_BASED_METHOD = "bandit_structured_model_based"
RECALIBRATED_LSVI_METHOD = "bandit_recalibrated_lsvi_lcb"
BATCHED_LSVI_METHOD = "bandit_batched_lsvi_lcb"
RECURSIVE_LCB_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    RECURSIVE_MC_METHOD,
    RECURSIVE_LSTDQ_METHOD,
    BATCHED_LSVI_METHOD,
)
RECURSIVE_LCB_PPO_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    "bandit_ppo",
    RECURSIVE_MC_METHOD,
    RECURSIVE_LSTDQ_METHOD,
)
RECURSIVE_LSTDQ_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    RECURSIVE_LSTDQ_METHOD,
)
SOLVE_CONTROLLER_SCREEN_METHODS = (
    "bandit_fixed_w1.6",
    RECURSIVE_LSTDQ_METHOD,
    RECURSIVE_LSTDQ_V2_METHOD,
    STRUCTURED_MODEL_BASED_METHOD,
    RECALIBRATED_LSVI_METHOD,
)
SOLVE_CONTROLLER_SEED_OFFSETS = {
    RECURSIVE_LSTDQ_METHOD: 1009,
    RECURSIVE_LSTDQ_V2_METHOD: 2018,
    STRUCTURED_MODEL_BASED_METHOD: 3027,
    RECALIBRATED_LSVI_METHOD: 4036,
}
BEHAVIOR_MODES = ("uniform", "uncertainty_lcb")

SHARED_ACTION_PROFILES = {
    "1to2_step0p1": {
        "weights": tuple(
            float(value) for value in np.linspace(1.0, 2.0, 11)
        ),
        "centers": tuple(
            float(value) for value in np.linspace(1.0, 2.0, 5)
        ),
    },
    "1to3_step0p05": {
        "weights": tuple(
            float(value) for value in np.linspace(1.0, 3.0, 41)
        ),
        "centers": tuple(
            float(value) for value in np.linspace(1.0, 3.0, 9)
        ),
    },
}


@dataclass(frozen=True)
class SarsaCandidate:
    behavior_mode: str
    alpha: float
    trace_lambda: float

    @property
    def name(self) -> str:
        alpha = f"{self.alpha:g}".replace(".", "p")
        trace_lambda = f"{self.trace_lambda:g}".replace(".", "p")
        return (
            f"bandit_sarsa_{self.behavior_mode}_"
            f"alpha_{alpha}_lambda_{trace_lambda}"
        )

    @property
    def family(self) -> str:
        return f"sarsa_{self.behavior_mode}"


@dataclass(frozen=True)
class LegacyStudy:
    """Resolved roster and setup-bandit ownership for one archived mode."""

    study_mode: str
    candidates: tuple[SarsaCandidate, ...]
    methods: tuple[str, ...]
    family_by_method: Mapping[str, str]
    bandit_methods: tuple[str, ...]


@dataclass(frozen=True)
class LegacySolveRuntime:
    """Built online controllers and optional frozen PPO policy for a legacy mode."""

    controller_bundles: Mapping[str, ControllerBundle]
    ppo_runner: Any | None


def candidate_grid(
    *,
    alphas: Sequence[float],
    trace_lambdas: Sequence[float],
) -> tuple[SarsaCandidate, ...]:
    return tuple(
        SarsaCandidate(behavior_mode, float(alpha), float(trace_lambda))
        for behavior_mode in BEHAVIOR_MODES
        for alpha in alphas
        for trace_lambda in trace_lambdas
    )


def resolve_legacy_study(args: Any) -> LegacyStudy:
    """Resolve a frozen legacy roster without constructing its controllers."""

    study_mode = str(args.study_mode)
    candidates: tuple[SarsaCandidate, ...] = ()
    if study_mode == "lsvi_lcb":
        methods = LSVI_METHODS
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            LSVI_METHOD: "stagewise_lsvi_lcb",
        }
    elif study_mode == "solve_controller_screen":
        if bool(args.include_default_setup_baseline):
            raise ValueError(
                "solve_controller_screen has a locked five-method roster"
            )
        methods = SOLVE_CONTROLLER_SCREEN_METHODS
        family_by_method = {
            "bandit_fixed_w1.6": "fixed_w1.6",
            RECURSIVE_LSTDQ_METHOD: "recursive_lstdq_lcb",
            RECURSIVE_LSTDQ_V2_METHOD: "recursive_lstdq_v2_lcb",
            STRUCTURED_MODEL_BASED_METHOD: "structured_model_based",
            RECALIBRATED_LSVI_METHOD: "recalibrated_lsvi_lcb",
        }
    elif study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        if study_mode == "recursive_lcb_suite":
            methods = RECURSIVE_LCB_METHODS
        elif study_mode == "recursive_lcb_ppo":
            methods = RECURSIVE_LCB_PPO_METHODS
        else:
            methods = RECURSIVE_LSTDQ_METHODS
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            **(
                {"bandit_ppo": "ppo"}
                if "bandit_ppo" in methods
                else {}
            ),
            **(
                {RECURSIVE_MC_METHOD: "recursive_mc_lcb"}
                if RECURSIVE_MC_METHOD in methods
                else {}
            ),
            **(
                {RECURSIVE_LSTDQ_METHOD: "recursive_lstdq_lcb"}
                if RECURSIVE_LSTDQ_METHOD in methods
                else {}
            ),
            **(
                {BATCHED_LSVI_METHOD: "batched_lsvi_lcb"}
                if BATCHED_LSVI_METHOD in methods
                else {}
            ),
        }
    elif study_mode == "sarsa":
        candidates = candidate_grid(
            alphas=parse_csv_values(args.alphas, float),
            trace_lambdas=parse_csv_values(args.trace_lambdas, float),
        )
        methods = (
            *REFERENCE_METHODS,
            *[candidate.name for candidate in candidates],
        )
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            "bandit_ppo": "ppo",
            **{
                candidate.name: candidate.family
                for candidate in candidates
            },
        }
    else:
        raise ValueError(f"Unknown legacy study mode: {study_mode}")

    if bool(args.include_default_setup_baseline):
        methods = (DEFAULT_SETUP_METHOD, *methods)
        family_by_method = {
            DEFAULT_SETUP_METHOD: "default_setup",
            **family_by_method,
        }
    resolved_methods = tuple(methods)
    return LegacyStudy(
        study_mode=study_mode,
        candidates=tuple(candidates),
        methods=resolved_methods,
        family_by_method=dict(family_by_method),
        bandit_methods=tuple(
            method
            for method in resolved_methods
            if method != DEFAULT_SETUP_METHOD
        ),
    )


def build_sarsa_controller_bundle(
    args: argparse.Namespace,
    candidate: SarsaCandidate,
    *,
    seed: int,
) -> ControllerBundle:
    """Build one archived true-online SARSA controller as a solve bundle."""

    controller, encoder = _build_sarsa_controller_parts(
        args,
        candidate,
        seed=seed,
    )
    return ControllerBundle(
        controller=controller,
        encoder=encoder,
        _protocol_metadata={
            "kind": "true_online_sarsa",
            "family": candidate.family,
            "state": "setup_full",
            "actions": [
                float(weight)
                for weight in parse_csv_values(args.weights, float)
            ],
            "behavior_mode": str(candidate.behavior_mode),
            "alpha": float(candidate.alpha),
            "trace_lambda": float(candidate.trace_lambda),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        },
    )


def make_sarsa_controller(
    args: argparse.Namespace,
    candidate: SarsaCandidate,
    *,
    seed: int,
) -> tuple[BehaviorPolicySarsaController, SolveStateEncoder]:
    """Compatibility tuple for archived diagnostics."""

    return build_sarsa_controller_bundle(
        args,
        candidate,
        seed=seed,
    ).as_legacy_tuple()


def _build_sarsa_controller_parts(
    args: argparse.Namespace,
    candidate: SarsaCandidate,
    *,
    seed: int,
) -> tuple[BehaviorPolicySarsaController, SolveStateEncoder]:
    encoder = make_encoder(args)
    weights = parse_csv_values(args.weights, float)
    config = ExpectedSarsaLambdaConfig(
        weights=tuple(float(weight) for weight in weights),
        anchor_weight=float(weights[0]),
        alpha=float(candidate.alpha),
        td_decay_power=0.0,
        gamma=1.0,
        trace_lambda=float(candidate.trace_lambda),
        epsilon_start=float(args.epsilon_start),
        epsilon_final=float(args.epsilon_final),
        epsilon_decay_steps=float(args.epsilon_decay_steps),
        initial_q_sec=0.0,
        monte_carlo_alpha=0.0,
        adaptive_cycles=None,
        exploration_mode="uniform",
        action_rbf_sigma=0.0,
        td_algorithm="true_online_sarsa",
        force_default_first_action=True,
    )
    behavior_spec = SarsaBehaviorSpec(
        action_mode="absolute",
        behavior_mode=str(candidate.behavior_mode),
        uncertainty_beta=float(args.uncertainty_beta),
        uncertainty_ridge=float(args.uncertainty_ridge),
        uncertainty_td_floor_sec=float(args.uncertainty_td_floor_sec),
    )
    controller = BehaviorPolicySarsaController(
        behavior_spec=behavior_spec,
        feature_dim=encoder.feature_dim,
        config=config,
        seed=int(seed),
        initial_parameters=encoder.constant_value_parameters(0.0),
    )
    return controller, encoder


def build_legacy_solve_runtime(
    args: Any,
    *,
    study: LegacyStudy,
    make_ppo_runner: Callable[[Any], Any],
) -> LegacySolveRuntime:
    """Construct only the controllers selected by a resolved legacy roster."""

    controller_seed = int(args.controller_seed)
    controller_bundles: dict[str, ControllerBundle] = {}
    study_mode = str(args.study_mode)
    if study.study_mode != study_mode:
        raise ValueError(
            "Legacy study/runtime mode mismatch: "
            f"{study.study_mode!r} != {study_mode!r}"
        )

    if study_mode == "lsvi_lcb":
        controller_bundles[LSVI_METHOD] = build_lsvi_controller_bundle(
            args,
            seed=controller_seed,
            refit_interval_episodes=1,
        )
    elif study_mode == "solve_controller_screen":
        factories = {
            RECURSIVE_LSTDQ_METHOD: (
                build_recursive_lstdq_controller_bundle
            ),
            RECURSIVE_LSTDQ_V2_METHOD: (
                build_recursive_lstdq_v2_controller_bundle
            ),
            STRUCTURED_MODEL_BASED_METHOD: (
                build_structured_model_based_controller_bundle
            ),
            RECALIBRATED_LSVI_METHOD: (
                build_recalibrated_lsvi_controller_bundle
            ),
        }
        for method, factory in factories.items():
            controller_bundles[method] = factory(
                args,
                seed=(
                    controller_seed
                    + SOLVE_CONTROLLER_SEED_OFFSETS[method]
                ),
            )
    elif study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        factories = {
            RECURSIVE_MC_METHOD: (
                build_recursive_mc_controller_bundle,
                0,
            ),
            RECURSIVE_LSTDQ_METHOD: (
                build_recursive_lstdq_controller_bundle,
                1009,
            ),
            BATCHED_LSVI_METHOD: (
                build_lsvi_controller_bundle,
                2018,
            ),
        }
        for method, (factory, seed_offset) in factories.items():
            if method not in study.methods:
                continue
            factory_kwargs: dict[str, Any] = {
                "seed": controller_seed + seed_offset,
            }
            if method == BATCHED_LSVI_METHOD:
                factory_kwargs["refit_interval_episodes"] = int(
                    args.lsvi_refit_interval_episodes
                )
            controller_bundles[method] = factory(
                args,
                **factory_kwargs,
            )
    elif study_mode == "sarsa":
        for candidate_index, candidate in enumerate(study.candidates):
            controller_bundles[candidate.name] = (
                build_sarsa_controller_bundle(
                    args,
                    candidate,
                    seed=controller_seed + candidate_index * 1009,
                )
            )
    else:
        raise ValueError(f"Unknown legacy study mode: {study_mode}")

    ppo_runner = (
        make_ppo_runner(args)
        if "bandit_ppo" in study.methods
        else None
    )
    return LegacySolveRuntime(
        controller_bundles=controller_bundles,
        ppo_runner=ppo_runner,
    )


def validate_shared_lcb_protocol(args: argparse.Namespace) -> None:
    """Validate the frozen action/controller controls of archived studies."""

    profile_name = str(
        getattr(args, "shared_action_profile", "1to2_step0p1")
    )
    if profile_name not in SHARED_ACTION_PROFILES:
        raise ValueError(f"Unknown shared-action profile: {profile_name}")
    profile = SHARED_ACTION_PROFILES[profile_name]
    expected_weights = profile["weights"]
    expected_centers = profile["centers"]
    if getattr(args, "weights", None) is None:
        args.weights = ",".join(
            f"{value:g}" for value in expected_weights
        )
    if getattr(args, "action_rbf_centers", None) is None:
        args.action_rbf_centers = ",".join(
            f"{value:g}" for value in expected_centers
        )
    weights = parse_csv_values(args.weights, float)
    centers = parse_csv_values(args.action_rbf_centers, float)
    if not np.allclose(
        weights,
        expected_weights,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"Weights do not match shared-action profile {profile_name}"
        )
    if not np.allclose(
        centers,
        expected_centers,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"RBF centers do not match shared-action profile {profile_name}"
        )
    locked_scalars = {
        "action_rbf_sigma": (float(args.action_rbf_sigma), 0.2),
        "lsvi_ridge": (float(args.lsvi_ridge), 1.0),
        "lsvi_beta": (float(args.lsvi_beta), 2.0),
        "lsvi_residual_floor_sec": (
            float(args.lsvi_residual_floor_sec),
            1.0e-3,
        ),
    }
    for name, (actual, expected) in locked_scalars.items():
        if not np.isclose(actual, expected, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"Locked shared-action {name} must equal {expected:g}"
            )
    if getattr(args, "study_mode", "lsvi_lcb") in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
        "solve_controller_screen",
    }:
        recursive_scalars = {
            "recursive_mc_ridge": (
                float(args.recursive_mc_ridge),
                1.0,
            ),
            "recursive_mc_beta": (
                float(args.recursive_mc_beta),
                2.0,
            ),
            "recursive_mc_episode_half_life": (
                float(args.recursive_mc_episode_half_life),
                500.0,
            ),
            "recursive_lstdq_ridge": (
                float(args.recursive_lstdq_ridge),
                1.0,
            ),
            "recursive_lstdq_beta": (
                float(args.recursive_lstdq_beta),
                2.0,
            ),
            "recursive_lstdq_lambda": (
                float(args.recursive_lstdq_lambda),
                0.8,
            ),
            "recursive_lstdq_lcb_lower_bound_sec": (
                float(args.recursive_lstdq_lcb_lower_bound_sec),
                0.0,
            ),
        }
        for name, (actual, expected) in recursive_scalars.items():
            if not np.isclose(
                actual,
                expected,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Locked shared-action {name} must equal {expected:g}"
                )
        if int(args.lsvi_refit_interval_episodes) != 100:
            raise ValueError(
                "Locked batched LSVI refit interval must equal 100"
            )
    if getattr(args, "study_mode", "") == "solve_controller_screen":
        locked_screen_values = {
            "recursive_lstdq_v2_coverage_ridge": (
                float(args.recursive_lstdq_v2_coverage_ridge),
                1.0,
            ),
            "recursive_lstdq_v2_residual_window": (
                int(args.recursive_lstdq_v2_residual_window),
                2048,
            ),
            "recursive_lstdq_v2_min_samples": (
                int(args.recursive_lstdq_v2_min_samples),
                32,
            ),
            "structured_model_ridge": (
                float(args.structured_model_ridge),
                1.0,
            ),
            "structured_model_min_samples": (
                int(args.structured_model_min_samples),
                32,
            ),
            "structured_model_scale_window": (
                int(args.structured_model_scale_window),
                2048,
            ),
            "recalibrated_lsvi_refit_sweeps": (
                int(args.recalibrated_lsvi_refit_sweeps),
                3,
            ),
            "recalibrated_lsvi_shrinkage_samples": (
                float(args.recalibrated_lsvi_shrinkage_samples),
                32.0,
            ),
        }
        for name, (actual, expected) in locked_screen_values.items():
            if actual != expected:
                raise ValueError(
                    f"Locked solve screen {name} must equal {expected:g}"
                )


def validate_lsvi_protocol(args: argparse.Namespace) -> None:
    """Backward-compatible alias used by the archived protocol tests."""

    validate_shared_lcb_protocol(args)


__all__ = [
    "BATCHED_LSVI_METHOD",
    "BEHAVIOR_MODES",
    "DEFAULT_SETUP_METHOD",
    "LSVI_METHOD",
    "LSVI_METHODS",
    "LegacySolveRuntime",
    "LegacyStudy",
    "RECALIBRATED_LSVI_METHOD",
    "RECURSIVE_LCB_METHODS",
    "RECURSIVE_LCB_PPO_METHODS",
    "RECURSIVE_LSTDQ_METHOD",
    "RECURSIVE_LSTDQ_METHODS",
    "RECURSIVE_LSTDQ_V2_METHOD",
    "RECURSIVE_MC_METHOD",
    "REFERENCE_METHODS",
    "SHARED_ACTION_PROFILES",
    "SOLVE_CONTROLLER_SCREEN_METHODS",
    "SOLVE_CONTROLLER_SEED_OFFSETS",
    "STRUCTURED_MODEL_BASED_METHOD",
    "SarsaCandidate",
    "build_legacy_solve_runtime",
    "build_sarsa_controller_bundle",
    "candidate_grid",
    "make_sarsa_controller",
    "resolve_legacy_study",
    "validate_lsvi_protocol",
    "validate_shared_lcb_protocol",
]
