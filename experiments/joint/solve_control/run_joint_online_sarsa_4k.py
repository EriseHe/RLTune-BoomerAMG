from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import csv
import json
import platform
import shlex
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from hypre.bindings import run_with_default_fallback

from setup_action_space import (
    DEFAULT_SETUP_PARAMS,
    SetupConfigurationSpace,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from SolvePhase.algorithms.sarsa import (
    BehaviorPolicySarsaController,
    SarsaBehaviorSpec,
)
from online_td_experiment_common import _json_ready, _write_json
from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _git_revision,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from run_online_methods_2k import (
    _as_feedback,
    _method_comparison,
    make_exp44_ppo_runner,
)
from setup_aware_compare_common import (
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    augment_setup_params,
    build_online_linucb_branch,
    classify_rl_failure,
    clone_branch_for_independent_updates,
    default_test_final_bandit_config_from_env,
    run_bandit_step_test_final,
    solve_default_baseline_case,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
    validate_expected_setup_action_count,
)

_build_bandit = build_online_linucb_branch
from SolvePhase.algorithms.lcb import (
    HierarchicalLsviLcbController,
    HierarchicalLsviLcbSpec,
    RecursiveBlstdqController,
    RecursiveBlstdqSpec,
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbController,
    RecursiveLstdqV2LcbSpec,
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
    StructuredModelBasedController,
    StructuredModelBasedSpec,
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

SETUP_BANDIT_KINDS = ("linucb", "lints")
SETUP_CANDIDATE_SAMPLING_METHODS = ("uniform512", "structured512")
COMPOSABLE_SETUP_KINDS = ("default", *SETUP_BANDIT_KINDS)
COMPOSABLE_SOLVE_KINDS = (
    "default",
    "fixed",
    "ppo",
    "recursive_mc",
    "recursive_lstdq_v1",
    "recursive_lstdq_v2",
    "rblspi",
    "stagewise_lsvi",
    "structured_model_based",
    "recalibrated_lsvi",
)


@dataclass(frozen=True)
class ComposableMethodSpec:
    """One independently learned setup/solve branch in a joint experiment."""

    name: str
    setup_kind: str
    solve_kind: str
    setup_space: str | None = None
    candidate_sampling: str = "uniform512"
    fixed_weight: float | None = None

    @property
    def family(self) -> str:
        if self.setup_kind == "default" and self.solve_kind == "default":
            return "default_setup"
        if self.setup_kind in SETUP_BANDIT_KINDS and self.solve_kind == "default":
            return "default"
        if self.setup_kind in SETUP_BANDIT_KINDS and self.solve_kind == "fixed":
            if np.isclose(float(self.fixed_weight), 1.6):
                return "fixed_w1.6"
            return f"fixed_w{float(self.fixed_weight):g}"
        family = {
            "ppo": "ppo",
            "recursive_mc": "recursive_mc_lcb",
            "recursive_lstdq_v1": "recursive_lstdq_lcb",
            "recursive_lstdq_v2": "recursive_lstdq_v2_lcb",
            "rblspi": "rblspi",
            "stagewise_lsvi": "stagewise_lsvi_lcb",
            "structured_model_based": "structured_model_based",
            "recalibrated_lsvi": "recalibrated_lsvi_lcb",
        }[self.solve_kind]
        if self.setup_kind == "default":
            return f"default_setup_{family}"
        return family

    @property
    def label(self) -> str:
        setup = {
            "default": "Default setup",
            "linucb": "Online LinUCB",
            "lints": "Online LinTS v2",
        }[self.setup_kind]
        if self.setup_space is not None:
            candidate = self.candidate_sampling.replace("512", "-512")
            setup = f"{setup} [{self.setup_space}; {candidate}]"
        if self.solve_kind == "fixed":
            solve = f"fixed w={float(self.fixed_weight):g}"
        else:
            solve = {
                "default": "default solve",
                "ppo": "PPO",
                "recursive_mc": "Recursive MC-LCB",
                "recursive_lstdq_v1": "Recursive LSTDQ-LCB",
                "recursive_lstdq_v2": "Recursive LSTDQ v2-LCB",
                "rblspi": "Recursive BLSTDQ / RBLSPI",
                "stagewise_lsvi": "Stagewise LSVI-LCB",
                "structured_model_based": "structured model-based control",
                "recalibrated_lsvi": "Recalibrated LSVI-LCB",
            }[self.solve_kind]
        return f"{setup} + {solve}"


def _parse_composable_method(raw: str) -> ComposableMethodSpec:
    """Parse ``name:setup:solve`` where fixed solves use ``fixed@weight``."""

    parts = str(raw).split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(
            "Composable methods must use name:setup:solve syntax"
        )
    name, setup_token, solve_token = (part.strip() for part in parts)
    if any(character in name for character in "/\\"):
        raise ValueError("Composable method names cannot contain path separators")
    setup_parts = setup_token.split("@")
    if len(setup_parts) > 3:
        raise ValueError(
            "Setup tokens use setup[@space[@candidate_sampling]] syntax"
        )
    setup_kind = setup_parts[0]
    setup_space = setup_parts[1].strip() if len(setup_parts) >= 2 else None
    candidate_sampling = (
        setup_parts[2].strip().lower().replace("-", "")
        if len(setup_parts) == 3
        else "uniform512"
    )
    if setup_space == "":
        raise ValueError("A setup-bandit space name cannot be empty")
    if setup_kind not in COMPOSABLE_SETUP_KINDS:
        raise ValueError(f"Unsupported setup kind: {setup_kind}")
    if setup_space is not None and setup_kind not in SETUP_BANDIT_KINDS:
        raise ValueError("Only setup-bandit methods can select a setup space")
    if candidate_sampling not in SETUP_CANDIDATE_SAMPLING_METHODS:
        raise ValueError(
            "candidate sampling must be uniform512 or structured512"
        )
    if setup_kind not in SETUP_BANDIT_KINDS and candidate_sampling != "uniform512":
        raise ValueError("Only setup-bandit methods select candidate sampling")
    fixed_weight = None
    if solve_token.startswith("fixed@"):
        solve_kind = "fixed"
        fixed_weight = float(solve_token.split("@", 1)[1])
        if not np.isfinite(fixed_weight) or fixed_weight <= 0.0:
            raise ValueError("A fixed solve weight must be finite and positive")
    else:
        solve_kind = solve_token
    if solve_kind not in COMPOSABLE_SOLVE_KINDS:
        raise ValueError(f"Unsupported solve kind: {solve_kind}")
    if solve_kind == "fixed" and fixed_weight is None:
        raise ValueError("Fixed solve methods must use fixed@weight")
    return ComposableMethodSpec(
        name=name,
        setup_kind=setup_kind,
        solve_kind=solve_kind,
        setup_space=setup_space,
        candidate_sampling=candidate_sampling,
        fixed_weight=fixed_weight,
    )


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


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in str(raw).split(",") if part.strip())


SHARED_ACTION_PROFILES = {
    "1to2_step0p1": {
        "weights": tuple(float(value) for value in np.linspace(1.0, 2.0, 11)),
        "centers": tuple(float(value) for value in np.linspace(1.0, 2.0, 5)),
    },
    "1to3_step0p05": {
        "weights": tuple(float(value) for value in np.linspace(1.0, 3.0, 41)),
        "centers": tuple(float(value) for value in np.linspace(1.0, 3.0, 9)),
    },
}


def _validate_shared_lcb_protocol(args: argparse.Namespace) -> None:
    profile_name = str(
        getattr(args, "shared_action_profile", "1to2_step0p1")
    )
    if profile_name not in SHARED_ACTION_PROFILES:
        raise ValueError(f"Unknown shared-action profile: {profile_name}")
    profile = SHARED_ACTION_PROFILES[profile_name]
    expected_weights = profile["weights"]
    expected_centers = profile["centers"]
    if getattr(args, "weights", None) is None:
        args.weights = ",".join(f"{value:g}" for value in expected_weights)
    if getattr(args, "action_rbf_centers", None) is None:
        args.action_rbf_centers = ",".join(
            f"{value:g}" for value in expected_centers
        )
    weights = _parse_values(args.weights, float)
    centers = _parse_values(args.action_rbf_centers, float)
    if not np.allclose(weights, expected_weights, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            f"Weights do not match shared-action profile {profile_name}"
        )
    if not np.allclose(centers, expected_centers, atol=1.0e-12, rtol=0.0):
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
            raise ValueError(f"Locked shared-action {name} must equal {expected:g}")
    if getattr(args, "study_mode", "lsvi_lcb") in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
        "solve_controller_screen",
    }:
        recursive_scalars = {
            "recursive_mc_ridge": (float(args.recursive_mc_ridge), 1.0),
            "recursive_mc_beta": (float(args.recursive_mc_beta), 2.0),
            "recursive_mc_episode_half_life": (
                float(args.recursive_mc_episode_half_life),
                500.0,
            ),
            "recursive_lstdq_ridge": (float(args.recursive_lstdq_ridge), 1.0),
            "recursive_lstdq_beta": (float(args.recursive_lstdq_beta), 2.0),
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
            if not np.isclose(actual, expected, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"Locked shared-action {name} must equal {expected:g}")
        if int(args.lsvi_refit_interval_episodes) != 100:
            raise ValueError("Locked batched LSVI refit interval must equal 100")
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
            "structured_model_ridge": (float(args.structured_model_ridge), 1.0),
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
                raise ValueError(f"Locked solve screen {name} must equal {expected:g}")


def _validate_lsvi_protocol(args: argparse.Namespace) -> None:
    """Backward-compatible name used by the existing protocol tests."""
    _validate_shared_lcb_protocol(args)


def _setup_configuration_spaces_from_args(
    args: argparse.Namespace,
) -> Dict[str, SetupConfigurationSpace]:
    raw_spaces = dict(
        getattr(args, "setup_configuration_spaces", {}) or {}
    )
    spaces: Dict[str, SetupConfigurationSpace] = {}
    for name, raw_space in raw_spaces.items():
        if not isinstance(raw_space, SetupConfigurationSpace):
            raise TypeError(
                f"setup configuration space {name!r} was not normalized by "
                "run_joint_experiment.py"
            )
        if str(name) != raw_space.name:
            raise ValueError(
                f"Setup configuration-space key {name!r} does not match "
                f"its name {raw_space.name!r}"
            )
        spaces[str(name)] = raw_space
    return spaces


def _validate_composable_protocol(
    args: argparse.Namespace,
) -> tuple[ComposableMethodSpec, ...]:
    specs = tuple(
        _parse_composable_method(raw)
        for raw in tuple(getattr(args, "method_specs", ()) or ())
    )
    if not specs:
        raise ValueError("Composable experiments require at least one --method")
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("Composable method names must be unique")

    configuration_spaces = _setup_configuration_spaces_from_args(args)
    referenced_spaces = {
        str(spec.setup_space)
        for spec in specs
        if spec.setup_space is not None
    }
    unknown_spaces = referenced_spaces - set(configuration_spaces)
    if unknown_spaces:
        raise ValueError(
            f"Methods reference unknown setup spaces: {sorted(unknown_spaces)}"
        )
    if configuration_spaces:
        missing_references = [
            spec.name
            for spec in specs
            if spec.setup_kind in SETUP_BANDIT_KINDS
            and spec.setup_space is None
        ]
        if missing_references:
            raise ValueError(
                "Every setup-bandit method must select setup_space when named "
                f"configuration spaces are present: {missing_references}"
            )
        unused_spaces = set(configuration_spaces) - referenced_spaces
        if unused_spaces:
            raise ValueError(
                f"Unused setup configuration spaces: {sorted(unused_spaces)}"
            )
        if str(args.setup_action_space) != "full_cartesian":
            raise ValueError(
                "Named setup configuration spaces require "
                "setup.action_space='full_cartesian'"
            )
    candidate_mode = str(
        getattr(args, "setup_candidate_mode", "explicit")
    ).strip().lower()
    if candidate_mode == "aot":
        if not configuration_spaces:
            raise ValueError(
                "AOT candidate schedules currently require named setup spaces"
            )
        if int(args.warmup_cases) != 0:
            raise ValueError("AOT candidate schedules require the 0+4K protocol")
        if int(args.aot_max_selections_per_case) < 3:
            raise ValueError(
                "AOT schedules need at least three selections per case for recovery"
            )
    elif candidate_mode != "explicit":
        raise ValueError("setup_candidate_mode must be explicit or aot")
    if candidate_mode != "aot" and any(
        spec.candidate_sampling == "structured512" for spec in specs
    ):
        raise ValueError("structured512 candidate sampling requires AOT mode")

    if args.weights is None or args.action_rbf_centers is None:
        raise ValueError(
            "Composable experiments require explicit --weights and "
            "--action-rbf-centers"
        )
    weights = np.asarray(_parse_values(args.weights, float), dtype=float)
    centers = np.asarray(_parse_values(args.action_rbf_centers, float), dtype=float)
    for label, values in (("weights", weights), ("RBF centers", centers)):
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Composable {label} must be finite and non-empty")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Composable {label} must be strictly increasing")
    if float(args.action_rbf_sigma) <= 0.0:
        raise ValueError("action_rbf_sigma must be positive")
    if any(spec.solve_kind == "ppo" for spec in specs) and not args.ppo_model.exists():
        raise FileNotFoundError(args.ppo_model)
    for name, value in (
        (
            "lin_ts_relative_sampling_scale",
            float(args.lin_ts_relative_sampling_scale),
        ),
        ("lin_ts_loss_scale_prior", float(args.lin_ts_loss_scale_prior)),
        ("rblspi_prior_precision", float(args.rblspi_prior_precision)),
        ("rblspi_noise_precision", float(args.rblspi_noise_precision)),
        ("rblspi_gram_ridge", float(args.rblspi_gram_ridge)),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return specs


def _make_encoder(args: argparse.Namespace) -> SolveStateEncoder:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
    )
    setup_encoder = SetupObsEncoder(
        parameter_spec,
        dict(DEFAULT_SETUP_PARAMS),
        tuple(parameter_spec.parameter_names),
    )
    return SolveStateEncoder(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        mode="setup_full",
        setup_obs_encoder=setup_encoder,
    )


def _make_frozen_ppo_runner(args: argparse.Namespace):
    ppo_args = SimpleNamespace(
        ppo_model=args.ppo_model,
        output_dir=args.output_dir,
        grid_n=int(args.grid_n),
        c_min=float(args.c_min),
        c_max=float(args.c_max),
        max_cycles=int(args.max_cycles),
        tol=float(args.tol),
        ppo_action_mode=str(args.ppo_action_mode),
        ppo_w_center=float(args.ppo_w_center),
        ppo_w_scale=float(args.ppo_w_scale),
        ppo_initial_observation_weight=float(
            args.ppo_initial_observation_weight
        ),
        ppo_force_default_first_action=bool(
            args.ppo_force_default_first_action
        ),
        ppo_default_first_weight=float(args.ppo_default_first_weight),
    )
    return make_exp44_ppo_runner(ppo_args)


def _make_controller(
    args: argparse.Namespace,
    candidate: SarsaCandidate,
    *,
    seed: int,
) -> tuple[BehaviorPolicySarsaController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    weights = _parse_values(args.weights, float)
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


def _make_shared_action_config(
    args: argparse.Namespace,
    *,
    trace_lambda: float,
) -> ExpectedSarsaLambdaConfig:
    weights = _parse_values(args.weights, float)
    return ExpectedSarsaLambdaConfig(
        weights=tuple(float(weight) for weight in weights),
        anchor_weight=float(weights[0]),
        alpha=0.001,
        td_decay_power=0.0,
        gamma=1.0,
        trace_lambda=float(trace_lambda),
        epsilon_start=float(args.epsilon_start),
        epsilon_final=float(args.epsilon_final),
        epsilon_decay_steps=float(args.epsilon_decay_steps),
        initial_q_sec=0.0,
        monte_carlo_alpha=0.0,
        adaptive_cycles=None,
        exploration_mode="uniform",
        action_rbf_sigma=float(args.action_rbf_sigma),
        action_basis_mode="compact_rbf",
        action_basis_centers=_parse_values(args.action_rbf_centers, float),
        td_algorithm="true_online_sarsa",
        force_default_first_action=True,
    )


def _make_lsvi_controller(
    args: argparse.Namespace,
    *,
    seed: int,
    refit_interval_episodes: int = 1,
) -> tuple[StagewiseLsviLcbController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    config = _make_shared_action_config(
        args,
        trace_lambda=0.8,
    )
    controller = StagewiseLsviLcbController(
        feature_dim=encoder.feature_dim,
        config=config,
        spec=StagewiseLsviLcbSpec(
            horizon=int(args.max_cycles),
            ridge=float(args.lsvi_ridge),
            uncertainty_beta=float(args.lsvi_beta),
            residual_floor_sec=float(args.lsvi_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            refit_interval_episodes=int(refit_interval_episodes),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_recursive_mc_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[RecursiveMonteCarloLcbController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    controller = RecursiveMonteCarloLcbController(
        feature_dim=encoder.feature_dim,
        config=_make_shared_action_config(args, trace_lambda=1.0),
        spec=RecursiveMonteCarloLcbSpec(
            ridge=float(args.recursive_mc_ridge),
            uncertainty_beta=float(args.recursive_mc_beta),
            residual_floor_sec=float(args.recursive_mc_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            episode_half_life=float(args.recursive_mc_episode_half_life),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_recursive_lstdq_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[RecursiveLstdqLcbController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    lcb_lower_bound = getattr(args, "recursive_lstdq_lcb_lower_bound_sec", 0.0)
    controller = RecursiveLstdqLcbController(
        feature_dim=encoder.feature_dim,
        config=_make_shared_action_config(
            args,
            trace_lambda=float(args.recursive_lstdq_lambda),
        ),
        spec=RecursiveLstdqLcbSpec(
            ridge=float(args.recursive_lstdq_ridge),
            uncertainty_beta=float(args.recursive_lstdq_beta),
            residual_floor_sec=float(args.recursive_lstdq_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            lcb_lower_bound_sec=(
                None if lcb_lower_bound is None else float(lcb_lower_bound)
            ),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_recursive_lstdq_v2_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[RecursiveLstdqV2LcbController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    lcb_lower_bound = getattr(args, "recursive_lstdq_lcb_lower_bound_sec", 0.0)
    controller = RecursiveLstdqV2LcbController(
        feature_dim=encoder.feature_dim,
        config=_make_shared_action_config(
            args,
            trace_lambda=float(args.recursive_lstdq_lambda),
        ),
        spec=RecursiveLstdqV2LcbSpec(
            ridge=float(args.recursive_lstdq_ridge),
            uncertainty_beta=float(args.recursive_lstdq_v2_beta),
            residual_floor_sec=float(args.recursive_lstdq_residual_floor_sec),
            lcb_lower_bound_sec=(
                None if lcb_lower_bound is None else float(lcb_lower_bound)
            ),
            coverage_ridge=float(args.recursive_lstdq_v2_coverage_ridge),
            residual_scale_window=int(
                args.recursive_lstdq_v2_residual_window
            ),
            residual_scale_min_samples=int(
                args.recursive_lstdq_v2_min_samples
            ),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_recursive_blstdq_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[RecursiveBlstdqController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    config = replace(
        _make_shared_action_config(args, trace_lambda=0.0),
        epsilon_start=0.0,
        epsilon_final=0.0,
    )
    controller = RecursiveBlstdqController(
        feature_dim=encoder.feature_dim,
        config=config,
        spec=RecursiveBlstdqSpec(
            prior_precision=float(args.rblspi_prior_precision),
            noise_precision=float(args.rblspi_noise_precision),
            gram_ridge=float(args.rblspi_gram_ridge),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_structured_model_based_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[StructuredModelBasedController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    controller = StructuredModelBasedController(
        feature_dim=encoder.feature_dim,
        config=_make_shared_action_config(args, trace_lambda=0.0),
        spec=StructuredModelBasedSpec(
            ridge=float(args.structured_model_ridge),
            minimum_samples=int(args.structured_model_min_samples),
            scale_window=int(args.structured_model_scale_window),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _make_recalibrated_lsvi_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[HierarchicalLsviLcbController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    controller = HierarchicalLsviLcbController(
        feature_dim=encoder.feature_dim,
        config=_make_shared_action_config(args, trace_lambda=0.8),
        spec=HierarchicalLsviLcbSpec(
            horizon=int(args.max_cycles),
            ridge=float(args.lsvi_ridge),
            uncertainty_beta=float(args.recalibrated_lsvi_beta),
            residual_floor_sec=float(args.lsvi_residual_floor_sec),
            refit_interval_episodes=int(args.lsvi_refit_interval_episodes),
            refit_sweeps=int(args.recalibrated_lsvi_refit_sweeps),
            residual_shrinkage_samples=float(
                args.recalibrated_lsvi_shrinkage_samples
            ),
        ),
        seed=int(seed),
    )
    return controller, encoder


def _controller_summary(controller: Any) -> Dict[str, Any]:
    summary = {
        "steps": int(controller.steps),
        "episodes": int(controller.episodes),
        "epsilon": float(controller.epsilon),
    }
    if hasattr(controller, "behavior_summary"):
        summary.update(controller.behavior_summary())
    elif hasattr(controller, "summary"):
        summary.update(controller.summary())
    return summary


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")


def _write_solve_screen_reproduction(
    args: argparse.Namespace,
    *,
    stream_hash: str,
) -> None:
    """Write an auditable command and short protocol README beside results."""

    output_default = args.output_dir.with_name(f"{args.output_dir.name}_reproduction")
    command = [
        sys.executable,
        "-u",
        "experiments/joint/solve_control/run_joint_online_sarsa_4k.py",
        "--output-dir",
        '"$OUTPUT_DIR"',
        "--study-mode",
        "solve_controller_screen",
        "--seed",
        str(args.seed),
        "--bandit-seed",
        str(args.bandit_seed),
        "--controller-seed",
        str(args.controller_seed),
        "--method-order-seed",
        str(args.method_order_seed),
        "--train-cases",
        str(args.train_cases),
        "--warmup-cases",
        str(args.warmup_cases),
        "--online-cases",
        str(args.online_cases),
        "--train-seed-groups",
        str(args.train_seed_groups),
        "--train-shuffle-seeds",
        str(args.train_shuffle_seeds),
        "--train-cases-per-seed",
        str(args.train_cases_per_seed),
        "--train-group-take",
        str(args.train_group_take),
        "--matrix-grid-n",
        str(args.grid_n),
        "--setup-param-resolution",
        str(args.setup_param_resolution),
        "--max-cycles",
        str(args.max_cycles),
        "--shared-action-profile",
        str(args.shared_action_profile),
        "--recursive-lstdq-v2-beta",
        str(args.recursive_lstdq_v2_beta),
        "--recalibrated-lsvi-beta",
        str(args.recalibrated_lsvi_beta),
        "--progress-every",
        str(args.progress_every),
    ]
    if bool(args.smoke):
        command.append("--smoke")
    quoted = " ".join(
        token if token == '"$OUTPUT_DIR"' else shlex.quote(token)
        for token in command
    )
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'OUTPUT_DIR="${{OUTPUT_DIR:-{output_default}}}"\n'
        f"{quoted}\n"
    )
    script_path = args.output_dir / "reproduce.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    readme = f"""# 60^3 Solve-Controller Screen

This directory contains the locked five-branch joint-online comparison.

- methods: {', '.join(SOLVE_CONTROLLER_SCREEN_METHODS)}
- stream SHA-256: `{stream_hash}`
- LSTDQ v2 beta: `{args.recursive_lstdq_v2_beta:g}`
- recalibrated LSVI beta: `{args.recalibrated_lsvi_beta:g}`
- action grid: `1.00:0.05:3.00`
- recovery and timing: active bounded-recovery joint protocol

Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting
this directory.  `config.json`, `stream_manifest.json`, trajectories, 1K
checkpoints, and final mutable states are written by the runner.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


def _empty_stream_summary() -> Dict[str, Any]:
    runtime_fields = (
        "setup_runtime",
        "native_solve_runtime",
        "native_total_runtime",
        "controller_runtime",
        "setup_bandit_overhead",
        "setup_bandit_select",
        "setup_bandit_loss_eval",
        "setup_bandit_update",
        "end_to_end_runtime",
    )
    return {
        "cases": 0,
        "failed_count": 0,
        "mean_runtime_sec": 0.0,
        "mean_runtime_with_overhead_sec": 0.0,
        "mean_setup_runtime_sec": 0.0,
        "mean_solve_runtime_sec": 0.0,
        "mean_solve_with_overhead_sec": 0.0,
        "mean_policy_overhead_sec": 0.0,
        "mean_feature_runtime_sec": 0.0,
        "mean_decision_runtime_sec": 0.0,
        "mean_update_runtime_sec": 0.0,
        "mean_iterations": 0.0,
        "totals_sec": {field: 0.0 for field in runtime_fields},
        "means_sec": {field: 0.0 for field in runtime_fields},
        "unique_setup_count": 0,
        "primary_failure_count": 0,
        "setup_fallback_count": 0,
        "recovered_failure_count": 0,
        "unrecovered_failure_count": 0,
        "bandit_update_count": 0,
        "controller_update_count": 0,
    }


def _comparison_windows(case_count: int) -> Dict[str, tuple[int, int]]:
    if int(case_count) == 2000:
        return {
            "all_2000": (0, 2000),
            "first_1000": (0, 1000),
            "last_1000": (1000, 2000),
            "last_500": (1500, 2000),
            "last_300": (1700, 2000),
        }
    if int(case_count) == 4000:
        return {
            "all_4000": (0, 4000),
            "first_2000": (0, 2000),
            "last_2000": (2000, 4000),
            "first_1000": (0, 1000),
            "last_1000": (3000, 4000),
            "last_500": (3500, 4000),
            "last_300": (3700, 4000),
        }
    return {f"all_{int(case_count)}": (0, int(case_count))}


def _warmup_bandit(
    args: argparse.Namespace,
    instances: Sequence[tuple[Dict[str, Any], np.ndarray]],
    *,
    stream_hash: str,
) -> tuple[Any, list[Dict[str, Any]], Dict[str, Any]]:
    state_path = args.output_dir / f"bandit_warmup_{len(instances)}.npz"
    records_path = args.output_dir / "warmup_trajectory.jsonl"
    if args.reuse_warmup:
        branch, _bandit_cfg = _build_bandit(
            seed=int(args.bandit_seed),
            tune_dim=7,
            tune7_variant="categorical",
        )
        metadata = branch.policy.model.load_mutable_state(state_path)
        if str(metadata.get("stream_hash")) != str(stream_hash):
            raise ValueError("Saved warmup state does not match the locked 4K stream")
        rows = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != len(instances):
            raise ValueError("Saved warmup trajectory length does not match")
        validate_expected_setup_action_count(branch)
        return branch, rows, dict(metadata["summary"])

    branch, bandit_cfg = _build_bandit(
        seed=int(args.bandit_seed),
        tune_dim=7,
        tune7_variant="categorical",
    )
    validate_expected_setup_action_count(branch)
    if not instances:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("", encoding="utf-8")
        summary = _empty_stream_summary()
        branch.policy.model.save_mutable_state(
            state_path,
            metadata={
                "stream_hash": str(stream_hash),
                "summary": summary,
            },
        )
        _write_json(
            args.output_dir / "warmup_progress.json",
            {"completed_instances": 0, "summary": summary},
        )
        return branch, [], summary

    previous_update = 0.0
    rows: list[Dict[str, Any]] = []
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as trajectory:
        for case_index, (mkw, context) in enumerate(instances):
            def solve_selected(params: Dict[str, Any]) -> Dict[str, Any]:
                native = solve_no_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                )
                return _as_feedback(native, include_controller=False)

            def solve_fallback(_params: Dict[str, Any]) -> Dict[str, Any]:
                native = solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                )
                return _as_feedback(native, include_controller=False)

            params, native, timing, fallback_used, update_sec = (
                run_bandit_step_test_final(
                    policy=branch.policy,
                    parameter_space=branch.parameter_space,
                    context=np.asarray(context, dtype=float),
                    solver_fn=solve_selected,
                    fallback_solver_fn=solve_fallback,
                    prev_update_est=float(previous_update),
                )
            )
            previous_update = float(update_sec)
            row = {
                "warmup_index": int(case_index),
                "mkw": dict(mkw),
                "context": np.asarray(context, dtype=float).tolist(),
                "params": dict(params),
                "arm_index": _policy_last_arm(branch.policy),
                "fallback_used": int(fallback_used),
                "bandit_timing": dict(timing),
                "outcome": _report_online_outcome(native, bandit_timing=timing),
            }
            rows.append(row)
            _write_json_line(trajectory, row)
            done = case_index + 1
            if done % max(1, int(args.progress_every)) == 0 or done == len(instances):
                trajectory.flush()
                _write_json(
                    args.output_dir / "warmup_progress.json",
                    {
                        "completed_instances": int(done),
                        "summary": _method_stream_summary(rows),
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": "joint_online_4k_warmup",
                            "done": int(done),
                            "total": int(len(instances)),
                        }
                    ),
                    flush=True,
                )

    summary = _method_stream_summary(rows)
    branch.policy.model.save_mutable_state(
        state_path,
        metadata={
            "stream_hash": str(stream_hash),
            "summary": summary,
        },
    )
    return branch, rows, summary


def _method_solver(
    method: str,
    *,
    args: argparse.Namespace,
    mkw: Dict[str, Any],
    case_progress: float,
    controllers: Dict[str, Any],
    encoders: Dict[str, SolveStateEncoder],
    ppo_runner: Any,
    composable_specs: Dict[str, ComposableMethodSpec] | None = None,
):
    spec = None if composable_specs is None else composable_specs.get(method)
    solve_kind = None if spec is None else spec.solve_kind
    if (spec is None and method == "bandit_default") or solve_kind == "default":
        def solve_default(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_no_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solver_tol=float(args.tol),
                solver_max_iter=int(args.max_cycles),
                augment_params=augment_setup_params,
            )
            return _as_feedback(native, include_controller=False)

        return solve_default
    if (spec is None and method == "bandit_fixed_w1.6") or solve_kind == "fixed":
        fixed_weight = 1.6 if spec is None else float(spec.fixed_weight)

        def solve_fixed(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_fixed_w_case(
                params=dict(params),
                mkw=dict(mkw),
                w=float(fixed_weight),
                sweeps_down=1,
                sweeps_up=1,
                solve_tol=float(args.tol),
                solve_max_cycles=int(args.max_cycles),
            )
            return _as_feedback(native, include_controller=False)

        return solve_fixed
    if (spec is None and method == "bandit_ppo") or solve_kind == "ppo":
        def solve_ppo(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_setup_aware_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solve_policy=ppo_runner,
                augment_params=augment_setup_params,
                classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                    residual_norm=float(residual_norm),
                    iterations=int(iterations),
                    solve_tol=float(args.tol),
                    solve_max_cycles=int(args.max_cycles),
                ),
                solve_max_cycles=int(args.max_cycles),
                case_progress=float(case_progress),
            )
            return _as_feedback(native, include_controller=True)

        return solve_ppo
    if method in controllers:
        def solve_sarsa(params: Dict[str, Any]) -> Dict[str, Any]:
            native = run_td_episode(
                mkw=dict(mkw),
                params=dict(params),
                controller=controllers[method],
                encoder=encoders[method],
                solve_tol=float(args.tol),
                solve_max_cycles=int(args.max_cycles),
                learn=True,
                explore=True,
                record_action_metadata=True,
                fallback_attempt=lambda: solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                ),
            )
            return _as_feedback(native, include_controller=True)

        return solve_sarsa
    raise ValueError(f"Unsupported method: {method}")


def _run_default_setup_method(
    *,
    spec: ComposableMethodSpec,
    solver_fn: Any,
    args: argparse.Namespace,
    mkw: Dict[str, Any],
    controller_methods: Sequence[str],
) -> Dict[str, Any]:
    """Execute one default-setup branch under the active recovery protocol."""

    if spec.solve_kind == "default":
        native = solve_default_baseline_case(
            mkw=dict(mkw),
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            augment_params=augment_setup_params,
        )
        feedback = _as_feedback(native, include_controller=False)
    elif spec.name in controller_methods:
        # Online controllers own their episode transaction and fallback so that
        # unrecovered failures can roll back the controller state atomically.
        feedback = solver_fn(dict(DEFAULT_SETUP_PARAMS))
    else:
        recovery = run_with_default_fallback(
            lambda: solver_fn(dict(DEFAULT_SETUP_PARAMS)),
            lambda: solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS),
                mkw=dict(mkw),
                solver_tol=float(args.tol),
                solver_max_iter=int(args.max_cycles),
                augment_params=augment_setup_params,
            ),
            primary_is_default=False,
        )
        native = recovery.to_result()
        native.update(
            {
                "recovery_protocol_applied": True,
                "bandit_update_committed": False,
                "controller_update_committed": False,
            }
        )
        feedback = _as_feedback(
            native,
            include_controller=spec.solve_kind == "ppo",
        )
    reported = _report_online_outcome(feedback, bandit_timing={})
    reported["bandit_update_committed"] = False
    return reported


def _window_result(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    seed: int,
) -> Dict[str, Any]:
    available_references = {
        "vs_bandit_default": "bandit_default",
        "vs_fixed_w1.6": "bandit_fixed_w1.6",
        "vs_ppo": "bandit_ppo",
    }
    references = {
        label: method
        for label, method in available_references.items()
        if method in records
    }
    return {
        "methods": {
            method: _method_stream_summary(rows)
            for method, rows in records.items()
        },
        "comparisons": {
            label: {
                method: _method_comparison(
                    rows,
                    records[reference],
                    seed=int(seed + reference_index * 100_000 + method_index * 101),
                )
                for method_index, (method, rows) in enumerate(records.items())
                if method != reference
            }
            for reference_index, (label, reference) in enumerate(references.items())
        },
    }


def _action_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actions: list[float] = []
    explored: list[bool] = []
    uncertainties: list[float] = []
    selected_q_values: list[float] = []
    selected_scores: list[float] = []
    calibration_ratios: list[float] = []
    for row in rows:
        outcome = row["outcome"]
        actions.extend(float(value) for value in outcome.get("cycle_actions", []))
        explored.extend(bool(value) for value in outcome.get("cycle_explored", []))
        uncertainties.extend(
            float(value)
            for value in outcome.get("cycle_selected_uncertainties", [])
        )
        selected_q_values.extend(
            float(value)
            for value in outcome.get("cycle_selected_q_values", [])
            if np.isfinite(float(value))
        )
        selected_scores.extend(
            float(value)
            for value in outcome.get("cycle_selection_scores", [])
            if np.isfinite(float(value))
        )
        errors = outcome.get("postfit_td_errors") or outcome.get("td_errors", [])
        widths = outcome.get("cycle_selected_uncertainties", [])
        for error, width in zip(errors, widths):
            if np.isfinite(float(error)) and float(width) > 0.0:
                calibration_ratios.append(abs(float(error)) / float(width))
    ratios = np.asarray(calibration_ratios, dtype=float)
    return {
        "decisions": int(len(actions)),
        "mean_weight": float(np.mean(actions)) if actions else float("nan"),
        "explored_rate": float(np.mean(explored)) if explored else 0.0,
        "mean_selected_uncertainty_sec": (
            float(np.mean(uncertainties)) if uncertainties else 0.0
        ),
        "mean_selected_q_sec": (
            float(np.mean(selected_q_values)) if selected_q_values else 0.0
        ),
        "lower_bound_saturation_count": int(
            sum(np.isclose(value, 0.0, atol=1.0e-12, rtol=0.0) for value in selected_scores)
        ),
        "lower_bound_saturation_rate": (
            float(
                np.mean(
                    np.isclose(
                        np.asarray(selected_scores, dtype=float),
                        0.0,
                        atol=1.0e-12,
                        rtol=0.0,
                    )
                )
            )
            if selected_scores
            else 0.0
        ),
        "confidence_calibration": {
            "paired_updates": int(ratios.size),
            "median_abs_error_over_uncertainty": (
                float(np.median(ratios)) if ratios.size else 0.0
            ),
            "p90_abs_error_over_uncertainty": (
                float(np.quantile(ratios, 0.9)) if ratios.size else 0.0
            ),
            "coverage_at_1x": (
                float(np.mean(ratios <= 1.0)) if ratios.size else 0.0
            ),
            "coverage_at_2x": (
                float(np.mean(ratios <= 2.0)) if ratios.size else 0.0
            ),
            "coverage_at_4x": (
                float(np.mean(ratios <= 4.0)) if ratios.size else 0.0
            ),
        },
    }


def _write_summary_csv(
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
) -> None:
    fields = (
        "method",
        "family",
        "cases",
        "mean_setup_runtime_sec",
        "mean_native_solve_runtime_sec",
        "mean_native_total_runtime_sec",
        "mean_controller_runtime_sec",
        "mean_setup_bandit_overhead_sec",
        "mean_end_to_end_runtime_sec",
        "failures",
        "mean_iterations",
        "setup_fallbacks",
        "primary_failures",
        "recovered_failures",
        "unrecovered_failures",
        "bandit_updates",
        "controller_updates",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in records.items():
            summary = _method_stream_summary(rows)
            means = summary["means_sec"]
            writer.writerow(
                {
                    "method": method,
                    "family": family_by_method[method],
                    "cases": len(rows),
                    "mean_setup_runtime_sec": means["setup_runtime"],
                    "mean_native_solve_runtime_sec": means["native_solve_runtime"],
                    "mean_native_total_runtime_sec": means["native_total_runtime"],
                    "mean_controller_runtime_sec": means["controller_runtime"],
                    "mean_setup_bandit_overhead_sec": means["setup_bandit_overhead"],
                    "mean_end_to_end_runtime_sec": means["end_to_end_runtime"],
                    "failures": summary["failed_count"],
                    "mean_iterations": summary["mean_iterations"],
                    "setup_fallbacks": summary["setup_fallback_count"],
                    "primary_failures": summary["primary_failure_count"],
                    "recovered_failures": summary["recovered_failure_count"],
                    "unrecovered_failures": summary["unrecovered_failure_count"],
                    "bandit_updates": summary["bandit_update_count"],
                    "controller_updates": summary["controller_update_count"],
                }
            )


def _write_solve_screen_report(
    path: Path,
    *,
    window_results: Dict[str, Any],
    window_actions: Dict[str, Dict[str, Any]],
) -> None:
    """Render the locked screening metrics without requiring plotting tools."""

    preferred_windows = ("all_4000", "first_1000", "last_1000", "last_500")
    has_fixed_reference = any(
        "vs_fixed_w1.6" in window.get("comparisons", {})
        for window in window_results.values()
    )
    comparison_note = (
        "Improvement and paired 95% intervals are relative to Online "
        "LinUCB + fixed `w=1.6`."
        if has_fixed_reference
        else "No fixed-`w=1.6` reference was included, so paired improvement "
        "columns are reported as n/a."
    )
    lines = [
        "# Solve-Controller Screening Report",
        "",
        "All runtimes are per-instance means in milliseconds.",
        comparison_note,
        "",
    ]
    report_windows = tuple(
        name for name in preferred_windows if name in window_results
    )
    if not report_windows:
        report_windows = tuple(window_results)
    for window_name in report_windows:
        window = window_results[window_name]
        comparisons = window.get("comparisons", {}).get("vs_fixed_w1.6", {})
        lines.extend(
            [
                f"## {window_name}",
                "",
                "| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, summary in window["methods"].items():
            means = summary["means_sec"]
            if method == "bandit_fixed_w1.6":
                improvement_text = "baseline"
                same_setup_text = "1.000"
            elif method in comparisons:
                comparison = comparisons[method]
                metric = comparison["end_to_end_runtime"]
                interval = metric["candidate_improvement_95pct"]
                improvement_text = (
                    f"{metric['candidate_improvement_pct']:.2f}% "
                    f"[{interval[0]:.2f}, {interval[1]:.2f}]"
                )
                same_setup_text = (
                    f"{float(comparison['same_setup_rate']):.3f}"
                )
            else:
                improvement_text = "n/a"
                same_setup_text = "n/a"
            lines.append(
                "| {method} | {setup:.3f} | {solve:.3f} | {native:.3f} | "
                "{controller:.3f} | {bandit:.3f} | {e2e:.3f} | {cycles:.2f} | "
                "{improvement} | {primary}/{recovered}/{unrecovered} | {same} |".format(
                    method=method,
                    setup=1000.0 * float(means["setup_runtime"]),
                    solve=1000.0 * float(means["native_solve_runtime"]),
                    native=1000.0 * float(means["native_total_runtime"]),
                    controller=1000.0 * float(means["controller_runtime"]),
                    bandit=1000.0 * float(means["setup_bandit_overhead"]),
                    e2e=1000.0 * float(means["end_to_end_runtime"]),
                    cycles=float(summary["mean_iterations"]),
                    improvement=improvement_text,
                    primary=int(summary["primary_failure_count"]),
                    recovered=int(summary["recovered_failure_count"]),
                    unrecovered=int(summary["unrecovered_failure_count"]),
                    same=same_setup_text,
                )
            )
        action_rows = window_actions.get(window_name, {})
        if action_rows:
            lines.extend(
                [
                    "",
                    "Controller diagnostics:",
                    "",
                    "| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for method, action in action_rows.items():
                calibration = action["confidence_calibration"]
                is_model_based = method == STRUCTURED_MODEL_BASED_METHOD
                uncertainty_text = (
                    "n/a"
                    if is_model_based
                    else f"{1000.0 * float(action['mean_selected_uncertainty_sec']):.3f}"
                )
                saturation_text = (
                    "n/a"
                    if is_model_based
                    else f"{float(action['lower_bound_saturation_rate']):.3f}"
                )
                calibration_text = (
                    "n/a"
                    if is_model_based
                    else (
                        f"{float(calibration['coverage_at_1x']):.3f}/"
                        f"{float(calibration['coverage_at_2x']):.3f}/"
                        f"{float(calibration['coverage_at_4x']):.3f}"
                    )
                )
                lines.append(
                    "| {method} | {decisions} | {weight:.3f} | {explored:.3f} | "
                    "{uncertainty} | {saturation} | {calibration} |".format(
                        method=method,
                        decisions=int(action["decisions"]),
                        weight=float(action["mean_weight"]),
                        explored=float(action["explored_rate"]),
                        uncertainty=uncertainty_text,
                        saturation=saturation_text,
                        calibration=calibration_text,
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    composable_specs_tuple: tuple[ComposableMethodSpec, ...] = ()
    setup_configuration_spaces: Dict[str, SetupConfigurationSpace] = {}
    if args.study_mode == "composable":
        composable_specs_tuple = _validate_composable_protocol(args)
        setup_configuration_spaces = _setup_configuration_spaces_from_args(args)
    if args.study_mode in {
        "lsvi_lcb",
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
        "solve_controller_screen",
    }:
        _validate_shared_lcb_protocol(args)
    if (
        args.study_mode in {"solve_controller_screen", "composable"}
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing to overwrite experiment output: {args.output_dir}"
        )
    if int(args.warmup_cases) + int(args.online_cases) != int(args.train_cases):
        raise ValueError("train_cases must equal warmup_cases + online_cases")
    partition = (int(args.warmup_cases), int(args.online_cases))
    if (
        args.study_mode != "composable"
        and not bool(args.smoke)
        and partition not in {(2000, 2000), (0, 4000)}
    ):
        raise ValueError(
            "This locked protocol requires either 2000 warmup + 2000 online "
            "cases or 0 warmup + 4000 joint-online cases"
        )
    if bool(args.reuse_warmup) and int(args.warmup_cases) == 0:
        raise ValueError("A joint-from-scratch run cannot reuse a warmup state")
    if args.study_mode in {
        "sarsa",
        "recursive_lcb_ppo",
    } and not args.ppo_model.exists():
        raise FileNotFoundError(args.ppo_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    checkpoints_dir = args.output_dir / "checkpoints"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _configure_paired_environment(args)
    full_stream, stream_manifest = _build_paired_instance_stream(args)
    if len(full_stream) != int(args.train_cases):
        raise ValueError("Generated stream length does not match train_cases")
    if (
        args.study_mode != "composable"
        and not bool(args.smoke)
        and len(full_stream) != 4000
    ):
        raise ValueError("Locked stream must contain exactly 4000 instances")
    expected_stream_hash = str(
        getattr(args, "expected_stream_hash", "") or ""
    ).strip()
    if (
        expected_stream_hash
        and str(stream_manifest["sha256"]) != expected_stream_hash
    ):
        raise ValueError(
            "Generated stream does not match --expected-stream-hash: "
            f"{stream_manifest['sha256']} != {expected_stream_hash}"
        )
    if args.study_mode == "solve_controller_screen" and not bool(args.smoke):
        if (int(args.warmup_cases), int(args.online_cases)) != (0, 4000):
            raise ValueError("The final solve screen must learn jointly for all 4K cases")
        if int(args.grid_n) != 60 or int(args.setup_param_resolution) != 20:
            raise ValueError("The final solve screen requires 60^3 and setup resolution 20")
        expected_hash = (
            "156e6fdbbed6733d98e2c5f7e550e5d230c45217d0833ac64567563b5434459b"
        )
        if str(stream_manifest["sha256"]) != expected_hash:
            raise ValueError("The final solve screen does not match the canonical stream")
    warmup_instances = full_stream[: int(args.warmup_cases)]
    online_instances = full_stream[int(args.warmup_cases) :]

    composable_specs: Dict[str, ComposableMethodSpec] = {}
    if args.study_mode == "composable":
        candidates = ()
        methods = tuple(spec.name for spec in composable_specs_tuple)
        composable_specs = {spec.name: spec for spec in composable_specs_tuple}
        family_by_method = {
            spec.name: spec.family for spec in composable_specs_tuple
        }
    elif args.study_mode == "lsvi_lcb":
        candidates = ()
        methods = LSVI_METHODS
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            LSVI_METHOD: "stagewise_lsvi_lcb",
        }
    elif args.study_mode == "solve_controller_screen":
        candidates = ()
        methods = SOLVE_CONTROLLER_SCREEN_METHODS
        family_by_method = {
            "bandit_fixed_w1.6": "fixed_w1.6",
            RECURSIVE_LSTDQ_METHOD: "recursive_lstdq_lcb",
            RECURSIVE_LSTDQ_V2_METHOD: "recursive_lstdq_v2_lcb",
            STRUCTURED_MODEL_BASED_METHOD: "structured_model_based",
            RECALIBRATED_LSVI_METHOD: "recalibrated_lsvi_lcb",
        }
    elif args.study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        candidates = ()
        if args.study_mode == "recursive_lcb_suite":
            methods = RECURSIVE_LCB_METHODS
        elif args.study_mode == "recursive_lcb_ppo":
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
    else:
        candidates = candidate_grid(
            alphas=_parse_values(args.alphas, float),
            trace_lambdas=_parse_values(args.trace_lambdas, float),
        )
        methods = (*REFERENCE_METHODS, *[candidate.name for candidate in candidates])
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            "bandit_ppo": "ppo",
            **{candidate.name: candidate.family for candidate in candidates},
        }
    if (
        args.study_mode == "solve_controller_screen"
        and bool(args.include_default_setup_baseline)
    ):
        raise ValueError("solve_controller_screen has a locked five-method roster")
    if (
        args.study_mode == "composable"
        and bool(args.include_default_setup_baseline)
    ):
        raise ValueError(
            "Composable experiments must express the default baseline as a "
            "normal --method instead of using the legacy baseline flag"
        )
    if bool(args.include_default_setup_baseline):
        methods = (DEFAULT_SETUP_METHOD, *methods)
        family_by_method = {
            DEFAULT_SETUP_METHOD: "default_setup",
            **family_by_method,
        }
    if args.study_mode == "composable":
        bandit_methods = tuple(
            spec.name
            for spec in composable_specs_tuple
            if spec.setup_kind in SETUP_BANDIT_KINDS
        )
    else:
        bandit_methods = tuple(
            method for method in methods if method != DEFAULT_SETUP_METHOD
        )
    resolved_setup_spaces = {
        name: configuration_space.as_dict()
        for name, configuration_space in setup_configuration_spaces.items()
    }
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "purpose": (
            "4K joint-online setup-bandit and solve-controller learning from scratch"
            if int(args.warmup_cases) == 0
            else "2K setup-bandit warmup + 2K persistent joint-online comparison"
        ),
        "stream": stream_manifest,
        "stream_partition": {
            "warmup": [0, int(args.warmup_cases)],
            "online": [int(args.warmup_cases), int(args.train_cases)],
        },
        "methods": list(methods),
        "families": family_by_method,
        "method_labels": {
            spec.name: spec.label for spec in composable_specs_tuple
        },
        "method_specs": [asdict(spec) for spec in composable_specs_tuple],
        "candidates": [asdict(candidate) | {"name": candidate.name} for candidate in candidates],
        "setup_bandit": {
            "seed": int(args.bandit_seed),
            "matrix_grid_n": int(args.grid_n),
            "problem": str(getattr(args, "problem", "scalar_anisotropic_diffusion")),
            "grid_shape": list(stream_manifest.get("grid", [int(args.grid_n)] * 3)),
            "setup_param_resolution": int(args.setup_param_resolution),
            "warmup_instances": int(args.warmup_cases),
            "joint_from_scratch": int(args.warmup_cases) == 0,
            "common_snapshot_mutable_state_cloned": not bool(
                setup_configuration_spaces
            ),
            "immutable_action_catalog_and_g_actions_shared": not bool(
                setup_configuration_spaces
            ),
            "per_branch_parameter_space": bool(setup_configuration_spaces),
            "candidate_mode": str(args.setup_candidate_mode),
            "aot_schedule": (
                {
                    "max_selections_per_case": int(
                        args.aot_max_selections_per_case
                    ),
                    "scheduled_rounds_per_branch": int(args.online_cases)
                    * int(args.aot_max_selections_per_case),
                    "pairing": "fixed-width block per online instance",
                    "payload": "candidate_ids_only",
                    "action_features": "shared_factorized_ram_cache",
                }
                if str(args.setup_candidate_mode) == "aot"
                else None
            ),
            "configuration_spaces": resolved_setup_spaces,
            "method_configuration_spaces": {
                spec.name: spec.setup_space
                for spec in composable_specs_tuple
                if spec.setup_kind in SETUP_BANDIT_KINDS
            },
            "frozen_after_warmup": False,
            "independent_updates_per_method": True,
            "fallback_runtime_included_in_feedback": True,
            "linear_ts_v2": {
                "posterior": "Gaussian linear posterior",
                "precision_factor": "rank-one Cholesky updates",
                "relative_sampling_scale": float(
                    args.lin_ts_relative_sampling_scale
                ),
                "loss_scale_prior_sec": float(args.lin_ts_loss_scale_prior),
                "runtime_scale_reference": "first successful observation",
                "posterior_rng_separate_from_candidate_rng": True,
            },
            "failure_head": {
                "representation": "shared_context_action_features",
                "ridge": 1.0,
                "beta": 2.0,
            },
            "setup_construction_reselection": {
                "max_learned_attempts_including_first": 3,
                "temporary_exact_arm_exclusion": True,
                "ranking": ["failure_ucb", "runtime_lcb"],
                "runtime_target": "realized_suffix_cost",
            },
            "solve_failure_reselection": False,
            "default_fallback_attempts": 1,
            "online_branches": list(bandit_methods),
        },
        "sarsa": {
            "algorithm": "true-online SARSA(lambda)",
            "update_frequency": "one update per AMG cycle plus terminal update",
            "state": "setup_full",
            "actions": list(_parse_values(args.weights, float)),
            "behavior_modes": list(BEHAVIOR_MODES),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "uncertainty": {
                "beta": float(args.uncertainty_beta),
                "ridge": float(args.uncertainty_ridge),
                "td_floor_sec": float(args.uncertainty_td_floor_sec),
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        },
        "ppo": {
            "model": str(args.ppo_model),
            "frozen": True,
            "action_mode": str(args.ppo_action_mode),
            "w_center": float(args.ppo_w_center),
            "w_scale": float(args.ppo_w_scale),
            "initial_observation_weight": float(
                args.ppo_initial_observation_weight
            ),
            "prepared_solver_default_forced_once": bool(
                args.ppo_force_default_first_action
            ),
        },
        "randomized_method_order_per_instance": True,
        "method_order_seed": int(args.method_order_seed),
    }
    if args.study_mode == "composable":
        protocol.pop("sarsa")
        if not any(
            spec.solve_kind == "ppo" for spec in composable_specs_tuple
        ):
            protocol.pop("ppo")
        shared_protocol = {
            "state": "setup_full",
            "action_profile": "explicit",
            "actions": list(_parse_values(args.weights, float)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(
                    _parse_values(args.action_rbf_centers, float)
                ),
                "sigma": float(args.action_rbf_sigma),
            },
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
                "kind": "uniform epsilon-greedy floor",
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        }
        solve_kinds = {spec.solve_kind for spec in composable_specs_tuple}
        if "recursive_lstdq_v1" in solve_kinds:
            protocol["recursive_lstdq_lcb"] = shared_protocol | {
                "version": "v1",
                "target": "on-policy LSTDQ(lambda) Bellman equation",
                "ridge": float(args.recursive_lstdq_ridge),
                "trace_lambda": float(args.recursive_lstdq_lambda),
                "uncertainty_beta": float(args.recursive_lstdq_beta),
                "uncertainty": "unclipped post-fit sandwich covariance",
                "lcb_lower_bound_sec": float(
                    args.recursive_lstdq_lcb_lower_bound_sec
                ),
            }
        if "recursive_lstdq_v2" in solve_kinds:
            protocol["recursive_lstdq_v2_lcb"] = shared_protocol | {
                "version": "v2",
                "target": "same recursive LSTDQ mean as v1",
                "ridge": float(args.recursive_lstdq_ridge),
                "trace_lambda": float(args.recursive_lstdq_lambda),
                "uncertainty_beta": float(args.recursive_lstdq_v2_beta),
                "uncertainty": "rolling-MAD-scaled feature coverage",
                "coverage_ridge": float(
                    args.recursive_lstdq_v2_coverage_ridge
                ),
                "residual_window": int(
                    args.recursive_lstdq_v2_residual_window
                ),
                "minimum_scale_samples": int(
                    args.recursive_lstdq_v2_min_samples
                ),
                "residual_floor_sec": float(
                    args.recursive_lstdq_residual_floor_sec
                ),
                "lcb_lower_bound_sec": float(
                    args.recursive_lstdq_lcb_lower_bound_sec
                ),
            }
        if "rblspi" in solve_kinds:
            protocol["recursive_blstdq_rblspi"] = shared_protocol | {
                "target": "off-policy BLSTDQ empirical Bellman equation",
                "posterior": (
                    "S^-1=alpha*I+beta*A.T*C^-1*A; "
                    "m=beta*S*A.T*C^-1*b"
                ),
                "exploration": "one sampled Q-function per solve episode",
                "epsilon": {"kind": "none"},
                "trace_lambda": 0.0,
                "prior_precision": float(args.rblspi_prior_precision),
                "noise_precision": float(args.rblspi_noise_precision),
                "gram_ridge": float(args.rblspi_gram_ridge),
                "history_storage": "none; recursive A/C/b statistics only",
            }
        if "recursive_mc" in solve_kinds:
            protocol["recursive_mc_lcb"] = shared_protocol | {
                "target": "undiscounted episodic cost-to-go",
                "ridge": float(args.recursive_mc_ridge),
                "uncertainty_beta": float(args.recursive_mc_beta),
                "residual_floor_sec": float(
                    args.recursive_mc_residual_floor_sec
                ),
                "episode_half_life": float(
                    args.recursive_mc_episode_half_life
                ),
                "uncertainty": "post-fit episode-cluster sandwich covariance",
            }
        if "stagewise_lsvi" in solve_kinds:
            protocol["stagewise_lsvi_lcb"] = shared_protocol | {
                "horizon": int(args.max_cycles),
                "ridge": float(args.lsvi_ridge),
                "uncertainty_beta": float(args.lsvi_beta),
                "residual_floor_sec": float(args.lsvi_residual_floor_sec),
                "refit_interval_episodes": int(
                    args.lsvi_refit_interval_episodes
                ),
            }
        if "structured_model_based" in solve_kinds:
            protocol["structured_model_based"] = shared_protocol | {
                "target": [
                    "native_cycle_cost",
                    "signed_log_residual_progress",
                ],
                "estimator": "shared recursive least squares",
                "planning": "receding time per log-residual reduction",
                "recovery_cost_in_physical_model": False,
                "ridge": float(args.structured_model_ridge),
                "minimum_samples": int(args.structured_model_min_samples),
                "scale_window": int(args.structured_model_scale_window),
            }
        if "recalibrated_lsvi" in solve_kinds:
            protocol["recalibrated_lsvi_lcb"] = shared_protocol | {
                "target": "stagewise optimistic Bellman backup",
                "horizon": int(args.max_cycles),
                "ridge": float(args.lsvi_ridge),
                "uncertainty_beta": float(args.recalibrated_lsvi_beta),
                "residual_floor_sec": float(args.lsvi_residual_floor_sec),
                "batch_size_episodes": int(
                    args.lsvi_refit_interval_episodes
                ),
                "backward_shared_sweeps": int(
                    args.recalibrated_lsvi_refit_sweeps
                ),
                "residual_shrinkage_samples": float(
                    args.recalibrated_lsvi_shrinkage_samples
                ),
                "value_cap": "maximum observed realized return",
            }
    elif args.study_mode == "lsvi_lcb":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        protocol["stagewise_lsvi_lcb"] = {
            "state": "setup_full",
            "action_profile": str(args.shared_action_profile),
            "actions": list(_parse_values(args.weights, float)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(_parse_values(args.action_rbf_centers, float)),
                "sigma": float(args.action_rbf_sigma),
            },
            "horizon": int(args.max_cycles),
            "ridge": float(args.lsvi_ridge),
            "uncertainty_beta": float(args.lsvi_beta),
            "residual_floor_sec": float(args.lsvi_residual_floor_sec),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "update": "backward refit after every solve",
            "frozen": False,
        }
    elif args.study_mode == "solve_controller_screen":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        shared_protocol = {
            "state": "setup_full",
            "action_profile": str(args.shared_action_profile),
            "actions": list(_parse_values(args.weights, float)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(_parse_values(args.action_rbf_centers, float)),
                "sigma": float(args.action_rbf_sigma),
            },
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
                "kind": "uniform epsilon-greedy floor",
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        }
        protocol["recursive_lstdq_lcb"] = shared_protocol | {
            "version": "v1",
            "target": "on-policy LSTDQ(lambda) Bellman equation",
            "ridge": float(args.recursive_lstdq_ridge),
            "trace_lambda": float(args.recursive_lstdq_lambda),
            "uncertainty_beta": float(args.recursive_lstdq_beta),
            "uncertainty": "unclipped post-fit sandwich covariance",
            "lcb_lower_bound_sec": float(
                args.recursive_lstdq_lcb_lower_bound_sec
            ),
        }
        protocol["recursive_lstdq_v2_lcb"] = shared_protocol | {
            "version": "v2",
            "target": "same recursive LSTDQ mean as v1",
            "ridge": float(args.recursive_lstdq_ridge),
            "trace_lambda": float(args.recursive_lstdq_lambda),
            "uncertainty_beta": float(args.recursive_lstdq_v2_beta),
            "uncertainty": "rolling-MAD-scaled feature coverage",
            "coverage_ridge": float(args.recursive_lstdq_v2_coverage_ridge),
            "residual_window": int(args.recursive_lstdq_v2_residual_window),
            "minimum_scale_samples": int(
                args.recursive_lstdq_v2_min_samples
            ),
            "residual_floor_sec": float(
                args.recursive_lstdq_residual_floor_sec
            ),
            "lcb_lower_bound_sec": float(
                args.recursive_lstdq_lcb_lower_bound_sec
            ),
        }
        protocol["structured_model_based"] = shared_protocol | {
            "target": ["native_cycle_cost", "signed_log_residual_progress"],
            "estimator": "shared recursive least squares",
            "planning": "receding time per log-residual reduction",
            "recovery_cost_in_physical_model": False,
            "ridge": float(args.structured_model_ridge),
            "minimum_samples": int(args.structured_model_min_samples),
            "scale_window": int(args.structured_model_scale_window),
        }
        protocol["recalibrated_lsvi_lcb"] = shared_protocol | {
            "target": "stagewise optimistic Bellman backup",
            "horizon": int(args.max_cycles),
            "ridge": float(args.lsvi_ridge),
            "uncertainty_beta": float(args.recalibrated_lsvi_beta),
            "residual_floor_sec": float(args.lsvi_residual_floor_sec),
            "batch_size_episodes": int(args.lsvi_refit_interval_episodes),
            "backward_shared_sweeps": int(
                args.recalibrated_lsvi_refit_sweeps
            ),
            "residual_shrinkage_samples": float(
                args.recalibrated_lsvi_shrinkage_samples
            ),
            "value_cap": "maximum observed realized return",
        }
    elif args.study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        protocol.pop("sarsa")
        if args.study_mode != "recursive_lcb_ppo":
            protocol.pop("ppo")
        shared_protocol = {
            "state": "setup_full",
            "action_profile": str(args.shared_action_profile),
            "actions": list(_parse_values(args.weights, float)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(_parse_values(args.action_rbf_centers, float)),
                "sigma": float(args.action_rbf_sigma),
            },
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
                "kind": "uniform epsilon-greedy floor",
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        }
        if RECURSIVE_MC_METHOD in methods:
            protocol["recursive_mc_lcb"] = shared_protocol | {
                "target": "undiscounted episodic cost-to-go",
                "estimator": "episode-forgetting recursive least squares",
                "ridge": float(args.recursive_mc_ridge),
                "uncertainty_beta": float(args.recursive_mc_beta),
                "residual_floor_sec": float(
                    args.recursive_mc_residual_floor_sec
                ),
                "episode_half_life": float(
                    args.recursive_mc_episode_half_life
                ),
                "uncertainty": "post-fit episode-cluster sandwich covariance",
                "history_storage": "none after episode update",
            }
        if RECURSIVE_LSTDQ_METHOD in methods:
            protocol["recursive_lstdq_lcb"] = shared_protocol | {
                "target": "on-policy LSTDQ(lambda) Bellman equation",
                "estimator": "recursive LSTD with Sherman-Morrison inverse",
                "ridge": float(args.recursive_lstdq_ridge),
                "trace_lambda": float(args.recursive_lstdq_lambda),
                "uncertainty_beta": float(args.recursive_lstdq_beta),
                "lcb_lower_bound_sec": float(
                    args.recursive_lstdq_lcb_lower_bound_sec
                ),
                "uncertainty": "unclipped post-fit sandwich covariance",
                "history_storage": "none",
            }
        if BATCHED_LSVI_METHOD in methods:
            protocol["batched_lsvi_lcb"] = shared_protocol | {
                "target": "stagewise optimistic Bellman backup",
                "horizon": int(args.max_cycles),
                "ridge": float(args.lsvi_ridge),
                "uncertainty_beta": float(args.lsvi_beta),
                "residual_floor_sec": float(args.lsvi_residual_floor_sec),
                "batch_size_episodes": int(args.lsvi_refit_interval_episodes),
                "policy_updates": (
                    int(args.online_cases)
                    // int(args.lsvi_refit_interval_episodes)
                ),
                "update": "exact all-history backward refit at fixed batch boundaries",
            }
    _write_json(args.output_dir / "config.json", protocol)
    _write_json(args.output_dir / "stream_manifest.json", stream_manifest)
    if args.study_mode == "solve_controller_screen":
        _write_solve_screen_reproduction(
            args, stream_hash=str(stream_manifest["sha256"])
        )

    warmup_trajectory_artifact: Any = str(
        args.output_dir / "warmup_trajectory.jsonl"
    )
    if setup_configuration_spaces:
        if warmup_instances or bool(args.reuse_warmup):
            raise ValueError(
                "Per-branch setup configuration spaces currently require the "
                "joint-from-scratch 0+4K protocol"
            )
        warmup_rows = []
        warmup_summary = _empty_stream_summary()
        warmup_path = args.output_dir / "warmup_trajectory.jsonl"
        warmup_path.write_text("", encoding="utf-8")
        _write_json(
            args.output_dir / "warmup_progress.json",
            {"completed_instances": 0, "summary": warmup_summary},
        )
        branches = {}
        warmup_state_artifact: Any = {}
        warmup_state_dir = args.output_dir / "bandit_warmup_states"
        warmup_state_dir.mkdir(parents=True, exist_ok=True)
        aot_enabled = str(args.setup_candidate_mode) == "aot"
        for method in bandit_methods:
            method_spec = composable_specs[method]
            configuration_space = setup_configuration_spaces[
                str(method_spec.setup_space)
            ]
            branch, _bandit_cfg = _build_bandit(
                seed=int(args.bandit_seed),
                learner_kind=str(method_spec.setup_kind),
                tune_dim=7,
                tune7_variant="categorical",
                action_space_mode=str(args.setup_action_space),
                solver_tol=float(args.tol),
                solver_max_iter=int(args.max_cycles),
                parameter_resolution=int(args.setup_param_resolution),
                configuration_space=configuration_space,
                candidate_schedule_dir=(
                    args.output_dir
                    / "aot_candidate_schedules"
                    / configuration_space.name
                    / method_spec.candidate_sampling
                    if aot_enabled
                    else None
                ),
                candidate_schedule_rounds=(
                    int(args.online_cases)
                    * int(args.aot_max_selections_per_case)
                    if aot_enabled
                    else None
                ),
                candidate_schedule_chunk_rounds=int(
                    args.aot_schedule_chunk_rounds
                ),
                lin_ts_relative_sampling_scale=float(
                    args.lin_ts_relative_sampling_scale
                ),
                lin_ts_loss_scale_prior=float(args.lin_ts_loss_scale_prior),
                candidate_sampling=str(method_spec.candidate_sampling),
            )
            branches[method] = branch
            state_path = warmup_state_dir / f"{method}.npz"
            branch.policy.model.save_mutable_state(
                state_path,
                metadata={
                    "stream_hash": str(stream_manifest["sha256"]),
                    "summary": warmup_summary,
                    "method": method,
                    "setup_space": configuration_space.name,
                },
            )
            warmup_state_artifact[method] = str(state_path)
    else:
        aot_enabled = False
        warmup_branch, warmup_rows, warmup_summary = _warmup_bandit(
            args,
            warmup_instances,
            stream_hash=str(stream_manifest["sha256"]),
        )
        branches = {
            method: clone_branch_for_independent_updates(warmup_branch)
            for method in bandit_methods
        }
        warmup_state_artifact = str(
            args.output_dir
            / f"bandit_warmup_{int(args.warmup_cases)}.npz"
        )
    if len({id(branch.policy) for branch in branches.values()}) != len(branches):
        raise RuntimeError("Bandit branches do not have independent policy objects")
    branch_models = [getattr(branch.policy, "model", None) for branch in branches.values()]
    if any(model is None for model in branch_models):
        raise RuntimeError("The locked protocol requires model-backed LinUCB branches")
    if len({id(model.A_inv) for model in branch_models}) != len(branch_models):
        raise RuntimeError("LinUCB mutable parameter matrices are not independent")
    if setup_configuration_spaces:
        if len({id(model.actions) for model in branch_models}) != len(branch_models):
            raise RuntimeError(
                "Named setup spaces must own distinct action catalogs"
            )
        if str(args.setup_candidate_mode) == "aot":
            if any(model._g_actions is not None for model in branch_models):
                raise RuntimeError(
                    "AOT LinUCB branches must not materialize _g_actions"
                )
            if any(
                model._candidate_schedule is None
                or model._action_feature_cache is None
                for model in branch_models
            ):
                raise RuntimeError(
                    "AOT branches require schedules and factorized caches"
                )
        elif any(model._g_actions is None for model in branch_models):
            raise RuntimeError("Explicit branches require cached _g_actions")
    elif branch_models and len(
        {id(model._g_actions) for model in branch_models}
    ) != 1:
        raise RuntimeError(
            "LinUCB branches must share the immutable _g_actions cache"
        )
    if any(
        model._g_actions is not None and model._g_actions.flags.writeable
        for model in branch_models
    ):
        raise RuntimeError("Cached LinUCB action features must be read-only")

    controllers: Dict[str, Any] = {}
    encoders: Dict[str, SolveStateEncoder] = {}
    if args.study_mode == "composable":
        controller_factories = {
            "recursive_mc": _make_recursive_mc_controller,
            "recursive_lstdq_v1": _make_recursive_lstdq_controller,
            "recursive_lstdq_v2": _make_recursive_lstdq_v2_controller,
            "rblspi": _make_recursive_blstdq_controller,
            "stagewise_lsvi": _make_lsvi_controller,
            "structured_model_based": _make_structured_model_based_controller,
            "recalibrated_lsvi": _make_recalibrated_lsvi_controller,
        }
        seed_offsets = {
            "recursive_mc": 0,
            "recursive_lstdq_v1": 1009,
            "recursive_lstdq_v2": 2018,
            "rblspi": 5045,
            "stagewise_lsvi": 2018,
            "structured_model_based": 3027,
            "recalibrated_lsvi": 4036,
        }
        for spec in composable_specs_tuple:
            factory = controller_factories.get(spec.solve_kind)
            if factory is None:
                continue
            factory_kwargs: Dict[str, Any] = {
                "seed": int(
                    args.controller_seed + seed_offsets[spec.solve_kind]
                )
            }
            if spec.solve_kind == "stagewise_lsvi":
                factory_kwargs["refit_interval_episodes"] = int(
                    args.lsvi_refit_interval_episodes
                )
            controller, encoder = factory(args, **factory_kwargs)
            controllers[spec.name] = controller
            encoders[spec.name] = encoder
        ppo_runner = (
            _make_frozen_ppo_runner(args)
            if any(
                spec.solve_kind == "ppo"
                for spec in composable_specs_tuple
            )
            else None
        )
    elif args.study_mode == "lsvi_lcb":
        controller, encoder = _make_lsvi_controller(
            args,
            seed=int(args.controller_seed),
        )
        controllers[LSVI_METHOD] = controller
        encoders[LSVI_METHOD] = encoder
        ppo_runner = None
    elif args.study_mode == "solve_controller_screen":
        factories = (
            (RECURSIVE_LSTDQ_METHOD, _make_recursive_lstdq_controller),
            (RECURSIVE_LSTDQ_V2_METHOD, _make_recursive_lstdq_v2_controller),
            (
                STRUCTURED_MODEL_BASED_METHOD,
                _make_structured_model_based_controller,
            ),
            (
                RECALIBRATED_LSVI_METHOD,
                _make_recalibrated_lsvi_controller,
            ),
        )
        for method, factory in factories:
            controller, encoder = factory(
                args,
                seed=int(
                    args.controller_seed
                    + SOLVE_CONTROLLER_SEED_OFFSETS[method]
                ),
            )
            controllers[method] = controller
            encoders[method] = encoder
        ppo_runner = None
    elif args.study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        factories = []
        if RECURSIVE_MC_METHOD in methods:
            factories.append(
                (RECURSIVE_MC_METHOD, _make_recursive_mc_controller)
            )
        if RECURSIVE_LSTDQ_METHOD in methods:
            factories.append(
                (RECURSIVE_LSTDQ_METHOD, _make_recursive_lstdq_controller)
            )
        if BATCHED_LSVI_METHOD in methods:
            factories.append((BATCHED_LSVI_METHOD, _make_lsvi_controller))
        seed_offsets = {
            RECURSIVE_MC_METHOD: 0,
            RECURSIVE_LSTDQ_METHOD: 1009,
            BATCHED_LSVI_METHOD: 2018,
        }
        for method, factory in factories:
            factory_kwargs: Dict[str, Any] = {
                "seed": int(args.controller_seed + seed_offsets[method]),
            }
            if method == BATCHED_LSVI_METHOD:
                factory_kwargs["refit_interval_episodes"] = int(
                    args.lsvi_refit_interval_episodes
                )
            controller, encoder = factory(args, **factory_kwargs)
            controllers[method] = controller
            encoders[method] = encoder
        ppo_runner = (
            _make_frozen_ppo_runner(args)
            if "bandit_ppo" in methods
            else None
        )
    else:
        for candidate_index, candidate in enumerate(candidates):
            controller, encoder = _make_controller(
                args,
                candidate,
                seed=int(args.controller_seed + candidate_index * 1009),
            )
            controllers[candidate.name] = controller
            encoders[candidate.name] = encoder

        ppo_runner = _make_frozen_ppo_runner(args)
    bandit_cfg = default_test_final_bandit_config_from_env()
    previous_update = {method: 0.0 for method in bandit_methods}
    bandit_online_steps = {method: 0 for method in bandit_methods}
    records: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    handles = {
        method: (trajectories_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in methods
    }
    order_handle = (trajectories_dir / "method_order.jsonl").open("w", encoding="utf-8")
    order_rng = np.random.default_rng(int(args.method_order_seed))
    progress_denom = max(1, len(online_instances) - 1)
    try:
        for online_index, (mkw, context) in enumerate(online_instances):
            order = [methods[int(index)] for index in order_rng.permutation(len(methods))]
            _write_json_line(
                order_handle,
                {"online_index": int(online_index), "method_order": order},
            )
            case_rows: Dict[str, Dict[str, Any]] = {}
            for execution_rank, method in enumerate(order):
                if method == DEFAULT_SETUP_METHOD and method not in composable_specs:
                    native = solve_default_baseline_case(
                        mkw=dict(mkw),
                        solver_tol=float(args.tol),
                        solver_max_iter=int(args.max_cycles),
                        augment_params=augment_setup_params,
                    )
                    outcome = _report_online_outcome(
                        _as_feedback(native, include_controller=False),
                        bandit_timing={},
                    )
                    case_rows[method] = {
                        "stream_index": int(args.warmup_cases + online_index),
                        "online_index": int(online_index),
                        "execution_rank": int(execution_rank),
                        "mkw": dict(mkw),
                        "context": np.asarray(context, dtype=float).tolist(),
                        "params": dict(DEFAULT_SETUP_PARAMS),
                        "arm_index": -1,
                        "fallback_used": 0,
                        "bandit_timing": {},
                        "outcome": outcome,
                    }
                    continue
                spec = composable_specs.get(method)
                solver_fn = _method_solver(
                    method,
                    args=args,
                    mkw=dict(mkw),
                    case_progress=float(online_index) / float(progress_denom),
                    controllers=controllers,
                    encoders=encoders,
                    ppo_runner=ppo_runner,
                    composable_specs=composable_specs,
                )
                if spec is not None and spec.setup_kind == "default":
                    outcome = _run_default_setup_method(
                        spec=spec,
                        solver_fn=solver_fn,
                        args=args,
                        mkw=dict(mkw),
                        controller_methods=tuple(controllers),
                    )
                    case_rows[method] = {
                        "stream_index": int(args.warmup_cases + online_index),
                        "online_index": int(online_index),
                        "execution_rank": int(execution_rank),
                        "mkw": dict(mkw),
                        "context": np.asarray(context, dtype=float).tolist(),
                        "params": dict(DEFAULT_SETUP_PARAMS),
                        "arm_index": -1,
                        "fallback_used": int(
                            bool(outcome.get("fallback_used", False))
                        ),
                        "bandit_timing": {},
                        "outcome": outcome,
                    }
                    continue
                def fallback_solver(_params: Dict[str, Any]) -> Dict[str, Any]:
                    fallback_native = solve_no_rl_case(
                        params=dict(DEFAULT_SETUP_PARAMS),
                        mkw=dict(mkw),
                        solver_tol=float(args.tol),
                        solver_max_iter=int(args.max_cycles),
                        augment_params=augment_setup_params,
                    )
                    return _as_feedback(fallback_native, include_controller=False)
                branch = branches[method]
                params, native, timing, fallback_used, update_sec = (
                    run_bandit_step_test_final(
                        policy=branch.policy,
                        parameter_space=branch.parameter_space,
                        context=np.asarray(context, dtype=float),
                        solver_fn=solver_fn,
                        fallback_solver_fn=fallback_solver,
                        prev_update_est=float(previous_update[method]),
                    )
                )
                previous_update[method] = float(update_sec)
                if aot_enabled:
                    branch.policy.model.finish_candidate_schedule_case(
                        max_selections=int(args.aot_max_selections_per_case)
                    )
                bandit_online_steps[method] += 1
                row = {
                    "stream_index": int(args.warmup_cases + online_index),
                    "online_index": int(online_index),
                    "execution_rank": int(execution_rank),
                    "mkw": dict(mkw),
                    "context": np.asarray(context, dtype=float).tolist(),
                    "params": dict(params),
                    "arm_index": _policy_last_arm(branch.policy),
                    "fallback_used": int(fallback_used),
                    "bandit_timing": dict(timing),
                    "outcome": _report_online_outcome(native, bandit_timing=timing),
                }
                case_rows[method] = row
            for method in methods:
                records[method].append(case_rows[method])
                _write_json_line(handles[method], case_rows[method])

            done = online_index + 1
            audit_episodes = {
                *range(1000, int(args.online_cases) + 1, 1000),
                int(args.online_cases),
            }
            if done in audit_episodes:
                for method, controller in controllers.items():
                    controller.save(
                        checkpoints_dir / f"{method}_episode_{done}.npz"
                    )
            if done % max(1, int(args.progress_every)) == 0 or done == len(online_instances):
                for handle in handles.values():
                    handle.flush()
                order_handle.flush()
                progress = {
                    "completed_online_instances": int(done),
                    "methods": {
                        method: _method_stream_summary(rows)
                        for method, rows in records.items()
                    },
                    "controllers": {
                        method: _controller_summary(controller)
                        for method, controller in controllers.items()
                    },
                    "bandit_online_steps": dict(bandit_online_steps),
                }
                _write_json(args.output_dir / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            "stage": "joint_online_4k_compare",
                            "done": int(done),
                            "total": int(len(online_instances)),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()
        order_handle.close()

    recovery_audit = {
        method: _validate_recovery_stream(
            rows,
            expect_bandit_transaction=method in bandit_methods,
        )
        for method, rows in records.items()
    }

    if (
        bandit_online_steps
        and set(bandit_online_steps.values()) != {len(online_instances)}
    ):
        raise RuntimeError(
            "Every LinUCB branch must process every online instance: "
            f"{bandit_online_steps}"
        )

    final_bandit_dir = args.output_dir / "final_bandit_states"
    final_bandit_dir.mkdir(parents=True, exist_ok=True)
    for method, branch in branches.items():
        branch.policy.model.save_mutable_state(
            final_bandit_dir / f"{method}.npz",
            metadata={"method": method, "online_steps": bandit_online_steps[method]},
        )
    for method, controller in controllers.items():
        controller.save(checkpoints_dir / f"{method}_final.npz")

    windows = _comparison_windows(int(args.online_cases))
    window_results = {
        name: _window_result(
            {method: rows[start:stop] for method, rows in records.items()},
            seed=int(args.method_order_seed + start + stop),
        )
        for name, (start, stop) in windows.items()
    }
    window_actions = {
        name: {
            method: _action_summary(records[method][start:stop])
            for method in controllers
        }
        for name, (start, stop) in windows.items()
    }
    summary_name = f"summary_{int(args.online_cases)}.csv"
    has_solve_screen_report = (
        args.study_mode == "solve_controller_screen"
        or (args.study_mode == "composable" and bool(controllers))
    )
    result = {
        "protocol": protocol,
        "warmup": {
            "summary": warmup_summary,
            "trajectory": warmup_trajectory_artifact,
            "state": warmup_state_artifact,
            "records": int(len(warmup_rows)),
        },
        "windows": window_results,
        "controllers": {
            method: _controller_summary(controller)
            for method, controller in controllers.items()
        },
        "bandit_online_steps": dict(bandit_online_steps),
        "recovery_audit": recovery_audit,
        "actions": {
            method: _action_summary(records[method]) for method in controllers
        },
        "window_actions": window_actions,
        "artifacts": {
            "trajectories": str(trajectories_dir),
            "summary_csv": str(args.output_dir / summary_name),
            "checkpoints": str(checkpoints_dir),
            "final_bandit_states": str(final_bandit_dir),
            **(
                {"screen_report": str(args.output_dir / "screen_report.md")}
                if has_solve_screen_report
                else {}
            ),
        },
    }
    if ppo_runner is not None:
        result["ppo"] = {
            "forced_initial_action_count": int(
                ppo_runner.forced_initial_action_count
            ),
        }
    _write_summary_csv(
        args.output_dir / summary_name,
        records,
        family_by_method,
    )
    if has_solve_screen_report:
        _write_solve_screen_report(
            args.output_dir / "screen_report.md",
            window_results=window_results,
            window_actions=window_actions,
        )
    _write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_online_4k_done",
                    "output": args.output_dir / "result.json",
                }
            )
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the shared low-level parser used by CLI and high-level configs."""

    parser = argparse.ArgumentParser(
        description=(
            "Run either the locked 2K+2K protocol or the same 4K stream with "
            "setup bandits and solve controllers learning jointly from scratch."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--study-mode",
        choices=(
            "sarsa",
            "lsvi_lcb",
            "recursive_lcb_suite",
            "recursive_lcb_ppo",
            "recursive_lstdq_lcb",
            "solve_controller_screen",
            "composable",
        ),
        default="sarsa",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/joint/mature_tune7_ppo_repro_20260423/run_logs/"
            "exp44_absolute_default_lstm_canonical_20260718/model_best.zip"
        ),
    )
    parser.add_argument(
        "--ppo-action-mode",
        choices=("continuous", "continuous_absolute"),
        default="continuous_absolute",
    )
    parser.add_argument("--ppo-w-center", type=float, default=1.5)
    parser.add_argument("--ppo-w-scale", type=float, default=0.5)
    parser.add_argument(
        "--ppo-initial-observation-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--ppo-force-default-first-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ppo-default-first-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=39860039)
    parser.add_argument("--bandit-seed", type=int, default=39860039)
    parser.add_argument("--controller-seed", type=int, default=39866039)
    parser.add_argument("--method-order-seed", type=int, default=39872039)
    parser.add_argument("--train-cases", type=int, default=4000)
    parser.add_argument("--warmup-cases", type=int, default=2000)
    parser.add_argument("--online-cases", type=int, default=2000)
    parser.add_argument("--instance-offset", type=int, default=0)
    parser.add_argument(
        "--train-seed-groups",
        default=(
            "39800039,39806039,39812039,39818039,"
            "39824039,39830039,39836039,39842039"
        ),
    )
    parser.add_argument("--train-shuffle-seeds", default="39848039")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-group-take", type=int, default=4000)
    parser.add_argument(
        "--expected-stream-hash",
        default="",
        help="Optional SHA-256 lock for the generated instance stream.",
    )
    parser.add_argument(
        "--problem",
        choices=("scalar_anisotropic_diffusion", "diffusion_convection", "difconv"),
        default="scalar_anisotropic_diffusion",
    )
    parser.add_argument(
        "--matrix-grid-n",
        "--grid-n",
        dest="grid_n",
        type=int,
        default=EXP44_MATRIX_GRID_N,
    )
    parser.add_argument(
        "--grid-shape",
        default=None,
        help="Optional nx,ny,nz grid; otherwise --matrix-grid-n is cubic.",
    )
    parser.add_argument(
        "--advection",
        default="0,0,0",
        help="ax,ay,az for diffusion_convection problems.",
    )
    parser.add_argument(
        "--setup-param-resolution",
        type=int,
        default=EXP44_SETUP_PARAM_RESOLUTION,
    )
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument(
        "--setup-action-space",
        choices=("full_cartesian",),
        default="full_cartesian",
    )
    parser.add_argument(
        "--setup-candidate-mode",
        choices=("explicit", "aot"),
        default="explicit",
    )
    parser.add_argument("--aot-max-selections-per-case", type=int, default=3)
    parser.add_argument("--aot-schedule-chunk-rounds", type=int, default=256)
    parser.add_argument(
        "--lin-ts-relative-sampling-scale", type=float, default=0.15
    )
    parser.add_argument("--lin-ts-loss-scale-prior", type=float, default=0.1)
    parser.add_argument(
        "--shared-action-profile",
        choices=tuple(SHARED_ACTION_PROFILES),
        default="1to2_step0p1",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Explicit solve action grid; locked modes require the profile grid.",
    )
    parser.add_argument(
        "--action-rbf-centers",
        default=None,
        help="Explicit action RBF centers; locked modes require the profile grid.",
    )
    parser.add_argument("--action-rbf-sigma", type=float, default=0.2)
    parser.add_argument("--alphas", default="0.001")
    parser.add_argument("--trace-lambdas", default="0.8")
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20_000.0)
    parser.add_argument("--q-max-sec", type=float, default=0.1)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--uncertainty-ridge", type=float, default=1.0)
    parser.add_argument("--uncertainty-td-floor-sec", type=float, default=1.0e-3)
    parser.add_argument("--lsvi-ridge", type=float, default=1.0)
    parser.add_argument("--lsvi-beta", type=float, default=2.0)
    parser.add_argument("--lsvi-residual-floor-sec", type=float, default=1.0e-3)
    parser.add_argument("--lsvi-refit-interval-episodes", type=int, default=100)
    parser.add_argument("--recursive-mc-ridge", type=float, default=1.0)
    parser.add_argument("--recursive-mc-beta", type=float, default=2.0)
    parser.add_argument(
        "--recursive-mc-residual-floor-sec", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--recursive-mc-episode-half-life", type=float, default=500.0
    )
    parser.add_argument("--recursive-lstdq-ridge", type=float, default=1.0)
    parser.add_argument("--recursive-lstdq-beta", type=float, default=2.0)
    parser.add_argument("--recursive-lstdq-lambda", type=float, default=0.8)
    parser.add_argument(
        "--recursive-lstdq-residual-floor-sec", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--recursive-lstdq-lcb-lower-bound-sec", type=float, default=0.0
    )
    parser.add_argument("--recursive-lstdq-v2-beta", type=float, default=2.0)
    parser.add_argument(
        "--recursive-lstdq-v2-coverage-ridge", type=float, default=1.0
    )
    parser.add_argument(
        "--recursive-lstdq-v2-residual-window", type=int, default=2048
    )
    parser.add_argument(
        "--recursive-lstdq-v2-min-samples", type=int, default=32
    )
    parser.add_argument("--rblspi-prior-precision", type=float, default=1.0e4)
    parser.add_argument("--rblspi-noise-precision", type=float, default=1.0e6)
    parser.add_argument("--rblspi-gram-ridge", type=float, default=1.0e-6)
    parser.add_argument("--structured-model-ridge", type=float, default=1.0)
    parser.add_argument(
        "--structured-model-min-samples", type=int, default=32
    )
    parser.add_argument(
        "--structured-model-scale-window", type=int, default=2048
    )
    parser.add_argument("--recalibrated-lsvi-beta", type=float, default=2.0)
    parser.add_argument(
        "--recalibrated-lsvi-refit-sweeps", type=int, default=3
    )
    parser.add_argument(
        "--recalibrated-lsvi-shrinkage-samples", type=float, default=32.0
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--method",
        dest="method_specs",
        action="append",
        default=[],
        help=(
            "Composable branch as name:setup:solve; setup is default/linucb/lints "
            "and a fixed solve uses fixed@weight. Repeat for each branch."
        ),
    )
    parser.add_argument(
        "--include-default-setup-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reuse-warmup", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
