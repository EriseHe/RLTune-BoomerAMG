"""Compatibility adapters from legacy experiment namespaces to solve specs.

The canonical solve-controller constructors live in :mod:`solve.registry`.
This module is deliberately limited to translating the historical argparse
namespace used by the 4K runner and diagnostics into those typed requests.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Dict

from setup.space import (
    DEFAULT_SETUP_PARAMS,
    SetupConfigurationSpace,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from solve.controllers.common import (
    ControllerBundle,
    EpsilonScheduleSpec,
    SharedActionSpec,
    SolveStateSpec,
)
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.registry import (
    build_online_solve_controller,
    make_online_controller_spec,
)


def parse_csv_values(raw: str, cast: Any) -> tuple[Any, ...]:
    """Parse the runner's compact comma-separated compatibility values."""

    return tuple(
        cast(part.strip())
        for part in str(raw).split(",")
        if part.strip()
    )


def make_setup_obs_encoder(
    *,
    configuration_space: SetupConfigurationSpace | None = None,
    parameter_resolution: int | None = None,
    strict_categories: bool = False,
) -> SetupObsEncoder:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
        parameter_resolution=parameter_resolution,
        coarsen_type_values=(
            configuration_space.coarsen_types if configuration_space else None
        ),
        interp_type_values=(
            configuration_space.interp_types if configuration_space else None
        ),
        agg_interp_type_values=(
            configuration_space.agg_interp_types if configuration_space else None
        ),
    )
    if strict_categories:
        # The native default remains a valid observation even outside a grid.
        parameter_spec = replace(parameter_spec, parameters=tuple(
            replace(param, values=(*param.values, param.default))
            if param.kind == "categorical" and param.default not in param.values
            else param
            for param in parameter_spec.parameters
        ))
    return SetupObsEncoder(
        parameter_spec,
        dict(DEFAULT_SETUP_PARAMS),
        tuple(parameter_spec.parameter_names),
        strict_categories=strict_categories,
    )


def make_solve_state_spec(args: argparse.Namespace) -> SolveStateSpec:
    return SolveStateSpec(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        mode="setup_full",
        problem_context_mode=str(
            getattr(
                args,
                "solve_problem_context_mode",
                "canonical",
            )
        ),
    )


def make_encoder(args: argparse.Namespace) -> SolveStateEncoder:
    return make_solve_state_spec(args).build_encoder(
        setup_obs_encoder=make_setup_obs_encoder()
    )


def make_shared_action_spec(args: argparse.Namespace) -> SharedActionSpec:
    weights = parse_csv_values(args.weights, float)
    return SharedActionSpec(
        weights=tuple(float(weight) for weight in weights),
        anchor_weight=float(weights[0]),
        rbf_sigma=float(args.action_rbf_sigma),
        rbf_centers=tuple(
            float(center)
            for center in parse_csv_values(args.action_rbf_centers, float)
        ),
        epsilon=EpsilonScheduleSpec(
            start=float(args.epsilon_start),
            final=float(args.epsilon_final),
            decay_steps=float(args.epsilon_decay_steps),
        ),
        force_default_first_action=True,
    )


def make_shared_action_config(
    args: argparse.Namespace,
    *,
    trace_lambda: float,
) -> ExpectedSarsaLambdaConfig:
    """Return the legacy TD config for diagnostics that still import it."""

    return make_shared_action_spec(args).to_td_config(
        trace_lambda=float(trace_lambda),
    )


def build_online_controller_bundle_from_args(
    args: argparse.Namespace,
    *,
    kind: str,
    seed: int,
    algorithm_parameters: Dict[str, Any],
    trace_lambda: float | None = None,
) -> ControllerBundle:
    """Translate one legacy namespace into the canonical solve factory."""

    request = make_online_controller_spec(
        kind=kind,
        state=make_solve_state_spec(args),
        actions=make_shared_action_spec(args),
        algorithm_parameters=algorithm_parameters,
        trace_lambda=trace_lambda,
    )
    return build_online_solve_controller(
        request,
        setup_obs_encoder=make_setup_obs_encoder(),
        seed=int(seed),
    )


def make_online_controller_from_args(
    args: argparse.Namespace,
    *,
    kind: str,
    seed: int,
    algorithm_parameters: Dict[str, Any],
    trace_lambda: float | None = None,
) -> tuple[Any, SolveStateEncoder]:
    """Compatibility tuple for diagnostics written before ControllerBundle."""

    return build_online_controller_bundle_from_args(
        args,
        kind=kind,
        seed=seed,
        algorithm_parameters=algorithm_parameters,
        trace_lambda=trace_lambda,
    ).as_legacy_tuple()


def build_lsvi_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
    refit_interval_episodes: int = 1,
) -> ControllerBundle:
    return build_online_controller_bundle_from_args(
        args,
        kind="stagewise_lsvi",
        seed=seed,
        algorithm_parameters=dict(
            horizon=int(args.max_cycles),
            ridge=float(args.lsvi_ridge),
            uncertainty_beta=float(args.lsvi_beta),
            residual_floor_sec=float(args.lsvi_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            refit_interval_episodes=int(refit_interval_episodes),
        ),
    )


def make_lsvi_controller(
    args: argparse.Namespace,
    *,
    seed: int,
    refit_interval_episodes: int = 1,
) -> tuple[Any, SolveStateEncoder]:
    return build_lsvi_controller_bundle(
        args,
        seed=seed,
        refit_interval_episodes=refit_interval_episodes,
    ).as_legacy_tuple()


def build_recursive_mc_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    return build_online_controller_bundle_from_args(
        args,
        kind="recursive_mc",
        seed=seed,
        algorithm_parameters=dict(
            ridge=float(args.recursive_mc_ridge),
            uncertainty_beta=float(args.recursive_mc_beta),
            residual_floor_sec=float(args.recursive_mc_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            episode_half_life=float(args.recursive_mc_episode_half_life),
        ),
    )


def make_recursive_mc_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recursive_mc_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_recursive_lstdq_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    lcb_lower_bound = getattr(
        args,
        "recursive_lstdq_lcb_lower_bound_sec",
        0.0,
    )
    return build_online_controller_bundle_from_args(
        args,
        kind="recursive_lstdq_v1",
        seed=seed,
        trace_lambda=float(args.recursive_lstdq_lambda),
        algorithm_parameters=dict(
            ridge=float(args.recursive_lstdq_ridge),
            uncertainty_beta=float(args.recursive_lstdq_beta),
            residual_floor_sec=float(args.recursive_lstdq_residual_floor_sec),
            q_max_sec=float(getattr(args, "q_max_sec", 0.1)),
            lcb_lower_bound_sec=(
                None
                if lcb_lower_bound is None
                else float(lcb_lower_bound)
            ),
        ),
    )


def make_recursive_lstdq_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recursive_lstdq_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_recursive_lstdq_v2_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    lcb_lower_bound = getattr(
        args,
        "recursive_lstdq_lcb_lower_bound_sec",
        0.0,
    )
    return build_online_controller_bundle_from_args(
        args,
        kind="recursive_lstdq_v2",
        seed=seed,
        trace_lambda=float(args.recursive_lstdq_lambda),
        algorithm_parameters=dict(
            ridge=float(args.recursive_lstdq_ridge),
            uncertainty_beta=float(args.recursive_lstdq_v2_beta),
            residual_floor_sec=float(args.recursive_lstdq_residual_floor_sec),
            lcb_lower_bound_sec=(
                None
                if lcb_lower_bound is None
                else float(lcb_lower_bound)
            ),
            coverage_ridge=float(
                args.recursive_lstdq_v2_coverage_ridge
            ),
            residual_scale_window=int(
                args.recursive_lstdq_v2_residual_window
            ),
            residual_scale_min_samples=int(
                args.recursive_lstdq_v2_min_samples
            ),
        ),
    )


def make_recursive_lstdq_v2_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recursive_lstdq_v2_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_recursive_lstdq_v3_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    lcb_lower_bound = getattr(
        args,
        "recursive_lstdq_lcb_lower_bound_sec",
        0.0,
    )
    return build_online_controller_bundle_from_args(
        args,
        kind="recursive_lstdq_v3",
        seed=seed,
        trace_lambda=float(args.recursive_lstdq_lambda),
        algorithm_parameters=dict(
            ridge=float(args.recursive_lstdq_ridge),
            uncertainty_beta=float(args.recursive_lstdq_v3_beta),
            residual_floor_sec=float(args.recursive_lstdq_residual_floor_sec),
            lcb_lower_bound_sec=(
                None
                if lcb_lower_bound is None
                else float(lcb_lower_bound)
            ),
        ),
    )


def make_recursive_lstdq_v3_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recursive_lstdq_v3_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_recursive_blstdq_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    return build_online_controller_bundle_from_args(
        args,
        kind="rblspi",
        seed=seed,
        algorithm_parameters=dict(
            prior_precision=float(args.rblspi_prior_precision),
            noise_precision=float(args.rblspi_noise_precision),
            gram_ridge=float(args.rblspi_gram_ridge),
        ),
    )


def make_recursive_blstdq_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recursive_blstdq_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_structured_model_based_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    return build_online_controller_bundle_from_args(
        args,
        kind="structured_model_based",
        seed=seed,
        algorithm_parameters=dict(
            ridge=float(args.structured_model_ridge),
            minimum_samples=int(args.structured_model_min_samples),
            scale_window=int(args.structured_model_scale_window),
        ),
    )


def make_structured_model_based_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_structured_model_based_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


def build_recalibrated_lsvi_controller_bundle(
    args: argparse.Namespace,
    *,
    seed: int,
) -> ControllerBundle:
    return build_online_controller_bundle_from_args(
        args,
        kind="recalibrated_lsvi",
        seed=seed,
        algorithm_parameters=dict(
            horizon=int(args.max_cycles),
            ridge=float(args.lsvi_ridge),
            uncertainty_beta=float(args.recalibrated_lsvi_beta),
            residual_floor_sec=float(args.lsvi_residual_floor_sec),
            refit_interval_episodes=int(
                args.lsvi_refit_interval_episodes
            ),
            refit_sweeps=int(args.recalibrated_lsvi_refit_sweeps),
            residual_shrinkage_samples=float(
                args.recalibrated_lsvi_shrinkage_samples
            ),
        ),
    )


def make_recalibrated_lsvi_controller(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, SolveStateEncoder]:
    return build_recalibrated_lsvi_controller_bundle(
        args,
        seed=seed,
    ).as_legacy_tuple()


__all__ = [
    "build_lsvi_controller_bundle",
    "build_online_controller_bundle_from_args",
    "build_recalibrated_lsvi_controller_bundle",
    "build_recursive_blstdq_controller_bundle",
    "build_recursive_lstdq_controller_bundle",
    "build_recursive_lstdq_v2_controller_bundle",
    "build_recursive_lstdq_v3_controller_bundle",
    "build_recursive_mc_controller_bundle",
    "build_structured_model_based_controller_bundle",
    "make_encoder",
    "make_lsvi_controller",
    "make_online_controller_from_args",
    "make_recalibrated_lsvi_controller",
    "make_recursive_blstdq_controller",
    "make_recursive_lstdq_controller",
    "make_recursive_lstdq_v2_controller",
    "make_recursive_lstdq_v3_controller",
    "make_recursive_mc_controller",
    "make_setup_obs_encoder",
    "make_shared_action_config",
    "make_shared_action_spec",
    "make_solve_state_spec",
    "make_structured_model_based_controller",
    "parse_csv_values",
]
