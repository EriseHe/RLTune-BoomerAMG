"""Composable setup/solve assembly for the canonical 4K experiment.

This module owns the composable method roster and the construction of its
setup learners and solve controllers.  The canonical runner remains
responsible for the experiment stream, retry/fallback protocol, and artifacts;
the mode-neutral online loop remains in :mod:`joint_4k_execution`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from joint_controller_build import (
    build_lsvi_controller_bundle,
    build_recalibrated_lsvi_controller_bundle,
    build_recursive_blstdq_controller_bundle,
    build_recursive_lstdq_controller_bundle,
    build_recursive_lstdq_v2_controller_bundle,
    build_recursive_lstdq_v3_controller_bundle,
    build_recursive_mc_controller_bundle,
    build_structured_model_based_controller_bundle,
    make_setup_obs_encoder,
    parse_csv_values,
)
from joint_method_spec import ComposableMethodSpec
from joint_reporting import _empty_stream_summary
from online_td_experiment_common import _write_json
from setup.registry import ONLINE_SETUP_KINDS
from setup.space import SetupConfigurationSpace
from setup_aware_compare_common import (
    build_online_linucb_branch,
)
from solve.controllers.common import ControllerBundle
from solve.controllers.ppo import (
    FrozenPpoConfig,
    build_frozen_ppo_runner,
)
from solve.registry import ONLINE_SOLVE_KINDS, build_online_solve_controller


SETUP_BANDIT_KINDS = ONLINE_SETUP_KINDS

_CONTROLLER_SEED_OFFSETS = {
    "recursive_mc": 0,
    "recursive_lstdq_v1": 1009,
    "recursive_lstdq_v2": 2018,
    "recursive_lstdq_v3": 2018,
    "rblspi": 5045,
    "stagewise_lsvi": 2018,
    "structured_model_based": 3027,
    "recalibrated_lsvi": 4036,
}

_COMPATIBILITY_BUNDLE_FACTORIES = {
    "recursive_mc": build_recursive_mc_controller_bundle,
    "recursive_lstdq_v1": build_recursive_lstdq_controller_bundle,
    "recursive_lstdq_v2": build_recursive_lstdq_v2_controller_bundle,
    "recursive_lstdq_v3": build_recursive_lstdq_v3_controller_bundle,
    "rblspi": build_recursive_blstdq_controller_bundle,
    "stagewise_lsvi": build_lsvi_controller_bundle,
    "structured_model_based": build_structured_model_based_controller_bundle,
    "recalibrated_lsvi": build_recalibrated_lsvi_controller_bundle,
}


@dataclass(frozen=True)
class ComposableStudy:
    """Validated method roster and named setup spaces for one experiment."""

    specs: tuple[ComposableMethodSpec, ...]
    specs_by_name: Mapping[str, ComposableMethodSpec]
    methods: tuple[str, ...]
    family_by_method: Mapping[str, str]
    setup_configuration_spaces: Mapping[str, SetupConfigurationSpace]
    bandit_methods: tuple[str, ...]


@dataclass(frozen=True)
class ComposableSetupArtifacts:
    """Per-method setup learners and their zero-warmup artifact contract."""

    branches: Mapping[str, Any]
    warmup_rows: tuple[Mapping[str, Any], ...]
    warmup_summary: Mapping[str, Any]
    warmup_trajectory: str
    warmup_state: Mapping[str, str]
    aot_enabled: bool


@dataclass(frozen=True)
class ComposableSolveRuntime:
    """Built online solve controllers plus the optional frozen PPO policy."""

    controller_bundles: Mapping[str, ControllerBundle]
    ppo_runner: Any | None


def parse_composable_method(raw: str) -> ComposableMethodSpec:
    return ComposableMethodSpec.from_runner_token(raw)


def setup_configuration_spaces_from_args(
    args: Any,
) -> Dict[str, SetupConfigurationSpace]:
    """Return already-decoded named setup spaces after identity validation."""

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


def _validate_specs(
    args: Any,
    *,
    configuration_spaces: Mapping[str, SetupConfigurationSpace],
) -> tuple[ComposableMethodSpec, ...]:
    # An argparse Namespace is always a compatibility input, even when it was
    # adapted from a typed config and therefore also carries ``methods``.
    # Preserve the historical contract that callers may edit ``method_specs``.
    typed_methods = (
        None
        if isinstance(args, argparse.Namespace)
        else getattr(args, "methods", None)
    )
    if typed_methods is not None:
        specs = tuple(typed_methods)
        if not all(
            isinstance(spec, ComposableMethodSpec)
            for spec in specs
        ):
            raise TypeError(
                "Canonical composable methods must be "
                "ComposableMethodSpec instances"
            )
    else:
        specs = tuple(
            parse_composable_method(raw)
            for raw in tuple(getattr(args, "method_specs", ()) or ())
        )
    if not specs:
        raise ValueError("Composable experiments require at least one --method")
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("Composable method names must be unique")

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
    weights = np.asarray(parse_csv_values(args.weights, float), dtype=float)
    centers = np.asarray(
        parse_csv_values(args.action_rbf_centers, float),
        dtype=float,
    )
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


def validate_composable_protocol(
    args: Any,
) -> tuple[ComposableMethodSpec, ...]:
    """Compatibility validator returning the ordered method specs."""

    return _validate_specs(
        args,
        configuration_spaces=setup_configuration_spaces_from_args(args),
    )


def resolve_composable_study(args: Any) -> ComposableStudy:
    """Resolve all composable method-level ownership in one validated object."""

    configuration_spaces = setup_configuration_spaces_from_args(args)
    specs = _validate_specs(
        args,
        configuration_spaces=configuration_spaces,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    return ComposableStudy(
        specs=specs,
        specs_by_name=specs_by_name,
        methods=tuple(spec.name for spec in specs),
        family_by_method={
            spec.name: spec.family
            for spec in specs
        },
        setup_configuration_spaces=configuration_spaces,
        bandit_methods=tuple(
            spec.name
            for spec in specs
            if spec.setup_kind in SETUP_BANDIT_KINDS
        ),
    )


def build_named_setup_branches(
    args: Any,
    *,
    study: ComposableStudy,
    warmup_instances: Sequence[tuple[Mapping[str, Any], np.ndarray]],
    stream_hash: str,
) -> ComposableSetupArtifacts:
    """Build independent setup learners for named composable spaces."""

    if not study.setup_configuration_spaces:
        raise ValueError("Named setup branch construction requires setup spaces")
    if warmup_instances or bool(args.reuse_warmup):
        raise ValueError(
            "Per-branch setup configuration spaces currently require the "
            "joint-from-scratch 0+4K protocol"
        )

    warmup_summary = _empty_stream_summary()
    warmup_path = args.output_dir / "warmup_trajectory.jsonl"
    warmup_path.write_text("", encoding="utf-8")
    _write_json(
        args.output_dir / "warmup_progress.json",
        {"completed_instances": 0, "summary": warmup_summary},
    )

    branches: Dict[str, Any] = {}
    warmup_state: Dict[str, str] = {}
    warmup_state_dir = args.output_dir / "bandit_warmup_states"
    warmup_state_dir.mkdir(parents=True, exist_ok=True)
    aot_enabled = str(args.setup_candidate_mode) == "aot"
    for method in study.bandit_methods:
        method_spec = study.specs_by_name[method]
        configuration_space = study.setup_configuration_spaces[
            str(method_spec.setup_space)
        ]
        branch, _bandit_config = build_online_linucb_branch(
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
                "stream_hash": str(stream_hash),
                "summary": warmup_summary,
                "method": method,
                "setup_space": configuration_space.name,
            },
        )
        warmup_state[method] = str(state_path)

    return ComposableSetupArtifacts(
        branches=branches,
        warmup_rows=(),
        warmup_summary=warmup_summary,
        warmup_trajectory=str(warmup_path),
        warmup_state=warmup_state,
        aot_enabled=aot_enabled,
    )


def validate_setup_branch_independence(
    branches: Mapping[str, Any],
    *,
    named_setup_spaces: bool,
    setup_candidate_mode: str,
) -> None:
    """Validate mutable-state isolation without changing learner behavior."""

    branch_values = tuple(branches.values())
    if len({id(branch.policy) for branch in branch_values}) != len(branch_values):
        raise RuntimeError("Bandit branches do not have independent policy objects")
    branch_models = [
        getattr(branch.policy, "model", None)
        for branch in branch_values
    ]
    if any(model is None for model in branch_models):
        raise RuntimeError("The locked protocol requires model-backed LinUCB branches")
    if len({id(model.A_inv) for model in branch_models}) != len(branch_models):
        raise RuntimeError("LinUCB mutable parameter matrices are not independent")
    if named_setup_spaces:
        if len({id(model.actions) for model in branch_models}) != len(branch_models):
            raise RuntimeError(
                "Named setup spaces must own distinct action catalogs"
            )
        if str(setup_candidate_mode) == "aot":
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


def make_frozen_ppo_runner(args: Any) -> Any:
    """Compatibility wrapper around the solve-owned PPO factory."""

    return build_frozen_ppo_runner(
        FrozenPpoConfig.from_runtime(args)
    )


def build_composable_solve_runtime(
    args: Any,
    *,
    specs: Sequence[ComposableMethodSpec],
) -> ComposableSolveRuntime:
    """Build each selected online solve controller exactly once."""

    controller_bundles: Dict[str, ControllerBundle] = {}
    has_typed_controller_specs = hasattr(args, "solve_controller_specs")
    typed_controller_specs = dict(
        getattr(args, "solve_controller_specs", {}) or {}
    )
    if has_typed_controller_specs:
        unknown_typed_kinds = (
            set(typed_controller_specs) - set(ONLINE_SOLVE_KINDS)
        )
        if unknown_typed_kinds:
            raise ValueError(
                "Unknown typed solve-controller kinds: "
                f"{sorted(unknown_typed_kinds)}"
            )
    for method_spec in specs:
        solve_kind = method_spec.solve_kind
        if solve_kind in {"default", "fixed", "ppo"}:
            continue
        if solve_kind not in ONLINE_SOLVE_KINDS:
            raise ValueError(f"Unknown online solve kind: {solve_kind!r}")
        controller_seed = int(
            args.controller_seed
            + _CONTROLLER_SEED_OFFSETS[solve_kind]
        )
        if has_typed_controller_specs:
            typed_controller_spec = typed_controller_specs.get(solve_kind)
            if typed_controller_spec is None:
                raise ValueError(
                    "Missing typed solve-controller spec for "
                    f"{solve_kind!r}"
                )
            if typed_controller_spec.kind != solve_kind:
                raise ValueError(
                    "Solve-controller spec key/kind mismatch: "
                    f"{solve_kind!r} maps to "
                    f"{typed_controller_spec.kind!r}"
                )
            bundle = build_online_solve_controller(
                typed_controller_spec,
                setup_obs_encoder=make_setup_obs_encoder(),
                seed=controller_seed,
            )
        else:
            compatibility_factory = _COMPATIBILITY_BUNDLE_FACTORIES.get(
                solve_kind
            )
            if compatibility_factory is None:
                raise ValueError(
                    f"No compatibility factory for {solve_kind!r}"
                )
            factory_kwargs: Dict[str, Any] = {
                "seed": controller_seed,
            }
            if solve_kind == "stagewise_lsvi":
                factory_kwargs["refit_interval_episodes"] = int(
                    args.lsvi_refit_interval_episodes
                )
            bundle = compatibility_factory(args, **factory_kwargs)
        controller_bundles[method_spec.name] = bundle

    ppo_runner = (
        make_frozen_ppo_runner(args)
        if any(spec.solve_kind == "ppo" for spec in specs)
        else None
    )
    return ComposableSolveRuntime(
        controller_bundles=controller_bundles,
        ppo_runner=ppo_runner,
    )


__all__ = [
    "ComposableSetupArtifacts",
    "ComposableSolveRuntime",
    "ComposableStudy",
    "FrozenPpoConfig",
    "SETUP_BANDIT_KINDS",
    "build_composable_solve_runtime",
    "build_named_setup_branches",
    "make_frozen_ppo_runner",
    "parse_composable_method",
    "resolve_composable_study",
    "setup_configuration_spaces_from_args",
    "validate_composable_protocol",
    "validate_setup_branch_independence",
]
