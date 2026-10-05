"""Composable setup/solve assembly for the canonical 4K experiment.

This module owns the composable method roster and the construction of its
setup learners and solve controllers.  The canonical runner remains
responsible for the experiment stream, retry/fallback protocol, and artifacts;
the mode-neutral online loop remains in :mod:`joint_4k_execution`.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from experiments.joint.solve_control.joint_controller_build import (
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
from experiments.joint.solve_control.joint_method_spec import ComposableMethodSpec
from experiments.joint.solve_control.joint_artifacts import _write_json_line
from experiments.joint.solve_control.joint_online_common import (
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
)
from experiments.joint.solve_control.joint_reporting import _empty_stream_summary
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from problems.amg import COMPACT_DIFFUSION_CONTEXT_INDICES
from problems.registry import (
    context_for_setup_method,
    learning_context_for_setup,
)
from setup.registry import ONLINE_SETUP_KINDS
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from hypre.bindings import augment_setup_params
from experiments.joint.solve_control.setup_branches import build_online_linucb_branch, run_bandit_step_test_final
from experiments.joint.solve_control.native_evaluation import solve_no_rl_case
from experiments.joint.solve_control.feedback import as_feedback
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
    """Per-method setup learners and resolved warmup artifacts."""

    branches: Mapping[str, Any]
    warmup_rows: tuple[Mapping[str, Any], ...]
    warmup_summary: Mapping[str, Any]
    warmup_trajectory: Any
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
    invalid_warmups = {
        spec.name: int(spec.setup_warmup_cases)
        for spec in specs
        if not 0 <= int(spec.setup_warmup_cases) <= int(args.warmup_cases)
    }
    if invalid_warmups:
        raise ValueError(
            "Method setup_warmup_cases must lie within the shared warmup "
            f"prefix [0, {int(args.warmup_cases)}]: {invalid_warmups}"
        )
    staged_methods = {
        spec.name: int(spec.solve_activation_case)
        for spec in specs
        if int(spec.solve_activation_case) > 0
    }
    dynamic_methods = [spec for spec in specs if spec.solve_activation is not None]
    if (staged_methods or dynamic_methods) and int(args.warmup_cases) != 0:
        raise ValueError(
            "solve_activation_case is an inclusive online-stream boundary "
            "and therefore requires stream.warmup_cases=0"
        )
    if dynamic_methods and getattr(args, "setup_replay_trajectory", None) is not None:
        raise ValueError(
            "Dynamic activation requires online setup learning, not setup replay"
        )
    for spec in dynamic_methods:
        horizon = spec.solve_activation.horizon
        if horizon is not None and horizon != int(args.online_cases) - 1:
            raise ValueError("Composite activation horizon must equal online_cases - 1")
    if getattr(args, "shared_online_prefix", False):
        if dynamic_methods:
            if len(specs) != 2 or len(staged_methods) != 1 or len(dynamic_methods) != 1:
                raise ValueError("Dynamic shared prefix requires one fixed-start and one dynamic-start method")
            fixed = next(spec for spec in specs if spec.solve_activation_case)
            dynamic = dynamic_methods[0]
            if fixed.setup_kind != "linucb" or replace(
                fixed, name=dynamic.name, solve_activation_case=0,
                solve_activation=dynamic.solve_activation,
            ) != dynamic:
                raise ValueError("Shared online prefix requires matching LinUCB methods and seeds apart from activation")
        else:
            references = [spec for spec in specs if spec.solve_kind == "default"]
            if len(references) > 1 or len(staged_methods) != len(specs) - len(references) or not staged_methods:
                raise ValueError("Nested shared prefix requires staged RL branches and at most one setup-only reference")
            staged = [spec for spec in specs if spec.solve_kind != "default"]
            reference = references[0] if references else staged[0]
            if reference.setup_kind != "linucb" or any(
                spec.solve_kind != "recursive_lstdq_v3" or replace(
                    spec, name=reference.name, solve_kind=reference.solve_kind,
                    solve_context=reference.solve_context, solve_activation_case=reference.solve_activation_case,
                    solve_tolerance=reference.solve_tolerance,
                ) != reference for spec in staged
            ):
                raise ValueError("Nested shared prefix requires matching LinUCB methods and seeds apart from solve activation")
            if any(replace(spec, name=staged[0].name,
                           solve_activation_case=staged[0].solve_activation_case) != staged[0]
                   for spec in staged[1:]):
                raise ValueError("Nested RL branches must differ only in name and activation boundary")
    compact_methods = any(
        spec.setup_context in COMPACT_DIFFUSION_CONTEXT_INDICES
        or spec.solve_context in COMPACT_DIFFUSION_CONTEXT_INDICES
        for spec in specs
    )
    if compact_methods and (
        getattr(args, "problem", "scalar_anisotropic_diffusion")
        != "scalar_anisotropic_diffusion"
    ):
        raise ValueError(
            "Compact diffusion contexts require scalar_anisotropic_diffusion"
        )
    invalid_activations = {
        name: activation
        for name, activation in staged_methods.items()
        if activation >= int(args.online_cases)
    }
    if invalid_activations:
        raise ValueError(
            "Method solve_activation_case must lie within the online stream "
            f"[1, {int(args.online_cases) - 1}]: {invalid_activations}"
        )

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
    if bool(args.reuse_warmup):
        raise ValueError(
            "Named per-method setup warmups do not support --reuse-warmup"
        )

    warmup_path = args.output_dir / "warmup_trajectory.jsonl"
    warmup_path.write_text("", encoding="utf-8")

    branches: Dict[str, Any] = {}
    warmup_state: Dict[str, str] = {}
    warmup_state_dir = args.output_dir / "bandit_warmup_states"
    warmup_state_dir.mkdir(parents=True, exist_ok=True)
    aot_enabled = str(args.setup_candidate_mode) == "aot"
    for method in study.bandit_methods:
        method_spec = study.specs_by_name[method]
        learning_context = learning_context_for_setup(
            getattr(args, "problem", "scalar_anisotropic_diffusion"),
            method_spec.setup_kind,
            method_spec.setup_context,
        )
        seed_offset = int(method_spec.seed_offset)
        configuration_space = study.setup_configuration_spaces[
            str(method_spec.setup_space)
        ]
        candidate_schedule_dir = (
            args.output_dir
            / "aot_candidate_schedules"
            / configuration_space.name
            / method_spec.candidate_sampling
        )
        if seed_offset:
            candidate_schedule_dir = (
                candidate_schedule_dir / f"seed_offset_{seed_offset}"
            )
        branch, _bandit_config = build_online_linucb_branch(
            seed=int(args.bandit_seed) + seed_offset,
            learner_kind=str(method_spec.setup_kind),
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode=str(args.setup_action_space),
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            parameter_resolution=int(args.setup_param_resolution),
            configuration_space=configuration_space,
            candidate_schedule_dir=(
                candidate_schedule_dir
                if aot_enabled
                else None
            ),
            candidate_schedule_rounds=(
                (len(warmup_instances) + int(args.online_cases))
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
            context_dim=int(learning_context.dimension),
            context_interaction_indices=(
                learning_context.interaction_indices
            ),
        )
        branches[method] = branch

    requested_depths = {
        method: int(study.specs_by_name[method].setup_warmup_cases)
        for method in study.bandit_methods
    }
    grouped_methods: Dict[tuple[str, str, str, str, int], list[str]] = {}
    for method in study.bandit_methods:
        spec = study.specs_by_name[method]
        key = (
            str(spec.setup_kind),
            str(spec.setup_space),
            str(spec.candidate_sampling),
            str(spec.setup_context),
            int(spec.seed_offset),
        )
        grouped_methods.setdefault(key, []).append(method)

    combined_rows: list[Mapping[str, Any]] = []
    method_summaries: Dict[str, Mapping[str, Any]] = {}
    method_trajectories: Dict[str, str] = {}
    checkpoint_root = args.output_dir / "bandit_warmup_checkpoints"
    trajectory_root = args.output_dir / "bandit_warmup_trajectories"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trajectory_root.mkdir(parents=True, exist_ok=True)
    evaluation_cursor = len(warmup_instances) * int(
        args.aot_max_selections_per_case
    )

    for group_index, methods in enumerate(grouped_methods.values()):
        source_method = methods[0]
        source_spec = study.specs_by_name[source_method]
        source_branch = branches[source_method]
        depths = sorted({requested_depths[method] for method in methods})
        max_depth = depths[-1]
        checkpoints = {
            depth: checkpoint_root
            / f"group_{group_index}_warmup_{depth}.npz"
            for depth in depths
        }
        if 0 in checkpoints:
            source_branch.policy.model.save_mutable_state(
                checkpoints[0],
                metadata={
                    "stream_hash": str(stream_hash),
                    "warmup_cases": 0,
                    "source_method": source_method,
                },
            )

        rows: list[Mapping[str, Any]] = []
        trajectory_path = trajectory_root / f"group_{group_index}.jsonl"
        previous_update = 0.0
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for case_index, (mkw, context) in enumerate(
                warmup_instances[:max_depth]
            ):
                learner_context = context_for_setup_method(
                    problem_kind=getattr(
                        args,
                        "problem",
                        "scalar_anisotropic_diffusion",
                    ),
                    setup_kind=source_spec.setup_kind,
                    matrix_kwargs=mkw,
                    stream_context=context,
                    grid_norm_div=getattr(args, "grid_n", None),
                    setup_context=source_spec.setup_context,
                )

                def solve_selected(params: Dict[str, Any]) -> Dict[str, Any]:
                    return as_feedback(
                        solve_no_rl_case(
                            params=dict(params),
                            mkw=dict(mkw),
                            solver_tol=float(args.tol),
                            solver_max_iter=int(args.max_cycles),
                            augment_params=augment_setup_params,
                        ),
                        include_controller=False,
                    )

                def solve_fallback(_params: Dict[str, Any]) -> Dict[str, Any]:
                    return as_feedback(
                        solve_no_rl_case(
                            params=dict(DEFAULT_SETUP_PARAMS),
                            mkw=dict(mkw),
                            solver_tol=float(args.tol),
                            solver_max_iter=int(args.max_cycles),
                            augment_params=augment_setup_params,
                        ),
                        include_controller=False,
                    )

                params, native, timing, fallback_used, update_sec = (
                    run_bandit_step_test_final(
                        policy=source_branch.policy,
                        parameter_space=source_branch.parameter_space,
                        problem_context=learner_context,
                        solver_fn=solve_selected,
                        fallback_solver_fn=solve_fallback,
                        prev_update_est=float(previous_update),
                    )
                )
                previous_update = float(update_sec)
                if aot_enabled:
                    source_branch.policy.model.finish_candidate_schedule_case(
                        max_selections=int(args.aot_max_selections_per_case)
                    )
                row = {
                    "warmup_index": int(case_index),
                    "source_method": source_method,
                    "mkw": dict(mkw),
                    "context": learner_context.tolist(),
                    "params": dict(params),
                    "arm_index": int(native.get("selected_arm_index", -1)),
                    "fallback_used": int(fallback_used),
                    "bandit_timing": dict(timing),
                    "outcome": _report_online_outcome(
                        native, bandit_timing=timing
                    ),
                }
                rows.append(row)
                _write_json_line(trajectory, dict(row))
                done = case_index + 1
                if done in checkpoints:
                    source_branch.policy.model.save_mutable_state(
                        checkpoints[done],
                        metadata={
                            "stream_hash": str(stream_hash),
                            "warmup_cases": int(done),
                            "source_method": source_method,
                            "summary": _method_stream_summary(rows),
                        },
                    )
                if (
                    done % max(1, int(args.progress_every)) == 0
                    or done == max_depth
                ):
                    trajectory.flush()
                    print(
                        json.dumps(
                            {
                                "stage": "setup_bandit_warmup",
                                "group": int(group_index),
                                "done": int(done),
                                "total": int(max_depth),
                            }
                        ),
                        flush=True,
                    )

        combined_rows.extend(rows)
        for method in methods:
            depth = requested_depths[method]
            model = branches[method].policy.model
            model.load_mutable_state(checkpoints[depth])
            if aot_enabled:
                model.set_candidate_schedule_cursor(evaluation_cursor)
            summary = (
                _empty_stream_summary()
                if depth == 0
                else _method_stream_summary(rows[:depth])
            )
            method_summaries[method] = summary
            method_trajectories[method] = str(trajectory_path)
            state_path = warmup_state_dir / f"{method}.npz"
            model.save_mutable_state(
                state_path,
                metadata={
                    "stream_hash": str(stream_hash),
                    "summary": summary,
                    "method": method,
                    "setup_space": study.specs_by_name[method].setup_space,
                    "setup_context": study.specs_by_name[method].setup_context,
                    "warmup_cases": int(depth),
                    "evaluation_schedule_cursor": int(evaluation_cursor),
                },
            )
            warmup_state[method] = str(state_path)

    with warmup_path.open("w", encoding="utf-8") as combined:
        for row in combined_rows:
            _write_json_line(combined, dict(row))
    warmup_summary: Mapping[str, Any] = (
        _empty_stream_summary()
        if not any(requested_depths.values())
        else method_summaries
    )
    _write_json(
        args.output_dir / "warmup_progress.json",
        {
            "completed_native_instances": len(combined_rows),
            "method_warmup_cases": requested_depths,
            "summary": warmup_summary,
            "trajectories": method_trajectories,
        },
    )

    return ComposableSetupArtifacts(
        branches=branches,
        warmup_rows=tuple(combined_rows),
        warmup_summary=warmup_summary,
        warmup_trajectory={
            "combined": str(warmup_path),
            "by_method": method_trajectories,
        },
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
            + int(method_spec.seed_offset)
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
            resolved_controller_spec = typed_controller_spec
            state_overrides: Dict[str, Any] = {}
            if method_spec.solve_tolerance is not None:
                state_overrides["tol"] = float(
                    method_spec.solve_tolerance
                )
            typed_state = getattr(
                typed_controller_spec,
                "state",
                None,
            )
            typed_context_mode = getattr(
                typed_state,
                "problem_context_mode",
                "canonical",
            )
            if method_spec.solve_context != typed_context_mode:
                state_overrides["problem_context_mode"] = (
                    method_spec.solve_context
                )
            if state_overrides:
                if typed_state is None:
                    raise TypeError(
                        "Per-method solve-state overrides require a typed "
                        "controller spec with a state"
                    )
                resolved_controller_spec = replace(
                    typed_controller_spec,
                    state=replace(
                        typed_state,
                        **state_overrides,
                    ),
                )
            if getattr(typed_state, "encoding_version", "legacy_v1") == "space_aware_v2":
                spaces = setup_configuration_spaces_from_args(args)
                space = spaces[method_spec.setup_space] if method_spec.setup_space else None
                setup_encoder = make_setup_obs_encoder(
                    configuration_space=space,
                    parameter_resolution=getattr(args, "setup_param_resolution", None),
                    strict_categories=True,
                )
            else:
                setup_encoder = make_setup_obs_encoder()
            bundle = build_online_solve_controller(
                resolved_controller_spec,
                setup_obs_encoder=setup_encoder,
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
            compatibility_args = args
            if (
                method_spec.solve_tolerance is not None
                or method_spec.solve_context
                != getattr(
                    args,
                    "solve_problem_context_mode",
                    "canonical",
                )
            ):
                compatibility_args = copy.copy(args)
            if method_spec.solve_tolerance is not None:
                compatibility_args.tol = float(method_spec.solve_tolerance)
            if method_spec.solve_context != getattr(
                args,
                "solve_problem_context_mode",
                "canonical",
            ):
                compatibility_args.solve_problem_context_mode = (
                    method_spec.solve_context
                )
            bundle = compatibility_factory(
                compatibility_args,
                **factory_kwargs,
            )
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
