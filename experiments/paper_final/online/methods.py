"""Composable setup/solve assembly for the canonical 4K experiment.

This module owns the composable method roster and the construction of its
setup learners and solve controllers.  The canonical runner remains
responsible for the experiment stream, retry/fallback protocol, and artifacts;
the mode-neutral online loop remains in :mod:`execution`.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from experiments.paper_final.online.controllers import make_setup_obs_encoder
from experiments.paper_final.online.method_spec import ComposableMethodSpec
from experiments.paper_final.online.artifacts import _write_json_line
from experiments.paper_final.online.case_loop import _method_stream_summary
from experiments.paper_final.online.reporting import _empty_stream_summary
from experiments.paper_final.online.io import _write_json
from problems.amg import COMPACT_DIFFUSION_CONTEXT_INDICES
from problems.registry import learning_context_for_setup
from setup.registry import ONLINE_SETUP_KINDS
from setup.space import SetupConfigurationSpace
from experiments.paper_final.online.setup_branches import build_online_linucb_branch
from solve.controllers.common import ControllerBundle
from solve.registry import ONLINE_SOLVE_KINDS, build_online_solve_controller

SETUP_BANDIT_KINDS = ONLINE_SETUP_KINDS
_CONTROLLER_SEED_OFFSETS = {"recursive_lstdq": 2018}


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
    """Built online solve controllers for the selected official methods."""

    controller_bundles: Mapping[str, ControllerBundle]


def setup_configuration_spaces_from_args(
    args: Any,
) -> Dict[str, SetupConfigurationSpace]:
    """Return already-decoded named setup spaces after identity validation."""
    raw_spaces = dict(getattr(args, "setup_configuration_spaces", {}) or {})
    spaces: Dict[str, SetupConfigurationSpace] = {}
    for name, raw_space in raw_spaces.items():
        if not isinstance(raw_space, SetupConfigurationSpace):
            raise TypeError(
                f"setup configuration space {name!r} was not normalized by run.py"
            )
        if str(name) != raw_space.name:
            raise ValueError(
                f"Setup configuration-space key {name!r} does not match its name {raw_space.name!r}"
            )
        spaces[str(name)] = raw_space
    return spaces


def _validate_specs(
    args: Any, *, configuration_spaces: Mapping[str, SetupConfigurationSpace]
) -> tuple[ComposableMethodSpec, ...]:
    if args.setup_candidate_mode != "aot":
        raise ValueError("The official study requires an AOT candidate schedule")
    specs = tuple(args.methods)
    if not all((isinstance(spec, ComposableMethodSpec) for spec in specs)):
        raise TypeError("Methods must be decoded official method specs")
    if not specs:
        raise ValueError("Composable experiments require at least one --method")
    names = tuple((spec.name for spec in specs))
    if len(set(names)) != len(names):
        raise ValueError("Composable method names must be unique")
    referenced_spaces = {
        str(spec.setup_space) for spec in specs if spec.setup_space is not None
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
            if spec.setup_kind in SETUP_BANDIT_KINDS and spec.setup_space is None
        ]
        if missing_references:
            raise ValueError(
                f"Every setup-bandit method must select setup_space when named configuration spaces are present: {missing_references}"
            )
        unused_spaces = set(configuration_spaces) - referenced_spaces
        if unused_spaces:
            raise ValueError(
                f"Unused setup configuration spaces: {sorted(unused_spaces)}"
            )
        if str(args.setup_action_space) != "full_cartesian":
            raise ValueError(
                "Named setup configuration spaces require setup.action_space='full_cartesian'"
            )
    candidate_mode = (
        str(getattr(args, "setup_candidate_mode", "explicit")).strip().lower()
    )
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
        (spec.candidate_sampling == "structured512" for spec in specs)
    ):
        raise ValueError("structured512 candidate sampling requires AOT mode")
    invalid_warmups = {
        spec.name: int(spec.setup_warmup_cases)
        for spec in specs
        if not 0 <= int(spec.setup_warmup_cases) <= int(args.warmup_cases)
    }
    if invalid_warmups:
        raise ValueError(
            f"Method setup_warmup_cases must lie within the shared warmup prefix [0, {int(args.warmup_cases)}]: {invalid_warmups}"
        )
    staged_methods = {
        spec.name: int(spec.solve_activation_case)
        for spec in specs
        if int(spec.solve_activation_case) > 0
    }
    if int(args.warmup_cases) != 0 or any((spec.setup_warmup_cases for spec in specs)):
        raise ValueError("The official study has no separate setup warmup")
    if any((spec.solve_activation is not None for spec in specs)):
        raise ValueError("The official study uses fixed solve activation")
    compact_methods = any(
        (
            spec.setup_context in COMPACT_DIFFUSION_CONTEXT_INDICES
            or spec.solve_context in COMPACT_DIFFUSION_CONTEXT_INDICES
            for spec in specs
        )
    )
    if (
        compact_methods
        and getattr(args, "problem", "scalar_anisotropic_diffusion")
        != "scalar_anisotropic_diffusion"
    ):
        raise ValueError(
            "Compact diffusion contexts require scalar_anisotropic_diffusion"
        )
    invalid_activations = {
        name: activation
        for (name, activation) in staged_methods.items()
        if activation >= int(args.online_cases)
    }
    if invalid_activations:
        raise ValueError(
            f"Method solve_activation_case must lie within the online stream [1, {int(args.online_cases) - 1}]: {invalid_activations}"
        )
    if args.weights is None or args.action_rbf_centers is None:
        raise ValueError(
            "Composable experiments require explicit --weights and --action-rbf-centers"
        )
    weights = np.asarray(
        tuple(
            (float(part.strip()) for part in args.weights.split(",") if part.strip())
        ),
        dtype=float,
    )
    centers = np.asarray(
        tuple(
            (
                float(part.strip())
                for part in args.action_rbf_centers.split(",")
                if part.strip()
            )
        ),
        dtype=float,
    )
    for label, values in (("weights", weights), ("RBF centers", centers)):
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Composable {label} must be finite and non-empty")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Composable {label} must be strictly increasing")
    if float(args.action_rbf_sigma) <= 0.0:
        raise ValueError("action_rbf_sigma must be positive")
    return specs


def validate_composable_protocol(args: Any) -> tuple[ComposableMethodSpec, ...]:
    """Compatibility validator returning the ordered method specs."""
    return _validate_specs(
        args, configuration_spaces=setup_configuration_spaces_from_args(args)
    )


def resolve_composable_study(args: Any) -> ComposableStudy:
    """Resolve all composable method-level ownership in one validated object."""
    configuration_spaces = setup_configuration_spaces_from_args(args)
    specs = _validate_specs(args, configuration_spaces=configuration_spaces)
    specs_by_name = {spec.name: spec for spec in specs}
    return ComposableStudy(
        specs=specs,
        specs_by_name=specs_by_name,
        methods=tuple((spec.name for spec in specs)),
        family_by_method={spec.name: spec.family for spec in specs},
        setup_configuration_spaces=configuration_spaces,
        bandit_methods=tuple(
            (spec.name for spec in specs if spec.setup_kind in SETUP_BANDIT_KINDS)
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
    if warmup_instances or any((spec.setup_warmup_cases for spec in study.specs)):
        raise ValueError("The official study has no separate setup warmup")
    if not study.setup_configuration_spaces:
        raise ValueError("Named setup branch construction requires setup spaces")
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
        (branch, _bandit_config) = build_online_linucb_branch(
            seed=int(args.bandit_seed) + seed_offset,
            learner_kind=str(method_spec.setup_kind),
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode=str(args.setup_action_space),
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            parameter_resolution=int(args.setup_param_resolution),
            configuration_space=configuration_space,
            candidate_schedule_dir=candidate_schedule_dir if aot_enabled else None,
            candidate_schedule_rounds=(len(warmup_instances) + int(args.online_cases))
            * int(args.aot_max_selections_per_case)
            if aot_enabled
            else None,
            candidate_schedule_chunk_rounds=int(args.aot_schedule_chunk_rounds),
            candidate_sampling=str(method_spec.candidate_sampling),
            context_dim=int(learning_context.dimension),
            context_interaction_indices=learning_context.interaction_indices,
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
    evaluation_cursor = len(warmup_instances) * int(args.aot_max_selections_per_case)
    for group_index, methods in enumerate(grouped_methods.values()):
        source_method = methods[0]
        source_spec = study.specs_by_name[source_method]
        source_branch = branches[source_method]
        depths = sorted({requested_depths[method] for method in methods})
        max_depth = depths[-1]
        checkpoints = {
            depth: checkpoint_root / f"group_{group_index}_warmup_{depth}.npz"
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
            pass
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
    branches: Mapping[str, Any], *, named_setup_spaces: bool, setup_candidate_mode: str
) -> None:
    """Validate mutable-state isolation without changing learner behavior."""
    branch_values = tuple(branches.values())
    if len({id(branch.policy) for branch in branch_values}) != len(branch_values):
        raise RuntimeError("Bandit branches do not have independent policy objects")
    branch_models = [getattr(branch.policy, "model", None) for branch in branch_values]
    if any((model is None for model in branch_models)):
        raise RuntimeError("The locked protocol requires model-backed LinUCB branches")
    if len({id(model.A_inv) for model in branch_models}) != len(branch_models):
        raise RuntimeError("LinUCB mutable parameter matrices are not independent")
    if named_setup_spaces:
        if len({id(model.actions) for model in branch_models}) != len(branch_models):
            raise RuntimeError("Named setup spaces must own distinct action catalogs")
        if str(setup_candidate_mode) == "aot":
            if any((model._g_actions is not None for model in branch_models)):
                raise RuntimeError(
                    "AOT LinUCB branches must not materialize _g_actions"
                )
            if any(
                (
                    model._candidate_schedule is None
                    or model._action_feature_cache is None
                    for model in branch_models
                )
            ):
                raise RuntimeError(
                    "AOT branches require schedules and factorized caches"
                )
        elif any((model._g_actions is None for model in branch_models)):
            raise RuntimeError("Explicit branches require cached _g_actions")
    elif branch_models and len({id(model._g_actions) for model in branch_models}) != 1:
        raise RuntimeError("LinUCB branches must share the immutable _g_actions cache")
    if any(
        (
            model._g_actions is not None and model._g_actions.flags.writeable
            for model in branch_models
        )
    ):
        raise RuntimeError("Cached LinUCB action features must be read-only")


def build_composable_solve_runtime(
    args: Any, *, specs: Sequence[ComposableMethodSpec]
) -> ComposableSolveRuntime:
    """Build each selected online solve controller exactly once."""
    controller_bundles: Dict[str, ControllerBundle] = {}
    has_typed_controller_specs = hasattr(args, "solve_controller_specs")
    typed_controller_specs = dict(getattr(args, "solve_controller_specs", {}) or {})
    if has_typed_controller_specs:
        unknown_typed_kinds = set(typed_controller_specs) - set(ONLINE_SOLVE_KINDS)
        if unknown_typed_kinds:
            raise ValueError(
                f"Unknown typed solve-controller kinds: {sorted(unknown_typed_kinds)}"
            )
    for method_spec in specs:
        solve_kind = method_spec.solve_kind
        if solve_kind in {"default"}:
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
                    f"Missing typed solve-controller spec for {solve_kind!r}"
                )
            if typed_controller_spec.kind != solve_kind:
                raise ValueError(
                    f"Solve-controller spec key/kind mismatch: {solve_kind!r} maps to {typed_controller_spec.kind!r}"
                )
            resolved_controller_spec = typed_controller_spec
            state_overrides: Dict[str, Any] = {}
            if method_spec.solve_tolerance is not None:
                state_overrides["tol"] = float(method_spec.solve_tolerance)
            typed_state = getattr(typed_controller_spec, "state", None)
            typed_context_mode = getattr(
                typed_state, "problem_context_mode", "canonical"
            )
            if method_spec.solve_context != typed_context_mode:
                state_overrides["problem_context_mode"] = method_spec.solve_context
            if state_overrides:
                if typed_state is None:
                    raise TypeError(
                        "Per-method solve-state overrides require a typed controller spec with a state"
                    )
                resolved_controller_spec = replace(
                    typed_controller_spec, state=replace(typed_state, **state_overrides)
                )
            if (
                getattr(typed_state, "encoding_version", "legacy_v1")
                == "space_aware_v2"
            ):
                spaces = setup_configuration_spaces_from_args(args)
                space = (
                    spaces[method_spec.setup_space] if method_spec.setup_space else None
                )
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
            raise TypeError(
                "The official runner requires decoded controller specifications"
            )
        controller_bundles[method_spec.name] = bundle
    return ComposableSolveRuntime(controller_bundles=controller_bundles)
