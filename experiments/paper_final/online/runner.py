"""Orchestrate one prespecified official joint-online comparison."""

from __future__ import annotations
from typing import Sequence
from solve.controllers.common import ControllerBundle
from .method_spec import ComposableMethodSpec
from typing import Any, Dict
from .io import _write_json
from .methods import (
    SETUP_BANDIT_KINDS,
    build_composable_solve_runtime,
    build_named_setup_branches,
    resolve_composable_study,
    validate_setup_branch_independence,
)
from .execution import (
    OnlineComparisonHooks,
    OnlineComparisonPlan,
    SolveExecutionConfig,
    WarmupArtifacts,
    make_method_solver,
    run_default_setup_method,
    run_online_comparison,
)
from .configuration import JointExperimentRuntimeConfig
from .case_loop import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _git_revision,
    _report_online_outcome,
)
from .protocol import build_joint_protocol
from .setup_branches import default_test_final_bandit_config_from_env
from problems.registry import context_for_setup_method

RunnerConfig = JointExperimentRuntimeConfig


def _method_solver(
    method: str,
    *,
    args: RunnerConfig,
    mkw: Dict[str, Any],
    case_progress: float,
    case_index: int = 0,
    problem_context: Sequence[float] | None = None,
    controller_bundles: Dict[str, ControllerBundle],
    composable_specs: Dict[str, ComposableMethodSpec] | None = None,
):
    return make_method_solver(
        method,
        solve=SolveExecutionConfig(
            tolerance=float(args.tol),
            max_cycles=int(args.max_cycles),
            failure_penalty_sec=getattr(args, "failure_penalty_sec", None),
        ),
        mkw=mkw,
        case_progress=case_progress,
        case_index=int(case_index),
        problem_context=problem_context,
        controller_bundles=controller_bundles,
        composable_specs=composable_specs,
    )


def _run_default_setup_method(
    *,
    spec: ComposableMethodSpec,
    solver_fn: Any,
    args: RunnerConfig,
    mkw: Dict[str, Any],
    controller_methods: Sequence[str],
    report_online_outcome: Any | None = None,
) -> Dict[str, Any]:
    """Execute one default-setup branch under the active recovery protocol."""
    return run_default_setup_method(
        spec=spec,
        solver_fn=solver_fn,
        solve=SolveExecutionConfig(
            tolerance=float(args.tol),
            max_cycles=int(args.max_cycles),
            failure_penalty_sec=getattr(args, "failure_penalty_sec", None),
        ),
        mkw=mkw,
        controller_methods=controller_methods,
        report_online_outcome=_report_online_outcome
        if report_online_outcome is None
        else report_online_outcome,
    )


def run(args: RunnerConfig) -> Dict[str, Any]:
    study = resolve_composable_study(args)
    composable_specs_tuple = study.specs
    setup_configuration_spaces = dict(study.setup_configuration_spaces)
    composable_specs = dict(study.specs_by_name)
    methods = study.methods
    family_by_method = dict(study.family_by_method)
    bandit_methods = study.bandit_methods
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite experiment output: {args.output_dir}"
        )
    if args.warmup_cases or int(args.online_cases) != int(args.train_cases):
        raise ValueError("The official stream includes every problem in online cost")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    checkpoints_dir = args.output_dir / "checkpoints"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _configure_paired_environment(args)
    (full_stream, stream_manifest) = _build_paired_instance_stream(args)
    if len(full_stream) != int(args.train_cases):
        raise ValueError("Generated stream length does not match train_cases")
    expected_stream_hash = str(args.expected_stream_hash or "").strip()
    if expected_stream_hash and str(stream_manifest["sha256"]) != expected_stream_hash:
        raise ValueError(
            f"Generated stream does not match expected hash: {stream_manifest['sha256']} != {expected_stream_hash}"
        )
    warmup_instances = full_stream[: int(args.warmup_cases)]
    online_instances = full_stream[int(args.warmup_cases) :]
    _write_json(args.output_dir / "stream_manifest.json", stream_manifest)
    setup_artifacts = build_named_setup_branches(
        args,
        study=study,
        warmup_instances=warmup_instances,
        stream_hash=str(stream_manifest["sha256"]),
    )
    branches = dict(setup_artifacts.branches)
    warmup_rows = list(setup_artifacts.warmup_rows)
    warmup_summary = dict(setup_artifacts.warmup_summary)
    warmup_trajectory_artifact = setup_artifacts.warmup_trajectory
    warmup_state_artifact = dict(setup_artifacts.warmup_state)
    aot_enabled = bool(setup_artifacts.aot_enabled)
    validate_setup_branch_independence(
        branches,
        named_setup_spaces=bool(setup_configuration_spaces),
        setup_candidate_mode=str(args.setup_candidate_mode),
    )
    solve_runtime = build_composable_solve_runtime(args, specs=composable_specs_tuple)
    controller_bundles = dict(solve_runtime.controller_bundles)
    protocol = build_joint_protocol(
        args,
        git_revision=_git_revision(),
        stream_manifest=stream_manifest,
        methods=methods,
        family_by_method=family_by_method,
        composable_specs=composable_specs_tuple,
        candidates=(),
        setup_configuration_spaces=setup_configuration_spaces,
        bandit_methods=bandit_methods,
        setup_bandit_kinds=SETUP_BANDIT_KINDS,
        solve_controller_protocols={
            method: bundle.protocol_metadata()
            for (method, bundle) in controller_bundles.items()
        },
    )
    _write_json(args.output_dir / "config.json", protocol)
    default_test_final_bandit_config_from_env()
    has_solve_screen_report = bool(controller_bundles)
    plan = OnlineComparisonPlan(
        output_dir=args.output_dir,
        trajectories_dir=trajectories_dir,
        checkpoints_dir=checkpoints_dir,
        methods=tuple(methods),
        family_by_method=family_by_method,
        bandit_methods=tuple(bandit_methods),
        online_instances=online_instances,
        composable_specs=composable_specs,
        branches=branches,
        controller_bundles=controller_bundles,
        protocol=protocol,
        warmup=WarmupArtifacts(
            summary=warmup_summary,
            trajectory=warmup_trajectory_artifact,
            state=warmup_state_artifact,
            record_count=len(warmup_rows),
        ),
        solve=SolveExecutionConfig(
            tolerance=float(args.tol),
            max_cycles=int(args.max_cycles),
            failure_penalty_sec=getattr(args, "failure_penalty_sec", None),
        ),
        warmup_cases=int(args.warmup_cases),
        online_cases=int(args.online_cases),
        method_order_seed=int(args.method_order_seed),
        progress_every=int(args.progress_every),
        aot_enabled=bool(aot_enabled),
        aot_max_selections_per_case=int(args.aot_max_selections_per_case),
        include_solve_screen_report=has_solve_screen_report,
    )
    hooks = OnlineComparisonHooks(
        method_solver=lambda method, **kwargs: _method_solver(
            method,
            args=args,
            controller_bundles=controller_bundles,
            composable_specs=composable_specs,
            **kwargs,
        ),
        run_default_setup_method=lambda **kwargs: _run_default_setup_method(
            args=args, **kwargs
        ),
        report_online_outcome=_report_online_outcome,
        setup_context=lambda method, mkw, context: context_for_setup_method(
            problem_kind=getattr(args, "problem", "scalar_anisotropic_diffusion"),
            setup_kind=composable_specs[method].setup_kind
            if method in composable_specs
            else "linucb",
            matrix_kwargs=mkw,
            stream_context=context,
            grid_norm_div=float(args.grid_n),
            setup_context=composable_specs[method].setup_context
            if method in composable_specs
            else "default",
        ),
    )
    return run_online_comparison(plan, hooks=hooks)
