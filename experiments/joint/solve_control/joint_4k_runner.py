"""Canonical 4K joint setup/solve experiment orchestrator."""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from setup.space import (
    DEFAULT_SETUP_PARAMS,
)
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from solve.controllers.common import (
    ControllerBundle,
)
from experiments.joint.solve_control.composable_joint_4k import (
    SETUP_BANDIT_KINDS,
    build_composable_solve_runtime,
    build_named_setup_branches,
    make_frozen_ppo_runner as _make_frozen_ppo_runner,
    parse_composable_method as _parse_composable_method,
    resolve_composable_study,
    setup_configuration_spaces_from_args as _setup_configuration_spaces_from_args,
    validate_composable_protocol as _validate_composable_protocol,
    validate_setup_branch_independence,
)
from experiments.joint.solve_control.joint_4k_cli import build_parser
from experiments.joint.solve_control.joint_4k_execution import (
    OnlineComparisonHooks,
    OnlineComparisonPlan,
    SolveExecutionConfig,
    WarmupArtifacts,
    make_method_solver,
    run_default_setup_method,
    run_online_comparison,
)
from experiments.joint.solve_control.joint_controller_build import (
    make_encoder as _make_encoder,
    make_lsvi_controller as _make_lsvi_controller,
    make_online_controller_from_args as _make_online_controller_from_args,
    make_recalibrated_lsvi_controller as _make_recalibrated_lsvi_controller,
    make_recursive_blstdq_controller as _make_recursive_blstdq_controller,
    make_recursive_lstdq_controller as _make_recursive_lstdq_controller,
    make_recursive_lstdq_v2_controller as _make_recursive_lstdq_v2_controller,
    make_recursive_lstdq_v3_controller as _make_recursive_lstdq_v3_controller,
    make_recursive_mc_controller as _make_recursive_mc_controller,
    make_setup_obs_encoder as _make_setup_obs_encoder,
    make_shared_action_config as _make_shared_action_config,
    make_shared_action_spec as _make_shared_action_spec,
    make_solve_state_spec as _make_solve_state_spec,
    make_structured_model_based_controller as _make_structured_model_based_controller,
    parse_csv_values as _parse_values,
)
from experiments.joint.solve_control.joint_experiment_config import JointExperimentRuntimeConfig
from hypre.bindings.recovery import validate_failure_penalty
from experiments.joint.solve_control.legacy_joint_studies import (
    BATCHED_LSVI_METHOD,
    BEHAVIOR_MODES,
    DEFAULT_SETUP_METHOD,
    LSVI_METHOD,
    LSVI_METHODS,
    RECALIBRATED_LSVI_METHOD,
    RECURSIVE_LCB_METHODS,
    RECURSIVE_LCB_PPO_METHODS,
    RECURSIVE_LSTDQ_METHOD,
    RECURSIVE_LSTDQ_METHODS,
    RECURSIVE_LSTDQ_V2_METHOD,
    RECURSIVE_MC_METHOD,
    REFERENCE_METHODS,
    SHARED_ACTION_PROFILES,
    SOLVE_CONTROLLER_SCREEN_METHODS,
    SOLVE_CONTROLLER_SEED_OFFSETS,
    STRUCTURED_MODEL_BASED_METHOD,
    SarsaCandidate,
    build_legacy_solve_runtime,
    candidate_grid,
    make_sarsa_controller as _make_controller,
    resolve_legacy_study,
    validate_lsvi_protocol as _validate_lsvi_protocol,
    validate_shared_lcb_protocol as _validate_shared_lcb_protocol,
)
from experiments.joint.solve_control.joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _git_revision,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from experiments.joint.solve_control.joint_method_spec import ComposableMethodSpec
from experiments.joint.solve_control.joint_artifacts import (
    _write_json_line,
    _write_solve_screen_reproduction,
)
from experiments.joint.solve_control.joint_reporting import (
    _action_summary,
    _comparison_windows,
    _empty_stream_summary,
    _window_result,
    _write_solve_screen_report,
    _write_summary_csv,
)
from experiments.joint.solve_control.joint_protocol import build_joint_protocol
from experiments.joint.solve_control.run_online_methods_2k import (
    _as_feedback,
)
from problems.registry import context_for_setup_method
from experiments.joint.solve_control.setup_aware_compare_common import (
    augment_setup_params,
    build_online_linucb_branch,
    clone_branch_for_independent_updates,
    default_test_final_bandit_config_from_env,
    run_bandit_step_test_final,
    solve_no_rl_case,
    validate_expected_setup_action_count,
)

_build_bandit = build_online_linucb_branch
RunnerConfig = JointExperimentRuntimeConfig | argparse.Namespace


def _load_setup_replay_trajectory(
    path: Path,
    *,
    online_instances: Sequence[tuple[Mapping[str, Any], np.ndarray]],
) -> tuple[tuple[Dict[str, Any], ...], Dict[str, Any]]:
    """Load and strictly align a recorded setup choice with each instance."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    source_rows = tuple(
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    )
    if len(source_rows) < len(online_instances):
        raise ValueError(
            "Setup replay trajectory is shorter than the online stream: "
            f"{len(source_rows)} < {len(online_instances)}"
        )
    rows = source_rows[: len(online_instances)]
    for index, (row, (mkw, context)) in enumerate(
        zip(rows, online_instances)
    ):
        if int(row.get("online_index", -1)) != index:
            raise ValueError(
                f"Setup replay row {index} has a mismatched online_index"
            )
        if dict(row.get("mkw", {})) != dict(mkw):
            raise ValueError(
                f"Setup replay row {index} does not match the generated PDE instance"
            )
        params = row.get("params")
        if not isinstance(params, Mapping) or not params:
            raise ValueError(
                f"Setup replay row {index} does not contain setup parameters"
            )
        source_context = row.get("context")
        if source_context is not None and not np.array_equal(
            np.asarray(source_context, dtype=float),
            np.asarray(context, dtype=float),
        ):
            raise ValueError(
                f"Setup replay row {index} does not match the generated context"
            )
    return rows, {
        "mode": "frozen_per_instance_trajectory",
        "source": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "cases": len(rows),
        "source_cases": len(source_rows),
        "instance_alignment": "exact mkw and context equality",
        "updates": False,
    }


def _annotate_setup_replay_protocol(
    protocol: Dict[str, Any],
    *,
    replay_metadata: Mapping[str, Any],
    specs: Sequence[ComposableMethodSpec],
) -> None:
    """Record that setup choices are replayed rather than learned online."""

    protocol["purpose"] = (
        f"{int(replay_metadata['cases'])}-instance frozen setup replay; "
        "only the configured solve controllers learn online"
    )
    protocol["setup_replay"] = dict(replay_metadata)
    setup_protocol = protocol["setup_bandit"]
    setup_protocol.update(
        {
            "mode": "frozen_per_instance_trajectory",
            "setup_from_scratch": False,
            "joint_from_scratch": False,
            "frozen_after_warmup": True,
            "independent_updates_per_method": False,
            "online_branches": [],
            "candidate_mode": "replay",
            "aot_schedule": None,
        }
    )
    labels = dict(protocol.get("method_labels", {}))
    for spec in specs:
        if spec.name not in labels:
            continue
        labels[spec.name] = labels[spec.name].replace(
            "Online LinUCB v5",
            "Frozen LinUCB v5 setup replay",
            1,
        )
    protocol["method_labels"] = labels


def _warmup_bandit(
    args: RunnerConfig,
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
                    problem_context=np.asarray(context, dtype=float),
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
                "arm_index": int(native.get("selected_arm_index", -1)),
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
    args: RunnerConfig,
    mkw: Dict[str, Any],
    case_progress: float,
    case_index: int = 0,
    problem_context: Sequence[float] | None = None,
    controller_enabled: bool | None = None,
    controller_bundles: Dict[str, ControllerBundle],
    ppo_runner: Any,
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
        controller_enabled=controller_enabled,
        controller_bundles=controller_bundles,
        ppo_runner=ppo_runner,
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
        report_online_outcome=(
            _report_online_outcome
            if report_online_outcome is None
            else report_online_outcome
        ),
    )


def run(args: RunnerConfig) -> Dict[str, Any]:
    composable_study = None
    composable_specs_tuple: tuple[ComposableMethodSpec, ...] = ()
    setup_configuration_spaces = {}
    if args.study_mode == "composable":
        composable_study = resolve_composable_study(args)
        composable_specs_tuple = composable_study.specs
        setup_configuration_spaces = dict(
            composable_study.setup_configuration_spaces
        )
    penalty = validate_failure_penalty(getattr(args, "failure_penalty_sec", None))
    if penalty is not None:
        if args.study_mode != "composable" or int(args.warmup_cases) or any(
            spec.setup_warmup_cases for spec in composable_specs_tuple
        ):
            raise ValueError("budgeted_penalty requires composable online setup learning from problem 1")
        if getattr(args, "setup_replay_trajectory", None) is not None or any(
            spec.solve_kind == "ppo" for spec in composable_specs_tuple
        ):
            raise ValueError("budgeted_penalty is not supported for frozen setup replay or PPO")
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
    setup_replay_rows: tuple[Dict[str, Any], ...] | None = None
    setup_replay_metadata: Dict[str, Any] | None = None
    setup_replay_path = getattr(args, "setup_replay_trajectory", None)
    if setup_replay_path is not None:
        if args.study_mode != "composable":
            raise ValueError(
                "Setup trajectory replay is supported only by composable experiments"
            )
        setup_replay_rows, setup_replay_metadata = (
            _load_setup_replay_trajectory(
                Path(setup_replay_path),
                online_instances=online_instances,
            )
        )

    composable_specs: Dict[str, ComposableMethodSpec] = {}
    if args.study_mode == "composable":
        assert composable_study is not None
        legacy_study = None
        candidates = ()
        methods = composable_study.methods
        composable_specs = dict(composable_study.specs_by_name)
        family_by_method = dict(composable_study.family_by_method)
    else:
        legacy_study = resolve_legacy_study(args)
        candidates = legacy_study.candidates
        methods = legacy_study.methods
        family_by_method = dict(legacy_study.family_by_method)
    if (
        args.study_mode == "composable"
        and bool(args.include_default_setup_baseline)
    ):
        raise ValueError(
            "Composable experiments must express the default baseline as a "
            "normal --method instead of using the legacy baseline flag"
        )
    if args.study_mode == "composable":
        assert composable_study is not None
        bandit_methods = (
            ()
            if setup_replay_rows is not None
            else composable_study.bandit_methods
        )
    else:
        assert legacy_study is not None
        bandit_methods = legacy_study.bandit_methods
    protocol: Dict[str, Any] | None = None
    if args.study_mode != "composable":
        protocol = build_joint_protocol(
            args,
            git_revision=_git_revision(),
            stream_manifest=stream_manifest,
            methods=methods,
            family_by_method=family_by_method,
            composable_specs=composable_specs_tuple,
            candidates=candidates,
            setup_configuration_spaces=setup_configuration_spaces,
            bandit_methods=bandit_methods,
            behavior_modes=BEHAVIOR_MODES,
            setup_bandit_kinds=SETUP_BANDIT_KINDS,
            recursive_mc_method=RECURSIVE_MC_METHOD,
            recursive_lstdq_method=RECURSIVE_LSTDQ_METHOD,
            batched_lsvi_method=BATCHED_LSVI_METHOD,
        )
        _write_json(args.output_dir / "config.json", protocol)
    _write_json(args.output_dir / "stream_manifest.json", stream_manifest)
    if args.study_mode == "solve_controller_screen":
        _write_solve_screen_reproduction(
            args, stream_hash=str(stream_manifest["sha256"])
        )

    warmup_trajectory_artifact: Any = str(
        args.output_dir / "warmup_trajectory.jsonl"
    )
    if setup_replay_rows is not None:
        assert setup_replay_metadata is not None
        branches = {}
        warmup_rows = []
        warmup_summary = _empty_stream_summary()
        warmup_trajectory_artifact = {
            "mode": "frozen_setup_replay",
            "source": setup_replay_metadata["source"],
        }
        warmup_state_artifact = dict(setup_replay_metadata)
        aot_enabled = False
        (args.output_dir / "warmup_trajectory.jsonl").write_text(
            "",
            encoding="utf-8",
        )
    elif setup_configuration_spaces:
        assert composable_study is not None
        setup_artifacts = build_named_setup_branches(
            args,
            study=composable_study,
            warmup_instances=warmup_instances,
            stream_hash=str(stream_manifest["sha256"]),
        )
        branches = dict(setup_artifacts.branches)
        warmup_rows = list(setup_artifacts.warmup_rows)
        warmup_summary = dict(setup_artifacts.warmup_summary)
        warmup_trajectory_artifact = setup_artifacts.warmup_trajectory
        warmup_state_artifact: Any = dict(setup_artifacts.warmup_state)
        aot_enabled = bool(setup_artifacts.aot_enabled)
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
    validate_setup_branch_independence(
        branches,
        named_setup_spaces=bool(setup_configuration_spaces),
        setup_candidate_mode=str(args.setup_candidate_mode),
    )

    controller_bundles: Dict[str, ControllerBundle] = {}
    if args.study_mode == "composable":
        solve_runtime = build_composable_solve_runtime(
            args,
            specs=composable_specs_tuple,
        )
        controller_bundles = dict(solve_runtime.controller_bundles)
        ppo_runner = solve_runtime.ppo_runner
    else:
        assert legacy_study is not None
        solve_runtime = build_legacy_solve_runtime(
            args,
            study=legacy_study,
            make_ppo_runner=_make_frozen_ppo_runner,
        )
        controller_bundles = dict(solve_runtime.controller_bundles)
        ppo_runner = solve_runtime.ppo_runner
    if args.study_mode == "composable":
        protocol = build_joint_protocol(
            args,
            git_revision=_git_revision(),
            stream_manifest=stream_manifest,
            methods=methods,
            family_by_method=family_by_method,
            composable_specs=composable_specs_tuple,
            candidates=candidates,
            setup_configuration_spaces=setup_configuration_spaces,
            bandit_methods=bandit_methods,
            behavior_modes=BEHAVIOR_MODES,
            setup_bandit_kinds=SETUP_BANDIT_KINDS,
            recursive_mc_method=RECURSIVE_MC_METHOD,
            recursive_lstdq_method=RECURSIVE_LSTDQ_METHOD,
            batched_lsvi_method=BATCHED_LSVI_METHOD,
            solve_controller_protocols={
                method: bundle.protocol_metadata()
                for method, bundle in controller_bundles.items()
            },
        )
        if setup_replay_metadata is not None:
            _annotate_setup_replay_protocol(
                protocol,
                replay_metadata=setup_replay_metadata,
                specs=composable_specs_tuple,
            )
        _write_json(args.output_dir / "config.json", protocol)
    assert protocol is not None
    if getattr(args, "shared_online_prefix", False):
        protocol["shared_online_prefix"] = True
        _write_json(args.output_dir / "config.json", protocol)
    # Preserve the legacy environment-derived validation at the same point in
    # the run lifecycle; the resolved value was never consumed by the loop.
    default_test_final_bandit_config_from_env()
    has_solve_screen_report = (
        args.study_mode == "solve_controller_screen"
        or (
            args.study_mode == "composable"
            and bool(controller_bundles)
        )
    )
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
        setup_replay_rows=setup_replay_rows,
        shared_online_prefix=bool(getattr(args, "shared_online_prefix", False)),
        controller_bundles=controller_bundles,
        ppo_runner=ppo_runner,
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
        aot_max_selections_per_case=int(
            args.aot_max_selections_per_case
        ),
        default_setup_method=DEFAULT_SETUP_METHOD,
        include_solve_screen_report=has_solve_screen_report,
    )
    hooks = OnlineComparisonHooks(
        method_solver=lambda method, **kwargs: _method_solver(
            method,
            args=args,
            controller_bundles=controller_bundles,
            ppo_runner=ppo_runner,
            composable_specs=composable_specs,
            **kwargs,
        ),
        run_default_setup_method=lambda **kwargs: (
            _run_default_setup_method(args=args, **kwargs)
        ),
        report_online_outcome=_report_online_outcome,
        setup_context=lambda method, mkw, context: context_for_setup_method(
            problem_kind=getattr(
                args,
                "problem",
                "scalar_anisotropic_diffusion",
            ),
            setup_kind=(
                composable_specs[method].setup_kind
                if method in composable_specs
                else "linucb"
            ),
            matrix_kwargs=mkw,
            stream_context=context,
            grid_norm_div=float(args.grid_n),
            setup_context=(
                composable_specs[method].setup_context
                if method in composable_specs
                else "default"
            ),
        ),
    )
    return run_online_comparison(plan, hooks=hooks)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
