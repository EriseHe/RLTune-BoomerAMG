from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from amg_setup_gym_env import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from online_sarsa_exploration import (
    BehaviorPolicySarsaController,
    SarsaBehaviorSpec,
)
from online_td_experiment_common import _json_ready, _write_json
from online_td_lambda import (
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from run_online_bandit_rl import _build_bandit
from run_online_bandit_td_lambda import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _git_revision,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
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
    classify_rl_failure,
    clone_branch_for_independent_updates,
    default_test_final_bandit_config_from_env,
    run_bandit_step_test_final,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
    validate_expected_setup_action_count,
)
from shared_action_rl import (
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
)


REFERENCE_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    "bandit_ppo",
)
LSVI_METHOD = "bandit_stagewise_lsvi_lcb"
LSVI_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    LSVI_METHOD,
)
RECURSIVE_MC_METHOD = "bandit_recursive_mc_lcb"
RECURSIVE_LSTDQ_METHOD = "bandit_recursive_lstdq_lcb"
BATCHED_LSVI_METHOD = "bandit_batched_lsvi_lcb"
RECURSIVE_LCB_METHODS = (
    "bandit_default",
    "bandit_fixed_w1.6",
    RECURSIVE_MC_METHOD,
    RECURSIVE_LSTDQ_METHOD,
    BATCHED_LSVI_METHOD,
)
BEHAVIOR_MODES = ("uniform", "uncertainty_lcb")


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


def _validate_shared_lcb_protocol(args: argparse.Namespace) -> None:
    expected_weights = tuple(float(value) for value in np.arange(1.0, 2.01, 0.1))
    expected_centers = (1.0, 1.25, 1.5, 1.75, 2.0)
    weights = _parse_values(args.weights, float)
    centers = _parse_values(args.action_rbf_centers, float)
    if not np.allclose(weights, expected_weights, atol=1.0e-12, rtol=0.0):
        raise ValueError("The locked shared-action space must be 1.0, 1.1, ..., 2.0")
    if not np.allclose(centers, expected_centers, atol=1.0e-12, rtol=0.0):
        raise ValueError("The locked shared-action RBF centers do not match")
    locked_scalars = {
        "action_rbf_sigma": (float(args.action_rbf_sigma), 0.2),
        "potential_scale_sec": (float(args.potential_scale_sec), 0.0),
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
    if getattr(args, "study_mode", "lsvi_lcb") == "recursive_lcb_suite":
        recursive_scalars = {
            "recursive_mc_ridge": (float(args.recursive_mc_ridge), 1.0),
            "recursive_mc_beta": (float(args.recursive_mc_beta), 2.0),
            "recursive_lstdq_ridge": (float(args.recursive_lstdq_ridge), 1.0),
            "recursive_lstdq_beta": (float(args.recursive_lstdq_beta), 2.0),
            "recursive_lstdq_lambda": (
                float(args.recursive_lstdq_lambda),
                0.8,
            ),
        }
        for name, (actual, expected) in recursive_scalars.items():
            if not np.isclose(actual, expected, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"Locked shared-action {name} must equal {expected:g}")
        if int(args.lsvi_refit_interval_episodes) != 100:
            raise ValueError("Locked batched LSVI refit interval must equal 100")


def _validate_lsvi_protocol(args: argparse.Namespace) -> None:
    """Backward-compatible name used by the existing protocol tests."""
    _validate_shared_lcb_protocol(args)


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
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
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
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
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
            q_max_sec=float(args.failure_penalty_sec),
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
            q_max_sec=float(args.failure_penalty_sec),
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
            q_max_sec=float(args.failure_penalty_sec),
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
    history = deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
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

            params, native, timing, failed_attempts, update_sec = (
                run_bandit_step_test_final(
                    policy=branch.policy,
                    parameter_space=branch.parameter_space,
                    context=np.asarray(context, dtype=float),
                    solver_fn=solve_selected,
                    prev_update_est=float(previous_update),
                    success_runtime_history=history,
                    b_min_runtime_sec=1.0e-3,
                    solver_tol=float(args.tol),
                    cfg=bandit_cfg,
                )
            )
            previous_update = float(update_sec)
            row = {
                "warmup_index": int(case_index),
                "mkw": dict(mkw),
                "context": np.asarray(context, dtype=float).tolist(),
                "params": dict(params),
                "arm_index": _policy_last_arm(branch.policy),
                "failed_attempts": int(failed_attempts),
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
):
    if method == "bandit_default":
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
    if method == "bandit_fixed_w1.6":
        def solve_fixed(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_fixed_w_case(
                params=dict(params),
                mkw=dict(mkw),
                w=1.6,
                sweeps_down=1,
                sweeps_up=1,
                solve_tol=float(args.tol),
                solve_max_cycles=int(args.max_cycles),
            )
            return _as_feedback(native, include_controller=False)

        return solve_fixed
    if method == "bandit_ppo":
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
            )
            return _as_feedback(native, include_controller=True)

        return solve_sarsa
    raise ValueError(f"Unsupported method: {method}")


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
    for row in rows:
        outcome = row["outcome"]
        actions.extend(float(value) for value in outcome.get("cycle_actions", []))
        explored.extend(bool(value) for value in outcome.get("cycle_explored", []))
        uncertainties.extend(
            float(value)
            for value in outcome.get("cycle_selected_uncertainties", [])
        )
    return {
        "decisions": int(len(actions)),
        "mean_weight": float(np.mean(actions)) if actions else float("nan"),
        "explored_rate": float(np.mean(explored)) if explored else 0.0,
        "mean_selected_uncertainty_sec": (
            float(np.mean(uncertainties)) if uncertainties else 0.0
        ),
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
        "setup_retries",
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
                    "setup_retries": summary["setup_retry_count"],
                }
            )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.study_mode in {"lsvi_lcb", "recursive_lcb_suite"}:
        _validate_shared_lcb_protocol(args)
    if int(args.warmup_cases) + int(args.online_cases) != int(args.train_cases):
        raise ValueError("train_cases must equal warmup_cases + online_cases")
    if (
        not bool(args.smoke)
        and (int(args.warmup_cases) != 2000 or int(args.online_cases) != 2000)
    ):
        raise ValueError("This locked protocol requires 2000 warmup + 2000 online cases")
    if args.study_mode == "sarsa" and not args.ppo_model.exists():
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
    if not bool(args.smoke) and len(full_stream) != 4000:
        raise ValueError("Locked stream must contain exactly 4000 instances")
    warmup_instances = full_stream[: int(args.warmup_cases)]
    online_instances = full_stream[int(args.warmup_cases) :]

    if args.study_mode == "lsvi_lcb":
        candidates = ()
        methods = LSVI_METHODS
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            LSVI_METHOD: "stagewise_lsvi_lcb",
        }
    elif args.study_mode == "recursive_lcb_suite":
        candidates = ()
        methods = RECURSIVE_LCB_METHODS
        family_by_method = {
            "bandit_default": "default",
            "bandit_fixed_w1.6": "fixed_w1.6",
            RECURSIVE_MC_METHOD: "recursive_mc_lcb",
            RECURSIVE_LSTDQ_METHOD: "recursive_lstdq_lcb",
            BATCHED_LSVI_METHOD: "batched_lsvi_lcb",
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
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "purpose": "2K setup-bandit warmup + 2K persistent joint-online comparison",
        "stream": stream_manifest,
        "stream_partition": {
            "warmup": [0, int(args.warmup_cases)],
            "online": [int(args.warmup_cases), int(args.train_cases)],
        },
        "methods": list(methods),
        "families": family_by_method,
        "candidates": [asdict(candidate) | {"name": candidate.name} for candidate in candidates],
        "setup_bandit": {
            "seed": int(args.bandit_seed),
            "matrix_grid_n": int(args.grid_n),
            "setup_param_resolution": int(args.setup_param_resolution),
            "warmup_instances": int(args.warmup_cases),
            "common_snapshot_mutable_state_cloned": True,
            "immutable_action_catalog_and_g_actions_shared": True,
            "frozen_after_warmup": False,
            "independent_updates_per_method": True,
            "retry_timing_accumulated": True,
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
    if args.study_mode == "lsvi_lcb":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        protocol["stagewise_lsvi_lcb"] = {
            "state": "setup_full",
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
            "potential_scale_sec": float(args.potential_scale_sec),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "update": "backward refit after every solve",
            "frozen": False,
        }
    elif args.study_mode == "recursive_lcb_suite":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        shared_protocol = {
            "state": "setup_full",
            "actions": list(_parse_values(args.weights, float)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(_parse_values(args.action_rbf_centers, float)),
                "sigma": float(args.action_rbf_sigma),
            },
            "potential_scale_sec": float(args.potential_scale_sec),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
                "kind": "uniform epsilon-greedy floor",
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        }
        protocol["recursive_mc_lcb"] = shared_protocol | {
            "target": "undiscounted episodic cost-to-go",
            "estimator": "recursive least squares via Sherman-Morrison",
            "ridge": float(args.recursive_mc_ridge),
            "uncertainty_beta": float(args.recursive_mc_beta),
            "residual_floor_sec": float(args.recursive_mc_residual_floor_sec),
            "history_storage": "none after episode update",
        }
        protocol["recursive_lstdq_lcb"] = shared_protocol | {
            "target": "on-policy LSTDQ(lambda) Bellman equation",
            "estimator": "recursive LSTD with Sherman-Morrison inverse",
            "ridge": float(args.recursive_lstdq_ridge),
            "trace_lambda": float(args.recursive_lstdq_lambda),
            "uncertainty_beta": float(args.recursive_lstdq_beta),
            "uncertainty": "sandwich estimating-equation covariance",
            "history_storage": "none",
        }
        protocol["batched_lsvi_lcb"] = shared_protocol | {
            "target": "stagewise optimistic Bellman backup",
            "horizon": int(args.max_cycles),
            "ridge": float(args.lsvi_ridge),
            "uncertainty_beta": float(args.lsvi_beta),
            "residual_floor_sec": float(args.lsvi_residual_floor_sec),
            "batch_size_episodes": int(args.lsvi_refit_interval_episodes),
            "policy_updates": (
                int(args.online_cases) // int(args.lsvi_refit_interval_episodes)
            ),
            "update": "exact all-history backward refit at fixed batch boundaries",
        }
    _write_json(args.output_dir / "config.json", protocol)
    _write_json(args.output_dir / "stream_manifest.json", stream_manifest)

    warmup_branch, warmup_rows, warmup_summary = _warmup_bandit(
        args,
        warmup_instances,
        stream_hash=str(stream_manifest["sha256"]),
    )
    branches = {
        method: clone_branch_for_independent_updates(warmup_branch)
        for method in methods
    }
    if len({id(branch.policy) for branch in branches.values()}) != len(branches):
        raise RuntimeError("Bandit branches do not have independent policy objects")
    branch_models = [getattr(branch.policy, "model", None) for branch in branches.values()]
    if any(model is None for model in branch_models):
        raise RuntimeError("The locked protocol requires model-backed LinUCB branches")
    if len({id(model.A_inv) for model in branch_models}) != len(branch_models):
        raise RuntimeError("LinUCB mutable parameter matrices are not independent")
    if len({id(model._g_actions) for model in branch_models}) != 1:
        raise RuntimeError("LinUCB branches must share the immutable _g_actions cache")
    if any(model._g_actions.flags.writeable for model in branch_models):
        raise RuntimeError("The shared LinUCB _g_actions cache must be read-only")

    controllers: Dict[str, Any] = {}
    encoders: Dict[str, SolveStateEncoder] = {}
    if args.study_mode == "lsvi_lcb":
        controller, encoder = _make_lsvi_controller(
            args,
            seed=int(args.controller_seed),
        )
        controllers[LSVI_METHOD] = controller
        encoders[LSVI_METHOD] = encoder
        ppo_runner = None
    elif args.study_mode == "recursive_lcb_suite":
        factories = (
            (RECURSIVE_MC_METHOD, _make_recursive_mc_controller),
            (RECURSIVE_LSTDQ_METHOD, _make_recursive_lstdq_controller),
            (BATCHED_LSVI_METHOD, _make_lsvi_controller),
        )
        for controller_index, (method, factory) in enumerate(factories):
            factory_kwargs: Dict[str, Any] = {
                "seed": int(args.controller_seed + controller_index * 1009),
            }
            if method == BATCHED_LSVI_METHOD:
                factory_kwargs["refit_interval_episodes"] = int(
                    args.lsvi_refit_interval_episodes
                )
            controller, encoder = factory(args, **factory_kwargs)
            controllers[method] = controller
            encoders[method] = encoder
        ppo_runner = None
    else:
        for candidate_index, candidate in enumerate(candidates):
            controller, encoder = _make_controller(
                args,
                candidate,
                seed=int(args.controller_seed + candidate_index * 1009),
            )
            controllers[candidate.name] = controller
            encoders[candidate.name] = encoder

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
        ppo_runner = make_exp44_ppo_runner(ppo_args)
    bandit_cfg = default_test_final_bandit_config_from_env()
    histories = {
        method: deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
        for method in methods
    }
    previous_update = {method: 0.0 for method in methods}
    bandit_online_steps = {method: 0 for method in methods}
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
                solver_fn = _method_solver(
                    method,
                    args=args,
                    mkw=dict(mkw),
                    case_progress=float(online_index) / float(progress_denom),
                    controllers=controllers,
                    encoders=encoders,
                    ppo_runner=ppo_runner,
                )
                branch = branches[method]
                params, native, timing, failed_attempts, update_sec = (
                    run_bandit_step_test_final(
                        policy=branch.policy,
                        parameter_space=branch.parameter_space,
                        context=np.asarray(context, dtype=float),
                        solver_fn=solver_fn,
                        prev_update_est=float(previous_update[method]),
                        success_runtime_history=histories[method],
                        b_min_runtime_sec=1.0e-3,
                        solver_tol=float(args.tol),
                        cfg=bandit_cfg,
                    )
                )
                previous_update[method] = float(update_sec)
                bandit_online_steps[method] += 1
                row = {
                    "stream_index": int(args.warmup_cases + online_index),
                    "online_index": int(online_index),
                    "execution_rank": int(execution_rank),
                    "mkw": dict(mkw),
                    "context": np.asarray(context, dtype=float).tolist(),
                    "params": dict(params),
                    "arm_index": _policy_last_arm(branch.policy),
                    "failed_attempts": int(failed_attempts),
                    "bandit_timing": dict(timing),
                    "outcome": _report_online_outcome(native, bandit_timing=timing),
                }
                case_rows[method] = row
            for method in methods:
                records[method].append(case_rows[method])
                _write_json_line(handles[method], case_rows[method])

            done = online_index + 1
            audit_episodes = {int(args.online_cases)}
            if int(args.online_cases) >= 1000:
                audit_episodes.add(1000)
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

    if set(bandit_online_steps.values()) != {len(online_instances)}:
        raise RuntimeError(
            "Every LinUCB branch must update once per online instance: "
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

    if int(args.online_cases) == 2000:
        windows = {
            "all_2000": (0, 2000),
            "first_1000": (0, 1000),
            "last_1000": (1000, 2000),
            "last_500": (1500, 2000),
            "last_300": (1700, 2000),
        }
    else:
        windows = {f"all_{int(args.online_cases)}": (0, int(args.online_cases))}
    window_results = {
        name: _window_result(
            {method: rows[start:stop] for method, rows in records.items()},
            seed=int(args.method_order_seed + start + stop),
        )
        for name, (start, stop) in windows.items()
    }
    result = {
        "protocol": protocol,
        "warmup": {
            "summary": warmup_summary,
            "trajectory": str(args.output_dir / "warmup_trajectory.jsonl"),
            "state": str(
                args.output_dir / f"bandit_warmup_{int(args.warmup_cases)}.npz"
            ),
            "records": int(len(warmup_rows)),
        },
        "windows": window_results,
        "controllers": {
            method: _controller_summary(controller)
            for method, controller in controllers.items()
        },
        "bandit_online_steps": dict(bandit_online_steps),
        "actions": {
            method: _action_summary(records[method]) for method in controllers
        },
        "artifacts": {
            "trajectories": str(trajectories_dir),
            "summary_csv": str(args.output_dir / "summary_2000.csv"),
            "checkpoints": str(checkpoints_dir),
            "final_bandit_states": str(final_bandit_dir),
        },
    }
    if ppo_runner is not None:
        result["ppo"] = {
            "forced_initial_action_count": int(
                ppo_runner.forced_initial_action_count
            ),
        }
    _write_summary_csv(
        args.output_dir / "summary_2000.csv",
        records,
        family_by_method,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a locked 2K setup-bandit warmup followed by a 2K persistent "
            "online comparison of setup/solve controller variants."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--study-mode",
        choices=("sarsa", "lsvi_lcb", "recursive_lcb_suite"),
        default="sarsa",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/mature_tune7_ppo_repro_20260423/run_logs/"
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
        "--matrix-grid-n",
        "--grid-n",
        dest="grid_n",
        type=int,
        default=EXP44_MATRIX_GRID_N,
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
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument(
        "--action-rbf-centers",
        default="1.0,1.25,1.5,1.75,2.0",
    )
    parser.add_argument("--action-rbf-sigma", type=float, default=0.2)
    parser.add_argument("--alphas", default="0.001")
    parser.add_argument("--trace-lambdas", default="0.8")
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20_000.0)
    parser.add_argument("--potential-scale-sec", type=float, default=0.001)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.1)
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
    parser.add_argument("--recursive-lstdq-ridge", type=float, default=1.0)
    parser.add_argument("--recursive-lstdq-beta", type=float, default=2.0)
    parser.add_argument("--recursive-lstdq-lambda", type=float, default=0.8)
    parser.add_argument(
        "--recursive-lstdq-residual-floor-sec", type=float, default=1.0e-3
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--reuse-warmup", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
