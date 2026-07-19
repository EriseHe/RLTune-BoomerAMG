from __future__ import annotations

import _project_paths  # noqa: F401

import copy
import json
import os
import pickle
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from amg_setup_gym_env import build_setup_parameter_spec
from online_bandit_step_env import OnlineBanditStepEnv
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    DIFCONV_CONTEXT_DIM,
    BranchRun,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    augment_setup_params,
    build_action_space_bundle,
    build_single_branch,
    build_test_final_bandit_policy,
    classify_rl_failure,
    default_test_final_bandit_config_from_env,
    default_branch_label,
    family_seed_map,
    generate_difconv_instances,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
    validate_expected_setup_action_count,
)


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_values(name: str, default: str, cast: Any) -> list[Any]:
    raw = _env_str(name, default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"cases": 0}
    runtimes = np.asarray([float(row["runtime"]) for row in results], dtype=float)
    setup = np.asarray([float(row.get("setup_runtime", 0.0)) for row in results], dtype=float)
    solve = np.asarray([float(row.get("solve_runtime", 0.0)) for row in results], dtype=float)
    infer = np.asarray([float(row.get("infer_runtime", 0.0)) for row in results], dtype=float)
    iterations = np.asarray([float(row.get("iterations", np.nan)) for row in results], dtype=float)
    failed = np.asarray([bool(row.get("failed", False)) for row in results], dtype=bool)
    success_runtimes = runtimes[~failed]
    return {
        "cases": int(len(results)),
        "failed_count": int(np.sum(failed)),
        "success_rate": float(1.0 - np.mean(failed)),
        "mean_runtime_sec": float(np.mean(runtimes)),
        "median_runtime_sec": float(np.median(runtimes)),
        "mean_success_runtime_sec": (
            float(np.mean(success_runtimes)) if success_runtimes.size else float("nan")
        ),
        "mean_setup_runtime_sec": float(np.mean(setup)),
        "mean_solve_runtime_sec": float(np.mean(solve)),
        "mean_infer_runtime_sec": float(np.mean(infer)),
        "mean_runtime_with_infer_sec": float(np.mean(runtimes + infer)),
        "mean_iterations": float(np.nanmean(iterations)),
    }


def _relative_improvement(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference <= 0.0 or not np.isfinite(candidate):
        return float("nan")
    return float(100.0 * (reference - candidate) / reference)


def _paired_speed_stats(
    reference_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    valid = (
        len(reference_rows) == len(candidate_rows)
        and bool(reference_rows)
        and not any(bool(row.get("failed", False)) for row in reference_rows)
        and not any(bool(row.get("failed", False)) for row in candidate_rows)
    )
    if not valid:
        return {"valid": False}
    reference = np.asarray(
        [float(row["runtime"]) + float(row.get("infer_runtime", 0.0)) for row in reference_rows],
        dtype=float,
    )
    candidate = np.asarray(
        [float(row["runtime"]) + float(row.get("infer_runtime", 0.0)) for row in candidate_rows],
        dtype=float,
    )
    improvement = 100.0 * (reference - candidate) / reference
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(2000, dtype=float)
    for index in range(bootstrap.size):
        sample = rng.integers(0, improvement.size, size=improvement.size)
        bootstrap[index] = float(np.mean(improvement[sample]))
    return {
        "valid": True,
        "mean_case_improvement_pct": float(np.mean(improvement)),
        "median_case_improvement_pct": float(np.median(improvement)),
        "win_rate": float(np.mean(improvement > 0.0)),
        "bootstrap_mean_95pct": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
    }


def _summarize_online_records(records: Sequence[Dict[str, Any]], window: int) -> Dict[str, Any]:
    def summarize(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        outcomes = [dict(record["outcome"]) for record in subset]
        base = _summarize_results(outcomes)
        if not subset:
            return base
        wall = np.asarray([float(record["episode_wall_sec"]) for record in subset], dtype=float)
        losses = np.asarray(
            [
                float(record["bandit_update"]["loss_sec"])
                for record in subset
                if record.get("bandit_update") is not None
            ],
            dtype=float,
        )
        final_w = np.asarray(
            [float(record["outcome"].get("final_w", np.nan)) for record in subset],
            dtype=float,
        )
        base.update(
            {
                "mean_episode_wall_sec": float(np.mean(wall)),
                "mean_bandit_loss_sec": float(np.mean(losses)) if losses.size else float("nan"),
                "mean_final_w": float(np.nanmean(final_w)),
            }
        )
        return base

    window = max(1, min(int(window), len(records))) if records else 1
    full = summarize(records)
    first = summarize(records[:window])
    last = summarize(records[-window:])
    return {
        "all": full,
        "window_size": int(window),
        "first_window": first,
        "last_window": last,
        "last_vs_first_pct": {
            "native_runtime": _relative_improvement(
                float(first.get("mean_runtime_sec", np.nan)),
                float(last.get("mean_runtime_sec", np.nan)),
            ),
            "episode_wall": _relative_improvement(
                float(first.get("mean_episode_wall_sec", np.nan)),
                float(last.get("mean_episode_wall_sec", np.nan)),
            ),
            "bandit_loss": _relative_improvement(
                float(first.get("mean_bandit_loss_sec", np.nan)),
                float(last.get("mean_bandit_loss_sec", np.nan)),
            ),
        },
    }


class EpisodeProgressCallback(BaseCallback):
    def __init__(self, env: OnlineBanditStepEnv, every_episodes: int) -> None:
        super().__init__(verbose=0)
        self.env = env
        self.every_episodes = max(1, int(every_episodes))
        self._last_reported = 0

    def _on_step(self) -> bool:
        count = len(self.env.completed_episodes)
        if count > self._last_reported and count % self.every_episodes == 0:
            recent = self.env.completed_episodes[-self.every_episodes :]
            mean_runtime = float(np.mean([row["outcome"]["runtime"] for row in recent]))
            failed = int(sum(bool(row["outcome"]["failed"]) for row in recent))
            print(
                json.dumps(
                    {
                        "stage": "online_train_progress",
                        "timesteps": int(self.num_timesteps),
                        "episodes": int(count),
                        "recent_mean_runtime_sec": mean_runtime,
                        "recent_failed": failed,
                        "bandit_updates": int(len(self.env.bandit_updates)),
                    }
                ),
                flush=True,
            )
            self._last_reported = count
        return True


def _safe_one_at_a_time_actions(parameter_spec: Any) -> list[Dict[str, Any]]:
    values = {
        "strong_threshold": _env_values("ONLINE_STRONG_THRESHOLD_VALUES", "0.15,0.35", float),
        "max_row_sum": _env_values("ONLINE_MAX_ROW_SUM_VALUES", "0.8,0.95", float),
        "trunc_factor": _env_values("ONLINE_TRUNC_FACTOR_VALUES", "0.05,0.1", float),
        "P_max_elmts": _env_values("ONLINE_P_MAX_ELMTS_VALUES", "2,6,8", int),
        "agg_num_levels": _env_values("ONLINE_AGG_NUM_LEVELS_VALUES", "1", int),
        "coarsen_type": _env_values("ONLINE_COARSEN_TYPE_VALUES", "6,8", int),
        "interp_type": _env_values("ONLINE_INTERP_TYPE_VALUES", "8", int),
    }
    actions = [dict(DEFAULT_SETUP_PARAMS)]
    seen = {tuple(sorted(DEFAULT_SETUP_PARAMS.items()))}
    params_by_name = {param.name: param for param in parameter_spec.parameters}
    for name, candidates in values.items():
        param = params_by_name[name]
        for value in candidates:
            if value not in param.values:
                if param.kind == "categorical":
                    raise ValueError(f"{name}={value!r} is not in {tuple(param.values)!r}")
                value = min(param.values, key=lambda candidate: abs(float(candidate) - float(value)))
            params = dict(DEFAULT_SETUP_PARAMS)
            params[name] = value
            key = tuple(sorted(params.items()))
            if key not in seen:
                actions.append(params)
                seen.add(key)
    return actions


def _build_bandit(seed: int, tune_dim: int, tune7_variant: str) -> Tuple[BranchRun, Any]:
    bandit_cfg = default_test_final_bandit_config_from_env()
    action_space_mode = _env_str("SETUP_ACTION_SPACE", "safe_one_at_a_time").strip().lower()
    if action_space_mode == "full_cartesian":
        bundle = build_action_space_bundle(final_tune_dims=[int(tune_dim)], tune7_variant=tune7_variant)
        branch = build_single_branch(
            method="linucbv4",
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(seed),
            solver_tol=_env_float("SOLVE_TOL", 1.0e-6),
            solver_max_iter=_env_int("SOLVE_MAX_CYCLES", 50),
            bandit_cfg=bandit_cfg,
            bundle=bundle,
        )
    elif action_space_mode == "safe_one_at_a_time":
        parameter_spec, _fixed_params = build_setup_parameter_spec(
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
        )
        actions = _safe_one_at_a_time_actions(parameter_spec)
        parameter_space = {"actions": actions, "context_dim": int(DIFCONV_CONTEXT_DIM)}
        family = "Shared LinUCB v4"
        policy = build_test_final_bandit_policy(
            method="linucbv4",
            tune_dim=int(tune_dim),
            actions=actions,
            context_dim=int(DIFCONV_CONTEXT_DIM),
            seed=int(family_seed_map(seed=int(seed))[family]),
            default_params=dict(DEFAULT_SETUP_PARAMS),
            default_arm_index=0,
            parameter_spec=parameter_spec,
            tune7_variant=tune7_variant,
            cfg=bandit_cfg,
        )
        branch = BranchRun(
            label=default_branch_label(
                method="linucbv4",
                tune_dim=int(tune_dim),
                tune7_variant=tune7_variant,
            ),
            family=family,
            tune_set="tune7",
            seed=int(family_seed_map(seed=int(seed))[family]),
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=_env_float("SOLVE_TOL", 1.0e-6),
            solver_max_iter=_env_int("SOLVE_MAX_CYCLES", 50),
        )
    else:
        raise ValueError(
            "SETUP_ACTION_SPACE must be safe_one_at_a_time or full_cartesian"
        )
    validate_expected_setup_action_count(branch)
    return branch, bandit_cfg


def _make_env(
    *,
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    branch: BranchRun,
    bandit_cfg: Any,
    seed: int,
    discrete_w_values: Sequence[float] | None = None,
    policy_step_cost_sec: float | None = None,
    potential_progress_scale: float | None = None,
) -> OnlineBanditStepEnv:
    action_mode = _env_str("ACTION_MODE", "discrete_w").strip().lower()
    if action_mode not in {"discrete_w", "continuous_residual"}:
        raise ValueError("online v1 requires ACTION_MODE=discrete_w or continuous_residual")
    return OnlineBanditStepEnv(
        instances=instances,
        branch=branch,
        bandit_cfg=bandit_cfg,
        failure_scale_min_runtime_sec=_env_float("FAILURE_SCALE_MIN_RUNTIME_SEC", 1.0e-3),
        max_setup_attempts=_env_int("MAX_SETUP_ATTEMPTS", 8),
        convergence_guard_start_fraction=_env_float("CONVERGENCE_GUARD_START_FRACTION", 0.6),
        convergence_guard_multiplier=_env_float("CONVERGENCE_GUARD_MULTIPLIER", 2.0),
        tune_dim=_env_int("SETUP_TUNE_DIM", 7),
        tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
        c_max=_env_float("C_MAX", 1000.0),
        tol=_env_float("SOLVE_TOL", 1.0e-6),
        max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
        w_only=True,
        action_mode=action_mode,
        discrete_w_values=(
            tuple(float(value) for value in discrete_w_values)
            if discrete_w_values is not None
            else tuple(_env_values("DISCRETE_W_VALUES", "1.2,1.4,1.5,1.6", float))
        ),
        w_center=_env_float("W_CENTER", 1.5),
        w_scale=_env_float("W_SCALE", 0.02),
        w_global_min=_env_float("W_GLOBAL_MIN", 1.0),
        w_global_max=_env_float("W_GLOBAL_MAX", 2.0),
        w_init_mode="fixed",
        sweeps_min=1,
        sweeps_max=1,
        reward_mode=3,
        term_bonus=_env_float("TERM_BONUS", 0.0),
        trunc_penalty=_env_float("TRUNC_PENALTY", 1.0),
        policy_step_cost_sec=(
            float(policy_step_cost_sec)
            if policy_step_cost_sec is not None
            else _env_float("POLICY_STEP_COST_SEC", 2.0e-4)
        ),
        potential_progress_scale=(
            float(potential_progress_scale)
            if potential_progress_scale is not None
            else _env_float("POTENTIAL_PROGRESS_SCALE", 1.0)
        ),
        obs_mode=_env_str("OBS_MODE", "cycle_action_setup").strip().lower(),
        obs_include_trace_progress=False,
        seed=int(seed),
    )


def _make_runner(model_path: Path, output_dir: Path) -> SetupAwareSolvePolicyRunner:
    matrix_n = _env_int("MATRIX_GRID_N", 30)
    return SetupAwareSolvePolicyRunner(
        SetupAwareRLConfig(
            tune_dim=_env_int("SETUP_TUNE_DIM", 7),
            tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
            algo="ppo",
            model_type=_env_str("MODEL_TYPE", "mlp").strip().lower(),
            model_path=model_path,
            vec_path=output_dir / "unused_vecnormalize.pkl",
            fixed_grid=(matrix_n, matrix_n, matrix_n),
            difconv_c_range=(_env_float("C_MIN", 1.0), _env_float("C_MAX", 1000.0)),
            w_only=True,
            w_center=_env_float("W_CENTER", 1.5),
            w_scale=_env_float("W_SCALE", 0.02),
            w_global_min=_env_float("W_GLOBAL_MIN", 1.0),
            w_global_max=_env_float("W_GLOBAL_MAX", 2.0),
            sweeps_min=1,
            sweeps_max=1,
            w_init=None,
            sweeps_init=None,
            solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
            solve_tol=_env_float("SOLVE_TOL", 1.0e-6),
            default_setup_params=dict(DEFAULT_SETUP_PARAMS),
            obs_mode=_env_str("OBS_MODE", "cycle_action_setup").strip().lower(),
            action_mode=_env_str("ACTION_MODE", "discrete_w").strip().lower(),
            discrete_w_values=tuple(_env_values("DISCRETE_W_VALUES", "1.2,1.4,1.5,1.6", float)),
        )
    )


def _select_frozen_eval_trace(
    branch: BranchRun,
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    candidate_arms: Sequence[int],
) -> Tuple[list[Tuple[Dict[str, Any], Dict[str, Any]]], list[float], float]:
    frozen_policy = copy.deepcopy(branch.policy)
    trace = []
    selection_times = []
    selection_wall_sec = 0.0
    for mkw, context in instances:
        started = time.perf_counter()
        selected = frozen_policy.recommend(
            context=np.asarray(context, dtype=float),
            candidate_arms=candidate_arms,
            alpha=0.0,
            parameter_space=branch.parameter_space,
        )
        selection_sec = float(time.perf_counter() - started)
        selection_times.append(selection_sec)
        selection_wall_sec += selection_sec
        params = selected[0] if isinstance(selected, tuple) else selected
        trace.append((dict(mkw), dict(params)))
    return trace, selection_times, float(selection_wall_sec)


def _deployment_safe_arms(
    branch: BranchRun,
    records: Sequence[Dict[str, Any]],
    *,
    max_cycle_fraction: float,
) -> Tuple[list[int], Dict[str, Any]]:
    if not 0.0 < max_cycle_fraction <= 1.0:
        raise ValueError("max_cycle_fraction must be in (0, 1]")
    max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    default_arm = next(
        index
        for index, action in enumerate(branch.parameter_space["actions"])
        if all(action.get(key) == value for key, value in DEFAULT_SETUP_PARAMS.items())
    )
    by_arm: Dict[int, list[Dict[str, Any]]] = {}
    for record in records:
        arm = int(record.get("selection", {}).get("arm_index", -1))
        if arm >= 0:
            by_arm.setdefault(arm, []).append(dict(record["outcome"]))
        for setup_failure in record.get("setup_failures", []):
            failed_arm = int(
                setup_failure.get("bandit_update", {}).get("arm_index", -1)
            )
            if failed_arm >= 0:
                by_arm.setdefault(failed_arm, []).append(dict(setup_failure["outcome"]))

    safe_arms = []
    arm_stats = {}
    for arm, outcomes in sorted(by_arm.items()):
        failed_count = int(sum(bool(outcome.get("failed", False)) for outcome in outcomes))
        max_iterations = int(
            max(int(outcome.get("iterations", max_cycles)) for outcome in outcomes)
        )
        safe = (
            failed_count == 0
            and max_iterations <= float(max_cycle_fraction) * float(max_cycles)
        )
        arm_stats[str(arm)] = {
            "observations": int(len(outcomes)),
            "failed_count": failed_count,
            "max_iterations": max_iterations,
            "safe": bool(safe),
            "params": dict(branch.parameter_space["actions"][arm]),
        }
        if safe:
            safe_arms.append(int(arm))
    if not safe_arms:
        safe_arms = [int(default_arm)]
    return safe_arms, {
        "max_cycle_fraction": float(max_cycle_fraction),
        "default_arm": int(default_arm),
        "safe_arms": [int(arm) for arm in safe_arms],
        "arm_stats": arm_stats,
    }


def _evaluate(
    *,
    trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    bandit_selection_times: Sequence[float],
    runner: SetupAwareSolvePolicyRunner,
    seed: int,
) -> Dict[str, Any]:
    if len(bandit_selection_times) != len(trace):
        raise ValueError("bandit_selection_times must align with trace")
    fixed_weights = _env_values("FIXED_W_VALUES", "1.4,1.6", float)
    solve_tol = _env_float("SOLVE_TOL", 1.0e-6)
    max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    results: Dict[str, list[Dict[str, Any]]] = {
        "default_setup_default_solve": [],
        "final_bandit_exploit_default_solve": [],
        "final_joint_exploit_ppo": [],
    }
    for fixed_w in fixed_weights:
        results[f"final_bandit_exploit_fixed_w_{fixed_w:.2f}"] = []
    per_case = []
    rng = np.random.default_rng(int(seed))
    for index, (mkw, params) in enumerate(trace):
        case_results: Dict[str, Dict[str, Any]] = {}
        methods = list(results)
        methods = [methods[int(i)] for i in rng.permutation(len(methods))]
        for method in methods:
            if method == "default_setup_default_solve":
                outcome = solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=solve_tol,
                    solver_max_iter=max_cycles,
                    augment_params=augment_setup_params,
                )
            elif method == "final_bandit_exploit_default_solve":
                outcome = solve_no_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solver_tol=solve_tol,
                    solver_max_iter=max_cycles,
                    augment_params=augment_setup_params,
                )
            elif method.startswith("final_bandit_exploit_fixed_w_"):
                fixed_w = float(method.rsplit("_", 1)[-1])
                outcome = solve_fixed_w_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    w=fixed_w,
                    sweeps_down=1,
                    sweeps_up=1,
                    solve_tol=solve_tol,
                    solve_max_cycles=max_cycles,
                )
            else:
                outcome = solve_setup_aware_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solve_policy=runner,
                    augment_params=augment_setup_params,
                    classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=solve_tol,
                        solve_max_cycles=max_cycles,
                    ),
                    solve_max_cycles=max_cycles,
                    case_progress=(0.0 if len(trace) <= 1 else float(index) / float(len(trace) - 1)),
                )
            outcome = dict(outcome)
            if method != "default_setup_default_solve":
                selection_sec = float(bandit_selection_times[index])
                outcome["bandit_select_runtime"] = selection_sec
                outcome["infer_runtime"] = (
                    float(outcome.get("infer_runtime", 0.0)) + selection_sec
                )
            case_results[method] = outcome
            results[method].append(outcome)
        per_case.append(
            {
                "case_index": int(index),
                "mkw": dict(mkw),
                "params": dict(params),
                "methods": case_results,
            }
        )
        if (index + 1) % max(1, _env_int("EVAL_PROGRESS_EVERY", 10)) == 0 or index + 1 == len(trace):
            print(
                json.dumps({"stage": "online_eval_progress", "done": index + 1, "total": len(trace)}),
                flush=True,
            )

    summaries = {name: _summarize_results(rows) for name, rows in results.items()}
    reference_summary = summaries["default_setup_default_solve"]
    reference = float(reference_summary["mean_runtime_with_infer_sec"])
    relative = {}
    valid_speed_comparison = {}
    for name, summary in summaries.items():
        if name == "default_setup_default_solve":
            continue
        valid = int(reference_summary["failed_count"]) == 0 and int(summary["failed_count"]) == 0
        valid_speed_comparison[name] = bool(valid)
        relative[f"{name}_vs_default_pct"] = (
            _relative_improvement(reference, float(summary["mean_runtime_with_infer_sec"]))
            if valid
            else float("nan")
        )
    paired_speed = {
        name: _paired_speed_stats(
            results["default_setup_default_solve"],
            rows,
            seed=int(seed + index + 1),
        )
        for index, (name, rows) in enumerate(results.items())
        if name != "default_setup_default_solve"
    }
    return {
        "methods": summaries,
        "relative_pct": relative,
        "valid_speed_comparison": valid_speed_comparison,
        "paired_speed": paired_speed,
        "per_case": per_case,
    }


def main() -> None:
    output_dir = Path(_env_str("OUTPUT_DIR", str(_REPO_ROOT / "results/joint/online_joint_v1/run_logs/manual"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = _env_int("ONLINE_SEED", 20260715)
    eval_seed = _env_int("EVAL_SEED", seed + 100_000)
    total_timesteps = _env_int("TOTAL_TIMESTEPS", 1024)
    n_steps = _env_int("N_STEPS", 128)
    batch_size = _env_int("BATCH_SIZE", n_steps)
    matrix_n = _env_int("MATRIX_GRID_N", 30)
    tune_dim = _env_int("SETUP_TUNE_DIM", 7)
    tune7_variant = _env_str("TUNE7_VARIANT", "categorical").strip().lower()
    if tune_dim != 7 or tune7_variant != "categorical":
        raise ValueError("online v1 is intentionally scoped to Tune7 categorical")
    if batch_size > n_steps:
        raise ValueError("BATCH_SIZE must be <= N_STEPS for one online environment")

    torch.set_num_threads(max(1, _env_int("TORCH_NUM_THREADS", 2)))
    config = {
        "git_revision": _git_revision(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "seed": seed,
        "eval_seed": eval_seed,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": _env_int("N_EPOCHS", 4),
        "learning_rate": _env_float("LEARNING_RATE", 3.0e-4),
        "gamma": _env_float("GAMMA", 1.0),
        "gae_lambda": _env_float("GAE_LAMBDA", 0.95),
        "matrix_grid_n": matrix_n,
        "setup_param_resolution": _env_int("SETUP_PARAM_RESOLUTION", 8),
        "setup_action_space": _env_str("SETUP_ACTION_SPACE", "safe_one_at_a_time"),
        "setup_initial_guess_rounds": _env_int("SETUP_INITIAL_GUESS_ROUNDS", 4),
        "eval_cases": _env_int("EVAL_CASES", 48),
        "signal_window": _env_int("SIGNAL_WINDOW", 25),
        "solve_max_cycles": _env_int("SOLVE_MAX_CYCLES", 50),
        "solve_tol": _env_float("SOLVE_TOL", 1.0e-6),
        "w_center": _env_float("W_CENTER", 1.5),
        "w_scale": _env_float("W_SCALE", 0.02),
        "action_mode": _env_str("ACTION_MODE", "discrete_w").strip().lower(),
        "discrete_w_values": _env_values("DISCRETE_W_VALUES", "1.2,1.4,1.5,1.6", float),
        "model_type": _env_str("MODEL_TYPE", "mlp").strip().lower(),
        "reward_mode": 3,
        "trunc_penalty": _env_float("TRUNC_PENALTY", 1.0),
        "policy_step_cost_sec": _env_float("POLICY_STEP_COST_SEC", 2.0e-4),
        "potential_progress_scale": _env_float("POTENTIAL_PROGRESS_SCALE", 1.0),
        "convergence_guard_start_fraction": _env_float("CONVERGENCE_GUARD_START_FRACTION", 0.6),
        "convergence_guard_multiplier": _env_float("CONVERGENCE_GUARD_MULTIPLIER", 2.0),
        "deployment_max_cycle_fraction": _env_float("DEPLOYMENT_MAX_CYCLE_FRACTION", 0.7),
    }
    _write_json(output_dir / "config.json", config)
    print(json.dumps({"stage": "online_start", "output_dir": str(output_dir), **config}), flush=True)

    branch, bandit_cfg = _build_bandit(seed=seed, tune_dim=tune_dim, tune7_variant=tune7_variant)
    model = branch.policy.model
    config["bandit"] = {
        "actions": int(model.K),
        "feature_dim": int(model.d_phi),
        "candidate_pool_size": int(bandit_cfg.tune7_candidate_pool_size),
        "candidate_pool_size_burnin": int(bandit_cfg.tune7_candidate_pool_size_burnin),
    }
    _write_json(output_dir / "config.json", config)
    print(json.dumps({"stage": "bandit_ready", **config["bandit"]}), flush=True)

    instances = generate_difconv_instances(
        T=max(total_timesteps + _env_int("MAX_SETUP_ATTEMPTS", 8), 64),
        seed=seed,
        grid_choices=[(matrix_n, matrix_n, matrix_n)],
        c_min=_env_float("C_MIN", 1.0),
        c_max=_env_float("C_MAX", 1000.0),
    )
    env = _make_env(instances=instances, branch=branch, bandit_cfg=bandit_cfg, seed=seed)
    vec_env = VecMonitor(DummyVecEnv([lambda: env]))
    model_type = _env_str("MODEL_TYPE", "mlp").strip().lower()
    if model_type == "lstm":
        policy_name = "MlpLstmPolicy"
        policy_kwargs: Dict[str, Any] = {
            "net_arch": {"pi": [64], "vf": [64]},
            "lstm_hidden_size": _env_int("LSTM_HIDDEN_SIZE", 64),
        }
        algorithm = RecurrentPPO
    elif model_type == "mlp":
        policy_name = "MlpPolicy"
        policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
        algorithm = PPO
    else:
        raise ValueError("MODEL_TYPE must be mlp or lstm")
    if _env_str("ACTION_MODE", "discrete_w").strip().lower() == "continuous_residual":
        policy_kwargs["log_std_init"] = _env_float("LOG_STD_INIT", -1.0)
    ppo = algorithm(
        policy_name,
        vec_env,
        learning_rate=_env_float("LEARNING_RATE", 3.0e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=_env_int("N_EPOCHS", 4),
        gamma=_env_float("GAMMA", 1.0),
        gae_lambda=_env_float("GAE_LAMBDA", 0.95),
        clip_range=_env_float("CLIP_RANGE", 0.2),
        ent_coef=_env_float("ENT_COEF", 0.0),
        vf_coef=_env_float("VF_COEF", 0.5),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=_env_int("RL_VERBOSE", 0),
        device="cpu",
    )

    training_started = time.perf_counter()
    ppo.learn(
        total_timesteps=total_timesteps,
        callback=EpisodeProgressCallback(
            env,
            every_episodes=_env_int("PROGRESS_EVERY_EPISODES", 10),
        ),
    )
    training_wall_sec = float(time.perf_counter() - training_started)
    env.discard_incomplete_episode()

    model_base = output_dir / "online_joint_model"
    ppo.save(str(model_base))
    model_path = model_base.with_suffix(".zip")
    with (output_dir / "online_bandit_state.pkl").open("wb") as handle:
        pickle.dump({"branch": branch}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    online_records = copy.deepcopy(env.completed_episodes)
    bandit_updates = copy.deepcopy(env.bandit_updates)
    interaction_wall_sec = float(env.interaction_wall_sec)
    vec_env.close()

    online_summary = _summarize_online_records(
        online_records,
        window=_env_int("SIGNAL_WINDOW", 10),
    )
    training_summary = {
        "requested_timesteps": total_timesteps,
        "actual_timesteps": int(ppo.num_timesteps),
        "completed_episodes": int(len(online_records)),
        "bandit_updates": int(len(bandit_updates)),
        "training_wall_sec": training_wall_sec,
        "environment_interaction_wall_sec": interaction_wall_sec,
        "ppo_and_policy_overhead_wall_sec": float(max(0.0, training_wall_sec - interaction_wall_sec)),
        "online_signal": online_summary,
    }
    print(json.dumps(_json_ready({"stage": "online_train_done", **training_summary})), flush=True)

    eval_cases = _env_int("EVAL_CASES", 48)
    eval_instances = generate_difconv_instances(
        T=eval_cases,
        seed=eval_seed,
        grid_choices=[(matrix_n, matrix_n, matrix_n)],
        c_min=_env_float("C_MIN", 1.0),
        c_max=_env_float("C_MAX", 1000.0),
    )
    deployment_arms, deployment_safety = _deployment_safe_arms(
        branch,
        online_records,
        max_cycle_fraction=_env_float("DEPLOYMENT_MAX_CYCLE_FRACTION", 0.7),
    )
    eval_trace, selection_times, selection_wall_sec = _select_frozen_eval_trace(
        branch,
        eval_instances,
        candidate_arms=deployment_arms,
    )
    runner = _make_runner(model_path, output_dir)
    evaluation_started = time.perf_counter()
    evaluation = _evaluate(
        trace=eval_trace,
        bandit_selection_times=selection_times,
        runner=runner,
        seed=eval_seed + 1,
    )
    evaluation_wall_sec = float(time.perf_counter() - evaluation_started)
    evaluation.update(
        {
            "cases": eval_cases,
            "bandit_selection_wall_sec": selection_wall_sec,
            "evaluation_wall_sec": evaluation_wall_sec,
            "deployment_safety": deployment_safety,
        }
    )

    payload = {
        "config": config,
        "artifacts": {
            "model": model_path,
            "bandit_state": output_dir / "online_bandit_state.pkl",
        },
        "training": training_summary,
        "evaluation": evaluation,
        "online_records": online_records,
        "bandit_updates": bandit_updates,
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, payload)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "online_done",
                    "result": result_path,
                    "relative_pct": evaluation["relative_pct"],
                    "online_last_vs_first_pct": online_summary["last_vs_first_pct"],
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
