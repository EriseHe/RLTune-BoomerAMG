from __future__ import annotations

import _project_paths  # noqa: F401

import math
import json
import os
from collections import Counter, deque
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO

from frozen_bandit_step_env import FrozenBanditStepEnv
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    augment_setup_params,
    build_test10_branches,
    classify_rl_failure,
    default_branch_label,
    default_test_final_bandit_config_from_env,
    generate_difconv_instances,
    run_bandit_step_test_final,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_schedule_case,
    solve_setup_aware_rl_case,
    validate_expected_setup_action_count,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_optional_float(name: str):
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    return float(raw)


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_flag(name: str, default: str = "0") -> bool:
    return _env_str(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _initialize_continuous_policy_weight(
    model,
    *,
    initial_weight: float,
    w_center: float,
    w_scale: float,
) -> float:
    if float(w_scale) <= 0.0:
        raise ValueError("W_SCALE must be positive")
    normalized_action = (float(initial_weight) - float(w_center)) / float(w_scale)
    if not -1.0 <= normalized_action <= 1.0:
        raise ValueError("INITIAL_POLICY_WEIGHT is outside the action range")
    action_net = getattr(model.policy, "action_net", None)
    if action_net is None or action_net.bias is None:
        raise ValueError("PPO policy does not expose an initializable action head")
    if int(action_net.out_features) != 1:
        raise ValueError("Default-weight initialization currently requires w-only PPO")
    with torch.no_grad():
        action_net.weight.zero_()
        action_net.bias.fill_(float(normalized_action))
    return float(normalized_action)


def _env_float_tuple(name: str, default: str = "") -> Tuple[float, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _env_int_tuple(name: str, default: str = "") -> Tuple[int, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _env_joint_actions(name: str, default: str = "") -> Tuple[Tuple[float, int, int], ...]:
    raw = _env_str(name, default).strip()
    out: List[Tuple[float, int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        w, sd, su = item.split(":")
        out.append((float(w), int(sd), int(su)))
    return tuple(out)


def _env_extended_actions(
    name: str, default: str = ""
) -> Tuple[Tuple[float, int, int, int, int, int, float, float], ...]:
    raw = _env_str(name, default).strip()
    out: List[Tuple[float, int, int, int, int, int, float, float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) < 6:
            raise ValueError(f"{name} item must have at least 6 fields: {item!r}")
        w, sd, su, sc, ct, rt = parts[:6]
        ow = parts[6] if len(parts) > 6 else "-1"
        arw = parts[7] if len(parts) > 7 else "-1"
        out.append((float(w), int(sd), int(su), int(sc), int(ct), int(rt), float(ow), float(arw)))
    return tuple(out)


def _env_teacher_schedule(name: str, default: str = "") -> Tuple[Tuple[int, int], ...]:
    raw = _env_str(name, default).strip()
    out: List[Tuple[int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        end_cycle, idx = item.split(":")
        out.append((int(end_cycle), int(idx)))
    return tuple(out)


def _param_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    out = []
    for k, v in sorted(dict(params).items()):
        out.append((k, round(float(v), 12) if isinstance(v, float) else v))
    return tuple(out)


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(r["runtime"]) for r in results]
    setup = [float(r["setup_runtime"]) for r in results]
    solve = [float(r["solve_runtime"]) for r in results]
    controller = [float(r.get("infer_runtime", 0.0)) for r in results]
    runtime_with_controller = [
        runtime + controller_runtime
        for runtime, controller_runtime in zip(runtimes, controller)
    ]
    failures = [bool(r.get("failed", False)) for r in results]
    iterations = [int(r.get("iterations", -1)) for r in results if int(r.get("iterations", -1)) >= 0]
    final_ws = [float(r.get("final_w")) for r in results if np.isfinite(float(r.get("final_w", np.nan)))]
    action_counts: Counter[int] = Counter()
    for r in results:
        for k, v in dict(r.get("action_counts", {})).items():
            action_counts[int(k)] += int(v)
    return {
        "cases": int(len(results)),
        "mean_runtime": float(mean(runtimes)),
        "total_runtime": float(sum(runtimes)),
        "mean_setup_runtime": float(mean(setup)),
        "total_setup_runtime": float(sum(setup)),
        "mean_solve_runtime": float(mean(solve)),
        "total_solve_runtime": float(sum(solve)),
        "mean_infer_runtime": float(mean(controller)),
        "total_infer_runtime": float(sum(controller)),
        "mean_runtime_with_controller": float(mean(runtime_with_controller)),
        "total_runtime_with_controller": float(sum(runtime_with_controller)),
        "failed_count": int(sum(failures)),
        "mean_iterations": float(mean(iterations)) if iterations else float("nan"),
        "mean_final_w": float(mean(final_ws)) if final_ws else float("nan"),
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
    }


def _make_branch(seed: int):
    tune_dim = _env_int("SETUP_TUNE_DIM", 7)
    bandit_method = _env_str("SETUP_BANDIT_METHOD", "linucbv4").strip().lower()
    tune7_variant = _env_str("TUNE7_VARIANT", "categorical").strip().lower()
    bandit_cfg = default_test_final_bandit_config_from_env()
    branches, _bundle = build_test10_branches(
        final_tune_dims=[tune_dim],
        tune7_variant=tune7_variant,
        seed=int(seed),
        solver_tol=_env_float("SOLVER_TOL", 1e-6),
        solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
        include_default=False,
        method_filter=bandit_method,
        branch_filter=_env_str(
            "BRANCH_FILTER",
            default_branch_label(method=bandit_method, tune_dim=tune_dim, tune7_variant=tune7_variant),
        ),
        bandit_cfg=bandit_cfg,
    )
    if len(branches) != 1:
        raise RuntimeError(f"Expected one branch, got {[b.label for b in branches]}")
    return branches[0], bandit_cfg


def _warmup_bandit():
    # Exp44 rebuilds the immutable action catalog and restores only mutable
    # LinUCB state, so multi-million-action catalogs are never serialized.
    saved_state = _env_str("BANDIT_STATE_PATH", "").strip()
    if saved_state and _env_flag("USE_SAVED_BANDIT", "0"):
        state_path = Path(saved_state)
        print(json.dumps({"stage": "bandit_state_load_start", "path": str(state_path)}), flush=True)
        warmup_seed = _env_int("WARMUP_SEED", 39393939)
        branch, _bandit_cfg = _make_branch(seed=warmup_seed)
        metadata = branch.policy.model.load_mutable_state(state_path)
        validate_expected_setup_action_count(branch)
        expected_metadata = {
            "matrix_grid_n": _env_int("MATRIX_GRID_N", EXP44_MATRIX_GRID_N),
            "setup_param_resolution": _env_int(
                "SETUP_PARAM_RESOLUTION", EXP44_SETUP_PARAM_RESOLUTION
            ),
            "warmup_seed": warmup_seed,
            "warmup_cases": _env_int("WARMUP_CASES", 1500),
        }
        for key, expected in expected_metadata.items():
            if int(metadata.get(key, -1)) != int(expected):
                raise ValueError(
                    f"Saved mature-bandit state has {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        print(json.dumps({"stage": "bandit_state_load_done", "path": str(state_path)}), flush=True)
        return branch

    grid_n = _env_int("MATRIX_GRID_N", EXP44_MATRIX_GRID_N)
    warmup_seed = _env_int("WARMUP_SEED", 39393939)
    warmup_cases = _env_int("WARMUP_CASES", 1500)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))
    print(
        json.dumps(
            {
                "stage": "branch_build_start",
                "grid": [grid_n, grid_n, grid_n],
                "warmup_cases": warmup_cases,
                "warmup_seed": warmup_seed,
                "bandit_method": _env_str("SETUP_BANDIT_METHOD", "linucbv4"),
                "tune_dim": _env_int("SETUP_TUNE_DIM", 7),
                "tune7_variant": _env_str("TUNE7_VARIANT", "categorical"),
            }
        ),
        flush=True,
    )
    branch, bandit_cfg = _make_branch(seed=warmup_seed)
    validate_expected_setup_action_count(branch)
    print(
        json.dumps(
            {
                "stage": "warmup_start",
                "grid": [grid_n, grid_n, grid_n],
                "warmup_cases": warmup_cases,
                "warmup_seed": warmup_seed,
                "bandit": branch.label,
            }
        ),
        flush=True,
    )
    instances = generate_difconv_instances(
        T=warmup_cases,
        seed=warmup_seed,
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=c_min,
        c_max=c_max,
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    prev_update_est = 0.0
    for i, (mkw, context) in enumerate(instances, 1):
        def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
            return solve_no_rl_case(
                params=selected_params,
                mkw=dict(mkw),
                solver_tol=_env_float("SOLVER_TOL", 1e-6),
                solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
                augment_params=augment_setup_params,
            )

        _params, _out, _timing, _fallback_used, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            fallback_solver_fn=solver_fn,
            prev_update_est=float(prev_update_est),
        )
        if i % max(1, _env_int("PROGRESS_EVERY", 100)) == 0 or i == warmup_cases:
            print(json.dumps({"stage": "warmup_progress", "done": i, "total": warmup_cases}), flush=True)
    print(json.dumps({"stage": "warmup_done"}), flush=True)
    save_state = _env_str("SAVE_BANDIT_STATE_PATH", "").strip()
    if save_state:
        state_path = Path(save_state)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        branch.policy.model.save_mutable_state(
            state_path,
            metadata={
                "matrix_grid_n": int(grid_n),
                "setup_param_resolution": _env_int(
                    "SETUP_PARAM_RESOLUTION", EXP44_SETUP_PARAM_RESOLUTION
                ),
                "warmup_seed": int(warmup_seed),
                "warmup_cases": int(warmup_cases),
                "action_count": int(branch.policy.model.K),
            },
        )
        print(json.dumps({"stage": "bandit_state_save_done", "path": str(state_path)}), flush=True)
    return branch


def _frozen_trace(branch, *, cases: int, seed: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # Numbered trace-building flow:
    # 1) regenerate diffusion-convection cases from the requested seed
    # 2) for each case, call branch.policy.select(...) on its context
    # 3) save (mkw, params) so solve-phase RL can reuse the mature setup choice
    grid_n = _env_int("MATRIX_GRID_N", EXP44_MATRIX_GRID_N)
    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=int(cases),
        seed=int(seed),
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=_env_float("C_MIN", 1.0),
        c_max=_env_float("C_MAX", 1000.0),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    trace: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, (mkw, context) in enumerate(instances, 1):
        params, _info = branch.policy.select(
            context=np.asarray(context, dtype=float),
            parameter_space=branch.parameter_space,
        )
        trace.append((dict(mkw), dict(params)))
        if i % max(1, _env_int("PROGRESS_EVERY", 100)) == 0 or i == int(cases):
            print(json.dumps({"stage": "trace_progress", "done": i, "total": int(cases), "seed": int(seed)}), flush=True)
    return trace


def _build_train_trace(branch) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    # Build one frozen training dataset by concatenating per-seed traces.
    train_seeds = _env_int_tuple("RL_TRAIN_SEEDS", "")
    if not train_seeds:
        train_seeds = (_env_int("RL_TRAIN_SEED", 39396939),)
    cases_per_seed = _env_int("RL_TRAIN_CASES", 500)
    traces: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = []
    for seed in train_seeds:
        print(
            json.dumps(
                {
                    "stage": "train_trace_build_start",
                    "seed": int(seed),
                    "cases": int(cases_per_seed),
                }
            ),
            flush=True,
        )
        traces.append(_frozen_trace(branch, cases=cases_per_seed, seed=int(seed)))
    train_trace = [item for trace in traces for item in trace]
    shuffle = _env_flag("TRAIN_TRACE_SHUFFLE", "1")
    shuffle_seed = _env_int("TRAIN_TRACE_SHUFFLE_SEED", _env_int("RL_SEED", _env_int("WARMUP_SEED", 39393939) + 77))
    if shuffle and len(train_trace) > 1:
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(len(train_trace))
        train_trace = [train_trace[int(i)] for i in order]
    meta = {
        "train_seeds": [int(x) for x in train_seeds],
        "cases_per_seed": int(cases_per_seed),
        "total_cases": int(len(train_trace)),
        "shuffle": bool(shuffle),
        "shuffle_seed": int(shuffle_seed),
    }
    print(json.dumps({"stage": "train_trace_build_done", **meta}), flush=True)
    return train_trace, meta


def _make_env(trace, *, seed: int):
    reference_mode = _env_str("REWARD_REFERENCE_MODE", "").strip().lower()
    reference_runtimes = None
    if reference_mode == "fixed_w":
        fixed_w = _env_float("FIXED_W", 1.60)
        fixed_sd = _env_int("FIXED_SWEEPS_DOWN", _env_int("SWEEPS_MIN", 1))
        fixed_su = _env_int("FIXED_SWEEPS_UP", _env_int("SWEEPS_MAX", 1))
        reference_runtimes = tuple(
            float(
                solve_fixed_w_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    w=float(fixed_w),
                    sweeps_down=int(fixed_sd),
                    sweeps_up=int(fixed_su),
                    solve_tol=_env_float("SOLVE_TOL", 1e-6),
                    solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
                )["runtime"]
            )
            for mkw, params in trace
        )
    return FrozenBanditStepEnv(
        trace=trace,
        tune_dim=_env_int("SETUP_TUNE_DIM", 7),
        tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
        c_max=_env_float("C_MAX", 1000.0),
        tol=_env_float("SOLVE_TOL", 1e-6),
        max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
        w_only=_env_flag("W_ONLY", "1"),
        action_mode=_env_str("ACTION_MODE", "continuous").strip().lower(),
        discrete_w_values=_env_float_tuple("DISCRETE_W_VALUES", "1.4,1.6,1.8"),
        discrete_joint_actions=_env_joint_actions("DISCRETE_ACTIONS"),
        discrete_extended_actions=_env_extended_actions("DISCRETE_EXTENDED_ACTIONS"),
        discrete_blend_alphas=_env_float_tuple("DISCRETE_BLEND_ALPHAS", "0.0,0.5,1.0"),
        w_center=_env_float("W_CENTER", 1.65),
        w_scale=_env_float("W_SCALE", 0.1),
        w_global_min=_env_float("W_GLOBAL_MIN", 1.0),
        w_global_max=_env_float("W_GLOBAL_MAX", 2.0),
        initial_observation_weight=_env_optional_float(
            "INITIAL_OBSERVATION_WEIGHT"
        ),
        w_init_mode=_env_str("W_INIT_MODE", "fixed"),
        w_init_min=_env_optional_float("W_INIT_MIN"),
        w_init_max=_env_optional_float("W_INIT_MAX"),
        sweeps_min=_env_int("SWEEPS_MIN", 1),
        sweeps_max=_env_int("SWEEPS_MAX", 1),
        reward_mode=_env_int("REWARD_MODE", 4),
        term_bonus=_env_float("TERM_BONUS", 0.0),
        reference_runtimes=reference_runtimes,
        relative_bonus_scale=_env_float("RELATIVE_BONUS_SCALE", 1.0),
        reference_margin_sec=_env_float("REFERENCE_MARGIN_SEC", 0.0),
        anchor_w=_env_float("ANCHOR_W", float("nan")),
        anchor_deviation_penalty=_env_float("ANCHOR_DEVIATION_PENALTY", 0.0),
        obs_mode=_env_str("OBS_MODE", "solve_only"),
        teacher_action_schedule=_env_teacher_schedule("TEACHER_ACTION_SCHEDULE"),
        teacher_match_bonus=_env_float("TEACHER_MATCH_BONUS", 0.0),
        teacher_mismatch_penalty=_env_float("TEACHER_MISMATCH_PENALTY", 0.0),
        seed=int(seed),
    )


def _score_eval_summary(summary: Dict[str, Any], per_eval_seed: Dict[str, Any] | None = None) -> float:
    fixed_w = _env_float("FIXED_W", 1.60)
    objective = _env_str("CHECKPOINT_OBJECTIVE", "avg_vs_fixed_bandit").strip().lower()
    rel = dict(summary["relative_pct"])
    key_fixed = f"rl_vs_fixed_w_{fixed_w:.2f}"
    if objective == "vs_fixed":
        return float(rel[key_fixed])
    if objective == "vs_bandit":
        return float(rel["rl_vs_bandit"])
    if objective == "avg_vs_fixed_bandit":
        return 0.5 * float(rel[key_fixed] + rel["rl_vs_bandit"])
    if objective == "min_vs_fixed_bandit":
        return float(min(rel[key_fixed], rel["rl_vs_bandit"]))
    if objective == "avg_vs_default_fixed_bandit":
        return float((rel["rl_vs_default_setup"] + rel[key_fixed] + rel["rl_vs_bandit"]) / 3.0)
    if objective == "seed_min_vs_fixed":
        if not per_eval_seed:
            raise ValueError("seed_min_vs_fixed requires per_eval_seed")
        return float(min(float(row["relative_pct"][key_fixed]) for row in per_eval_seed.values()))
    if objective == "seed_mean_minus_std_vs_fixed":
        if not per_eval_seed:
            raise ValueError("seed_mean_minus_std_vs_fixed requires per_eval_seed")
        vals = np.asarray([float(row["relative_pct"][key_fixed]) for row in per_eval_seed.values()], dtype=float)
        return float(np.mean(vals) - np.std(vals))
    if objective == "seed_mean_minus_std_vs_bandit":
        if not per_eval_seed:
            raise ValueError("seed_mean_minus_std_vs_bandit requires per_eval_seed")
        vals = np.asarray(
            [float(row["relative_pct"]["rl_vs_bandit"]) for row in per_eval_seed.values()],
            dtype=float,
        )
        return float(np.mean(vals) - np.std(vals))
    if objective == "seed_mean_minus_std_avg_fixed_bandit":
        if not per_eval_seed:
            raise ValueError("seed_mean_minus_std_avg_fixed_bandit requires per_eval_seed")
        vals = np.asarray(
            [
                0.5 * (
                    float(row["relative_pct"][key_fixed])
                    + float(row["relative_pct"]["rl_vs_bandit"])
                )
                for row in per_eval_seed.values()
            ],
            dtype=float,
        )
        return float(np.mean(vals) - np.std(vals))
    raise ValueError(f"Unsupported CHECKPOINT_OBJECTIVE={objective!r}")


def _evaluate_model_on_traces(
    eval_traces: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]], model_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per_eval_seed = {
        str(seed): _evaluate(trace, model_path) for seed, trace in eval_traces.items()
    }
    combined_eval = _combine_eval_summaries(list(per_eval_seed.values()))
    return per_eval_seed, combined_eval


def _train_rl(train_trace, eval_traces) -> Tuple[Path, Dict[str, Any]]:
    model_type = _env_str("MODEL_TYPE", "mlp").strip().lower()
    init_model_path = _env_str("INIT_MODEL_PATH", "").strip()
    seed = _env_int("RL_SEED", _env_int("WARMUP_SEED", 39393939) + 77)
    model_base = Path(_env_str("MODEL_BASENAME", "/tmp/mature_bandit_pipeline_rl"))
    model_base.parent.mkdir(parents=True, exist_ok=True)
    venv = VecMonitor(DummyVecEnv([lambda: _make_env(train_trace, seed=seed)]))
    common = {
        "learning_rate": _env_float("LEARNING_RATE", 3e-4),
        "gamma": _env_float("GAMMA", 0.99),
        "verbose": _env_int("RL_VERBOSE", 1),
        "device": "cpu",
        "seed": seed,
    }
    if init_model_path:
        init_path = str(Path(init_model_path))
        if model_type == "lstm":
            model = RecurrentPPO.load(init_path, env=venv, device="cpu")
        else:
            model = PPO.load(init_path, env=venv, device="cpu")
        if _env_flag("OVERRIDE_LOADED_LR", "1"):
            override_lr = _env_float("LEARNING_RATE", 3e-4)
            model.learning_rate = override_lr
            if hasattr(model, "lr_schedule"):
                model.lr_schedule = lambda _progress: float(override_lr)
            policy_opt = getattr(getattr(model, "policy", None), "optimizer", None)
            if policy_opt is not None:
                for group in policy_opt.param_groups:
                    group["lr"] = override_lr
            print(json.dumps({"stage": "loaded_model_lr_override", "learning_rate": override_lr}), flush=True)
    elif model_type == "lstm":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            venv,
            n_steps=_env_int("N_STEPS", 64),
            batch_size=_env_int("BATCH_SIZE", 64),
            n_epochs=_env_int("N_EPOCHS", 10),
            gae_lambda=_env_float("GAE_LAMBDA", 0.95),
            clip_range=_env_float("CLIP_RANGE", 0.2),
            ent_coef=_env_float("ENT_COEF", 0.0),
            vf_coef=_env_float("VF_COEF", 0.5),
            policy_kwargs={"net_arch": {"pi": [128], "vf": [128]}, "lstm_hidden_size": 128},
            **common,
        )
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=_env_int("N_STEPS", 64),
            batch_size=_env_int("BATCH_SIZE", 64),
            n_epochs=_env_int("N_EPOCHS", 10),
            gae_lambda=_env_float("GAE_LAMBDA", 0.95),
            clip_range=_env_float("CLIP_RANGE", 0.2),
            ent_coef=_env_float("ENT_COEF", 0.0),
            vf_coef=_env_float("VF_COEF", 0.5),
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            **common,
        )
    initialized_policy_action = None
    initial_policy_weight = _env_optional_float("INITIAL_POLICY_WEIGHT")
    if not init_model_path and initial_policy_weight is not None:
        action_mode = _env_str("ACTION_MODE", "continuous").strip().lower()
        if action_mode not in {"continuous", "continuous_absolute"}:
            raise ValueError(
                "INITIAL_POLICY_WEIGHT requires an absolute continuous action mode"
            )
        initialized_policy_action = _initialize_continuous_policy_weight(
            model,
            initial_weight=float(initial_policy_weight),
            w_center=_env_float("W_CENTER", 1.65),
            w_scale=_env_float("W_SCALE", 0.1),
        )
    print(
        json.dumps(
                {
                    "stage": "rl_train_start",
                    "algo": "ppo",
                "train_trace": len(train_trace),
                "eval_trace_cases_per_seed": len(next(iter(eval_traces.values()))),
                "eval_trace_seeds": [int(x) for x in eval_traces.keys()],
                "train_episodes": _env_int("TRAIN_EPISODES", 0),
                "timesteps": _env_int("TOTAL_TIMESTEPS", 100000),
                "obs_mode": _env_str("OBS_MODE", "solve_only"),
                "action_mode": _env_str("ACTION_MODE", "continuous"),
                "w_center": _env_float("W_CENTER", 1.65),
                "w_scale": _env_float("W_SCALE", 0.1),
                "initial_observation_weight": _env_optional_float(
                    "INITIAL_OBSERVATION_WEIGHT"
                ),
                "initial_policy_weight": initial_policy_weight,
                "initialized_policy_action": initialized_policy_action,
            }
        ),
        flush=True,
    )
    total_timesteps = _env_int("TOTAL_TIMESTEPS", 100000)
    train_episodes = _env_int("TRAIN_EPISODES", 0)
    if train_episodes < 0:
        raise ValueError("TRAIN_EPISODES must be non-negative")
    if train_episodes > 0 and train_episodes != len(train_trace):
        raise ValueError(
            "Episode-budget training requires TRAIN_EPISODES to equal the "
            "number of unique training cases"
        )
    checkpoint_interval = _env_int("CHECKPOINT_INTERVAL", total_timesteps)
    checkpoint_interval = max(1, checkpoint_interval)
    select_model = _env_str("SELECT_MODEL", "final").strip().lower()
    keep_temp_checkpoints = _env_flag("KEEP_TEMP_CHECKPOINTS", "0")
    if select_model not in {"final", "best"}:
        raise ValueError(f"Unsupported SELECT_MODEL={select_model!r}")

    steps_done = 0
    chunk_index = 0
    best_score = -math.inf
    best_model_path = model_base.parent / f"{model_base.name}_best.zip"
    best_selection: Dict[str, Any] | None = None
    checkpoint_history: List[Dict[str, Any]] = []

    if train_episodes > 0:
        max_training_steps = int(
            train_episodes * _env_int("SOLVE_MAX_CYCLES", 50)
        )
        stop_callback = StopTrainingOnMaxEpisodes(
            max_episodes=int(train_episodes),
            verbose=1,
        )
        model.learn(
            total_timesteps=max_training_steps,
            reset_num_timesteps=True,
            callback=stop_callback,
        )
        if int(stop_callback.n_episodes) != int(train_episodes):
            raise RuntimeError(
                "PPO episode-budget training did not consume exactly the "
                f"requested {train_episodes} episodes"
            )
        actual_timesteps = int(model.num_timesteps)
        chunk_index = 1
        checkpoint_path = (
            model_base.parent
            / f"{model_base.name}_episodes_{train_episodes}.zip"
        )
        model.save(str(checkpoint_path.with_suffix("")))
        per_eval_seed, combined_eval = _evaluate_model_on_traces(
            eval_traces, checkpoint_path
        )
        score = _score_eval_summary(combined_eval, per_eval_seed)
        record = {
            "chunk_index": int(chunk_index),
            "episodes": int(train_episodes),
            "actual_timesteps": int(actual_timesteps),
            "model": str(checkpoint_path),
            "score": float(score),
            "relative_pct": dict(combined_eval["relative_pct"]),
        }
        checkpoint_history.append(record)
        print(
            json.dumps({"stage": "checkpoint_eval_done", **record}),
            flush=True,
        )
        best_score = float(score)
        best_selection = dict(record)
        best_selection["model"] = str(best_model_path)
        best_selection["per_eval_seed"] = per_eval_seed
        best_selection["combined_eval"] = combined_eval
        checkpoint_path.replace(best_model_path)
    while train_episodes == 0 and steps_done < total_timesteps:
        learn_steps = min(checkpoint_interval, total_timesteps - steps_done)
        model.learn(
            total_timesteps=learn_steps,
            reset_num_timesteps=(steps_done == 0),
        )
        steps_done += learn_steps
        actual_timesteps = int(model.num_timesteps)
        chunk_index += 1
        checkpoint_path = model_base.parent / f"{model_base.name}_ckpt_{steps_done}.zip"
        model.save(str(checkpoint_path.with_suffix("")))
        per_eval_seed, combined_eval = _evaluate_model_on_traces(eval_traces, checkpoint_path)
        score = _score_eval_summary(combined_eval, per_eval_seed)
        record = {
            "chunk_index": int(chunk_index),
            "timesteps": int(steps_done),
            "actual_timesteps": actual_timesteps,
            "model": str(checkpoint_path),
            "score": float(score),
            "relative_pct": dict(combined_eval["relative_pct"]),
        }
        checkpoint_history.append(record)
        print(json.dumps({"stage": "checkpoint_eval_done", **record}), flush=True)
        if score > best_score:
            best_score = score
            best_selection = dict(record)
            best_selection["model"] = str(best_model_path)
            best_selection["per_eval_seed"] = per_eval_seed
            best_selection["combined_eval"] = combined_eval
            checkpoint_path.replace(best_model_path)
        elif not keep_temp_checkpoints:
            checkpoint_path.unlink(missing_ok=True)

    model.save(str(model_base))
    venv.close()

    final_model_path = model_base.with_suffix(".zip")
    selected_model_path = final_model_path
    if select_model == "best":
        if best_selection is None:
            raise RuntimeError("No checkpoint selection results were produced")
        selected_model_path = best_model_path
    selection_info = {
        "select_model": select_model,
        "checkpoint_objective": _env_str("CHECKPOINT_OBJECTIVE", "avg_vs_fixed_bandit"),
        "budget_unit": "episodes" if train_episodes > 0 else "transitions",
        "train_episodes": int(train_episodes),
        "checkpoint_interval": int(checkpoint_interval),
        "total_timesteps": int(total_timesteps),
        "actual_total_timesteps": int(model.num_timesteps),
        "final_model": str(final_model_path),
        "selected_model": str(selected_model_path),
        "best_checkpoint": best_selection,
        "checkpoint_history": checkpoint_history,
    }
    print(json.dumps({"stage": "rl_train_done", **selection_info}), flush=True)
    return selected_model_path, selection_info


def _make_runner(model_path: Path) -> SetupAwareSolvePolicyRunner:
    return SetupAwareSolvePolicyRunner(
        SetupAwareRLConfig(
            tune_dim=_env_int("SETUP_TUNE_DIM", 7),
            tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
            algo="ppo",
            model_type=_env_str("MODEL_TYPE", "mlp").strip().lower(),
            model_path=model_path,
            vec_path=Path("/tmp/nonexistent_vecnormalize.pkl"),
            fixed_grid=(_env_int("MATRIX_GRID_N", EXP44_MATRIX_GRID_N),) * 3,
            difconv_c_range=(_env_float("C_MIN", 1.0), _env_float("C_MAX", 1000.0)),
            w_only=_env_flag("W_ONLY", "1"),
            w_center=_env_float("W_CENTER", 1.65),
            w_scale=_env_float("W_SCALE", 0.1),
            w_global_min=_env_float("W_GLOBAL_MIN", 1.0),
            w_global_max=_env_float("W_GLOBAL_MAX", 2.0),
            sweeps_min=_env_int("SWEEPS_MIN", 1),
            sweeps_max=_env_int("SWEEPS_MAX", 1),
            w_init=None,
            sweeps_init=None,
            solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
            solve_tol=_env_float("SOLVE_TOL", 1e-6),
            default_setup_params=dict(DEFAULT_SETUP_PARAMS),
            obs_mode=_env_str("OBS_MODE", "solve_only"),
            action_mode=_env_str("ACTION_MODE", "continuous").strip().lower(),
            discrete_w_values=_env_float_tuple("DISCRETE_W_VALUES", "1.4,1.6,1.8"),
            discrete_joint_actions=_env_joint_actions("DISCRETE_ACTIONS"),
            discrete_extended_actions=_env_extended_actions("DISCRETE_EXTENDED_ACTIONS"),
            discrete_blend_alphas=_env_float_tuple("DISCRETE_BLEND_ALPHAS", "0.0,0.5,1.0"),
            initial_observation_weight=_env_optional_float(
                "INITIAL_OBSERVATION_WEIGHT"
            ),
            force_default_first_action=_env_flag(
                "FORCE_DEFAULT_FIRST_ACTION", "0"
            ),
            default_first_weight=_env_optional_float("DEFAULT_FIRST_WEIGHT"),
        )
    )


def _evaluate(eval_trace, model_path: Path) -> Dict[str, Any]:
    fixed_w = _env_float("FIXED_W", 1.60)
    runner = _make_runner(model_path)
    default_setup_results = []
    bandit_results = []
    fixed_results = []
    rl_results = []
    setup_counter = Counter()
    rng = np.random.default_rng(_env_int("EVAL_ORDER_SEED", _env_int("EVAL_SEED", 39394939) + 9001))
    permute_methods = _env_flag("EVAL_PERMUTE_METHODS", "1")
    for i, (mkw, params) in enumerate(eval_trace, 1):
        setup_counter[_param_key(params)] += 1
        case_outputs: Dict[str, Dict[str, Any]] = {}
        method_order = ["default", "bandit", "fixed", "rl"]
        if permute_methods:
            method_order = [method_order[j] for j in rng.permutation(len(method_order))]
        for method in method_order:
            if method == "default":
                case_outputs[method] = solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=_env_float("SOLVER_TOL", 1e-6),
                    solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
                    augment_params=augment_setup_params,
                )
            elif method == "bandit":
                case_outputs[method] = solve_no_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solver_tol=_env_float("SOLVER_TOL", 1e-6),
                    solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
                    augment_params=augment_setup_params,
                )
            elif method == "fixed":
                case_outputs[method] = solve_schedule_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    schedule=[(_env_int("SOLVE_MAX_CYCLES", 50), fixed_w, 1, 1)],
                    solve_tol=_env_float("SOLVE_TOL", 1e-6),
                    solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
                )
            elif method == "rl":
                case_outputs[method] = solve_setup_aware_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solve_policy=runner,
                    augment_params=augment_setup_params,
                    classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=_env_float("SOLVE_TOL", 1e-6),
                        solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
                    ),
                    solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
                    case_progress=0.0 if len(eval_trace) <= 1 else float(i - 1) / float(len(eval_trace) - 1),
                )
        default_setup_results.append(case_outputs["default"])
        bandit_results.append(case_outputs["bandit"])
        fixed_results.append(case_outputs["fixed"])
        rl_results.append(case_outputs["rl"])
        if i % max(1, _env_int("PROGRESS_EVERY", 100)) == 0 or i == len(eval_trace):
            print(json.dumps({"stage": "eval_progress", "done": i, "total": len(eval_trace)}), flush=True)

    s_default = _summarize(default_setup_results)
    s_bandit = _summarize(bandit_results)
    s_fixed = _summarize(fixed_results)
    s_rl = _summarize(rl_results)
    setup_top = [
        {"count": int(c), "share_pct": float(100.0 * c / len(eval_trace)), "params": dict(k)}
        for k, c in setup_counter.most_common(10)
    ]
    return {
        "methods": {
            "default_setup_default_solve": s_default,
            "mature_bandit_default_solve": s_bandit,
            f"mature_bandit_fixed_w_{fixed_w:.2f}": s_fixed,
            "mature_bandit_rl": s_rl,
        },
        "relative_pct": {
            "bandit_vs_default_setup": float(100.0 * (s_default["mean_runtime"] - s_bandit["mean_runtime"]) / s_default["mean_runtime"]),
            f"fixed_w_{fixed_w:.2f}_vs_bandit": float(100.0 * (s_bandit["mean_runtime"] - s_fixed["mean_runtime"]) / s_bandit["mean_runtime"]),
            "rl_vs_bandit": float(100.0 * (s_bandit["mean_runtime"] - s_rl["mean_runtime"]) / s_bandit["mean_runtime"]),
            f"rl_vs_fixed_w_{fixed_w:.2f}": float(100.0 * (s_fixed["mean_runtime"] - s_rl["mean_runtime"]) / s_fixed["mean_runtime"]),
            "rl_vs_default_setup": float(100.0 * (s_default["mean_runtime"] - s_rl["mean_runtime"]) / s_default["mean_runtime"]),
        },
        "selected_setup_distribution": {
            "unique_setups": int(len(setup_counter)),
            "top10": setup_top,
        },
    }


def _combine_eval_summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    method_names = list(rows[0]["methods"].keys())
    combined_methods: Dict[str, Any] = {}
    for method in method_names:
        cases = sum(int(row["methods"][method]["cases"]) for row in rows)
        totals = {
            key: sum(float(row["methods"][method][key]) for row in rows)
            for key in (
                "total_runtime",
                "total_setup_runtime",
                "total_solve_runtime",
                "total_infer_runtime",
                "total_runtime_with_controller",
            )
        }
        failed_count = sum(int(row["methods"][method]["failed_count"]) for row in rows)
        combined_methods[method] = {
            "cases": int(cases),
            "mean_runtime": float(totals["total_runtime"] / max(1, cases)),
            "total_runtime": float(totals["total_runtime"]),
            "mean_setup_runtime": float(totals["total_setup_runtime"] / max(1, cases)),
            "total_setup_runtime": float(totals["total_setup_runtime"]),
            "mean_solve_runtime": float(totals["total_solve_runtime"] / max(1, cases)),
            "total_solve_runtime": float(totals["total_solve_runtime"]),
            "mean_infer_runtime": float(totals["total_infer_runtime"] / max(1, cases)),
            "total_infer_runtime": float(totals["total_infer_runtime"]),
            "mean_runtime_with_controller": float(
                totals["total_runtime_with_controller"] / max(1, cases)
            ),
            "total_runtime_with_controller": float(totals["total_runtime_with_controller"]),
            "failed_count": int(failed_count),
        }
    fixed_w = _env_float("FIXED_W", 1.60)
    bandit_key = "mature_bandit_default_solve"
    fixed_key = f"mature_bandit_fixed_w_{fixed_w:.2f}"
    rl_key = "mature_bandit_rl"
    default_key = "default_setup_default_solve"
    return {
        "methods": combined_methods,
        "relative_pct": {
            "bandit_vs_default_setup": float(
                100.0
                * (combined_methods[default_key]["mean_runtime"] - combined_methods[bandit_key]["mean_runtime"])
                / combined_methods[default_key]["mean_runtime"]
            ),
            f"fixed_w_{fixed_w:.2f}_vs_bandit": float(
                100.0
                * (combined_methods[bandit_key]["mean_runtime"] - combined_methods[fixed_key]["mean_runtime"])
                / combined_methods[bandit_key]["mean_runtime"]
            ),
            "rl_vs_bandit": float(
                100.0
                * (combined_methods[bandit_key]["mean_runtime"] - combined_methods[rl_key]["mean_runtime"])
                / combined_methods[bandit_key]["mean_runtime"]
            ),
            f"rl_vs_fixed_w_{fixed_w:.2f}": float(
                100.0
                * (combined_methods[fixed_key]["mean_runtime"] - combined_methods[rl_key]["mean_runtime"])
                / combined_methods[fixed_key]["mean_runtime"]
            ),
            "rl_vs_default_setup": float(
                100.0
                * (combined_methods[default_key]["mean_runtime"] - combined_methods[rl_key]["mean_runtime"])
                / combined_methods[default_key]["mean_runtime"]
            ),
        },
    }


def main() -> None:
    branch = _warmup_bandit()
    train_trace, train_meta = _build_train_trace(branch)
    eval_seeds = _env_int_tuple("EVAL_SEEDS", "")
    if not eval_seeds:
        eval_seeds = (_env_int("EVAL_SEED", 39394939),)
    eval_cases = _env_int("EVAL_CASES", 500)
    eval_traces: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for seed in eval_seeds:
        print(
            json.dumps(
                {
                    "stage": "eval_trace_build_start",
                    "seed": int(seed),
                    "cases": int(eval_cases),
                }
            ),
            flush=True,
        )
        eval_traces[int(seed)] = _frozen_trace(branch, cases=eval_cases, seed=int(seed))
    first_eval_seed = int(eval_seeds[0])
    model_path, selection_info = _train_rl(train_trace, eval_traces)
    per_eval_seed, combined_eval = _evaluate_model_on_traces(eval_traces, model_path)
    result = {
        "stage": "final",
        "protocol": {
            "grid": [_env_int("MATRIX_GRID_N", EXP44_MATRIX_GRID_N)] * 3,
            "setup_param_resolution": _env_int(
                "SETUP_PARAM_RESOLUTION", EXP44_SETUP_PARAM_RESOLUTION
            ),
            "warmup_cases": _env_int("WARMUP_CASES", 1500),
            "warmup_seed": _env_int("WARMUP_SEED", 39393939),
            "bandit_frozen_after_warmup": True,
            "rl_train_cases_per_seed": _env_int("RL_TRAIN_CASES", 500),
            "rl_train_seeds": train_meta["train_seeds"],
            "rl_train_total_cases": train_meta["total_cases"],
            "rl_train_episode_budget": _env_int("TRAIN_EPISODES", 0),
            "train_trace_shuffle": train_meta["shuffle"],
            "train_trace_shuffle_seed": train_meta["shuffle_seed"],
            "eval_cases": int(eval_cases),
            "eval_seeds": [int(x) for x in eval_seeds],
            "eval_bandit_updates": False,
        },
        "rl_setting": {
            "algo": "ppo",
            "obs_mode": _env_str("OBS_MODE", "solve_only"),
            "action_mode": _env_str("ACTION_MODE", "continuous"),
            "w_center": _env_float("W_CENTER", 1.65),
            "w_scale": _env_float("W_SCALE", 0.1),
            "initial_observation_weight": _env_optional_float(
                "INITIAL_OBSERVATION_WEIGHT"
            ),
            "initial_policy_weight": _env_optional_float(
                "INITIAL_POLICY_WEIGHT"
            ),
            "reward_mode": _env_int("REWARD_MODE", 4),
            "total_timesteps": _env_int("TOTAL_TIMESTEPS", 100000),
            "actual_total_timesteps": int(
                selection_info.get(
                    "actual_total_timesteps", _env_int("TOTAL_TIMESTEPS", 100000)
                )
            ),
            "actual_train_episodes": int(
                selection_info.get("train_episodes", 0)
            ),
            "model": str(model_path),
        },
        "checkpoint_selection": selection_info,
        "per_eval_seed": per_eval_seed,
        "combined_eval": combined_eval,
    }
    result_path_raw = _env_str("RESULT_PATH", "").strip()
    if result_path_raw:
        result_path = Path(result_path_raw)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "result_write", "path": str(result_path)}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
