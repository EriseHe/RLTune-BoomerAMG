from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from solve.core.amg_gym_env import build_policy_obs
from experiments.joint.solve_control.frozen_bandit_step_env import FrozenBanditStepEnv
from solve.controllers.ppo import SetupAwareRLConfig, SetupAwareSolvePolicyRunner
from hypre.bindings import augment_setup_params
from experiments.joint.solve_control.evaluation import eval_runner, fixed_trace
from experiments.joint.solve_control.native_evaluation import solve_no_rl_case
from hypre.bindings import create_env

# Active Exp44 path:
# - MODEL_TYPE=lstm
# - ACTION_MODE=continuous_residual
# - no retained VecNormalize file on disk
#
# Older BC warmstart, DQN, and VecNormalize branches were removed from the
# retained Exp44 training path. This file now keeps only the PPO-based frozen
# trace trainer used by the current workflow.


def _trace_cache_dir() -> Path:
    return Path(os.environ.get("TRACE_CACHE_DIR", "/tmp/frozen_bandit_trace_cache"))


def _trace_cache_key(
    *,
    T: int,
    grid: tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str,
) -> str:
    relevant_env = {
        k: v
        for k, v in os.environ.items()
        if (
            k.startswith("TRACE_")
            or k.startswith("DIFCONV_")
            or k.startswith("C_")
            or k.startswith("SOLVE_")
            or k.startswith("SOLVER_")
            or k.startswith("SETUP_")
            or k in {"TUNE7_VARIANT", "BRANCH_FILTER", "METHOD_FILTER", "MATRIX_GRID_N"}
        )
    }
    payload = {
        "T": int(T),
        "grid": tuple(int(x) for x in grid),
        "seed": int(seed),
        "tune_dim": int(tune_dim),
        "bandit_method": str(bandit_method),
        "solve_mode": str(solve_mode),
        "env": dict(sorted(relevant_env.items())),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cached_fixed_trace(
    *,
    T: int,
    grid: tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str = "no_rl",
    solve_policy: SetupAwareSolvePolicyRunner | None = None,
):
    cache_dir = _trace_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _trace_cache_key(
        T=int(T),
        grid=tuple(int(x) for x in grid),
        seed=int(seed),
        tune_dim=int(tune_dim),
        bandit_method=str(bandit_method),
        solve_mode=str(solve_mode),
    )
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            trace = pickle.load(fh)
        print(
            f"trace_cache_hit path={cache_path} T={int(T)} grid={tuple(int(x) for x in grid)} seed={int(seed)} mode={solve_mode}",
            flush=True,
        )
        return trace
    trace = fixed_trace(
        T=int(T),
        grid=tuple(int(x) for x in grid),
        seed=int(seed),
        tune_dim=int(tune_dim),
        bandit_method=str(bandit_method),
        solve_mode=str(solve_mode),
        solve_policy=solve_policy,
    )
    with cache_path.open("wb") as fh:
        pickle.dump(trace, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"trace_cache_write path={cache_path} T={int(T)} grid={tuple(int(x) for x in grid)} seed={int(seed)} mode={solve_mode}",
        flush=True,
    )
    return trace


def _skip_trace_prefix(trace, skip_prefix: int):
    skip = max(0, int(skip_prefix))
    if skip <= 0:
        return list(trace)
    if skip >= len(trace):
        return [trace[-1]]
    return list(trace[skip:])


def _setup_action_key(params: dict) -> tuple:
    items = []
    for key, value in sorted(dict(params).items()):
        if isinstance(value, float):
            items.append((key, round(float(value), 12)))
        else:
            items.append((key, value))
    return tuple(items)


def _filter_trace_to_cluster(trace, *, cluster_rank: int, cluster_max_cases: int):
    if not trace:
        return []
    rank = max(1, int(cluster_rank))
    max_cases = int(cluster_max_cases)
    counts = {}
    order = []
    for _mkw, params in trace:
        k = _setup_action_key(params)
        if k not in counts:
            counts[k] = 0
            order.append(k)
        counts[k] += 1
    sorted_keys = sorted(order, key=lambda k: (-counts[k], k))
    idx = min(rank - 1, len(sorted_keys) - 1)
    target = sorted_keys[idx]
    filtered = [(mkw, params) for (mkw, params) in trace if _setup_action_key(params) == target]
    if max_cases > 0:
        filtered = filtered[:max_cases]
    stats = {
        "num_unique": len(sorted_keys),
        "selected_rank": rank,
        "selected_count": counts[target],
        "total_count": len(trace),
        "selected_key": target,
    }
    return filtered, stats


def _parse_discrete_joint_actions(raw: str) -> tuple[tuple[float, int, int], ...]:
    actions = []
    for spec in raw.split(";"):
        parts = [p.strip() for p in spec.split(":") if p.strip()]
        if len(parts) != 3:
            continue
        actions.append((float(parts[0]), int(parts[1]), int(parts[2])))
    return tuple(actions)


def _parse_discrete_extended_actions(raw: str) -> tuple[tuple[float, int, int, int, int, int, float, float], ...]:
    actions = []
    for spec in raw.split(";"):
        parts = [p.strip() for p in spec.split(":") if p.strip()]
        if len(parts) not in {6, 7, 8}:
            continue
        actions.append(
            (
                float(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
                (-1.0 if len(parts) <= 6 else float(parts[6])),
                (-1.0 if len(parts) <= 7 else float(parts[7])),
            )
        )
    return tuple(actions)


def _parse_bc_action_schedule(raw: str, max_cycles: int) -> tuple[tuple[int, int], ...]:
    items = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            continue
        end_s, idx_s = spec.split(":", 1)
        items.append((int(end_s.strip()), int(idx_s.strip())))
    if not items:
        return ((int(max_cycles), 0),)
    items.sort(key=lambda x: x[0])
    if items[-1][0] < int(max_cycles):
        items.append((int(max_cycles), items[-1][1]))
    return tuple(items)


def _parse_teacher_action_schedule(raw: str, max_cycles: int) -> tuple[tuple[int, int], ...]:
    return _parse_bc_action_schedule(raw, max_cycles)


def _teacher_index_for_cycle(schedule: tuple[tuple[int, int], ...], cycle: int) -> int:
    chosen = schedule[-1][1]
    for end_cycle, idx in schedule:
        if cycle < int(end_cycle):
            chosen = int(idx)
            break
    return int(chosen)


def _parse_float_tuple_env(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(raw).split(",") if x.strip())


def _parse_single_discrete_extended_action(raw: str, default_spec: tuple[float, int, int, int, int, int, float, float]) -> tuple[float, int, int, int, int, int, float, float]:
    parsed = _parse_discrete_extended_actions(str(raw))
    if parsed:
        return parsed[0]
    return tuple(default_spec)


def _parse_grid_specs_env(raw: str, *, default_grid: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    specs = []
    for spec in str(raw).split(";"):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 3:
            continue
        specs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return tuple(specs) or (tuple(int(x) for x in default_grid),)


def _make_env(trace, *, seed: int):
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "30,30,30").split(","))
    del grid
    action_mode = os.environ.get("ACTION_MODE", "continuous").strip().lower()
    discrete_w_values = tuple(
        float(x.strip()) for x in os.environ.get("DISCRETE_W_VALUES", "1.4,1.6,1.8").split(",") if x.strip()
    )
    discrete_joint_actions = _parse_discrete_joint_actions(os.environ.get("DISCRETE_ACTIONS", ""))
    discrete_extended_actions = _parse_discrete_extended_actions(os.environ.get("DISCRETE_EXTENDED_ACTIONS", ""))
    discrete_blend_alphas = _parse_float_tuple_env(os.environ.get("DISCRETE_BLEND_ALPHAS", "0.0,0.5,1.0"))
    blend_safe_action = _parse_single_discrete_extended_action(
        os.environ.get("BLEND_SAFE_ACTION", ""),
        (1.6, 1, 1, 1, 1, 18, -1.0, -1.0),
    )
    blend_aggr_action = _parse_single_discrete_extended_action(
        os.environ.get("BLEND_AGGR_ACTION", ""),
        (1.6, 1, 1, 2, 1, 18, -1.0, 0.1),
    )
    blend_decay_tau = float(os.environ.get("BLEND_DECAY_TAU", "0.0"))
    blend_cutoff_cycles = int(os.environ.get("BLEND_CUTOFF_CYCLES", "-1"))
    blend_progress_start = float(os.environ.get("BLEND_PROGRESS_START", "0.0"))
    blend_progress_end = float(os.environ.get("BLEND_PROGRESS_END", "0.0"))
    max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    teacher_action_schedule = _parse_teacher_action_schedule(
        os.environ.get("TEACHER_ACTION_SCHEDULE", ""),
        max_cycles=max_cycles,
    )
    return FrozenBanditStepEnv(
        trace=trace,
        tune_dim=int(os.environ.get("SETUP_TUNE_DIM", "5")),
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        c_max=float(os.environ.get("C_MAX", os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")[-1])),
        tol=float(os.environ.get("SOLVE_TOL", "1e-6")),
        max_cycles=max_cycles,
        w_only=os.environ.get("W_ONLY", "1").strip().lower() not in {"0", "false", "no"},
        action_mode=action_mode,
        discrete_w_values=discrete_w_values,
        discrete_joint_actions=discrete_joint_actions,
        discrete_extended_actions=discrete_extended_actions,
        discrete_blend_alphas=discrete_blend_alphas,
        blend_safe_action=blend_safe_action,
        blend_aggr_action=blend_aggr_action,
        blend_decay_tau=blend_decay_tau,
        blend_cutoff_cycles=blend_cutoff_cycles,
        blend_progress_start=blend_progress_start,
        blend_progress_end=blend_progress_end,
        w_center=float(os.environ.get("W_CENTER", "1.25")),
        w_scale=float(os.environ.get("W_SCALE", "0.75")),
        w_global_min=float(os.environ.get("W_GLOBAL_MIN", "1.0")),
        w_global_max=float(os.environ.get("W_GLOBAL_MAX", "2.0")),
        sweeps_min=int(os.environ.get("SWEEPS_MIN", "1")),
        sweeps_max=int(os.environ.get("SWEEPS_MAX", "1")),
        reward_mode=int(os.environ.get("REWARD_MODE", "3")),
        term_bonus=float(os.environ.get("TERM_BONUS", "0.0")),
        relative_bonus_scale=float(os.environ.get("RELATIVE_BONUS_SCALE", "1.0")),
        obs_mode=os.environ.get("OBS_MODE", "full"),
        obs_include_trace_progress=os.environ.get("OBS_INCLUDE_TRACE_PROGRESS", "0").strip().lower() not in {"0", "false", "no"},
        teacher_action_schedule=teacher_action_schedule,
        teacher_match_bonus=float(os.environ.get("TEACHER_MATCH_BONUS", "0.0")),
        teacher_mismatch_penalty=float(os.environ.get("TEACHER_MISMATCH_PENALTY", "0.0")),
        seed=int(seed),
    )


def _make_trace_policy(
    *,
    grid: tuple[int, int, int],
    tune_dim: int,
    action_mode: str,
    discrete_w_values,
    discrete_joint_actions,
    discrete_extended_actions,
    discrete_blend_alphas,
    blend_safe_action,
    blend_aggr_action,
    blend_decay_tau,
    blend_cutoff_cycles,
    blend_progress_start,
    blend_progress_end,
):
    trace_policy_mode = os.environ.get("TRACE_POLICY_MODE", "model").strip().lower()
    if trace_policy_mode == "schedule":
        return _ScheduledTracePolicyRunner(
            solve_tol=float(os.environ.get("TRACE_SOLVE_TOL", os.environ.get("SOLVE_TOL", "1e-6"))),
            solve_max_cycles=int(os.environ.get("TRACE_SOLVE_MAX_CYCLES", os.environ.get("SOLVE_MAX_CYCLES", "50"))),
            action_schedule=_parse_teacher_action_schedule(
                os.environ.get("TRACE_ACTION_SCHEDULE", "4:1;50:0"),
                int(os.environ.get("TRACE_SOLVE_MAX_CYCLES", os.environ.get("SOLVE_MAX_CYCLES", "50"))),
            ),
            discrete_extended_actions=(
                _parse_discrete_extended_actions(os.environ.get("TRACE_DISCRETE_EXTENDED_ACTIONS", ""))
                or discrete_extended_actions
            ),
        )
    trace_model_path = Path(os.environ["TRACE_MODEL_PATH"])
    trace_vec_path = Path(os.environ.get("TRACE_VEC_PATH", "/tmp/none.pkl"))
    return SetupAwareSolvePolicyRunner(
        SetupAwareRLConfig(
            tune_dim=int(tune_dim),
            tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
            algo=os.environ.get("TRACE_ALGO", os.environ.get("ALGO", "ppo")).strip().lower(),
            model_type=os.environ.get("TRACE_MODEL_TYPE", os.environ.get("MODEL_TYPE", "mlp")).strip().lower(),
            model_path=trace_model_path,
            vec_path=trace_vec_path,
            fixed_grid=grid,
            difconv_c_range=tuple(float(x) for x in os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")),
            w_only=os.environ.get("TRACE_W_ONLY", os.environ.get("W_ONLY", "1")).strip().lower() not in {"0", "false", "no"},
            w_center=float(os.environ.get("TRACE_W_CENTER", os.environ.get("W_CENTER", "1.25"))),
            w_scale=float(os.environ.get("TRACE_W_SCALE", os.environ.get("W_SCALE", "0.75"))),
            sweeps_min=int(os.environ.get("TRACE_SWEEPS_MIN", os.environ.get("SWEEPS_MIN", "1"))),
            sweeps_max=int(os.environ.get("TRACE_SWEEPS_MAX", os.environ.get("SWEEPS_MAX", "1"))),
            w_init=None,
            sweeps_init=None,
            solve_max_cycles=int(os.environ.get("TRACE_SOLVE_MAX_CYCLES", os.environ.get("SOLVE_MAX_CYCLES", "50"))),
            solve_tol=float(os.environ.get("TRACE_SOLVE_TOL", os.environ.get("SOLVE_TOL", "1e-6"))),
            default_setup_params={},
            obs_mode=os.environ.get("TRACE_OBS_MODE", os.environ.get("OBS_MODE", "full")),
            obs_include_trace_progress=os.environ.get("TRACE_OBS_INCLUDE_TRACE_PROGRESS", os.environ.get("OBS_INCLUDE_TRACE_PROGRESS", "0")).strip().lower() not in {"0", "false", "no"},
            action_mode=os.environ.get("TRACE_ACTION_MODE", action_mode).strip().lower(),
            discrete_w_values=tuple(
                float(x.strip())
                for x in os.environ.get("TRACE_DISCRETE_W_VALUES", "").split(",")
                if x.strip()
            ) or discrete_w_values,
            discrete_joint_actions=_parse_discrete_joint_actions(os.environ.get("TRACE_DISCRETE_ACTIONS", ""))
            or discrete_joint_actions,
            discrete_extended_actions=_parse_discrete_extended_actions(
                os.environ.get("TRACE_DISCRETE_EXTENDED_ACTIONS", "")
            )
            or discrete_extended_actions,
            discrete_blend_alphas=_parse_float_tuple_env(
                os.environ.get("TRACE_DISCRETE_BLEND_ALPHAS", ",".join(str(x) for x in discrete_blend_alphas))
            )
            or discrete_blend_alphas,
            blend_safe_action=_parse_single_discrete_extended_action(
                os.environ.get("TRACE_BLEND_SAFE_ACTION", ""),
                blend_safe_action,
            ),
            blend_aggr_action=_parse_single_discrete_extended_action(
                os.environ.get("TRACE_BLEND_AGGR_ACTION", ""),
                blend_aggr_action,
            ),
            blend_decay_tau=float(os.environ.get("TRACE_BLEND_DECAY_TAU", str(blend_decay_tau))),
            blend_cutoff_cycles=int(os.environ.get("TRACE_BLEND_CUTOFF_CYCLES", str(blend_cutoff_cycles))),
            blend_progress_start=float(os.environ.get("TRACE_BLEND_PROGRESS_START", str(blend_progress_start))),
            blend_progress_end=float(os.environ.get("TRACE_BLEND_PROGRESS_END", str(blend_progress_end))),
        )
    )


class _ScheduledTracePolicyRunner:
    def __init__(
        self,
        *,
        solve_tol: float,
        solve_max_cycles: int,
        action_schedule: tuple[tuple[int, int], ...],
        discrete_extended_actions,
    ) -> None:
        self.cfg = SimpleNamespace(solve_tol=float(solve_tol), solve_max_cycles=int(solve_max_cycles))
        self.action_schedule = tuple((int(end_cycle), int(idx)) for end_cycle, idx in action_schedule)
        self.discrete_extended_actions = tuple(
            (
                float(spec[0]),
                int(spec[1]),
                int(spec[2]),
                int(spec[3]),
                int(spec[4]),
                int(spec[5]),
                (-1.0 if len(spec) < 7 else float(spec[6])),
                (-1.0 if len(spec) < 8 else float(spec[7])),
            )
            for spec in discrete_extended_actions
        )
        if not self.discrete_extended_actions:
            raise ValueError("TRACE_POLICY_MODE=schedule requires TRACE_DISCRETE_EXTENDED_ACTIONS or discrete_extended_actions")

    def run(self, env, *, mkw: Dict[str, Any], setup_params: Dict[str, Any]) -> Dict[str, Any]:
        del mkw, setup_params
        solve_runtime = 0.0
        cycles = 0
        action_counts: Dict[int, int] = {}
        last = self.discrete_extended_actions[0]
        r_cur = float(env.r0)
        for cycle in range(int(self.cfg.solve_max_cycles)):
            idx = _teacher_index_for_cycle(self.action_schedule, cycle)
            idx = int(np.clip(idx, 0, len(self.discrete_extended_actions) - 1))
            action_counts[idx] = int(action_counts.get(idx, 0)) + 1
            last = self.discrete_extended_actions[idx]
            w, sd, su, sc, ct, rt, ow, arw = last
            r_cur, dt = env.step_rl(
                relax_weight=float(w),
                sweeps_down=int(sd),
                sweeps_up=int(su),
                coarse_sweeps=int(sc),
                cycle_type=(None if int(ct) < 0 else int(ct)),
                relax_type=(None if int(rt) < 0 else int(rt)),
                outer_weight=(None if float(ow) < 0.0 else float(ow)),
                add_relax_weight=(None if float(arw) < 0.0 else float(arw)),
                tol=float(self.cfg.solve_tol),
                max_cycles=int(self.cfg.solve_max_cycles),
            )
            solve_runtime += float(dt)
            cycles = cycle + 1
            if int(env.last_step.status) != 0:
                break
        w, sd, su, sc, ct, rt, ow, arw = last
        return {
            "solve_runtime": float(solve_runtime),
            "infer_runtime": 0.0,
            "residual_norm": float(r_cur),
            "iterations": int(cycles),
            "failed": int(env.last_step.status) != 1,
            "attempt_status": (
                "success" if int(env.last_step.status) == 1 else "nonconvergence"
            ),
            "native_status": env.last_step.status.name.lower(),
            "final_w": float(w),
            "final_sweeps_down": int(sd),
            "final_sweeps_up": int(su),
            "final_coarse_sweeps": int(sc),
            "final_cycle_type": int(ct),
            "final_relax_type": int(rt),
            "final_outer_weight": float(ow),
            "final_add_relax_weight": float(arw),
            "action_counts": dict(action_counts),
        }


class _PeriodicTraceEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_traces,
        eval_cfg_kwargs,
        bandit_only_means,
        eval_freq: int,
        best_model_basename: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_traces = list(eval_traces)
        self.eval_cfg_kwargs = dict(eval_cfg_kwargs)
        self.bandit_only_means = [float(x) for x in bandit_only_means]
        self.eval_freq = max(1, int(eval_freq))
        self.best_model_basename = Path(best_model_basename)
        self.best_delta = float("inf")
        self.best_summary = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        tmp_base = self.best_model_basename.with_name(self.best_model_basename.name + "_tmp_eval")
        self.model.save(str(tmp_base))
        runner = SetupAwareSolvePolicyRunner(
            SetupAwareRLConfig(
                model_path=tmp_base.with_suffix(".zip"),
                **self.eval_cfg_kwargs,
            )
        )
        per_trace = []
        for eval_trace, bandit_only_mean in zip(self.eval_traces, self.bandit_only_means):
            summary = eval_runner(runner, eval_trace)
            delta = float(summary["mean_runtime"] - float(bandit_only_mean))
            per_trace.append(
                {
                    "summary": dict(summary),
                    "bandit_only_mean": float(bandit_only_mean),
                    "delta": float(delta),
                }
            )
        delta = float(np.mean([item["delta"] for item in per_trace]))
        if self.verbose:
            details = " ".join(
                f"trace{i}=({item['bandit_only_mean']:.6f}->{item['summary']['mean_runtime']:.6f},{item['delta']:+.6f})"
                for i, item in enumerate(per_trace)
            )
            print(
                f"trace_eval step={self.num_timesteps} avg_delta={delta:+.6f} {details}",
                flush=True,
            )
        if delta < self.best_delta:
            self.best_delta = delta
            self.best_summary = {
                "avg_delta": float(delta),
                "per_trace": per_trace,
            }
            self.model.save(str(self.best_model_basename))
            if self.verbose:
                print(f"trace_eval best step={self.num_timesteps} delta={delta:+.6f}", flush=True)
        return True


def main() -> None:
    seed = int(os.environ.get("SEED", "39393939"))
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "30,30,30").split(","))
    train_grids = _parse_grid_specs_env(os.environ.get("TRAIN_GRID_SIZES", ""), default_grid=grid)
    eval_grids = _parse_grid_specs_env(os.environ.get("EVAL_GRID_SIZES", ""), default_grid=grid)
    tune_dim = int(os.environ.get("SETUP_TUNE_DIM", "5"))
    bandit_method = os.environ.get("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    train_T = int(os.environ.get("TRAIN_TRACE_T", "64"))
    eval_T = int(os.environ.get("EVAL_TRACE_T", "32"))
    action_mode = os.environ.get("ACTION_MODE", "continuous").strip().lower()
    discrete_w_values = tuple(
        float(x.strip()) for x in os.environ.get("DISCRETE_W_VALUES", "1.4,1.6,1.8").split(",") if x.strip()
    )
    discrete_joint_actions = _parse_discrete_joint_actions(os.environ.get("DISCRETE_ACTIONS", ""))
    discrete_extended_actions = _parse_discrete_extended_actions(os.environ.get("DISCRETE_EXTENDED_ACTIONS", ""))
    discrete_blend_alphas = _parse_float_tuple_env(os.environ.get("DISCRETE_BLEND_ALPHAS", "0.0,0.5,1.0"))
    blend_safe_action = _parse_single_discrete_extended_action(
        os.environ.get("BLEND_SAFE_ACTION", ""),
        (1.6, 1, 1, 1, 1, 18, -1.0, -1.0),
    )
    blend_aggr_action = _parse_single_discrete_extended_action(
        os.environ.get("BLEND_AGGR_ACTION", ""),
        (1.6, 1, 1, 2, 1, 18, -1.0, 0.1),
    )
    blend_decay_tau = float(os.environ.get("BLEND_DECAY_TAU", "0.0"))
    blend_cutoff_cycles = int(os.environ.get("BLEND_CUTOFF_CYCLES", "-1"))
    blend_progress_start = float(os.environ.get("BLEND_PROGRESS_START", "0.0"))
    blend_progress_end = float(os.environ.get("BLEND_PROGRESS_END", "0.0"))
    trace_solve_mode = os.environ.get("TRACE_SOLVE_MODE", "no_rl").strip().lower()
    trace_solve_policy = None
    if trace_solve_mode == "rl":
        trace_solve_policy = _make_trace_policy(
            grid=grid,
            tune_dim=tune_dim,
            action_mode=action_mode,
            discrete_w_values=discrete_w_values,
            discrete_joint_actions=discrete_joint_actions,
            discrete_extended_actions=discrete_extended_actions,
            discrete_blend_alphas=discrete_blend_alphas,
            blend_safe_action=blend_safe_action,
            blend_aggr_action=blend_aggr_action,
            blend_decay_tau=blend_decay_tau,
            blend_cutoff_cycles=blend_cutoff_cycles,
            blend_progress_start=blend_progress_start,
            blend_progress_end=blend_progress_end,
        )
    train_seed_offsets = tuple(
        int(x.strip())
        for x in os.environ.get("TRAIN_TRACE_SEED_OFFSETS", "0").split(",")
        if x.strip()
    )
    train_trace_skip_prefix = int(os.environ.get("TRAIN_TRACE_SKIP_PREFIX", "0"))
    train_cluster_rank = int(os.environ.get("TRAIN_TRACE_CLUSTER_RANK", "0"))
    train_cluster_max_cases = int(os.environ.get("TRAIN_TRACE_CLUSTER_MAX_CASES", "0"))
    train_trace_parts = [
        item
        for grid_item in train_grids
        for offset in train_seed_offsets
        for item in _skip_trace_prefix(
            _cached_fixed_trace(
                T=train_T,
                grid=tuple(int(x) for x in grid_item),
                seed=seed + int(offset),
                tune_dim=tune_dim,
                bandit_method=bandit_method,
                solve_mode=trace_solve_mode,
                solve_policy=trace_solve_policy,
            ),
            train_trace_skip_prefix,
        )
    ]
    if train_cluster_rank > 0:
        train_trace, train_cluster_stats = _filter_trace_to_cluster(
            train_trace_parts,
            cluster_rank=train_cluster_rank,
            cluster_max_cases=train_cluster_max_cases,
        )
    else:
        train_trace = list(train_trace_parts)
        train_cluster_stats = None
    eval_seed_offsets = tuple(
        int(x.strip())
        for x in os.environ.get("TRACE_EVAL_SEED_OFFSETS", "1000").split(",")
        if x.strip()
    )
    eval_trace_skip_prefix = int(os.environ.get("EVAL_TRACE_SKIP_PREFIX", "0"))
    eval_cluster_rank = int(os.environ.get("EVAL_TRACE_CLUSTER_RANK", "0"))
    eval_cluster_max_cases = int(os.environ.get("EVAL_TRACE_CLUSTER_MAX_CASES", "0"))
    eval_traces = []
    eval_cluster_stats = []
    for grid_item in eval_grids:
        for offset in eval_seed_offsets:
            base_trace = _skip_trace_prefix(
                _cached_fixed_trace(
                    T=eval_T,
                    grid=tuple(int(x) for x in grid_item),
                    seed=seed + int(offset),
                    tune_dim=tune_dim,
                    bandit_method=bandit_method,
                    solve_mode=trace_solve_mode,
                    solve_policy=trace_solve_policy,
                ),
                eval_trace_skip_prefix,
            )
            if eval_cluster_rank > 0:
                filtered, stats = _filter_trace_to_cluster(
                    base_trace,
                    cluster_rank=eval_cluster_rank,
                    cluster_max_cases=eval_cluster_max_cases,
                )
                eval_traces.append(filtered)
                eval_cluster_stats.append(stats)
            else:
                eval_traces.append(base_trace)
                eval_cluster_stats.append(None)
    print(f"train_grids={train_grids}")
    print(f"eval_grids={eval_grids}")
    print(f"trace_solve_mode={trace_solve_mode}")
    if train_cluster_stats is not None:
        print(
            "train_cluster_stats="
            f"rank={train_cluster_stats['selected_rank']} "
            f"selected_count={train_cluster_stats['selected_count']} "
            f"total_count={train_cluster_stats['total_count']} "
            f"num_unique={train_cluster_stats['num_unique']}"
        )
    if any(s is not None for s in eval_cluster_stats):
        compact = [
            None
            if s is None
            else {
                "rank": s["selected_rank"],
                "selected_count": s["selected_count"],
                "total_count": s["total_count"],
                "num_unique": s["num_unique"],
            }
            for s in eval_cluster_stats
        ]
        print(f"eval_cluster_stats={compact}")
    if trace_solve_policy is not None:
        trace_model_path = getattr(getattr(trace_solve_policy, "cfg", None), "model_path", None)
        if trace_model_path is not None:
            print(f"trace_model={trace_model_path}")
        else:
            print(f"trace_policy={type(trace_solve_policy).__name__}")

    venv = DummyVecEnv([lambda: _make_env(train_trace, seed=seed)])
    venv = VecMonitor(venv)
    model_type = os.environ.get("MODEL_TYPE", "mlp").strip().lower()
    init_model_path = os.environ.get("INIT_MODEL_PATH", "").strip()
    common_kwargs = dict(
        learning_rate=float(os.environ.get("LEARNING_RATE", "3e-4")),
        gamma=float(os.environ.get("GAMMA", "0.99")),
        verbose=1,
        device="cpu",
    )
    if init_model_path:
        init_path = str(Path(init_model_path))
        if model_type == "lstm":
            model = RecurrentPPO.load(init_path, env=venv, device="cpu")
        else:
            model = PPO.load(init_path, env=venv, device="cpu")
        if os.environ.get("OVERRIDE_LOADED_LR", "1").strip().lower() not in {"0", "false", "no"}:
            override_lr = float(os.environ.get("LEARNING_RATE", "3e-4"))
            model.learning_rate = float(override_lr)
            if hasattr(model, "lr_schedule"):
                model.lr_schedule = lambda _progress: float(override_lr)
            policy_opt = getattr(getattr(model, "policy", None), "optimizer", None)
            if policy_opt is not None:
                for group in policy_opt.param_groups:
                    group["lr"] = float(override_lr)
            print(f"loaded_model_lr_override={override_lr}")
    else:
        if model_type == "lstm":
            model = RecurrentPPO(
                "MlpLstmPolicy",
                venv,
                n_steps=int(os.environ.get("N_STEPS", "64")),
                batch_size=int(os.environ.get("BATCH_SIZE", "64")),
                n_epochs=int(os.environ.get("N_EPOCHS", "10")),
                gae_lambda=float(os.environ.get("GAE_LAMBDA", "0.95")),
                clip_range=float(os.environ.get("CLIP_RANGE", "0.2")),
                ent_coef=float(os.environ.get("ENT_COEF", "0.0")),
                vf_coef=float(os.environ.get("VF_COEF", "0.5")),
                policy_kwargs={"net_arch": {"pi": [128], "vf": [128]}, "lstm_hidden_size": 128},
                **common_kwargs,
            )
        else:
            model = PPO(
                "MlpPolicy",
                venv,
                n_steps=int(os.environ.get("N_STEPS", "64")),
                batch_size=int(os.environ.get("BATCH_SIZE", "64")),
                n_epochs=int(os.environ.get("N_EPOCHS", "10")),
                gae_lambda=float(os.environ.get("GAE_LAMBDA", "0.95")),
                clip_range=float(os.environ.get("CLIP_RANGE", "0.2")),
                ent_coef=float(os.environ.get("ENT_COEF", "0.0")),
                vf_coef=float(os.environ.get("VF_COEF", "0.5")),
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                **common_kwargs,
            )

    model_basename = Path(os.environ.get("MODEL_BASENAME", "ppo_frozen_bandit_step"))
    vec_name = Path(os.environ.get("VECNORMALIZE_NAME", "/tmp/unused_vecnormalize.pkl"))
    eval_bandit_means = [
        float(
            np.mean(
                [
                    solve_no_rl_case(
                        params=p,
                        mkw=dict(m),
                        solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
                        solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
                        augment_params=augment_setup_params,
                    )["runtime"]
                    for m, p in eval_trace
                ]
            )
        )
        for eval_trace in eval_traces
    ]
    eval_bandit = float(np.mean(eval_bandit_means))
    callback = None
    if os.environ.get("USE_TRACE_EVAL_CALLBACK", "0").strip().lower() not in {"0", "false", "no"}:
        callback = _PeriodicTraceEvalCallback(
            eval_traces=eval_traces,
            eval_cfg_kwargs=dict(
                tune_dim=int(tune_dim),
                tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
                algo="ppo",
                model_type=model_type,
                vec_path=vec_name,
                fixed_grid=grid,
                difconv_c_range=tuple(float(x) for x in os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")),
                w_only=os.environ.get("W_ONLY", "1").strip().lower() not in {"0", "false", "no"},
                w_center=float(os.environ.get("W_CENTER", "1.25")),
                w_scale=float(os.environ.get("W_SCALE", "0.75")),
                sweeps_min=int(os.environ.get("SWEEPS_MIN", "1")),
                sweeps_max=int(os.environ.get("SWEEPS_MAX", "1")),
                w_init=None,
                sweeps_init=None,
                solve_max_cycles=int(os.environ.get("SOLVE_MAX_CYCLES", "50")),
                solve_tol=float(os.environ.get("SOLVE_TOL", "1e-6")),
                default_setup_params={},
                obs_mode=os.environ.get("OBS_MODE", "full"),
                action_mode=action_mode,
                discrete_w_values=discrete_w_values,
                discrete_joint_actions=discrete_joint_actions,
                discrete_extended_actions=discrete_extended_actions,
                discrete_blend_alphas=discrete_blend_alphas,
                blend_safe_action=blend_safe_action,
                blend_aggr_action=blend_aggr_action,
                blend_decay_tau=blend_decay_tau,
                blend_cutoff_cycles=blend_cutoff_cycles,
                blend_progress_start=blend_progress_start,
                blend_progress_end=blend_progress_end,
            ),
            bandit_only_means=eval_bandit_means,
            eval_freq=int(os.environ.get("TRACE_EVAL_FREQ", "4000")),
            best_model_basename=Path(os.environ.get("BEST_MODEL_BASENAME", str(model_basename) + "_best")),
            verbose=1,
        )

    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "20000"))
    reset_num_timesteps = os.environ.get("RESET_NUM_TIMESTEPS", "1").strip().lower() not in {"0", "false", "no"}
    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps, callback=callback)
    else:
        print("Skipping RL fine-tuning because TOTAL_TIMESTEPS<=0")

    model.save(str(model_basename))
    if vec_name.exists():
        vec_name.unlink()

    cfg = SetupAwareRLConfig(
        tune_dim=int(tune_dim),
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        algo="ppo",
        model_type=model_type,
        model_path=model_basename.with_suffix(".zip"),
        vec_path=vec_name,
        fixed_grid=grid,
        difconv_c_range=tuple(float(x) for x in os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")),
        w_only=os.environ.get("W_ONLY", "1").strip().lower() not in {"0", "false", "no"},
        w_center=float(os.environ.get("W_CENTER", "1.25")),
        w_scale=float(os.environ.get("W_SCALE", "0.75")),
        sweeps_min=int(os.environ.get("SWEEPS_MIN", "1")),
        sweeps_max=int(os.environ.get("SWEEPS_MAX", "1")),
        w_init=None,
        sweeps_init=None,
        solve_max_cycles=int(os.environ.get("SOLVE_MAX_CYCLES", "50")),
        solve_tol=float(os.environ.get("SOLVE_TOL", "1e-6")),
        default_setup_params={},
        obs_mode=os.environ.get("OBS_MODE", "full"),
        obs_include_trace_progress=os.environ.get("OBS_INCLUDE_TRACE_PROGRESS", "0").strip().lower() not in {"0", "false", "no"},
        action_mode=action_mode,
        discrete_w_values=discrete_w_values,
        discrete_joint_actions=discrete_joint_actions,
        discrete_extended_actions=discrete_extended_actions,
        discrete_blend_alphas=discrete_blend_alphas,
        blend_safe_action=blend_safe_action,
        blend_aggr_action=blend_aggr_action,
        blend_decay_tau=blend_decay_tau,
        blend_cutoff_cycles=blend_cutoff_cycles,
        blend_progress_start=blend_progress_start,
        blend_progress_end=blend_progress_end,
    )
    runner = SetupAwareSolvePolicyRunner(cfg)
    eval_summary = eval_runner(runner, eval_traces[0])
    print(f"saved_model={model_basename}.zip")
    print(f"saved_vec={vec_name}")
    print(f"eval_bandit_only_mean={eval_bandit:.6f}")
    print(f"eval_rl_summary={eval_summary}")
    print(f"eval_delta_rl_vs_bandit={eval_summary['mean_runtime'] - eval_bandit:+.6f}")


if __name__ == "__main__":
    main()
