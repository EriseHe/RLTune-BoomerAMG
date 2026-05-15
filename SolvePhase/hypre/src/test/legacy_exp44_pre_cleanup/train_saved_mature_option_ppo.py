from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _solve_schedule_case
from setup_aware_compare_common import DEFAULT_SETUP_PARAMS, generate_difconv_instances, solve_no_rl_case


SETUP_KEYS = (
    "P_max_elmts",
    "agg_Pmx",
    "agg_interp_type",
    "agg_num_levels",
    "agg_tr",
    "coarsen_type",
    "interp_type",
    "max_row_sum",
    "strong_threshold",
    "trunc_factor",
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_flag(name: str, default: str = "0") -> bool:
    return _env_str(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(r["runtime"]) for r in results]
    setup = [float(r["setup_runtime"]) for r in results]
    solve = [float(r["solve_runtime"]) for r in results]
    failures = [bool(r.get("failed", False)) for r in results]
    iterations = [int(r.get("iterations", -1)) for r in results if int(r.get("iterations", -1)) >= 0]
    return {
        "cases": int(len(results)),
        "mean_runtime": float(mean(runtimes)),
        "total_runtime": float(sum(runtimes)),
        "mean_setup_runtime": float(mean(setup)),
        "mean_solve_runtime": float(mean(solve)),
        "failed_count": int(sum(failures)),
        "mean_iterations": float(mean(iterations)) if iterations else float("nan"),
    }


def _features(mkw: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    c_max = max(_env_float("C_MAX", 1000.0), 1.0)
    denom = max(float(np.log(c_max)), 1e-12)
    vals: List[float] = [
        float(np.log(float(mkw.get("k", 1.0)) + 1e-30) / denom),
        float(np.log(float(mkw.get("c", 1.0)) + 1e-30) / denom),
        float(np.log(float(mkw.get("a0", 1.0)) + 1e-30) / denom),
        float(mkw.get("nx", 1)) / 100.0,
        float(mkw.get("ny", 1)) / 100.0,
        float(mkw.get("nz", 1)) / 100.0,
    ]
    vals.extend(float(params.get(k, DEFAULT_SETUP_PARAMS.get(k, 0.0))) for k in SETUP_KEYS)
    return np.asarray(vals, dtype=np.float32)


def _load_branch():
    state_path = Path(_env_str("BANDIT_STATE_PATH", "/tmp/mature40_tune7_bandit_state.pkl"))
    print(json.dumps({"stage": "bandit_state_load_start", "path": str(state_path)}), flush=True)
    with state_path.open("rb") as fh:
        branch = pickle.load(fh)["branch"]
    print(json.dumps({"stage": "bandit_state_load_done", "path": str(state_path)}), flush=True)
    return branch


def _trace_from_branch(branch, *, cases: int, seed: int):
    grid_n = _env_int("GRID_N", 40)
    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=int(cases),
        seed=int(seed),
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=_env_float("C_MIN", 1.0),
        c_max=_env_float("C_MAX", 1000.0),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    trace = []
    for i, (mkw, context) in enumerate(instances, 1):
        params, _info = branch.policy.select(np.asarray(context, dtype=float), parameter_space=branch.parameter_space)
        trace.append((dict(mkw), dict(params)))
        if i % max(1, _env_int("PROGRESS_EVERY", 100)) == 0 or i == int(cases):
            print(json.dumps({"stage": "trace_progress", "done": i, "total": int(cases), "seed": int(seed)}), flush=True)
    return trace


def _option_specs(solve_max_cycles: int):
    raw = _env_str("OPTION_SCHEDULES", "").strip()
    if raw:
        specs = [("no_rl", None)]
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            name, schedule_raw = item.split("=")
            schedule = []
            for phase in schedule_raw.split("|"):
                end, w, sd, su = phase.split(":")
                schedule.append((int(end), float(w), int(sd), int(su)))
            specs.append((name.strip(), schedule))
        return specs
    return [
        ("no_rl", None),
        ("const_w_1.60", [(solve_max_cycles, 1.6, 1, 1)]),
        ("tp_6_1.8_to_1.6", [(6, 1.8, 1, 1), (solve_max_cycles, 1.6, 1, 1)]),
        ("tp_8_1.8_to_1.6", [(8, 1.8, 1, 1), (solve_max_cycles, 1.6, 1, 1)]),
        ("tp_12_1.8_to_1.6", [(12, 1.8, 1, 1), (solve_max_cycles, 1.6, 1, 1)]),
        ("tp_4_1.8_to_1.5", [(4, 1.8, 1, 1), (solve_max_cycles, 1.5, 1, 1)]),
        ("tp_8_1.8_to_1.5", [(8, 1.8, 1, 1), (solve_max_cycles, 1.5, 1, 1)]),
        ("tp_12_1.8_to_1.5", [(12, 1.8, 1, 1), (solve_max_cycles, 1.5, 1, 1)]),
        ("tp_12_1.6_to_1.5", [(12, 1.6, 1, 1), (solve_max_cycles, 1.5, 1, 1)]),
    ]


def _eval_option(mkw: Dict[str, Any], params: Dict[str, Any], schedule, *, solve_tol: float, max_cycles: int):
    if schedule is None:
        return solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=float(solve_tol),
            solver_max_iter=int(max_cycles),
            augment_params=_augment_params,
        )
    return _solve_schedule_case(
        params=dict(params),
        mkw=dict(mkw),
        schedule=list(schedule),
        solve_tol=float(solve_tol),
        solve_max_cycles=int(max_cycles),
    )


def _table_cache_key(kind: str, cases: int, seed: int, option_names: Sequence[str]) -> Path:
    tag = "_".join(name.replace(":", "").replace("|", "_").replace(" ", "") for name in option_names)
    return Path(_env_str("TABLE_CACHE_DIR", "/private/tmp/mature40_saved_bandit_option_tables")) / (
        f"{kind}_grid{_env_int('GRID_N', 40)}_n{int(cases)}_seed{int(seed)}_{tag}.pkl"
    )


def _build_table(trace, *, kind: str, seed: int, solve_tol: float, max_cycles: int):
    specs = _option_specs(max_cycles)
    option_names = [name for name, _sched in specs]
    cache_path = _table_cache_key(kind, len(trace), seed, option_names)
    if _env_flag("USE_TABLE_CACHE", "1") and cache_path.exists():
        with cache_path.open("rb") as fh:
            table = pickle.load(fh)
        if list(table.get("options", [])) == option_names:
            print(json.dumps({"stage": "table_cache_hit", "path": str(cache_path)}), flush=True)
            return table
    features = []
    option_outputs: List[List[Dict[str, Any]]] = []
    runtimes = []
    failed = []
    iterations = []
    rng = np.random.default_rng(seed + 70001)
    for i, (mkw, params) in enumerate(trace, 1):
        features.append(_features(dict(mkw), dict(params)))
        order = [int(j) for j in rng.permutation(len(specs))]
        outs = [None] * len(specs)
        for j in order:
            _name, sched = specs[j]
            outs[j] = _eval_option(dict(mkw), dict(params), sched, solve_tol=solve_tol, max_cycles=max_cycles)
        case_outs = [dict(x) for x in outs]
        option_outputs.append(case_outs)
        runtimes.append([float(x["runtime"]) for x in case_outs])
        failed.append([bool(x.get("failed", False)) for x in case_outs])
        iterations.append([int(x.get("iterations", max_cycles)) for x in case_outs])
        if i % max(1, _env_int("PROGRESS_EVERY", 100)) == 0 or i == len(trace):
            print(json.dumps({"stage": "table_progress", "kind": kind, "done": i, "total": len(trace)}), flush=True)
    table = {
        "features": np.asarray(features, dtype=np.float32),
        "runtimes": np.asarray(runtimes, dtype=np.float32),
        "failed": np.asarray(failed, dtype=bool),
        "iterations": np.asarray(iterations, dtype=np.int32),
        "option_outputs": option_outputs,
        "options": option_names,
    }
    if _env_flag("USE_TABLE_CACHE", "1"):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(table, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps({"stage": "table_cache_write", "path": str(cache_path)}), flush=True)
    return table


class OptionPPOEnv(gym.Env):
    def __init__(self, table: Dict[str, Any], *, reward_scale: float, failure_penalty: float, seed: int):
        super().__init__()
        self.table = table
        self.reward_scale = float(reward_scale)
        self.failure_penalty = float(failure_penalty)
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=table["features"].shape[1:], dtype=np.float32)
        self.action_space = spaces.Discrete(int(table["runtimes"].shape[1]))
        self.idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.idx = int(self.rng.integers(0, int(self.table["features"].shape[0])))
        return self.table["features"][self.idx].astype(np.float32), {}

    def step(self, action):
        a = int(np.clip(int(action), 0, self.action_space.n - 1))
        baseline = float(self.table["runtimes"][self.idx, 0])
        runtime = float(self.table["runtimes"][self.idx, a])
        reward = float((baseline - runtime) * self.reward_scale)
        if bool(self.table["failed"][self.idx, a]):
            reward -= self.failure_penalty
        obs = self.table["features"][self.idx].astype(np.float32)
        return obs, reward, True, False, {"case": int(self.idx), "action": a, "runtime": runtime, "baseline": baseline}


def _oracle_actions(table: Dict[str, Any]) -> np.ndarray:
    actions = []
    for i in range(int(table["runtimes"].shape[0])):
        cand = list(range(int(table["runtimes"].shape[1])))
        safe = [a for a in cand if not bool(table["failed"][i, a])]
        if safe:
            cand = safe
        actions.append(min(cand, key=lambda a: float(table["runtimes"][i, a])))
    return np.asarray(actions, dtype=np.int64)


def _actions_summary(table: Dict[str, Any], actions: Sequence[int]) -> Dict[str, Any]:
    selected = [dict(table["option_outputs"][i][int(a)]) for i, a in enumerate(actions)]
    summary = _summarize(selected)
    base = _summarize([dict(row[0]) for row in table["option_outputs"]])
    summary["vs_bandit_pct"] = float(100.0 * (base["mean_runtime"] - summary["mean_runtime"]) / base["mean_runtime"])
    summary["action_hist"] = {str(int(k)): int(v) for k, v in Counter(int(a) for a in actions).items()}
    summary["action_names"] = {str(i): name for i, name in enumerate(table["options"])}
    return summary


def _predict_actions(model: PPO, table: Dict[str, Any]) -> np.ndarray:
    actions = []
    for x in table["features"]:
        action, _ = model.predict(x.astype(np.float32), deterministic=True)
        actions.append(int(action))
    return np.asarray(actions, dtype=np.int64)


def main() -> None:
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    branch = _load_branch()
    train_cases = _env_int("RL_TRAIN_CASES", 500)
    eval_cases = _env_int("EVAL_CASES", 500)
    train_seed = _env_int("RL_TRAIN_SEED", 39396939)
    eval_seed = _env_int("EVAL_SEED", 39394939)
    train_trace = _trace_from_branch(branch, cases=train_cases, seed=train_seed)
    eval_trace = _trace_from_branch(branch, cases=eval_cases, seed=eval_seed)
    train_table = _build_table(train_trace, kind="train", seed=train_seed, solve_tol=solve_tol, max_cycles=max_cycles)
    eval_table = _build_table(eval_trace, kind="eval", seed=eval_seed, solve_tol=solve_tol, max_cycles=max_cycles)

    env = VecMonitor(
        DummyVecEnv(
            [
                lambda: OptionPPOEnv(
                    train_table,
                    reward_scale=_env_float("REWARD_SCALE", 100.0),
                    failure_penalty=_env_float("FAILURE_PENALTY", 10.0),
                    seed=_env_int("RL_SEED", train_seed + 77),
                )
            ]
        )
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=_env_float("LEARNING_RATE", 3e-4),
        n_steps=_env_int("N_STEPS", 1024),
        batch_size=_env_int("BATCH_SIZE", 256),
        n_epochs=_env_int("N_EPOCHS", 10),
        ent_coef=_env_float("ENT_COEF", 0.01),
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        verbose=_env_int("RL_VERBOSE", 0),
        seed=_env_int("RL_SEED", train_seed + 77),
        device="cpu",
    )
    print(
        json.dumps(
            {
                "stage": "ppo_train_start",
                "train_cases": train_cases,
                "eval_cases": eval_cases,
                "options": train_table["options"],
                "timesteps": _env_int("TOTAL_TIMESTEPS", 50000),
            }
        ),
        flush=True,
    )
    model.learn(total_timesteps=_env_int("TOTAL_TIMESTEPS", 50000))
    model_base = Path(_env_str("MODEL_BASENAME", "/private/tmp/saved_mature_option_ppo"))
    model.save(str(model_base))
    print(json.dumps({"stage": "ppo_train_done", "model": str(model_base.with_suffix(".zip"))}), flush=True)

    train_actions = _predict_actions(model, train_table)
    eval_actions = _predict_actions(model, eval_table)
    oracle_train = _oracle_actions(train_table)
    oracle_eval = _oracle_actions(eval_table)
    fixed_idx = int(train_table["options"].index("const_w_1.60")) if "const_w_1.60" in train_table["options"] else -1

    result = {
        "stage": "final",
        "protocol": {
            "grid": [_env_int("GRID_N", 40)] * 3,
            "bandit_state_path": _env_str("BANDIT_STATE_PATH", "/tmp/mature40_tune7_bandit_state.pkl"),
            "bandit_frozen": True,
            "train_cases": train_cases,
            "train_seed": train_seed,
            "eval_cases": eval_cases,
            "eval_seed": eval_seed,
        },
        "options": list(eval_table["options"]),
        "methods": {
            "mature_bandit_default_solve": _summarize([dict(row[0]) for row in eval_table["option_outputs"]]),
            "mature_bandit_fixed_w_1.60": (
                _summarize([dict(row[fixed_idx]) for row in eval_table["option_outputs"]]) if fixed_idx >= 0 else None
            ),
            "mature_bandit_option_ppo": _actions_summary(eval_table, eval_actions),
            "oracle_best_option_per_case": _actions_summary(eval_table, oracle_eval),
        },
        "train_policy": _actions_summary(train_table, train_actions),
        "train_oracle": _actions_summary(train_table, oracle_train),
    }
    result["relative_pct"] = {
        "option_ppo_vs_bandit": result["methods"]["mature_bandit_option_ppo"]["vs_bandit_pct"],
        "oracle_vs_bandit": result["methods"]["oracle_best_option_per_case"]["vs_bandit_pct"],
    }
    if result["methods"]["mature_bandit_fixed_w_1.60"] is not None:
        fixed = result["methods"]["mature_bandit_fixed_w_1.60"]
        ppo = result["methods"]["mature_bandit_option_ppo"]
        result["relative_pct"]["option_ppo_vs_fixed_w_1.60"] = float(
            100.0 * (fixed["mean_runtime"] - ppo["mean_runtime"]) / fixed["mean_runtime"]
        )

    result_path = _env_str("RESULT_PATH", "").strip()
    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "result_write", "path": result_path}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
