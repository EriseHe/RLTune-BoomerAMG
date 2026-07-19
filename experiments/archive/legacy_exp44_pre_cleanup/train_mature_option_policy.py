from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3 import DQN

from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _solve_schedule_case
from setup_aware_compare_common import solve_no_rl_case


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _load_trace(path: Path, *, skip: int, cases: int):
    trace = pickle.load(path.open("rb"))
    out = list(trace[int(skip) : int(skip) + int(cases)])
    if len(out) != int(cases):
        raise RuntimeError(f"expected {cases} cases from {path}, got {len(out)}")
    return out


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


def _features(mkw: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    vals = []
    vals.append(np.log(float(mkw.get("c", 1.0)) + 1e-30) / np.log(1000.0))
    vals.append(float(mkw.get("nx", 1)) / 100.0)
    vals.append(float(mkw.get("ny", 1)) / 100.0)
    vals.append(float(mkw.get("nz", 1)) / 100.0)
    for key in SETUP_KEYS:
        vals.append(float(params.get(key, 0.0)))
    return np.asarray(vals, dtype=np.float32)


def _option_specs():
    # Option 0 is exact bandit_only/no-RL fallback. Other options use cycle-based schedules.
    specs = [
        ("no_rl", None),
        ("const_w_1.5", [(50, 1.5, 1, 1)]),
        ("const_w_1.7", [(50, 1.7, 1, 1)]),
        ("twophase_8_1.7_to_1.5", [(8, 1.7, 1, 1), (50, 1.5, 1, 1)]),
        ("twophase_4_1.9_to_1.5", [(4, 1.9, 1, 1), (50, 1.5, 1, 1)]),
    ]
    keep_raw = os.environ.get("OPTION_INDICES", "").strip()
    if keep_raw:
        keep = [int(x.strip()) for x in keep_raw.split(",") if x.strip()]
        specs = [specs[i] for i in keep]
    return specs


def _eval_option(mkw: dict[str, Any], params: dict[str, Any], schedule, *, solve_tol: float, max_cycles: int):
    if schedule is None:
        out = solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=float(solve_tol),
            solver_max_iter=int(max_cycles),
            augment_params=_augment_params,
        )
        return {
            "runtime": float(out["runtime"]),
            "setup_runtime": float(out["setup_runtime"]),
            "solve_runtime": float(out["solve_runtime"]),
            "failed": bool(out.get("failed", False)),
            "iterations": int(out["iterations"]),
        }
    return _solve_schedule_case(
        params=dict(params),
        mkw=dict(mkw),
        schedule=list(schedule),
        solve_tol=float(solve_tol),
        solve_max_cycles=int(max_cycles),
    )


def _build_table(trace, *, solve_tol: float, max_cycles: int):
    options = _option_specs()
    feats = []
    runtimes = []
    failed = []
    iterations = []
    for case_idx, (mkw, params) in enumerate(trace):
        feats.append(_features(dict(mkw), dict(params)))
        case_runtimes = []
        case_failed = []
        case_iters = []
        for _name, schedule in options:
            out = _eval_option(dict(mkw), dict(params), schedule, solve_tol=solve_tol, max_cycles=max_cycles)
            case_runtimes.append(float(out["runtime"]))
            case_failed.append(bool(out.get("failed", False)))
            case_iters.append(int(out.get("iterations", max_cycles)))
        runtimes.append(case_runtimes)
        failed.append(case_failed)
        iterations.append(case_iters)
        if (case_idx + 1) % 50 == 0:
            print(json.dumps({"stage": "table_progress", "done": case_idx + 1, "total": len(trace)}), flush=True)
    return {
        "features": np.asarray(feats, dtype=np.float32),
        "runtimes": np.asarray(runtimes, dtype=np.float32),
        "failed": np.asarray(failed, dtype=bool),
        "iterations": np.asarray(iterations, dtype=np.int32),
        "options": [name for name, _schedule in options],
    }


class OptionTableEnv(gym.Env):
    def __init__(self, table: dict[str, Any], *, reward_scale: float = 100.0, failure_penalty: float = 0.0, seed: int = 0):
        super().__init__()
        self.table = table
        self.reward_scale = float(reward_scale)
        self.failure_penalty = float(failure_penalty)
        self.n = int(table["features"].shape[0])
        self.i = 0
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=table["features"].shape[1:], dtype=np.float32)
        self.action_space = spaces.Discrete(int(table["runtimes"].shape[1]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.i = int(self.rng.integers(0, self.n))
        return self.table["features"][self.i].astype(np.float32), {}

    def step(self, action):
        a = int(np.clip(int(action), 0, self.action_space.n - 1))
        baseline = float(self.table["runtimes"][self.i, 0])
        runtime = float(self.table["runtimes"][self.i, a])
        reward = float((baseline - runtime) * self.reward_scale)
        if bool(self.table["failed"][self.i, a]):
            reward -= float(self.failure_penalty)
        info = {"case": int(self.i), "runtime": runtime, "baseline": baseline, "action": a}
        return self.table["features"][self.i].astype(np.float32), reward, True, False, info


def _policy_eval(model, table: dict[str, Any]):
    actions = []
    rows = []
    for i, obs in enumerate(table["features"]):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        actions.append(a)
        rows.append(
            {
                "runtime": float(table["runtimes"][i, a]),
                "baseline": float(table["runtimes"][i, 0]),
                "failed": bool(table["failed"][i, a]),
                "iterations": int(table["iterations"][i, a]),
            }
        )
    return {
        "mean_runtime": float(mean(r["runtime"] for r in rows)),
        "mean_baseline": float(mean(r["baseline"] for r in rows)),
        "delta": float(mean(r["runtime"] for r in rows) - mean(r["baseline"] for r in rows)),
        "failed_count": int(sum(r["failed"] for r in rows)),
        "mean_iterations": float(mean(r["iterations"] for r in rows)),
        "action_hist": {int(a): int(actions.count(a)) for a in sorted(set(actions))},
    }


def _oracle_eval(table: dict[str, Any], *, require_no_fail: bool):
    actions = []
    runtimes = []
    fails = []
    for i in range(int(table["runtimes"].shape[0])):
        candidates = list(range(int(table["runtimes"].shape[1])))
        if require_no_fail:
            safe = [a for a in candidates if not bool(table["failed"][i, a])]
            if safe:
                candidates = safe
        a = min(candidates, key=lambda j: float(table["runtimes"][i, j]))
        actions.append(a)
        runtimes.append(float(table["runtimes"][i, a]))
        fails.append(bool(table["failed"][i, a]))
    base = [float(x) for x in table["runtimes"][:, 0]]
    return {
        "mean_runtime": float(mean(runtimes)),
        "mean_baseline": float(mean(base)),
        "delta": float(mean(runtimes) - mean(base)),
        "failed_count": int(sum(fails)),
        "action_hist": {int(a): int(actions.count(a)) for a in sorted(set(actions))},
    }


def _oracle_labels(table: dict[str, Any], *, require_no_fail: bool) -> np.ndarray:
    labels = []
    for i in range(int(table["runtimes"].shape[0])):
        candidates = list(range(int(table["runtimes"].shape[1])))
        if require_no_fail:
            safe = [a for a in candidates if not bool(table["failed"][i, a])]
            if safe:
                candidates = safe
        labels.append(int(min(candidates, key=lambda j: float(table["runtimes"][i, j]))))
    return np.asarray(labels, dtype=np.int64)


def _bc_warmstart_qnet(model: DQN, table: dict[str, Any], *, require_no_fail: bool) -> None:
    labels = _oracle_labels(table, require_no_fail=require_no_fail)
    x = torch.as_tensor(table["features"], dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    epochs = _env_int("BC_EPOCHS", 500)
    batch = _env_int("BC_BATCH_SIZE", 128)
    lr = _env_float("BC_LR", 1e-3)
    class_balance = _env_str("BC_CLASS_BALANCE", "1").strip().lower() not in {"0", "false", "no"}
    class_weights = None
    if class_balance:
        bincount = torch.bincount(y, minlength=int(table["runtimes"].shape[1])).float().clamp(min=1.0)
        class_weights = torch.sum(bincount) / (float(len(bincount)) * bincount)
        print(json.dumps({"stage": "bc_class_weights", "weights": [float(v) for v in class_weights]}), flush=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.q_net.parameters(), lr=lr)
    n = int(x.shape[0])
    for epoch in range(int(epochs)):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            logits = model.q_net(x[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            pred = torch.argmax(model.q_net(x), dim=1)
            acc = float((pred == y).float().mean().item())
            print(
                json.dumps(
                    {
                        "stage": "bc_progress",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "loss": float(np.mean(losses)) if losses else float("nan"),
                        "train_acc": acc,
                    }
                ),
                flush=True,
            )
    model.q_net_target.load_state_dict(model.q_net.state_dict())


def main():
    train_trace = _load_trace(Path(_env_str("TRAIN_TRACE_PATH", "/tmp/frozen_bandit_trace_cache_mature40_warmup1000/99366365838415a0f150c5b7c4da77c550da958f898c2f5592125a21805bfaf0.pkl")), skip=_env_int("TRAIN_SKIP", 1000), cases=_env_int("TRAIN_CASES", 300))
    eval_trace = _load_trace(Path(_env_str("EVAL_TRACE_PATH", "/tmp/frozen_bandit_trace_cache_mature40_warmup1000/f4644aa61f8718ed9074de8cbb60778fb64ff38bfa98106d3dcb26f6744b5063.pkl")), skip=_env_int("EVAL_SKIP", 1000), cases=_env_int("EVAL_CASES", 500))
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    table_cache = Path(_env_str("TABLE_CACHE", "/tmp/mature40_option_table.pkl"))
    if table_cache.exists():
        blob = pickle.load(table_cache.open("rb"))
        train_table = blob["train"]
        eval_table = blob["eval"]
    else:
        print(json.dumps({"stage": "build_train_table"}), flush=True)
        train_table = _build_table(train_trace, solve_tol=solve_tol, max_cycles=max_cycles)
        print(json.dumps({"stage": "build_eval_table"}), flush=True)
        eval_table = _build_table(eval_trace, solve_tol=solve_tol, max_cycles=max_cycles)
        pickle.dump({"train": train_table, "eval": eval_table}, table_cache.open("wb"))
    print(json.dumps({"stage": "tables_ready", "options": train_table["options"], "train_n": len(train_table["features"]), "eval_n": len(eval_table["features"])}), flush=True)
    print(json.dumps({"stage": "train_oracle", "oracle": _oracle_eval(train_table, require_no_fail=False), "oracle_no_fail": _oracle_eval(train_table, require_no_fail=True)}, indent=2), flush=True)
    print(json.dumps({"stage": "eval_oracle", "oracle": _oracle_eval(eval_table, require_no_fail=False), "oracle_no_fail": _oracle_eval(eval_table, require_no_fail=True)}, indent=2), flush=True)

    env = OptionTableEnv(train_table, reward_scale=_env_float("REWARD_SCALE", 100.0), failure_penalty=_env_float("FAILURE_PENALTY", 1.0), seed=0)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=_env_float("LEARNING_RATE", 1e-3),
        buffer_size=_env_int("BUFFER_SIZE", 50000),
        learning_starts=_env_int("LEARNING_STARTS", 1000),
        batch_size=_env_int("BATCH_SIZE", 128),
        gamma=0.0,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=_env_float("EXPLORATION_FRACTION", 0.2),
        exploration_final_eps=_env_float("EXPLORATION_FINAL_EPS", 0.02),
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device="cpu",
    )
    if _env_str("BC_WARMSTART", "1").strip().lower() not in {"0", "false", "no"}:
        _bc_warmstart_qnet(
            model,
            train_table,
            require_no_fail=_env_str("BC_REQUIRE_NO_FAIL", "1").strip().lower() not in {"0", "false", "no"},
        )
        print(
            json.dumps(
                {
                    "stage": "post_bc",
                    "train_eval": _policy_eval(model, train_table),
                    "eval": _policy_eval(model, eval_table),
                },
                indent=2,
            ),
            flush=True,
        )
    total_timesteps = _env_int("TOTAL_TIMESTEPS", 30000)
    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps)
    else:
        print(json.dumps({"stage": "skip_rl_finetune"}), flush=True)
    out_path = Path(_env_str("MODEL_PATH", "/tmp/dqn_mature40_option_policy.zip"))
    model.save(str(out_path))
    print(json.dumps({"stage": "final", "model_path": str(out_path), "train_eval": _policy_eval(model, train_table), "eval": _policy_eval(model, eval_table)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
