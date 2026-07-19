from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from torch import nn

from explore_step_rl_dynamic_control import _fixed_trace, _solve_schedule_case
from explore_bandit_solve_control import _augment_params
from setup_aware_compare_common import solve_no_rl_case


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


SETUP_KEYS = (
    "P_max_elmts", "agg_Pmx", "agg_interp_type", "agg_num_levels", "agg_tr",
    "coarsen_type", "interp_type", "max_row_sum", "strong_threshold", "trunc_factor",
)


def _features(mkw: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    vals = [
        np.log(float(mkw.get("c", 1.0)) + 1e-30) / np.log(1000.0),
        float(mkw.get("nx", 1)) / 100.0,
        float(mkw.get("ny", 1)) / 100.0,
        float(mkw.get("nz", 1)) / 100.0,
    ]
    vals.extend(float(params.get(k, 0.0)) for k in SETUP_KEYS)
    return np.asarray(vals, dtype=np.float32)


def _option_specs():
    mode = _env_str("OPTION_MODE", "base")
    if mode == "base":
        return [
            ("no_rl", None),
            ("const_w_1.7", [(50, 1.7, 1, 1)]),
            ("twophase_8_1.7_to_1.5", [(8, 1.7, 1, 1), (50, 1.5, 1, 1)]),
            ("twophase_4_1.9_to_1.5", [(4, 1.9, 1, 1), (50, 1.5, 1, 1)]),
        ]
    if mode == "expanded":
        return [
            ("no_rl", None),
            ("const_w_1.5", [(50, 1.5, 1, 1)]),
            ("const_w_1.6", [(50, 1.6, 1, 1)]),
            ("const_w_1.7", [(50, 1.7, 1, 1)]),
            ("const_w_1.8", [(50, 1.8, 1, 1)]),
            ("twophase_4_1.8_to_1.5", [(4, 1.8, 1, 1), (50, 1.5, 1, 1)]),
            ("twophase_4_1.9_to_1.5", [(4, 1.9, 1, 1), (50, 1.5, 1, 1)]),
            ("twophase_8_1.7_to_1.5", [(8, 1.7, 1, 1), (50, 1.5, 1, 1)]),
            ("twophase_8_1.7_to_1.6", [(8, 1.7, 1, 1), (50, 1.6, 1, 1)]),
            ("twophase_8_1.8_to_1.5", [(8, 1.8, 1, 1), (50, 1.5, 1, 1)]),
            ("twophase_12_1.7_to_1.5", [(12, 1.7, 1, 1), (50, 1.5, 1, 1)]),
        ]
    raise ValueError(f"unknown OPTION_MODE={mode!r}")


def _trace_cache_path(
    cache_dir: Path,
    *,
    T: int,
    seed: int,
    grid: tuple[int, int, int],
    tune_dim: int,
    bandit_method: str,
    tune7_variant: str,
) -> Path:
    return cache_dir / (
        f"option_trace_T{T}_seed{seed}_grid{grid[0]}"
        f"_tune{int(tune_dim)}_{str(bandit_method)}_{str(tune7_variant)}.pkl"
    )


def _load_or_build_trace(
    *,
    cache_dir: Path,
    T: int,
    seed: int,
    grid: tuple[int, int, int],
    skip: int,
    cases: int,
    tune_dim: int,
    bandit_method: str,
    tune7_variant: str,
):
    path = _trace_cache_path(
        cache_dir,
        T=T,
        seed=seed,
        grid=grid,
        tune_dim=tune_dim,
        bandit_method=bandit_method,
        tune7_variant=tune7_variant,
    )
    if path.exists():
        trace = pickle.load(path.open("rb"))
        print(json.dumps({"stage": "trace_cache_hit", "path": str(path)}), flush=True)
    else:
        trace = _fixed_trace(
            T=T,
            grid=grid,
            seed=seed,
            tune_dim=int(tune_dim),
            bandit_method=str(bandit_method),
            solve_mode="no_rl",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(trace, path.open("wb"))
        print(
            json.dumps(
                {
                    "stage": "trace_cache_write",
                    "path": str(path),
                    "T": T,
                    "seed": seed,
                    "tune_dim": int(tune_dim),
                    "bandit_method": str(bandit_method),
                    "tune7_variant": str(tune7_variant),
                }
            ),
            flush=True,
        )
    out = list(trace[int(skip): int(skip) + int(cases)])
    if len(out) != int(cases):
        raise RuntimeError(f"expected {cases} after skip={skip}, got {len(out)}")
    return out


def _eval_option(mkw, params, schedule, *, solve_tol, max_cycles):
    if schedule is None:
        return solve_no_rl_case(params=dict(params), mkw=dict(mkw), solver_tol=solve_tol, solver_max_iter=max_cycles, augment_params=_augment_params)
    return _solve_schedule_case(params=dict(params), mkw=dict(mkw), schedule=list(schedule), solve_tol=solve_tol, solve_max_cycles=max_cycles)


def _build_table(trace, *, solve_tol, max_cycles, cache_path: Path | None = None):
    opts = _option_specs()
    opt_names = [name for name, _ in opts]
    if cache_path is not None and cache_path.exists():
        table = pickle.load(cache_path.open("rb"))
        if list(table.get("options", [])) == opt_names:
            print(json.dumps({"stage": "table_cache_hit", "path": str(cache_path)}), flush=True)
            return table
        print(json.dumps({"stage": "table_cache_mismatch", "path": str(cache_path), "cached_options": table.get("options", []), "requested_options": opt_names}), flush=True)
    feats, runtimes, failed, iterations = [], [], [], []
    for i, (mkw, params) in enumerate(trace):
        feats.append(_features(dict(mkw), dict(params)))
        rr, ff, ii = [], [], []
        for _name, sched in opts:
            out = _eval_option(dict(mkw), dict(params), sched, solve_tol=solve_tol, max_cycles=max_cycles)
            rr.append(float(out["runtime"])); ff.append(bool(out.get("failed", False))); ii.append(int(out["iterations"]))
        runtimes.append(rr); failed.append(ff); iterations.append(ii)
        if (i + 1) % 100 == 0:
            print(json.dumps({"stage": "table_progress", "done": i + 1, "total": len(trace)}), flush=True)
    table = {
        "features": np.asarray(feats, dtype=np.float32),
        "runtimes": np.asarray(runtimes, dtype=np.float32),
        "failed": np.asarray(failed, dtype=bool),
        "iterations": np.asarray(iterations, dtype=np.int32),
        "options": [name for name, _ in opts],
    }
    if cache_path is not None:
        pickle.dump(table, cache_path.open("wb"))
        print(json.dumps({"stage": "table_cache_write", "path": str(cache_path)}), flush=True)
    return table


def _oracle_labels(table, *, require_no_fail=True):
    ys = []
    for i in range(table["runtimes"].shape[0]):
        cand = list(range(table["runtimes"].shape[1]))
        if require_no_fail:
            safe = [a for a in cand if not bool(table["failed"][i, a])]
            if safe:
                cand = safe
        ys.append(min(cand, key=lambda j: float(table["runtimes"][i, j])))
    return np.asarray(ys, dtype=np.int64)


def _oracle_eval(table):
    y = _oracle_labels(table, require_no_fail=True)
    rt = [float(table["runtimes"][i, a]) for i, a in enumerate(y)]
    base = [float(x) for x in table["runtimes"][:, 0]]
    return {"mean_runtime": mean(rt), "mean_baseline": mean(base), "delta": mean(rt)-mean(base), "failed_count": int(sum(bool(table["failed"][i,a]) for i,a in enumerate(y))), "action_hist": {int(k): int(np.sum(y == k)) for k in sorted(set(y.tolist()))}}


def _eval_actions(table, actions):
    rt = [float(table["runtimes"][i, int(a)]) for i, a in enumerate(actions)]
    base = [float(x) for x in table["runtimes"][:, 0]]
    return {"mean_runtime": mean(rt), "mean_baseline": mean(base), "delta": mean(rt)-mean(base), "failed_count": int(sum(bool(table["failed"][i,int(a)]) for i,a in enumerate(actions))), "action_hist": {int(k): int(np.sum(np.asarray(actions) == k)) for k in sorted(set(int(a) for a in actions))}}


def _make_model(input_dim: int, n_actions: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_actions),
    )


def _predict_payload(payload: dict[str, Any], table: dict[str, Any]) -> np.ndarray:
    model = _make_model(int(payload["input_dim"]), int(payload["n_actions"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x = torch.as_tensor(table["features"], dtype=torch.float32)
    mu = payload["mu"]
    sig = payload["sig"]
    with torch.no_grad():
        return model((x - mu) / sig).argmax(1).cpu().numpy()


def _supervised(train, evalt):
    x = torch.as_tensor(train["features"], dtype=torch.float32)
    y = torch.as_tensor(_oracle_labels(train), dtype=torch.long)
    xe = torch.as_tensor(evalt["features"], dtype=torch.float32)
    mu = x.mean(0); sig = x.std(0).clamp(min=1e-6)
    xn = (x - mu) / sig; xen = (xe - mu) / sig
    model = _make_model(x.shape[1], train["runtimes"].shape[1])
    weights = torch.bincount(y, minlength=train["runtimes"].shape[1]).float().clamp(min=1)
    weights = weights.sum() / (len(weights) * weights)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    fail_penalty = _env_float("SL_FAIL_PENALTY", 0.2)
    cost = torch.as_tensor(train["runtimes"], dtype=torch.float32) + fail_penalty * torch.as_tensor(train["failed"], dtype=torch.float32)
    # Per-case centering keeps gradients focused on action differences, not case difficulty.
    cost = cost - cost.min(dim=1, keepdim=True).values
    denom = cost.detach().std().clamp(min=1e-6)
    cost = cost / denom
    loss_mode = _env_str("SL_LOSS_MODE", "ce")
    cost_weight = _env_float("SL_COST_WEIGHT", 1.0)
    tau = _env_float("SL_COST_TAU", 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=_env_float("SL_LR", 5e-4), weight_decay=_env_float("SL_WD", 1e-2))
    best = None
    best_payload = None
    for e in range(1, _env_int("SL_EPOCHS", 1200) + 1):
        perm = torch.randperm(xn.shape[0])
        for st in range(0, xn.shape[0], _env_int("SL_BATCH_SIZE", 128)):
            idx = perm[st:st+_env_int("SL_BATCH_SIZE", 128)]
            logits = model(xn[idx])
            ce_loss = loss_fn(logits, y[idx])
            prob = torch.softmax(logits / tau, dim=1)
            cost_loss = (prob * cost[idx]).sum(dim=1).mean()
            if loss_mode == "ce":
                loss = ce_loss
            elif loss_mode == "cost":
                loss = cost_loss
            elif loss_mode == "mixed":
                loss = ce_loss + cost_weight * cost_loss
            else:
                raise ValueError(f"unknown SL_LOSS_MODE={loss_mode!r}")
            opt.zero_grad(); loss.backward(); opt.step()
        if e % _env_int("SL_EVAL_EVERY", 100) == 0 or e == 1:
            with torch.no_grad():
                a_tr = model(xn).argmax(1).cpu().numpy()
                a_ev = model(xen).argmax(1).cpu().numpy()
            ev = _eval_actions(evalt, a_ev)
            tr = _eval_actions(train, a_tr)
            rec = {"epoch": e, "train": tr, "eval": ev}
            if best is None or ev["delta"] < best["eval"]["delta"]:
                best = rec
                best_payload = {
                    "epoch": e,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "mu": mu.detach().clone(),
                    "sig": sig.detach().clone(),
                    "input_dim": int(x.shape[1]),
                    "n_actions": int(train["runtimes"].shape[1]),
                    "options": list(train["options"]),
                    "setup_keys": list(SETUP_KEYS),
                    "best": best,
                }
                save_path = _env_str("SL_SAVE_PATH", "")
                if save_path:
                    torch.save(best_payload, save_path)
                    print(json.dumps({"stage": "sl_save_best", "path": save_path, "epoch": e, "eval_delta": ev["delta"]}), flush=True)
            print(json.dumps({"stage": "sl_progress", **rec}), flush=True)
    print(json.dumps({"stage": "sl_best", "best": best}, indent=2), flush=True)
    return best, best_payload


def main():
    grid_n = _env_int("GRID_N_FIXED", 40)
    grid = (grid_n, grid_n, grid_n)
    cache_dir = Path(_env_str("TRACE_CACHE_DIR", "/tmp/frozen_bandit_trace_cache_mature40_warmup1000"))
    table_dir = Path(_env_str("TABLE_CACHE_DIR", "/tmp/mature40_option_tables_multi")); table_dir.mkdir(parents=True, exist_ok=True)
    T = _env_int("TRACE_T", 1500); skip = _env_int("TRACE_SKIP_PREFIX", 1000)
    tune_dim = _env_int("SETUP_TUNE_DIM", 5)
    bandit_method = _env_str("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    tune7_variant = _env_str("TUNE7_VARIANT", "categorical").strip().lower()
    train_cases = _env_int("TRAIN_CASES_PER_SEED", 500)
    eval_cases = _env_int("EVAL_CASES", 500)
    train_seeds = [int(x) for x in _env_str("TRAIN_SEEDS", "39393939,39395939,39396939").split(",") if x.strip()]
    eval_seeds_raw = _env_str("EVAL_SEEDS", "")
    eval_seeds = [int(x) for x in eval_seeds_raw.split(",") if x.strip()] if eval_seeds_raw else [_env_int("EVAL_SEED", 39394939)]
    eval_seed = eval_seeds[0]
    option_mode = _env_str("OPTION_MODE", "base")
    bg_tag = f"tune{int(tune_dim)}_{bandit_method}_{tune7_variant}"
    train_tables = []
    for seed in train_seeds:
        trace = _load_or_build_trace(
            cache_dir=cache_dir,
            T=T,
            seed=seed,
            grid=grid,
            skip=skip,
            cases=train_cases,
            tune_dim=tune_dim,
            bandit_method=bandit_method,
            tune7_variant=tune7_variant,
        )
        train_tables.append(
            _build_table(
                trace,
                solve_tol=1e-6,
                max_cycles=50,
                cache_path=table_dir / f"train_seed{seed}_n{train_cases}_{bg_tag}_{option_mode}.pkl",
            )
        )
    eval_trace = _load_or_build_trace(
        cache_dir=cache_dir,
        T=T,
        seed=eval_seed,
        grid=grid,
        skip=skip,
        cases=eval_cases,
        tune_dim=tune_dim,
        bandit_method=bandit_method,
        tune7_variant=tune7_variant,
    )
    eval_table = _build_table(
        eval_trace,
        solve_tol=1e-6,
        max_cycles=50,
        cache_path=table_dir / f"eval_seed{eval_seed}_n{eval_cases}_{bg_tag}_{option_mode}.pkl",
    )
    train_table = {
        "features": np.concatenate([t["features"] for t in train_tables], axis=0),
        "runtimes": np.concatenate([t["runtimes"] for t in train_tables], axis=0),
        "failed": np.concatenate([t["failed"] for t in train_tables], axis=0),
        "iterations": np.concatenate([t["iterations"] for t in train_tables], axis=0),
        "options": train_tables[0]["options"],
    }
    print(
        json.dumps(
            {
                "stage": "ready",
                "background": {
                    "tune_dim": int(tune_dim),
                    "bandit_method": str(bandit_method),
                    "tune7_variant": str(tune7_variant),
                },
                "options": train_table["options"],
                "train_n": len(train_table["features"]),
                "eval_n": len(eval_table["features"]),
                "eval_seed": eval_seed,
                "train_oracle": _oracle_eval(train_table),
                "eval_oracle": _oracle_eval(eval_table),
            },
            indent=2,
        ),
        flush=True,
    )
    load_path = _env_str("SL_LOAD_PATH", "")
    if load_path:
        payload = torch.load(load_path, map_location="cpu")
        best = payload.get("best")
        print(json.dumps({"stage": "sl_load", "path": load_path, "best": best}, indent=2), flush=True)
    else:
        best, payload = _supervised(train_table, eval_table)
    if payload is not None:
        multi = []
        for seed in eval_seeds:
            trace = _load_or_build_trace(
                cache_dir=cache_dir,
                T=T,
                seed=seed,
                grid=grid,
                skip=skip,
                cases=eval_cases,
                tune_dim=tune_dim,
                bandit_method=bandit_method,
                tune7_variant=tune7_variant,
            )
            table = _build_table(
                trace,
                solve_tol=1e-6,
                max_cycles=50,
                cache_path=table_dir / f"eval_seed{seed}_n{eval_cases}_{bg_tag}_{option_mode}.pkl",
            )
            actions = _predict_payload(payload, table)
            multi.append({"seed": seed, "oracle": _oracle_eval(table), "policy": _eval_actions(table, actions)})
        print(json.dumps({"stage": "multi_eval", "best": best, "results": multi}, indent=2), flush=True)


if __name__ == "__main__":
    main()
