from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
_SETUP_ROOT = _REPO_ROOT / "SetupPhase"
_SETUP_SCRIPTS = _SETUP_ROOT / "scripts"
for _path in (str(_SETUP_ROOT), str(_SETUP_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from amg_gym_env import build_policy_obs, decode_policy_action
from amg_setup_gym_env import DEFAULT_SETUP_PARAMS, SetupObsEncoder, build_setup_param_space
from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _best_two_phase_schedule_for_case, _fixed_trace
from setup_aware_compare_common import solve_no_rl_case
from solver import create_env


class OracleStepMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_obs_encoder(tune_dim: int, tune7_variant: str) -> Tuple[SetupObsEncoder, Tuple[str, ...]]:
    setup_param_space = build_setup_param_space(tune_dim=int(tune_dim), tune7_variant=str(tune7_variant))
    setup_obs_keys = tuple(setup_param_space.parameter_spec.parameter_names)
    return (
        SetupObsEncoder(setup_param_space.parameter_spec, dict(DEFAULT_SETUP_PARAMS), setup_obs_keys),
        setup_obs_keys,
    )


def _policy_obs(
    *,
    r: float,
    r_prev: float,
    cycle: int,
    max_cycles: int,
    mkw: Dict[str, Any],
    last_w: float,
    setup_params: Dict[str, Any],
    obs_encoder: SetupObsEncoder,
    c_max: float,
) -> np.ndarray:
    eps = 1e-30
    c_norm_div = max(1.0, float(c_max))
    c_denom = max(float(np.log(float(c_norm_div) + eps)), eps)
    s1 = float(np.log(float(mkw["k"]) + eps) / c_denom)
    s2 = float(np.log(float(mkw["c"]) + eps) / c_denom)
    s3 = float(np.log(float(mkw["a0"]) + eps) / c_denom)
    grid_norm_div = max(float(mkw["nx"]), float(mkw["ny"]), float(mkw["nz"]), 1.0)
    n_denom = max(float(np.log(float(grid_norm_div) + eps)), eps)
    solve_obs = build_policy_obs(
        r=float(r),
        r_prev=float(r_prev),
        cycle=int(cycle),
        max_cycles=int(max_cycles),
        coeff_triplet=(s1, s2, s3),
        grid_triplet=(
            float(np.log(float(mkw["nx"]) + eps) / n_denom),
            float(np.log(float(mkw["ny"]) + eps) / n_denom),
            float(np.log(float(mkw["nz"]) + eps) / n_denom),
        ),
        last_w=float(last_w),
    ).astype(np.float32)
    setup_obs = obs_encoder.encode(dict(setup_params)).astype(np.float32)
    return np.concatenate([solve_obs, setup_obs], axis=0).astype(np.float32)


def _teacher_schedule(max_cycles: int) -> List[Tuple[int, float, int, int]]:
    switch = int(os.environ.get("TEACHER_SWITCH", "3"))
    w1 = float(os.environ.get("TEACHER_W1", "1.8"))
    w2 = float(os.environ.get("TEACHER_W2", "1.6"))
    sd1 = int(os.environ.get("TEACHER_SD1", "1"))
    su1 = int(os.environ.get("TEACHER_SU1", "1"))
    sd2 = int(os.environ.get("TEACHER_SD2", "1"))
    su2 = int(os.environ.get("TEACHER_SU2", "1"))
    return [(switch, w1, sd1, su1), (int(max_cycles), w2, sd2, su2)]


def _teacher_schedule_for_case(
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    max_cycles: int,
    solve_tol: float,
) -> List[Tuple[int, float, int, int]]:
    mode = os.environ.get("TEACHER_MODE", "fixed").strip().lower()
    if mode != "per_case_two_phase":
        return _teacher_schedule(max_cycles)
    w_grid = [float(x) for x in os.environ.get("ORACLE_W_GRID", "1.0,1.2,1.4,1.6,1.8").split(",") if x.strip()]
    sweep_values = [int(x) for x in os.environ.get("ORACLE_SWEEPS", "1,2").split(",") if x.strip()]
    switch_points = [int(x) for x in os.environ.get("ORACLE_SWITCHES", "3,5,8,12,20").split(",") if x.strip()]
    best = _best_two_phase_schedule_for_case(
        params=dict(params),
        mkw=dict(mkw),
        solve_tol=float(solve_tol),
        solve_max_cycles=int(max_cycles),
        w_grid=w_grid,
        sweep_values=sweep_values,
        switch_points=switch_points,
    )
    return list(best["schedule"])


def _teacher_action_for_cycle(schedule: Sequence[Tuple[int, float, int, int]], cycle: int) -> Tuple[float, int, int]:
    chosen = schedule[-1]
    for end_cycle, w, sd, su in schedule:
        if cycle < int(end_cycle):
            chosen = (int(end_cycle), float(w), int(sd), int(su))
            break
    _end, w, sd, su = chosen
    return float(w), int(sd), int(su)


def _raw_action_from_controls(
    *,
    w: float,
    sd: int,
    su: int,
    w_only: bool,
    w_center: float,
    w_scale: float,
    sweeps_min: int,
    sweeps_max: int,
) -> np.ndarray:
    a_w = float(np.clip((float(w) - float(w_center)) / max(float(w_scale), 1e-12), -1.0, 1.0))
    if w_only:
        return np.asarray([a_w], dtype=np.float32)
    sweeps_center = 0.5 * (int(sweeps_min) + int(sweeps_max))
    sweeps_half = max(0.5 * (int(sweeps_max) - int(sweeps_min)), 1e-12)
    a_d = float(np.clip((float(sd) - sweeps_center) / sweeps_half, -1.0, 1.0))
    a_u = float(np.clip((float(su) - sweeps_center) / sweeps_half, -1.0, 1.0))
    return np.asarray([a_w, a_d, a_u], dtype=np.float32)


def _build_dataset(
    *,
    trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    max_cycles: int,
    solve_tol: float,
    obs_encoder: SetupObsEncoder,
    c_max: float,
    w_only: bool,
    w_center: float,
    w_scale: float,
    sweeps_min: int,
    sweeps_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    for mkw, params in trace:
        aug_params = _augment_params(params)
        schedule = _teacher_schedule_for_case(
            mkw=dict(mkw),
            params=aug_params,
            max_cycles=int(max_cycles),
            solve_tol=float(solve_tol),
        )
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=aug_params)
            del prep
            r_prev_obs = float(env.r0)
            r_curr = float(env.r0)
            last_w = float(w_center)
            for cycle in range(int(max_cycles)):
                obs = _policy_obs(
                    r=float(r_curr),
                    r_prev=float(r_prev_obs),
                    cycle=int(cycle),
                    max_cycles=int(max_cycles),
                    mkw=mkw,
                    last_w=float(last_w),
                    setup_params=aug_params,
                    obs_encoder=obs_encoder,
                    c_max=float(c_max),
                )
                w, sd, su = _teacher_action_for_cycle(schedule, cycle)
                raw_action = _raw_action_from_controls(
                    w=float(w),
                    sd=int(sd),
                    su=int(su),
                    w_only=bool(w_only),
                    w_center=float(w_center),
                    w_scale=float(w_scale),
                    sweeps_min=int(sweeps_min),
                    sweeps_max=int(sweeps_max),
                )
                obs_rows.append(obs)
                act_rows.append(raw_action)
                r_new, _dt = env.step_rl(relax_weight=float(w), sweeps_down=int(sd), sweeps_up=int(su))
                last_w = float(w)
                if float(r_new) <= float(solve_tol):
                    break
                r_prev_obs = float(r_curr)
                r_curr = float(r_new)
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.float32)


def _eval_model(
    *,
    model: OracleStepMLP,
    trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    max_cycles: int,
    solve_tol: float,
    obs_encoder: SetupObsEncoder,
    c_max: float,
    w_only: bool,
    w_center: float,
    w_scale: float,
    sweeps_min: int,
    sweeps_max: int,
) -> Dict[str, float]:
    model.eval()
    runtimes = []
    failures = 0
    with torch.no_grad():
        for mkw, params in trace:
            aug_params = _augment_params(params)
            with create_env(**mkw) as env:
                prep = env.prepare_rl(params=aug_params)
                setup_runtime = float(prep.setup_runtime_sec)
                r_prev_obs = float(env.r0)
                r_curr = float(env.r0)
                last_w = float(w_center)
                solve_runtime = 0.0
                failed = True
                for cycle in range(int(max_cycles)):
                    obs = _policy_obs(
                        r=float(r_curr),
                        r_prev=float(r_prev_obs),
                        cycle=int(cycle),
                        max_cycles=int(max_cycles),
                        mkw=mkw,
                        last_w=float(last_w),
                        setup_params=aug_params,
                        obs_encoder=obs_encoder,
                        c_max=float(c_max),
                    )
                    raw_action = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
                    w, sd, su = decode_policy_action(
                        raw_action,
                        w_only=bool(w_only),
                        w_center=float(w_center),
                        w_scale=float(w_scale),
                        sweeps_min=int(sweeps_min),
                        sweeps_max=int(sweeps_max),
                        cycle=int(cycle),
                        last_w=float(last_w),
                        w_init=None,
                        sweeps_init=None,
                        w_smooth_alpha=0.0,
                    )
                    r_new, dt = env.step_rl(relax_weight=float(w), sweeps_down=int(sd), sweeps_up=int(su))
                    solve_runtime += float(dt)
                    last_w = float(w)
                    if float(r_new) <= float(solve_tol):
                        failed = False
                        break
                    r_prev_obs = float(r_curr)
                    r_curr = float(r_new)
                failures += int(failed)
                runtimes.append(float(setup_runtime + solve_runtime))
    return {"mean_runtime": float(np.mean(runtimes)), "failed_count": int(failures)}


def main() -> None:
    seed = int(os.environ.get("SEED", "39393939"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "30,30,30").split(","))
    tune_dim = int(os.environ.get("SETUP_TUNE_DIM", "5"))
    tune7_variant = os.environ.get("TUNE7_VARIANT", "categorical").strip().lower()
    bandit_method = os.environ.get("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    w_only = os.environ.get("W_ONLY", "1").strip().lower() not in {"0", "false", "no"}
    w_center = float(os.environ.get("W_CENTER", "1.25"))
    w_scale = float(os.environ.get("W_SCALE", "0.75"))
    sweeps_min = int(os.environ.get("SWEEPS_MIN", "1"))
    sweeps_max = int(os.environ.get("SWEEPS_MAX", "1"))
    c_max = float(os.environ.get("C_MAX", os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")[-1]))
    train_T = int(os.environ.get("BC_TRAIN_T", "64"))
    eval_T = int(os.environ.get("BC_EVAL_T", "32"))
    train_seed = int(os.environ.get("BC_TRAIN_SEED", str(seed)))
    eval_seed = int(os.environ.get("BC_EVAL_SEED", str(seed + 1000)))
    epochs = int(os.environ.get("BC_EPOCHS", "200"))
    lr = float(os.environ.get("BC_LR", "1e-3"))
    batch_size = int(os.environ.get("BC_BATCH_SIZE", "128"))
    model_path = Path(os.environ.get("BC_MODEL_PATH", "oracle_step_bc.pt"))

    obs_encoder, obs_keys = _make_obs_encoder(tune_dim, tune7_variant)
    del obs_keys

    train_trace = _fixed_trace(T=train_T, grid=grid, seed=train_seed, tune_dim=tune_dim, bandit_method=bandit_method)
    eval_trace = _fixed_trace(T=eval_T, grid=grid, seed=eval_seed, tune_dim=tune_dim, bandit_method=bandit_method)
    x_train, y_train = _build_dataset(
        trace=train_trace,
        max_cycles=max_cycles,
        solve_tol=solve_tol,
        obs_encoder=obs_encoder,
        c_max=c_max,
        w_only=w_only,
        w_center=w_center,
        w_scale=w_scale,
        sweeps_min=sweeps_min,
        sweeps_max=sweeps_max,
    )
    action_dim = int(y_train.shape[1])
    model = OracleStepMLP(obs_dim=int(x_train.shape[1]), action_dim=action_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    n = int(x_tensor.shape[0])
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_tensor[idx])
            loss = loss_fn(pred, y_tensor[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(idx.numel())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch={epoch + 1} mean_loss={total_loss / max(n, 1):.6e}")

    torch.save({"state_dict": model.state_dict(), "obs_dim": int(x_train.shape[1]), "action_dim": action_dim}, model_path)
    print(f"saved_model={model_path}")

    train_eval = _eval_model(
        model=model,
        trace=train_trace,
        max_cycles=max_cycles,
        solve_tol=solve_tol,
        obs_encoder=obs_encoder,
        c_max=c_max,
        w_only=w_only,
        w_center=w_center,
        w_scale=w_scale,
        sweeps_min=sweeps_min,
        sweeps_max=sweeps_max,
    )
    eval_eval = _eval_model(
        model=model,
        trace=eval_trace,
        max_cycles=max_cycles,
        solve_tol=solve_tol,
        obs_encoder=obs_encoder,
        c_max=c_max,
        w_only=w_only,
        w_center=w_center,
        w_scale=w_scale,
        sweeps_min=sweeps_min,
        sweeps_max=sweeps_max,
    )
    train_bandit = float(np.mean([
        solve_no_rl_case(
            params=p,
            mkw=dict(m),
            solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
            solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
            augment_params=_augment_params,
        )["runtime"]
        for m, p in train_trace
    ]))
    eval_bandit = float(np.mean([
        solve_no_rl_case(
            params=p,
            mkw=dict(m),
            solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
            solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
            augment_params=_augment_params,
        )["runtime"]
        for m, p in eval_trace
    ]))

    print(f"train_bandit_only_mean={train_bandit:.6f}")
    print(f"train_bc_policy={train_eval}")
    print(f"train_delta_bc_vs_bandit={train_eval['mean_runtime'] - train_bandit:+.6f}")
    print(f"eval_bandit_only_mean={eval_bandit:.6f}")
    print(f"eval_bc_policy={eval_eval}")
    print(f"eval_delta_bc_vs_bandit={eval_eval['mean_runtime'] - eval_bandit:+.6f}")


if __name__ == "__main__":
    main()
