from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hypre.bindings import SolveStatus
from setup.space import SetupObsEncoder, build_setup_parameter_spec
from solve.core.amg_gym_env import (
    build_policy_obs,
    decode_policy_action,
    decode_policy_action_hierarchical,
    decode_policy_action_residual,
)

from .config import SetupAwareRLConfig


class _SpaceOnlyEnv(gym.Env):
    """Minimal VecNormalize host for legacy normalized PPO checkpoints."""

    metadata = {"render_modes": []}

    def __init__(self, *, observation_space: Any, action_space: Any) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("Only Box observations are supported")
        observation = np.zeros(
            self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )
        return observation, {}

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        del action
        raise RuntimeError(
            "_SpaceOnlyEnv is only used to load VecNormalize state"
        )


class SetupAwareSolvePolicyRunner:
    """Load and execute a frozen setup-aware PPO/DQN solve policy."""

    def __init__(self, cfg: SetupAwareRLConfig) -> None:
        self.cfg = cfg
        self.algo = str(cfg.algo).strip().lower()
        self.model_type = str(cfg.model_type).strip().lower()
        self.use_lstm = self.algo == "ppo" and self.model_type == "lstm"
        if self.algo == "dqn":
            self.model = DQN.load(str(cfg.model_path))
        else:
            self.model = (
                RecurrentPPO.load(str(cfg.model_path))
                if self.use_lstm
                else PPO.load(str(cfg.model_path))
            )
        action_shape = getattr(self.model.policy.action_space, "shape", ())
        self.w_only = bool(
            cfg.w_only or (action_shape and int(action_shape[0]) == 1)
        )
        self.w_init = cfg.w_init
        self.sweeps_init = cfg.sweeps_init
        self.initial_observation_weight = (
            None
            if cfg.initial_observation_weight is None
            else float(cfg.initial_observation_weight)
        )
        self._default_first_action_pending = bool(
            cfg.force_default_first_action
        )
        self.forced_initial_action_count = 0
        self.default_first_weight = (
            self.initial_observation_weight
            if cfg.default_first_weight is None
            else float(cfg.default_first_weight)
        )
        self.action_mode = str(cfg.action_mode).strip().lower()
        self.discrete_w_values = tuple(
            float(value) for value in cfg.discrete_w_values
        )
        self.discrete_joint_actions = tuple(
            (float(weight), int(sweeps_down), int(sweeps_up))
            for weight, sweeps_down, sweeps_up in cfg.discrete_joint_actions
        )
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
            for spec in cfg.discrete_extended_actions
        )
        self.discrete_blend_alphas = tuple(
            float(value) for value in cfg.discrete_blend_alphas
        )
        self.blend_safe_action = (
            float(cfg.blend_safe_action[0]),
            int(cfg.blend_safe_action[1]),
            int(cfg.blend_safe_action[2]),
            int(cfg.blend_safe_action[3]),
            int(cfg.blend_safe_action[4]),
            int(cfg.blend_safe_action[5]),
            float(cfg.blend_safe_action[6]),
            float(cfg.blend_safe_action[7]),
        )
        self.blend_aggr_action = (
            float(cfg.blend_aggr_action[0]),
            int(cfg.blend_aggr_action[1]),
            int(cfg.blend_aggr_action[2]),
            int(cfg.blend_aggr_action[3]),
            int(cfg.blend_aggr_action[4]),
            int(cfg.blend_aggr_action[5]),
            float(cfg.blend_aggr_action[6]),
            float(cfg.blend_aggr_action[7]),
        )
        self.blend_decay_tau = float(cfg.blend_decay_tau)
        self.blend_cutoff_cycles = int(cfg.blend_cutoff_cycles)
        self.sweeps_center = 0.5 * (
            int(cfg.sweeps_min) + int(cfg.sweeps_max)
        )
        self.sweeps_half = 0.5 * (
            int(cfg.sweeps_max) - int(cfg.sweeps_min)
        )
        self.sweeps_default = int(
            np.clip(
                np.round(self.sweeps_center),
                int(cfg.sweeps_min),
                int(cfg.sweeps_max),
            )
        )
        self.obs_mode = str(cfg.obs_mode).strip().lower()
        if (
            self.action_mode == "discrete_switch"
            and len(self.discrete_extended_actions) < 2
        ):
            raise ValueError(
                "discrete_switch requires at least two "
                "discrete_extended_actions"
            )
        if (
            self.action_mode == "discrete_blend"
            and not self.discrete_blend_alphas
        ):
            raise ValueError(
                "discrete_blend requires discrete_blend_alphas"
            )

        self.setup_parameter_spec, _fixed_setup_params = (
            build_setup_parameter_spec(
                tune_dim=int(cfg.tune_dim),
                tune7_variant=str(cfg.tune7_variant),
            )
        )
        self.setup_obs_keys = tuple(
            self.setup_parameter_spec.parameter_names
        )
        self.setup_obs_encoder = SetupObsEncoder(
            self.setup_parameter_spec,
            dict(cfg.default_setup_params),
            self.setup_obs_keys,
        )
        self.vec_norm = None
        if cfg.vec_path.exists():
            raw_env = DummyVecEnv(
                [
                    lambda: _SpaceOnlyEnv(
                        observation_space=self.model.observation_space,
                        action_space=self.model.action_space,
                    )
                ]
            )
            self.vec_norm = VecNormalize.load(str(cfg.vec_path), raw_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False

    def _make_obs(
        self,
        *,
        r: float,
        r_prev: float,
        cycle: int,
        case_progress: float,
        last_w: float,
        last_coarse_sweeps: int,
        last_cycle_type: int,
        last_relax_type: int,
        last_outer_weight: float,
        last_add_relax_weight: float,
        switched_to_safe: bool,
        mkw: Dict[str, Any],
        setup_params: Dict[str, Any],
    ) -> np.ndarray:
        eps = 1e-30
        c_norm_div = max(1.0, float(self.cfg.difconv_c_range[1]))
        c_denom = max(float(np.log(float(c_norm_div) + eps)), eps)
        s1 = float(np.log(float(mkw["k"]) + eps) / c_denom)
        s2 = float(np.log(float(mkw["c"]) + eps) / c_denom)
        s3 = float(np.log(float(mkw["a0"]) + eps) / c_denom)

        grid_norm_div = max(
            float(mkw["nx"]),
            float(mkw["ny"]),
            float(mkw["nz"]),
            1.0,
        )
        n_denom = max(
            float(np.log(float(grid_norm_div) + eps)),
            eps,
        )
        nx_norm = float(np.log(float(mkw["nx"]) + eps) / n_denom)
        ny_norm = float(np.log(float(mkw["ny"]) + eps) / n_denom)
        nz_norm = float(np.log(float(mkw["nz"]) + eps) / n_denom)

        solve_obs = build_policy_obs(
            r=float(r),
            r_prev=float(r_prev),
            cycle=int(cycle),
            max_cycles=int(self.cfg.solve_max_cycles),
            coeff_triplet=(s1, s2, s3),
            grid_triplet=(nx_norm, ny_norm, nz_norm),
            last_w=float(last_w),
        ).astype(np.float32)
        setup_obs = self.setup_obs_encoder.encode(
            dict(setup_params)
        ).astype(np.float32)
        action_state = np.array(
            [
                float(last_coarse_sweeps),
                float(last_cycle_type),
                float(last_relax_type),
                float(last_outer_weight),
                float(last_add_relax_weight),
                1.0 if switched_to_safe else 0.0,
            ],
            dtype=np.float32,
        )
        compact_solve_obs = solve_obs[[0, 1, 2, 9]]
        if self.obs_mode == "full":
            obs = np.concatenate([solve_obs, setup_obs], axis=0).astype(
                np.float32
            )
        elif self.obs_mode == "solve_only":
            obs = solve_obs.astype(np.float32)
        elif self.obs_mode == "full_action":
            obs = np.concatenate(
                [solve_obs, action_state, setup_obs],
                axis=0,
            ).astype(np.float32)
        elif self.obs_mode == "cycle_setup":
            obs = np.concatenate(
                [compact_solve_obs, setup_obs],
                axis=0,
            ).astype(np.float32)
        elif self.obs_mode == "cycle_action_setup":
            obs = np.concatenate(
                [compact_solve_obs, action_state, setup_obs],
                axis=0,
            ).astype(np.float32)
        elif self.obs_mode == "cycle_only":
            obs = compact_solve_obs.astype(np.float32)
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
        if bool(self.cfg.obs_include_trace_progress):
            obs = np.concatenate(
                [
                    obs,
                    np.asarray(
                        [float(case_progress)],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            ).astype(np.float32)
        if self.vec_norm is not None:
            obs = self.vec_norm.normalize_obs(obs)
        return obs

    def _map_action(
        self,
        action: np.ndarray,
        cycle: int,
        last_w: float,
        switched_to_safe: bool,
        case_progress: float = 0.0,
    ) -> Tuple[float, int, int, int, int, int, float, float]:
        if self.action_mode == "discrete_switch":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, 1))
            if idx == 1:
                switched_to_safe = True
            chosen_idx = 1 if switched_to_safe else 0
            return self.discrete_extended_actions[chosen_idx]
        if self.action_mode == "discrete_extended":
            if not self.discrete_extended_actions:
                raise ValueError(
                    "discrete_extended_actions must be non-empty when "
                    "action_mode='discrete_extended'"
                )
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(
                np.clip(idx, 0, len(self.discrete_extended_actions) - 1)
            )
            return self.discrete_extended_actions[idx]
        if self.action_mode == "discrete_joint":
            if not self.discrete_joint_actions:
                raise ValueError(
                    "discrete_joint_actions must be non-empty when "
                    "action_mode='discrete_joint'"
                )
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(
                np.clip(idx, 0, len(self.discrete_joint_actions) - 1)
            )
            weight, sweeps_down, sweeps_up = (
                self.discrete_joint_actions[idx]
            )
            return (
                float(weight),
                int(sweeps_down),
                int(sweeps_up),
                1,
                -1,
                -1,
                -1.0,
                -1.0,
            )
        if self.action_mode == "discrete_w":
            if not self.discrete_w_values:
                raise ValueError(
                    "discrete_w_values must be non-empty when "
                    "action_mode='discrete_w'"
                )
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_w_values) - 1))
            weight = float(self.discrete_w_values[idx])
            return (
                weight,
                self.sweeps_default,
                self.sweeps_default,
                1,
                -1,
                -1,
                -1.0,
                -1.0,
            )
        if self.action_mode == "discrete_blend":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(
                np.clip(idx, 0, len(self.discrete_blend_alphas) - 1)
            )
            alpha = float(self.discrete_blend_alphas[idx])
            if (
                self.blend_cutoff_cycles >= 0
                and int(cycle) >= int(self.blend_cutoff_cycles)
            ):
                alpha = 0.0
            if self.blend_decay_tau > 0.0:
                alpha *= float(
                    np.exp(-float(cycle) / float(self.blend_decay_tau))
                )
            if (
                self.cfg.blend_progress_end
                > self.cfg.blend_progress_start
            ):
                if (
                    float(case_progress)
                    <= float(self.cfg.blend_progress_start)
                ):
                    alpha = 0.0
                else:
                    ramp = (
                        float(case_progress)
                        - float(self.cfg.blend_progress_start)
                    ) / (
                        float(self.cfg.blend_progress_end)
                        - float(self.cfg.blend_progress_start)
                    )
                    alpha *= float(np.clip(ramp, 0.0, 1.0))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            (
                safe_w,
                safe_down,
                safe_up,
                safe_coarse,
                safe_cycle,
                safe_relax,
                safe_outer,
                safe_add_relax,
            ) = self.blend_safe_action
            (
                aggressive_w,
                aggressive_down,
                aggressive_up,
                aggressive_coarse,
                aggressive_cycle,
                aggressive_relax,
                aggressive_outer,
                aggressive_add_relax,
            ) = self.blend_aggr_action
            weight = float(
                safe_w + alpha * (aggressive_w - safe_w)
            )
            sweeps_down = int(
                np.clip(
                    np.rint(
                        safe_down
                        + alpha * (aggressive_down - safe_down)
                    ),
                    self.cfg.sweeps_min,
                    self.cfg.sweeps_max,
                )
            )
            sweeps_up = int(
                np.clip(
                    np.rint(
                        safe_up + alpha * (aggressive_up - safe_up)
                    ),
                    self.cfg.sweeps_min,
                    self.cfg.sweeps_max,
                )
            )
            coarse_sweeps = max(
                1,
                int(
                    np.rint(
                        safe_coarse
                        + alpha * (aggressive_coarse - safe_coarse)
                    )
                ),
            )
            cycle_type = int(
                safe_cycle
                if alpha < 0.5 or int(aggressive_cycle) < 0
                else aggressive_cycle
            )
            relax_type = int(
                safe_relax
                if alpha < 0.5 or int(aggressive_relax) < 0
                else aggressive_relax
            )
            if float(safe_outer) < 0.0 and float(aggressive_outer) < 0.0:
                outer_weight = -1.0
            elif (
                float(safe_outer) < 0.0
                or float(aggressive_outer) < 0.0
            ):
                outer_weight = float(
                    safe_outer if alpha < 0.5 else aggressive_outer
                )
            else:
                outer_weight = float(
                    safe_outer
                    + alpha * (aggressive_outer - safe_outer)
                )
            if (
                float(safe_add_relax) < 0.0
                and float(aggressive_add_relax) < 0.0
            ):
                add_relax_weight = -1.0
            elif (
                float(safe_add_relax) < 0.0
                or float(aggressive_add_relax) < 0.0
            ):
                add_relax_weight = float(
                    safe_add_relax
                    if alpha < 0.5
                    else aggressive_add_relax
                )
            else:
                add_relax_weight = float(
                    safe_add_relax
                    + alpha
                    * (aggressive_add_relax - safe_add_relax)
                )
            return (
                weight,
                sweeps_down,
                sweeps_up,
                coarse_sweeps,
                cycle_type,
                relax_type,
                outer_weight,
                add_relax_weight,
            )
        if self.action_mode == "continuous_residual":
            weight, sweeps_down, sweeps_up = (
                decode_policy_action_residual(
                    action,
                    w_only=self.w_only,
                    w_center=float(self.cfg.w_center),
                    w_scale=float(self.cfg.w_scale),
                    w_min=float(self.cfg.w_global_min),
                    w_max=float(self.cfg.w_global_max),
                    sweeps_min=int(self.cfg.sweeps_min),
                    sweeps_max=int(self.cfg.sweeps_max),
                    cycle=int(cycle),
                    last_w=float(last_w),
                    w_init=self.w_init,
                    sweeps_init=self.sweeps_init,
                )
            )
            return (
                float(weight),
                int(sweeps_down),
                int(sweeps_up),
                1,
                -1,
                -1,
                -1.0,
                -1.0,
            )
        if self.action_mode == "continuous_hierarchical":
            weight, sweeps_down, sweeps_up = (
                decode_policy_action_hierarchical(
                    action,
                    w_only=self.w_only,
                    sweeps_min=int(self.cfg.sweeps_min),
                    sweeps_max=int(self.cfg.sweeps_max),
                )
            )
            return (
                float(weight),
                int(sweeps_down),
                int(sweeps_up),
                1,
                -1,
                -1,
                -1.0,
                -1.0,
            )
        weight, sweeps_down, sweeps_up = decode_policy_action(
            action,
            w_only=self.w_only,
            w_center=float(self.cfg.w_center),
            w_scale=float(self.cfg.w_scale),
            sweeps_min=int(self.cfg.sweeps_min),
            sweeps_max=int(self.cfg.sweeps_max),
            cycle=int(cycle),
            last_w=float(last_w),
            w_init=self.w_init,
            sweeps_init=self.sweeps_init,
            w_smooth_alpha=0.0,
        )
        return (
            float(weight),
            int(sweeps_down),
            int(sweeps_up),
            1,
            -1,
            -1,
            -1.0,
            -1.0,
        )

    def run(
        self,
        env: Any,
        *,
        mkw: Dict[str, Any],
        setup_params: Dict[str, Any],
        case_progress: float = 0.0,
    ) -> Dict[str, Any]:
        r_prev_obs = float(env.r0)
        r_curr = float(env.r0)
        if self.w_init is not None:
            last_w = float(self.w_init)
        elif self.initial_observation_weight is not None:
            last_w = float(self.initial_observation_weight)
        else:
            last_w = float(self.cfg.w_center)
        solve_runtime = 0.0
        infer_runtime = 0.0
        cycles = 0
        action_counts: Dict[int, int] = {}
        cycle_actions: List[float] = []
        cycle_forced_default_actions: List[bool] = []
        cycle_residuals: List[float] = []
        cycle_times: List[float] = []
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        last_sd = self.sweeps_default
        last_su = self.sweeps_default
        last_sc = 1
        last_ct = -1
        last_rt = -1
        last_ow = -1.0
        last_arw = -1.0
        switched_to_safe = False
        r_new = float(r_curr)
        native_status = SolveStatus.CONTINUE

        for cycle in range(int(self.cfg.solve_max_cycles)):
            obs = self._make_obs(
                r=r_curr,
                r_prev=r_prev_obs,
                cycle=cycle,
                case_progress=float(case_progress),
                last_w=last_w,
                last_coarse_sweeps=last_sc,
                last_cycle_type=last_ct,
                last_relax_type=last_rt,
                last_outer_weight=last_ow,
                last_add_relax_weight=last_arw,
                switched_to_safe=switched_to_safe,
                mkw=mkw,
                setup_params=setup_params,
            )
            infer_start = time.perf_counter()
            if self.use_lstm:
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )
            infer_runtime += float(time.perf_counter() - infer_start)

            action_idx: Optional[int] = None
            if self.action_mode == "discrete_switch":
                chosen = int(np.asarray(action).reshape(-1)[0])
                chosen = int(np.clip(chosen, 0, 1))
                if chosen == 1:
                    switched_to_safe = True
                action_idx = 1 if switched_to_safe else 0
                (
                    weight,
                    sweeps_down,
                    sweeps_up,
                    coarse_sweeps,
                    cycle_type,
                    relax_type,
                    outer_weight,
                    add_relax_weight,
                ) = self.discrete_extended_actions[action_idx]
            else:
                (
                    weight,
                    sweeps_down,
                    sweeps_up,
                    coarse_sweeps,
                    cycle_type,
                    relax_type,
                    outer_weight,
                    add_relax_weight,
                ) = self._map_action(
                    action,
                    cycle,
                    last_w,
                    switched_to_safe,
                    case_progress=float(case_progress),
                )
                if self.action_mode in {
                    "discrete_extended",
                    "discrete_joint",
                    "discrete_w",
                    "discrete_blend",
                }:
                    action_idx = int(np.asarray(action).reshape(-1)[0])
            forced_default = bool(self._default_first_action_pending)
            if forced_default:
                if self.default_first_weight is None:
                    raise ValueError(
                        "force_default_first_action requires a default weight"
                    )
                weight = float(self.default_first_weight)
                self._default_first_action_pending = False
                self.forced_initial_action_count += 1
            if action_idx is not None:
                action_idx = int(action_idx)
                action_counts[action_idx] = (
                    int(action_counts.get(action_idx, 0)) + 1
                )
            r_new, duration = env.step_rl(
                relax_weight=weight,
                sweeps_down=sweeps_down,
                sweeps_up=sweeps_up,
                coarse_sweeps=coarse_sweeps,
                cycle_type=(
                    None if int(cycle_type) < 0 else int(cycle_type)
                ),
                relax_type=(
                    None if int(relax_type) < 0 else int(relax_type)
                ),
                outer_weight=(
                    None
                    if float(outer_weight) < 0.0
                    else float(outer_weight)
                ),
                add_relax_weight=(
                    None
                    if float(add_relax_weight) < 0.0
                    else float(add_relax_weight)
                ),
                tol=float(self.cfg.solve_tol),
                max_cycles=int(self.cfg.solve_max_cycles),
            )
            solve_runtime += float(duration)
            cycles = cycle + 1
            cycle_actions.append(float(weight))
            cycle_forced_default_actions.append(bool(forced_default))
            cycle_residuals.append(float(r_new))
            cycle_times.append(float(duration))
            native_status = env.last_step.status
            last_w = float(weight)
            last_sd = int(sweeps_down)
            last_su = int(sweeps_up)
            last_sc = int(coarse_sweeps)
            last_ct = int(cycle_type)
            last_rt = int(relax_type)
            last_ow = float(outer_weight)
            last_arw = float(add_relax_weight)
            if native_status is not SolveStatus.CONTINUE:
                r_curr = float(r_new)
                break
            r_prev_obs = float(r_curr)
            r_curr = float(r_new)
            episode_start[...] = False
        else:
            r_curr = float(r_new)

        return {
            "solve_runtime": float(solve_runtime),
            "infer_runtime": float(infer_runtime),
            "residual_norm": float(r_curr),
            "iterations": int(cycles),
            "failed": native_status is not SolveStatus.CONVERGED,
            "attempt_status": (
                "success"
                if native_status is SolveStatus.CONVERGED
                else "nonconvergence"
            ),
            "native_status": native_status.name.lower(),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
            "final_coarse_sweeps": int(last_sc),
            "final_cycle_type": int(last_ct),
            "final_relax_type": int(last_rt),
            "final_outer_weight": float(last_ow),
            "final_add_relax_weight": float(last_arw),
            "action_counts": dict(action_counts),
            "cycle_actions": cycle_actions,
            "cycle_forced_default_actions": (
                cycle_forced_default_actions
            ),
            "cycle_residuals": cycle_residuals,
            "cycle_times": cycle_times,
        }


__all__ = ["SetupAwareSolvePolicyRunner", "_SpaceOnlyEnv"]
