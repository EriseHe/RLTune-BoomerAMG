from __future__ import annotations

import _project_paths  # noqa: F401

import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]

from amg_gym_env import (
    build_policy_obs,
    decode_policy_action,
    decode_policy_action_hierarchical,
    decode_policy_action_residual,
)
from amg_setup_gym_env import DEFAULT_SETUP_PARAMS, SetupObsEncoder, build_setup_parameter_spec
from hypre.bindings import create_env
from setup_aware_compare_common import augment_setup_params, solve_no_rl_case


class FrozenBanditStepEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        tune_dim: int,
        tune7_variant: str,
        c_max: float,
        tol: float,
        max_cycles: int,
        w_only: bool,
        action_mode: str = "continuous",
        discrete_w_values: Sequence[float] = (),
        discrete_joint_actions: Sequence[Tuple[float, int, int]] = (),
        discrete_extended_actions: Sequence[Tuple[float, int, int, int, int, int, float, float]] = (),
        discrete_blend_alphas: Sequence[float] = (),
        blend_safe_action: Tuple[float, int, int, int, int, int, float, float] = (1.6, 1, 1, 1, 1, 18, -1.0, -1.0),
        blend_aggr_action: Tuple[float, int, int, int, int, int, float, float] = (1.6, 1, 1, 2, 1, 18, -1.0, 0.1),
        blend_decay_tau: float = 0.0,
        blend_cutoff_cycles: int = -1,
        blend_progress_start: float = 0.0,
        blend_progress_end: float = 0.0,
        w_center: float,
        w_scale: float,
        w_global_min: float = 1.0,
        w_global_max: float = 2.0,
        initial_observation_weight: float | None = None,
        w_init_mode: str = "fixed",
        w_init_min: float | None = None,
        w_init_max: float | None = None,
        sweeps_min: int,
        sweeps_max: int,
        reward_mode: int = 3,
        term_bonus: float = 0.0,
        trunc_penalty: float = 0.0,
        policy_step_cost_sec: float = 0.0,
        potential_progress_scale: float = 0.0,
        episode_stride: int = 1,
        baseline_runtimes: Sequence[float] | None = None,
        reference_runtimes: Sequence[float] | None = None,
        relative_bonus_scale: float = 1.0,
        reference_margin_sec: float = 0.0,
        anchor_w: float = float("nan"),
        anchor_deviation_penalty: float = 0.0,
        obs_mode: str = "full",
        obs_include_trace_progress: bool = False,
        teacher_action_schedule: Sequence[Tuple[int, int]] = (),
        teacher_match_bonus: float = 0.0,
        teacher_mismatch_penalty: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.trace = [(dict(mkw), dict(params)) for mkw, params in trace]
        if not self.trace:
            raise ValueError("trace must be non-empty")
        self.rng = np.random.default_rng(seed)
        self.tol = float(tol)
        self.max_cycles = int(max_cycles)
        self.w_only = bool(w_only)
        self.action_mode = str(action_mode).strip().lower()
        self.discrete_w_values = tuple(float(x) for x in discrete_w_values)
        self.discrete_joint_actions = tuple((float(w), int(sd), int(su)) for (w, sd, su) in discrete_joint_actions)
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
        self.discrete_blend_alphas = tuple(float(x) for x in discrete_blend_alphas)
        self.blend_safe_action = (
            float(blend_safe_action[0]),
            int(blend_safe_action[1]),
            int(blend_safe_action[2]),
            int(blend_safe_action[3]),
            int(blend_safe_action[4]),
            int(blend_safe_action[5]),
            float(blend_safe_action[6]),
            float(blend_safe_action[7]),
        )
        self.blend_aggr_action = (
            float(blend_aggr_action[0]),
            int(blend_aggr_action[1]),
            int(blend_aggr_action[2]),
            int(blend_aggr_action[3]),
            int(blend_aggr_action[4]),
            int(blend_aggr_action[5]),
            float(blend_aggr_action[6]),
            float(blend_aggr_action[7]),
        )
        self.blend_decay_tau = float(blend_decay_tau)
        self.blend_cutoff_cycles = int(blend_cutoff_cycles)
        self.blend_progress_start = float(blend_progress_start)
        self.blend_progress_end = float(blend_progress_end)
        self.w_center = float(w_center)
        self.w_scale = float(w_scale)
        self.w_global_min = float(w_global_min)
        self.w_global_max = float(w_global_max)
        self.initial_observation_weight = float(
            self.w_center
            if initial_observation_weight is None
            else initial_observation_weight
        )
        if not (
            self.w_global_min
            <= self.initial_observation_weight
            <= self.w_global_max
        ):
            raise ValueError(
                "initial_observation_weight must be inside the global w range"
            )
        self.w_init_mode = str(w_init_mode).strip().lower()
        self.w_init_min = float(self.w_global_min if w_init_min is None else w_init_min)
        self.w_init_max = float(self.w_global_max if w_init_max is None else w_init_max)
        self.sweeps_min = int(sweeps_min)
        self.sweeps_max = int(sweeps_max)
        self.sweeps_center = 0.5 * (self.sweeps_min + self.sweeps_max)
        self.sweeps_default = int(np.clip(np.round(self.sweeps_center), self.sweeps_min, self.sweeps_max))
        self.reward_mode = int(reward_mode)
        self.term_bonus = float(term_bonus)
        self.trunc_penalty = float(trunc_penalty)
        self.policy_step_cost_sec = float(policy_step_cost_sec)
        self.potential_progress_scale = float(potential_progress_scale)
        self.episode_stride = max(1, int(episode_stride))
        self.next_index = 0
        self.c_max = float(c_max)
        self.relative_bonus_scale = float(relative_bonus_scale)
        self.reference_margin_sec = float(reference_margin_sec)
        self.anchor_w = float(anchor_w)
        self.anchor_deviation_penalty = float(anchor_deviation_penalty)
        self.obs_mode = str(obs_mode).strip().lower()
        self.obs_include_trace_progress = bool(obs_include_trace_progress)
        self.teacher_action_schedule = tuple((int(end_cycle), int(idx)) for end_cycle, idx in teacher_action_schedule)
        self.teacher_match_bonus = float(teacher_match_bonus)
        self.teacher_mismatch_penalty = float(teacher_mismatch_penalty)
        if baseline_runtimes is None:
            self.baseline_runtimes = tuple(
                float(
                    solve_no_rl_case(
                        params=dict(params),
                        mkw=dict(mkw),
                        solver_tol=float(self.tol),
                        solver_max_iter=int(self.max_cycles),
                        augment_params=augment_setup_params,
                    )["runtime"]
                )
                for mkw, params in self.trace
            )
        else:
            self.baseline_runtimes = tuple(float(x) for x in baseline_runtimes)
        if len(self.baseline_runtimes) != len(self.trace):
            raise ValueError("baseline_runtimes must match trace length")
        if reference_runtimes is None:
            self.reference_runtimes = None
        else:
            self.reference_runtimes = tuple(float(x) for x in reference_runtimes)
            if len(self.reference_runtimes) != len(self.trace):
                raise ValueError("reference_runtimes must match trace length")

        parameter_spec, _fixed_params = build_setup_parameter_spec(tune_dim=int(tune_dim), tune7_variant=str(tune7_variant))
        self.setup_obs_encoder = SetupObsEncoder(
            parameter_spec,
            dict(DEFAULT_SETUP_PARAMS),
            tuple(parameter_spec.parameter_names),
        )
        self._setup_obs_dim = len(parameter_spec.parameter_names)

        if self.action_mode == "discrete_extended":
            if not self.discrete_extended_actions:
                raise ValueError("discrete_extended_actions must be non-empty when action_mode='discrete_extended'")
            self.action_space = spaces.Discrete(len(self.discrete_extended_actions))
        elif self.action_mode == "discrete_switch":
            if len(self.discrete_extended_actions) < 2:
                raise ValueError("discrete_switch requires at least two discrete_extended_actions: aggressive;safe")
            self.action_space = spaces.Discrete(2)
        elif self.action_mode == "discrete_joint":
            if not self.discrete_joint_actions:
                raise ValueError("discrete_joint_actions must be non-empty when action_mode='discrete_joint'")
            self.action_space = spaces.Discrete(len(self.discrete_joint_actions))
        elif self.action_mode == "discrete_w":
            if not self.discrete_w_values:
                raise ValueError("discrete_w_values must be non-empty when action_mode='discrete_w'")
            self.action_space = spaces.Discrete(len(self.discrete_w_values))
        elif self.action_mode == "discrete_blend":
            if not self.discrete_blend_alphas:
                raise ValueError("discrete_blend_alphas must be non-empty when action_mode='discrete_blend'")
            self.action_space = spaces.Discrete(len(self.discrete_blend_alphas))
        elif self.action_mode == "continuous_residual":
            if self.w_only:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        elif self.action_mode == "continuous_hierarchical":
            if self.w_only:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        elif self.w_only:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._action_state_dim = 6
        if self.obs_mode == "full":
            self.obs_dim = 10 + self._setup_obs_dim
        elif self.obs_mode == "solve_only":
            self.obs_dim = 10
        elif self.obs_mode == "full_action":
            self.obs_dim = 10 + self._action_state_dim + self._setup_obs_dim
        elif self.obs_mode == "cycle_setup":
            self.obs_dim = 4 + self._setup_obs_dim
        elif self.obs_mode == "cycle_action_setup":
            self.obs_dim = 4 + self._action_state_dim + self._setup_obs_dim
        elif self.obs_mode == "cycle_only":
            self.obs_dim = 4
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
        if self.obs_include_trace_progress:
            self.obs_dim += 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self._env = None
        self._mkw: Dict[str, Any] | None = None
        self._params: Dict[str, Any] | None = None
        self._setup_time = 0.0
        self._solve_runtime = 0.0
        self._r_prev = 0.0
        self._r_cur = 0.0
        self._cycle = 0
        self._last_w = self.initial_observation_weight
        self._last_coarse_sweeps = 1
        self._last_cycle_type = -1
        self._last_relax_type = -1
        self._last_outer_weight = -1.0
        self._last_add_relax_weight = -1.0
        self._baseline_runtime = float("nan")
        self._reference_runtime = float("nan")
        self._switched_to_safe = False
        self._trace_progress = 0.0

    def _teacher_index_for_cycle(self, cycle: int) -> int | None:
        if not self.teacher_action_schedule:
            return None
        chosen = int(self.teacher_action_schedule[-1][1])
        for end_cycle, idx in self.teacher_action_schedule:
            if int(cycle) < int(end_cycle):
                chosen = int(idx)
                break
        return int(chosen)

    def _obs(self) -> np.ndarray:
        assert self._mkw is not None and self._params is not None
        eps = 1e-30
        c_denom = max(float(np.log(float(max(1.0, self.c_max)) + eps)), eps)
        s1 = float(np.log(float(self._mkw["k"]) + eps) / c_denom)
        s2 = float(np.log(float(self._mkw["c"]) + eps) / c_denom)
        s3 = float(np.log(float(self._mkw["a0"]) + eps) / c_denom)
        grid_norm_div = max(float(self._mkw["nx"]), float(self._mkw["ny"]), float(self._mkw["nz"]), 1.0)
        n_denom = max(float(np.log(float(grid_norm_div) + eps)), eps)
        solve_obs = build_policy_obs(
            r=float(self._r_cur),
            r_prev=float(self._r_prev),
            cycle=int(self._cycle),
            max_cycles=int(self.max_cycles),
            coeff_triplet=(s1, s2, s3),
            grid_triplet=(
                float(np.log(float(self._mkw["nx"]) + eps) / n_denom),
                float(np.log(float(self._mkw["ny"]) + eps) / n_denom),
                float(np.log(float(self._mkw["nz"]) + eps) / n_denom),
            ),
            last_w=float(self._last_w),
        ).astype(np.float32)
        setup_obs = self.setup_obs_encoder.encode(self._params).astype(np.float32)
        action_state = np.array(
            [
                float(self._last_coarse_sweeps),
                float(self._last_cycle_type),
                float(self._last_relax_type),
                float(self._last_outer_weight),
                float(self._last_add_relax_weight),
                1.0 if self._switched_to_safe else 0.0,
            ],
            dtype=np.float32,
        )
        compact_solve_obs = solve_obs[[0, 1, 2, 9]]
        if self.obs_mode == "full":
            obs = np.concatenate([solve_obs, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "solve_only":
            obs = solve_obs.astype(np.float32)
        elif self.obs_mode == "full_action":
            obs = np.concatenate([solve_obs, action_state, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "cycle_setup":
            obs = np.concatenate([compact_solve_obs, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "cycle_action_setup":
            obs = np.concatenate([compact_solve_obs, action_state, setup_obs], axis=0).astype(np.float32)
        else:
            obs = compact_solve_obs.astype(np.float32)
        if self.obs_include_trace_progress:
            obs = np.concatenate([obs, np.asarray([float(self._trace_progress)], dtype=np.float32)], axis=0).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._env is not None:
            self._env.close()
            self._env = None
        idx = self.next_index
        self.next_index = (self.next_index + self.episode_stride) % len(self.trace)
        self._trace_progress = 0.0 if len(self.trace) <= 1 else float(idx) / float(len(self.trace) - 1)
        mkw, params = self.trace[idx]
        self._mkw = dict(mkw)
        self._params = augment_setup_params(params)
        self._env = create_env(**self._mkw)
        prep = self._env.prepare_rl(params=self._params)
        self._setup_time = float(prep.setup_runtime_sec)
        self._solve_runtime = 0.0
        self._r_prev = float(prep.initial_residual_norm)
        self._r_cur = float(prep.initial_residual_norm)
        self._cycle = 0
        if self.action_mode == "continuous_residual" and self.w_init_mode == "uniform_random":
            init_lo = float(np.clip(self.w_init_min, self.w_global_min, self.w_global_max))
            init_hi = float(np.clip(self.w_init_max, self.w_global_min, self.w_global_max))
            if init_hi < init_lo:
                init_lo, init_hi = init_hi, init_lo
            self._last_w = float(self.rng.uniform(init_lo, init_hi))
        else:
            self._last_w = float(self.initial_observation_weight)
        self._last_coarse_sweeps = 1
        self._last_cycle_type = -1
        self._last_relax_type = -1
        self._last_outer_weight = -1.0
        self._last_add_relax_weight = -1.0
        self._baseline_runtime = float(self.baseline_runtimes[idx])
        self._reference_runtime = (
            float("nan") if self.reference_runtimes is None else float(self.reference_runtimes[idx])
        )
        self._switched_to_safe = False
        info = {
            "setup_time": float(self._setup_time),
            "index": int(idx),
            "nx": int(self._mkw["nx"]),
            "ny": int(self._mkw["ny"]),
            "nz": int(self._mkw["nz"]),
            "baseline_runtime": float(self._baseline_runtime),
            "reference_runtime": float(self._reference_runtime),
        }
        return self._obs(), info

    def step(self, action):
        assert self._env is not None
        action_idx = None
        if self.action_mode == "discrete_extended":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_extended_actions) - 1))
            action_idx = int(idx)
            w, sd, su, sc, ct, rt, ow, arw = self.discrete_extended_actions[idx]
        elif self.action_mode == "discrete_switch":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, 1))
            if int(idx) == 1:
                self._switched_to_safe = True
            action_idx = (1 if self._switched_to_safe else 0)
            w, sd, su, sc, ct, rt, ow, arw = self.discrete_extended_actions[action_idx]
        elif self.action_mode == "discrete_joint":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_joint_actions) - 1))
            action_idx = int(idx)
            w, sd, su = self.discrete_joint_actions[idx]
            sc, ct, rt, ow, arw = 1, -1, -1, -1.0, -1.0
        elif self.action_mode == "discrete_w":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_w_values) - 1))
            action_idx = int(idx)
            w = float(self.discrete_w_values[idx])
            sd = self.sweeps_default
            su = self.sweeps_default
            sc, ct, rt, ow, arw = 1, -1, -1, -1.0, -1.0
        elif self.action_mode == "discrete_blend":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_blend_alphas) - 1))
            action_idx = int(idx)
            alpha = float(self.discrete_blend_alphas[idx])
            if self.blend_cutoff_cycles >= 0 and int(self._cycle) >= int(self.blend_cutoff_cycles):
                alpha = 0.0
            if self.blend_decay_tau > 0.0:
                alpha *= float(np.exp(-float(self._cycle) / float(self.blend_decay_tau)))
            if self.blend_progress_end > self.blend_progress_start:
                if float(self._trace_progress) <= float(self.blend_progress_start):
                    alpha = 0.0
                else:
                    ramp = (float(self._trace_progress) - float(self.blend_progress_start)) / (
                        float(self.blend_progress_end) - float(self.blend_progress_start)
                    )
                    alpha *= float(np.clip(ramp, 0.0, 1.0))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            sw, sdown, sup, ssc, sct, srt, sow, sarw = self.blend_safe_action
            aw, adown, aup, asc, act, art, aow, aarw = self.blend_aggr_action
            w = float(sw + alpha * (aw - sw))
            sd = int(np.clip(np.rint(sdown + alpha * (adown - sdown)), self.sweeps_min, self.sweeps_max))
            su = int(np.clip(np.rint(sup + alpha * (aup - sup)), self.sweeps_min, self.sweeps_max))
            sc = max(1, int(np.rint(ssc + alpha * (asc - ssc))))
            ct = int(sct if alpha < 0.5 or int(act) < 0 else act)
            rt = int(srt if alpha < 0.5 or int(art) < 0 else art)
            if float(sow) < 0.0 and float(aow) < 0.0:
                ow = -1.0
            elif float(sow) < 0.0 or float(aow) < 0.0:
                ow = float(sow if alpha < 0.5 else aow)
            else:
                ow = float(sow + alpha * (aow - sow))
            if float(sarw) < 0.0 and float(aarw) < 0.0:
                arw = -1.0
            elif float(sarw) < 0.0 or float(aarw) < 0.0:
                arw = float(sarw if alpha < 0.5 else aarw)
            else:
                arw = float(sarw + alpha * (aarw - sarw))
        elif self.action_mode == "continuous_residual":
            w, sd, su = decode_policy_action_residual(
                action,
                w_only=self.w_only,
                w_center=self.w_center,
                w_scale=self.w_scale,
                w_min=self.w_global_min,
                w_max=self.w_global_max,
                sweeps_min=self.sweeps_min,
                sweeps_max=self.sweeps_max,
                cycle=self._cycle,
                last_w=self._last_w,
                w_init=None,
                sweeps_init=None,
            )
            sc, ct, rt, ow, arw = 1, -1, -1, -1.0, -1.0
        elif self.action_mode == "continuous_hierarchical":
            w, sd, su = decode_policy_action_hierarchical(
                action,
                w_only=self.w_only,
                sweeps_min=self.sweeps_min,
                sweeps_max=self.sweeps_max,
            )
            sc, ct, rt, ow, arw = 1, -1, -1, -1.0, -1.0
        else:
            w, sd, su = decode_policy_action(
                action,
                w_only=self.w_only,
                w_center=self.w_center,
                w_scale=self.w_scale,
                sweeps_min=self.sweeps_min,
                sweeps_max=self.sweeps_max,
                cycle=self._cycle,
                last_w=self._last_w,
                w_init=None,
                sweeps_init=None,
                w_smooth_alpha=0.0,
            )
            sc, ct, rt, ow, arw = 1, -1, -1, -1.0, -1.0
        r_new, dt = self._env.step_rl(
            relax_weight=float(w),
            sweeps_down=int(sd),
            sweeps_up=int(su),
            coarse_sweeps=int(sc),
            cycle_type=(None if int(ct) < 0 else int(ct)),
            relax_type=(None if int(rt) < 0 else int(rt)),
            outer_weight=(None if float(ow) < 0.0 else float(ow)),
            add_relax_weight=(None if float(arw) < 0.0 else float(arw)),
        )
        self._solve_runtime += float(dt)
        self._cycle += 1
        self._last_w = float(w)
        self._last_coarse_sweeps = int(sc)
        self._last_cycle_type = int(ct)
        self._last_relax_type = int(rt)
        self._last_outer_weight = float(ow)
        self._last_add_relax_weight = float(arw)

        eps = 1e-30
        # Match the original AMG env semantics:
        # the next action sees (current residual, previous residual),
        # while the step reward compares the incoming current residual to the new one.
        r_prev = max(float(self._r_cur), eps)
        r_cur = max(float(r_new), eps)
        log_prev = math.log(r_prev + eps)
        log_cur = math.log(r_cur + eps)
        log_drop = max(0.0, log_prev - log_cur)
        rel_drop = (r_prev - r_cur) / r_prev
        dt_eff = max(float(dt), 1.0e-6)
        if self.reward_mode == 0:
            reward = log_drop / dt_eff
        elif self.reward_mode == 1:
            reward = rel_drop / dt_eff
        else:
            reward = -dt_eff
        reward -= self.policy_step_cost_sec
        potential_prev = math.log(max(r_prev, self.tol) + eps)
        potential_cur = math.log(max(r_cur, self.tol) + eps)
        reward += self.potential_progress_scale * (potential_prev - potential_cur)
        terminated = bool(np.isfinite(r_cur) and r_cur <= float(self.tol))
        truncated = bool((not terminated) and self._cycle >= int(self.max_cycles))
        if self.reward_mode == 4 and (terminated or truncated) and np.isfinite(float(self._baseline_runtime)):
            total_runtime = float(self._setup_time + self._solve_runtime)
            reward += float(self.relative_bonus_scale) * float(self._baseline_runtime - total_runtime)
        if self.reward_mode == 5:
            if np.isfinite(float(self.anchor_w)) and float(self.anchor_deviation_penalty) > 0.0:
                reward -= float(self.anchor_deviation_penalty) * abs(float(w) - float(self.anchor_w))
            if (terminated or truncated) and np.isfinite(float(self._reference_runtime)):
                total_runtime = float(self._setup_time + self._solve_runtime)
                reward += float(self.relative_bonus_scale) * float(
                    float(self._reference_runtime) - float(self.reference_margin_sec) - total_runtime
                )
        if action_idx is not None:
            teacher_idx = self._teacher_index_for_cycle(self._cycle - 1)
            if teacher_idx is not None:
                if int(action_idx) == int(teacher_idx):
                    reward += self.teacher_match_bonus
                else:
                    reward -= self.teacher_mismatch_penalty
        if terminated:
            reward += self.term_bonus
        if truncated:
            reward -= self.trunc_penalty
        self._r_prev = float(r_prev)
        self._r_cur = float(r_cur)
        info = {
            "r": float(r_cur),
            "dt_solver": float(dt),
            "setup_time": float(self._setup_time),
            "solve_time": float(self._solve_runtime),
            "time_with_setup": float(self._setup_time + self._solve_runtime),
            "cycle": int(self._cycle),
            "w": float(w),
            "sweeps_down": int(sd),
            "sweeps_up": int(su),
            "coarse_sweeps": int(sc),
            "cycle_type": int(ct),
            "relax_type": int(rt),
            "outer_weight": float(ow),
            "add_relax_weight": float(arw),
            "action_idx": (-1 if action_idx is None else int(action_idx)),
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
