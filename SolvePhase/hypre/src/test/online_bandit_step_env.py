from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np

from frozen_bandit_step_env import FrozenBanditStepEnv
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    FAIL_RUNTIME_SEC,
    BranchRun,
    TestFinalBanditConfig,
    rolling_success_scale_sec,
    runtime_loss_sec_test_final,
)


class OnlineBanditStepEnv(FrozenBanditStepEnv):
    """Train setup LinUCB per problem and solve PPO per AMG cycle."""

    def __init__(
        self,
        *,
        instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
        branch: BranchRun,
        bandit_cfg: TestFinalBanditConfig,
        failure_scale_min_runtime_sec: float = 1.0e-3,
        max_setup_attempts: int = 8,
        convergence_guard_start_fraction: float = 0.6,
        convergence_guard_multiplier: float = 2.0,
        **kwargs: Any,
    ) -> None:
        if not instances:
            raise ValueError("instances must be non-empty")
        if max_setup_attempts <= 0:
            raise ValueError("max_setup_attempts must be positive")
        if not 0.0 <= convergence_guard_start_fraction < 1.0:
            raise ValueError("convergence_guard_start_fraction must be in [0, 1)")
        if convergence_guard_multiplier < 0.0:
            raise ValueError("convergence_guard_multiplier must be non-negative")

        self.instances = [(dict(mkw), np.asarray(context, dtype=float)) for mkw, context in instances]
        self.branch = branch
        self.bandit_cfg = bandit_cfg
        self.failure_scale_min_runtime_sec = float(failure_scale_min_runtime_sec)
        self.max_setup_attempts = int(max_setup_attempts)
        self.convergence_guard_start_fraction = float(convergence_guard_start_fraction)
        self.convergence_guard_multiplier = float(convergence_guard_multiplier)
        self.next_instance_index = 0
        self.success_runtime_history: Deque[float] = deque(
            maxlen=max(1, int(bandit_cfg.failure_scale_window))
        )
        self.previous_update_sec = 0.0
        self.completed_episodes: list[Dict[str, Any]] = []
        self.bandit_updates: list[Dict[str, Any]] = []
        self.interaction_wall_sec = 0.0

        self._instance_index = -1
        self._context = np.zeros(1, dtype=float)
        self._selection_info: Dict[str, Any] = {}
        self._episode_select_sec = 0.0
        self._episode_bandit_enabled = False
        self._episode_actions: list[Dict[str, Any]] = []
        self._episode_setup_failures: list[Dict[str, Any]] = []
        self._episode_wall_start = 0.0
        self._episode_finalized = True

        placeholder_mkw = dict(self.instances[0][0])
        super().__init__(
            trace=[(placeholder_mkw, dict(DEFAULT_SETUP_PARAMS))],
            baseline_runtimes=[float("nan")],
            **kwargs,
        )

    def _select_setup(self, context: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        started = time.perf_counter()
        selected = self.branch.policy.select(
            context=np.asarray(context, dtype=float),
            parameter_space=self.branch.parameter_space,
        )
        elapsed = float(time.perf_counter() - started)
        if isinstance(selected, tuple):
            params, info = selected
        else:
            params, info = selected, {}
        return dict(params), dict(info), elapsed

    def _update_bandit(
        self,
        *,
        context: np.ndarray,
        params: Dict[str, Any],
        outcome: Dict[str, Any],
        select_sec: float,
        phase: str,
    ) -> Dict[str, Any]:
        loss_started = time.perf_counter()
        success_scale = rolling_success_scale_sec(
            self.success_runtime_history,
            b_min_runtime_sec=float(self.failure_scale_min_runtime_sec),
        )
        policy_cost_sec = max(0.0, float(self.policy_step_cost_sec)) * max(
            0, int(outcome.get("iterations", 0))
        )
        loss_outcome = dict(outcome)
        native_runtime = float(loss_outcome.get("runtime", FAIL_RUNTIME_SEC))
        if np.isfinite(native_runtime) and native_runtime < float(FAIL_RUNTIME_SEC):
            loss_outcome["runtime"] = native_runtime + policy_cost_sec
            loss_outcome["solve_runtime"] = (
                float(loss_outcome.get("solve_runtime", 0.0)) + policy_cost_sec
            )
        base_loss = runtime_loss_sec_test_final(
            outcome=loss_outcome,
            fail_runtime_sec=float(FAIL_RUNTIME_SEC),
            solver_tol=float(self.tol),
            success_runtime_scale_sec=float(success_scale),
            failure_penalty_multiplier=float(self.bandit_cfg.failure_penalty_multiplier),
            failure_severity_cap=float(self.bandit_cfg.failure_severity_cap),
            structural_failure_surcharge_multiplier=float(
                self.bandit_cfg.structural_failure_surcharge_multiplier
            ),
        )
        convergence_guard_sec = 0.0
        if not bool(outcome.get("failed", False)):
            cycle_fraction = float(outcome.get("iterations", 0)) / float(max(1, self.max_cycles))
            if cycle_fraction > self.convergence_guard_start_fraction:
                guard_progress = (cycle_fraction - self.convergence_guard_start_fraction) / (
                    1.0 - self.convergence_guard_start_fraction
                )
                convergence_guard_sec = float(
                    success_scale * self.convergence_guard_multiplier * guard_progress**2
                )
                base_loss += convergence_guard_sec
        loss_eval_sec = float(time.perf_counter() - loss_started)
        loss = float(base_loss + select_sec + loss_eval_sec + self.previous_update_sec)

        update_started = time.perf_counter()
        self.branch.policy.update(
            loss=loss,
            context=np.asarray(context, dtype=float),
            params=dict(params),
            outcome=dict(outcome),
        )
        update_sec = float(time.perf_counter() - update_started)
        self.previous_update_sec = update_sec

        if not bool(outcome.get("failed", False)):
            runtime = float(loss_outcome["runtime"])
            if np.isfinite(runtime) and runtime > 0.0:
                self.success_runtime_history.append(runtime)

        record = {
            "update_index": int(len(self.bandit_updates)),
            "phase": str(phase),
            "instance_index": int(self._instance_index),
            "arm_index": int(self._selection_info.get("arm_index", -1)),
            "loss_sec": float(loss),
            "base_loss_sec": float(base_loss),
            "success_scale_sec": float(success_scale),
            "convergence_guard_sec": float(convergence_guard_sec),
            "policy_cost_sec": float(policy_cost_sec),
            "select_sec": float(select_sec),
            "loss_eval_sec": float(loss_eval_sec),
            "update_sec": float(update_sec),
            "failed": bool(outcome.get("failed", False)),
            "structural_fail": bool(outcome.get("structural_fail", False)),
        }
        self.bandit_updates.append(record)
        return record

    def _setup_failure_outcome(self, exc: Exception) -> Dict[str, Any]:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "failed": True,
            "structural_fail": True,
            "failure_reason": f"setup_exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": 0,
        }

    def reset(self, *, seed=None, options=None):
        if not self._episode_finalized:
            self.discard_incomplete_episode()
        call_started = time.perf_counter()
        self._instance_index = int(self.next_instance_index)
        self.next_instance_index = (self.next_instance_index + 1) % len(self.instances)
        mkw, context = self.instances[self._instance_index]
        self._context = np.asarray(context, dtype=float)
        self._episode_actions = []
        self._episode_setup_failures = []
        self._episode_wall_start = time.perf_counter()
        self._episode_finalized = False

        first_reset_seed = seed
        for _attempt in range(self.max_setup_attempts):
            params, selection_info, select_sec = self._select_setup(self._context)
            self._selection_info = dict(selection_info)
            self._episode_select_sec = float(select_sec)
            self.trace = [(dict(mkw), dict(params))]
            self.baseline_runtimes = (float("nan"),)
            self.next_index = 0
            try:
                obs, info = super().reset(seed=first_reset_seed, options=options)
            except Exception as exc:
                first_reset_seed = None
                outcome = self._setup_failure_outcome(exc)
                update = self._update_bandit(
                    context=self._context,
                    params=params,
                    outcome=outcome,
                    select_sec=select_sec,
                    phase="setup_failure",
                )
                self._episode_setup_failures.append(
                    {"params": dict(params), "outcome": outcome, "bandit_update": update}
                )
                continue

            self._episode_bandit_enabled = True
            info = dict(info)
            info.update(
                {
                    "online_instance_index": int(self._instance_index),
                    "setup_params": dict(self._params or params),
                    "setup_attempts": int(len(self._episode_setup_failures) + 1),
                    **{f"bandit_{key}": value for key, value in selection_info.items()},
                }
            )
            self.interaction_wall_sec += float(time.perf_counter() - call_started)
            return obs, info

        self.trace = [(dict(mkw), dict(DEFAULT_SETUP_PARAMS))]
        self.baseline_runtimes = (float("nan"),)
        self.next_index = 0
        try:
            obs, info = super().reset(seed=first_reset_seed, options=options)
        except Exception as exc:
            raise RuntimeError("Default setup failed after all online setup attempts") from exc
        self._selection_info = {"fallback_default": True, "arm_index": -1}
        self._episode_select_sec = 0.0
        self._episode_bandit_enabled = False
        info = dict(info)
        info.update(
            {
                "online_instance_index": int(self._instance_index),
                "setup_params": dict(self._params or DEFAULT_SETUP_PARAMS),
                "setup_attempts": int(len(self._episode_setup_failures) + 1),
                "bandit_fallback_default": True,
            }
        )
        self.interaction_wall_sec += float(time.perf_counter() - call_started)
        return obs, info

    def _finalize_episode(self, outcome: Dict[str, Any]) -> Dict[str, Any] | None:
        if self._episode_finalized:
            return None
        update = None
        if self._episode_bandit_enabled:
            update = self._update_bandit(
                context=self._context,
                params=dict(self._params or {}),
                outcome=outcome,
                select_sec=float(self._episode_select_sec),
                phase="joint_solve",
            )
        record = {
            "episode_index": int(len(self.completed_episodes)),
            "instance_index": int(self._instance_index),
            "mkw": dict(self._mkw or {}),
            "context": self._context.tolist(),
            "params": dict(self._params or {}),
            "selection": dict(self._selection_info),
            "setup_failures": list(self._episode_setup_failures),
            "outcome": dict(outcome),
            "actions": list(self._episode_actions),
            "bandit_update": update,
            "episode_wall_sec": float(time.perf_counter() - self._episode_wall_start),
        }
        self.completed_episodes.append(record)
        self._episode_finalized = True
        return record

    def step(self, action):
        call_started = time.perf_counter()
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        except Exception as exc:
            outcome = {
                "runtime": float(FAIL_RUNTIME_SEC),
                "setup_runtime": float(self._setup_time),
                "solve_runtime": float(self._solve_runtime),
                "failed": True,
                "structural_fail": True,
                "failure_reason": f"solve_exception:{type(exc).__name__}:{exc}",
                "residual_norm": float("inf"),
                "iterations": int(self._cycle),
            }
            record = self._finalize_episode(outcome)
            info = {
                "failed": True,
                "failure_reason": outcome["failure_reason"],
                "online_episode_index": int(record["episode_index"] if record else -1),
            }
            self.interaction_wall_sec += float(time.perf_counter() - call_started)
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                -float(self.trunc_penalty),
                False,
                True,
                info,
            )

        info = dict(info)
        self._episode_actions.append(
            {
                "cycle": int(info["cycle"]),
                "w": float(info["w"]),
                "sweeps_down": int(info["sweeps_down"]),
                "sweeps_up": int(info["sweeps_up"]),
                "dt_solver": float(info["dt_solver"]),
                "residual_norm": float(info["r"]),
            }
        )
        if terminated or truncated:
            outcome = {
                "runtime": float(info["time_with_setup"]),
                "setup_runtime": float(info["setup_time"]),
                "solve_runtime": float(info["solve_time"]),
                "failed": bool(truncated or not terminated),
                "structural_fail": False,
                "failure_reason": "" if terminated else "max_cycles_reached_without_convergence",
                "residual_norm": float(info["r"]),
                "iterations": int(info["cycle"]),
                "final_w": float(info["w"]),
            }
            record = self._finalize_episode(outcome)
            if record is not None:
                info["online_episode_index"] = int(record["episode_index"])
                if record["bandit_update"] is not None:
                    info["bandit_loss_sec"] = float(record["bandit_update"]["loss_sec"])
        self.interaction_wall_sec += float(time.perf_counter() - call_started)
        return obs, reward, terminated, truncated, info

    def discard_incomplete_episode(self) -> None:
        if self._episode_finalized:
            return
        if self._episode_bandit_enabled and hasattr(self.branch.policy, "cancel_pending"):
            self.branch.policy.cancel_pending()
        self._episode_finalized = True
        if self._env is not None:
            self._env.close()
            self._env = None
