from __future__ import annotations

import _project_paths  # noqa: F401

import time
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from frozen_bandit_step_env import FrozenBanditStepEnv
from hypre.bindings import (
    AttemptOutcome,
    RecoveryOutcome,
    execute_attempt,
)
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    BranchRun,
    TestFinalBanditConfig,
    augment_setup_params,
    solve_no_rl_case,
)


class OnlineBanditStepEnv(FrozenBanditStepEnv):
    """Joint LinUCB/PPO environment with one default recovery attempt."""

    def __init__(
        self,
        *,
        instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
        branch: BranchRun,
        bandit_cfg: TestFinalBanditConfig,
        **kwargs: Any,
    ) -> None:
        if not instances:
            raise ValueError("instances must be non-empty")
        self.instances = [
            (dict(mkw), np.asarray(context, dtype=float))
            for mkw, context in instances
        ]
        self.branch = branch
        self.bandit_cfg = bandit_cfg
        self.next_instance_index = 0
        self.previous_update_sec = 0.0
        self.completed_episodes: list[Dict[str, Any]] = []
        self.bandit_updates: list[Dict[str, Any]] = []
        self.interaction_wall_sec = 0.0

        self._instance_index = -1
        self._context = np.zeros(1, dtype=float)
        self._selection_info: Dict[str, Any] = {}
        self._episode_select_sec = 0.0
        self._episode_actions: list[Dict[str, Any]] = []
        self._episode_wall_start = 0.0
        self._episode_finalized = True

        placeholder_mkw = dict(self.instances[0][0])
        super().__init__(
            trace=[(placeholder_mkw, dict(DEFAULT_SETUP_PARAMS))],
            baseline_runtimes=[float("nan")],
            **kwargs,
        )

    def _select_setup(
        self,
        context: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
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

    def _cancel_pending(self) -> None:
        model = getattr(self.branch.policy, "model", self.branch.policy)
        cancel_pending = getattr(model, "cancel_pending", None)
        if callable(cancel_pending):
            cancel_pending()

    def _update_bandit(
        self,
        *,
        params: Dict[str, Any],
        outcome: Dict[str, Any],
        select_sec: float,
        phase: str,
    ) -> Dict[str, Any] | None:
        if bool(outcome.get("unrecovered_failure", outcome.get("failed", False))):
            self._cancel_pending()
            return None

        loss_started = time.perf_counter()
        loss_eval_sec = float(time.perf_counter() - loss_started)
        loss = float(
            outcome["runtime"]
            + select_sec
            + loss_eval_sec
            + self.previous_update_sec
        )
        update_started = time.perf_counter()
        self.branch.policy.update(
            loss=loss,
            context=np.asarray(self._context, dtype=float),
            params=dict(params),
            outcome=dict(outcome),
        )
        update_sec = float(time.perf_counter() - update_started)
        self.previous_update_sec = update_sec
        record = {
            "update_index": int(len(self.bandit_updates)),
            "phase": str(phase),
            "instance_index": int(self._instance_index),
            "arm_index": int(self._selection_info.get("arm_index", -1)),
            "loss_sec": loss,
            "select_sec": float(select_sec),
            "loss_eval_sec": loss_eval_sec,
            "update_sec": update_sec,
            "bandit_update_committed": True,
        }
        self.bandit_updates.append(record)
        return record

    def _default_fallback(self) -> Dict[str, Any]:
        return solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(self._mkw or {}),
            solver_tol=float(self.tol),
            solver_max_iter=int(self.max_cycles),
            augment_params=augment_setup_params,
        )

    def _record_episode(
        self,
        *,
        outcome: Dict[str, Any],
        update: Dict[str, Any] | None,
        params: Dict[str, Any],
        actions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        record = {
            "episode_index": int(len(self.completed_episodes)),
            "instance_index": int(self._instance_index),
            "mkw": dict(self._mkw or {}),
            "context": self._context.tolist(),
            "params": dict(params),
            "selection": dict(self._selection_info),
            "setup_failures": [],
            "outcome": dict(outcome),
            "actions": [dict(action) for action in actions],
            "bandit_update": update,
            "episode_wall_sec": float(
                time.perf_counter() - self._episode_wall_start
            ),
        }
        self.completed_episodes.append(record)
        self._episode_finalized = True
        return record

    def _setup_failure_recovery(
        self,
        *,
        exc: Exception,
        params: Dict[str, Any],
        select_sec: float,
        started_at: float,
    ) -> None:
        primary = AttemptOutcome.from_exception(
            exc,
            elapsed_sec=float(time.perf_counter() - started_at),
        )
        recovery = RecoveryOutcome(
            primary=primary,
            fallback=execute_attempt(self._default_fallback),
        )
        outcome = recovery.to_result()
        outcome.update(
            {
                "recovery_protocol_applied": True,
                "bandit_update_committed": False,
                "controller_update_committed": False,
            }
        )
        update = self._update_bandit(
            params=params,
            outcome=outcome,
            select_sec=select_sec,
            phase="setup_failure_recovery",
        )
        outcome["bandit_update_committed"] = update is not None
        self._record_episode(
            outcome=outcome,
            update=update,
            params=params,
            actions=(),
        )

    def reset(self, *, seed=None, options=None):
        if not self._episode_finalized:
            self.discard_incomplete_episode()
        call_started = time.perf_counter()
        first_seed = seed

        for _ in range(len(self.instances)):
            self._instance_index = int(self.next_instance_index)
            self.next_instance_index = (
                self.next_instance_index + 1
            ) % len(self.instances)
            mkw, context = self.instances[self._instance_index]
            self._mkw = dict(mkw)
            self._context = np.asarray(context, dtype=float)
            self._episode_actions = []
            self._episode_wall_start = time.perf_counter()
            self._episode_finalized = False

            params, selection_info, select_sec = self._select_setup(self._context)
            self._selection_info = dict(selection_info)
            self._episode_select_sec = float(select_sec)
            self.trace = [(dict(mkw), dict(params))]
            self.baseline_runtimes = (float("nan"),)
            self.next_index = 0
            setup_started = time.perf_counter()
            try:
                obs, info = super().reset(seed=first_seed, options=options)
            except Exception as exc:
                first_seed = None
                if self._env is not None:
                    self._env.close()
                    self._env = None
                self._setup_failure_recovery(
                    exc=exc,
                    params=params,
                    select_sec=select_sec,
                    started_at=setup_started,
                )
                continue

            info = dict(info)
            info.update(
                {
                    "online_instance_index": int(self._instance_index),
                    "setup_params": dict(self._params or params),
                    "setup_attempts": 1,
                    **{
                        f"bandit_{key}": value
                        for key, value in selection_info.items()
                    },
                }
            )
            self.interaction_wall_sec += float(time.perf_counter() - call_started)
            return obs, info

        self.interaction_wall_sec += float(time.perf_counter() - call_started)
        raise RuntimeError(
            "Every available instance had an unrecoverable setup failure"
        )

    def _finalize_primary(
        self,
        *,
        primary: AttemptOutcome,
        params: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        recovery = RecoveryOutcome(
            primary=primary,
            fallback=(
                None
                if primary.succeeded
                else execute_attempt(self._default_fallback)
            ),
        )
        outcome = recovery.to_result()
        outcome["recovery_protocol_applied"] = True
        outcome["controller_update_committed"] = not recovery.unrecovered_failure
        update = self._update_bandit(
            params=params,
            outcome=outcome,
            select_sec=float(self._episode_select_sec),
            phase="joint_solve",
        )
        outcome["bandit_update_committed"] = update is not None
        return outcome, update

    def step(self, action):
        call_started = time.perf_counter()
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        except Exception as exc:
            primary = AttemptOutcome.from_exception(
                exc,
                elapsed_sec=float(time.perf_counter() - call_started),
            )
            primary = AttemptOutcome(
                status=primary.status,
                setup_runtime_sec=float(self._setup_time),
                solve_runtime_sec=float(
                    self._solve_runtime + primary.solve_runtime_sec
                ),
                controller_runtime_sec=0.0,
                failure_reason=primary.failure_reason,
                residual_norm=float(self._r_cur),
                cycles=int(self._cycle),
                result=primary.result,
            )
            outcome, update = self._finalize_primary(
                primary=primary,
                params=dict(self._params or {}),
            )
            record = self._record_episode(
                outcome=outcome,
                update=update,
                params=dict(self._params or {}),
                actions=self._episode_actions,
            )
            self.interaction_wall_sec += float(time.perf_counter() - call_started)
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                -float(outcome["runtime"]),
                bool(outcome.get("recovered", False)),
                bool(outcome.get("unrecovered_failure", True)),
                {
                    **outcome,
                    "online_episode_index": int(record["episode_index"]),
                },
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
            primary = AttemptOutcome.from_mapping(
                {
                    "runtime": float(info["time_with_setup"]),
                    "setup_runtime": float(info["setup_time"]),
                    "solve_runtime": float(info["solve_time"]),
                    "failed": bool(truncated or not terminated),
                    "failure_reason": (
                        "" if terminated
                        else "max_cycles_reached_without_convergence"
                    ),
                    "failure_stage": "" if terminated else "solve",
                    "residual_norm": float(info["r"]),
                    "iterations": int(info["cycle"]),
                    "final_w": float(info["w"]),
                }
            )
            outcome, update = self._finalize_primary(
                primary=primary,
                params=dict(self._params or {}),
            )
            if truncated:
                reward += float(self.trunc_penalty)
                if outcome.get("fallback_used", False):
                    reward -= float(
                        outcome["fallback_setup_runtime"]
                        + outcome["fallback_solve_runtime"]
                    )
                terminated = bool(outcome.get("recovered", False))
                truncated = bool(outcome.get("unrecovered_failure", False))
            record = self._record_episode(
                outcome=outcome,
                update=update,
                params=dict(self._params or {}),
                actions=self._episode_actions,
            )
            info.update(outcome)
            info["online_episode_index"] = int(record["episode_index"])
            if update is not None:
                info["bandit_loss_sec"] = float(update["loss_sec"])
        self.interaction_wall_sec += float(time.perf_counter() - call_started)
        return obs, reward, terminated, truncated, info

    def discard_incomplete_episode(self) -> None:
        if self._episode_finalized:
            return
        self._cancel_pending()
        self._episode_finalized = True
        if self._env is not None:
            self._env.close()
            self._env = None
