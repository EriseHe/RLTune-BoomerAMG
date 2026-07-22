"""Typed native attempt and default-fallback outcomes for AMG experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Callable, Mapping

from .boomeramg import AMGNativeError, NativeCode


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    SETUP_FAILURE = "setup_failure"
    SOLVE_FAILURE = "solve_failure"
    NONFINITE_RESULT = "nonfinite_result"
    NONCONVERGENCE = "nonconvergence"


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) and result >= 0.0 else float(default)


def _status_from_mapping(result: Mapping[str, Any]) -> AttemptStatus:
    explicit = str(result.get("attempt_status", "")).strip().lower()
    if explicit:
        try:
            return AttemptStatus(explicit)
        except ValueError:
            pass
    native_status = str(result.get("native_status", "")).strip().lower()
    if native_status == "max_cycles":
        return AttemptStatus.NONCONVERGENCE
    if native_status == "converged" and not bool(result.get("failed", False)):
        return AttemptStatus.SUCCESS
    if not bool(result.get("failed", False)):
        return AttemptStatus.SUCCESS
    reason = str(result.get("failure_reason", "")).lower()
    stage = str(result.get("failure_stage", "")).lower()
    residual = result.get("residual_norm", float("nan"))
    if bool(result.get("structural_fail", False)) or stage in {"create", "setup"}:
        return AttemptStatus.SETUP_FAILURE
    if "setup" in reason or "create" in reason:
        return AttemptStatus.SETUP_FAILURE
    try:
        residual_is_finite = math.isfinite(float(residual))
    except (TypeError, ValueError):
        residual_is_finite = False
    if not residual_is_finite or "nonfinite" in reason or "non_finite" in reason:
        return AttemptStatus.NONFINITE_RESULT
    if "max_cycle" in reason or "max_iter" in reason or "residual_above" in reason:
        return AttemptStatus.NONCONVERGENCE
    return AttemptStatus.SOLVE_FAILURE


@dataclass(frozen=True)
class AttemptOutcome:
    status: AttemptStatus
    setup_runtime_sec: float
    solve_runtime_sec: float
    controller_runtime_sec: float
    failure_reason: str = ""
    residual_norm: float = float("nan")
    cycles: int = 0
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status is AttemptStatus.SUCCESS

    @property
    def failure_stage(self) -> str:
        if self.status is AttemptStatus.SETUP_FAILURE:
            return "setup"
        if self.status is AttemptStatus.SUCCESS:
            return ""
        return "solve"

    @property
    def native_runtime_sec(self) -> float:
        return float(self.setup_runtime_sec + self.solve_runtime_sec)

    @property
    def end_to_end_runtime_sec(self) -> float:
        return float(self.native_runtime_sec + self.controller_runtime_sec)

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> "AttemptOutcome":
        payload = dict(result)
        setup = _finite_nonnegative(payload.get("setup_runtime", 0.0))
        controller = _finite_nonnegative(payload.get("infer_runtime", 0.0))
        solve = _finite_nonnegative(
            payload.get(
                "native_solve_runtime",
                max(0.0, _finite_nonnegative(payload.get("solve_runtime", 0.0)) - controller),
            )
        )
        reported = _finite_nonnegative(payload.get("runtime", setup + solve + controller))
        if setup + solve <= 0.0 and reported > controller:
            setup = reported - controller
        status = _status_from_mapping(payload)
        return cls(
            status=status,
            setup_runtime_sec=setup,
            solve_runtime_sec=solve,
            controller_runtime_sec=controller,
            failure_reason=str(payload.get("failure_reason", "")),
            residual_norm=float(payload.get("residual_norm", float("nan"))),
            cycles=int(payload.get("iterations", payload.get("cycles", 0))),
            result=payload,
        )

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        elapsed_sec: float,
    ) -> "AttemptOutcome":
        if isinstance(exc, AMGNativeError):
            status = (
                AttemptStatus.SETUP_FAILURE
                if exc.code in {NativeCode.CREATE_ERROR, NativeCode.SETUP_ERROR}
                else AttemptStatus.NONFINITE_RESULT
                if exc.code == NativeCode.NONFINITE_RESULT
                else AttemptStatus.SOLVE_FAILURE
            )
            setup = float(exc.setup_runtime_sec)
            solve = float(exc.solve_runtime_sec)
        else:
            status = AttemptStatus.SETUP_FAILURE
            setup = _finite_nonnegative(elapsed_sec)
            solve = 0.0
        reason = f"exception:{type(exc).__name__}:{exc}"
        return cls(
            status=status,
            setup_runtime_sec=setup,
            solve_runtime_sec=solve,
            controller_runtime_sec=0.0,
            failure_reason=reason,
            result={
                "runtime": float(setup + solve),
                "setup_runtime": setup,
                "solve_runtime": solve,
                "infer_runtime": 0.0,
                "failed": True,
                "failure_reason": reason,
                "failure_stage": "setup" if status is AttemptStatus.SETUP_FAILURE else "solve",
                "residual_norm": float("nan"),
                "iterations": 0,
            },
        )


@dataclass(frozen=True, init=False)
class RecoveryOutcome:
    primary_attempts: tuple[AttemptOutcome, ...]
    fallback: AttemptOutcome | None

    def __init__(
        self,
        primary: AttemptOutcome | None = None,
        fallback: AttemptOutcome | None = None,
        *,
        primary_attempts: tuple[AttemptOutcome, ...] | list[AttemptOutcome] | None = None,
    ) -> None:
        attempts = (
            tuple(primary_attempts)
            if primary_attempts is not None
            else (() if primary is None else (primary,))
        )
        if not attempts:
            raise ValueError("RecoveryOutcome requires at least one primary attempt")
        if primary is not None and primary_attempts is not None:
            raise ValueError("Provide primary or primary_attempts, not both")
        object.__setattr__(self, "primary_attempts", attempts)
        object.__setattr__(self, "fallback", fallback)

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> "RecoveryOutcome":
        """Reconstruct typed attempts from a previously reported recovery row."""

        payload = dict(result)
        primary_rows = payload.get("primary_attempts")
        attempts: list[AttemptOutcome] = []
        if isinstance(primary_rows, list) and primary_rows:
            for row in primary_rows:
                attempt_payload = dict(payload)
                attempt_payload.update(
                    {
                        "attempt_status": row.get("status", "solve_failure"),
                        "setup_runtime": row.get("setup_runtime", 0.0),
                        "native_solve_runtime": row.get("solve_runtime", 0.0),
                        "solve_runtime": row.get("solve_runtime", 0.0),
                        "infer_runtime": row.get("controller_runtime", 0.0),
                        "failure_reason": row.get("failure_reason", ""),
                        "failure_stage": row.get("failure_stage", ""),
                        "residual_norm": row.get("residual_norm", float("nan")),
                        "iterations": row.get("cycles", 0),
                        "failed": row.get("status", "success") != "success",
                    }
                )
                attempts.append(AttemptOutcome.from_mapping(attempt_payload))
        else:
            primary_payload = dict(payload)
            primary_payload.update(
                {
                    "attempt_status": payload.get(
                        "primary_status",
                        payload.get("attempt_status", "success"),
                    ),
                    "setup_runtime": payload.get(
                        "primary_setup_runtime",
                        payload.get("setup_runtime", 0.0),
                    ),
                    "native_solve_runtime": payload.get(
                        "primary_solve_runtime",
                        payload.get("native_solve_runtime", 0.0),
                    ),
                    "solve_runtime": payload.get(
                        "primary_solve_runtime",
                        payload.get("native_solve_runtime", 0.0),
                    ),
                    "infer_runtime": payload.get(
                        "primary_controller_runtime",
                        payload.get("infer_runtime", 0.0),
                    ),
                    "failure_reason": payload.get(
                        "primary_failure_reason",
                        payload.get("failure_reason", ""),
                    ),
                    "residual_norm": payload.get(
                        "primary_residual_norm",
                        payload.get("residual_norm", float("nan")),
                    ),
                    "iterations": payload.get(
                        "primary_cycles",
                        payload.get("iterations", 0),
                    ),
                    "failed": str(
                        payload.get(
                            "primary_status",
                            payload.get("attempt_status", "success"),
                        )
                    )
                    != "success",
                }
            )
            attempts.append(AttemptOutcome.from_mapping(primary_payload))

        fallback = None
        if bool(payload.get("fallback_used", False)):
            fallback_payload = {
                "attempt_status": payload.get("fallback_status", "solve_failure"),
                "setup_runtime": payload.get("fallback_setup_runtime", 0.0),
                "native_solve_runtime": payload.get("fallback_solve_runtime", 0.0),
                "solve_runtime": payload.get("fallback_solve_runtime", 0.0),
                "infer_runtime": payload.get("fallback_controller_runtime", 0.0),
                "failure_reason": payload.get("fallback_failure_reason", ""),
                "failed": str(payload.get("fallback_status", "success"))
                != "success",
                "residual_norm": payload.get("fallback_residual_norm", float("nan")),
                "iterations": payload.get("fallback_cycles", 0),
            }
            fallback = AttemptOutcome.from_mapping(fallback_payload)
        return cls(primary_attempts=tuple(attempts), fallback=fallback)

    @property
    def primary(self) -> AttemptOutcome:
        """The final learned attempt, preserving the legacy singular API."""

        return self.primary_attempts[-1]

    @property
    def first_primary(self) -> AttemptOutcome:
        return self.primary_attempts[0]

    @property
    def primary_attempt_count(self) -> int:
        return len(self.primary_attempts)

    @property
    def learned_reselection_count(self) -> int:
        return max(0, self.primary_attempt_count - 1)

    @property
    def learned_recovered(self) -> bool:
        return bool(
            self.primary_attempt_count > 1
            and not self.first_primary.succeeded
            and self.primary.succeeded
        )

    @property
    def fallback_used(self) -> bool:
        return self.fallback is not None

    @property
    def recovered(self) -> bool:
        return bool(
            not self.first_primary.succeeded
            and (
                self.learned_recovered
                or (
                    self.fallback is not None
                    and self.fallback.succeeded
                )
            )
        )

    @property
    def unrecovered_failure(self) -> bool:
        return bool(
            not self.primary.succeeded
            and (self.fallback is None or not self.fallback.succeeded)
        )

    @property
    def native_runtime_sec(self) -> float:
        fallback = 0.0 if self.fallback is None else self.fallback.native_runtime_sec
        return float(
            sum(attempt.native_runtime_sec for attempt in self.primary_attempts)
            + fallback
        )

    @property
    def controller_runtime_sec(self) -> float:
        fallback = 0.0 if self.fallback is None else self.fallback.controller_runtime_sec
        return float(
            sum(attempt.controller_runtime_sec for attempt in self.primary_attempts)
            + fallback
        )

    @property
    def end_to_end_runtime_sec(self) -> float:
        return float(self.native_runtime_sec + self.controller_runtime_sec)

    def to_result(self) -> dict[str, Any]:
        result = dict(self.primary.result)
        fallback = self.fallback
        primary_setup_runtime = float(
            sum(attempt.setup_runtime_sec for attempt in self.primary_attempts)
        )
        primary_solve_runtime = float(
            sum(attempt.solve_runtime_sec for attempt in self.primary_attempts)
        )
        primary_controller_runtime = float(
            sum(attempt.controller_runtime_sec for attempt in self.primary_attempts)
        )
        attempt_rows = [
            {
                "attempt_index": int(index),
                "status": attempt.status.value,
                "failure_stage": attempt.failure_stage,
                "failure_reason": attempt.failure_reason,
                "setup_runtime": float(attempt.setup_runtime_sec),
                "solve_runtime": float(attempt.solve_runtime_sec),
                "controller_runtime": float(attempt.controller_runtime_sec),
                "residual_norm": float(attempt.residual_norm),
                "cycles": int(attempt.cycles),
            }
            for index, attempt in enumerate(self.primary_attempts, start=1)
        ]
        result.update(
            {
                "runtime": self.end_to_end_runtime_sec,
                "native_runtime": self.native_runtime_sec,
                "setup_runtime": float(
                    primary_setup_runtime
                    + (0.0 if fallback is None else fallback.setup_runtime_sec)
                ),
                "native_solve_runtime": float(
                    primary_solve_runtime
                    + (0.0 if fallback is None else fallback.solve_runtime_sec)
                ),
                "solve_runtime": float(
                    primary_solve_runtime
                    + (0.0 if fallback is None else fallback.solve_runtime_sec)
                    + self.controller_runtime_sec
                ),
                "infer_runtime": self.controller_runtime_sec,
                "failed": self.unrecovered_failure,
                "failure_reason": (
                    self.primary.failure_reason if self.unrecovered_failure else ""
                ),
                "failure_stage": self.primary.failure_stage,
                "primary_status": self.primary.status.value,
                "primary_failure_reason": self.primary.failure_reason,
                "primary_setup_runtime": primary_setup_runtime,
                "primary_solve_runtime": primary_solve_runtime,
                "primary_controller_runtime": primary_controller_runtime,
                "primary_residual_norm": self.primary.residual_norm,
                "primary_cycles": self.primary.cycles,
                "first_primary_status": self.first_primary.status.value,
                "primary_attempt_count": self.primary_attempt_count,
                "learned_reselection_count": self.learned_reselection_count,
                "learned_recovered": self.learned_recovered,
                "primary_attempts": attempt_rows,
                "fallback_used": self.fallback_used,
                "fallback_status": "not_run" if fallback is None else fallback.status.value,
                "fallback_failure_reason": "" if fallback is None else fallback.failure_reason,
                "fallback_setup_runtime": 0.0 if fallback is None else fallback.setup_runtime_sec,
                "fallback_solve_runtime": 0.0 if fallback is None else fallback.solve_runtime_sec,
                "fallback_controller_runtime": 0.0 if fallback is None else fallback.controller_runtime_sec,
                "recovered": self.recovered,
                "unrecovered_failure": self.unrecovered_failure,
            }
        )
        return result


def execute_attempt(attempt: Callable[[], Mapping[str, Any] | AttemptOutcome]) -> AttemptOutcome:
    started = time.perf_counter()
    try:
        result = attempt()
    except Exception as exc:
        return AttemptOutcome.from_exception(
            exc,
            elapsed_sec=float(time.perf_counter() - started),
        )
    if isinstance(result, AttemptOutcome):
        return result
    return AttemptOutcome.from_mapping(result)


def run_with_default_fallback(
    primary_attempt: Callable[[], Mapping[str, Any] | AttemptOutcome],
    fallback_attempt: Callable[[], Mapping[str, Any] | AttemptOutcome] | None,
    *,
    primary_is_default: bool = False,
) -> RecoveryOutcome:
    """Run one primary attempt and, only on failure, one default attempt."""

    primary = execute_attempt(primary_attempt)
    if primary.succeeded or primary_is_default:
        return RecoveryOutcome(primary=primary)
    if fallback_attempt is None:
        return RecoveryOutcome(primary=primary)
    return RecoveryOutcome(primary=primary, fallback=execute_attempt(fallback_attempt))
