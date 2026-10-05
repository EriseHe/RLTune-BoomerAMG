"""Native and frozen-policy evaluations on one resolved setup."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence, Tuple
import numpy as np
from solve.controllers.ppo import SetupAwareSolvePolicyRunner
from hypre.bindings import (
    AMGNativeError,
    SolveStatus,
    augment_setup_params,
    create_env,
    run_with_default_fallback,
    solve,
)
from solve.core.outcomes import classify_rl_failure
from .action_spaces import DEFAULT_SETUP_PARAMS


def _actual_failure_result(
    exc: Exception,
    *,
    started_at: float,
    setup_runtime: float = 0.0,
    solve_runtime: float = 0.0,
    controller_runtime: float = 0.0,
    iterations: int = 0,
    residual_norm: float = float("nan"),
) -> Dict[str, Any]:
    setup_sec = max(0.0, float(setup_runtime))
    solve_sec = max(0.0, float(solve_runtime))
    controller_sec = max(0.0, float(controller_runtime))
    if isinstance(exc, AMGNativeError):
        setup_sec += float(exc.setup_runtime_sec)
        solve_sec += float(exc.solve_runtime_sec)
        stage = "setup" if exc.operation in {"create", "setup"} else "solve"
    else:
        stage = "setup" if setup_sec <= 0.0 else "solve"
        elapsed = max(0.0, float(time.perf_counter() - started_at))
        unaccounted = max(0.0, elapsed - setup_sec - solve_sec - controller_sec)
        if stage == "setup":
            setup_sec += unaccounted
        else:
            controller_sec += unaccounted
    reason = f"exception:{type(exc).__name__}:{exc}"
    return {
        "failure_origin": "solver" if isinstance(exc, AMGNativeError) else "execution",
        "runtime": float(setup_sec + solve_sec),
        "setup_runtime": setup_sec,
        "solve_runtime": solve_sec,
        "infer_runtime": controller_sec,
        "failed": True,
        "failure_reason": reason,
        "failure_stage": stage,
        "residual_norm": float(residual_norm),
        "iterations": int(iterations),
        "structural_fail": stage == "setup",
    }


def solve_fixed_w_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    w: float,
    sweeps_down: int,
    sweeps_up: int,
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    prep = None
    solve_runtime = 0.0
    residual_norm = float("nan")
    iterations = 0
    cycle_residuals: List[float] = []
    cycle_times: List[float] = []
    native_status = SolveStatus.CONTINUE
    try:
        params = augment_setup_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            residual_norm = float(env.r0)
            for cycle in range(int(solve_max_cycles)):
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sweeps_down),
                    sweeps_up=int(sweeps_up),
                    tol=float(solve_tol),
                    max_cycles=int(solve_max_cycles),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                cycle_residuals.append(float(residual_norm))
                cycle_times.append(float(dt))
                native_status = env.last_step.status
                if native_status is not SolveStatus.CONTINUE:
                    break
        failure_reason = (
            ""
            if native_status is SolveStatus.CONVERGED
            else "max_cycles_reached_without_convergence"
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": 0.0,
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": native_status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
            "cycle_actions": [float(w)] * int(iterations),
            "cycle_residuals": cycle_residuals,
            "cycle_times": cycle_times,
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=solve_runtime,
            iterations=iterations,
            residual_norm=residual_norm,
        )
        result.update(
            {
                "final_w": float(w),
                "final_sweeps_down": int(sweeps_down),
                "final_sweeps_up": int(sweeps_up),
                "cycle_actions": [],
                "cycle_residuals": [],
                "cycle_times": [],
            }
        )
        return result


def solve_schedule_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    prep = None
    solve_runtime = 0.0
    decision_runtime = 0.0
    cycle_actions: List[float] = []
    cycle_residuals: List[float] = []
    cycle_times: List[float] = []
    residual_norm = float("nan")
    iterations = 0
    native_status = SolveStatus.CONTINUE
    try:
        params = augment_setup_params(params)
        schedule_sorted = sorted(
            (int(end), float(w), int(sd), int(su)) for end, w, sd, su in schedule
        )
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            residual_norm = float(env.r0)
            last_w = float("nan")
            last_sd = -1
            last_su = -1
            for cycle in range(int(solve_max_cycles)):
                decision_started = time.perf_counter()
                chosen = schedule_sorted[-1]
                for end_cycle, w, sd, su in schedule_sorted:
                    if cycle < int(end_cycle):
                        chosen = (end_cycle, w, sd, su)
                        break
                _end, w, sd, su = chosen
                decision_runtime += time.perf_counter() - decision_started
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sd),
                    sweeps_up=int(su),
                    tol=float(solve_tol),
                    max_cycles=int(solve_max_cycles),
                )
                solve_runtime += float(dt)
                cycle_actions.append(float(w))
                cycle_residuals.append(float(residual_norm))
                cycle_times.append(float(dt))
                iterations = cycle + 1
                last_w = float(w)
                last_sd = int(sd)
                last_su = int(su)
                native_status = env.last_step.status
                if native_status is not SolveStatus.CONTINUE:
                    break
        failure_reason = (
            ""
            if native_status is SolveStatus.CONVERGED
            else "max_cycles_reached_without_convergence"
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime + decision_runtime),
            "native_runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime + decision_runtime),
            "native_solve_runtime": float(solve_runtime),
            "infer_runtime": float(decision_runtime),
            "cycle_actions": cycle_actions,
            "cycle_residuals": cycle_residuals,
            "cycle_times": cycle_times,
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": native_status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=solve_runtime,
            iterations=iterations,
            residual_norm=residual_norm,
        )
        result.update(
            {
                "final_w": float("nan"),
                "final_sweeps_down": -1,
                "final_sweeps_up": -1,
                "cycle_actions": cycle_actions,
                "cycle_residuals": cycle_residuals,
                "cycle_times": cycle_times,
                "infer_runtime": float(decision_runtime),
                "native_solve_runtime": float(result["solve_runtime"]),
            }
        )
        result["runtime"] += decision_runtime
        result["solve_runtime"] += decision_runtime
        return result


def solve_setup_aware_rl_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solve_policy: SetupAwareSolvePolicyRunner,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
    classify_rl_failure: Callable[..., str],
    solve_max_cycles: int,
    case_progress: float = 0.0,
) -> Dict[str, Any]:
    # Bandit and RL are combined sequentially here:
    # 1) setup params have already been chosen upstream by the mature bandit
    # 2) prepare_rl(params=...) builds that setup in Hypre
    # 3) solve_policy.run(...) lets RL control solve-phase weights/sweeps on
    #    top of the fixed setup
    started_at = time.perf_counter()
    prep = None
    rl_out: Dict[str, Any] = {}
    try:
        params = augment_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            rl_out = solve_policy.run(
                env, mkw=mkw, setup_params=params, case_progress=float(case_progress)
            )
        res_norm = float(rl_out["residual_norm"])
        iters = int(rl_out["iterations"])
        total_runtime = float(prep.setup_runtime_sec + rl_out["solve_runtime"])
        failure_reason = classify_rl_failure(residual_norm=res_norm, iterations=iters)
        converged = not bool(failure_reason)
        return {
            "runtime": total_runtime,
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(rl_out["solve_runtime"]),
            "infer_runtime": float(rl_out["infer_runtime"]),
            "failed": (not converged),
            "failure_reason": failure_reason,
            "residual_norm": res_norm,
            "iterations": iters,
            "final_w": float(rl_out["final_w"]),
            "final_sweeps_down": int(rl_out["final_sweeps_down"]),
            "final_sweeps_up": int(rl_out["final_sweeps_up"]),
            "action_counts": dict(rl_out.get("action_counts", {})),
            "cycle_actions": list(rl_out.get("cycle_actions", [])),
            "cycle_residuals": list(rl_out.get("cycle_residuals", [])),
            "cycle_times": list(rl_out.get("cycle_times", [])),
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=float(rl_out.get("solve_runtime", 0.0)),
            controller_runtime=float(rl_out.get("infer_runtime", 0.0)),
            iterations=int(rl_out.get("iterations", 0)),
            residual_norm=float(rl_out.get("residual_norm", float("nan"))),
        )
        result.update(
            {
                "final_w": float("nan"),
                "final_sweeps_down": -1,
                "final_sweeps_up": -1,
                "action_counts": {},
                "cycle_actions": [],
                "cycle_residuals": [],
                "cycle_times": [],
            }
        )
        return result


def classify_no_rl_failure(
    *, residual_norm: float, iterations: int, solver_tol: float, solver_max_iter: int
) -> str:
    reasons: List[str] = []
    if not np.isfinite(float(residual_norm)):
        reasons.append("non_finite_residual_norm")
    elif float(residual_norm) > float(solver_tol):
        reasons.append("residual_above_solver_tol")
    if float(residual_norm) > float(solver_tol) and int(iterations) >= int(
        solver_max_iter
    ):
        reasons.append("max_iter_reached_without_convergence")
    return ";".join(reasons)


def solve_no_rl_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    try:
        res = solve(
            params=augment_params(dict(params)),
            tol=float(solver_tol),
            max_iter=int(solver_max_iter),
            **mkw,
        )
        residual_norm = float(res.residual_norm)
        iterations = int(res.iterations)
        failure_reason = (
            ""
            if res.status is SolveStatus.CONVERGED
            else "max_iter_reached_without_convergence"
        )
        return {
            "runtime": float(res.runtime_sec),
            "setup_runtime": float(res.setup_runtime_sec),
            "solve_runtime": float(res.solve_runtime_sec),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": res.status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "structural_fail": False,
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }
    except Exception as exc:
        result = _actual_failure_result(exc, started_at=started_at)
        result.update(
            {
                "final_w": float("nan"),
                "final_sweeps_down": -1,
                "final_sweeps_up": -1,
            }
        )
        return result


def solve_default_baseline_case(
    *,
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]] = augment_setup_params,
) -> Dict[str, Any]:
    """Run the default baseline once, without retrying the same attempt."""

    recovery = run_with_default_fallback(
        lambda: solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            augment_params=augment_params,
        ),
        None,
        primary_is_default=True,
    )
    result = recovery.to_result()
    result.update(
        {
            "recovery_protocol_applied": True,
            "bandit_update_committed": False,
            "controller_update_committed": False,
        }
    )
    return result
