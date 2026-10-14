"""Execution and verification helpers for the official studies."""
from __future__ import annotations
import math

from pathlib import Path

import time

from experiments.paper_final.common.artifacts import ROOT, file_hash

from experiments.paper_final.online.case_loop import report_online_outcome

from experiments.paper_final.recovery.protocol import RecoveryVariant

from experiments.paper_final.recovery.audit import audit

VARIANT = RecoveryVariant("no_cost", "Three learned + default; no RL recovery cost", 3, True, False, True)


TRACE_FIELDS = (
    "cycle_explored", "cycle_epsilons", "cycle_selected_uncertainties",
    "cycle_selected_q_values", "cycle_selection_scores", "cycle_forced_default_actions",
    "td_errors", "postfit_td_errors",
)


def native_hashes():
    from hypre.bindings import AMG_RUNTIME_LIBRARY
    paths = [Path(AMG_RUNTIME_LIBRARY)] + sorted((ROOT / "hypre/install/lib").glob("libHYPRE.*"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.is_file()}


def execute_default_case(runtime, index):
    from experiments.paper_final.online.native_evaluation import solve_default_baseline_case
    from setup.space import DEFAULT_SETUP_PARAMS
    mkw, context = runtime.stream[index]
    started = time.perf_counter()
    out = solve_default_baseline_case(mkw=dict(mkw), solver_tol=runtime.args.tol,
                                     solver_max_iter=runtime.args.max_cycles)
    out = report_online_outcome(out, bandit_timing={})
    out["method_wall_runtime"] = time.perf_counter() - started
    row = dict(problem_index=index+1, stream_index=index, method="default", mkw=mkw,
               context=context.tolist(), params=dict(DEFAULT_SETUP_PARAMS), outcome=out,
               rl_active=False, shared_prefix=False)
    audit_default_case(row)
    return row


def audit_default_case(row):
    out = row["outcome"]
    if row["rl_active"] or out["bandit_update_committed"] or out["controller_update_committed"]:
        raise ValueError("Default baseline unexpectedly used a learner")
    if out["primary_attempt_count"] != 1 or out["fallback_used"] or len(out["primary_attempts"]) != 1:
        raise ValueError("Default baseline must execute one ordinary default solve")
    attempt = out["primary_attempts"][0]
    if bool(out["failed"]) != (attempt["status"] != "success"):
        raise ValueError("Default baseline failure classification differs")
    for actual, expected in ((out["runtime"], attempt["setup_runtime"]+attempt["solve_runtime"]),
                             (out["end_to_end_runtime"], out["runtime"])):
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Default baseline runtime accounting does not close")
    if out["infer_runtime"] or out["bandit_overhead_runtime"]:
        raise ValueError("Default baseline contains learner runtime")


def audit_case(row, beta, *, variant=VARIANT, feedback_mode="attempt_2T"):
    audit([dict(row, problem_index=1)], variant, feedback_mode=feedback_mode)
    out = row["outcome"]
    if out.get("replayed_transitions", 0) or out.get("target_replay_runtime", 0.0):
        raise ValueError("No-recovery-cost branch performed a target replay")
    for attempt in out["primary_attempts"]:
        if attempt["rl_recovery_cost"] not in (None, 0.0):
            raise ValueError("Recovery expense leaked into an RL target")
        if not row["rl_active"] and (attempt["rl_episode_recorded"] or attempt["cycle_actions"]):
            raise ValueError("RL ran during the setup-only prefix")
        if row["rl_active"]:
            arrays = [attempt[key] for key in (
                "cycle_actions", "cycle_selected_q_values", "cycle_selected_uncertainties",
                "cycle_selection_scores", "cycle_explored", "cycle_forced_default_actions")]
            if len({len(a) for a in arrays}) != 1:
                raise ValueError("Incomplete per-cycle beta diagnostics")
            for q, width, score in zip(*arrays[1:4]):
                if any(v is None or not math.isfinite(v) for v in (q, width, score)) or width < 0:
                    raise ValueError("Invalid per-cycle beta diagnostics")
                if not math.isclose(max(q-beta*width, 0.0), score, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError("Recorded score does not match the branch beta")


def metrics(records):
    outcomes = [r["outcome"] for r in records]
    attempts = [a for o in outcomes for a in o["primary_attempts"]]
    scores = [s for a in attempts for s in a.get("cycle_selection_scores", []) if s is not None]
    actions = [w for a in attempts for w in a.get("cycle_actions", [])]
    return dict(cases=len(records), failures=sum(o["failed"] for o in outcomes),
        first_failures=sum(o["first_primary_status"] != "success" for o in outcomes),
        fallbacks=sum(o["fallback_used"] for o in outcomes),
        native_sec=sum(o["runtime"] for o in outcomes), total_sec=sum(o["end_to_end_runtime"] for o in outcomes),
        native_fallback_sec=sum(o["fallback_setup_runtime"]+o["fallback_solve_runtime"] for o in outcomes),
        cycles=sum(a["cycles"] for a in attempts)+sum(o["fallback_cycles"] for o in outcomes),
        rl_actions=len(actions), zero_selected_score_fraction=sum(s <= 0 for s in scores)/len(scores) if scores else None,
        random_actions=sum(v for a in attempts for v in a.get("cycle_explored", [])))
