"""Execution and verification helpers for the official studies."""
from __future__ import annotations
import math

def audit(records, variant, *, feedback_mode="attempt_2T"):
    if feedback_mode not in ("attempt_2T", "adjacent_delayed", "suffix_delayed"):
        raise ValueError("Unknown setup feedback mode")
    for index, record in enumerate(records):
        if record["problem_index"] != index + 1:
            raise ValueError("Incomplete or reordered problem stream")
        out = record["outcome"]
        attempts = out["primary_attempts"]
        if not 1 <= len(attempts) <= variant.attempts:
            raise ValueError("Learned attempt budget violated")
        if any(row["status"] == "success" for row in attempts[:-1]):
            raise ValueError("Retried after success")
        if not variant.unified and any(row["status"] != "setup_failure" for row in attempts[:-1]):
            raise ValueError("Original policy retried a solve failure")
        if variant.unified and out["fallback_used"] and len(attempts) != variant.attempts:
            raise ValueError("Fallback preceded exhaustion of the shared budget")
        if out["fallback_used"] and attempts[-1]["status"] == "success":
            raise ValueError("Fallback followed learned success")
        if out["fallback_used"] and not variant.default_fallback:
            raise ValueError("Default fallback exceeds the total attempt budget")
        if variant.default_fallback and attempts[-1]["status"] != "success" and not out["fallback_used"]:
            raise ValueError("Missing required default fallback after learned failure")
        committed = not out["failed"]
        if not out["bandit_update_committed"] or out["bandit_observation_count"] != len(attempts):
            raise ValueError("Every valid learned attempt must update the setup model")
        if out["failure_feedback_mode"] != feedback_mode:
            raise ValueError("Setup feedback differs from the requested protocol")
        for i, attempt in enumerate(attempts):
            expected_target = attempt["attempt_cost"]
            if attempt["status"] != "success":
                if feedback_mode == "attempt_2T":
                    expected_target *= 2
                elif feedback_mode == "suffix_delayed":
                    expected_target += sum(a["attempt_cost"] for a in attempts[i+1:])
                    expected_target += sum(out[f"fallback_{part}_runtime"] for part in ("setup", "solve", "controller"))
                elif i+1 < len(attempts):
                    expected_target += attempts[i+1]["attempt_cost"]
                else:
                    expected_target += sum(out[f"fallback_{part}_runtime"] for part in ("setup", "solve", "controller"))
            if attempt["attempt_cost"] < 0 or not math.isfinite(expected_target):
                raise ValueError("Invalid setup target")
            if not math.isclose(attempt["learning_cost"], expected_target, rel_tol=1e-12):
                raise ValueError("Setup target does not match the requested attempt-cost rule")
        if not committed and out["controller_update_committed"]:
            raise ValueError("Failed problem retained RL updates")
        if variant.unified and len({a["arm_index"] for a in attempts}) != len(attempts):
            raise ValueError("Repeated exact setup within a problem")
        native = sum(a["setup_runtime"] + a["solve_runtime"] for a in attempts)
        native += out["fallback_setup_runtime"] + out["fallback_solve_runtime"]
        controller = sum(a["controller_runtime"] for a in attempts)
        controller += out["fallback_controller_runtime"] + out.get("case_learning_runtime", 0.0)
        expected = native + controller + out["bandit_overhead_runtime"]
        for actual, wanted in ((out["runtime"], native), (out["infer_runtime"], controller),
                               (out["end_to_end_runtime"], expected)):
            if abs(actual - wanted) > 1e-8 * max(1, abs(wanted)):
                raise ValueError("Attempt/recovery timing accounting does not close")
        if variant.unified and committed:
            suffix = out["fallback_setup_runtime"] + out["fallback_solve_runtime"]
            for attempt in reversed(attempts):
                expected_cost = suffix if (variant.rl_recovery_cost
                    and attempt["status"] != "success"
                    and attempt["rl_episode_recorded"]) else 0.0
                if abs(attempt["rl_recovery_cost"] - expected_cost) > 1e-9:
                    raise ValueError("RL recovery target differs from executed subsequent native work")
                suffix += attempt["setup_runtime"] + attempt["solve_runtime"]
