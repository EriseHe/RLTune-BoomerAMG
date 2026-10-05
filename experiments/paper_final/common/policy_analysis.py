"""Matched-policy trial keys, costs and aggregation of prescribed repetitions."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

COST_FIELDS = (
    "native_continuation_sec",
    "inclusive_continuation_sec",
    "native_total_sec",
    "inclusive_total_sec",
    "wall_sec",
)


def cell_key(row):
    return (
        row["seed"],
        row["source"],
        row["case_id"],
        row.get("repeat", 0),
        row["policy"],
    )


def compact_row(row):
    outcome = row["outcome"]
    return {
        **{
            k: row[k]
            for k in (
                "seed",
                "source",
                "case_id",
                "policy",
                "kind",
                "success",
                *COST_FIELDS,
            )
        },
        "repeat": row.get("repeat", 0),
        "recovery": bool(outcome.get("fallback_used", False)),
        "setup_sec": outcome["setup_runtime"],
        "solve_sec": outcome["solve_runtime"],
        "controller_sec": outcome.get("infer_runtime", 0),
        "recovery_setup_sec": outcome.get("fallback_setup_runtime", 0),
        "primary_cycles": outcome.get("primary_cycles", outcome.get("iterations", 0)),
        "fallback_cycles": outcome.get("fallback_cycles", 0),
    }


def totals(rows):
    return {
        "cases": len(rows),
        "failures": sum(not r["success"] for r in rows),
        "recoveries": sum(r["recovery"] for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                *COST_FIELDS,
                "setup_sec",
                "solve_sec",
                "controller_sec",
                "recovery_setup_sec",
                "primary_cycles",
                "fallback_cycles",
            )
        },
    }


def average_repetitions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["source"], row["case_id"], row["policy"])].append(row)
    result = []
    for values in grouped.values():
        first = values[0]
        averaged = {
            **first,
            "repeat": "mean",
            "repetitions": len(values),
            "success": all(v["success"] for v in values),
            "recovery": any(v["recovery"] for v in values),
        }
        for key in (
            *COST_FIELDS,
            "setup_sec",
            "solve_sec",
            "controller_sec",
            "recovery_setup_sec",
            "primary_cycles",
            "fallback_cycles",
        ):
            averaged[key] = sum(v[key] for v in values) / len(values)
        result.append(averaged)
    return result


def fixed_hindsight(rows):
    """min_w sum_i C_i(w), separately from sum_i min_w C_i(w)."""
    by_policy, by_case = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["kind"] == "fixed":
            by_policy[row["policy"]].append(row)
            by_case[row["case_id"]].append(row)
    candidates = [
        {"policy": name, **totals(values)} for name, values in by_policy.items()
    ]
    successful = [r for r in candidates if r["failures"] == 0]
    best = (
        min(successful, key=lambda r: (r["native_continuation_sec"], r["policy"]))
        if successful
        else None
    )
    coverage_choice = min(
        candidates,
        key=lambda r: (r["failures"], r["native_continuation_sec"], r["policy"]),
    )
    individual = []
    uncovered = []
    for case, values in sorted(by_case.items()):
        eligible = [r for r in values if r["success"]]
        if eligible:
            individual.append(
                min(eligible, key=lambda r: (r["native_continuation_sec"], r["policy"]))
            )
        else:
            uncovered.append(case)
    return {
        "best_fixed": best,
        "maximum_coverage_fixed": coverage_choice,
        "per_case": individual,
        "uncovered_case_ids": uncovered,
        "fixed_grid": candidates,
    }


def reduction(reference, value):
    return 100 * (1 - value / reference) if reference > 0 else None


def expected_cells(jobs):
    keys = [cell_key({**j, "policy": name}) for j in jobs for name in j["policy_order"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate planned execution")
    return set(keys)


def statistics(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": a.tolist(),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def collect_cells(rows, expected_repeats):
    groups = defaultdict(list)
    for r in rows:
        groups[r["seed"], r["case_id"], r["policy"]].append(r)
    result = {}
    for key, values in groups.items():
        ids = [r["repeat"] for r in values]
        if len(ids) != len(set(ids)) or (
            expected_repeats is not None and set(ids) != set(expected_repeats)
        ):
            raise ValueError(f"Missing or duplicated timing repetitions: {key}")
        average = average_repetitions([compact_row(r) for r in values])
        if len(average) != 1:
            raise AssertionError("Mixed source in a matched cell")
        result[key] = average[0]
    return result
