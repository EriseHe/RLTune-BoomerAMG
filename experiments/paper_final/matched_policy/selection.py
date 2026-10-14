"""Success-constrained fixed comparators from a complete, repeated scan."""

from collections import defaultdict
import math
import statistics


def trial_key(row):
    return row["seed"], row["source"], row["case_id"], row["repeat"], row["policy"]


def verify_coverage(rows, jobs):
    expected = {trial_key({**j, "policy": p}): j for j in jobs for p in j["policy_order"]}
    if len(expected) != sum(len(j["policy_order"]) for j in jobs):
        raise ValueError("Duplicate planned trial")
    seen = set()
    for row in rows:
        key = trial_key(row)
        if key in seen or key not in expected:
            raise ValueError("Duplicate or foreign recorded trial")
        seen.add(key)
        for field in ("input_id", "hierarchy_id"):
            if row[field] != expected[key][field]:
                raise ValueError(f"Trial {field} differs from the frozen plan")
        if not isinstance(row["success"], bool) or not math.isfinite(row["native_continuation_sec"]) or row["native_continuation_sec"] < 0:
            raise ValueError("Invalid scan outcome or time")
    if seen != set(expected):
        raise ValueError("Incomplete trial coverage")


def select_fixed(rows, jobs):
    verify_coverage(rows, jobs)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"], row["case_id"], row["policy"]].append(row)
    means = {key: (statistics.mean(r["native_continuation_sec"] for r in values),
                   all(r["success"] for r in values)) for key, values in grouped.items()}
    result = {}
    for seed in sorted({j["seed"] for j in jobs}):
        cases = sorted({j["case_id"] for j in jobs if j["seed"] == seed})
        weights = sorted({p for j in jobs if j["seed"] == seed for p in j["policy_order"]})
        valid = [p for p in weights if all(means[seed, c, p][1] for c in cases)]
        total = lambda p: sum(means[seed, c, p][0] for c in cases)
        per_case = {}
        for case in cases:
            eligible = [p for p in weights if means[seed, case, p][1]]
            per_case[str(case)] = min(eligible, key=lambda p: (means[seed, case, p][0], p)) if eligible else None
        result[str(seed)] = {
            "global": min(valid, key=lambda p: (total(p), p)) if valid else None,
            "per_instance": per_case,
            "uncovered_case_ids": [c for c in cases if per_case[str(c)] is None],
            "global_candidates": [{"policy": p, "native_continuation_sec": total(p),
                                   "failed_cases": sum(not means[seed, c, p][1] for c in cases)}
                                  for p in weights],
        }
    return result
