"""Read-only convergence audit of paper runs; never read past case 1000.

The optional output is a derived diagnostic artifact, not an experiment run.
Binomial bounds below are local-stationarity diagnostics, not anytime-valid
certificates for an adaptive learner or guarantees about future RL behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom


PREFIX = 1000
PROPOSED_RULE = "block100_95_x3"
RULES = {
    "streak50": (50, 0, 1, 1),
    "ratio100_95_x3": (100, 5, 50, 3),
    "ratio200_98_x3": (200, 4, 50, 3),
    "cp100_95_x1": (100, 1, 50, 1),
    "cp100_95_x3": (100, 1, 50, 3),
    "cp200_95_x1": (200, 4, 50, 1),
    "cp200_95_x3": (200, 4, 50, 3),
    "block100_95_x2": (100, 5, 100, 2),
    "block100_95_x3": (100, 5, 100, 3),
    "block100_98_x2": (100, 2, 100, 2),
}


def trigger(failures: np.ndarray, rule: tuple[int, int, int, int]):
    window, allowed, stride, persistence = rule
    sums = np.r_[0, np.cumsum(failures)]
    consecutive = 0
    for end in range(window, len(failures) + 1, stride):
        count = int(sums[end] - sums[end - window])
        consecutive = consecutive + 1 if count <= allowed else 0
        if consecutive >= persistence:
            following = failures[end : end + 200]
            return {
                "ready_after": end,
                "rl_start_case": end + 1,
                "window_failures": count,
                "next200_available": len(following),
                "next200_failures": int(following.sum()),
                "remaining_available": len(failures) - end,
                "remaining_failures": int(failures[end:].sum()),
            }
    return None


def summarize(branches):
    """Descriptive cohorts, not independent-replicate confidence estimates."""

    groups = {}
    for row in branches:
        if row["is_baseline"]:
            continue
        key = f"n{row['grid_n']}:{row['problem']}:tol={row['tolerance']}:cap={row['max_cycles']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, rows in groups.items():
        periods = []
        for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 600), (600, 1000)):
            counts = [(sum(lo < i <= hi for i in r["failure_indices"]), min(r["n"], hi) - lo)
                      for r in rows if r["n"] > lo]
            failures, n = map(sum, zip(*counts))
            periods.append({"start": lo + 1, "end": hi, "n": n, "failures": failures,
                            "convergence_rate": 1 - failures / n})
        gates = [r["rules"][PROPOSED_RULE] for r in rows if r["rules"][PROPOSED_RULE]]
        complete = [g for g in gates if g["next200_available"] == 200]
        times = [g["ready_after"] for g in gates]
        result[key] = {
            "branches": len(rows), "periods": periods,
            "proposed_gate": {
                "triggered_by1000": len(gates),
                "triggered_before1000": sum(t < 1000 for t in times),
                "ready_after_min_median_max": [min(times), float(np.median(times)), max(times)] if times else None,
                "complete_next200_branches": len(complete),
                "next200_n": 200 * len(complete),
                "next200_failures": sum(g["next200_failures"] for g in complete),
                "next200_max_failure_rate": max(g["next200_failures"] / 200 for g in complete) if complete else None,
            },
        }
    return result


def audit(root: Path):
    branches, excluded = [], []
    for config_path in sorted(root.glob("**/paper*/experiment_config.json")):
        config = json.loads(config_path.read_text())
        grid = config["problem"]["grid"]
        if grid not in [[n, n, n] for n in (40, 60, 80)]:
            excluded.append({"path": str(config_path.parent), "reason": f"grid={grid}"})
            continue
        solve = config["solve"]
        for method in config["methods"]:
            path = config_path.parent / "trajectories" / (method["id"] + ".jsonl")
            if not path.exists():
                excluded.append({"path": str(path), "reason": "missing trajectory"})
                continue
            limit = PREFIX
            if method["solve"] not in {"default", "fixed"}:
                limit = min(limit, int(method.get("solve_activation_case", 0)))
            if limit == 0:
                excluded.append({"path": str(path), "reason": "RL active from first case"})
                continue
            with path.open() as stream:
                lines = list(itertools.islice(stream, limit))
            rows = [json.loads(line) for line in lines]
            assert [r["online_index"] for r in rows] == list(range(len(rows))), path
            if not rows:
                excluded.append({"path": str(path), "reason": "empty trajectory"})
                continue
            statuses, stages, cycles, residuals = [], [], [], []
            tol = float(method.get("solve_tolerance") or solve["tolerance"])
            cap = int(solve["max_cycles"])
            residual_disagreements = 0
            fallback_disagreements = 0
            for row in rows:
                outcome = row["outcome"]
                assert not outcome.get("controller_update_committed", False), path
                first = outcome["primary_attempts"][0]
                status = first["status"]
                assert status == outcome["first_primary_status"], (path, row["online_index"])
                statuses.append(status)
                stages.append(first["failure_stage"])
                cycles.append(int(first["cycles"]))
                residual = first.get("residual_norm")
                residuals.append(residual)
                numeric_success = residual is not None and np.isfinite(residual) and residual <= tol
                residual_disagreements += int((status == "success") != numeric_success)
                fallback_disagreements += int((status != "success") != bool(outcome["fallback_used"]))
            failures = np.array([s != "success" for s in statuses], dtype=int)
            numeric_failures = np.array([
                r is None or not np.isfinite(r) or r > tol for r in residuals
            ], dtype=int)
            numeric_gate = trigger(numeric_failures, RULES[PROPOSED_RULE])
            successful_cycles = [c for c, f in zip(cycles, failures) if not f]
            context = method.get("setup_context", "canonical8d" if method["setup"] == "linucb_v5" else method["setup"])
            bins = []
            for start in range(0, len(rows), 100):
                sub = failures[start : start + 100]
                good_cycles = [c for c, f in zip(cycles[start : start + 100], sub) if not f]
                bins.append({
                    "start": start + 1, "end": start + len(sub),
                    "n": len(sub), "failures": int(sub.sum()),
                    "convergence_rate": 1 - float(sub.mean()),
                    "successful_cycle_fraction_mean": float(np.mean(good_cycles) / cap) if good_cycles else None,
                })
            branches.append({
                "run": config_path.parent.name, "path": str(path.resolve()),
                "method": method["id"], "setup_context": context,
                "is_baseline": method["setup"] == "default",
                "planned_solve": method["solve"], "seed_offset": method.get("seed_offset", 0),
                "grid_n": grid[0], "problem": config["problem"]["kind"],
                "smoother_profile": solve.get("smoother_profile", "legacy_implicit"),
                "tolerance": tol, "max_cycles": cap, "n": len(rows),
                "first1000_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
                "input_hash": hashlib.sha256(json.dumps([r["mkw"] for r in rows], sort_keys=True).encode()).hexdigest(),
                "action_outcome_hash": hashlib.sha256(json.dumps([(r.get("arm_index"), s) for r, s in zip(rows, statuses)]).encode()).hexdigest(),
                "status_counts": dict(Counter(statuses)), "failure_stages": dict(Counter(stages)),
                "residual_disagreements": residual_disagreements,
                "residual_based_ready_after": numeric_gate["ready_after"] if numeric_gate else None,
                "fallback_disagreements": fallback_disagreements,
                "failure_indices": (np.flatnonzero(failures) + 1).tolist(),
                "success_cycle_fraction_mean": float(np.mean(successful_cycles) / cap) if successful_cycles else None,
                "bins100": bins,
                "rules": {name: trigger(failures, rule) for name, rule in RULES.items()},
            })
    bounds = {str(w): [float(beta.ppf(.95, k + 1, w - k)) for k in range(7)] for w in (100, 200)}
    # This reference calculation is a screening bound under a persistent bad
    # regime, NOT a certificate that future failure probability is below 5%.
    # If every conditional failure probability is >= .10, a block is
    # stochastically bounded by Binomial(100, .10).  Iterated conditioning
    # bounds three disjoint passing blocks by q**3, even with adaptive actions.
    q = float(binom.cdf(5, 100, .1))
    return {
        "prefix_limit": PREFIX, "rule_definitions": RULES, "proposed_rule": PROPOSED_RULE,
        "cp_upper95": bounds, "branches": branches, "excluded": excluded,
        "cohorts": summarize(branches),
        "screening_reference": {
            "null": "Every pre-activation conditional first-attempt failure probability is at least 0.10.",
            "block_pass_probability_upper": q,
            "false_trigger_upper_by_cases": {str(n): min(1., (n // 100 - 2) * q**3) for n in (1000, 5000)},
            "caveat": "Not a current/future 5% risk certificate. Threshold comparison is retrospective; new joint runs are needed to measure the effect of earlier RL activation.",
        },
        "sources": [
            "https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm",
            "https://arxiv.org/abs/1810.08240",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs": len({r["run"] for r in result["branches"]}), "branches": len(result["branches"]), "excluded": result["excluded"]}, indent=2))
    for r in result["branches"]:
        if r["is_baseline"]:
            continue
        triggers = {k: v["ready_after"] if v else None for k, v in r["rules"].items()}
        print(json.dumps({"run": r["run"], "method": r["method"], "n": r["n"], "failures100": [b["failures"] for b in r["bins100"]], "triggers": triggers, "residual_disagreements": r["residual_disagreements"]}))


if __name__ == "__main__":
    main()
