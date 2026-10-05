"""Verify the Run 05 Module 05 archive using only Python's standard library."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ("reference", "fixed", "oracle", "periodic", "rl")
FIELDS = {
    "native": "native_continuation_sec",
    "inclusive": "inclusive_continuation_sec",
    "native_total": "native_total_sec",
    "inclusive_total": "inclusive_total_sec",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11):
        raise AssertionError(f"Numerical mismatch: {a} != {b}")


def verify(root, *, check_manifest=True):
    root = Path(root)
    if check_manifest:
        expected = {}
        for line in (root / "MANIFEST.sha256").read_text().splitlines():
            sha, name = line.split("  ", 1)
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            expected[name] = sha
        actual = {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "MANIFEST.sha256"
        }
        assert actual == set(expected), "Missing or extra archive files"
        for name, sha in expected.items():
            assert digest(root / name) == sha, f"Hash mismatch: {name}"

    summary = read(root / "data/summary.json")
    protocol = read(root / "data/protocol.json")
    assert summary["seeds"] == list(range(1, 7))
    assert tuple(summary["methods"]) == METHODS
    assert "comparison_costs" not in summary and "run03_comparison" not in summary
    assert protocol["run_number"] == 5
    assert {
        k: v for k, v in protocol["policies"].items() if v["kind"] == "periodic"
    } == {"periodic_2.85_1.10": {"kind": "periodic", "pattern": [2.85, 1.10]}}
    assert protocol["sources"] == ["bandit_lstdq"]
    assert protocol["test_cases"] == 100 and protocol["repetitions_per_case"] == 3
    assert len(read(root / "data/inputs.json")["test"]) == 100
    gains = {m: [] for m in METHODS}
    scan_counts = {}
    n_eval = n_fail = n_recovered = 0
    for si, seed in enumerate(summary["seeds"]):
        directory = root / f"data/seeds/seed_{seed}"
        jobs = read(directory / "jobs.json")
        jobmap = {(j["case_id"], j["repeat"]): j for j in jobs}
        assert set(jobmap) == {(c, r) for c in range(100) for r in range(3)}
        expected = {
            (j["case_id"], p, j["repeat"]) for j in jobs for p in j["policy_order"]
        }
        rows = {}
        for row in records(directory / "evaluation.jsonl"):
            key = row["case_id"], row["policy"], row["repeat"]
            assert key not in rows
            assert row["seed"] == seed and row["source"] == "bandit_lstdq"
            assert row["phase"] == "test" and row["run_number"] == 5
            job = jobmap[row["case_id"], row["repeat"]]
            assert row["input_id"] == job["input_id"]
            assert row["hierarchy_id"] == job["hierarchy_id"]
            spec = protocol["policies"][row["policy"]]
            assert job["method_policies"]["periodic"] == "periodic_2.85_1.10"
            assert set(job["method_policies"]) == set(METHODS)
            assert set(job["policy_order"]) == set(job["method_policies"].values())
            assert not {
                "periodic_2.50_1.00",
                "periodic_2.60_1.00",
                "periodic_1.00_3.00",
            }.intersection(job["policy_order"])
            out = row["outcome"]
            if spec["kind"] in ("periodic", "fixed"):
                actions = out.get("cycle_actions", [])
                pattern = spec["pattern"]
                assert actions == [
                    pattern[i % len(pattern)] for i in range(len(actions))
                ]

            assert not out.get("controller_update_committed", False)
            close(
                row["native_continuation_sec"],
                out["solve_runtime"] + out.get("fallback_setup_runtime", 0),
            )
            close(
                row["inclusive_continuation_sec"],
                row["native_continuation_sec"] + out.get("infer_runtime", 0),
            )
            rows[key] = row
            n_eval += 1
            n_fail += not row["success"]
            n_recovered += bool(out.get("fallback_used", False))
        assert set(rows) == expected
        totals = {}
        for method in METHODS:
            costs = []
            for case in range(100):
                policy = jobmap[case, 0]["method_policies"][method]
                assert all(
                    jobmap[case, rep]["method_policies"][method] == policy
                    for rep in range(3)
                )
                rr = [rows[case, policy, rep] for rep in range(3)]
                for field, key in FIELDS.items():
                    mean = statistics.mean(r[key] for r in rr)
                    close(mean, summary["costs"][method][field][si][case])
                costs.append(statistics.mean(r["native_continuation_sec"] for r in rr))
            totals[method] = sum(costs)
        for method in METHODS:
            gains[method].append(100 * (1 - totals[method] / totals["reference"]))

        scan = defaultdict(list)
        seen = set()
        for row in records(directory / "fixed_weight_scan.jsonl"):
            key = row["case_id"], row["policy"], row["repeat"]
            assert key not in seen
            seen.add(key)
            assert row["seed"] == seed and row["source"] == "bandit_lstdq"
            assert row["kind"] == "fixed"
            job = jobmap[row["case_id"], 0]
            assert (
                row["input_id"] == job["input_id"]
                and row["hierarchy_id"] == job["hierarchy_id"]
            )
            scan[row["case_id"], row["policy"]].append(row)
        weights = [f"fixed_{1 + i * 0.05:.2f}" for i in range(41)]
        assert set(scan) == {(c, p) for c in range(100) for p in weights}
        means = {}
        for (case, policy), rr in scan.items():
            expected_repeats = (
                {0, 1, 2} if case in protocol["scan_extra_repeat_case_ids"] else {0}
            )
            assert {r["repeat"] for r in rr} == expected_repeats
            means[case, policy] = (
                statistics.mean(r["native_continuation_sec"] for r in rr),
                all(r["success"] for r in rr),
            )
        valid = [p for p in weights if all(means[c, p][1] for c in range(100))]
        fixed = min(valid, key=lambda p: (sum(means[c, p][0] for c in range(100)), p))
        for case in range(100):
            oracle = min(
                (p for p in weights if means[case, p][1]),
                key=lambda p: (means[case, p][0], p),
            )
            assert jobmap[case, 0]["method_policies"]["fixed"] == fixed
            assert jobmap[case, 0]["method_policies"]["oracle"] == oracle
        scan_counts[str(seed)] = len(seen)
        assert len(seen) == 4920
    assert n_eval == summary["audit"]["trials"] == 8460
    assert n_fail == summary["audit"]["failures"]
    assert n_recovered == summary["audit"]["recovered_trials"]
    for method in METHODS:
        aggregate = next(a for a in summary["aggregate"] if a["method"] == method)
        close(
            statistics.mean(gains[method]),
            aggregate["native_reduction_vs_w1_pct"]["mean"],
        )
        close(
            statistics.stdev(gains[method]),
            aggregate["native_reduction_vs_w1_pct"]["sd"],
        )
    for comparison in summary["rl_comparisons"]:
        comp, field = comparison["comparator"], comparison["cost"]
        values = [
            100
            * (
                1
                - sum(summary["costs"]["rl"][field][i])
                / sum(summary["costs"][comp][field][i])
            )
            for i in range(6)
        ]
        for actual, recorded in zip(values, comparison["values"]):
            close(actual, recorded)
    math_audit = read(root / "theory/period_two_minimax_verification_20260928.json")
    assert math_audit["passed"] and math_audit["unordered_pairs_checked"] == 861
    assert math_audit["grid_minimizers_ordered"] == [["2.85", "1.10"], ["1.10", "2.85"]]
    selection = read(root / "figures/best_seed_selection.json")
    assert selection["seed"] == 1 + max(range(6), key=lambda i: gains["rl"][i])
    close(selection["reduction_pct"], max(gains["rl"]))
    for name, sha in read(root / "provenance/checkpoint_hashes.json").items():
        assert digest(root / name) == sha
    return {
        "status": "passed",
        "seeds": 6,
        "cases_per_seed": 100,
        "repetitions_per_case": 3,
        "final_evaluations": n_eval,
        "fixed_scan_evaluations": sum(scan_counts.values()),
        "fixed_scan_by_seed": scan_counts,
        "unrecovered_failures": n_fail,
        "recovered_evaluations": n_recovered,
        "all_fixed_choices_reconstructed": True,
        "all_current_cost_arrays_recomputed": True,
        "manifest_verified": check_manifest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2))
