"""Audit recorded stage-03 branches and compare the two development runs.

No PDE solves or learner updates. Cross-run differences are descriptive: both
the input/learning seeds and the roster changed. Native costs retain the
original timer scope; this script does not relabel them as complete wall time.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.diagnostics.solve_control.analyze_context_activation_mechanism import summarize
from experiments.archive.paper_development.run_03_activation import audit_run
from experiments.joint.solve_control.online_td_experiment_common import _write_json


def summarize_window(rows):
    summary = summarize(rows)
    outs = [r["outcome"] for r in rows]
    summary["unrecovered"] = sum(o["unrecovered_failure"] for o in outs)
    summary["primary_setup_sec"] = sum(o["primary_setup_runtime"] for o in outs)
    summary["fallback_native_sec"] = sum(
        o["fallback_setup_runtime"] + o["fallback_solve_runtime"] for o in outs
    )
    success = [r for r in rows if r["outcome"]["first_primary_status"] == "success"]
    summary["primary_success_cases"] = len(success)
    summary["primary_success_mean_cycles"] = float(np.mean([
        r["outcome"]["primary_cycles"] for r in success
    ]))
    summary["primary_success_mean_cost_sec"] = float(np.mean([
        r["outcome"]["end_to_end_runtime"] for r in success
    ]))
    summary["ranks"] = {}
    for rank in sorted({r["execution_rank"] for r in rows}):
        group = [r for r in rows if r["execution_rank"] == rank]
        times = [t for r in group for t in r["outcome"].get("cycle_times", [])]
        summary["ranks"][rank] = {
            "cases": len(group),
            "mean_cost_sec": float(np.mean([r["outcome"]["end_to_end_runtime"] for r in group])),
            "mean_primary_cycles": float(np.mean([r["outcome"]["primary_cycles"] for r in group])),
            "mean_cycle_ms": float(np.mean(times) * 1000) if times else None,
        }
    pairs = [(a, b) for r in rows for a, b, ea, eb in zip(
        r["outcome"].get("cycle_actions", []), r["outcome"].get("cycle_actions", [])[1:],
        r["outcome"].get("cycle_explored", []), r["outcome"].get("cycle_explored", [])[1:]
    ) if not ea and not eb]
    summary["nongreedy_excluded_adjacent_pairs"] = len(pairs)
    summary["exact_1_3_alternation_fraction"] = (
        sum({a, b} == {1.0, 3.0} for a, b in pairs) / len(pairs) if pairs else None
    )
    return summary


def read_run(path):
    config = json.loads((path / "experiment_config.json").read_text())
    audit = audit_run(path, config)
    rows = {m["id"]: [json.loads(line) for line in
            (path / "trajectories" / (m["id"] + ".jsonl")).open()]
            for m in config["methods"]}
    orders = [json.loads(line) for line in (path / "trajectories/method_order.jsonl").open()]
    rng = np.random.default_rng(config["seeds"]["method_order"])
    methods = list(rows)
    source = audit["shared_prefix_source"]
    boundaries = {m["id"]: m.get("solve_activation_case", 5000) for m in config["methods"]}
    max_cycle_sum_error = 0.0
    for i, order_row in enumerate(orders):
        active = {m for m in methods if m == source or i >= boundaries[m]}
        expected = [methods[int(j)] for j in rng.permutation(len(methods)) if methods[int(j)] in active]
        assert order_row["online_index"] == i and order_row["method_order"] == expected
        for rank, method in enumerate(expected):
            assert rows[method][i]["execution_rank"] == rank
        for method in methods:
            out = rows[method][i]["outcome"]
            if out.get("cycle_times"):
                err = abs(sum(out["cycle_times"]) - out["primary_solve_runtime"])
                max_cycle_sum_error = max(max_cycle_sum_error, err)
                assert err < 1e-7
    assert len(orders) == 5000
    for method, data in rows.items():
        first = next((i + 1 for i, r in enumerate(data) if r["outcome"].get("cycle_actions")), None)
        assert first == (None if method == "setup_only" else boundaries[method] + 1)
    windows = {"all": (0, 5000), "prefix_750": (0, 750), "post_750": (750, 5000),
               "last_1000": (4000, 5000), **{f"block_{lo+1}_{lo+1000}": (lo, lo+1000)
                                           for lo in range(0, 5000, 1000)}}
    report = {
        "path": str(path.resolve()), "seeds": config["seeds"], "audit": audit,
        "order_and_rank_exactly_reproduced": True,
        "max_cycle_sum_error_sec": max_cycle_sum_error,
        "windows": {m: {name: summarize_window(data[lo:hi]) for name, (lo, hi) in windows.items()}
                    for m, data in rows.items()},
    }
    reference = rows["start_750"]
    chi = np.array([max(r["mkw"][a] / ((r["mkw"][n]+1)*r["mkw"][c])
                          for a, n, c in zip(("a1", "a2", "a3"), ("nx", "ny", "nz"), ("k", "c", "a0")))
                    for r in reference])
    union = np.array([any(data[i]["outcome"]["unrecovered_failure"] for data in rows.values())
                      for i in range(5000)])
    intersection = np.array([all(data[i]["outcome"]["unrecovered_failure"] for data in rows.values())
                             for i in range(5000)])
    report["input_tail"] = {
        "chi_quantiles": dict(zip(("median", "p90", "p99", "max"), np.quantile(chi, [.5, .9, .99, 1]).tolist())),
        "chi_gt_2": int(sum(chi > 2)), "chi_gt_2_all_fail": int(sum((chi > 2) & intersection)),
        "chi_le_1": int(sum(chi <= 1)), "chi_le_1_any_fail": int(sum((chi <= 1) & union)),
        "union_unrecovered": int(sum(union)), "intersection_unrecovered": int(sum(intersection)),
        "min_chi_unrecovered": float(min(chi[union])),
    }
    report["within_seed_phases"] = {}
    for method, data in rows.items():
        if method == "start_750":
            continue
        differences = [i+1 for i, (a, b) in enumerate(zip(data, reference)) if a["params"] != b["params"]]
        report["within_seed_phases"][method] = {
            "first_setup_difference_vs_750": differences[0] if differences else None,
            "same_setup_last_1000": sum(a["params"] == b["params"] for a, b in zip(data[-1000:], reference[-1000:])),
            "cost_delta_vs_750_sec": {f"{lo+1}_{hi}": sum(
                data[i]["outcome"]["end_to_end_runtime"] - reference[i]["outcome"]["end_to_end_runtime"]
                for i in range(lo, hi)) for lo, hi in ((0,250),(250,500),(500,750),(750,5000),(4000,5000))},
        }
    manifest = json.loads((path / "provenance/source_manifest.json").read_text())
    report["source"] = manifest
    report["input_signatures"] = [json.dumps(r["mkw"], sort_keys=True) for r in reference]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/paper_final/03_activation"))
    args = parser.parse_args()
    reports = {str(seed): read_run(args.root / f"advection_80_s{seed}") for seed in (1, 2)}
    comparison = {"runs": reports, "native_solves_executed": 0}
    comparison["input_overlap"] = len(set(reports["1"].pop("input_signatures")) & set(reports["2"].pop("input_signatures")))
    a, b = [reports[str(s)]["source"]["source"]["files"] for s in (1, 2)]
    comparison["changed_source_paths_between_runs"] = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    comparison["run2_source_changed_since_launch"] = [k for k, v in b.items()
        if not Path(k).is_file() or hashlib.sha256(Path(k).read_bytes()).hexdigest() != v]
    comparison["start_750_differences"] = {}
    for window in reports["1"]["windows"]["start_750"]:
        first, second = [reports[str(s)]["windows"]["start_750"][window] for s in (1, 2)]
        delta = {k: second["totals_sec"][k] - v for k, v in first["totals_sec"].items()}
        item = {"cost_component_delta_sec": delta,
                "cost_increase_pct": 100 * delta["end_to_end_runtime"] / first["totals_sec"]["end_to_end_runtime"],
                "primary_setup_delta_sec": second["primary_setup_sec"] - first["primary_setup_sec"],
                "fallback_native_delta_sec": second["fallback_native_sec"] - first["fallback_native_sec"]}
        # Exact accounting identity, not a causal/counterfactual decomposition.
        if first["controlled_cycles"] and second["controlled_cycles"]:
            n1, n2 = first["controlled_cycles"], second["controlled_cycles"]
            t1, t2 = first["cycle_ms_mean"]/1000, second["cycle_ms_mean"]/1000
            item["controlled_solve_identity"] = {
                "cycles_s1": n1, "cycles_s2": n2,
                "cycle_ms_s1": t1*1000, "cycle_ms_s2": t2*1000,
                "count_component_sec": (n2-n1)*t1,
                "mean_time_component_sec": n2*(t2-t1),
                "controlled_time_delta_sec": n2*t2-n1*t1,
            }
        comparison["start_750_differences"][window] = item
    output = args.root / "advection_80_s2/analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "comparison.json", comparison)
    print(output / "comparison.json")


if __name__ == "__main__":
    main()
