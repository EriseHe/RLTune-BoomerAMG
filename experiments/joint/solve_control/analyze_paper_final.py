"""Audit PAPER_FINAL logs and report Modules 1 and 2; never run PDE solves."""
from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import math
from pathlib import Path
import statistics

from joint_online_common import method_stream_summary, validate_recovery_stream
from online_td_experiment_common import _write_json
from run_paper_final import DEFAULT_OUTPUT, METHOD_LABELS, SUITE, load_suite


COMPONENTS = ("setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime")
WINDOWS = {"all_5000": (0, 5000), "last_1000": (4000, 5000)}


def audit_run(path: Path, expected_config: dict, *, require_completed_residuals: bool = False) -> dict:
    config = json.loads((path / "experiment_config.json").read_text())
    if {k: v for k, v in config.items() if k != "output_dir"} != {
        k: v for k, v in expected_config.items() if k != "output_dir"
    }:
        raise ValueError(f"Recorded experiment config differs from the prescribed config: {path}")
    result = json.loads((path / "result.json").read_text())
    feedback = expected_config.get("failure_feedback", {})
    feedback_mode = feedback.get("mode", "rollback_unrecovered")
    if feedback_mode == "budgeted_penalty":
        recorded = result["protocol"].get("failure_feedback", {})
        if recorded.get("mode") != feedback_mode or recorded.get("penalty_sec") != feedback["penalty_sec"]:
            raise ValueError("Failure-feedback objective differs from the prescribed config")
    stream = json.loads((path / "stream_manifest.json").read_text())
    progress = json.loads((path / "progress.json").read_text())
    if (stream["sha256"] != config["stream"]["expected_sha256"]
            or result["protocol"]["stream"]["sha256"] != stream["sha256"]
            or progress["completed_online_instances"] != 5000):
        raise ValueError(f"Incomplete or mismatched input stream: {path}")
    windows = {name: {} for name in WINDOWS}
    inputs = None
    unrecovered = 0
    for method in METHOD_LABELS:
        with (path / "trajectories" / f"{method}.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 5000 or any(r["online_index"] != i for i, r in enumerate(rows)):
            raise ValueError(f"Missing, reordered, or duplicated problems: {path}/{method}")
        matrix_inputs = [r["mkw"] for r in rows]
        if inputs is not None and inputs != matrix_inputs:
            raise ValueError(f"Methods do not share exactly the same matrix/RHS inputs: {path}")
        inputs = matrix_inputs
        recovery = validate_recovery_stream(rows, expect_bandit_transaction=method != "default_setup_default_solve")
        unrecovered += recovery["unrecovered_failures"]
        for i, row in enumerate(rows):
            outcome = row["outcome"]
            if feedback_mode == "budgeted_penalty":
                penalty = float(feedback["penalty_sec"]) if outcome.get("unrecovered_failure", False) else 0.0
                if outcome.get("failure_feedback_mode") != feedback_mode or outcome.get("failure_penalty_sec") != penalty:
                    raise ValueError("Per-problem failure penalty differs from the prescribed objective")
                if method != "default_setup_default_solve" and row["arm_index"] != outcome.get("selected_arm_index"):
                    raise ValueError("Logged arm does not match the immutable attempted arm")
            if require_completed_residuals or "completed_status" in outcome:
                required = ("completed_residual_norm", "completed_cycles", "completed_status")
                if any(key not in outcome for key in required):
                    raise ValueError(f"Missing completed residual/status: {path}/{method}, problem {i + 1}")
                prefix = "fallback" if outcome.get("fallback_used", False) else "primary"
                residual_value = outcome["completed_residual_norm"]
                expected_value = outcome[f"{prefix}_residual_norm"]
                residual = float("nan") if residual_value is None else float(residual_value)
                expected_residual = float("nan") if expected_value is None else float(expected_value)
                if not (residual == expected_residual or (
                    math.isnan(residual) and math.isnan(expected_residual)
                )):
                    raise ValueError("Completed residual does not match the final attempt")
                if (outcome["completed_status"] != outcome[f"{prefix}_status"]
                        or outcome["completed_cycles"] != outcome[f"{prefix}_cycles"]):
                    raise ValueError("Completed status/cycles do not match the final attempt")
                succeeded = outcome["completed_status"] == "success"
                if succeeded == bool(outcome.get("unrecovered_failure", False)):
                    raise ValueError("Completed status contradicts unrecovered failure flag")
                if succeeded and not (
                    math.isfinite(residual) and 0.0 <= residual < float(expected_config["solve"]["tolerance"])
                    and 0 <= outcome["completed_cycles"] < int(expected_config["solve"]["max_cycles"])
                ):
                    raise ValueError("Successful final attempt violates residual or cycle limit")
            values = [float(outcome[k]) for k in (*COMPONENTS, "end_to_end_runtime")]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Invalid time: {path}/{method}, problem {i + 1}")
            if not math.isclose(sum(values[:-1]), values[-1], rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f"Runtime components do not sum to online E2E: {path}/{method}")
            if result["protocol"].get("timing", {}).get("schema_version", 1) >= 2:
                wall = float(outcome.get("method_wall_runtime", float("nan")))
                if not math.isfinite(wall) or wall < 0.0:
                    raise ValueError(f"Invalid method-call wall time: {path}/{method}, problem {i + 1}")
                if "lifecycle_runtime" in outcome:
                    phases = [float(outcome.get(key, float("nan"))) for key in (
                        "feature_runtime", "decision_runtime", "update_runtime", "lifecycle_runtime"
                    )]
                    if not all(math.isfinite(v) and v >= 0 for v in phases) or not math.isclose(
                        sum(phases), float(outcome["infer_runtime"]), rel_tol=1e-8, abs_tol=1e-10
                    ):
                        raise ValueError(f"Controller timing phases do not sum: {path}/{method}, problem {i + 1}")
            if (method != "bandit_lstdq" or i < 1000) and (
                outcome.get("cycle_actions") or outcome.get("controller_update_committed", False)
            ):
                raise ValueError(f"Solve control used outside the prescribed schedule: {path}/{method}")
        if method == "bandit_lstdq" and not any(r["outcome"].get("cycle_actions") for r in rows[1000:]):
            raise ValueError(f"Solve controller never acted after activation: {path}")
        for name, (start, stop) in WINDOWS.items():
            subset = rows[start:stop]
            summary = method_stream_summary(subset)
            reported = result["windows"][name]["methods"][method]
            for key, value in summary["totals_sec"].items():
                if not math.isclose(value, reported["totals_sec"][key], rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Summary/trajectory mismatch: {path}/{method}/{key}")
            summary["fallback_native_sec"] = sum(
                float(r["outcome"].get("fallback_setup_runtime", 0))
                + float(r["outcome"].get("fallback_solve_runtime", 0)) for r in subset)
            summary["failed_first_attempt_problem_e2e_sec"] = sum(
                r["outcome"]["end_to_end_runtime"] for r in subset
                if r["outcome"]["first_primary_status"] != "success")
            windows[name][method] = summary
    for methods in windows.values():
        default = methods["default_setup_default_solve"]["totals_sec"]["end_to_end_runtime"]
        bandit = methods["bandit_default"]["totals_sec"]["end_to_end_runtime"]
        if default <= 0 or bandit <= 0:
            raise ValueError("Nonpositive reference time")
        for summary in methods.values():
            cost = summary["totals_sec"]["end_to_end_runtime"]
            summary["time_reduction_vs_default_pct"] = 100 * (1 - cost / default)
            summary["time_reduction_vs_linucb_pct"] = 100 * (1 - cost / bandit)
            summary["speedup_vs_default"] = default / cost if cost > 0 else None
    return {"path": str(path.resolve()), "stream_sha256": stream["sha256"],
            "windows": windows, "unrecovered_failures": unrecovered,
            "input_pairing_verified": True, "cost_accounting_verified": True}


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups = []
    for family in ("diffusion", "diffusion_advection"):
        for grid in (40, 60, 80):
            matching = [r for r in runs if r["family"] == family and r["grid"] == grid]
            for cap in sorted({r.get("cap", 0) for r in matching}):
                selected = [r for r in matching if r.get("cap", 0) == cap]
                for window in WINDOWS:
                    for method in METHOD_LABELS:
                        metrics = [r["windows"][window][method] for r in selected]
                        reductions = [m["time_reduction_vs_default_pct"] for m in metrics]
                        groups.append({
                            "family": family, "grid": grid, "cap": cap or None,
                            "window": window, "method": method,
                            "replicates": len(selected), "seeds": [r["seed"] for r in selected],
                            "mean_time_reduction_pct": statistics.mean(reductions),
                            "sample_sd_time_reduction_pct": statistics.stdev(reductions) if len(reductions) > 1 else None,
                            "mean_e2e_sec": statistics.mean(m["totals_sec"]["end_to_end_runtime"] for m in metrics),
                        })
    return groups


def write_reports(suite: Path, output_root: Path, *, allow_partial: bool = False) -> dict:
    _manifest, configs = load_suite(suite)
    runs, missing = [], []
    for entry, _path, raw in configs:
        if not (output_root / f"{entry['name']}.complete.json").exists():
            missing.append(entry["name"])
            continue
        runs.append({**entry, "cap": int(raw["solve"]["max_cycles"]),
                     **audit_run(output_root / entry["name"], raw, require_completed_residuals=True)})
    if missing and not allow_partial:
        raise ValueError(f"PAPER_FINAL incomplete: {len(missing)}/{len(configs)} runs missing")
    summary = {"complete": not missing, "completed_runs": len(runs), "missing": missing,
               "runs": runs, "aggregate": aggregate_runs(runs)}
    report_dir = output_root / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "modules_1_2.json", summary)
    lines = ["# Module 04 — cycle-cap comparison" if _manifest.get("caps") else "# PAPER_FINAL — Modules 1 and 2", "",
             f"Completed: {len(runs)}/{len(configs)} runs. Status: {'complete' if not missing else 'PARTIAL — not final paper results'}.", "",
             "Primary: summed online E2E seconds across all 5000 problems, including the first 1000, "
             "bounded recovery and recurring controller/bandit overhead. One-time model initialization, "
             "AOT schedule generation, matrix/RHS assembly, logging, and plotting are outside this metric. "
             "Subprocess elapsed time is stored separately and covers all three methods together.", "",
             "When a group has unrecovered failures, its cost is a budgeted attempted-solve cost, "
             "not the time to successfully solve every problem. Interpret time reductions together "
             "with the unrecovered counts; all failed attempts and recovery costs are retained.", "",
             "Time reduction = 100(1 − method/default); speedup factor = default/method. "
             "Each reduction uses the paired Default from the same run. Replicate SD describes variability "
             "across the completed training streams; it is not a confidence interval from 5000 independent observations.", "",
             "## Module 1: every prespecified replicate", "",
             "| PDE | Grid | Cap | Seed | Window | Method | Online E2E (s) | Reduction vs Default (%) | Reduction vs LinUCB (%) | Unrecovered |",
             "|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for run in runs:
        for window, methods in run["windows"].items():
            for method, label in METHOD_LABELS.items():
                m = methods[method]
                lines.append(f"| {run['family']} | {run['grid']}³ | {run['cap']} | {run['seed']} | {window} | {label} | "
                             f"{m['totals_sec']['end_to_end_runtime']:.3f} | {m['time_reduction_vs_default_pct']:.2f} | "
                             f"{m['time_reduction_vs_linucb_pct']:.2f} | {m['unrecovered_failure_count']} |")
    lines += ["", "## Module 1: replicate summary", "",
              "Mean ± sample SD of the per-replicate time reductions; all prescribed seeds are retained.", "",
              "Caps are distinct experimental settings and are never pooled as independent replicates.", "",
              "| PDE | Grid | Cap | Window | Method | Replicates | Reduction mean ± SD (percentage points) |",
              "|---|---:|---:|---|---|---:|---:|"]
    for row in summary["aggregate"]:
        sd = row["sample_sd_time_reduction_pct"]
        sd_text = "NA" if sd is None else f"{sd:.2f}"
        lines.append(f"| {row['family']} | {row['grid']}³ | {row['cap']} | {row['window']} | {METHOD_LABELS[row['method']]} | "
                     f"{row['replicates']} | {row['mean_time_reduction_pct']:.2f} ± {sd_text} |")
    lines += ["", "## Module 2: 60³ runtime components", "",
              "Components are totals in seconds. Fallback native time is a subset of setup + solve; "
              "it must not be added again. Unrecovered cases remain visible in all summaries.", "",
              "| PDE | Cap | Seed | Window | Method | Setup | Solve | Controller | Bandit | Online E2E | First-attempt failures | Fallback native |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        if run["grid"] != 60:
            continue
        for window, methods in run["windows"].items():
            for method, label in METHOD_LABELS.items():
                m = methods[method]
                t = m["totals_sec"]
                values = " | ".join(f"{t[k]:.3f}" for k in (
                    "setup_runtime", "native_solve_runtime", "controller_runtime", "setup_bandit_overhead", "end_to_end_runtime"))
                lines.append(f"| {run['family']} | {run['cap']} | {run['seed']} | {window} | {label} | {values} | "
                             f"{m['primary_failure_count']} | {m['fallback_native_sec']:.3f} |")
    if any("method_wall_runtime" in m["totals_sec"]
           for r in runs for methods in r["windows"].values() for m in methods.values()):
        lines += ["", "## Additional timing audit", "",
                  "Controller lifecycle is already included in Controller/Online E2E, not an extra charge. "
                  "Method-call wall time is an independent stopwatch including matrix construction and wrapper work; "
                  "it excludes outer trajectory I/O, checkpoints and plotting, and is not fed to either learner.", "",
                  "| PDE | Grid | Cap | Seed | Window | Method | Controller lifecycle (s) | Method-call wall (s) |",
                  "|---|---:|---:|---:|---|---|---:|---:|"]
        for run in runs:
            for window, methods in run["windows"].items():
                for method, m in methods.items():
                    t = m["totals_sec"]
                    if "method_wall_runtime" in t:
                        lines.append(f"| {run['family']} | {run['grid']}³ | {run['cap']} | {run['seed']} | {window} | "
                                     f"{METHOD_LABELS[method]} | {t.get('controller_lifecycle_runtime', 0.0):.6f} | "
                                     f"{t['method_wall_runtime']:.6f} |")
    if any("penalized_cost" in m["totals_sec"]
           for r in runs for methods in r["windows"].values() for m in methods.values()):
        lines += ["", "## Failure-aware objective", "",
                  "Penalized cost = measured online E2E + the prescribed penalty for final failures. "
                  "The penalty is not elapsed time and is excluded from every runtime reduction above.", "",
                  "| PDE | Grid | Cap | Window | Method | Measured E2E (s) | Failure penalty (s) | Penalized cost (s) |",
                  "|---|---:|---:|---|---|---:|---:|---:|"]
        for run in runs:
            for window, methods in run["windows"].items():
                for method, m in methods.items():
                    t = m["totals_sec"]
                    if "penalized_cost" in t:
                        lines.append(f"| {run['family']} | {run['grid']}³ | {run['cap']} | {window} | {METHOD_LABELS[method]} | "
                                     f"{t['end_to_end_runtime']:.3f} | {t['failure_penalty']:.3f} | {t['penalized_cost']:.3f} |")
    lines += ["", "LinUCB and LinUCB–LSTDQ learn independent setup paths with their own realized costs. "
              "Their difference measures the complete adaptive frameworks, not an isolated causal effect "
              "of changing relaxation on a matched hierarchy. The fixed-weight oracle / matched-hierarchy "
              "comparison belongs to deferred Module 3.", ""]
    (report_dir / "modules_1_2.md").write_text("\n".join(lines))
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    summary = write_reports(args.suite, args.output_root, allow_partial=args.allow_partial)
    print(json.dumps({"complete": summary["complete"], "completed_runs": summary["completed_runs"],
                      "report": str(args.output_root / 'analysis/modules_1_2.md')}))


if __name__ == "__main__":
    main()
