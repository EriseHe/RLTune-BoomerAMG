"""Validate or run the prespecified six-path activation study sequentially."""

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.joint.solve_control.composable_joint_4k import build_composable_solve_runtime
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_online_common import method_stream_summary, validate_recovery_stream
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from experiments.joint.solve_control.run_joint_experiment import _validate_resolved
from experiments.joint.solve_control.run_paper_final import file_hash, require_disk_space, run_suite

ROOT = REPOSITORY_ROOT
SUITE = Path(__file__).with_name("03_activation") / "suite.json"
OUTPUT = ROOT / "results/paper_final/03_activation"
BOUNDARIES = (750, 1000, 1250, 1500, 2000)
METHODS = ("setup_only", *(f"start_{tau}" for tau in BOUNDARIES))


def load_suite(path=SUITE):
    manifest = json.loads(path.read_text())
    if (manifest["boundaries"] != list(BOUNDARIES) or manifest["cases"] != 5000
            or manifest["grid"] != 80 or len(manifest["runs"]) != 3):
        raise ValueError("Activation suite must retain the prescribed design")
    configs = []
    for entry in manifest["runs"]:
        config_path = path.parent / entry["config"]
        raw = json.loads(config_path.read_text())
        if file_hash(config_path) != entry["config_sha256"]:
            raise ValueError(f"Prescribed configuration changed: {config_path}")
        if raw["name"] != entry["name"] or raw["stream"]["expected_sha256"] != entry["stream_sha256"]:
            raise ValueError("Config and manifest disagree")
        configs.append((entry, config_path, raw))
    return manifest, configs


def validate_suite(path=SUITE):
    manifest, configs = load_suite(path)
    reports = []
    reference = json.loads((Path(__file__).with_name("04_online") / "advection_80_s1.json").read_text())
    for entry, _config_path, raw in configs:
        if raw["solve"] != reference["solve"] or raw["problem"] != reference["problem"]:
            raise ValueError("Activation study changed the agreed PDE or solve settings")
        if {k: v for k, v in raw["setup"].items() if k != "shared_online_prefix"} != reference["setup"]:
            raise ValueError("Activation study changed the agreed setup learner")
        runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
        validation = _validate_resolved(runtime)
        if (not runtime.shared_online_prefix or runtime.online_cases != 5000 or runtime.warmup_cases != 0
                or tuple(m.name for m in runtime.methods) != METHODS):
            raise ValueError("Incorrect activation methods or stream partition")
        if [m.solve_activation_case for m in runtime.methods] != [0, *BOUNDARIES]:
            raise ValueError("Incorrect activation boundaries")
        bundle = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles["start_1000"]
        if (bundle.encoder.feature_dim != 34 or bundle.controller.joint_dim != 306
                or bundle.encoder.encoding_version != "space_aware_v2"
                or bundle.encoder.problem_context_mode != "canonical_no_c_mean"):
            raise ValueError("Unexpected advection model encoding")
        reports.append({"name": entry["name"], "stream_sha256": validation["stream"]["sha256"],
                        "state_dim": 34, "q_dim": 306, "physical_primary_executions": 23500})
    if len({r["stream_sha256"] for r in reports}) != 3:
        raise ValueError("Activation replicates must use distinct input streams")
    return {"suite": manifest["name"], "valid": True, "native_experiments_executed": 0, "runs": reports}


def audit_run(path, expected_config):
    cases = expected_config["stream"]["online_cases"]
    boundaries = {m["id"]: m.get("solve_activation_case", cases) for m in expected_config["methods"]}
    methods = tuple(boundaries)
    source = next((m["id"] for m in expected_config["methods"] if m["solve"] == "default"),
                  max(methods, key=boundaries.get))
    rl_methods = {m["id"] for m in expected_config["methods"] if m["solve"] == "recursive_lstdq_v3"}
    tail = min(1000, cases)
    recorded = json.loads((path / "experiment_config.json").read_text())
    if {k: v for k, v in recorded.items() if k != "output_dir"} != {
        k: v for k, v in expected_config.items() if k != "output_dir"
    }:
        raise ValueError("Recorded activation configuration differs from the prescribed one")
    stream = json.loads((path / "stream_manifest.json").read_text())
    if stream["sha256"] != expected_config["stream"]["expected_sha256"]:
        raise ValueError("Activation input hash differs")
    windows = {f"all_{cases}": {}, f"last_{tail}": {}}
    with (path / "trajectories" / f"{source}.jsonl").open() as handle:
        reference_rows = [json.loads(line) for line in handle]
    if len(reference_rows) != cases:
        raise ValueError("Incomplete shared-prefix source trajectory")
    checkpoint_audits = {}
    unrecovered = 0
    for name in methods:
        with (path / "trajectories" / f"{name}.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != cases or any(row["online_index"] != i for i, row in enumerate(rows)):
            raise ValueError("Incomplete or reordered activation trajectory")
        recovery = validate_recovery_stream(rows, expect_bandit_transaction=True)
        unrecovered += recovery["unrecovered_failures"]
        tau = boundaries[name]
        for i, row in enumerate(rows):
            out = row["outcome"]
            values = [out[k] for k in ("setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime")]
            if not all(math.isfinite(v) and v >= 0 for v in values) or not math.isclose(
                sum(values), out["end_to_end_runtime"], rel_tol=1e-8, abs_tol=1e-10
            ):
                raise ValueError("Activation cost accounting failed")
            if row["mkw"] != reference_rows[i]["mkw"]:
                raise ValueError("Activation branches have unpaired matrix/RHS inputs")
            if i < tau:
                if out.get("cycle_actions") or out.get("controller_update_committed", False):
                    raise ValueError("RL acted before its activation boundary")
                if name != source and any(row[k] != reference_rows[i][k]
                                               for k in ("params", "bandit_timing", "outcome")):
                    raise ValueError("Copied prefix differs from the measured reference")
        if name in rl_methods:
            if not any(row["outcome"].get("cycle_actions") for row in rows[tau:]):
                raise ValueError("Activated RL branch never acted")
            if name != source:
                fork = json.loads((path / f"shared_prefix_{name}.json").read_text())
                if not fork["valid"] or fork["completed_instances"] != tau or fork["source"] != source:
                    raise ValueError("Missing or invalid full-state fork audit")
                if fork["controller_summaries_at_fork"][name]["steps"] != 0:
                    raise ValueError("RL was trained before its prescribed start")
            checkpoint = path / "checkpoints" / f"{name}_final.npz"
            with np.load(checkpoint, allow_pickle=False) as saved:
                # Reuse stage 02's completed-episode inverse/batch audit.
                from types import SimpleNamespace
                from experiments.archive.paper_development.run_02_diagnostics import audit_controller
                checkpoint_audits[name] = audit_controller(SimpleNamespace(
                    joint_dim=len(saved["theta"]), **{key: saved[key] for key in (
                        "a_matrix", "a_inverse", "theta", "b", "inverse_is_valid", "inverse_rebuild_count"
                    )}))
        for window, selected in ((f"all_{cases}", rows), (f"last_{tail}", rows[-tail:])):
            windows[window][name] = method_stream_summary(selected)
    with (path / "trajectories/method_order.jsonl").open() as handle:
        physical = sum(len(json.loads(line)["method_order"]) for line in handle)
    if physical != cases + sum(cases - boundaries[name] for name in methods if name != source):
        raise ValueError("Unexpected nested-prefix execution count")
    comparison = "start_1000" if "start_1000" in methods else source
    for summaries in windows.values():
        standard = summaries[comparison]["totals_sec"]["end_to_end_runtime"]
        for value in summaries.values():
            cost = value["totals_sec"]["end_to_end_runtime"]
            if "setup_only" in summaries:
                baseline = summaries["setup_only"]["totals_sec"]["end_to_end_runtime"]
                value["reduction_vs_setup_only_pct"] = 100 * (1 - cost / baseline)
            value[f"difference_vs_{comparison}_sec"] = cost - standard
    return {"windows": windows, "unrecovered_failures": unrecovered,
            "physical_primary_executions": physical, "prefix_and_cost_audit_valid": True,
            "shared_prefix_source": source, "comparison_reference": comparison,
            "checkpoint_audits": checkpoint_audits}


def write_reports(path, output_root, *, allow_partial=False):
    _manifest, configs = load_suite(path)
    reports = []
    for entry, _config_path, raw in configs:
        if not (output_root / f"{entry['name']}.complete.json").exists():
            if not allow_partial:
                raise ValueError("Activation suite is incomplete")
            continue
        reports.append({"name": entry["name"], "replicate": entry["replicate"],
                        **audit_run(output_root / entry["name"], raw)})
    _write_json(output_root / "summary.json", {"complete": len(reports) == 3, "runs": reports})
    lines = ["# 03 — Activation results", "", f"Completed replicates: {len(reports)}/3.", "",
             "All 5000 logical problems include their prefix cost once; recovery is included.", "",
             "| Replicate | Method | Recorded total (s) | Reduction vs setup-only |", "|---|---|---:|---:|"]
    for report in reports:
        for name in METHODS:
            value = report["windows"]["all_5000"][name]
            lines.append(f"| {report['replicate']} | {name} | {value['totals_sec']['end_to_end_runtime']:.3f} | {value['reduction_vs_setup_only_pct']:.2f}% |")
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    validation = validate_suite()
    print(json.dumps(validation, indent=2), flush=True)
    if args.run:
        # Six paths produce more trajectory data than the three-path main suite.
        require_disk_space(args.output, 12)
        _write_json(args.output / "validation.json", validation)
        run_suite(SUITE, args.output.resolve(), suite_loader=load_suite,
                  run_audit=audit_run, report_writer=write_reports)


if __name__ == "__main__":
    main()
