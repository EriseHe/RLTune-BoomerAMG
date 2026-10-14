"""Prepare or run a configurable sequence of four-branch Module 04 seed batches.

Preparation and verification execute no numerical solves. Only the explicit run
command starts workers. Each seed batch completes every family/grid experiment
before the next batch begins. Completed batches are combined without rerunning.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import shutil
import traceback

from experiments.paper_final.common.artifacts import (
    dump, read, file_hash, digest, read_records, now, write_csv,
)
from experiments.paper_final import run_04_factorial as runner
from experiments.paper_final.online.factorial import METHODS, LABELS
from experiments.runtime import prevent_sleep, stop_sleep_prevention
import numpy as np


def safe_label(value):
    label = str(value)
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError(f"Unsafe label: {label!r}")
    return label


def load_plan(path):
    path = Path(path).resolve()
    plan = read(path)
    if set(plan) != {"name", "max_workers", "seed_batches", "experiments"}:
        raise ValueError("Campaign needs name, max_workers, seed_batches and experiments")
    limit = plan["max_workers"]
    if type(limit) is not int or limit < 1:
        raise ValueError("max_workers must be a positive integer")
    batches = [[safe_label(s) for s in batch] for batch in plan["seed_batches"]]
    seeds = [s for batch in batches for s in batch]
    if not batches or any(not b or len(b) > limit for b in batches) or len(set(seeds)) != len(seeds):
        raise ValueError("Seed batches must be nonempty, disjoint, and within max_workers")
    experiments = []
    for item in plan["experiments"]:
        if set(item) != {"label", "suite"}:
            raise ValueError("Each experiment needs label and suite")
        label = safe_label(item["label"])
        suite_path = (path.parent / item["suite"]).resolve()
        suite = read(suite_path)
        if set(suite) != {"name", "seeds"}:
            raise ValueError("Each experiment must reference a factorial suite")
        configs = {}
        for entry in suite["seeds"]:
            seed = safe_label(entry["label"])
            if seed in configs:
                raise ValueError("Duplicate seed in an experiment")
            config_path = (suite_path.parent / entry["config"]).resolve()
            raw = read(config_path)
            runner.factorial.validate_methods(runner.parse_joint_experiment_config(raw))
            configs[seed] = {"path": str(config_path), "raw": raw}
        if set(configs) != set(seeds):
            raise ValueError("Every experiment must contain exactly the campaign seeds")
        first = configs[seeds[0]]["raw"]
        for config in configs.values():
            raw = config["raw"]
            for key in ("problem", "setup", "solve", "methods"):
                if raw[key] != first[key]:
                    raise ValueError(f"Seed configurations differ in {key} for {label}")
            if raw["stream"]["online_cases"] != first["stream"]["online_cases"]:
                raise ValueError("Seed stream lengths differ within an experiment")
        experiments.append(dict(label=label, name=suite["name"], configs=configs))
    if not experiments or len({e["label"] for e in experiments}) != len(experiments):
        raise ValueError("Experiment labels must be nonempty and unique")
    return plan, batches, experiments


def prepare(plan_path, output, *, cases=None):
    plan, batches, experiments = load_plan(plan_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = runner.source_hashes()
    jobs = []
    for number, seeds in enumerate(batches, 1):
        for experiment in experiments:
            label = experiment["label"]
            relative = f"batches/batch_{number:02d}/{label}"
            suite_path = output / "suites" / f"batch_{number:02d}_{label}.json"
            dump(suite_path, dict(name=f"{experiment['name']} — seeds {', '.join(seeds)}",
                seeds=[dict(label=s, config=experiment["configs"][s]["path"]) for s in seeds]))
            protocol = runner.prepare(suite_path, output / relative, cases=cases)
            if read(output / relative / "source_manifest.json") != frozen:
                raise ValueError("Source changed during campaign preparation")
            jobs.append(dict(batch=number, experiment=label, seeds=seeds, output=relative,
                protocol_sha256=file_hash(output / relative / "protocol.json"),
                inputs=sum(v["cases"] for v in protocol["stages"].values())))
    campaign = dict(format="module04-zero-prefix-campaign-v1", name=plan["name"],
        prepared_at=now(), plan_path=str(Path(plan_path).resolve()),
        plan_sha256=file_hash(plan_path), max_workers=plan["max_workers"],
        seed_batches=batches, experiments=[e["label"] for e in experiments],
        jobs=jobs, methods=list(METHODS), cases_override=cases,
        planned_seed_experiments=sum(len(j["seeds"]) for j in jobs),
        planned_paired_inputs=sum(j["inputs"] for j in jobs),
        planned_method_evaluations=4 * sum(j["inputs"] for j in jobs),
        execution="Batches are sequential. Each batch completes all experiments in plan order; its seed workers run concurrently, with four serial methods per input.")
    dump(output / "campaign.json", campaign)
    dump(output / "source_manifest.json", frozen)
    dump(output / "prepared_identity.json", dict(campaign_sha256=file_hash(output / "campaign.json")))
    dump(output / "status.json", dict(status="prepared", native_solves_executed=0, completed_jobs=0))
    verify(output)
    return campaign


def verify(output, *, require_unstarted=True):
    output = Path(output).resolve()
    campaign = read(output / "campaign.json")
    if file_hash(output / "campaign.json") != read(output / "prepared_identity.json")["campaign_sha256"]:
        raise ValueError("Prepared campaign schedule changed")
    if runner.source_hashes() != read(output / "source_manifest.json"):
        raise ValueError("Source changed since preparation")
    if require_unstarted and read(output / "status.json")["status"] != "prepared":
        raise ValueError("Campaign is not unstarted")
    verified = []
    for job in campaign["jobs"]:
        root = output / job["output"]
        if file_hash(root / "protocol.json") != job["protocol_sha256"]:
            raise ValueError("Prepared job protocol changed")
        runner.check_frozen(root)
        protocol = read(root / "protocol.json")
        if protocol["seeds"] != job["seeds"] or len(job["seeds"]) > campaign["max_workers"]:
            raise ValueError("Job seed coverage or worker limit changed")
        if protocol["shared_prefix"] or protocol["copied_measurements"]:
            raise ValueError("Campaign cannot contain shared or copied measurements")
        if require_unstarted and ((root / "release.json").exists() or read(root / "status.json")["status"] != "prepared"):
            raise ValueError("A job already started")
        for seed in job["seeds"]:
            stage = root / f"seed_{seed}"
            checks = protocol["stages"][seed]
            config = read(stage / "experiment_config.json")
            runner.factorial.validate_methods(runner.parse_joint_experiment_config(config))
            if file_hash(stage / "experiment_config.json") != checks["config_sha256"]:
                raise ValueError("Prepared experiment config changed")
            inputs = list(read_records(stage / "inputs.jsonl"))
            if len(inputs) != checks["cases"] or digest(inputs) != checks["inputs_sha256"]:
                raise ValueError("Frozen input file changed")
            if checks["joint_activation_case"] != 0 or config["stream"]["warmup_cases"] != 0:
                raise ValueError("Every learner must start cold at input 1")
            if require_unstarted and (read(stage / "status.json")["status"] != "prepared" or
                                      (stage / "trajectories").exists() or (stage / "result.json").exists()):
                raise ValueError("A seed already executed numerical work")
            verified.append(dict(experiment=job["experiment"], batch=job["batch"], seed=seed,
                cases=checks["cases"], config_sha256=checks["config_sha256"],
                inputs_sha256=checks["inputs_sha256"], full_stream_sha256=checks["full_stream_sha256"]))
    if len(verified) != campaign["planned_seed_experiments"]:
        raise ValueError("Prepared seed coverage is incomplete")
    audit = dict(passed=True, checked_at=now(), unstarted=require_unstarted,
        max_workers=campaign["max_workers"], seed_batches=campaign["seed_batches"],
        seed_experiments=len(verified), paired_inputs=sum(v["cases"] for v in verified),
        method_evaluations=4 * sum(v["cases"] for v in verified), entries=verified)
    dump(output / "preparation_audit.json", audit)
    return audit


def preflight(output):
    """Build and audit cold learners and candidate schedules, without solves."""
    output = Path(output).resolve()
    verify(output)
    evidence = []
    for job in read(output / "campaign.json")["jobs"]:
        root = output / job["output"]
        for seed in job["seeds"]:
            runtimes = runner.initialize(root, seed)
            audit = read(root / f"seed_{seed}" / "initial_audit.json")
            evidence.append(dict(experiment=job["experiment"], seed=seed, passed=audit["passed"],
                features=audit["features"], independence=audit["independence"], candidates=audit["candidates"]))
            del runtimes
            gc.collect()
            print(f"Cold learner preflight: {job['experiment']} seed {seed}; no solves", flush=True)
    dump(output / "cold_start_preflight.json", dict(passed=True, native_solves_executed=0,
        checked_at=now(), seed_experiments=len(evidence), entries=evidence))
    verify(output)
    return evidence


def combine_experiment(output, label, jobs):
    """Combine completed seed batches using links to original measurements."""
    output = Path(output).resolve()
    combined = output / "combined" / label
    combined.mkdir(parents=True, exist_ok=False)
    roots = [output / job["output"] for job in jobs]
    campaign = read(output / "campaign.json")
    protocol = copy.deepcopy(read(roots[0] / "protocol.json"))
    protocol.update(name=f"{campaign['name']} — {label}", seeds=[], stages={},
        concurrency=f"Recorded seed batches, at most {campaign['max_workers']} workers concurrently; see campaign.json and component_runs.json")
    source = read(roots[0] / "source_manifest.json")
    components = []
    for root in roots:
        saved = read(root / "protocol.json")
        if read(root / "status.json")["status"] != "complete" or not read(root / "completion_audit.json")["passed"]:
            raise ValueError("Cannot combine an incomplete experiment")
        if read(root / "source_manifest.json") != source or saved["native_sha256"] != protocol["native_sha256"]:
            raise ValueError("Component source/native identities differ")
        for seed in saved["seeds"]:
            if seed in protocol["seeds"]:
                raise ValueError("Duplicate seed while combining batches")
            protocol["seeds"].append(seed)
            protocol["stages"][seed] = saved["stages"][seed]
            (combined / f"seed_{seed}").symlink_to(os.path.relpath(root / f"seed_{seed}", combined), target_is_directory=True)
        components.append(dict(output=os.path.relpath(root, combined), seeds=saved["seeds"],
            protocol_sha256=file_hash(root / "protocol.json"),
            summary_sha256=file_hash(root / "summary.json"),
            completion_audit_sha256=file_hash(root / "completion_audit.json")))
    (combined / "source").symlink_to(os.path.relpath(roots[0] / "source", combined), target_is_directory=True)
    shutil.copy2(roots[0] / "source_manifest.json", combined / "source_manifest.json")
    dump(combined / "component_runs.json", components)
    dump(combined / "protocol.json", protocol)
    rows = runner.analyze(combined)
    dump(combined / "status.json", dict(status="complete", passed=True, numerical_rerun=False, at=now()))
    return rows


def analyze(output):
    output = Path(output).resolve()
    campaign = read(output / "campaign.json")
    rows = []
    for label in campaign["experiments"]:
        jobs = [j for j in campaign["jobs"] if j["experiment"] == label]
        combined = output / "combined" / label
        if combined.exists():
            if not (combined / "summary.json").exists():
                raise ValueError("Incomplete combined view; inspect before regenerating")
            group = runner.analyze(combined)
        else:
            group = combine_experiment(output, label, jobs)
        rows.extend(dict(experiment=label, **r) for r in group)
    aggregates = []
    for label in campaign["experiments"]:
        for window in ("full", "last_1000"):
            for method in METHODS:
                group = [r for r in rows if r["experiment"] == label and r["window"] == window and r["method"] == method]
                if not group:
                    continue
                a = dict(experiment=label, window=window, method=method, seeds=len(group),
                    final_failures=sum(r["failures"] for r in group),
                    common_success_cases_min=min(r["common_success_cases"] for r in group))
                for field in ("native_reduction_pct", "inclusive_reduction_pct", "common_success_native_reduction_pct"):
                    values = [r[field] for r in group]
                    defined = all(v is not None for v in values)
                    a[field + "_mean"] = float(np.mean(values)) if defined else None
                    a[field + "_sd"] = float(np.std(values, ddof=1)) if defined and len(values) > 1 else None
                aggregates.append(a)
    dump(output / "summary.json", dict(passed=True, rows=rows, aggregates=aggregates))
    write_csv(output / "summary.csv", rows)
    write_csv(output / "aggregate.csv", aggregates)
    text = [campaign["name"], "", "Combined views use original measurements from the recorded seed batches. "
        "All-input costs include every attempt and recovery; final failures are reported separately. "
        "Common-success reductions use the same successful original inputs for all four methods. "
        "Dispersion is sample SD of per-seed ratios of summed costs.", "",
        "| Experiment | Method | Full native reduction | Inclusive reduction | Final failures |",
        "|---|---|---:|---:|---:|"]
    for a in aggregates:
        if a["window"] == "full":
            fmt = lambda key: f"{a[key+'_mean']:.2f} ± {a[key+'_sd']:.2f}%" if a[key+'_sd'] is not None else f"{a[key+'_mean']:.2f}%"
            text.append(f"| {a['experiment']} | {LABELS[a['method']]} | {fmt('native_reduction_pct')} | {fmt('inclusive_reduction_pct')} | {a['final_failures']} |")
    text.extend(["", "Per-experiment figures and raw-data links:", ""])
    text.extend(f"- [{label}](combined/{label}/report.md) · [Paper figures](combined/{label}/paper_figures/module04_paper_figures.pdf)" for label in campaign["experiments"])
    (output / "report.md").write_text("\n".join(text) + "\n")
    return aggregates


def launch(output):
    output = Path(output).resolve()
    verify(output)
    campaign = read(output / "campaign.json")
    sleep_guard = prevent_sleep()
    try:
        for index, job in enumerate(campaign["jobs"]):
            dump(output / "status.json", dict(status="running", pid=os.getpid(),
                completed_jobs=index, total_jobs=len(campaign["jobs"]), active_job=job, at=now()))
            print(f"Starting batch {job['batch']}: {job['experiment']}, seeds {', '.join(job['seeds'])}", flush=True)
            runner.launch(output / job["output"])
        verify(output, require_unstarted=False)
        analyze(output)
        dump(output / "status.json", dict(status="complete", passed=True, completed_jobs=len(campaign["jobs"]), at=now()))
    except BaseException:
        dump(output / "status.json", dict(status="failed", at=now(), traceback=traceback.format_exc()))
        raise
    finally:
        stop_sleep_prevention(sleep_guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "preflight", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--cases", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.plan is None:
            parser.error("prepare requires --plan")
        prepare(args.plan, args.output, cases=args.cases)
    elif args.command == "verify":
        verify(args.output)
    elif args.command == "preflight":
        preflight(args.output)
    elif args.command == "run":
        launch(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
