"""Resumable fresh scans followed by independent five-policy measurements."""

from __future__ import annotations

import fcntl
from pathlib import Path
import signal
import statistics

import numpy as np

from experiments.paper_final.common import artifacts as a
from experiments.paper_final.common import frozen_policy as policy
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from experiments.paper_final.common.workers import run_frozen_worker, run_frozen_phase
from experiments.runtime import prevent_sleep, stop_sleep_prevention

from .configuration import METHODS, LABELS, fixed_name
from .planning import evaluation_jobs
from .selection import select_fixed, verify_coverage, trial_key

RUNNER = "experiments.paper_final.run_policy"


def verify(output):
    output = Path(output)
    saved = a.read(output / "prepared.json")
    for name, sha in saved["files_sha256"].items():
        if a.file_hash(output / name) != sha:
            raise ValueError(f"Prepared input changed: {name}")
    source = a.read(output / "source_manifest.json")
    if set(source) != execution_source_paths():
        raise ValueError("Source file inventory changed after preparation")
    for name, sha in source.items():
        if a.file_hash(a.ROOT / name) != sha:
            raise ValueError(f"Execution source changed: {name}")
    marker = output / "selection_complete.json"
    if marker.is_file():
        for name, sha in a.read(marker)["files_sha256"].items():
            if a.file_hash(output / name) != sha:
                raise ValueError(f"Fixed-selection evidence changed: {name}")
    policy.environment_check(output)


class FrozenRuntime:
    def __init__(self, output):
        self.protocol = a.read(Path(output) / "protocol.json")

    def load_bundle(self, output, seed):
        source = self.protocol["sources"][str(seed)]
        directory = Path(output) / source["directory"]
        bundle = policy.build_bundle(a.read(directory / "experiment_config.json"),
                                     source["method"], directory / "controller.npz",
                                     beta=source["effective_beta"])
        bundle.controller.start_episode(initial_environment_weight=1.0)
        bundle.controller.finish_episode(learned=False)
        return bundle

    def run_policy(self, job, name, spec, bundle):
        return policy.run_policy(job, name, spec, bundle,
                                 tolerance=self.protocol["tolerance"],
                                 max_cycles=self.protocol["max_cycles"], profile=self.protocol["profile"])

    frozen_snapshot = staticmethod(policy.frozen_snapshot)
    verify_frozen = staticmethod(policy.verify_frozen)


def worker(output, phase, worker_id):
    protocol = a.read(output / "protocol.json")
    if phase == "test" and not (output / "selection_complete.json").is_file():
        raise ValueError("Complete fresh scans and freeze selections before evaluation")
    run_frozen_worker(output, phase, worker_id, protocol["max_workers"], run_number=5,
                      verify=verify, policy=FrozenRuntime(output))


def preflight(output):
    runtime = FrozenRuntime(output)
    jobs = a.read(output / "jobs_base.json")
    checks = []
    for seed in runtime.protocol["training_seeds"]:
        job = next(j for j in jobs if j["seed"] == seed)
        bundle = runtime.load_bundle(output, seed)
        before = policy.frozen_snapshot(bundle)
        policies = runtime.protocol["policies"]
        rows = [runtime.run_policy(job, name, policies[name], bundle)
                for name in (fixed_name(1), "periodic", "rl_frozen", fixed_name(1))]
        np.testing.assert_allclose(rows[0]["outcome"].get("cycle_residuals", []),
                                   rows[-1]["outcome"].get("cycle_residuals", []), rtol=1e-12, atol=1e-15)
        if rows[0]["success"] != rows[-1]["success"]:
            raise ValueError("Repeated reference outcome changed")
        policy.verify_frozen(bundle, before)
        for row in rows:
            audit_trial(row, runtime.protocol)
        checks.append({"seed": seed, "same_reference_residuals": True, "controller_frozen": True})
    a.dump(output / "preflight.json", {"passed": True, "checks": checks})


def audit_trial(row, protocol):
    policy.reject_execution_error(row["outcome"])
    policy.audit_outcome(row["outcome"], tolerance=protocol["tolerance"])
    if row["outcome"].get("controller_update_committed", False):
        raise ValueError("Frozen controller learned")
    spec = protocol["policies"][row["policy"]]
    if spec["kind"] in ("fixed", "periodic"):
        actions = row["outcome"].get("cycle_actions", [])
        pattern = spec["pattern"]
        if actions != [pattern[i % len(pattern)] for i in range(len(actions))]:
            raise ValueError("Executed weights differ from prescribed schedule")


def phase_rows(output, phase):
    marker = a.read(output / "raw" / phase / "complete.json")
    protocol = a.read(output / "protocol.json")
    rows = []
    for name, sha in marker["raw_sha256"].items():
        if a.file_hash(output / name) != sha:
            raise ValueError("Recorded trial file changed")
        for row in a.read_records(output / name):
            if row["phase"] != phase:
                raise ValueError("Wrong measurement phase")
            audit_trial(row, protocol)
            rows.append(row)
    verify_coverage(rows, a.read(output / f"jobs_{phase}.json"))
    if len(rows) != marker["trials"]:
        raise ValueError("Completion marker trial count differs")
    return rows


def select(output):
    verify(output)
    if (output / "selection_complete.json").is_file():
        return a.read(output / "selections.json")
    rows = phase_rows(output, "scan")
    choices = select_fixed(rows, a.read(output / "jobs_scan.json"))
    a.dump(output / "selections.json", choices)
    jobs = evaluation_jobs(a.read(output / "jobs_base.json"), choices,
                           a.read(output / "configuration.json"))
    a.dump(output / "jobs_test.json", jobs)
    evidence = ["selections.json", "jobs_test.json", "raw/scan/complete.json"]
    evidence += list(a.read(output / "raw/scan/complete.json")["raw_sha256"])
    a.dump(output / "selection_complete.json", {"at": a.now(), "fresh_scan": True,
           "files_sha256": {name: a.file_hash(output / name) for name in evidence}})
    return choices


def analyze(output):
    verify(output)
    # Both datasets remain independently auditable after completion.
    scans = phase_rows(output, "scan")
    selections = a.read(output / "selections.json")
    if select_fixed(scans, a.read(output / "jobs_scan.json")) != selections:
        raise ValueError("Saved fixed selections do not match the fresh scan")
    rows = phase_rows(output, "test")
    jobs = a.read(output / "jobs_test.json")
    protocol = a.read(output / "protocol.json")
    by_key = {trial_key(r): r for r in rows}
    fields = ("native_continuation_sec", "inclusive_continuation_sec", "native_total_sec", "inclusive_total_sec")
    per_seed, per_case = [], []
    for seed in protocol["training_seeds"]:
        base = [j for j in jobs if j["seed"] == seed and j["repeat"] == 0]
        for method in METHODS:
            cases = []
            for job in base:
                name = job["method_policies"][method]
                if name is None:
                    cases.append(None)
                    continue
                observations = [by_key[(seed, job["source"], job["case_id"], rep, name)]
                                for rep in range(protocol["repetitions_per_case"])]
                case = {"seed": seed, "case_id": job["case_id"], "method": method, "policy": name,
                        "success": all(r["success"] for r in observations),
                        "recovered": any(r["outcome"].get("fallback_used", False) for r in observations),
                        **{k: statistics.mean(r[k] for r in observations) for k in fields}}
                cases.append(case)
                per_case.append(case)
            available = [c for c in cases if c is not None]
            complete = len(available) == len(cases)
            per_seed.append({"seed": seed, "method": method, "label": LABELS[method],
                             "defined": complete, "cases": len(available), "expected_cases": len(cases),
                             "failed_cases": sum(not c["success"] for c in available),
                             **{k: sum(c[k] for c in available) if complete else None for k in fields}})
    comparisons = []
    for method in METHODS[:-1]:
        for field in fields[:2]:
            values = []
            for seed in protocol["training_seeds"]:
                rr = {r["method"]: r for r in per_seed if r["seed"] == seed}
                if (rr[method]["defined"] and rr[method][field] > 0
                        and rr[method]["failed_cases"] == 0 and rr["rl"]["failed_cases"] == 0):
                    values.append(100 * (1 - rr["rl"][field] / rr[method][field]))
            comparisons.append({"comparator": method, "metric": field, "seeds": len(values),
                                "mean_pct": statistics.mean(values) if values else None,
                                "sample_sd_pct": statistics.stdev(values) if len(values) > 1 else None})
    summary = {"protocol": protocol, "selections": selections, "per_seed": per_seed,
               "rl_comparisons": comparisons, "audit": {"fresh_scan_trials": len(scans),
               "evaluation_trials": len(rows), "exact_coverage": True,
               "evaluation_failures": sum(not r["success"] for r in rows),
               "recovered_evaluations": sum(bool(r["outcome"].get("fallback_used", False)) for r in rows)}}
    a.dump(output / "analysis/summary.json", summary)
    a.write_csv(output / "analysis/per_seed.csv", per_seed)
    a.write_csv(output / "analysis/per_case.csv", per_case)
    lines = [f"# Module 2 — {protocol['name']}", "",
             f"Training seed labels: {protocol['training_seeds']}. {protocol['test_cases']} paired problems; "
             f"{protocol['repetitions_per_case']} independent evaluation repetitions.", "",
             f"Fresh fixed-weight scan: {len(scans)} trials. Independent evaluation: {len(rows)} trials.",
             f"Evaluation failures: {summary['audit']['evaluation_failures']}; recovered evaluations: {summary['audit']['recovered_evaluations']}.",
             "Costs include executed recovery. Common primary setup is excluded from continuation costs.",
             "Fixed choices are successful best-observed grid choices on these evaluation cases, then independently retimed.",
             "", "| Frozen RL compared with | Cost | Mean reduction (%) | Sample SD | Seeds |",
             "|---|---|---:|---:|---:|"]
    for row in comparisons:
        mean = 'undefined' if row['mean_pct'] is None else f"{row['mean_pct']:.2f}"
        sd = '—' if row['sample_sd_pct'] is None else f"{row['sample_sd_pct']:.2f}"
        lines.append(f"| {LABELS[row['comparator']]} | {row['metric']} | {mean} | {sd} | {row['seeds']} |")
    lines += ["", "A comparator is undefined where the scan found no qualifying successful weight; it is never replaced by a failed minimum.",
              "Speedup summaries include only seeds where both compared methods complete every evaluation successfully; failure counts remain in the tables and raw records.",
              "Raw scan/evaluation JSONL files retain per-cycle actions, residuals, timing and recovery outcomes."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    from experiments.paper_final.plot_policy import generate
    generate(output)
    with (output / "REPORT.md").open("a") as handle:
        handle.write("\n[Paper figures](paper_figures/module2_paper_figures.pdf) · [Five-policy weights](paper_figures/matched_policy_weights.png) · [Plot data and provenance](paper_figures/manifest.json)\n")
    return summary


def run(output, *, scan_only=False):
    output = Path(output).resolve()
    verify(output)
    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        awake = prevent_sleep()
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise KeyboardInterrupt("Termination requested; durable records can be resumed")
        signal.signal(signal.SIGTERM, terminate)
        try:
            if not (output / "preflight.json").is_file():
                preflight(output)
            workers = a.read(output / "protocol.json")["max_workers"]
            if not (output / "raw/scan/complete.json").is_file():
                run_frozen_phase(output, "scan", runner_module=RUNNER, run_number=5, workers=workers)
            select(output)
            if scan_only:
                a.dump(output / "status.json", {"status": "scanned", "selections_frozen": True})
                return
            if not (output / "raw/test/complete.json").is_file():
                run_frozen_phase(output, "test", runner_module=RUNNER, run_number=5, workers=workers)
            result = analyze(output)
            a.dump(output / "complete.json", {"at": a.now(), "status": "complete", "audit": result["audit"]})
            a.dump(output / "status.json", {"status": "complete", "audit": result["audit"]})
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            stop_sleep_prevention(awake)
