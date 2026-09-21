"""Prespecified, opt-in hierarchy attribution and timing stability experiments.

Reuse the production setup learner, controller transaction, recovery and timers.
Default execution validates inputs only. --run starts the serialized study;
--smoke is a separate small-grid integration check, excluded from conclusions.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

from experiments.diagnostics.solve_control import _project_paths  # noqa: E402,F401
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from composable_joint_4k import resolve_composable_study
from hypre.bindings import run_with_default_fallback
from hypre.bindings.recovery import InvalidObservationError
from hypre.bindings.config import configure_smoother_profile
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from joint_online_common import report_online_outcome, _build_paired_instance_stream
from online_td_experiment_common import _json_ready, _write_json
from problems.registry import context_for_setup_method, learning_context_for_setup
from problems.streams import generate_scalar_anisotropic_diffusion_advection_instances
from run_online_methods_2k import _as_feedback
from run_paper_final import file_hash, source_state
from setup.space import DEFAULT_SETUP_PARAMS
from setup_aware_compare_common import (augment_setup_params, build_online_linucb_branch,
    run_bandit_step_test_final, solve_no_rl_case, solve_schedule_case)
from solve.controllers.common import OnlineSolveCase
from experiments.paper_final.run_02_diagnostics import (
    build_bundle, audit_controller, audit_frozen_policies)
from experiments.paper_final.timing_selection import TimingSelection

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/paper_final/05_policy/protocol.json"
OUTPUT = ROOT / "results/paper_final/05_policy/controlled"


def digest(value):
    return hashlib.sha256(json.dumps(_json_ready(value), sort_keys=True).encode()).hexdigest()


def check_execution_errors(outcome):
    """Programming/interface errors must never become valid fallback samples."""
    for attempt in [outcome, *outcome.get("primary_attempts", [])]:
        origins = (attempt.get("failure_origin"), attempt.get("fallback_failure_origin"))
        reasons = [str(attempt.get(key, "")) for key in
                   ("failure_reason", "primary_failure_reason", "fallback_failure_reason")]
        execution_reasons = [reason for reason in reasons if reason.startswith("exception:")
                             and not reason.startswith("exception:AMGNativeError:")]
        if "execution" in origins or execution_reasons:
            raise InvalidObservationError("Execution error in study attempt: " + "; ".join(execution_reasons or reasons))


def check_outcome(outcome):
    check_execution_errors(outcome)
    total = sum(float(outcome.get(k, 0)) for k in
                ("setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime"))
    if not np.isfinite(total) or total < 0 or not np.isclose(
            total, outcome["end_to_end_runtime"], rtol=1e-9, atol=1e-10):
        raise AssertionError("Invalid component accounting")
    if not outcome["failed"] and not (np.isfinite(outcome["completed_residual_norm"])
                                      and outcome["completed_residual_norm"] <= 1e-6):
        raise AssertionError("Successful final status exceeds tolerance")
    outcome["solve_control_runtime"] = float(outcome["end_to_end_runtime"] -
        outcome.get("primary_setup_runtime", outcome["setup_runtime"] - outcome.get("fallback_setup_runtime", 0)))
    outcome["algorithm_runtime"] = float(outcome["end_to_end_runtime"])


def match_hierarchies(rows):
    hashes = [r["outcome"].get("hierarchy_fingerprint") for r in rows]
    if any(h is not None for h in hashes):
        if len(set(hashes)) != 1 or any(r["outcome"].get("initial_cycle") != 0 for r in rows):
            raise AssertionError("Paired policies received different prepared hierarchies/initial states")
    elif any(r["outcome"].get("primary_status") != "setup_failure" for r in rows):
        raise AssertionError("Missing hierarchy audit for a prepared primary attempt")
    physical = [r for r in rows if "shared_prefix_source" not in r]
    common_setup = float(np.mean([r["outcome"].get("primary_setup_runtime",
        r["outcome"]["setup_runtime"] - r["outcome"].get("fallback_setup_runtime", 0)) for r in physical]))
    for row in rows:
        row["outcome"]["common_primary_setup_runtime"] = common_setup
        row["outcome"]["logical_end_to_end_runtime"] = (
            row["outcome"]["solve_control_runtime"] + common_setup + row.get("source_decision_runtime", 0))


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row["phase"], row["source"], row["policy"], row.get("repeat", 0))].append(row)
    summaries = []
    for (phase, source, policy, repeat), values in sorted(cells.items()):
        outcomes = [v["outcome"] for v in values]
        summaries.append({"phase": phase, "source": source, "policy": policy, "repeat": repeat,
            "cases": len(values), "final_failures": sum(bool(o["failed"]) for o in outcomes),
            "primary_failures": sum(o.get("primary_status", "success") != "success" for o in outcomes),
            "primary_cycles": sum(o.get("primary_cycles", o["iterations"]) for o in outcomes),
            **{key: float(sum(o.get(key, 0) for o in outcomes)) for key in (
                "solve_control_runtime", "end_to_end_runtime", "logical_end_to_end_runtime",
                "algorithm_runtime", "calibration_charge", "infer_runtime", "bandit_overhead_runtime",
                "hierarchy_audit_runtime", "method_wall_runtime")}})
    return summaries


def choose_baselines(rows, sources):
    cells = summarize(rows)
    selected = {}
    for source in sources:
        selected[source] = {}
        for family in ("fixed", "schedule"):
            eligible = [c for c in cells if c["phase"] == "selection" and c["source"] == source
                        and c["policy"].startswith(family + "_")]
            winner = min(eligible, key=lambda c: (c["final_failures"], c["solve_control_runtime"], c["policy"]))
            selected[source][family] = winner["policy"]
    return selected


def paired_comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["phase"] not in {"evaluation", "online", "robustness"}:
            continue
        group = groups[(row["phase"], row["source"], row.get("repeat", 0))]
        group.setdefault(row["policy"], {})[row["index"]] = row["outcome"]
    comparisons = []
    for (phase, source, repeat), policies in sorted(groups.items()):
        references = (["fixed_1", "old_rl"] if phase == "evaluation" else
                      ["w1"] if phase == "online" else ["raw", "stable"])
        metric = ("solve_control_runtime" if phase == "evaluation" else
                  "logical_end_to_end_runtime" if phase == "online" else "algorithm_runtime")
        for reference in references:
            if reference not in policies:
                continue
            baseline = policies[reference]
            for name, candidate in sorted(policies.items()):
                if name == reference:
                    continue
                if set(baseline) != set(candidate):
                    raise AssertionError("Unpaired cost comparison")
                indices = sorted(baseline)
                costs = np.array([baseline[i][metric] for i in indices])
                savings = costs - [candidate[i][metric] for i in indices]
                cumulative = np.cumsum(savings)
                last_negative = np.flatnonzero(cumulative < 0)
                payback = (None if cumulative[-1] < 0 else
                           int(indices[int(last_negative[-1]) + 1] + 1) if len(last_negative) else indices[0] + 1)
                comparisons.append({"phase": phase, "source": source, "repeat": repeat,
                    "reference": reference, "candidate": name, "metric": metric,
                    "cases": len(indices), "saved_sec": float(sum(savings)),
                    "reduction_pct": float(100 * sum(savings) / sum(costs)),
                    "observed_final_nonnegative_stretch_from": payback})
    return comparisons


def write_report(directory, rows, title, complete=False):
    directory.mkdir(parents=True, exist_ok=True)
    cells = summarize(rows)
    comparisons = paired_comparisons(rows)
    repeat_statistics = []
    for name in sorted({c["policy"] for c in cells if c["phase"] == "robustness"}):
        totals = [c["algorithm_runtime"] for c in cells if c["phase"] == "robustness" and c["policy"] == name]
        repeat_statistics.append({"policy": name, "executions": len(totals), "totals_sec": totals,
            "mean_sec": float(np.mean(totals)), "sample_sd_sec": float(np.std(totals, ddof=1)) if len(totals) > 1 else None})
    _write_json(directory / "summary.json", {"complete": complete, "cells": cells,
        "paired_comparisons": comparisons, "execution_statistics": repeat_statistics})
    lines = [f"# {title}", "", "Development evidence; timings include unsuccessful attempts and recovery.",
             "Frozen timing repeats are not independent training repetitions.", "",
             "| Phase | Source | Policy | Repeat | Cases | Solve/control/recovery (s) | Logical total (s) | Algorithm cost (s) | Final failures |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['phase']} | {c['source']} | {c['policy']} | {c['repeat']} | {c['cases']} | "
                     f"{c['solve_control_runtime']:.3f} | {c['logical_end_to_end_runtime']:.3f} | "
                     f"{c['algorithm_runtime']:.3f} | {c['final_failures']} |")
    lines += ["", "Algorithm cost preserves physically measured method costs and adds the full calibration charge to near_tie.",
              "Matched-hierarchy logical totals instead add an identical primary setup and source decision cost to all paired policies.",
              "Source selection is an externally supplied frozen policy; historical source-training cost is not included.",
              "Hashing, file IO, matrix assembly and uninstrumented wrapper work are outside the algorithm timer; enclosing wall time is retained.",
              "No timing-repeat or within-trajectory bootstrap is used to claim training uncertainty."]
    if comparisons:
        lines += ["", "| Phase/source/repeat | Reference → candidate | Saved (s) | Reduction | Observed payback input |",
                  "|---|---|---:|---:|---:|"]
        for c in comparisons:
            lines.append(f"| {c['phase']}/{c['source']}/{c['repeat']} | {c['reference']} → {c['candidate']} | "
                         f"{c['saved_sec']:.3f} | {c['reduction_pct']:.2f}% | {c['observed_final_nonnegative_stretch_from']} |")
        lines += ["", "Payback denotes the final nonnegative savings stretch within this observed horizon, not a future guarantee."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    if complete and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 5))
        by_policy = defaultdict(list)
        for row in rows:
            if row["phase"] in {"online", "robustness"}:
                by_policy[(row["policy"], row.get("repeat", 0))].append(row)
        if by_policy:
            for (policy, repeat), values in sorted(by_policy.items()):
                values.sort(key=lambda r: r["index"])
                field = "logical_end_to_end_runtime" if values[0]["phase"] == "online" else "algorithm_runtime"
                axis.plot(np.arange(1, len(values) + 1), np.cumsum([r["outcome"][field] for r in values]),
                          label=f"{policy} / {repeat + 1}")
            axis.set(xlabel="Input", ylabel="Cumulative recorded cost (s)", title=title)
        else:
            held = [c for c in cells if c["phase"] == "evaluation" and c["repeat"] == 0]
            labels = [f"{c['source']}\n{c['policy']}" for c in held]
            axis.bar(np.arange(len(held)), [c["solve_control_runtime"] for c in held])
            axis.set_xticks(np.arange(len(held)), labels, rotation=60, ha="right", fontsize=7)
            axis.set(ylabel="Solve/control/recovery cost (s)", title=title)
        if by_policy:
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / "cost.png", dpi=160)
        plt.close(figure)


class Study:
    def __init__(self, protocol, output, smoke=False):
        self.protocol, self.output, self.smoke = protocol, output, smoke
        self.old, self.new = ROOT / protocol["old_run"], ROOT / protocol["new_run"]
        self.raw = json.loads((self.new / "experiment_config.json").read_text())
        self.runtime = runtime_config_from_spec(parse_joint_experiment_config(self.raw))
        self.space = resolve_composable_study(self.runtime).setup_configuration_spaces["recommended"]
        self.contract = learning_context_for_setup(self.raw["problem"]["kind"], "linucb", "canonical_no_c_mean")
        self.order = np.random.default_rng(protocol["order_seed"])
        self.sources = {}
        self.status = {"status": "preparing", "smoke": smoke, "completed_records": 0}
        self.timing_output = output / "timing" if smoke else ROOT / protocol["timing_output"]
        self.last_power_check = 0.0

    def update_status(self, phase, **fields):
        if phase != self.status.get("phase"):
            for key in ("index", "total", "repeat"):
                self.status.pop(key, None)
        self.status["status"] = "running"
        self.status.update(phase=phase, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
        _write_json(self.output / "status.json", self.status)
        print(json.dumps(self.status), flush=True)

    def environment(self, phase):
        if shutil.disk_usage(self.output).free < 1.0 * 2**30:
            raise RuntimeError("Less than 1 GiB free; preserving partial results and stopping")
        if self.smoke or time.monotonic() - self.last_power_check < 30:
            return
        while True:
            power = subprocess.check_output(["/usr/bin/pmset", "-g", "batt"], text=True)
            with (self.output / "environment.jsonl").open("a") as handle:
                handle.write(json.dumps({"utc": time.time(), "phase": phase, "power": power,
                                         "load": os.getloadavg()}) + "\n")
            if "AC Power" in power:
                break
            self.update_status(phase, status="waiting_for_ac")
            time.sleep(30)
        self.last_power_check = time.monotonic()
        self.status["status"] = "running"

    def branch(self, directory, cases=5000):
        branch, _ = build_online_linucb_branch(
            seed=int(self.raw["seeds"]["bandit"]), learner_kind="linucb", tune_dim=7,
            tune7_variant="categorical", action_space_mode="full_cartesian", solver_tol=1e-6,
            solver_max_iter=50, parameter_resolution=20, configuration_space=self.space,
            candidate_schedule_dir=directory, candidate_schedule_rounds=cases * 3,
            candidate_sampling="structured512", context_dim=self.contract.dimension,
            context_interaction_indices=self.contract.interaction_indices)
        return branch

    def stream(self, name, count):
        p = self.protocol
        result = generate_scalar_anisotropic_diffusion_advection_instances(
            count=count, seed=p[name + "_seed"], grid_choices=[(p["grid"],) * 3],
            c_min=1, c_max=1000, advection_min=1, advection_max=1000)
        return [(dict(mkw), np.asarray(context)) for mkw, context in result]

    def context(self, mkw, context):
        return context_for_setup_method(problem_kind=self.raw["problem"]["kind"], setup_kind="linucb",
            setup_context="canonical_no_c_mean", matrix_kwargs=mkw, stream_context=context)

    def bundle(self, checkpoint=None):
        return build_bundle(self.raw, "bandit_lstdq", checkpoint)

    def prepare_sources(self):
        files = {}
        for source in self.protocol["sources"]:
            root = self.old if source.startswith("old") else self.new
            method = "bandit_default" if source.endswith("setup") else "bandit_lstdq"
            state = root / "final_bandit_states" / f"{method}.npz"
            branch = self.branch(self.output / "source_candidates")
            branch.policy.model.load_mutable_state(state)
            self.sources[source] = branch.policy.model
            files[str(state.relative_to(ROOT))] = file_hash(state)
        for root in (self.old, self.new):
            for name in ("bandit_lstdq_final.npz", "bandit_lstdq_final.npz.encoder.json"):
                path = root / "checkpoints" / name
                files[str(path.relative_to(ROOT))] = file_hash(path)
        bank = set()
        for model in self.sources.values():
            bank.update(map(int, model._cand.always_include_arms))
            bank.update(map(int, model._cand.elite_arms))
        rng = np.random.default_rng(self.protocol["source_candidate_seed"])
        count = self.protocol["source_candidate_count"]
        if len(bank) > count:
            raise ValueError("Frozen elite union exceeds prescribed bank size")
        while len(bank) < count:
            bank.add(int(rng.integers(next(iter(self.sources.values())).K)))
        self.bank = np.array(sorted(bank), dtype=np.int64)
        np.save(self.output / "frozen_candidate_bank.npy", self.bank)
        _write_json(self.output / "input_artifacts.json", {"files": files, "bank_sha256": digest(self.bank.tolist())})
        for name, model in self.sources.items():
            model.save_mutable_state(self.output / "frozen_sources" / f"{name}_before.npz")

    def hierarchy(self, source, mkw, context):
        if source == "default":
            return dict(DEFAULT_SETUP_PARAMS), 0.0
        model = self.sources[source]
        started = time.perf_counter()
        params, _ = model.recommend(self.context(mkw, context), candidate_arms=self.bank,
                                    alpha=model._effective_alpha())
        return params, time.perf_counter() - started

    def fallback(self, mkw):
        return solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(mkw),
            solver_tol=1e-6, solver_max_iter=50, augment_params=augment_setup_params)

    def solve(self, mkw, params, *, pattern=None, bundle=None, learn=False, audit=True):
        started = time.perf_counter()
        if bundle is None:
            pattern = [1.] if pattern is None else pattern
            schedule = [(i + 1, pattern[i % len(pattern)], 1, 1) for i in range(50)]
            native = run_with_default_fallback(
                lambda: solve_schedule_case(params=dict(params), mkw=dict(mkw), schedule=schedule,
                    solve_tol=1e-6, solve_max_cycles=50, audit_hierarchy=audit),
                lambda: self.fallback(mkw)).to_result()
        else:
            native = bundle.run_case(OnlineSolveCase(mkw=mkw, params=params, solve_tol=1e-6,
                solve_max_cycles=50, learn=learn, explore=learn, record_action_metadata=True,
                fallback_attempt=lambda: self.fallback(mkw), audit_hierarchy=audit))
            # The controller transaction leaves setup-construction failures to
            # its caller. This fixed-hierarchy study permits one Default fallback,
            # not the setup learner's alternative-configuration reselection.
            if not native.get("recovery_protocol_applied", False):
                native = run_with_default_fallback(lambda: native, lambda: self.fallback(mkw)).to_result()
        outcome = report_online_outcome(_as_feedback(native, include_controller=True), bandit_timing={})
        outcome["method_wall_runtime"] = time.perf_counter() - started
        outcome["completed_residual_norm"] = float(outcome.get("fallback_residual_norm", np.nan)
            if outcome.get("fallback_used") else outcome["residual_norm"])
        check_outcome(outcome)
        return outcome

    def warmup(self, phase):
        self.environment(phase)
        rows = []
        for mkw, _ in self.stream("calibration", 4):
            rows.append(self.solve(mkw, DEFAULT_SETUP_PARAMS, audit=False))
        _write_json(self.output / f"warmup_{phase}.json", rows)

    def append(self, path, rows, retained):
        with path.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(_json_ready(row), allow_nan=True) + "\n")
        # Full traces stay on disk. Do not progressively increase memory pressure
        # across timing repetitions by keeping cycle arrays and RNG logs alive.
        for row in rows:
            retained.append({**{key: row[key] for key in ("phase", "source", "policy", "index")},
                "repeat": row.get("repeat", 0), "outcome": {key: value for key, value in row["outcome"].items()
                    if key in {"setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime",
                        "end_to_end_runtime", "solve_control_runtime", "algorithm_runtime", "calibration_charge",
                        "logical_end_to_end_runtime", "primary_cycles", "iterations", "primary_status", "failed",
                        "hierarchy_audit_runtime", "method_wall_runtime"}}})
        self.status["completed_records"] += len(rows)

    def calibrate(self):
        self.warmup("calibration")
        rows, deviations = [], []
        for index, (mkw, context) in enumerate(self.stream("calibration", self.protocol["calibration_cases"])):
            self.environment("calibration")
            group = [{"phase": "calibration", "source": "default", "policy": "w1", "index": index,
                      "repeat": repeat, "mkw": mkw, "params": dict(DEFAULT_SETUP_PARAMS),
                      "outcome": self.solve(mkw, DEFAULT_SETUP_PARAMS)}
                     for repeat in range(self.protocol["calibration_repeats"])]
            match_hierarchies(group)
            numerical = [(r["outcome"].get("primary_cycles", r["outcome"]["iterations"]),
                          r["outcome"].get("primary_residual_norm", r["outcome"]["residual_norm"])) for r in group]
            if len(set(numerical)) != 1:
                raise AssertionError("Calibration changed numerical work")
            deviations.append(float(np.std([r["outcome"]["end_to_end_runtime"] for r in group], ddof=1)))
            self.append(self.output / "calibration.jsonl", group, rows)
        result = {"sigma_sec": float(np.quantile(deviations, .75)), "within_input_sd_sec": deviations,
                  "charge_sec": sum(r["outcome"]["end_to_end_runtime"] for r in rows),
                  "rule": self.protocol["sigma_rule"]}
        _write_json(self.output / "calibration.json", result)
        self.update_status("calibration", status="completed_phase", sigma_sec=result["sigma_sec"])
        return result

    def frozen(self):
        directory = self.output / "frozen"
        directory.mkdir()
        bundles = {name: self.bundle((self.old if name == "old_rl" else self.new) /
                    "checkpoints/bandit_lstdq_final.npz") for name in ("old_rl", "new_rl")}
        before = {n: b.controller.snapshot_learning_state() for n, b in bundles.items()}
        rng_before = {n: json.dumps(b.controller.rng.bit_generator.state) for n, b in bundles.items()}
        for bundle in bundles.values():
            audit_controller(bundle.controller)
        patterns = {f"fixed_{w:g}": [w] for w in self.protocol["fixed_weights"]}
        patterns.update({"schedule_" + n: v for n, v in self.protocol["schedules"].items()})
        self.warmup("frozen")
        rows, selections = [], None
        for phase, count in (("selection", self.protocol["selection_cases"]),
                             ("evaluation", self.protocol["evaluation_cases"])):
            for index, (mkw, context) in enumerate(self.stream(phase, count)):
                self.environment("frozen_" + phase)
                for source in self.protocol["sources"]:
                    params, source_time = self.hierarchy(source, mkw, context)
                    policies = list(patterns) if phase == "selection" else list(dict.fromkeys(
                        ["fixed_1", selections[source]["fixed"], selections[source]["schedule"], "old_rl", "new_rl"]))
                    repeats = (self.protocol["timing_repeats"] if phase == "evaluation" and
                               index < self.protocol["timing_repeat_cases"] else 1)
                    for repeat in range(repeats):
                        group = []
                        for name in self.order.permutation(policies):
                            outcome = self.solve(mkw, params, pattern=patterns.get(name), bundle=bundles.get(name))
                            group.append({"phase": phase, "source": source, "policy": str(name), "index": index,
                                "repeat": repeat, "mkw": mkw, "params": params, "source_decision_runtime": source_time,
                                "outcome": outcome})
                        match_hierarchies(group)
                        self.append(directory / "trajectories.jsonl", group, rows)
                if index % 8 == 0:
                    self.update_status("frozen_" + phase, index=index + 1, total=count)
                    write_report(directory, rows, "Frozen controller cross evaluation")
            if phase == "selection":
                selections = choose_baselines(rows, self.protocol["sources"])
                _write_json(directory / "selection.json", selections)
        audit_frozen_policies(bundles, before, rng_before)
        for bundle in bundles.values():
            audit_controller(bundle.controller)
        _write_json(directory / "frozen_audit.json", {"learning_state_and_rng_unchanged": True})
        write_report(directory, rows, "Frozen controller cross evaluation", complete=True)
        return selections, patterns

    def online(self, selections, patterns):
        directory = self.output / "online"
        directory.mkdir()
        source = self.protocol["online_source"]
        controller = self.bundle()
        policies = ["w1", "selected_fixed", "selected_schedule", "online_rl"]
        static = {"w1": [1.], "selected_fixed": patterns[selections[source]["fixed"]],
                  "selected_schedule": patterns[selections[source]["schedule"]]}
        rows = []
        self.update_status("online", index=0, total=self.protocol["online_cases"])
        self.warmup("online")
        for index, (mkw, context) in enumerate(self.stream("online", self.protocol["online_cases"])):
            self.environment("online")
            params, source_time = self.hierarchy(source, mkw, context)
            group = []
            physical_policies = policies if index >= self.protocol["online_activation"] else policies[:-1]
            for policy in self.order.permutation(physical_policies):
                active = policy == "online_rl" and index >= self.protocol["online_activation"]
                outcome = self.solve(mkw, params, pattern=static.get(policy, [1.]),
                                     bundle=controller if active else None, learn=active)
                group.append({"phase": "online", "source": source, "policy": str(policy), "index": index,
                    "mkw": mkw, "params": params, "source_decision_runtime": source_time,
                    "rl_active": active, "outcome": outcome})
            if index < self.protocol["online_activation"]:
                reference = next(row for row in group if row["policy"] == "w1")
                group.append({**copy.deepcopy(reference), "policy": "online_rl", "shared_prefix_source": "w1"})
            match_hierarchies(group)
            self.append(directory / "trajectories.jsonl", group, rows)
            if (index + 1) % 100 == 0:
                self.update_status("online", index=index + 1, total=self.protocol["online_cases"])
                write_report(directory, rows, "Online solve learning on common hierarchies")
            if (index + 1) % 1000 == 0:
                controller.save(directory / f"rl_{index + 1}.npz")
        controller.save(directory / "rl_final.npz")
        _write_json(directory / "controller_audit.json", audit_controller(controller.controller))
        write_report(directory, rows, "Online solve learning on common hierarchies", complete=True)

    def robustness(self, calibration):
        self.timing_output.mkdir(parents=True, exist_ok=False)
        p = self.protocol
        records = [json.loads(line) for line in (self.new / "trajectories/bandit_default.jsonl").open()]
        records = records[:p["robustness_cases"]]
        if self.smoke:
            records = [{"mkw": m, "context": self.context(m, c).tolist(), "problem_context": c.tolist()}
                       for m, c in self.stream("online", p["robustness_cases"])]
        else:
            # Saved trajectory `context` is the seven-value setup-learner view.
            # The solve API accepts the full eight-value PDE context and applies
            # its own projection. Rebuild it through the production stream helper.
            stream, _ = _build_paired_instance_stream(self.runtime)
            for entry, (mkw, context) in zip(records, stream):
                if entry["mkw"] != mkw:
                    raise AssertionError("Robustness input differs from the production stream")
                np.testing.assert_array_equal(entry["context"], self.context(mkw, context))
                entry["problem_context"] = context.tolist()
        _write_json(self.timing_output / "protocol.json", p)
        _write_json(self.timing_output / "calibration.json", calibration)
        all_rows = []
        for repeat in range(p["robustness_repetitions"]):
            self.update_status("robustness", repeat=repeat + 1, index=0, total=p["robustness_cases"],
                controller_steps={name: 0 for name in p["robustness_variants"]})
            directory = self.timing_output / f"repeat_{repeat + 1}"
            directory.mkdir()
            branches = {name: self.branch(directory / "candidates", cases=p["robustness_cases"])
                        for name in p["robustness_variants"]}
            controllers = {name: self.bundle() for name in branches}
            selectors = {name: TimingSelection(name, calibration["sigma_sec"], p["near_tie_multiplier"])
                         for name in branches}
            previous_updates = dict.fromkeys(branches, 0.0)
            for name, branch in branches.items():
                branch.policy.model.experimental_score_selector = selectors[name]
            if not self.smoke:
                original = self.new / "aot_candidate_schedules/recommended/structured512/candidate_ids.npy"
                if file_hash(original) != file_hash(directory / "candidates/candidate_ids.npy"):
                    raise AssertionError("Robustness AOT candidates differ from the original protocol")
            order = np.random.default_rng(self.raw["seeds"]["method_order"])
            # Rotate run order independently of learner seeds; every run records it.
            names = list(branches)
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            rows = []
            self.warmup(f"robustness_{repeat + 1}")
            for index, entry in enumerate(records):
                self.environment("robustness")
                mkw, context = entry["mkw"], np.asarray(entry["context"])
                solve_context = np.asarray(entry["problem_context"])
                for rank, name in enumerate(order.permutation(names)):
                    branch, bundle = branches[name], controllers[name]
                    active = index >= p["online_activation"]
                    def solver(params):
                        if not active:
                            return _as_feedback(solve_no_rl_case(params=dict(params), mkw=dict(mkw),
                                solver_tol=1e-6, solver_max_iter=50, augment_params=augment_setup_params),
                                include_controller=False)
                        native = bundle.run_case(OnlineSolveCase(mkw=mkw, params=params, solve_tol=1e-6,
                            solve_max_cycles=50, learn=True, explore=True, problem_context=solve_context,
                            record_action_metadata=True, fallback_attempt=lambda: self.fallback(mkw)))
                        check_execution_errors(native)
                        return _as_feedback(native, include_controller=True)
                    started = time.perf_counter()
                    params, native, timing, _fallback, update = run_bandit_step_test_final(
                        policy=branch.policy, parameter_space=branch.parameter_space, problem_context=context,
                        solver_fn=solver, fallback_solver_fn=lambda _: self.fallback(mkw),
                        prev_update_est=previous_updates[name])
                    previous_updates[name] = update
                    branch.policy.model.finish_candidate_schedule_case(max_selections=3)
                    outcome = report_online_outcome(native, bandit_timing=timing)
                    outcome["method_wall_runtime"] = time.perf_counter() - started
                    outcome["completed_residual_norm"] = float(outcome.get("fallback_residual_norm", np.nan)
                        if outcome.get("fallback_used") else outcome["residual_norm"])
                    check_outcome(outcome)
                    outcome["calibration_charge"] = calibration["charge_sec"] if name == "near_tie" and index == 0 else 0.
                    outcome["algorithm_runtime"] += outcome["calibration_charge"]
                    row = {"phase": "robustness", "source": "adaptive_joint", "policy": str(name), "repeat": repeat,
                        "index": index, "execution_rank": rank, "mkw": mkw, "context": context.tolist(),
                        "problem_context": solve_context.tolist(), "params": params, "outcome": outcome,
                        "decision": copy.deepcopy(selectors[name].last)}
                    self.append(directory / "trajectories.jsonl", [row], rows)
                if (index + 1) % 100 == 0:
                    self.update_status("robustness", repeat=repeat + 1, index=index + 1, total=len(records),
                        controller_steps={name: int(bundle.controller.steps) for name, bundle in controllers.items()})
                    write_report(directory, rows, f"Timing stability execution {repeat + 1}")
            for name, branch in branches.items():
                if len(records) > p["online_activation"] and controllers[name].controller.steps == 0:
                    raise AssertionError(f"No RL training occurred in completed robustness branch {name}")
                branch.policy.model.save_mutable_state(directory / f"{name}_setup.npz", metadata={
                    "variant": name, "sigma_sec": calibration["sigma_sec"],
                    "band_multiplier": p["near_tie_multiplier"], "numeric_tolerance": 1e-12})
                controllers[name].save(directory / f"{name}_rl.npz")
                audit_controller(controllers[name].controller)
            write_report(directory, rows, f"Timing stability execution {repeat + 1}", complete=True)
            all_rows.extend(rows)
        write_report(self.timing_output, all_rows, "Timing stability: independent execution repeats", complete=True)

    def run(self):
        configure_smoother_profile(self.raw["solve"]["smoother_profile"])
        self.environment("preparing")
        _write_json(self.output / "protocol.json", self.protocol)
        state = source_state()
        state["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        _write_json(self.output / "source_manifest.json", state)
        self.prepare_sources()
        streams = {name: [{"mkw": m, "context": c.tolist()} for m, c in self.stream(name, self.protocol[count])]
                   for name, count in (("calibration", "calibration_cases"), ("selection", "selection_cases"),
                                       ("evaluation", "evaluation_cases"), ("online", "online_cases"))}
        sets = [set(digest(r["mkw"]) for r in stream) for stream in streams.values()]
        training = {digest(json.loads(line)["mkw"]) for line in (self.new / "trajectories/bandit_default.jsonl").open()}
        if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]) or any(s & training for s in sets):
            raise AssertionError("Fresh phase inputs overlap each other or the development stream")
        _write_json(self.output / "inputs.json", streams)
        calibration = self.calibrate()
        selections, patterns = self.frozen()
        self.online(selections, patterns)
        self.robustness(calibration)
        for name, model in self.sources.items():
            after = self.output / "frozen_sources" / f"{name}_after.npz"
            model.save_mutable_state(after)
            with np.load(self.output / "frozen_sources" / f"{name}_before.npz") as initial, np.load(after) as final:
                if initial.files != final.files:
                    raise AssertionError("Frozen setup checkpoint schema changed")
                for key in initial.files:
                    np.testing.assert_array_equal(initial[key], final[key], err_msg=f"Frozen setup changed: {name}/{key}")
        _write_json(self.output / "complete.json", {"complete": True, "frozen_sources_unchanged": True,
            "source_manifest_sha256": file_hash(self.output / "source_manifest.json"),
            "timing_results": str(self.timing_output)})
        self.update_status("complete", status="complete")

    def run_timing_only(self, previous):
        """Rerun the invalid timing stage without repeating valid attribution work."""
        for phase in ("frozen", "online"):
            if not json.loads((previous / phase / "summary.json").read_text())["complete"]:
                raise ValueError(f"The retained {phase} phase is incomplete")
        configure_smoother_profile(self.raw["solve"]["smoother_profile"])
        self.environment("preparing")
        _write_json(self.output / "protocol.json", self.protocol)
        state = source_state()
        state["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        _write_json(self.output / "source_manifest.json", state)
        _write_json(self.output / "continuation.json", {
            "retained_attribution_results": str(previous),
            "retained_phases": ["frozen", "online"], "timing_restarts_from_scratch": True,
            "reason": "Correct solve-context dimension and require evidence of actual RL training",
            "fresh_calibration": True})
        calibration = self.calibrate()
        self.robustness(calibration)
        _write_json(self.output / "complete.json", {"complete": True, "scope": "corrected timing study only",
            "timing_results": str(self.timing_output), "retained_attribution_results": str(previous)})
        self.update_status("complete", status="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--timing-only-from", type=Path)
    parser.add_argument("--repetitions", type=int, choices=(2, 3))
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.timing_only_from:
        if args.output_root.resolve() == OUTPUT.resolve():
            raise ValueError("Timing-only reruns need a fresh --output-root")
        protocol["timing_output"] = str(args.output_root.resolve() / "runs")
    if args.repetitions is not None:
        protocol["robustness_repetitions"] = args.repetitions
    for path, expected in json.loads(PROTOCOL.with_name("input_manifest.json").read_text()).items():
        if not (ROOT / path).is_file() or file_hash(ROOT / path) != expected:
            raise ValueError(f"Historical input artifact changed or is missing: {path}")
    for run in (protocol["old_run"], protocol["new_run"]):
        for method in ("bandit_default", "bandit_lstdq"):
            assert (ROOT / run / "final_bandit_states" / f"{method}.npz").is_file()
        assert (ROOT / run / "checkpoints/bandit_lstdq_final.npz").is_file()
    if not args.run and not args.smoke:
        print(json.dumps({"valid": True, "protocol": protocol, "native_experiments_executed": 0}, indent=2))
        return
    if args.smoke:
        if args.output_root.resolve() == OUTPUT.resolve():
            raise ValueError("Smoke needs its own --output-root")
        protocol.update(grid=8, calibration_cases=2, selection_cases=2, evaluation_cases=3,
                        timing_repeat_cases=1, online_cases=8, online_activation=3,
                        robustness_cases=8, robustness_repetitions=1, source_candidate_count=512)
    elif subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit source/configuration changes before the timing study")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite study results: {args.output_root}")
    if not args.smoke and (ROOT / protocol["timing_output"]).exists():
        raise FileExistsError("Timing output already exists")
    args.output_root.mkdir(parents=True)
    with (ROOT / "results/paper_final/.native_timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study(protocol, args.output_root, args.smoke)
        try:
            if args.timing_only_from:
                study.run_timing_only(args.timing_only_from.resolve())
            else:
                study.run()
        except BaseException as exc:
            study.update_status(study.status.get("phase", "preparing"), status="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
