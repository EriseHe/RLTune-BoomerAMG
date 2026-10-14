"""Freeze hierarchies and plan independent scan/evaluation measurements."""

from __future__ import annotations

import copy
import random
from pathlib import Path
import shutil

import numpy as np

from experiments.paper_final.common import artifacts as a
from experiments.paper_final.common import frozen_policy as policy
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from experiments.paper_final.online.configuration import ProblemSpec, parse_joint_experiment_config
from experiments.paper_final.online.problem_stream import generate_problem_instances
from experiments.paper_final.online.setup_branches import build_online_linucb_branch
from problems.registry import context_for_setup_method, learning_context_for_setup
from setup.space import SetupConfigurationSpace
from hypre.bindings.config import SMOOTHER_PROFILES

from .configuration import METHODS, LABELS, fixed_name, load_config, policy_library


def make_inputs(cfg):
    if cfg["solver"]["profile"] not in SMOOTHER_PROFILES:
        raise ValueError(f"solver.profile must be one of {SMOOTHER_PROFILES}")
    problem = ProblemSpec.from_mapping(cfg["problem"])
    if cfg.get("inputs_file"):
        rows = a.read(cfg["inputs_file"])
        if isinstance(rows, dict):
            rows = rows["test"]
        if len(rows) != cfg["cases"]:
            raise ValueError("inputs_file length differs from cases")
    else:
        instances = generate_problem_instances(
            T=cfg["cases"], seed=cfg["seeds"]["inputs"], grid_choices=[problem.grid],
            c_min=problem.c_min, c_max=problem.c_max, problem=problem.kind,
            difconv_a=problem.advection, advection_min=problem.advection_min,
            advection_max=problem.advection_max,
        )
        rows = [{"case_id": i, "mkw": mkw, "context": context.tolist()}
                for i, (mkw, context) in enumerate(instances)]
    result = []
    for row in rows:
        if not isinstance(row["case_id"], int) or row["case_id"] < 0:
            raise ValueError("Case IDs must be nonnegative integers")
        context = np.asarray(row["context"], dtype=float)
        if context.ndim != 1 or not np.all(np.isfinite(context)):
            raise ValueError("Invalid problem context")
        result.append({"case_id": row["case_id"], "mkw": row["mkw"],
                       "context": context.tolist(), "input_id": a.digest(row["mkw"])})
    if len({r["case_id"] for r in result}) != len(result) or len({r["input_id"] for r in result}) != len(result):
        raise ValueError("Duplicate evaluation cases")
    return result


def freeze_hierarchies(output, cfg, inputs):
    jobs, audits, sources = [], [], {}
    for source in cfg["training_runs"]:
        seed = source["seed"]
        directory = output / "checkpoints" / f"seed_{seed}"
        directory.mkdir(parents=True)
        raw = source["training_config"]
        spec = parse_joint_experiment_config(raw)
        method = next(m for m in spec.methods if m.name == source["method"])
        a.dump(directory / "experiment_config.json", raw)
        for key, target in (("controller_checkpoint", "controller.npz"),
                            ("encoder", "controller.npz.encoder.json"),
                            ("setup_checkpoint", "setup_original.npz")):
            shutil.copy2(source[key], directory / target)
        # Training cursors refer to the old AOT stream. Only its copied cursor
        # is reset; regression statistics and RNG are retained.
        with np.load(directory / "setup_original.npz", allow_pickle=False) as state:
            if int(state.get("checkpoint_version", np.asarray(1))) != 4:
                raise ValueError(
                    f"Seed {seed}: fresh studies require a current setup checkpoint (version 4). "
                    "Use the released Module 2 configuration and checkpoints."
                )
            copied = {k: state[k].copy() for k in state.files}
        copied["candidate_schedule_cursor"] = np.asarray(0, dtype=np.int64)
        np.savez_compressed(directory / "setup_evaluation.npz", **copied)
        assert a.checkpoint_payload_hash(directory / "setup_original.npz", exclude=("candidate_schedule_cursor",)) == a.checkpoint_payload_hash(directory / "setup_evaluation.npz", exclude=("candidate_schedule_cursor",))
        context_spec = learning_context_for_setup(cfg["problem"]["kind"], method.setup_kind, method.setup_context)
        space = SetupConfigurationSpace.from_mapping(method.setup_space,
                    raw["setup"]["configuration_spaces"][method.setup_space])
        branch, _ = build_online_linucb_branch(
            seed=raw["seeds"]["bandit"] + method.seed_offset,
            action_space_mode=raw["setup"]["action_space"],
            parameter_resolution=raw["setup"]["parameter_resolution"],
            configuration_space=space, solver_tol=cfg["solver"]["tolerance"],
            solver_max_iter=cfg["solver"]["max_cycles"],
            candidate_schedule_dir=output / "candidate_schedules" / f"seed_{seed}",
            candidate_schedule_rounds=len(inputs), candidate_schedule_seed=cfg["seeds"]["candidates"],
            candidate_pool_size=spec.setup.candidate_schedule.pool_size,
            structured_quotas=spec.setup.candidate_schedule.structured_quotas,
            candidate_sampling=method.candidate_sampling, context_dim=context_spec.dimension,
            context_interaction_indices=context_spec.interaction_indices,
        )
        model = branch.policy.model
        model.load_mutable_state(directory / "setup_evaluation.npz")
        model.save_mutable_state(directory / "setup_before.npz")
        for row in inputs:
            context = context_for_setup_method(
                problem_kind=cfg["problem"]["kind"], setup_kind=method.setup_kind,
                matrix_kwargs=row["mkw"], stream_context=np.asarray(row["context"]),
                setup_context=method.setup_context,
            )
            params, _ = branch.policy.select(context=context, parameter_space=branch.parameter_space)
            branch.policy.cancel_pending()
            jobs.append({**row, "seed": seed, "source": method.name, "params": params,
                         "hierarchy_id": a.digest({"mkw": row["mkw"], "params": params,
                                                   "profile": cfg["solver"]["profile"]})})
        model.save_mutable_state(directory / "setup_after.npz")
        excluded = ("candidate_schedule_cursor", "rng_state", "metadata")
        if a.checkpoint_payload_hash(directory / "setup_before.npz", exclude=excluded) != a.checkpoint_payload_hash(directory / "setup_after.npz", exclude=excluded):
            raise ValueError("Setup learning statistics changed during frozen selection")
        # Decode and load each controller before any expensive native scans.
        bundle = policy.build_bundle(raw, method.name, directory / "controller.npz", beta=cfg["rl_beta"])
        sources[str(seed)] = {"method": method.name, "directory": str(directory.relative_to(output)),
                             "effective_beta": bundle.controller.spec.uncertainty_beta,
                             "original_files": {k: {"path": source[k], "sha256": a.file_hash(source[k])}
                                                for k in ("config", "controller_checkpoint", "encoder", "setup_checkpoint")}}
        audits.append({"seed": seed, "source": method.name, "choices": len(inputs),
                       "setup_learning_unchanged": True})
    return jobs, audits, sources


def scan_jobs(base_jobs, cfg, extra_cases):
    names = [fixed_name(w) for w in cfg["weights"]]
    rng = random.Random(cfg["seeds"]["order"])
    jobs = []
    for row in base_jobs:
        repeats = cfg["scan_repetitions"] + (cfg["scan_extra_repetitions"] if row["case_id"] in extra_cases else 0)
        for repeat in range(repeats):
            order = names.copy()
            rng.shuffle(order)
            jobs.append({**row, "repeat": repeat, "policy_order": order})
    rng.shuffle(jobs)
    return jobs


def evaluation_jobs(base_jobs, selections, cfg):
    rng = random.Random(cfg["seeds"]["order"] + 1)
    jobs = []
    for row in base_jobs:
        choice = selections[str(row["seed"])]
        methods = {"reference": fixed_name(1), "fixed": choice["global"],
                   "oracle": choice["per_instance"][str(row["case_id"])],
                   "periodic": "periodic", "rl": "rl_frozen"}
        # Undefined all-success comparators stay missing, never silently become
        # a fastest failed weight. Other policies can still be evaluated.
        for repeat in range(cfg["evaluation_repetitions"]):
            order = sorted({p for p in methods.values() if p is not None})
            rng.shuffle(order)
            jobs.append({**row, "repeat": repeat, "method_policies": methods,
                         "policy_order": order})
    rng.shuffle(jobs)
    return jobs


def prepare(config, output):
    cfg = load_config(config)
    inputs = make_inputs(cfg)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("analysis", "progress", "raw/scan", "raw/test", "provenance/source"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    a.dump(output / "configuration.json", cfg)
    jobs, audits, sources = freeze_hierarchies(output, cfg, inputs)
    count = round(len(inputs) * float(cfg["scan_repeat_fraction"]))
    extra = sorted(random.Random(cfg["seeds"]["repeats"]).sample([r["case_id"] for r in inputs], count))
    protocol = {
        "module": "05", "format": "matched-policy-fresh-v1", "name": cfg["name"],
        "methods": list(METHODS), "labels": LABELS, "training_seeds": [s["seed"] for s in cfg["training_runs"]],
        "problem": cfg["problem"], "test_cases": len(inputs), "repetitions_per_case": cfg["evaluation_repetitions"],
        "scan_repetitions": cfg["scan_repetitions"], "scan_extra_repetitions": cfg["scan_extra_repetitions"],
        "scan_extra_repeat_case_ids": extra, "weights": cfg["weights"], "policies": policy_library(cfg),
        "sources": sources, "max_workers": cfg["workers"], "tolerance": cfg["solver"]["tolerance"],
        "max_cycles": cfg["solver"]["max_cycles"], "profile": cfg["solver"]["profile"],
        "seeds": cfg["seeds"], "created_at": a.now(),
        "selection": "Fresh complete fixed-weight scans, mean native continuation cost; all-success global weight and successful per-case weights; deterministic weight-name ties",
        "recovery": "One primary matched hierarchy, then one fresh Default fallback on numerical failure; no setup reselection or learning",
        "evaluation": "Freeze scan selections, independently retime five policy roles; share identical execution cells; all failures and recovery costs retained",
    }
    a.dump(output / "protocol.json", protocol)
    a.dump(output / "inputs.json", {"test": inputs})
    a.dump(output / "jobs_base.json", jobs)
    a.dump(output / "jobs_scan.json", scan_jobs(jobs, cfg, extra))
    a.dump(output / "setup_freeze_audit.json", audits)
    hashes = {name: a.file_hash(a.ROOT / name) for name in sorted(execution_source_paths())}
    for name in hashes:
        dest = output / "provenance/source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.ROOT / name, dest)
    a.dump(output / "source_manifest.json", hashes)
    environment = policy.environment_check(output)
    files = ["configuration.json", "protocol.json", "inputs.json", "jobs_base.json", "jobs_scan.json", "setup_freeze_audit.json", "source_manifest.json"]
    files += [str(p.relative_to(output)) for p in (output / "checkpoints").rglob("*") if p.is_file()]
    a.dump(output / "prepared.json", {"at": a.now(), "files_sha256": {name: a.file_hash(output / name) for name in files},
                                      "execution_environment": policy.environment_identity(environment)})
    a.dump(output / "status.json", {"status": "prepared", "native_solves_executed": 0})
    a.emit(output, "Fresh Module 2 prepared", seeds=protocol["training_seeds"], cases=len(inputs),
           scan_trials=sum(len(j["policy_order"]) for j in a.read(output / "jobs_scan.json")))
    return protocol
