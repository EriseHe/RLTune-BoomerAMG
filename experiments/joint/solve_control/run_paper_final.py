"""Validate or sequentially run the prespecified PAPER_FINAL Modules 1 and 2.

The default is validation only. Native experiments require the explicit --run
flag and still execute through run_joint_experiment.py, without a second runner.
"""
from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.joint.solve_control.composable_joint_4k import build_composable_solve_runtime
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_experiment_plotting import _compact_method_labels
from experiments.joint.solve_control.joint_online_common import _build_paired_instance_stream
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from problems.registry import context_for_setup_method, learning_context_for_setup
from experiments.joint.solve_control.run_joint_experiment import _validate_resolved
from setup.space import DEFAULT_SETUP_PARAMS


ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "experiments/paper_final/04_online/20260920_formal/suite.json"
DEFAULT_OUTPUT = ROOT / "results/paper_final/04_online"
METHOD_LABELS = {
    "default_setup_default_solve": "Default",
    "bandit_default": "LinUCB",
    "bandit_lstdq": "LinUCB–LSTDQ",
}
CONTEXTS = {"diffusion": "diffusion3d", "diffusion_advection": "canonical_no_c_mean"}
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path = SUITE):
    manifest = json.loads(path.read_text())
    entries = manifest["runs"]
    if not set(manifest.get("allow_unrecovered_families", [])).issubset(CONTEXTS):
        raise ValueError("Unknown family in the unrecovered-failure policy")
    seeds = manifest["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (1, 2, 3) for seed in seeds):
        raise ValueError("PAPER_FINAL requires distinct prescribed replicate labels in 1..3")
    families = manifest.get("families", list(CONTEXTS))
    if not families or len(set(families)) != len(families) or not set(families).issubset(CONTEXTS):
        raise ValueError("PAPER_FINAL requires distinct known PDE families")
    grids = manifest.get("grids", [40, 60, 80])
    if not grids or len(set(grids)) != len(grids) or not set(grids).issubset({40, 60, 80}):
        raise ValueError("PAPER_FINAL requires distinct prescribed grids in 40/60/80")
    caps = manifest.get("caps")
    if caps is not None and (not caps or any(type(cap) is not int or cap <= 0 for cap in caps)
                             or len(set(caps)) != len(caps)):
        raise ValueError("PAPER_FINAL requires distinct positive integer cycle caps")
    expected = {(family, n, seed, cap) for family in families
                for n in grids for seed in seeds for cap in (caps or [None])}
    if len(entries) != len(expected) or {(e["family"], e["grid"], e["seed"], e.get("cap")) for e in entries} != expected:
        raise ValueError("PAPER_FINAL requires the prescribed families × grids × seeds")
    if len({e["name"] for e in entries}) != len(expected):
        raise ValueError("PAPER_FINAL run names must be unique")
    configs = []
    for entry in entries:
        config_path = path.parent / entry["config"]
        if file_hash(config_path) != entry["config_sha256"]:
            raise ValueError(f"Prespecified config changed: {config_path}")
        raw = json.loads(config_path.read_text())
        if raw["name"] != entry["name"] or raw["stream"]["expected_sha256"] != entry["stream_sha256"]:
            raise ValueError(f"Config/manifest mismatch: {config_path}")
        configs.append((entry, config_path, raw))
    return manifest, configs


def validate_suite(path: Path = SUITE) -> dict:
    manifest, configs = load_suite(path)
    reports = []
    # Preserve the settings of the completed encoder-corrected study.
    previous = json.loads((Path(__file__).with_name("configs") / "paper_test_n60_context_activation_seed_d.json").read_text())
    for entry, _config_path, raw in configs:
        mode = CONTEXTS[entry["family"]]
        dimension = 4 if entry["family"] == "diffusion" else 7
        spec = parse_joint_experiment_config(raw)
        runtime = runtime_config_from_spec(spec)
        validation = _validate_resolved(runtime)
        expected_solve = dict(previous["solve"])
        expected_solve["max_cycles"] = entry.get("cap", manifest.get("cycle_caps", {}).get(
            entry["family"], previous["solve"]["max_cycles"]
        ))
        if raw["setup"] != previous["setup"] or raw["solve"] != expected_solve:
            raise ValueError("PAPER_FINAL must retain the agreed setup and solve settings")
        if manifest.get("caps"):
            feedback = manifest["failure_feedback"]
            if validation["failure_feedback"] != {
                "mode": feedback["mode"], "penalty_sec": feedback.get("penalty_sec")
            }:
                raise ValueError("Cap comparisons must keep the failure objective fixed")
        expected_problem = {
            "kind": "scalar_anisotropic_" + entry["family"],
            "grid": [entry["grid"]] * 3, "c_min": 1, "c_max": 1000,
            **({"advection": [0, 0, 0]} if entry["family"] == "diffusion"
               else {"advection_min": 1, "advection_max": 1000}),
        }
        if raw["problem"] != expected_problem:
            raise ValueError("Unexpected PAPER_FINAL PDE definition")
        if runtime.train_cases != 5000 or runtime.online_cases != 5000 or runtime.warmup_cases != 0:
            raise ValueError("The 1000-problem setup prefix must be included in the 5000 problems")
        offset = manifest["seed_offsets"][entry["seed"] - 1]
        if raw["seeds"] != {k: v + offset for k, v in previous["seeds"].items()}:
            raise ValueError("Seed tuple differs from the prespecified replicate")
        expected_stream = dict(previous["stream"])
        for key in ("seed_groups", "shuffle_seeds"):
            expected_stream[key] = [v + offset for v in previous["stream"][key]]
        expected_stream["expected_sha256"] = entry["stream_sha256"]
        if raw["stream"] != expected_stream:
            raise ValueError("Input-stream sampling differs from the prespecified replicate")
        if [m.name for m in runtime.methods] != list(METHOD_LABELS):
            raise ValueError("Unexpected PAPER_FINAL method roster")
        baseline, bandit, joint = runtime.methods
        if baseline.setup_kind != "default" or baseline.solve_kind != "default":
            raise ValueError("Default must use reference setup and solve")
        for method in (bandit, joint):
            if (method.setup_kind != "linucb" or method.setup_context != mode
                    or method.seed_offset != 0 or method.setup_warmup_cases != 0
                    or method.setup_space != "recommended" or method.candidate_sampling != "structured512"
                    or method.solve_activation is not None):
                raise ValueError("Unexpected setup/context/activation contract")
        if (bandit.solve_kind != "default" or joint.solve_kind != "recursive_lstdq_v3"
                or joint.solve_activation_case != 1000 or joint.solve_context != mode
                or joint.solve_tolerance != 1e-6):
            raise ValueError("LSTDQ must start on problem 1001 with the shared problem context")
        bundle = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles[joint.name]
        contract = learning_context_for_setup(raw["problem"]["kind"], "linucb", mode)
        if contract.dimension != dimension or bundle.encoder.feature_dim != 27 + dimension:
            raise ValueError("Unexpected problem or solve-state dimension")
        stream, _ = _build_paired_instance_stream(runtime)
        for index in (0, 1, 2499, 4999):
            mkw, context = stream[index]
            setup_view = context_for_setup_method(
                problem_kind=raw["problem"]["kind"], setup_kind="linucb", setup_context=mode,
                matrix_kwargs=mkw, stream_context=context,
            )
            state = bundle.encoder.encode(
                mkw=mkw, problem_context=context, initial_residual=1., residual=.1,
                previous_residual=.2, cycle=2, last_weight=2.5, last_cycle_time=.003,
                setup_params=DEFAULT_SETUP_PARAMS,
            )
            np.testing.assert_array_equal(setup_view, np.r_[state[0], state[7:7 + dimension - 1]])
            if not np.all(np.isfinite(state)):
                raise ValueError("Nonfinite solve state")
        labels = _compact_method_labels({"methods": list(METHOD_LABELS),
                                        "method_specs": [asdict(m) for m in runtime.methods]})
        if labels != METHOD_LABELS:
            raise ValueError("Paper figure labels do not match the method roster")
        reports.append({"name": entry["name"], "stream_sha256": validation["stream"]["sha256"],
                        "max_cycles": runtime.max_cycles, "failure_feedback": validation["failure_feedback"],
                        "problem_context_fields": list(bundle.encoder.problem_context_fields),
                        "problem_context_dim": dimension, "solve_state_dim": bundle.encoder.feature_dim,
                        "solve_joint_dim": bundle.controller.joint_dim, "rl_first_problem": 1001})
    streams = {}
    for (entry, _path, _raw), report in zip(configs, reports):
        key = (entry["family"], entry["grid"], entry["seed"])
        streams.setdefault(key, set()).add(report["stream_sha256"])
    if any(len(hashes) != 1 for hashes in streams.values()):
        raise ValueError("All caps must use the same paired input stream")
    if len({r["stream_sha256"] for r in reports}) != len(streams):
        raise ValueError("Different family/grid/seed groups must have distinct stream hashes")
    return {"suite": manifest["name"], "valid": True, "native_experiments_executed": 0,
            "runs": reports, "free_disk_gib": shutil.disk_usage(ROOT).free / 2**30}


def source_state() -> dict:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
         "experiments", "hypre", "problems", "setup", "solve"], cwd=ROOT,
    ).decode().split("\0")
    files = {p: file_hash(ROOT / p) for p in sorted(set(paths))
             if p and (ROOT / p).is_file() and Path(p).suffix in {".py", ".c", ".h", ".json", ".sh", ".command"}}
    for path in (ROOT / "hypre/interfaces/libamg_runtime.dylib",
                 ROOT / "hypre/install/lib/libHYPRE.3.0.0.dylib"):
        if path.exists():
            files[str(path.relative_to(ROOT))] = file_hash(path)
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files}


def require_disk_space(output_root: Path, remaining: int) -> None:
    parent = output_root
    while not parent.exists():
        parent = parent.parent
    # Planning allowance, not a hard bound on trajectory sizes: 256 MiB/run
    # plus 1 GiB reserve. Rechecked before each run; no old files are removed.
    required = (1 + remaining / 4) * 2**30
    available = shutil.disk_usage(parent).free
    if available < required:
        raise RuntimeError(f"Insufficient disk space: {available / 2**30:.2f} GiB free; "
                           f"allow at least {required / 2**30:.2f} GiB for {remaining} remaining runs")


def run_suite(path: Path, output_root: Path, *, suite_loader=None, run_audit=None,
              report_writer=None, family: str | None = None, seed: int | None = None) -> None:
    import fcntl
    from experiments.joint.solve_control.analyze_paper_final import audit_run, write_reports

    suite_loader = load_suite if suite_loader is None else suite_loader
    run_audit = partial(audit_run, require_completed_residuals=True) if run_audit is None else run_audit
    report_writer = write_reports if report_writer is None else report_writer
    manifest, configs = suite_loader(path)
    total_configs = len(configs)
    configs = [(entry, config, raw) for entry, config, raw in configs
               if (family is None or entry["family"] == family)
               and (seed is None or entry["seed"] == seed)]
    if not configs:
        raise ValueError("No prescribed runs match the requested family and seed")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This experiment suite is already running") from exc
        current = source_state()
        provenance_path = output_root / "source_manifest.json"
        if provenance_path.exists():
            if json.loads(provenance_path.read_text())["source"]["sha256"] != current["sha256"]:
                raise RuntimeError("Source/config/binary changed since suite launch; refusing to mix versions")
            if json.loads((output_root / "suite.json").read_text()) != manifest:
                raise RuntimeError("Suite manifest changed since launch")
        else:
            require_disk_space(output_root, len(configs))
            _write_json(output_root / "suite.json", manifest)
            _write_json(provenance_path, {
                "source": current, "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "thread_environment": THREAD_ENV,
            })
            patch = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD", "--", "experiments", "hypre", "problems", "setup", "solve"], cwd=ROOT)
            (output_root / "source_changes.patch").write_bytes(patch)
            untracked = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
            for relative in untracked:
                if relative in current["files"]:
                    target = output_root / "untracked_source" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / relative, target)
        environment = dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        for name in ("SETUP_RELAX_TYPE", "SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS",
                     "AMG_RELAX_TYPE", "AMG_COARSE_RELAX_TYPE", "AMG_CYCLE_TYPE"):
            environment.pop(name, None)
        logs = output_root / "logs"
        logs.mkdir(exist_ok=True)
        _write_json(output_root / "selection.json", {
            "family": family, "seed": seed, "runs": [entry["name"] for entry, _path, _raw in configs],
        })
        for index, (entry, config_path, raw) in enumerate(configs):
            destination = output_root / entry["name"]
            marker = output_root / f"{entry['name']}.complete.json"
            if marker.exists():
                completed = json.loads(marker.read_text())
                if (completed["config_sha256"] != entry["config_sha256"]
                        or completed["stream_sha256"] != entry["stream_sha256"]):
                    raise RuntimeError(f"Completion marker mismatch: {marker}")
                if (run_audit(destination, raw)["unrecovered_failures"]
                        and entry["family"] not in manifest.get("allow_unrecovered_families", [])):
                    raise RuntimeError("Existing run has unrecovered failures; inspect before continuing")
                print(f"Verified completed run {index + 1}/{len(configs)}: {entry['name']}", flush=True)
                continue
            log_path = logs / f"{entry['name']}.log"
            if destination.exists() or log_path.exists() or (output_root / log_path.name).exists():
                raise RuntimeError(f"Incomplete/existing run requires inspection; refusing overwrite: {destination}")
            require_disk_space(output_root, len(configs) - index)
            if source_state()["sha256"] != current["sha256"]:
                raise RuntimeError("Source/config/binary changed during suite execution")
            command = [sys.executable, "-u", str(Path(__file__).with_name("run_joint_experiment.py")),
                       "--config", str(config_path), "--output-dir", str(destination)]
            started = time.perf_counter()
            print(f"Starting {index + 1}/{len(configs)}: {entry['name']} (log: {log_path})", flush=True)
            with log_path.open("x") as handle:
                subprocess.run(command, cwd=ROOT, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            audited = run_audit(destination, raw)
            _write_json(marker, {"config_sha256": entry["config_sha256"],
                                 "stream_sha256": entry["stream_sha256"],
                                 "subprocess_elapsed_sec": time.perf_counter() - started,
                                 "unrecovered_failures": audited["unrecovered_failures"]})
            report_writer(path, output_root, allow_partial=True)
            if (audited["unrecovered_failures"]
                    and entry["family"] not in manifest.get("allow_unrecovered_families", [])):
                raise RuntimeError("Run completed with unrecovered failures; inspect before continuing")
            if audited["unrecovered_failures"]:
                print(f"Retained {audited['unrecovered_failures']} unrecovered outcomes in {entry['name']}; "
                      "continuing under the prescribed failure-reporting policy.", flush=True)
        report_writer(path, output_root, allow_partial=len(configs) < total_configs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Start the selected native experiments sequentially")
    mode.add_argument("--validate-only", action="store_true", help="Default: validate without native solves")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--family", choices=tuple(CONTEXTS), help="Run only this PDE family")
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), help="Run only this prescribed replicate")
    args = parser.parse_args(argv)
    report = validate_suite(args.suite.resolve())
    if args.validation_report:
        _write_json(args.validation_report, report)
    print(json.dumps(report, indent=2), flush=True)
    if args.run:
        run_suite(args.suite.resolve(), args.output_root.resolve(), family=args.family, seed=args.seed)


if __name__ == "__main__":
    main()
