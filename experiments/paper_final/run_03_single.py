"""Run one explicit activation configuration, with artifacts inside its run folder."""
from experiments.paper_final.run_03_activation import OUTPUT, ROOT, audit_run

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from online_td_experiment_common import _write_json
from run_paper_final import THREAD_ENV, file_hash, require_disk_space, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    raw = json.loads(config.read_text())
    grid_label = "×".join(str(n) for n in raw["problem"]["grid"])
    destination = (ROOT / raw["output_dir"]).resolve()
    if destination.parent != OUTPUT or destination.name != raw["name"]:
        raise ValueError("Single activation runs must have a named folder directly under 03_activation")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {destination}")

    # Keep metadata outside the destination until the native overwrite guard
    # has passed. Renaming the logs preserves their open file descriptors.
    staging = OUTPUT / f".{destination.name}.launch"
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (staging / "launch.json").exists() or destination.exists():
            raise FileExistsError("This run has already been launched")
        require_disk_space(OUTPUT, 4)
        provenance, logs = staging / "provenance", staging / "logs"
        for path in (provenance, logs, staging / "audits"):
            path.mkdir(exist_ok=True)
        source = source_state()
        entry = {"name": raw["name"], "config_sha256": file_hash(config),
                 "stream_sha256": raw["stream"]["expected_sha256"]}
        _write_json(provenance / "source_manifest.json", {
            "source": source, "python": sys.version,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "thread_environment": THREAD_ENV,
        })
        shutil.copy2(config, provenance / "requested_config.json")
        (provenance / "source_changes.patch").write_bytes(subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "experiments", "hypre", "problems", "setup", "solve"], cwd=ROOT))
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
        for relative in untracked:
            if relative in source["files"]:
                target = provenance / "untracked_source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
        environment = dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        for name in ("SETUP_RELAX_TYPE", "SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS",
                     "AMG_RELAX_TYPE", "AMG_COARSE_RELAX_TYPE", "AMG_CYCLE_TYPE"):
            environment.pop(name, None)
        command = [sys.executable, "-u", str(ROOT / "experiments/joint/solve_control/run_joint_experiment.py"),
                   "--config", str(config), "--output-dir", str(destination)]
        started = time.perf_counter()
        with (logs / "native.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            _write_json(staging / "launch.json", {
                **entry, "supervisor_pid": os.getpid(), "native_pid": child.pid,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                "source_sha256": source["sha256"], "supervisor_sha256": file_hash(Path(__file__)),
                "methods": [m["id"] for m in raw["methods"]], "requested_runs": [raw["name"]],
            })
            while child.poll() is None and not (destination / "trajectories").is_dir():
                time.sleep(.1)
            destination.mkdir(exist_ok=True)
            for artifact in staging.iterdir():
                target = destination / artifact.name
                if target.exists():
                    raise FileExistsError(f"Refusing to replace runner artifact: {target}")
                artifact.rename(target)
            staging.rmdir()
            (destination / "REPORT.md").write_text(
                f"# 03 — {raw['name']}\n\nRunning one development seed.\n\n"
                f"Branches: {', '.join(m['id'] for m in raw['methods'])}.\n\n"
                "[Progress](progress.json) · [Native log](logs/native.log) · "
                "[Preflight](audits/preflight/) · [Provenance](provenance/)\n\n"
                "The supervisor will audit the completed run and generate the usual figures.\n")
            print(f"Running {raw['name']}; native PID {child.pid}; logs: {destination / 'logs'}", flush=True)
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    progress = destination / "progress.json"
                    if progress.exists():
                        try:
                            value = json.loads(progress.read_text())
                            print(json.dumps({"elapsed_sec": round(time.perf_counter() - started),
                                              "progress": value.get("completed_online_instances")}), flush=True)
                        except json.JSONDecodeError:
                            pass
        elapsed = time.perf_counter() - started
        _write_json(destination / "exit.json", {"returncode": code,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": elapsed})
        if code:
            (destination / "REPORT.md").write_text(
                f"# 03 — {raw['name']}\n\nNative run stopped with code {code}; incomplete.\n\n"
                "See [native log](logs/native.log) and [exit record](exit.json).\n")
            raise SystemExit(code if code > 0 else 128 - code)
        try:
            audited = audit_run(destination, raw)
            _write_json(destination / "audits/final_audit.json", audited)
            with (destination / "logs/plots.log").open("x") as log:
                subprocess.run([
                    sys.executable, str(ROOT / "experiments/joint/solve_control/generate_joint_experiment_plots.py"),
                    "--result-dir", str(destination), "--rolling-window", str(raw["reporting"]["rolling_window"]),
                ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            figures = sorted((destination / "figures").glob("*.png"))
            if not figures:
                raise RuntimeError("Plot generator produced no figures")
            (destination / "figures/README.md").write_text("\n".join([
                "# Figures", "", "[Run report](../REPORT.md)", "",
                f"Shared settings: LinUCB (7D), LSTDQ v3, {grid_label} diffusion–advection.", "",
                *[f"- [{p.stem.replace('five_family_', '').replace('_', ' ')}]({p.name})" for p in figures], "",
            ]))
            cases, tail = raw["stream"]["online_cases"], min(raw["stream"]["online_cases"], 1000)
            comparison = audited["comparison_reference"]
            lines = [f"# 03 — {raw['name']}", "", "Completed and audited: one development seed.", "",
                     "[Figures](figures/README.md) · [Audit](audits/final_audit.json) · "
                     "[Logs](logs/) · [Provenance](provenance/)", "",
                     f"{grid_label} diffusion–advection; LinUCB v4 (7D, including intercept); recursive LSTDQ v3.",
                     "Each branch learns setup throughout. RL starts on the input after its boundary.", "",
                     f"Shared-prefix source: {audited['shared_prefix_source']}. "
                     f"Physical primary executions: {audited['physical_primary_executions']:,}; recovery attempts are additional.",
                     "Prefix costs count once in each logical trajectory. All failed/recovered attempts remain in the totals.",
                     "Recorded cost is not an all-success completion time when unrecovered failures occur.", "",
                     f"| Branch | All {cases} cost (s) | Difference vs {comparison} (s) | Unrecovered | Last {tail} cost (s) |",
                     "|---|---:|---:|---:|---:|"]
            for method in raw["methods"]:
                name = method["id"]
                value, last = audited["windows"][f"all_{cases}"][name], audited["windows"][f"last_{tail}"][name]
                lines.append(f"| {name} | {value['totals_sec']['end_to_end_runtime']:.3f} | "
                             f"{value[f'difference_vs_{comparison}_sec']:.3f} | {value['unrecovered_failure_count']} | "
                             f"{last['totals_sec']['end_to_end_runtime']:.3f} |")
            lines.extend(["", "A negative difference means lower recorded cost. Compare branches within this seed.",
                          "The earlier s1 run used a different input stream and branch roster; these are development",
                          "diagnostics, not a formal seed-robustness or fixed-hierarchy policy comparison.", "",
                          "Full-state prefix forks, paired inputs, activation boundaries, failure/recovery accounting,",
                          "and final inverse/batch identities passed the completion audit.", ""])
            (destination / "REPORT.md").write_text("\n".join(lines))
            _write_json(destination / "summary.json", {
                "complete": True, "requested_runs": [raw["name"]], "development_only": True,
                "runs": [{"name": raw["name"], **audited}],
            })
            _write_json(destination / "complete.json", {
                **entry, "subprocess_elapsed_sec": elapsed, "figures": len(figures),
                "unrecovered_failures": audited["unrecovered_failures"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            _write_json(destination / "audits/postprocess_error.json", {"error": repr(exc), "native_returncode": code})
            (destination / "REPORT.md").write_text(
                f"# 03 — {raw['name']}\n\nNative run finished; postprocessing needs attention.\n\n"
                "See [error record](audits/postprocess_error.json) and [logs](logs/).\n")
            raise
        print(f"Single-seed experiment, completion audit and {len(figures)} figures finished.", flush=True)


if __name__ == "__main__":
    main()
