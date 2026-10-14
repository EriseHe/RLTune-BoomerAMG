"""Run three fresh seed-5 studies with immediate T/2T setup feedback.

Type-2 beta 2/5/6, then Module 04 diffusion beta 2/3/4 and beta 5/6/7.
Every group includes both controls and its own freshly executed setup prefix.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys

from experiments.paper_final import run_06_beta_sweep as sweep
from experiments.paper_final.common.artifacts import (
    dump, emit, file_hash, now, read, write_csv,
)

STAGES = {
    "stage_1_type2_beta_2_5_6": (2.0, 5.0, 6.0),
    "stage_2_diffusion_beta_2_3_4": (2.0, 3.0, 4.0),
    "stage_3_diffusion_beta_5_6_7": (5.0, 6.0, 7.0),
}
METRIC = "Sum end_to_end_runtime over all 5000 inputs, including failures, all attempts, fallback and learner overhead"


def validate_stage(protocol, values):
    if protocol["betas"] != sweep.branch_betas(values) or protocol["baselines"] != sweep.BASELINES:
        raise ValueError("Each stage must have the requested three betas and both controls")
    if (protocol["cases"], protocol["prefix_cases"], protocol["joint_cases"]) != (5000, 1000, 4000):
        raise ValueError("Each stage requires the full 1000+4000 budget")
    if (protocol["learned_attempt_budget"], protocol["default_fallback_budget"],
        protocol["total_attempt_budget"], protocol["rl_recovery_cost"]) != (3, 1, 4, False):
        raise ValueError("Expected three learned attempts plus default, with no RL recovery cost")
    if protocol["setup_feedback"] != "attempt_2T":
        raise ValueError("Expected immediate T/2T setup feedback")
    if "reused_prefix" in protocol:
        raise ValueError("Every stage must execute a fresh prefix")


def prepare(output, *, config_path=sweep.CONFIG, diffusion_config=sweep.DIFFUSION_CONFIG):
    output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for name, values in STAGES.items():
        # Deliberately never pass prefix_from: each native prefix runs afresh.
        stage_config = config_path if name == next(iter(STAGES)) else diffusion_config
        protocol = sweep.prepare(output / name, betas=values, config_path=stage_config)
        validate_stage(protocol, values)
        prepared[name] = dict(
            betas=list(values), protocol_sha256=file_hash(output / name / "protocol.json"),
            source_manifest_sha256=file_hash(output / name / "source_manifest.json"),
            inputs_sha256=file_hash(output / name / "inputs.jsonl"),
        )
    diffusion_stages = list(STAGES)[1:]
    if len({prepared[name]["inputs_sha256"] for name in diffusion_stages}) != 1:
        raise ValueError("Diffusion input streams differ")
    if len({stage["source_manifest_sha256"] for stage in prepared.values()}) != 1:
        raise ValueError("Stages have different source snapshots")
    plan = dict(schema_version=2, prepared_at=now(), stages=prepared,
                metric=METRIC, unconditional=True, setup_feedback="attempt_2T",
                prefix_reuse=False, controls="Default and LinUCB-only in each stage",
                native_solves_executed_during_preparation=0)
    dump(output / "sequence_protocol.json", plan)
    dump(output / "status.json", dict(status="prepared", experiment_started=False,
                                     native_solves_executed=0))
    (output / "RUN_INSTRUCTIONS.txt").write_text(
        "Three fresh studies, in order, with no performance gate:\n"
        + "\n".join(STAGES) + "\n"
        "Each group: 5000 identical inputs per method; default and LinUCB-only controls.\n"
        "Fresh common 1000-input setup prefix, then 4000 continuation inputs.\n"
        "Three learned attempts plus default; immediate T/2T setup feedback; no RL recovery cost.\n"
        f'From {sweep.ROOT}:\n{sys.executable} -u -m experiments.paper_final.run_06_beta_sequence --run --output "{output}"\n'
    )
    return plan


def validate_prepared(output):
    if read(output / "status.json")["status"] != "prepared":
        raise ValueError("Only a prepared, unstarted sequence may be launched")
    plan = read(output / "sequence_protocol.json")
    if set(plan["stages"]) != set(STAGES) or not plan["unconditional"] or plan["metric"] != METRIC:
        raise ValueError("Sequence plan changed")
    current_source, current_native = sweep.source_hashes(), sweep.native_hashes()
    for name, values in STAGES.items():
        directory, expected = output / name, plan["stages"][name]
        protocol = read(directory / "protocol.json")
        validate_stage(protocol, values)
        if read(directory / "status.json")["status"] != "prepared" or (directory / "prefix").exists():
            raise ValueError("A sequence stage has already started or contains an old prefix")
        for filename, key in (("protocol.json", "protocol_sha256"),
                              ("source_manifest.json", "source_manifest_sha256"),
                              ("inputs.jsonl", "inputs_sha256")):
            if file_hash(directory / filename) != expected[key]:
                raise ValueError(f"Prepared stage artifact changed: {name}/{filename}")
        if current_source != read(directory / "source_manifest.json") or current_native != protocol["native_sha256"]:
            raise ValueError("Source or native libraries changed after preparation")
        for branch in [*protocol["betas"], *protocol["baselines"]]:
            if file_hash(directory / "configs" / f"{branch}.json") != protocol["preparation_checks"][branch]["config_sha256"]:
                raise ValueError("Prepared branch configuration changed")
    return plan


def prefix_check(output, name):
    directory = output / name
    result = read(directory / "prefix" / "result.json")
    if not result["complete"] or result["cases"] != 1000:
        raise ValueError("The 1000-input setup check is incomplete")
    if result["controller"]["steps"] or result["controller"]["episodes"]:
        raise ValueError("RL was trained in the setup-only prefix")
    trajectory = directory / "prefix" / "trajectory.jsonl"
    if file_hash(trajectory) != result["trajectory_sha256"]:
        raise ValueError("Prefix trajectory changed")
    import json
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    if len(records) != 1000 or any(r["rl_active"] for r in records):
        raise ValueError("Invalid setup-only prefix records")
    checkpoint = dict(stage=name, checked_at=now(), passed=True,
                      controller_steps=0, controller_episodes=0,
                      trajectory_sha256=result["trajectory_sha256"],
                      elapsed_sec=result["elapsed_sec"], **sweep.metrics(records))
    dump(directory / "checkpoint_1000.json", checkpoint)
    emit(output, "Fresh 1000-input setup prefix checked", stage=name,
         total_sec=checkpoint["total_sec"], failures=checkpoint["failures"],
         fallbacks=checkpoint["fallbacks"], rl_steps=0)


def completed_totals(directory, values):
    status, report = read(directory / "status.json"), read(directory / "comparison.json")
    if status["status"] != "complete" or not status.get("audits_passed"):
        raise ValueError("The stage has not completed all worker audits")
    if not report["complete"] or not report["audits_passed"]:
        raise ValueError("Comparison is incomplete or unaudited")
    validate_stage(report["protocol"], values)
    expected = {*sweep.branch_betas(values), *sweep.BASELINES}
    totals = {}
    for row in report["rows"]:
        if row["window"] != "all_5000":
            continue
        name = row["branch"]
        if name in totals or name not in expected:
            raise ValueError("Duplicate or unexpected full-run comparison row")
        if row["cases"] != 5000:
            raise ValueError("A method has not finished all 5000 inputs")
        if not math.isfinite(row["total_sec"]) or row["total_sec"] < 0:
            raise ValueError("Invalid full-run total time")
        totals[name] = row
    if set(totals) != expected:
        raise ValueError("Full-run totals must include all three betas and both controls")
    return totals


def write_report(output, totals_by_stage):
    rows = [dict(stage=name, **row) for name, totals in totals_by_stage.items() for row in totals.values()]
    write_csv(output / "sequence_comparison.csv", rows)
    lines = ["Seed-5 T/2T setup feedback study: type-2 and original Module 04 diffusion.", METRIC + ".", ""]
    for row in rows:
        lines.append(f"{row['stage']} / {row['branch']}: total={row['total_sec']:.3f}s, "
                     f"failures={row['failures']}, fallbacks={row['fallbacks']}")
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n")


def launch(output):
    validate_prepared(output)
    totals_by_stage = {}
    active = None
    try:
        for active, values in STAGES.items():
            dump(output / "status.json", dict(status="running", active_stage=active,
                 completed_stages=list(totals_by_stage), started_at=now()))
            emit(output, "Starting fresh five-method group", stage=active, betas=list(values))
            sweep.launch(output / active, prefix_completed=lambda _, name=active: prefix_check(output, name))
            totals_by_stage[active] = completed_totals(output / active, values)
            write_report(output, totals_by_stage)
            emit(output, "Group completed and audited", stage=active,
                 totals={name: {key: row[key] for key in ("total_sec", "failures", "fallbacks")}
                         for name, row in totals_by_stage[active].items()})
        dump(output / "status.json", dict(status="complete", completed_at=now(), audits_passed=True,
             completed_stages=list(totals_by_stage)))
        emit(output, "All three groups completed", completed_stages=list(totals_by_stage))
    except BaseException as error:
        dump(output / "status.json", dict(status="interrupted" if isinstance(error, (SystemExit, KeyboardInterrupt))
             else "failed", active_stage=active, error=str(error), failed_at=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=sweep.CONFIG)
    parser.add_argument("--diffusion-config", type=Path, default=sweep.DIFFUSION_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output.resolve(), config_path=args.config.resolve(),
                diffusion_config=args.diffusion_config.resolve())
        print(f"Prepared all three fresh studies; no native solves executed: {args.output.resolve()}")
    else:
        dump(args.output.resolve() / "manager.json", dict(pid=os.getpid(), started_at=now(),
             command=sys.argv, python=sys.executable))
        launch(args.output.resolve())


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    main()
