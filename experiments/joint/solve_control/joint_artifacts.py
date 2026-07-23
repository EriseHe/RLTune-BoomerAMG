from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any, Dict

from online_td_experiment_common import _json_ready


_SOLVE_CONTROLLER_SCREEN_METHODS = (
    "bandit_fixed_w1.6",
    "bandit_recursive_lstdq_lcb",
    "bandit_recursive_lstdq_v2_lcb",
    "bandit_structured_model_based",
    "bandit_recalibrated_lsvi_lcb",
)


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")


def _write_solve_screen_reproduction(
    args: argparse.Namespace,
    *,
    stream_hash: str,
) -> None:
    """Write an auditable command and short protocol README beside results."""

    output_default = args.output_dir.with_name(f"{args.output_dir.name}_reproduction")
    command = [
        sys.executable,
        "-u",
        "experiments/joint/solve_control/run_joint_online_sarsa_4k.py",
        "--output-dir",
        '"$OUTPUT_DIR"',
        "--study-mode",
        "solve_controller_screen",
        "--seed",
        str(args.seed),
        "--bandit-seed",
        str(args.bandit_seed),
        "--controller-seed",
        str(args.controller_seed),
        "--method-order-seed",
        str(args.method_order_seed),
        "--train-cases",
        str(args.train_cases),
        "--warmup-cases",
        str(args.warmup_cases),
        "--online-cases",
        str(args.online_cases),
        "--train-seed-groups",
        str(args.train_seed_groups),
        "--train-shuffle-seeds",
        str(args.train_shuffle_seeds),
        "--train-cases-per-seed",
        str(args.train_cases_per_seed),
        "--train-group-take",
        str(args.train_group_take),
        "--matrix-grid-n",
        str(args.grid_n),
        "--setup-param-resolution",
        str(args.setup_param_resolution),
        "--max-cycles",
        str(args.max_cycles),
        "--shared-action-profile",
        str(args.shared_action_profile),
        "--recursive-lstdq-v2-beta",
        str(args.recursive_lstdq_v2_beta),
        "--recalibrated-lsvi-beta",
        str(args.recalibrated_lsvi_beta),
        "--progress-every",
        str(args.progress_every),
    ]
    if bool(args.smoke):
        command.append("--smoke")
    quoted = " ".join(
        token if token == '"$OUTPUT_DIR"' else shlex.quote(token)
        for token in command
    )
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'OUTPUT_DIR="${{OUTPUT_DIR:-{output_default}}}"\n'
        f"{quoted}\n"
    )
    script_path = args.output_dir / "reproduce.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    readme = f"""# 60^3 Solve-Controller Screen

This directory contains the locked five-branch joint-online comparison.

- methods: {', '.join(_SOLVE_CONTROLLER_SCREEN_METHODS)}
- stream SHA-256: `{stream_hash}`
- LSTDQ v2 beta: `{args.recursive_lstdq_v2_beta:g}`
- recalibrated LSVI beta: `{args.recalibrated_lsvi_beta:g}`
- action grid: `1.00:0.05:3.00`
- recovery and timing: active bounded-recovery joint protocol

Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting
this directory.  `config.json`, `stream_manifest.json`, trajectories, 1K
checkpoints, and final mutable states are written by the runner.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
