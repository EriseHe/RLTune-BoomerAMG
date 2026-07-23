"""Prepare and optionally run the gated 500-case LSTDQ v2/v3 joint smoke."""

from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_config(
    *,
    calibration_result_path: Path,
    stability_result_path: Path,
    experiment_output_dir: Path,
) -> Dict[str, Any]:
    calibration = _load_json(calibration_result_path)
    stability = _load_json(stability_result_path)
    selected = calibration["selected"]["lstdq_v3"]
    if not bool(selected.get("eligible", False)):
        raise ValueError(
            "Calibration did not produce an eligible LSTDQ v3 candidate"
        )
    selected_method = str(selected["method"])
    calibration_gate = calibration["candidates"][selected_method][
        "v3_calibration_gate"
    ]
    if not bool(calibration_gate["passed"]):
        raise ValueError("LSTDQ v3 calibration gate has not passed")
    if not bool(stability["stability"]["acceptance"]["passed"]):
        raise ValueError(
            "LSTDQ v3 execution-order stability gate has not passed"
        )

    repo_root = Path(__file__).resolve().parents[3]
    base_path = (
        repo_root
        / "experiments"
        / "joint"
        / "solve_control"
        / "configs"
        / "n40_recommended_lstdq_ucb_vs_rblspi_joint4k.json"
    )
    config = _load_json(base_path)
    config["name"] = "n40_recommended_lstdq_v2_vs_v3_joint500_smoke"
    config["description"] = (
        "Gate-controlled 40^3/500-case comparison of default setup/solve, "
        "structured-512 LinUCB plus recursive LSTDQ v2, and the same setup "
        "learner plus episode-cluster sandwich recursive LSTDQ v3."
    )
    config["output_dir"] = str(experiment_output_dir)
    first_seed = int(config["stream"]["seed_groups"][0])
    config["stream"] = {
        "cases": 500,
        "warmup_cases": 0,
        "online_cases": 500,
        "instance_offset": 0,
        "seed_groups": [first_seed],
        "shuffle_seeds": list(config["stream"]["shuffle_seeds"]),
        "cases_per_seed": 500,
        "group_take": 500,
        "expected_sha256": "",
        "smoke": True,
    }
    config["solve"]["lstdq_v2"]["beta"] = 4.0
    config["solve"]["lstdq_v3"] = {
        "beta": float(selected["beta"]),
    }
    config["methods"] = [
        {
            "id": "default_setup_default_solve",
            "setup": "default",
            "solve": "default",
        },
        {
            "id": "linucb_structured512_lstdq_v2",
            "setup": "linucb",
            "setup_space": "recommended",
            "candidate_sampling": "structured512",
            "solve": "recursive_lstdq_v2",
        },
        {
            "id": "linucb_structured512_lstdq_v3",
            "setup": "linucb",
            "setup_space": "recommended",
            "candidate_sampling": "structured512",
            "solve": "recursive_lstdq_v3",
        },
    ]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--stability-result", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = prepare_config(
        calibration_result_path=args.calibration_result,
        stability_result_path=args.stability_result,
        experiment_output_dir=args.experiment_output_dir,
    )
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-u",
        "experiments/joint/solve_control/run_joint_experiment.py",
        "--config",
        str(args.config_output),
    ]
    print(json.dumps({"config": str(args.config_output), "command": command}))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
