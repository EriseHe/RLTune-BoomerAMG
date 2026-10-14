from __future__ import annotations
from experiments.paper_final.online.methods import (
    validate_composable_protocol as _validate_composable_protocol,
)
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from experiments.paper_final.online.configuration import (
    JointExperimentSpec,
    JointExperimentRuntimeConfig,
    parse_joint_experiment_config,
    runtime_config_from_spec,
)
from experiments.paper_final.online.case_loop import (
    _build_paired_instance_stream,
    _configure_paired_environment,
)
from experiments.paper_final.online.io import _json_ready, _write_json
from experiments.paper_final.online.runner import run


def run_typed_experiment(spec: JointExperimentSpec) -> Dict[str, Any]:
    """Execute a typed experiment through the canonical runner."""
    return run(runtime_config_from_spec(spec))


def _generate_requested_plots(
    *, spec: JointExperimentSpec, result_dir: Path
) -> Dict[str, Any] | None:
    """Invoke the separate plot-only entry point when reporting requests it."""
    if not spec.reporting.generate_plots:
        return None
    from experiments.paper_final.online.plot_run import generate_experiment_plots

    return generate_experiment_plots(
        result_dir=Path(result_dir), rolling_window=int(spec.reporting.rolling_window)
    )


def _validate_resolved(args: JointExperimentRuntimeConfig) -> Dict[str, Any]:
    specs = _validate_composable_protocol(args)
    _configure_paired_environment(args)
    (stream, manifest) = _build_paired_instance_stream(args)
    expected_hash = str(args.expected_stream_hash or "").strip()
    if expected_hash and str(manifest["sha256"]) != expected_hash:
        raise ValueError(
            f"Stream hash mismatch: {manifest['sha256']} != {expected_hash}"
        )
    if len(stream) != int(args.train_cases):
        raise ValueError("Generated stream length does not match train_cases")
    configuration_spaces = dict(getattr(args, "setup_configuration_spaces", {}) or {})
    resolved_spaces = {
        name: configuration_space.as_dict()
        for (name, configuration_space) in configuration_spaces.items()
    }
    return {
        "output_dir": str(args.output_dir),
        "smoother_profile": args.smoother_profile,
        "failure_feedback": {
            "mode": "rollback_unrecovered"
            if args.failure_penalty_sec is None
            else "budgeted_penalty",
            "penalty_sec": args.failure_penalty_sec,
        },
        "stream": manifest,
        "methods": [
            {
                "id": spec.name,
                "setup": spec.setup_kind,
                "setup_space": spec.setup_space,
                "setup_context": spec.setup_context,
                "candidate_sampling": spec.candidate_sampling,
                "solve": spec.solve_kind,
                "fixed_weight": spec.fixed_weight,
                "seed_offset": int(spec.seed_offset),
                "setup_warmup_cases": int(spec.setup_warmup_cases),
                "solve_activation_case": int(spec.solve_activation_case),
                **(
                    {"solve_activation": asdict(spec.solve_activation)}
                    if spec.solve_activation is not None
                    else {}
                ),
                "solve_tolerance": spec.solve_tolerance,
                "solve_context": spec.solve_context,
            }
            for spec in specs
        ],
        "setup_configuration_spaces": resolved_spaces,
        "setup_candidate_mode": str(args.setup_candidate_mode),
        "aot_max_selections_per_case": int(args.aot_max_selections_per_case),
        "lin_ts": {
            "relative_sampling_scale": float(args.lin_ts_relative_sampling_scale),
            "loss_scale_prior_sec": float(args.lin_ts_loss_scale_prior),
        },
        "actions": list((float(value) for value in args.weights.split(","))),
        "rbf_centers": list(
            (float(value) for value in args.action_rbf_centers.split(","))
        ),
    }


def _write_reproduction_artifacts(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    args: JointExperimentRuntimeConfig,
    validation: Mapping[str, Any],
) -> None:
    output_dir = args.output_dir.resolve()
    resolved_config = dict(config)
    resolved_config["output_dir"] = str(output_dir)
    _write_json(output_dir / "experiment_config.json", resolved_config)
    _write_json(output_dir / "high_level_validation.json", dict(validation))
    script = "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            'REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"',
            'PYTHON_BIN="${PYTHON_BIN:-python}"',
            'OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}_reproduction}"',
            "",
            'cd "$REPO_ROOT"',
            '"$PYTHON_BIN" -u -m experiments.paper_final.online.run \\\n  --config "$SCRIPT_DIR/experiment_config.json" \\\n  --output-dir "$OUTPUT_DIR"',
            "",
        )
    )
    reproduce_path = output_dir / "reproduce.sh"
    reproduce_path.write_text(script, encoding="utf-8")
    reproduce_path.chmod(493)
    methods = validation["methods"]
    readme = "\n".join(
        (
            f"# {config.get('name', 'Joint AMG experiment')}",
            "",
            str(config.get("description", "Composable joint-online experiment.")),
            "",
            f"- frozen config: `experiment_config.json` (source filename: `{config_path.name}`)",
            f"- stream SHA-256: `{validation['stream']['sha256']}`",
            f"- instances: `{args.train_cases}` (`{args.warmup_cases}` warmup + `{args.online_cases}` online)",
            f"- problem/grid: `{validation['stream'].get('problem')}` / `{validation['stream'].get('grid')}`",
            f"- methods: `{', '.join((method['id'] for method in methods))}`",
            "- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order",
            "- timing/recovery: active joint-online bounded-recovery protocol",
            "",
            "`stream_manifest.json` records the exact deterministic input stream.",
            "Follow the repository README once to create the environment and build HYPRE.",
            "Then run `OUTPUT_DIR=/new/path ./reproduce.sh`; the script uses the active Python environment.",
            "When `reporting.generate_plots` is true, the high-level entry point invokes the separate plot-only generator after the runner finishes.",
            "Plots can also be regenerated with `plot_run.py`.",
            "Raw trajectories and checkpoints remain local; the paper bundle keeps compact evidence only.",
            "",
        )
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a composable setup x solve AMG experiment from JSON."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    cli = parser.parse_args(argv)
    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = parse_joint_experiment_config(config, output_dir_override=cli.output_dir)
    runtime = runtime_config_from_spec(spec)
    validation = _validate_resolved(runtime)
    print(json.dumps(_json_ready(validation), indent=2), flush=True)
    if cli.validate_only:
        return
    run_typed_experiment(spec)
    _write_reproduction_artifacts(
        config=config, config_path=config_path, args=runtime, validation=validation
    )
    plots = _generate_requested_plots(spec=spec, result_dir=runtime.output_dir)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_experiment_complete",
                    "output_dir": runtime.output_dir,
                    "plots_generated": plots is not None,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
