from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from joint_experiment_config import (
    JointExperimentSpec,
    JointExperimentRuntimeConfig,
    expand_grid as _expand_grid,
    legacy_namespace_from_spec,
    parse_joint_experiment_config,
    runtime_config_from_spec,
)
from joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
)
from online_td_experiment_common import _json_ready, _write_json
from plot_joint_online_sarsa_4k import generate_plots
from joint_4k_runner import (
    _validate_composable_protocol,
    run,
)


def config_to_args(
    config: Mapping[str, Any],
    *,
    output_dir_override: Path | None = None,
) -> argparse.Namespace:
    return legacy_namespace_from_spec(
        parse_joint_experiment_config(
            config,
            output_dir_override=output_dir_override,
        )
    )


def run_typed_experiment(spec: JointExperimentSpec) -> Dict[str, Any]:
    """Execute a typed experiment through the canonical runner."""

    return run(runtime_config_from_spec(spec))


def _validate_resolved(
    args: JointExperimentRuntimeConfig,
) -> Dict[str, Any]:
    specs = _validate_composable_protocol(args)
    _configure_paired_environment(args)
    stream, manifest = _build_paired_instance_stream(args)
    expected_hash = str(args.expected_stream_hash or "").strip()
    if expected_hash and str(manifest["sha256"]) != expected_hash:
        raise ValueError(
            f"Stream hash mismatch: {manifest['sha256']} != {expected_hash}"
        )
    if len(stream) != int(args.train_cases):
        raise ValueError("Generated stream length does not match train_cases")
    configuration_spaces = dict(
        getattr(args, "setup_configuration_spaces", {}) or {}
    )
    resolved_spaces = {
        name: configuration_space.as_dict()
        for name, configuration_space in configuration_spaces.items()
    }
    return {
        "output_dir": str(args.output_dir),
        "stream": manifest,
        "methods": [
            {
                "id": spec.name,
                "setup": spec.setup_kind,
                "setup_space": spec.setup_space,
                "candidate_sampling": spec.candidate_sampling,
                "solve": spec.solve_kind,
                "fixed_weight": spec.fixed_weight,
            }
            for spec in specs
        ],
        "setup_configuration_spaces": resolved_spaces,
        "setup_candidate_mode": str(args.setup_candidate_mode),
        "aot_max_selections_per_case": int(args.aot_max_selections_per_case),
        "lin_ts": {
            "relative_sampling_scale": float(
                args.lin_ts_relative_sampling_scale
            ),
            "loss_scale_prior_sec": float(args.lin_ts_loss_scale_prior),
        },
        "actions": list(float(value) for value in args.weights.split(",")),
        "rbf_centers": list(
            float(value) for value in args.action_rbf_centers.split(",")
        ),
    }


def _write_reproduction_artifacts(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    args: JointExperimentRuntimeConfig,
    validation: Mapping[str, Any],
    plot_summary: Mapping[str, Any] | None,
) -> None:
    output_dir = args.output_dir.resolve()
    resolved_config = dict(config)
    resolved_config["output_dir"] = str(output_dir)
    _write_json(output_dir / "experiment_config.json", resolved_config)
    _write_json(output_dir / "high_level_validation.json", dict(validation))
    if plot_summary is not None:
        _write_json(output_dir / "high_level_plot_summary.json", dict(plot_summary))

    repo_root = Path(__file__).resolve().parents[3]
    default_reproduction = output_dir.with_name(f"{output_dir.name}_reproduction")
    script = "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'REPO_ROOT="${{REPO_ROOT:-{repo_root}}}"',
            f'PYTHON_BIN="${{PYTHON_BIN:-{sys.executable}}}"',
            f'OUTPUT_DIR="${{OUTPUT_DIR:-{default_reproduction}}}"',
            'cd "$REPO_ROOT"',
            '"$PYTHON_BIN" -u experiments/joint/solve_control/run_joint_experiment.py \\\n  --config "' + str(output_dir / "experiment_config.json") + '" \\\n  --output-dir "$OUTPUT_DIR"',
            "",
        )
    )
    reproduce_path = output_dir / "reproduce.sh"
    reproduce_path.write_text(script, encoding="utf-8")
    reproduce_path.chmod(0o755)

    methods = validation["methods"]
    readme = "\n".join(
        (
            f"# {config.get('name', 'Joint AMG experiment')}",
            "",
            str(config.get("description", "Composable joint-online experiment.")),
            "",
            f"- source config: `{config_path}`",
            f"- stream SHA-256: `{validation['stream']['sha256']}`",
            f"- instances: `{args.train_cases}` (`{args.warmup_cases}` warmup + `{args.online_cases}` online)",
            f"- problem/grid: `{validation['stream'].get('problem')}` / `{validation['stream'].get('grid')}`",
            f"- methods: `{', '.join(method['id'] for method in methods)}`",
            "- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order",
            "- timing/recovery: active joint-online bounded-recovery protocol",
            "",
            "`config.json` and `stream_manifest.json` are the resolved low-level protocol;",
            "`experiment_config.json` is the reusable high-level configuration.",
            "Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting this directory.",
            "",
        )
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a composable setup x solve AMG experiment from JSON."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    cli = parser.parse_args()

    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = parse_joint_experiment_config(
        config,
        output_dir_override=cli.output_dir,
    )
    runtime = runtime_config_from_spec(spec)
    validation = _validate_resolved(runtime)
    print(json.dumps(_json_ready(validation), indent=2), flush=True)
    if cli.validate_only:
        return

    run_typed_experiment(spec)
    plots_enabled = spec.reporting.generate_plots and not cli.no_plots
    plot_summary = (
        generate_plots(
            result_dir=runtime.output_dir,
            rolling_window=spec.reporting.rolling_window,
        )
        if plots_enabled
        else None
    )
    _write_reproduction_artifacts(
        config=config,
        config_path=config_path,
        args=runtime,
        validation=validation,
        plot_summary=plot_summary,
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_experiment_complete",
                    "output_dir": runtime.output_dir,
                    "plots": plot_summary,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
