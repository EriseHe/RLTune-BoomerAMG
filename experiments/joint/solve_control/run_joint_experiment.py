from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
)
from online_td_experiment_common import _json_ready, _write_json
from plot_joint_online_sarsa_4k import generate_plots
from run_joint_online_sarsa_4k import (
    _validate_composable_protocol,
    build_parser,
    run,
)
from setup_action_space import (
    SetupConfigurationSpace,
)


SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "description",
    "output_dir",
    "problem",
    "stream",
    "seeds",
    "setup",
    "solve",
    "methods",
    "reporting",
}
SETUP_KEYS = {
    "parameter_resolution",
    "action_space",
    "candidate_schedule",
    "configuration_spaces",
    "lin_ts",
}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _expand_grid(value: Any, *, name: str) -> tuple[float, ...]:
    """Expand an explicit list or inclusive start/stop/step specification."""

    if isinstance(value, Mapping):
        spec = _mapping(value, name=name)
        missing = {"start", "stop", "step"} - set(spec)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
        start = Decimal(str(spec["start"]))
        stop = Decimal(str(spec["stop"]))
        step = Decimal(str(spec["step"]))
        if step <= 0 or stop < start:
            raise ValueError(f"{name} requires start <= stop and step > 0")
        span = stop - start
        integral_steps = span / step
        if integral_steps != integral_steps.to_integral_value():
            raise ValueError(f"{name} stop must lie exactly on its grid")
        count = int(integral_steps) + 1
        return tuple(float(start + index * step) for index in range(count))
    values = tuple(float(item) for item in _sequence(value, name=name))
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def _csv(values: Iterable[Any]) -> str:
    return ",".join(f"{value:g}" if isinstance(value, float) else str(value) for value in values)


def _append_option(argv: list[str], option: str, value: Any) -> None:
    if value is not None:
        argv.extend((option, str(value)))


def _method_token(method: Mapping[str, Any]) -> str:
    allowed = {
        "id",
        "setup",
        "setup_space",
        "candidate_sampling",
        "solve",
        "fixed_weight",
    }
    unknown = set(method) - allowed
    if unknown:
        raise ValueError(
            f"method {method.get('id', '<unknown>')!r} has unknown keys: "
            f"{sorted(unknown)}"
        )
    method_id = str(method["id"])
    setup = str(method["setup"])
    setup_space = method.get("setup_space")
    candidate_sampling = str(
        method.get("candidate_sampling", "uniform512")
    ).strip().lower().replace("-", "")
    if setup_space is not None:
        if setup not in {"linucb", "lints"}:
            raise ValueError(
                f"method {method_id!r} can only set setup_space for a setup bandit"
            )
        if candidate_sampling not in {"uniform512", "structured512"}:
            raise ValueError(
                f"method {method_id!r} candidate_sampling must be "
                "uniform512 or structured512"
            )
        setup = f"{setup}@{str(setup_space)}@{candidate_sampling}"
    elif "candidate_sampling" in method:
        raise ValueError(
            f"method {method_id!r} requires setup_space when setting "
            "candidate_sampling"
        )
    solve = str(method["solve"])
    if solve == "fixed":
        if "fixed_weight" not in method:
            raise ValueError(f"method {method_id!r} requires fixed_weight")
        solve = f"fixed@{float(method['fixed_weight']):g}"
    return f"{method_id}:{setup}:{solve}"


def _configuration_spaces(
    setup: Mapping[str, Any],
) -> Dict[str, SetupConfigurationSpace]:
    unknown = set(setup) - SETUP_KEYS
    if unknown:
        raise ValueError(f"Unknown setup config keys: {sorted(unknown)}")
    raw_spaces = _mapping(
        setup.get("configuration_spaces", {}),
        name="setup.configuration_spaces",
    )
    spaces = {
        str(name): SetupConfigurationSpace.from_mapping(
            str(name),
            _mapping(raw, name=f"setup.configuration_spaces.{name}"),
        )
        for name, raw in raw_spaces.items()
    }
    return spaces


def config_to_args(
    config: Mapping[str, Any],
    *,
    output_dir_override: Path | None = None,
) -> argparse.Namespace:
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")

    problem = _mapping(config.get("problem"), name="problem")
    stream = _mapping(config.get("stream"), name="stream")
    seeds = _mapping(config.get("seeds"), name="seeds")
    setup = _mapping(config.get("setup"), name="setup")
    configuration_spaces = _configuration_spaces(setup)
    candidate_schedule = _mapping(
        setup.get("candidate_schedule", {}),
        name="setup.candidate_schedule",
    )
    unknown_candidate_keys = set(candidate_schedule) - {
        "mode",
        "max_selections_per_case",
        "chunk_rounds",
    }
    if unknown_candidate_keys:
        raise ValueError(
            "Unknown setup.candidate_schedule keys: "
            f"{sorted(unknown_candidate_keys)}"
        )
    lin_ts = _mapping(setup.get("lin_ts", {}), name="setup.lin_ts")
    unknown_lin_ts_keys = set(lin_ts) - {
        "relative_sampling_scale",
        "loss_scale_prior_sec",
    }
    if unknown_lin_ts_keys:
        raise ValueError(
            f"Unknown setup.lin_ts keys: {sorted(unknown_lin_ts_keys)}"
        )
    solve = _mapping(config.get("solve"), name="solve")
    reporting = _mapping(config.get("reporting", {}), name="reporting")
    methods = tuple(
        _mapping(item, name="methods[]")
        for item in _sequence(config.get("methods"), name="methods")
    )
    if not methods:
        raise ValueError("methods cannot be empty")

    configured_output = config.get("output_dir")
    output_dir = output_dir_override or (
        None if configured_output is None else Path(str(configured_output))
    )
    if output_dir is None:
        raise ValueError("Set output_dir in the config or pass --output-dir")

    grid_raw = problem.get("grid")
    if isinstance(grid_raw, int):
        grid = (int(grid_raw),) * 3
    else:
        grid = tuple(
            int(value) for value in _sequence(grid_raw, name="problem.grid")
        )
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError("problem.grid must contain three positive integers")
    advection = tuple(
        float(value)
        for value in _sequence(
            problem.get("advection", [0.0, 0.0, 0.0]),
            name="problem.advection",
        )
    )

    actions = _expand_grid(solve.get("action_grid"), name="solve.action_grid")
    rbf = _mapping(solve.get("rbf"), name="solve.rbf")
    centers = _expand_grid(rbf.get("centers"), name="solve.rbf.centers")
    epsilon = _mapping(solve.get("epsilon", {}), name="solve.epsilon")
    lstdq = _mapping(solve.get("lstdq", {}), name="solve.lstdq")
    lstdq_v2 = _mapping(solve.get("lstdq_v2", {}), name="solve.lstdq_v2")
    rblspi = _mapping(solve.get("rblspi", {}), name="solve.rblspi")
    recursive_mc = _mapping(
        solve.get("recursive_mc", {}), name="solve.recursive_mc"
    )
    lsvi = _mapping(solve.get("lsvi", {}), name="solve.lsvi")
    structured = _mapping(
        solve.get("structured_model", {}), name="solve.structured_model"
    )
    recalibrated = _mapping(
        solve.get("recalibrated_lsvi", {}),
        name="solve.recalibrated_lsvi",
    )

    argv = [
        "--output-dir",
        str(output_dir),
        "--study-mode",
        "composable",
        "--problem",
        str(problem.get("kind", "scalar_anisotropic_diffusion")),
        "--matrix-grid-n",
        str(max(grid)),
        "--grid-shape",
        _csv(grid),
        "--advection",
        _csv(advection),
        "--train-cases",
        str(int(stream["cases"])),
        "--warmup-cases",
        str(int(stream.get("warmup_cases", 0))),
        "--online-cases",
        str(int(stream.get("online_cases", stream["cases"]))),
        "--instance-offset",
        str(int(stream.get("instance_offset", 0))),
        "--train-seed-groups",
        _csv(stream.get("seed_groups", ())),
        "--train-shuffle-seeds",
        _csv(stream.get("shuffle_seeds", ())),
        "--train-cases-per-seed",
        str(int(stream.get("cases_per_seed", stream["cases"]))),
        "--train-group-take",
        str(int(stream.get("group_take", stream["cases"]))),
        "--seed",
        str(int(seeds["base"])),
        "--bandit-seed",
        str(int(seeds["bandit"])),
        "--controller-seed",
        str(int(seeds["controller"])),
        "--method-order-seed",
        str(int(seeds["method_order"])),
        "--setup-param-resolution",
        str(int(setup["parameter_resolution"])),
        "--setup-action-space",
        str(setup.get("action_space", "full_cartesian")),
        "--setup-candidate-mode",
        str(candidate_schedule.get("mode", "explicit")),
        "--aot-max-selections-per-case",
        str(int(candidate_schedule.get("max_selections_per_case", 3))),
        "--aot-schedule-chunk-rounds",
        str(int(candidate_schedule.get("chunk_rounds", 256))),
        "--lin-ts-relative-sampling-scale",
        str(float(lin_ts.get("relative_sampling_scale", 0.15))),
        "--lin-ts-loss-scale-prior",
        str(float(lin_ts.get("loss_scale_prior_sec", 0.1))),
        "--c-min",
        str(float(problem.get("c_min", 1.0))),
        "--c-max",
        str(float(problem.get("c_max", 1000.0))),
        "--tol",
        str(float(solve.get("tolerance", 1.0e-6))),
        "--max-cycles",
        str(int(solve.get("max_cycles", 50))),
        "--weights",
        _csv(actions),
        "--action-rbf-centers",
        _csv(centers),
        "--action-rbf-sigma",
        str(float(rbf.get("sigma", 0.2))),
        "--epsilon-start",
        str(float(epsilon.get("start", 0.30))),
        "--epsilon-final",
        str(float(epsilon.get("final", 0.03))),
        "--epsilon-decay-steps",
        str(float(epsilon.get("decay_steps", 20_000.0))),
        "--progress-every",
        str(int(reporting.get("progress_every", 100))),
    ]
    option_values = (
        ("--expected-stream-hash", stream.get("expected_sha256")),
        ("--recursive-lstdq-ridge", lstdq.get("ridge")),
        ("--recursive-lstdq-beta", lstdq.get("beta")),
        ("--recursive-lstdq-lambda", lstdq.get("trace_lambda")),
        (
            "--recursive-lstdq-residual-floor-sec",
            lstdq.get("residual_floor_sec"),
        ),
        (
            "--recursive-lstdq-lcb-lower-bound-sec",
            lstdq.get("lcb_lower_bound_sec"),
        ),
        ("--recursive-lstdq-v2-beta", lstdq_v2.get("beta")),
        (
            "--recursive-lstdq-v2-coverage-ridge",
            lstdq_v2.get("coverage_ridge"),
        ),
        (
            "--recursive-lstdq-v2-residual-window",
            lstdq_v2.get("residual_window"),
        ),
        (
            "--recursive-lstdq-v2-min-samples",
            lstdq_v2.get("min_samples"),
        ),
        ("--rblspi-prior-precision", rblspi.get("prior_precision")),
        ("--rblspi-noise-precision", rblspi.get("noise_precision")),
        ("--rblspi-gram-ridge", rblspi.get("gram_ridge")),
        ("--recursive-mc-ridge", recursive_mc.get("ridge")),
        ("--recursive-mc-beta", recursive_mc.get("beta")),
        (
            "--recursive-mc-residual-floor-sec",
            recursive_mc.get("residual_floor_sec"),
        ),
        (
            "--recursive-mc-episode-half-life",
            recursive_mc.get("episode_half_life"),
        ),
        ("--lsvi-ridge", lsvi.get("ridge")),
        ("--lsvi-beta", lsvi.get("beta")),
        ("--lsvi-residual-floor-sec", lsvi.get("residual_floor_sec")),
        (
            "--lsvi-refit-interval-episodes",
            lsvi.get("refit_interval_episodes"),
        ),
        ("--structured-model-ridge", structured.get("ridge")),
        (
            "--structured-model-min-samples",
            structured.get("min_samples"),
        ),
        (
            "--structured-model-scale-window",
            structured.get("scale_window"),
        ),
        ("--recalibrated-lsvi-beta", recalibrated.get("beta")),
        (
            "--recalibrated-lsvi-refit-sweeps",
            recalibrated.get("refit_sweeps"),
        ),
        (
            "--recalibrated-lsvi-shrinkage-samples",
            recalibrated.get("shrinkage_samples"),
        ),
        ("--ppo-model", solve.get("ppo_model")),
    )
    for option, value in option_values:
        _append_option(argv, option, value)
    for method in methods:
        argv.extend(("--method", _method_token(method)))
    if bool(stream.get("smoke", False)):
        argv.append("--smoke")
    args = build_parser().parse_args(argv)
    args.setup_configuration_spaces = configuration_spaces
    return args


def _validate_resolved(args: argparse.Namespace) -> Dict[str, Any]:
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
    args: argparse.Namespace,
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
    args = config_to_args(config, output_dir_override=cli.output_dir)
    validation = _validate_resolved(args)
    print(json.dumps(_json_ready(validation), indent=2), flush=True)
    if cli.validate_only:
        return

    run(args)
    reporting = _mapping(config.get("reporting", {}), name="reporting")
    plots_enabled = bool(reporting.get("generate_plots", True)) and not cli.no_plots
    plot_summary = (
        generate_plots(
            result_dir=args.output_dir,
            rolling_window=int(reporting.get("rolling_window", 100)),
        )
        if plots_enabled
        else None
    )
    _write_reproduction_artifacts(
        config=config,
        config_path=config_path,
        args=args,
        validation=validation,
        plot_summary=plot_summary,
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_experiment_complete",
                    "output_dir": args.output_dir,
                    "plots": plot_summary,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
