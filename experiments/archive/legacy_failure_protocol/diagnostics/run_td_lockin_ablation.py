from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from setup_action_space import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from online_td_experiment_common import (
    _action_diagnostics,
    _git_revision,
    _method_outcome,
    _paired,
    _summarize,
    _write_json,
)


TD_VARIANTS = (
    "setup_lambda0p8",
    "setup_lambda0p2",
    "full_lambda0p8",
    "setup_lambda0p8_uniform",
    "setup_lambda0p8_no_shaping",
    "setup_expected_a0p005_uniform",
    "setup_sarsa_a0p005_uniform",
    "setup_true_online_a0p02",
    "setup_true_online_a0p005",
    "setup_true_online_a0p001",
    "setup_true_online_a0p005_uniform",
)


def _make_encoder(
    *,
    mode: str,
    tol: float,
    max_cycles: int,
    c_max: float,
) -> SolveStateEncoder:
    setup_encoder = None
    if mode == "setup_full":
        parameter_spec, _fixed_params = build_setup_parameter_spec(
            tune_dim=7,
            tune7_variant="categorical",
        )
        setup_encoder = SetupObsEncoder(
            parameter_spec,
            dict(DEFAULT_SETUP_PARAMS),
            tuple(parameter_spec.parameter_names),
        )
    return SolveStateEncoder(
        tol=tol,
        max_cycles=max_cycles,
        c_max=c_max,
        mode=mode,
        setup_obs_encoder=setup_encoder,
    )


def _make_variants(
    *,
    base_config: ExpectedSarsaLambdaConfig,
    tol: float,
    max_cycles: int,
    c_max: float,
    seed: int,
    selected_variants: tuple[str, ...],
) -> tuple[
    Dict[str, ExpectedSarsaLambda],
    Dict[str, SolveStateEncoder],
]:
    definitions = {
        "setup_lambda0p8": ("setup_full", base_config),
        "setup_lambda0p2": (
            "setup_full",
            replace(base_config, trace_lambda=0.2),
        ),
        "full_lambda0p8": ("full", base_config),
        "setup_lambda0p8_uniform": (
            "setup_full",
            replace(base_config, exploration_mode="uniform"),
        ),
        "setup_lambda0p8_no_shaping": (
            "setup_full",
            replace(base_config, potential_scale_sec=0.0),
        ),
        "setup_expected_a0p005_uniform": (
            "setup_full",
            replace(
                base_config,
                alpha=5.0e-3,
                td_decay_power=0.0,
                exploration_mode="uniform",
            ),
        ),
        "setup_sarsa_a0p005_uniform": (
            "setup_full",
            replace(
                base_config,
                alpha=5.0e-3,
                td_decay_power=0.0,
                td_algorithm="sarsa",
                exploration_mode="uniform",
            ),
        ),
        "setup_true_online_a0p02": (
            "setup_full",
            replace(
                base_config,
                alpha=2.0e-2,
                td_decay_power=0.0,
                td_algorithm="true_online_sarsa",
            ),
        ),
        "setup_true_online_a0p005": (
            "setup_full",
            replace(
                base_config,
                alpha=5.0e-3,
                td_decay_power=0.0,
                td_algorithm="true_online_sarsa",
            ),
        ),
        "setup_true_online_a0p001": (
            "setup_full",
            replace(
                base_config,
                alpha=1.0e-3,
                td_decay_power=0.0,
                td_algorithm="true_online_sarsa",
            ),
        ),
        "setup_true_online_a0p005_uniform": (
            "setup_full",
            replace(
                base_config,
                alpha=5.0e-3,
                td_decay_power=0.0,
                td_algorithm="true_online_sarsa",
                exploration_mode="uniform",
            ),
        ),
    }
    controllers: Dict[str, ExpectedSarsaLambda] = {}
    encoders: Dict[str, SolveStateEncoder] = {}
    unknown = sorted(set(selected_variants) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown TD variants: {unknown}")
    for name in selected_variants:
        mode, config = definitions[name]
        encoder = _make_encoder(
            mode=mode,
            tol=tol,
            max_cycles=max_cycles,
            c_max=c_max,
        )
        encoders[name] = encoder
        controllers[name] = ExpectedSarsaLambda(
            feature_dim=encoder.feature_dim,
            config=config,
            seed=seed,
            initial_parameters=encoder.constant_value_parameters(
                config.initial_q_sec
            ),
        )
    return controllers, encoders


def _window_summary(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    start: int,
    stop: int,
    weights: tuple[float, ...],
    seed: int,
    td_variants: tuple[str, ...],
) -> Dict[str, Any]:
    fixed = records["fixed_w1.6"][start:stop]
    return {
        "start": int(start),
        "stop": int(stop),
        "methods": {
            method: _summarize(rows[start:stop])
            for method, rows in records.items()
        },
        "comparisons_vs_fixed_w1.6": {
            method: _paired(
                fixed,
                rows[start:stop],
                seed=seed + index * 17,
            )
            for index, (method, rows) in enumerate(records.items())
            if method != "fixed_w1.6"
        },
        "actions": {
            method: _action_diagnostics(records[method][start:stop], weights)
            for method in td_variants
        },
    }


def _run_stream(
    *,
    source_rows: list[Dict[str, Any]],
    methods: tuple[str, ...],
    controllers: Dict[str, ExpectedSarsaLambda],
    encoders: Dict[str, SolveStateEncoder],
    td_variants: tuple[str, ...],
    weights: tuple[float, ...],
    tol: float,
    max_cycles: int,
    seed: int,
    learn: bool,
    progress_every: int,
    progress_path: Path,
) -> Dict[str, list[Dict[str, Any]]]:
    records: Dict[str, list[Dict[str, Any]]] = {
        method: [] for method in methods
    }
    rng = np.random.default_rng(int(seed))
    stage = "td_lockin_online_ablation" if learn else "td_lockin_frozen_evaluation"
    reference_name = td_variants[0]
    for index, source_row in enumerate(source_rows):
        order = list(methods)
        rng.shuffle(order)
        for method in order:
            mkw = dict(source_row["mkw"])
            params = dict(source_row["params"])
            if method == "fixed_w1.6":
                outcome = _method_outcome(
                    method,
                    mkw=mkw,
                    params=params,
                    controller=controllers[reference_name],
                    encoder=encoders[reference_name],
                    tol=tol,
                    max_cycles=max_cycles,
                    learn=False,
                    explore=False,
                )
            else:
                outcome = run_td_episode(
                    mkw=mkw,
                    params=params,
                    controller=controllers[method],
                    encoder=encoders[method],
                    solve_tol=tol,
                    solve_max_cycles=max_cycles,
                    learn=learn,
                    explore=learn,
                )
            row = dict(outcome)
            row["instance_index"] = int(source_row["instance_index"])
            row["params"] = params
            records[method].append(row)

        done = index + 1
        if done % max(1, int(progress_every)) == 0 or done == len(source_rows):
            progress = {
                "stage": stage,
                "done": done,
                "episodes": len(source_rows),
                "methods": {
                    method: _summarize(rows)
                    for method, rows in records.items()
                },
                "dominant_actions": {
                    method: _action_diagnostics(records[method], weights)
                    for method in td_variants
                },
            }
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "done": done,
                        "episodes": len(source_rows),
                        "modal_weight": {
                            method: progress["dominant_actions"][method][
                                "modal_weight"
                            ]
                            for method in td_variants
                        },
                    }
                ),
                flush=True,
            )
    return records


def run(args: argparse.Namespace) -> None:
    source = json.loads(args.source_result.read_text())
    source_rows = source["online_records"]["bandit_td"]
    episodes = min(int(args.episodes), len(source_rows))
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    saved = np.load(args.source_checkpoint)
    base_config = ExpectedSarsaLambdaConfig(**json.loads(str(saved["config"])))
    protocol = source["protocol"]
    tol = float(protocol["tol"])
    max_cycles = int(protocol["max_cycles"])
    c_max = float(protocol["coefficient_range"][1])
    selected_variants = tuple(
        part.strip() for part in args.variants.split(",") if part.strip()
    )
    if not selected_variants:
        raise ValueError("At least one TD variant is required")
    controllers, encoders = _make_variants(
        base_config=base_config,
        tol=tol,
        max_cycles=max_cycles,
        c_max=c_max,
        seed=int(args.seed),
        selected_variants=selected_variants,
    )
    if args.load_controller_dir is not None:
        for name, controller in controllers.items():
            controller.load(
                args.load_controller_dir / f"{name}_controller.npz",
                restore_counters=True,
            )

    methods = ("fixed_w1.6", *selected_variants)
    if args.load_controller_dir is None:
        records = _run_stream(
            source_rows=source_rows[:episodes],
            methods=methods,
            controllers=controllers,
            encoders=encoders,
            td_variants=selected_variants,
            weights=base_config.weights,
            tol=tol,
            max_cycles=max_cycles,
            seed=int(args.seed + 101_003),
            learn=True,
            progress_every=int(args.progress_every),
            progress_path=args.output_dir / "progress.json",
        )
    else:
        records = {method: [] for method in methods}
    evaluation_start = (
        int(args.eval_start) if int(args.eval_start) >= 0 else episodes
    )
    evaluation_episodes = min(
        max(0, int(args.eval_episodes)),
        max(0, len(source_rows) - evaluation_start),
    )
    evaluation_records = _run_stream(
        source_rows=source_rows[
            evaluation_start : evaluation_start + evaluation_episodes
        ],
        methods=methods,
        controllers=controllers,
        encoders=encoders,
        td_variants=selected_variants,
        weights=base_config.weights,
        tol=tol,
        max_cycles=max_cycles,
        seed=int(args.seed + 202_007),
        learn=False,
        progress_every=int(args.progress_every),
        progress_path=args.output_dir / "evaluation_progress.json",
    ) if evaluation_episodes else None

    windows = [] if args.load_controller_dir is not None else [(0, episodes)]
    if args.load_controller_dir is None and episodes >= 2:
        midpoint = episodes // 2
        windows.extend(((0, midpoint), (midpoint, episodes)))
    result = {
        "protocol": {
            "git_revision": _git_revision(),
            "source_result": str(args.source_result),
            "source_checkpoint": str(args.source_checkpoint),
            "episodes": episodes,
            "training_executed": args.load_controller_dir is None,
            "loaded_controller_dir": (
                None
                if args.load_controller_dir is None
                else str(args.load_controller_dir)
            ),
            "evaluation_start": evaluation_start,
            "evaluation_episodes": evaluation_episodes,
            "information_model": "same recorded (instance, setup) pairs; one trajectory per method",
            "randomized_method_order_per_instance": True,
            "base_controller": base_config.__dict__,
            "variants": {
                name: {
                    "state_mode": encoders[name].mode,
                    "config": controllers[name].config.__dict__,
                }
                for name in selected_variants
            },
        },
        "windows": {
            f"{start}:{stop}": _window_summary(
                records,
                start=start,
                stop=stop,
                weights=base_config.weights,
                seed=int(args.seed + start + stop),
                td_variants=selected_variants,
            )
            for start, stop in windows
        },
        "records": records,
        "evaluation": (
            None
            if evaluation_records is None
            else _window_summary(
                evaluation_records,
                start=0,
                stop=evaluation_episodes,
                weights=base_config.weights,
                seed=int(args.seed + 303_011),
                td_variants=selected_variants,
            )
        ),
        "evaluation_records": evaluation_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, controller in controllers.items():
        controller.save(args.output_dir / f"{name}_controller.npz")
    _write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired online TD variants on an identical recorded setup stream."
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-start", type=int, default=-1)
    parser.add_argument("--load-controller-dir", type=Path)
    parser.add_argument("--seed", type=int, default=39394016)
    parser.add_argument("--variants", default=",".join(TD_VARIANTS))
    parser.add_argument("--progress-every", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
