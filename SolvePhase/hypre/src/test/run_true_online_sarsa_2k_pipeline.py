from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from plot_online_methods_2k import generate_plots
from online_td_experiment_common import _write_json
from run_online_methods_2k import run as run_final_2k
from run_true_online_sarsa_tuning import run as run_tuning


def _tuning_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=args.output_dir / "tuning",
        bandit_state=args.initial_bandit_state,
        train_seeds="39396939,39402939,39408939",
        train_cases_per_seed=500,
        train_take=1000,
        trace_shuffle_seed=39394016,
        controller_seeds="39394016,39394017,39394018",
        eval_seeds="39414939,39420939,39426939",
        eval_cases=96,
        alphas="0.001,0.005",
        trace_lambdas="0.2,0.5,0.8",
        weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
        anchor_weight=1.0,
        force_default_first_action=True,
        grid_n=40,
        setup_param_resolution=20,
        c_min=1.0,
        c_max=1000.0,
        tol=1.0e-6,
        max_cycles=50,
        progress_every=int(args.progress_every),
    )


def _final_args(
    args: argparse.Namespace,
    *,
    selected: Dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=args.output_dir / "final_2k",
        initial_bandit_state=args.initial_bandit_state,
        warmup_instances=1000,
        ppo_model=args.ppo_model,
        ppo_training_result=args.ppo_training_result,
        seed=39394016,
        train_cases=2000,
        instance_offset=1500,
        train_seed_groups="39394939,39400939;39406939,39412939",
        train_shuffle_seeds="39394016,39394022",
        train_cases_per_seed=500,
        train_group_take=1000,
        grid_n=40,
        setup_param_resolution=20,
        c_min=1.0,
        c_max=1000.0,
        tol=1.0e-6,
        max_cycles=50,
        setup_action_space="full_cartesian",
        fixed_weight=1.6,
        weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
        anchor_weight=1.0,
        alpha=float(selected["alpha"]),
        td_algorithm="true_online_sarsa",
        exploration_mode="uniform",
        td_decay_power=0.0,
        gamma=1.0,
        trace_lambda=float(selected["trace_lambda"]),
        epsilon_start=0.30,
        epsilon_final=0.03,
        epsilon_decay_steps=20_000.0,
        potential_scale_sec=0.001,
        failure_penalty_sec=0.1,
        initial_q_sec=0.0,
        force_default_first_action=True,
        progress_every=int(args.progress_every),
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuning_summary_path = args.output_dir / "tuning" / "tuning_summary.json"
    if args.reuse_tuning:
        tuning_summary = json.loads(tuning_summary_path.read_text(encoding="utf-8"))
    else:
        tuning_summary = run_tuning(_tuning_args(args))

    selected = tuning_summary.get("selected")
    if selected is None:
        summary = {
            "status": "stopped_after_tuning",
            "reason": "all true-online SARSA candidates were rejected",
            "tuning_summary": str(tuning_summary_path),
            "final_2k": None,
        }
        _write_json(args.output_dir / "pipeline_summary.json", summary)
        return summary

    if args.tuning_only:
        summary = {
            "status": "tuning_complete",
            "selected": selected,
            "tuning_summary": str(tuning_summary_path),
            "final_2k": None,
        }
        _write_json(args.output_dir / "pipeline_summary.json", summary)
        return summary

    final_dir = args.output_dir / "final_2k"
    result_path = final_dir / "result.json"
    if args.reuse_final:
        if not result_path.exists():
            raise FileNotFoundError(result_path)
    else:
        run_final_2k(_final_args(args, selected=selected))

    visualization = generate_plots(
        result_path=result_path,
        output_dir=final_dir / "figures",
        rolling_window=100,
    )
    summary = {
        "status": "complete",
        "selected": selected,
        "tuning_summary": str(tuning_summary_path),
        "final_2k": str(result_path),
        "visualization": visualization,
    }
    _write_json(args.output_dir / "pipeline_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune true-online SARSA(lambda), then run the locked five-method 2K protocol."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-bandit-state",
        type=Path,
        default=Path(
            "results/online_td_lambda_v1/run_logs/"
            "paired_online_bandit_warmup1000_seed39393939/bandit_state.pkl"
        ),
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/mature_tune7_ppo_repro_20260423/run_logs/"
            "exp44_warm1000_original_2500_20260716/model_best.zip"
        ),
    )
    parser.add_argument(
        "--ppo-training-result",
        type=Path,
        default=Path(
            "results/mature_tune7_ppo_repro_20260423/run_logs/"
            "exp44_warm1000_original_2500_20260716/result.json"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--reuse-tuning", action="store_true")
    parser.add_argument("--tuning-only", action="store_true")
    parser.add_argument("--reuse-final", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
