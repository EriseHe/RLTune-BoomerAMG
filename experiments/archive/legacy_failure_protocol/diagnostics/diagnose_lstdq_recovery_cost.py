"""Isolate LSTDQ-LCB behavior after one recovered solve failure.

The diagnostic reuses the locked 60^3 instance stream and the final successful
setup sequence from the legacy retry run.  It substitutes the new run's first
failed primary setup at exactly one case, then compares policy-only changes.
No setup bandit is active, so a controller failure cannot alter later setups.
"""

from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict

import numpy as np

from online_td_experiment_common import _json_ready, _write_json
from run_joint_online_sarsa_4k import _make_recursive_lstdq_controller
from setup_action_space import DEFAULT_SETUP_PARAMS
from setup_aware_compare_common import (
    augment_setup_params,
    solve_fixed_w_case,
    solve_no_rl_case,
)
from SolvePhase.algorithms.sarsa import run_td_episode


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instrument_controller(
    controller: Any,
    *,
    nonnegative_lcb: bool,
    covariance_td_clip_sec: float | None,
    postfit_covariance: bool,
) -> None:
    if nonnegative_lcb:
        original_values = controller._values

        def bounded_values(
            self: Any,
            features: np.ndarray,
            *,
            cycle: int | None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            means, uncertainty, scores = original_values(features, cycle=cycle)
            return means, uncertainty, np.maximum(scores, 0.0)

        controller._values = MethodType(bounded_values, controller)

    if covariance_td_clip_sec is not None and postfit_covariance:
        raise ValueError("Choose either clipped or post-fit covariance residuals")

    if covariance_td_clip_sec is not None:
        clip = float(covariance_td_clip_sec)
        if clip <= 0.0:
            raise ValueError("covariance TD clip must be positive")
        original_update = controller.update

        def robust_covariance_update(self: Any, **kwargs: Any) -> float:
            td_error = float(original_update(**kwargs))
            raw_moment = self.trace * td_error
            robust_moment = self.trace * float(np.clip(td_error, -clip, clip))
            self.moment_covariance += (
                np.outer(robust_moment, robust_moment)
                - np.outer(raw_moment, raw_moment)
            )
            return td_error

        controller.update = MethodType(robust_covariance_update, controller)

    if postfit_covariance:
        original_update = controller.update

        def postfit_covariance_update(self: Any, **kwargs: Any) -> float:
            prefit_td_error = float(original_update(**kwargs))
            joint = self.state_action_features(kwargs["features"])[
                int(kwargs["action_index"])
            ]
            if bool(kwargs["terminal"]):
                next_joint = np.zeros(self.joint_dim, dtype=float)
            else:
                next_joint = self.state_action_features(kwargs["next_features"])[
                    int(kwargs["next_action_index"])
                ]
            postfit_td_error = float(
                kwargs["cost"]
                + self.config.gamma * (next_joint @ self.theta)
                - joint @ self.theta
            )
            prefit_moment = self.trace * prefit_td_error
            postfit_moment = self.trace * postfit_td_error
            self.moment_covariance += (
                np.outer(postfit_moment, postfit_moment)
                - np.outer(prefit_moment, prefit_moment)
            )
            return prefit_td_error

        controller.update = MethodType(postfit_covariance_update, controller)


def _controller_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        weights=args.weights,
        action_rbf_centers=args.action_rbf_centers,
        action_rbf_sigma=0.2,
        epsilon_start=0.30,
        epsilon_final=0.03,
        epsilon_decay_steps=20_000.0,
        recursive_lstdq_lambda=0.8,
        recursive_lstdq_ridge=1.0,
        recursive_lstdq_beta=2.0,
        recursive_lstdq_residual_floor_sec=1.0e-3,
        recursive_lstdq_lcb_lower_bound_sec=None,
        recursive_lstdq_covariance_td_clip_sec=None,
        q_max_sec=0.1,
    )


def _summary(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    primary_success = [
        str(row.get("primary_status", "success")) == "success" for row in rows
    ]
    cycle_actions = [
        float(action)
        for row in rows
        for action in row.get("cycle_actions", ())
    ]
    uncertainties = [
        float(value)
        for row in rows
        for value in row.get("cycle_selected_uncertainties", ())
        if np.isfinite(float(value))
    ]
    scores = [
        float(value)
        for row in rows
        for value in row.get("cycle_selection_scores", ())
        if np.isfinite(float(value))
    ]
    return {
        "cases": len(rows),
        "primary_successes": int(sum(primary_success)),
        "primary_failures": int(len(rows) - sum(primary_success)),
        "fallbacks": int(sum(bool(row.get("fallback_used", False)) for row in rows)),
        "mean_primary_solve_sec": float(
            np.mean([float(row.get("primary_solve_runtime", row["solve_runtime"])) for row in rows])
        ),
        "mean_total_native_sec": float(np.mean([float(row["runtime"]) for row in rows])),
        "mean_weight": float(np.mean(cycle_actions)) if cycle_actions else float("nan"),
        "median_uncertainty_sec": (
            float(np.median(uncertainties)) if uncertainties else 0.0
        ),
        "max_uncertainty_sec": max(uncertainties, default=0.0),
        "min_selection_score_sec": min(scores, default=0.0),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    old_rows = _read_jsonl(
        args.old_result_dir / "trajectories" / "bandit_recursive_lstdq_lcb.jsonl"
    )
    new_rows = _read_jsonl(
        args.new_result_dir / "trajectories" / "bandit_recursive_lstdq_lcb.jsonl"
    )
    case_count = min(int(args.cases), len(old_rows), len(new_rows))
    first_failure = next(
        row["online_index"]
        for row in new_rows[:case_count]
        if bool(row["outcome"].get("fallback_used", False))
    )
    if int(first_failure) != 1:
        raise ValueError(f"Expected the locked first failure at index 1, got {first_failure}")

    all_variants = {
        "raw": (False, None, False),
        "nonnegative_lcb": (True, None, False),
        "robust_covariance": (
            False,
            float(args.covariance_td_clip_sec),
            False,
        ),
        "nonnegative_lcb_robust_covariance": (
            True,
            float(args.covariance_td_clip_sec),
            False,
        ),
        "postfit_covariance": (False, None, True),
        "nonnegative_lcb_postfit_covariance": (True, None, True),
    }
    requested_variants = tuple(
        name.strip() for name in str(args.variants).split(",") if name.strip()
    )
    unknown_variants = sorted(set(requested_variants) - set(all_variants))
    if unknown_variants:
        raise ValueError(f"Unknown diagnostic variants: {unknown_variants}")
    variants = {name: all_variants[name] for name in requested_variants}
    controller_args = _controller_args(args)
    controllers: Dict[str, Any] = {}
    encoders: Dict[str, Any] = {}
    for offset, (
        name,
        (lower_bound, covariance_clip, postfit_covariance),
    ) in enumerate(variants.items()):
        controller, encoder = _make_recursive_lstdq_controller(
            controller_args,
            seed=int(args.controller_seed + offset),
        )
        # Use an identical behavior-policy RNG across variants.  Differences then
        # come from the value/confidence rule, not different exploration draws.
        controller.rng = np.random.default_rng(int(args.controller_seed))
        _instrument_controller(
            controller,
            nonnegative_lcb=lower_bound,
            covariance_td_clip_sec=covariance_clip,
            postfit_covariance=postfit_covariance,
        )
        controllers[name] = controller
        encoders[name] = encoder

    records: Dict[str, list[Dict[str, Any]]] = {
        name: [] for name in (*variants.keys(), "fixed_w1.6")
    }
    output_trajectories = args.output_dir / "trajectories"
    output_trajectories.mkdir(parents=True, exist_ok=True)
    handles = {
        name: (output_trajectories / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in records
    }
    order_rng = np.random.default_rng(int(args.order_seed))
    try:
        for index in range(case_count):
            mkw = dict(new_rows[index]["mkw"])
            params = (
                dict(new_rows[index]["params"])
                if index == int(first_failure)
                else dict(old_rows[index]["params"])
            )
            for name in order_rng.permutation(tuple(records)):
                if name == "fixed_w1.6":
                    outcome = solve_fixed_w_case(
                        params=params,
                        mkw=mkw,
                        w=1.6,
                        sweeps_down=1,
                        sweeps_up=1,
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                    )
                    outcome.update(
                        {
                            "primary_status": (
                                "nonconvergence" if outcome["failed"] else "success"
                            ),
                            "fallback_used": False,
                        }
                    )
                else:
                    def fallback() -> Dict[str, Any]:
                        return solve_no_rl_case(
                            params=dict(DEFAULT_SETUP_PARAMS),
                            mkw=mkw,
                            solver_tol=float(args.tol),
                            solver_max_iter=int(args.max_cycles),
                            augment_params=augment_setup_params,
                        )

                    outcome = run_td_episode(
                        params=params,
                        mkw=mkw,
                        encoder=encoders[name],
                        controller=controllers[name],
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=True,
                        explore=True,
                        record_action_metadata=True,
                        initial_environment_weight_override=1.0,
                        fallback_attempt=fallback,
                    )
                row = {
                    "online_index": int(index),
                    "injected_failure_setup": bool(index == int(first_failure)),
                    "params": params,
                    "outcome": outcome,
                }
                records[name].append(outcome)
                handles[name].write(json.dumps(_json_ready(row), separators=(",", ":")))
                handles[name].write("\n")
            if (index + 1) % max(1, int(args.progress_every)) == 0:
                print(
                    json.dumps({"stage": "lstdq_recovery_diagnostic", "done": index + 1}),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    result = {
        "protocol": {
            "cases": int(case_count),
            "first_failure_index": int(first_failure),
            "setup_sequence": "legacy final successful setups except one new primary failure setup",
            "covariance_td_clip_sec": float(args.covariance_td_clip_sec),
            "weights": args.weights,
            "action_rbf_centers": args.action_rbf_centers,
        },
        "summaries": {name: _summary(rows) for name, rows in records.items()},
        "controllers": {
            name: controller.summary() | {
                "steps": int(controller.steps),
                "episodes": int(controller.episodes),
            }
            for name, controller in controllers.items()
        },
    }
    _write_json(args.output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-result-dir", type=Path, required=True)
    parser.add_argument("--new-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    # The locked recursive_lcb_ppo runner constructs MC first and LSTDQ second,
    # offsetting the common controller seed by 1009 for LSTDQ.
    parser.add_argument("--controller-seed", type=int, default=39967048)
    parser.add_argument("--order-seed", type=int, default=39972039)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--covariance-td-clip-sec", type=float, default=0.05)
    parser.add_argument(
        "--variants",
        default=(
            "raw,nonnegative_lcb,robust_covariance,"
            "nonnegative_lcb_robust_covariance"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--weights",
        default=",".join(f"{value:.2f}" for value in np.arange(1.0, 3.001, 0.05)),
    )
    parser.add_argument(
        "--action-rbf-centers",
        default=",".join(f"{value:.2f}" for value in np.arange(1.0, 3.001, 0.25)),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_json_ready(run(args)), indent=2))


if __name__ == "__main__":
    main()
