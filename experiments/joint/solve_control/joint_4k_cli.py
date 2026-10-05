"""Low-level command-line compatibility for the historical 4K runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.joint.solve_control.legacy_joint_studies import SHARED_ACTION_PROFILES
from problems.registry import SUPPORTED_PROBLEM_KINDS
from experiments.joint.solve_control.setup_aware_compare_common import (
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy CLI without coupling it to experiment execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Run either the locked 2K+2K protocol or the same 4K stream with "
            "setup bandits and solve controllers learning jointly from scratch."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--failure-penalty-sec", type=float, default=None,
        help="Opt into finite-budget failure feedback with this explicit final-failure penalty (seconds).",
    )
    parser.add_argument(
        "--study-mode",
        choices=(
            "sarsa",
            "lsvi_lcb",
            "recursive_lcb_suite",
            "recursive_lcb_ppo",
            "recursive_lstdq_lcb",
            "solve_controller_screen",
            "composable",
        ),
        default="sarsa",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/joint/mature_tune7_ppo_repro_20260423/run_logs/"
            "exp44_absolute_default_lstm_canonical_20260718/model_best.zip"
        ),
    )
    parser.add_argument(
        "--ppo-action-mode",
        choices=("continuous", "continuous_absolute"),
        default="continuous_absolute",
    )
    parser.add_argument("--ppo-w-center", type=float, default=1.5)
    parser.add_argument("--ppo-w-scale", type=float, default=0.5)
    parser.add_argument(
        "--ppo-initial-observation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ppo-force-default-first-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ppo-default-first-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=39860039)
    parser.add_argument("--bandit-seed", type=int, default=39860039)
    parser.add_argument("--controller-seed", type=int, default=39866039)
    parser.add_argument("--method-order-seed", type=int, default=39872039)
    parser.add_argument("--train-cases", type=int, default=4000)
    parser.add_argument("--warmup-cases", type=int, default=2000)
    parser.add_argument("--online-cases", type=int, default=2000)
    parser.add_argument("--instance-offset", type=int, default=0)
    parser.add_argument(
        "--train-seed-groups",
        default=(
            "39800039,39806039,39812039,39818039,"
            "39824039,39830039,39836039,39842039"
        ),
    )
    parser.add_argument("--train-shuffle-seeds", default="39848039")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-group-take", type=int, default=4000)
    parser.add_argument(
        "--expected-stream-hash",
        default="",
        help="Optional SHA-256 lock for the generated instance stream.",
    )
    parser.add_argument(
        "--problem",
        choices=SUPPORTED_PROBLEM_KINDS,
        default="scalar_anisotropic_diffusion",
    )
    parser.add_argument(
        "--matrix-grid-n",
        "--grid-n",
        dest="grid_n",
        type=int,
        default=EXP44_MATRIX_GRID_N,
    )
    parser.add_argument(
        "--grid-shape",
        default=None,
        help="Optional nx,ny,nz grid; otherwise --matrix-grid-n is cubic.",
    )
    parser.add_argument(
        "--advection",
        default="0,0,0",
        help="Fixed ax,ay,az for diffusion_convection problems.",
    )
    parser.add_argument(
        "--advection-min",
        type=float,
        default=None,
        help=(
            "Per-component lower bound for "
            "scalar_anisotropic_diffusion_advection."
        ),
    )
    parser.add_argument(
        "--advection-max",
        type=float,
        default=None,
        help=(
            "Per-component upper bound for "
            "scalar_anisotropic_diffusion_advection."
        ),
    )
    parser.add_argument(
        "--setup-param-resolution",
        type=int,
        default=EXP44_SETUP_PARAM_RESOLUTION,
    )
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument(
        "--setup-action-space",
        choices=("full_cartesian",),
        default="full_cartesian",
    )
    parser.add_argument(
        "--setup-candidate-mode",
        choices=("explicit", "aot"),
        default="explicit",
    )
    parser.add_argument("--aot-max-selections-per-case", type=int, default=3)
    parser.add_argument("--aot-schedule-chunk-rounds", type=int, default=256)
    parser.add_argument(
        "--lin-ts-relative-sampling-scale",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--lin-ts-loss-scale-prior",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--shared-action-profile",
        choices=tuple(SHARED_ACTION_PROFILES),
        default="1to2_step0p1",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Explicit solve action grid; locked modes require the profile grid.",
    )
    parser.add_argument(
        "--action-rbf-centers",
        default=None,
        help="Explicit action RBF centers; locked modes require the profile grid.",
    )
    parser.add_argument("--action-rbf-sigma", type=float, default=0.2)
    parser.add_argument("--alphas", default="0.001")
    parser.add_argument("--trace-lambdas", default="0.8")
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20_000.0)
    parser.add_argument("--q-max-sec", type=float, default=0.1)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--uncertainty-ridge", type=float, default=1.0)
    parser.add_argument(
        "--uncertainty-td-floor-sec",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--lsvi-ridge", type=float, default=1.0)
    parser.add_argument("--lsvi-beta", type=float, default=2.0)
    parser.add_argument(
        "--lsvi-residual-floor-sec",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--lsvi-refit-interval-episodes",
        type=int,
        default=100,
    )
    parser.add_argument("--recursive-mc-ridge", type=float, default=1.0)
    parser.add_argument("--recursive-mc-beta", type=float, default=2.0)
    parser.add_argument(
        "--recursive-mc-residual-floor-sec",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--recursive-mc-episode-half-life",
        type=float,
        default=500.0,
    )
    parser.add_argument("--recursive-lstdq-ridge", type=float, default=1.0)
    parser.add_argument("--recursive-lstdq-beta", type=float, default=2.0)
    parser.add_argument("--recursive-lstdq-lambda", type=float, default=0.8)
    parser.add_argument(
        "--recursive-lstdq-residual-floor-sec",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--recursive-lstdq-lcb-lower-bound-sec",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--recursive-lstdq-v2-beta",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--recursive-lstdq-v2-coverage-ridge",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--recursive-lstdq-v2-residual-window",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--recursive-lstdq-v2-min-samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--recursive-lstdq-v3-beta",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--rblspi-prior-precision",
        type=float,
        default=1.0e4,
    )
    parser.add_argument(
        "--rblspi-noise-precision",
        type=float,
        default=1.0e6,
    )
    parser.add_argument("--rblspi-gram-ridge", type=float, default=1.0e-6)
    parser.add_argument("--structured-model-ridge", type=float, default=1.0)
    parser.add_argument(
        "--structured-model-min-samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--structured-model-scale-window",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--recalibrated-lsvi-beta",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--recalibrated-lsvi-refit-sweeps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--recalibrated-lsvi-shrinkage-samples",
        type=float,
        default=32.0,
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--method",
        dest="method_specs",
        action="append",
        default=[],
        help=(
            "Composable branch as name:setup:solve; setup is "
            "default/linucb/linucb_v5/linucb_v5_rbf/linucb_v6/lints and a fixed solve uses "
            "fixed@weight. "
            "Repeat for each branch."
        ),
    )
    parser.add_argument(
        "--include-default-setup-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reuse-warmup", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


__all__ = ["build_parser"]
