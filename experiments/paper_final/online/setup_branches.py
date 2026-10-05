"""Setup policy adapters and construction for joint experiments."""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np
from setup.space import SetupConfigurationSpace, build_setup_param_space
from setup.learners.linucb import run_same_context_setup_reselection
from setup.learners.common import (
    AOTCandidateSchedule,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    SharedSetupLearnerSpec,
    resolve_tune7_candidate_strategy,
)
from setup.registry import build_online_setup_learner, make_setup_learner_spec
from problems.amg import DIFCONV_CONTEXT_DIM
from .action_spaces import DEFAULT_SETUP_PARAMS


@dataclass(frozen=True)
class BranchRun:
    label: str
    family: str
    tune_set: str
    seed: int | None
    policy: Any
    parameter_space: Dict[str, Any]
    solver_tol: float
    solver_max_iter: int


def validate_expected_setup_action_count(branch: BranchRun) -> int:
    model = getattr(branch.policy, "model", branch.policy)
    observed = int(model.K)
    raw_expected = os.environ.get("EXPECTED_SETUP_ACTION_COUNT", "").strip()
    if raw_expected:
        expected = int(raw_expected)
        if observed != expected:
            raise ValueError(
                f"Setup action-count mismatch: expected {expected:,}, observed {observed:,}. MATRIX_GRID_N and SETUP_PARAM_RESOLUTION must remain independent."
            )
    return observed


@dataclass(frozen=True)
class TestFinalBanditConfig:
    alpha: float
    l2: float
    candidate_pool_size: int
    elite_cache_size: int
    tune7_candidate_pool_size: int
    tune7_candidate_pool_size_burnin: int
    tune7_candidate_pool_burnin_rounds: int
    tune7_alpha_decay_burnin_rounds: int
    tune7_candidate_strategy: str
    tune7_local_neighbor_radius: int
    tune7_candidate_local_fraction: float
    tune7_candidate_elite_fraction: float


def default_test_final_bandit_config_from_env() -> TestFinalBanditConfig:
    return TestFinalBanditConfig(
        alpha=float(os.environ.get("ALPHA", "1.0")),
        l2=float(os.environ.get("L2", "1.0")),
        candidate_pool_size=int(os.environ.get("CANDIDATE_POOL_SIZE", "512")),
        elite_cache_size=int(os.environ.get("ELITE_CACHE_SIZE", "64")),
        tune7_candidate_pool_size=int(
            os.environ.get("TUNE7_CANDIDATE_POOL_SIZE", "1024")
        ),
        tune7_candidate_pool_size_burnin=int(
            os.environ.get("TUNE7_CANDIDATE_POOL_SIZE_BURNIN", "4096")
        ),
        tune7_candidate_pool_burnin_rounds=int(
            os.environ.get("TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS", "200")
        ),
        tune7_alpha_decay_burnin_rounds=int(
            os.environ.get("TUNE7_ALPHA_DECAY_BURNIN_ROUNDS", "250")
        ),
        tune7_candidate_strategy=os.environ.get("TUNE7_CANDIDATE_STRATEGY", "")
        .strip()
        .lower(),
        tune7_local_neighbor_radius=int(
            os.environ.get("TUNE7_LOCAL_NEIGHBOR_RADIUS", "1")
        ),
        tune7_candidate_local_fraction=float(
            os.environ.get("TUNE7_CANDIDATE_LOCAL_FRACTION", "0.60")
        ),
        tune7_candidate_elite_fraction=float(
            os.environ.get("TUNE7_CANDIDATE_ELITE_FRACTION", "0.20")
        ),
    )


class GenericBanditPolicy:
    def __init__(self, model):
        self.model = model

    def select(self, context, **_):
        params = self.model.predict(np.asarray(context, dtype=float))
        info = {}
        if getattr(self.model, "history", None):
            step = self.model.history[-1]
            info = {
                "arm_index": int(getattr(step, "arm_index", -1)),
                "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
            }
        return (dict(params), info)

    def update(self, loss, **_):
        self.model.update(float(loss))

    def cancel_pending(self) -> None:
        self.model.cancel_pending()

    def recommend(self, context, *, candidate_arms, alpha: float = 0.0, **_):
        return self.model.recommend(
            np.asarray(context, dtype=float),
            candidate_arms=candidate_arms,
            alpha=float(alpha),
        )


def build_test_final_bandit_policy(
    *,
    method: str,
    tune_dim: int,
    actions: Sequence[Dict[str, Any]],
    context_dim: int,
    seed: int,
    default_params: Dict[str, Any],
    default_arm_index: int,
    parameter_spec: Optional[ParameterSpaceSpec],
    tune7_variant: str,
    cfg: TestFinalBanditConfig,
    candidate_schedule: AOTCandidateSchedule | None = None,
    action_feature_cache: FactorizedActionFeatureCache | None = None,
    candidate_sampling: str = "uniform512",
    context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
) -> GenericBanditPolicy:
    method_key = str(method).strip().lower()
    if int(tune_dim) == 7:
        if parameter_spec is None:
            raise ValueError("parameter_spec is required for tune_dim=7")
        tune7_kwargs: Dict[str, Any] = {
            "parameter_spec": parameter_spec,
            "context_interaction_indices": tuple(
                (int(index) for index in context_interaction_indices)
            ),
            "always_include_arms": [int(default_arm_index)],
            "elite_cache_size": int(cfg.elite_cache_size),
            "initial_guess": [
                default_params[param.name] for param in parameter_spec.parameters
            ],
            "initial_guess_rounds": int(
                os.environ.get("SETUP_INITIAL_GUESS_ROUNDS", "1")
            ),
        }
        strategy = resolve_tune7_candidate_strategy(
            tune7_variant=tune7_variant,
            configured_strategy=cfg.tune7_candidate_strategy,
        )
        sampling = str(candidate_sampling).strip().lower().replace("-", "")
        if sampling not in AOTCandidateSchedule.SAMPLING_METHODS:
            raise ValueError("candidate_sampling must be uniform512 or structured512")
        if sampling == "structured512":
            if candidate_schedule is None:
                raise ValueError("structured512 requires an AOT candidate schedule")
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.candidate_pool_size),
                    "candidate_strategy": "structured",
                    "elite_rank_metric": "mean_loss",
                    "local_neighbor_radius": 2,
                }
            )
        else:
            raise ValueError("The official study requires structured512 candidates")
        learner_kwargs: Dict[str, Any] = {}
        model = build_online_setup_learner(
            make_setup_learner_spec(
                kind=method_key,
                shared=SharedSetupLearnerSpec(
                    actions=actions,
                    context_dim=int(context_dim),
                    seed=int(seed),
                    alpha=float(cfg.alpha),
                    l2_reg=float(cfg.l2),
                    candidate_schedule=candidate_schedule,
                    action_feature_cache=action_feature_cache,
                    **tune7_kwargs,
                ),
                algorithm_parameters=learner_kwargs,
            )
        )
        return GenericBanditPolicy(model)
    raise ValueError(
        f"The official setup path supports generic tune7 LinUCB, got method={method!r}"
    )


def build_online_linucb_branch(
    *,
    seed: int,
    learner_kind: str = "linucb",
    tune_dim: int = 7,
    tune7_variant: str = "categorical",
    action_space_mode: str | None = None,
    solver_tol: float | None = None,
    solver_max_iter: int | None = None,
    parameter_resolution: int | None = None,
    configuration_space: SetupConfigurationSpace | None = None,
    candidate_schedule_dir: Path | None = None,
    candidate_schedule_rounds: int | None = None,
    candidate_schedule_seed: int | None = None,
    candidate_schedule_chunk_rounds: int = 256,
    candidate_sampling: str = "uniform512",
    context_dim: int = DIFCONV_CONTEXT_DIM,
    context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
) -> tuple[BranchRun, TestFinalBanditConfig]:
    """Build a canonical online setup-bandit branch without a Gym dependency."""
    if learner_kind != "linucb":
        raise ValueError("The official setup learner is LinUCB")
    learner_token = "linucb"
    learner_method = "linucb"
    family = "Shared LinUCB"
    cfg = default_test_final_bandit_config_from_env()
    resolved_context_dim = int(context_dim)
    resolved_interaction_indices = tuple(
        (int(index) for index in context_interaction_indices)
    )
    if resolved_context_dim <= 0:
        raise ValueError("context_dim must be positive")
    if any(
        (
            index < 0 or index >= resolved_context_dim
            for index in resolved_interaction_indices
        )
    ):
        raise ValueError("context_interaction_indices must lie within context_dim")
    mode = (
        str(
            os.environ.get("SETUP_ACTION_SPACE", "safe_one_at_a_time")
            if action_space_mode is None
            else action_space_mode
        )
        .strip()
        .lower()
    )
    resolved_tol = float(
        os.environ.get("SOLVE_TOL", "1e-6") if solver_tol is None else solver_tol
    )
    resolved_max_iter = int(
        os.environ.get("SOLVE_MAX_CYCLES", "50")
        if solver_max_iter is None
        else solver_max_iter
    )
    if mode == "full_cartesian" and configuration_space is not None:
        if int(tune_dim) != 7 or str(tune7_variant).strip().lower() != "categorical":
            raise ValueError(
                "Named setup configuration spaces require tune_dim=7 and tune7_variant='categorical'"
            )
        setup_space = build_setup_param_space(
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            parameter_resolution=parameter_resolution,
            configuration_space=configuration_space,
            materialize=candidate_schedule_dir is None,
        )
        parameter_space = {
            "actions": setup_space.actions,
            "context_dim": resolved_context_dim,
        }
        family_seed = int(seed) + 13003
        candidate_schedule = None
        action_feature_cache = None
        if candidate_schedule_dir is not None:
            if candidate_schedule_rounds is None or candidate_schedule_rounds <= 0:
                raise ValueError(
                    "candidate_schedule_rounds must be positive for AOT mode"
                )
            encoder = GenericActionFeatureEncoder(setup_space.parameter_spec)
            action_feature_cache = FactorizedActionFeatureCache(
                setup_space.actions, encoder
            )
            candidate_schedule = AOTCandidateSchedule(
                directory=Path(candidate_schedule_dir),
                catalog=setup_space.actions,
                rounds=int(candidate_schedule_rounds),
                pool_size=int(cfg.candidate_pool_size),
                seed=int(
                    family_seed
                    if candidate_schedule_seed is None
                    else candidate_schedule_seed
                ),
                chunk_rounds=int(candidate_schedule_chunk_rounds),
                sampling_method=str(candidate_sampling),
                factorized_cache=action_feature_cache,
            )
        policy = build_test_final_bandit_policy(
            method=learner_method,
            tune_dim=int(tune_dim),
            actions=setup_space.actions,
            context_dim=resolved_context_dim,
            seed=family_seed,
            default_params=dict(DEFAULT_SETUP_PARAMS),
            default_arm_index=int(setup_space.default_arm_index),
            parameter_spec=setup_space.parameter_spec,
            tune7_variant=tune7_variant,
            cfg=cfg,
            candidate_schedule=candidate_schedule,
            action_feature_cache=action_feature_cache,
            candidate_sampling=str(candidate_sampling),
            context_interaction_indices=resolved_interaction_indices,
        )
        branch = BranchRun(
            label=f"{family} | {configuration_space.name} | {str(candidate_sampling)}",
            family=family,
            tune_set=configuration_space.name,
            seed=family_seed,
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
        )
    else:
        raise ValueError(
            "The official study requires a named full-cartesian setup space"
        )
    if configuration_space is None:
        validate_expected_setup_action_count(branch)
    return (branch, cfg)


def run_bandit_step_test_final(
    *,
    policy: Any,
    parameter_space: Dict[str, Any],
    problem_context: np.ndarray | None = None,
    context: np.ndarray | None = None,
    solver_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    fallback_solver_fn: Callable[[Dict[str, Any]], Dict[str, Any]] | None,
    prev_update_est: float,
    primary_is_default: bool = False,
    failure_penalty_sec: float | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], int, float]:
    """Run one setup decision through the shared PDE-context boundary.

    ``context`` remains a compatibility alias for historical callers. New
    setup/solve orchestration should use the same ``problem_context`` keyword
    and values passed to the solve controller.
    """
    if problem_context is not None and context is not None:
        raise ValueError("pass exactly one of problem_context or legacy context")
    resolved_context = problem_context if problem_context is not None else context
    if resolved_context is None:
        raise ValueError("problem_context is required")
    result = run_same_context_setup_reselection(
        policy=policy,
        parameter_space=parameter_space,
        context=np.asarray(resolved_context, dtype=float),
        solver_fn=solver_fn,
        fallback_solver_fn=fallback_solver_fn,
        default_params=DEFAULT_SETUP_PARAMS,
        prev_update_est=float(prev_update_est),
        primary_is_default=bool(primary_is_default),
        max_learned_attempts=3,
        failure_penalty_sec=failure_penalty_sec,
    )
    return (
        dict(result.params),
        dict(result.outcome),
        dict(result.timing),
        int(result.fallback_used),
        float(result.update_runtime_sec),
    )
