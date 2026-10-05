"""Setup policy adapters and construction for joint experiments."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from setup.space import SetupConfigurationSpace, build_setup_parameter_spec, build_setup_param_space
from setup.learners.linucb import run_same_context_setup_reselection, validate_linucb_v5_paper_contract, validate_linucb_v6_experimental_contract
from setup.learners.common import AOTCandidateSchedule, FactorizedActionFeatureCache, GenericActionFeatureEncoder, ParameterSpaceSpec, RBFActionFeatureEncoder, SharedSetupLearnerSpec, resolve_tune7_candidate_strategy
from setup.registry import build_online_setup_learner, make_setup_learner_spec, normalize_setup_kind
from problems.amg import DIFCONV_CONTEXT_DIM
from problems.registry import SCALAR_ANISOTROPIC_DIFFUSION, SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION, normalize_problem_kind
from problems.streams import generate_difconv_instances as _generate_difconv_instances, generate_scalar_anisotropic_diffusion_instances, generate_scalar_anisotropic_diffusion_advection_instances
from .action_spaces import ActionSpaceBundle, DEFAULT_SETUP_PARAMS, build_action_space_bundle


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

def clone_branch_for_independent_updates(branch: BranchRun) -> BranchRun:
    """Clone online learner state while sharing immutable action-space storage."""
    source_policy = branch.policy
    source_model = getattr(source_policy, "model", None)
    if source_model is not None and hasattr(
        source_model, "clone_for_independent_updates"
    ):
        cloned_policy = type(source_policy)(
            source_model.clone_for_independent_updates()
        )
    else:
        cloned_policy = copy.deepcopy(source_policy)
    return BranchRun(
        label=str(branch.label),
        family=str(branch.family),
        tune_set=str(branch.tune_set),
        seed=branch.seed,
        policy=cloned_policy,
        parameter_space=branch.parameter_space,
        solver_tol=float(branch.solver_tol),
        solver_max_iter=int(branch.solver_max_iter),
    )

def validate_expected_setup_action_count(branch: BranchRun) -> int:
    model = getattr(branch.policy, "model", branch.policy)
    observed = int(model.K)
    raw_expected = os.environ.get("EXPECTED_SETUP_ACTION_COUNT", "").strip()
    if raw_expected:
        expected = int(raw_expected)
        if observed != expected:
            raise ValueError(
                "Setup action-count mismatch: "
                f"expected {expected:,}, observed {observed:,}. "
                "MATRIX_GRID_N and SETUP_PARAM_RESOLUTION must remain independent."
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
        tune7_candidate_pool_size=int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE", "1024")),
        tune7_candidate_pool_size_burnin=int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE_BURNIN", "4096")),
        tune7_candidate_pool_burnin_rounds=int(os.environ.get("TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS", "200")),
        tune7_alpha_decay_burnin_rounds=int(os.environ.get("TUNE7_ALPHA_DECAY_BURNIN_ROUNDS", "250")),
        tune7_candidate_strategy=os.environ.get("TUNE7_CANDIDATE_STRATEGY", "").strip().lower(),
        tune7_local_neighbor_radius=int(os.environ.get("TUNE7_LOCAL_NEIGHBOR_RADIUS", "1")),
        tune7_candidate_local_fraction=float(os.environ.get("TUNE7_CANDIDATE_LOCAL_FRACTION", "0.60")),
        tune7_candidate_elite_fraction=float(os.environ.get("TUNE7_CANDIDATE_ELITE_FRACTION", "0.20")),
    )

def normalize_method_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

class RandomPolicy:
    def __init__(self, actions: Sequence[Dict[str, Any]], seed: int) -> None:
        self._actions = [dict(a) for a in actions]
        self._rng = np.random.default_rng(int(seed))

    def select(self, context, **_):
        idx = int(self._rng.integers(0, len(self._actions)))
        return dict(self._actions[idx]), {"arm_index": idx}

    def update(self, loss, **_):
        return None

def family_seed_map(*, seed: int) -> Dict[str, int]:
    return {
        "Shared LinUCB v4": int(seed + 13003),
        # v5 keeps v4's candidate/tie RNG so context comparisons stay paired.
        "Shared LinUCB v5": int(seed + 13003),
        # RBF keeps the same RNG so only the action representation changes.
        "Shared LinUCB v5 RBF": int(seed + 13003),
        # v6 keeps the same RNG so only the PDE context representation changes.
        "Shared LinUCB v6": int(seed + 13003),
        # LinTS shares candidate/tie RNGs and uses an independent posterior RNG.
        "Shared LinTS v2": int(seed + 13003),
    }

def default_branch_label(*, method: str, tune_dim: int, tune7_variant: str) -> str:
    if str(method).strip().lower() == "default":
        return "default (fixed)"
    family = {
        "linucbv4": "Shared LinUCB v4",
        "linucbv5": "Shared LinUCB v5",
        "linucbv5rbf": "Shared LinUCB v5 RBF",
        "linucbv6": "Shared LinUCB v6",
        "lints_v2": "Shared LinTS v2",
    }.get(str(method).strip().lower(), str(method))
    if int(tune_dim) == 7:
        suffix = "tune7" if str(tune7_variant).strip().lower() == "agg_conditional" else "tune7-categorical"
        return f"{family} | {suffix}"
    return f"{family} | tune{int(tune_dim)}"

def build_single_branch(
    *,
    method: str,
    tune_dim: int,
    tune7_variant: str,
    seed: int,
    solver_tol: float,
    solver_max_iter: int,
    bandit_cfg: TestFinalBanditConfig,
    bundle: ActionSpaceBundle,
    context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
) -> BranchRun:
    method_key = str(method).strip().lower()
    label = default_branch_label(method=method_key, tune_dim=int(tune_dim), tune7_variant=str(tune7_variant))
    if method_key == "default":
        parameter_space = bundle.parameter_space_tune3 if int(tune_dim) in {3, 0} else (
            bundle.parameter_space_tune5 if int(tune_dim) == 5 else bundle.parameter_space_tune7
        )
        if parameter_space is None:
            raise ValueError(f"No parameter space available for tune_dim={tune_dim}")
        return BranchRun(
            label=label,
            family="default",
            tune_set="default",
            seed=None,
            policy=FixedPolicy(DEFAULT_SETUP_PARAMS),
            parameter_space=parameter_space,
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        )

    seed_map = family_seed_map(seed=int(seed))
    family = {
        "linucbv4": "Shared LinUCB v4",
        "linucbv5": "Shared LinUCB v5",
        "linucbv5rbf": "Shared LinUCB v5 RBF",
        "linucbv6": "Shared LinUCB v6",
        "lints_v2": "Shared LinTS v2",
    }.get(method_key)
    if family is None:
        raise ValueError(f"Unsupported method: {method}")

    if int(tune_dim) == 3:
        parameter_space = bundle.parameter_space_tune3
        default_arm_index = int(bundle.default_arm_index_tune3)
        parameter_spec = None
        tune_set = "tune3"
    elif int(tune_dim) == 5:
        parameter_space = bundle.parameter_space_tune5
        default_arm_index = int(bundle.default_arm_index_tune5)
        parameter_spec = None
        tune_set = "tune5"
    elif int(tune_dim) == 7:
        parameter_space = bundle.parameter_space_tune7
        default_arm_index = int(bundle.default_arm_index_tune7)
        parameter_spec = bundle.parameter_spec_tune7
        tune_set = "tune7"
    else:
        raise ValueError(f"Unsupported tune_dim: {tune_dim}")
    if parameter_space is None:
        raise ValueError(f"No parameter space available for tune_dim={tune_dim}")

    policy = build_test_final_bandit_policy(
        method=method_key,
        tune_dim=int(tune_dim),
        actions=parameter_space["actions"],
        context_dim=int(parameter_space["context_dim"]),
        seed=int(seed_map[family]),
        default_params=DEFAULT_SETUP_PARAMS,
        default_arm_index=int(default_arm_index),
        parameter_spec=parameter_spec,
        tune7_variant=tune7_variant,
        cfg=bandit_cfg,
        context_interaction_indices=context_interaction_indices,
    )
    return BranchRun(
        label=label,
        family=family,
        tune_set=tune_set,
        seed=int(seed_map[family]),
        policy=policy,
        parameter_space=parameter_space,
        solver_tol=float(solver_tol),
        solver_max_iter=int(solver_max_iter),
    )

def generate_difconv_instances(
    *,
    T: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    difconv_a: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    problem: str | None = None,
    advection_min: float | None = None,
    advection_max: float | None = None,
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    if problem is None:
        # Historical callers predate named problem contracts and retain the
        # original DifConv context so locked stream hashes stay reproducible.
        return _generate_difconv_instances(
            count=int(T),
            seed=int(seed),
            grid_choices=grid_choices,
            c_min=float(c_min),
            c_max=float(c_max),
            advection=difconv_a,
        )
    problem_kind = normalize_problem_kind(problem)
    if problem_kind == SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION:
        return generate_scalar_anisotropic_diffusion_advection_instances(
            count=int(T),
            seed=int(seed),
            grid_choices=grid_choices,
            c_min=float(c_min),
            c_max=float(c_max),
            advection_min=float(
                c_min if advection_min is None else advection_min
            ),
            advection_max=float(
                c_max if advection_max is None else advection_max
            ),
        )
    if problem_kind == SCALAR_ANISOTROPIC_DIFFUSION:
        if any(float(value) != 0.0 for value in difconv_a):
            raise ValueError(
                "scalar_anisotropic_diffusion requires zero advection"
            )
        return generate_scalar_anisotropic_diffusion_instances(
            count=int(T),
            seed=int(seed),
            grid_choices=grid_choices,
            c_min=float(c_min),
            c_max=float(c_max),
        )
    return _generate_difconv_instances(
        count=int(T),
        seed=int(seed),
        grid_choices=grid_choices,
        c_min=float(c_min),
        c_max=float(c_max),
        advection=difconv_a,
    )

class FixedPolicy:
    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)

    def select(self, context, **_):
        return dict(self._params), {}

    def update(self, loss, **_):
        return None

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
        return dict(params), info

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

def build_test10_branches(
    *,
    final_tune_dims: Sequence[int],
    tune7_variant: str,
    seed: int,
    solver_tol: float,
    solver_max_iter: int,
    include_default: bool,
    method_filter: str,
    branch_filter: str,
    bandit_cfg: TestFinalBanditConfig,
) -> Tuple[List[BranchRun], ActionSpaceBundle]:
    bundle = build_action_space_bundle(
        final_tune_dims=final_tune_dims,
        tune7_variant=tune7_variant,
    )

    branches: List[BranchRun] = []
    if include_default:
        branches.append(
            build_single_branch(
                method="default",
                tune_dim=3,
                tune7_variant=tune7_variant,
                seed=int(seed),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                bandit_cfg=bandit_cfg,
                bundle=bundle,
            )
        )

    method_specs: List[Tuple[str, int]] = []
    for tune_dim in sorted(int(v) for v in final_tune_dims):
        if tune_dim == 7:
            method_specs.append(("linucbv4", tune_dim))
        else:
            raise ValueError(
                f"The retained Exp44 active path only supports tune_dim=7, got tune_dim={tune_dim}"
            )

    for method, tune_dim in method_specs:
        branches.append(
            build_single_branch(
                method=method,
                tune_dim=int(tune_dim),
                tune7_variant=tune7_variant,
                seed=int(seed),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                bandit_cfg=bandit_cfg,
                bundle=bundle,
            )
        )

    if str(method_filter).strip():
        method_key = normalize_method_name(method_filter)
        branches = [
            branch
            for branch in branches
            if method_key in normalize_method_name(branch.family)
            or method_key in normalize_method_name(branch.label)
        ]
        if not branches:
            raise ValueError(f"METHOD_FILTER={method_filter!r} matched no branches")

    if str(branch_filter).strip():
        filters = [normalize_method_name(part) for part in str(branch_filter).split(",") if part.strip()]
        branches = [
            branch
            for branch in branches
            if any(filt in normalize_method_name(branch.label) for filt in filters)
        ]
        if not branches:
            raise ValueError(f"BRANCH_FILTER={branch_filter!r} matched no branches")

    for branch in branches:
        if normalize_method_name(branch.family) == normalize_method_name("Shared LinUCB v4"):
            validate_expected_setup_action_count(branch)

    return branches, bundle

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
    lin_ts_relative_sampling_scale: float = 0.15,
    lin_ts_loss_scale_prior: float = 0.1,
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
                int(index) for index in context_interaction_indices
            ),
            "always_include_arms": [int(default_arm_index)],
            "elite_cache_size": int(cfg.elite_cache_size),
            "initial_guess": [default_params[param.name] for param in parameter_spec.parameters],
            "initial_guess_rounds": int(os.environ.get("SETUP_INITIAL_GUESS_ROUNDS", "1")),
        }
        strategy = resolve_tune7_candidate_strategy(
            tune7_variant=tune7_variant,
            configured_strategy=cfg.tune7_candidate_strategy,
        )
        sampling = str(candidate_sampling).strip().lower().replace("-", "")
        if sampling not in AOTCandidateSchedule.SAMPLING_METHODS:
            raise ValueError(
                "candidate_sampling must be uniform512 or structured512"
            )
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
        elif strategy == "adaptive_local":
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.tune7_candidate_pool_size),
                    "candidate_strategy": "adaptive_local",
                    "candidate_pool_size_burnin": int(cfg.tune7_candidate_pool_size_burnin),
                    "candidate_pool_burnin_rounds": int(cfg.tune7_candidate_pool_burnin_rounds),
                    "alpha_decay_burnin_rounds": int(cfg.tune7_alpha_decay_burnin_rounds),
                    "elite_rank_metric": "mean_loss",
                    "local_neighbor_radius": int(cfg.tune7_local_neighbor_radius),
                    "candidate_local_fraction": float(cfg.tune7_candidate_local_fraction),
                    "candidate_elite_fraction": float(cfg.tune7_candidate_elite_fraction),
                }
            )
        else:
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.candidate_pool_size),
                }
            )
        learner_kwargs: Dict[str, Any] = {}
        if method_key == "lints_v2":
            learner_kwargs.update(
                {
                    "relative_sampling_scale": float(
                        lin_ts_relative_sampling_scale
                    ),
                    "loss_scale_prior": float(lin_ts_loss_scale_prior),
                }
            )
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
        "The retained Exp44 active path only supports generic tune7 setup "
        f"learners, got method={method!r}"
    )

def _safe_one_at_a_time_actions(parameter_spec: Any) -> list[Dict[str, Any]]:
    values = {
        "strong_threshold": (0.15, 0.35),
        "max_row_sum": (0.8, 0.95),
        "trunc_factor": (0.05, 0.1),
        "P_max_elmts": (2, 6, 8),
        "agg_num_levels": (1,),
        "coarsen_type": (6, 8),
        "interp_type": (8,),
    }
    actions = [dict(DEFAULT_SETUP_PARAMS)]
    seen = {tuple(sorted(DEFAULT_SETUP_PARAMS.items()))}
    params_by_name = {param.name: param for param in parameter_spec.parameters}
    for name, candidates in values.items():
        parameter = params_by_name[name]
        for candidate in candidates:
            value = candidate
            if value not in parameter.values:
                if parameter.kind == "categorical":
                    raise ValueError(
                        f"{name}={value!r} is not in {tuple(parameter.values)!r}"
                    )
                value = min(
                    parameter.values,
                    key=lambda item: abs(float(item) - float(value)),
                )
            params = dict(DEFAULT_SETUP_PARAMS)
            params[name] = value
            key = tuple(sorted(params.items()))
            if key not in seen:
                actions.append(params)
                seen.add(key)
    return actions

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
    lin_ts_relative_sampling_scale: float = 0.15,
    lin_ts_loss_scale_prior: float = 0.1,
    candidate_sampling: str = "uniform512",
    context_dim: int = DIFCONV_CONTEXT_DIM,
    context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
) -> tuple[BranchRun, TestFinalBanditConfig]:
    """Build a canonical online setup-bandit branch without a Gym dependency."""

    learner_token = normalize_setup_kind(learner_kind)
    learner_method = {
        "linucb": "linucbv4",
        "linucb_v5": "linucbv5",
        "linucb_v5_rbf": "linucbv5rbf",
        "linucb_v6": "linucbv6",
        "lints": "lints_v2",
    }.get(learner_token)
    if learner_method is None:
        raise ValueError(f"Unsupported setup learner kind: {learner_kind!r}")
    family = {
        "linucb": "Shared LinUCB v4",
        "linucb_v5": "Shared LinUCB v5",
        "linucb_v5_rbf": "Shared LinUCB v5 RBF",
        "linucb_v6": "Shared LinUCB v6",
        "lints": "Shared LinTS v2",
    }[learner_token]

    cfg = default_test_final_bandit_config_from_env()
    resolved_context_dim = int(context_dim)
    resolved_interaction_indices = tuple(
        int(index) for index in context_interaction_indices
    )
    if resolved_context_dim <= 0:
        raise ValueError("context_dim must be positive")
    if any(
        index < 0 or index >= resolved_context_dim
        for index in resolved_interaction_indices
    ):
        raise ValueError(
            "context_interaction_indices must lie within context_dim"
        )
    mode = str(
        os.environ.get("SETUP_ACTION_SPACE", "safe_one_at_a_time")
        if action_space_mode is None
        else action_space_mode
    ).strip().lower()
    if learner_token in {"linucb_v5", "linucb_v5_rbf", "linucb_v6"}:
        contract_kwargs = dict(
            context_dim=resolved_context_dim,
            context_interaction_indices=resolved_interaction_indices,
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            action_space_mode=mode,
            setup_space_name=(
                None
                if configuration_space is None
                else configuration_space.name
            ),
            coarsen_types=(
                None
                if configuration_space is None
                else configuration_space.coarsen_types
            ),
            interp_types=(
                None
                if configuration_space is None
                else configuration_space.interp_types
            ),
            agg_interp_types=(
                None
                if configuration_space is None
                else configuration_space.agg_interp_types
            ),
        )
        if learner_token == "linucb_v6":
            validate_linucb_v6_experimental_contract(**contract_kwargs)
        else:
            validate_linucb_v5_paper_contract(**contract_kwargs)
    resolved_tol = float(
        os.environ.get("SOLVE_TOL", "1e-6")
        if solver_tol is None
        else solver_tol
    )
    resolved_max_iter = int(
        os.environ.get("SOLVE_MAX_CYCLES", "50")
        if solver_max_iter is None
        else solver_max_iter
    )
    if mode == "full_cartesian" and configuration_space is not None:
        if int(tune_dim) != 7 or str(tune7_variant).strip().lower() != "categorical":
            raise ValueError(
                "Named setup configuration spaces require tune_dim=7 and "
                "tune7_variant='categorical'"
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
        family_seed = int(family_seed_map(seed=int(seed))[family])
        candidate_schedule = None
        action_feature_cache = None
        if candidate_schedule_dir is not None:
            if candidate_schedule_rounds is None or candidate_schedule_rounds <= 0:
                raise ValueError(
                    "candidate_schedule_rounds must be positive for AOT mode"
                )
            encoder = (
                RBFActionFeatureEncoder(setup_space.parameter_spec)
                if learner_token == "linucb_v5_rbf"
                else GenericActionFeatureEncoder(setup_space.parameter_spec)
            )
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
            lin_ts_relative_sampling_scale=float(
                lin_ts_relative_sampling_scale
            ),
            lin_ts_loss_scale_prior=float(lin_ts_loss_scale_prior),
            candidate_sampling=str(candidate_sampling),
            context_interaction_indices=resolved_interaction_indices,
        )
        branch = BranchRun(
            label=(
                f"{family} | {configuration_space.name} | "
                f"{str(candidate_sampling)}"
            ),
            family=family,
            tune_set=configuration_space.name,
            seed=family_seed,
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
        )
    elif mode == "full_cartesian":
        bundle = build_action_space_bundle(
            final_tune_dims=[int(tune_dim)],
            tune7_variant=tune7_variant,
            context_dim=resolved_context_dim,
        )
        branch = build_single_branch(
            method=learner_method,
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(seed),
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
            bandit_cfg=cfg,
            bundle=bundle,
            context_interaction_indices=resolved_interaction_indices,
        )
    elif mode == "safe_one_at_a_time":
        parameter_spec, _fixed_params = build_setup_parameter_spec(
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
        )
        actions = _safe_one_at_a_time_actions(parameter_spec)
        parameter_space = {
            "actions": actions,
            "context_dim": resolved_context_dim,
        }
        family_seed = int(family_seed_map(seed=int(seed))[family])
        policy = build_test_final_bandit_policy(
            method=learner_method,
            tune_dim=int(tune_dim),
            actions=actions,
            context_dim=resolved_context_dim,
            seed=family_seed,
            default_params=dict(DEFAULT_SETUP_PARAMS),
            default_arm_index=0,
            parameter_spec=parameter_spec,
            tune7_variant=tune7_variant,
            cfg=cfg,
            lin_ts_relative_sampling_scale=float(
                lin_ts_relative_sampling_scale
            ),
            lin_ts_loss_scale_prior=float(lin_ts_loss_scale_prior),
            context_interaction_indices=resolved_interaction_indices,
        )
        branch = BranchRun(
            label=default_branch_label(
                method=learner_method,
                tune_dim=int(tune_dim),
                tune7_variant=tune7_variant,
            ),
            family=family,
            tune_set="tune7",
            seed=family_seed,
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
        )
    else:
        raise ValueError(
            "SETUP_ACTION_SPACE must be safe_one_at_a_time or full_cartesian"
        )
    if configuration_space is None:
        validate_expected_setup_action_count(branch)
    return branch, cfg

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
        raise ValueError(
            "pass exactly one of problem_context or legacy context"
        )
    resolved_context = (
        problem_context if problem_context is not None else context
    )
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
