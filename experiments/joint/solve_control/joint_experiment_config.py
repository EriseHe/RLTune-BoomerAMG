from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, cast

import numpy as np

from joint_method_spec import ComposableMethodSpec
from hypre.bindings.config import SMOOTHER_PROFILES
from hypre.bindings.recovery import validate_failure_penalty
from problems.registry import (
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    normalize_problem_kind,
)
from setup.space import SetupConfigurationSpace
from setup.registry import ONLINE_SETUP_KINDS
from solve.controllers.common import (
    EpsilonScheduleSpec,
    SharedActionSpec,
    SolveStateSpec,
)
from solve.controllers.lsvi import (
    HierarchicalLsviLcbSpec,
    LsviFamilySpecs,
    StagewiseLsviLcbSpec,
)
from solve.controllers.model_based import StructuredModelBasedSpec
from solve.controllers.rblspi import RecursiveBlstdqSpec
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqFamilySpecs,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbSpec,
)
from solve.controllers.recursive_mc import RecursiveMonteCarloLcbSpec
from solve.registry import (
    ONLINE_SOLVE_KINDS,
    OnlineControllerBuildSpec,
    OnlineSolveKind,
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
    "failure_feedback",
}
SETUP_KEYS = {
    "parameter_resolution",
    "action_space",
    "candidate_schedule",
    "configuration_spaces",
    "lin_ts",
    "replay_trajectory",
    "shared_online_prefix",
}
CANDIDATE_SCHEDULE_KEYS = {
    "mode",
    "max_selections_per_case",
    "chunk_rounds",
}
LIN_TS_KEYS = {
    "relative_sampling_scale",
    "loss_scale_prior_sec",
}
PROBLEM_KEYS = {
    "kind",
    "grid",
    "c_min",
    "c_max",
    "advection",
    "advection_min",
    "advection_max",
}
STREAM_KEYS = {
    "cases",
    "warmup_cases",
    "online_cases",
    "instance_offset",
    "seed_groups",
    "shuffle_seeds",
    "cases_per_seed",
    "group_take",
    "expected_sha256",
    "smoke",
}
SEED_KEYS = {"base", "bandit", "controller", "method_order"}
REPORTING_KEYS = {"progress_every", "rolling_window", "generate_plots"}
SOLVE_KEYS = {
    "state_encoding",
    "smoother_profile",
    "action_grid",
    "rbf",
    "epsilon",
    "tolerance",
    "max_cycles",
    "lstdq",
    "lstdq_v2",
    "lstdq_v3",
    "rblspi",
    "recursive_mc",
    "lsvi",
    "structured_model",
    "recalibrated_lsvi",
    "ppo_model",
}
RBF_KEYS = {"centers", "sigma"}
EPSILON_KEYS = {"start", "final", "decay_steps"}
DEFAULT_PPO_MODEL = Path(
    "results/joint/mature_tune7_ppo_repro_20260423/run_logs/"
    "exp44_absolute_default_lstm_canonical_20260718/model_best.zip"
)


def mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def reject_unknown_keys(
    raw: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def expand_grid(value: Any, *, name: str) -> tuple[float, ...]:
    """Expand an explicit list or inclusive start/stop/step specification."""

    if isinstance(value, Mapping):
        spec = mapping(value, name=name)
        missing = {"start", "stop", "step"} - set(spec)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
        start = Decimal(str(spec["start"]))
        stop = Decimal(str(spec["stop"]))
        step = Decimal(str(spec["step"]))
        if step <= 0 or stop < start:
            raise ValueError(f"{name} requires start <= stop and step > 0")
        integral_steps = (stop - start) / step
        if integral_steps != integral_steps.to_integral_value():
            raise ValueError(f"{name} stop must lie exactly on its grid")
        return tuple(
            float(start + index * step)
            for index in range(int(integral_steps) + 1)
        )
    values = tuple(float(item) for item in sequence(value, name=name))
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def csv(values: Iterable[Any]) -> str:
    return ",".join(
        f"{value:g}" if isinstance(value, float) else str(value)
        for value in values
    )


@dataclass(frozen=True)
class ProblemSpec:
    kind: str
    grid: tuple[int, int, int]
    c_min: float
    c_max: float
    advection: tuple[float, ...]
    advection_min: float
    advection_max: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProblemSpec":
        reject_unknown_keys(raw, PROBLEM_KEYS, name="problem")
        kind = str(raw.get("kind", "scalar_anisotropic_diffusion"))
        canonical_kind = normalize_problem_kind(kind)
        grid_raw = raw.get("grid")
        if isinstance(grid_raw, int):
            grid = (int(grid_raw),) * 3
        else:
            grid = tuple(
                int(value)
                for value in sequence(grid_raw, name="problem.grid")
            )
        if len(grid) != 3 or any(value <= 0 for value in grid):
            raise ValueError(
                "problem.grid must contain three positive integers"
            )
        advection = tuple(
            float(value)
            for value in sequence(
                raw.get("advection", [0.0, 0.0, 0.0]),
                name="problem.advection",
            )
        )
        if len(advection) != 3 or not all(
            np.isfinite(value) for value in advection
        ):
            raise ValueError(
                "problem.advection must contain three finite values"
            )
        c_min = float(raw.get("c_min", 1.0))
        c_max = float(raw.get("c_max", 1000.0))
        if (
            not np.isfinite(c_min)
            or not np.isfinite(c_max)
            or c_min <= 0.0
            or c_max < c_min
        ):
            raise ValueError(
                "problem c_min/c_max must be finite and satisfy "
                "0 < c_min <= c_max"
            )
        sampled_advection = (
            canonical_kind == SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION
        )
        advection_min = float(
            raw.get("advection_min", c_min if sampled_advection else 0.0)
        )
        advection_max = float(
            raw.get("advection_max", c_max if sampled_advection else 0.0)
        )
        if (
            not np.isfinite(advection_min)
            or not np.isfinite(advection_max)
            or advection_max < advection_min
        ):
            raise ValueError(
                "problem advection_min/advection_max must be finite and "
                "satisfy advection_min <= advection_max"
            )
        return cls(
            kind=kind,
            grid=grid,
            c_min=c_min,
            c_max=c_max,
            advection=advection,
            advection_min=advection_min,
            advection_max=advection_max,
        )


@dataclass(frozen=True)
class StreamSpec:
    cases: int
    warmup_cases: int
    online_cases: int
    instance_offset: int
    seed_groups: tuple[Any, ...]
    shuffle_seeds: tuple[Any, ...]
    cases_per_seed: int
    group_take: int
    expected_sha256: str = ""
    smoke: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StreamSpec":
        reject_unknown_keys(raw, STREAM_KEYS, name="stream")
        cases = int(raw["cases"])
        warmup_cases = int(raw.get("warmup_cases", 0))
        online_cases = int(raw.get("online_cases", cases))
        instance_offset = int(raw.get("instance_offset", 0))
        cases_per_seed = int(raw.get("cases_per_seed", cases))
        group_take = int(raw.get("group_take", cases))
        if cases <= 0:
            raise ValueError("stream.cases must be positive")
        if warmup_cases < 0 or online_cases < 0:
            raise ValueError(
                "stream warmup_cases and online_cases must be non-negative"
            )
        if warmup_cases + online_cases != cases:
            raise ValueError(
                "stream.cases must equal warmup_cases + online_cases"
            )
        if instance_offset < 0:
            raise ValueError("stream.instance_offset must be non-negative")
        if cases_per_seed <= 0 or group_take <= 0:
            raise ValueError(
                "stream cases_per_seed and group_take must be positive"
            )
        return cls(
            cases=cases,
            warmup_cases=warmup_cases,
            online_cases=online_cases,
            instance_offset=instance_offset,
            seed_groups=tuple(raw.get("seed_groups", ())),
            shuffle_seeds=tuple(raw.get("shuffle_seeds", ())),
            cases_per_seed=cases_per_seed,
            group_take=group_take,
            expected_sha256=str(raw.get("expected_sha256") or ""),
            smoke=bool(raw.get("smoke", False)),
        )


@dataclass(frozen=True)
class SeedSpec:
    base: int
    bandit: int
    controller: int
    method_order: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SeedSpec":
        reject_unknown_keys(raw, SEED_KEYS, name="seeds")
        return cls(
            base=int(raw["base"]),
            bandit=int(raw["bandit"]),
            controller=int(raw["controller"]),
            method_order=int(raw["method_order"]),
        )


@dataclass(frozen=True)
class CandidateScheduleSpec:
    mode: str = "explicit"
    max_selections_per_case: int = 3
    chunk_rounds: int = 256

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "CandidateScheduleSpec":
        unknown = set(raw) - CANDIDATE_SCHEDULE_KEYS
        if unknown:
            raise ValueError(
                "Unknown setup.candidate_schedule keys: "
                f"{sorted(unknown)}"
            )
        mode = str(raw.get("mode", "explicit"))
        if mode not in {"explicit", "aot"}:
            raise ValueError(
                "setup.candidate_schedule.mode must be explicit or aot"
            )
        return cls(
            mode=mode,
            max_selections_per_case=int(
                raw.get("max_selections_per_case", 3)
            ),
            chunk_rounds=int(raw.get("chunk_rounds", 256)),
        )


@dataclass(frozen=True)
class LinTsSpec:
    relative_sampling_scale: float = 0.15
    loss_scale_prior_sec: float = 0.1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LinTsSpec":
        unknown = set(raw) - LIN_TS_KEYS
        if unknown:
            raise ValueError(
                f"Unknown setup.lin_ts keys: {sorted(unknown)}"
            )
        return cls(
            relative_sampling_scale=float(
                raw.get("relative_sampling_scale", 0.15)
            ),
            loss_scale_prior_sec=float(
                raw.get("loss_scale_prior_sec", 0.1)
            ),
        )


@dataclass(frozen=True)
class SetupExperimentSpec:
    parameter_resolution: int
    action_space: str
    candidate_schedule: CandidateScheduleSpec
    configuration_spaces: Mapping[str, SetupConfigurationSpace]
    lin_ts: LinTsSpec
    replay_trajectory: Path | None
    shared_online_prefix: bool = False

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "SetupExperimentSpec":
        unknown = set(raw) - SETUP_KEYS
        if unknown:
            raise ValueError(f"Unknown setup config keys: {sorted(unknown)}")
        raw_spaces = mapping(
            raw.get("configuration_spaces", {}),
            name="setup.configuration_spaces",
        )
        spaces = {
            str(name): SetupConfigurationSpace.from_mapping(
                str(name),
                mapping(
                    space,
                    name=f"setup.configuration_spaces.{name}",
                ),
            )
            for name, space in raw_spaces.items()
        }
        action_space = str(raw.get("action_space", "full_cartesian"))
        if action_space != "full_cartesian":
            raise ValueError(
                "setup.action_space must equal full_cartesian"
            )
        replay_raw = raw.get("replay_trajectory")
        replay_token = (
            None if replay_raw is None else str(replay_raw).strip()
        )
        if replay_token == "":
            raise ValueError("setup.replay_trajectory cannot be empty")
        shared_online_prefix = raw.get("shared_online_prefix", False)
        if type(shared_online_prefix) is not bool:
            raise ValueError("setup.shared_online_prefix must be boolean")
        replay_trajectory = (
            None
            if replay_token is None
            else Path(replay_token)
        )
        return cls(
            parameter_resolution=int(raw["parameter_resolution"]),
            action_space=action_space,
            candidate_schedule=CandidateScheduleSpec.from_mapping(
                mapping(
                    raw.get("candidate_schedule", {}),
                    name="setup.candidate_schedule",
                )
            ),
            configuration_spaces=spaces,
            lin_ts=LinTsSpec.from_mapping(
                mapping(raw.get("lin_ts", {}), name="setup.lin_ts")
            ),
            replay_trajectory=replay_trajectory,
            shared_online_prefix=shared_online_prefix,
        )


@dataclass(frozen=True)
class ReportingSpec:
    progress_every: int = 100
    rolling_window: int = 100
    generate_plots: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReportingSpec":
        reject_unknown_keys(raw, REPORTING_KEYS, name="reporting")
        return cls(
            progress_every=int(raw.get("progress_every", 100)),
            rolling_window=int(raw.get("rolling_window", 100)),
            generate_plots=bool(raw.get("generate_plots", True)),
        )


@dataclass(frozen=True)
class JointSolveSpec:
    state: SolveStateSpec
    actions: SharedActionSpec
    recursive_mc: RecursiveMonteCarloLcbSpec
    recursive_lstdq_v1: RecursiveLstdqLcbSpec
    recursive_lstdq_v2: RecursiveLstdqV2LcbSpec
    recursive_lstdq_v3: RecursiveLstdqV3LcbSpec
    rblspi: RecursiveBlstdqSpec
    stagewise_lsvi: StagewiseLsviLcbSpec
    structured_model: StructuredModelBasedSpec
    recalibrated_lsvi: HierarchicalLsviLcbSpec
    trace_lambda: float
    smoother_profile: str = "legacy_l1_jacobi"
    controller_specs: Mapping[str, OnlineControllerBuildSpec] = field(
        default_factory=dict
    )
    ppo_model: Path = DEFAULT_PPO_MODEL

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        problem: ProblemSpec,
        solve_kinds: Iterable[str],
    ) -> "JointSolveSpec":
        reject_unknown_keys(raw, SOLVE_KEYS, name="solve")
        smoother_profile = str(raw.get("smoother_profile", "legacy_l1_jacobi"))
        if smoother_profile not in SMOOTHER_PROFILES:
            raise ValueError(f"solve.smoother_profile must be one of {SMOOTHER_PROFILES}")
        actions = expand_grid(
            raw.get("action_grid"),
            name="solve.action_grid",
        )
        rbf = mapping(raw.get("rbf"), name="solve.rbf")
        reject_unknown_keys(rbf, RBF_KEYS, name="solve.rbf")
        epsilon = mapping(
            raw.get("epsilon", {}),
            name="solve.epsilon",
        )
        reject_unknown_keys(
            epsilon,
            EPSILON_KEYS,
            name="solve.epsilon",
        )
        action_spec = SharedActionSpec(
            weights=actions,
            rbf_centers=expand_grid(
                rbf.get("centers"),
                name="solve.rbf.centers",
            ),
            rbf_sigma=float(rbf.get("sigma", 0.2)),
            epsilon=EpsilonScheduleSpec(
                start=float(epsilon.get("start", 0.30)),
                final=float(epsilon.get("final", 0.03)),
                decay_steps=float(epsilon.get("decay_steps", 20_000.0)),
            ),
            anchor_weight=actions[0],
            force_default_first_action=True,
        )
        max_cycles = int(raw.get("max_cycles", 50))
        state_spec = SolveStateSpec(
            tol=float(raw.get("tolerance", 1.0e-6)),
            max_cycles=max_cycles,
            c_max=float(problem.c_max),
            mode="setup_full",
            encoding_version=str(raw.get("state_encoding", "legacy_v1")),
        )

        lstdq = mapping(raw.get("lstdq", {}), name="solve.lstdq")
        lstdq_v2 = mapping(
            raw.get("lstdq_v2", {}),
            name="solve.lstdq_v2",
        )
        lstdq_v3 = mapping(
            raw.get("lstdq_v3", {}),
            name="solve.lstdq_v3",
        )
        rblspi_raw = mapping(
            raw.get("rblspi", {}),
            name="solve.rblspi",
        )
        recursive_mc_raw = mapping(
            raw.get("recursive_mc", {}),
            name="solve.recursive_mc",
        )
        lsvi = mapping(raw.get("lsvi", {}), name="solve.lsvi")
        structured = mapping(
            raw.get("structured_model", {}),
            name="solve.structured_model",
        )
        recalibrated = mapping(
            raw.get("recalibrated_lsvi", {}),
            name="solve.recalibrated_lsvi",
        )
        lstdq_specs = RecursiveLstdqFamilySpecs.from_mappings(
            lstdq,
            lstdq_v2,
            lstdq_v3,
        )
        lsvi_specs = LsviFamilySpecs.from_mappings(
            lsvi,
            recalibrated,
            horizon=max_cycles,
        )
        lstdq_v1_spec = lstdq_specs.v1
        lstdq_v2_spec = lstdq_specs.v2
        lstdq_v3_spec = lstdq_specs.v3
        trace_lambda = lstdq_specs.trace_lambda
        recursive_mc_spec = RecursiveMonteCarloLcbSpec.from_mapping(
            recursive_mc_raw
        )
        rblspi_spec = RecursiveBlstdqSpec.from_mapping(rblspi_raw)
        stagewise_spec = lsvi_specs.stagewise
        structured_spec = StructuredModelBasedSpec.from_mapping(structured)
        recalibrated_spec = lsvi_specs.hierarchical
        algorithms = {
            "recursive_mc": recursive_mc_spec,
            "recursive_lstdq_v1": lstdq_v1_spec,
            "recursive_lstdq_v2": lstdq_v2_spec,
            "recursive_lstdq_v3": lstdq_v3_spec,
            "rblspi": rblspi_spec,
            "stagewise_lsvi": stagewise_spec,
            "structured_model_based": structured_spec,
            "recalibrated_lsvi": recalibrated_spec,
        }
        requested_kinds = {
            str(kind) for kind in solve_kinds if kind in ONLINE_SOLVE_KINDS
        }
        controller_specs: Dict[str, OnlineControllerBuildSpec] = {}
        for kind in requested_kinds:
            algorithm = algorithms[kind]
            requested_trace = (
                trace_lambda
                if kind
                in {
                    "recursive_lstdq_v1",
                    "recursive_lstdq_v2",
                    "recursive_lstdq_v3",
                }
                else None
            )
            controller_specs[kind] = OnlineControllerBuildSpec(
                kind=cast(OnlineSolveKind, kind),
                state=state_spec,
                actions=action_spec,
                algorithm=algorithm,
                trace_lambda=requested_trace,
            )

        ppo_model_raw = raw.get("ppo_model")
        return cls(
            state=state_spec,
            actions=action_spec,
            recursive_mc=recursive_mc_spec,
            recursive_lstdq_v1=lstdq_v1_spec,
            recursive_lstdq_v2=lstdq_v2_spec,
            recursive_lstdq_v3=lstdq_v3_spec,
            rblspi=rblspi_spec,
            stagewise_lsvi=stagewise_spec,
            structured_model=structured_spec,
            recalibrated_lsvi=recalibrated_spec,
            trace_lambda=trace_lambda,
            smoother_profile=smoother_profile,
            controller_specs=controller_specs,
            ppo_model=(
                DEFAULT_PPO_MODEL
                if ppo_model_raw is None
                else Path(str(ppo_model_raw))
            ),
        )


@dataclass(frozen=True)
class JointExperimentSpec:
    schema_version: int
    name: str
    description: str
    output_dir: Path
    problem: ProblemSpec
    stream: StreamSpec
    seeds: SeedSpec
    setup: SetupExperimentSpec
    solve: JointSolveSpec
    methods: tuple[ComposableMethodSpec, ...]
    reporting: ReportingSpec
    failure_penalty_sec: float | None = None


@dataclass(frozen=True)
class JointExperimentRuntimeConfig:
    """Flat, typed configuration consumed by the canonical 4K runner."""

    output_dir: Path
    study_mode: str
    ppo_model: Path
    ppo_action_mode: str
    ppo_w_center: float
    ppo_w_scale: float
    ppo_initial_observation_weight: float
    ppo_force_default_first_action: bool
    ppo_default_first_weight: float
    seed: int
    bandit_seed: int
    controller_seed: int
    method_order_seed: int
    train_cases: int
    warmup_cases: int
    online_cases: int
    instance_offset: int
    train_seed_groups: str
    train_shuffle_seeds: str
    train_cases_per_seed: int
    train_group_take: int
    expected_stream_hash: str
    problem: str
    grid_n: int
    grid_shape: str
    advection: str
    advection_min: float
    advection_max: float
    setup_param_resolution: int
    c_min: float
    c_max: float
    tol: float
    max_cycles: int
    smoother_profile: str
    setup_action_space: str
    setup_candidate_mode: str
    setup_replay_trajectory: Path | None
    aot_max_selections_per_case: int
    aot_schedule_chunk_rounds: int
    lin_ts_relative_sampling_scale: float
    lin_ts_loss_scale_prior: float
    shared_action_profile: str
    weights: str
    action_rbf_centers: str
    action_rbf_sigma: float
    alphas: str
    trace_lambdas: str
    epsilon_start: float
    epsilon_final: float
    epsilon_decay_steps: float
    q_max_sec: float
    uncertainty_beta: float
    uncertainty_ridge: float
    uncertainty_td_floor_sec: float
    lsvi_ridge: float
    lsvi_beta: float
    lsvi_residual_floor_sec: float
    lsvi_refit_interval_episodes: int
    recursive_mc_ridge: float
    recursive_mc_beta: float
    recursive_mc_residual_floor_sec: float
    recursive_mc_episode_half_life: float
    recursive_lstdq_ridge: float
    recursive_lstdq_beta: float
    recursive_lstdq_lambda: float
    recursive_lstdq_residual_floor_sec: float
    recursive_lstdq_lcb_lower_bound_sec: float | None
    recursive_lstdq_v2_beta: float
    recursive_lstdq_v2_coverage_ridge: float
    recursive_lstdq_v2_residual_window: int
    recursive_lstdq_v2_min_samples: int
    recursive_lstdq_v3_beta: float
    rblspi_prior_precision: float
    rblspi_noise_precision: float
    rblspi_gram_ridge: float
    structured_model_ridge: float
    structured_model_min_samples: int
    structured_model_scale_window: int
    recalibrated_lsvi_beta: float
    recalibrated_lsvi_refit_sweeps: int
    recalibrated_lsvi_shrinkage_samples: float
    progress_every: int
    methods: tuple[ComposableMethodSpec, ...]
    include_default_setup_baseline: bool
    reuse_warmup: bool
    smoke: bool
    setup_configuration_spaces: Dict[str, SetupConfigurationSpace]
    solve_controller_specs: Dict[str, OnlineControllerBuildSpec]
    joint_experiment_spec: JointExperimentSpec
    shared_online_prefix: bool = False
    failure_penalty_sec: float | None = None


def parse_joint_experiment_config(
    config: Mapping[str, Any],
    *,
    output_dir_override: Path | None = None,
) -> JointExperimentSpec:
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    schema_version = int(config.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")

    configured_output = config.get("output_dir")
    output_dir = output_dir_override or (
        None
        if configured_output is None
        else Path(str(configured_output))
    )
    if output_dir is None:
        raise ValueError("Set output_dir in the config or pass --output-dir")

    problem = ProblemSpec.from_mapping(
        mapping(config.get("problem"), name="problem")
    )
    stream = StreamSpec.from_mapping(
        mapping(config.get("stream"), name="stream")
    )
    seeds = SeedSpec.from_mapping(
        mapping(config.get("seeds"), name="seeds")
    )
    setup = SetupExperimentSpec.from_mapping(
        mapping(config.get("setup"), name="setup")
    )
    methods = tuple(
        ComposableMethodSpec.from_mapping(
            mapping(item, name="methods[]")
        )
        for item in sequence(config.get("methods"), name="methods")
    )
    if not methods:
        raise ValueError("methods cannot be empty")
    names = tuple(method.name for method in methods)
    if len(set(names)) != len(names):
        raise ValueError("method ids must be unique")
    if setup.replay_trajectory is not None:
        invalid_setup_methods = [
            method.name
            for method in methods
            if method.setup_kind not in ONLINE_SETUP_KINDS
        ]
        if invalid_setup_methods:
            raise ValueError(
                "setup.replay_trajectory requires online setup methods: "
                f"{invalid_setup_methods}"
            )
        if stream.warmup_cases or any(
            method.setup_warmup_cases for method in methods
        ):
            raise ValueError(
                "setup.replay_trajectory requires zero setup warmup cases"
            )
    solve = JointSolveSpec.from_mapping(
        mapping(config.get("solve"), name="solve"),
        problem=problem,
        solve_kinds=(method.solve_kind for method in methods),
    )
    reporting = ReportingSpec.from_mapping(
        mapping(config.get("reporting", {}), name="reporting")
    )
    feedback = mapping(config.get("failure_feedback", {}), name="failure_feedback")
    reject_unknown_keys(feedback, {"mode", "penalty_sec"}, name="failure_feedback")
    mode = feedback.get("mode", "rollback_unrecovered")
    if mode not in {"rollback_unrecovered", "budgeted_penalty"}:
        raise ValueError("Unknown failure_feedback.mode")
    penalty = validate_failure_penalty(feedback.get("penalty_sec"))
    if (mode == "budgeted_penalty") != (penalty is not None):
        raise ValueError("budgeted_penalty requires explicit penalty_sec; rollback_unrecovered must omit it")
    if penalty is not None:
        if stream.warmup_cases or any(m.setup_warmup_cases for m in methods):
            raise ValueError("budgeted_penalty requires online setup learning from problem 1")
        if setup.replay_trajectory is not None or any(m.solve_kind == "ppo" for m in methods):
            raise ValueError("budgeted_penalty is not supported for frozen setup replay or PPO")
    return JointExperimentSpec(
        schema_version=schema_version,
        name=str(config.get("name", "")),
        description=str(config.get("description", "")),
        output_dir=Path(output_dir),
        problem=problem,
        stream=stream,
        seeds=seeds,
        setup=setup,
        solve=solve,
        methods=methods,
        reporting=reporting,
        failure_penalty_sec=penalty,
    )


def runtime_config_from_spec(
    spec: JointExperimentSpec,
) -> JointExperimentRuntimeConfig:
    """Resolve a hierarchical experiment spec for the canonical 4K runner."""

    solve = spec.solve
    actions = solve.actions
    epsilon = actions.epsilon
    v1 = solve.recursive_lstdq_v1
    v2 = solve.recursive_lstdq_v2
    v3 = solve.recursive_lstdq_v3
    recursive_mc = solve.recursive_mc
    lsvi = solve.stagewise_lsvi
    structured = solve.structured_model
    recalibrated = solve.recalibrated_lsvi
    return JointExperimentRuntimeConfig(
        output_dir=spec.output_dir,
        study_mode="composable",
        ppo_model=solve.ppo_model,
        ppo_action_mode="continuous_absolute",
        ppo_w_center=1.5,
        ppo_w_scale=0.5,
        ppo_initial_observation_weight=1.0,
        ppo_force_default_first_action=True,
        ppo_default_first_weight=1.0,
        seed=spec.seeds.base,
        bandit_seed=spec.seeds.bandit,
        controller_seed=spec.seeds.controller,
        method_order_seed=spec.seeds.method_order,
        train_cases=spec.stream.cases,
        warmup_cases=spec.stream.warmup_cases,
        online_cases=spec.stream.online_cases,
        instance_offset=spec.stream.instance_offset,
        train_seed_groups=csv(spec.stream.seed_groups),
        train_shuffle_seeds=csv(spec.stream.shuffle_seeds),
        train_cases_per_seed=spec.stream.cases_per_seed,
        train_group_take=spec.stream.group_take,
        expected_stream_hash=spec.stream.expected_sha256,
        problem=spec.problem.kind,
        grid_n=max(spec.problem.grid),
        grid_shape=csv(spec.problem.grid),
        advection=csv(spec.problem.advection),
        advection_min=spec.problem.advection_min,
        advection_max=spec.problem.advection_max,
        setup_param_resolution=spec.setup.parameter_resolution,
        c_min=spec.problem.c_min,
        c_max=spec.problem.c_max,
        tol=solve.state.tol,
        max_cycles=solve.state.max_cycles,
        smoother_profile=solve.smoother_profile,
        setup_action_space=spec.setup.action_space,
        setup_candidate_mode=spec.setup.candidate_schedule.mode,
        setup_replay_trajectory=spec.setup.replay_trajectory,
        aot_max_selections_per_case=(
            spec.setup.candidate_schedule.max_selections_per_case
        ),
        aot_schedule_chunk_rounds=(
            spec.setup.candidate_schedule.chunk_rounds
        ),
        lin_ts_relative_sampling_scale=(
            spec.setup.lin_ts.relative_sampling_scale
        ),
        lin_ts_loss_scale_prior=(
            spec.setup.lin_ts.loss_scale_prior_sec
        ),
        shared_action_profile="1to2_step0p1",
        weights=csv(actions.weights),
        action_rbf_centers=csv(actions.rbf_centers),
        action_rbf_sigma=actions.rbf_sigma,
        alphas="0.001",
        trace_lambdas="0.8",
        epsilon_start=epsilon.start,
        epsilon_final=epsilon.final,
        epsilon_decay_steps=epsilon.decay_steps,
        q_max_sec=0.1,
        uncertainty_beta=1.0,
        uncertainty_ridge=1.0,
        uncertainty_td_floor_sec=1.0e-3,
        lsvi_ridge=lsvi.ridge,
        lsvi_beta=lsvi.uncertainty_beta,
        lsvi_residual_floor_sec=lsvi.residual_floor_sec,
        lsvi_refit_interval_episodes=lsvi.refit_interval_episodes,
        recursive_mc_ridge=recursive_mc.ridge,
        recursive_mc_beta=recursive_mc.uncertainty_beta,
        recursive_mc_residual_floor_sec=(
            recursive_mc.residual_floor_sec
        ),
        recursive_mc_episode_half_life=(
            recursive_mc.episode_half_life
        ),
        recursive_lstdq_ridge=v1.ridge,
        recursive_lstdq_beta=v1.uncertainty_beta,
        recursive_lstdq_lambda=solve.trace_lambda,
        recursive_lstdq_residual_floor_sec=(
            v1.residual_floor_sec
        ),
        recursive_lstdq_lcb_lower_bound_sec=(
            v1.lcb_lower_bound_sec
        ),
        recursive_lstdq_v2_beta=v2.uncertainty_beta,
        recursive_lstdq_v2_coverage_ridge=v2.coverage_ridge,
        recursive_lstdq_v2_residual_window=(
            v2.residual_scale_window
        ),
        recursive_lstdq_v2_min_samples=(
            v2.residual_scale_min_samples
        ),
        recursive_lstdq_v3_beta=v3.uncertainty_beta,
        rblspi_prior_precision=solve.rblspi.prior_precision,
        rblspi_noise_precision=solve.rblspi.noise_precision,
        rblspi_gram_ridge=solve.rblspi.gram_ridge,
        structured_model_ridge=structured.ridge,
        structured_model_min_samples=structured.minimum_samples,
        structured_model_scale_window=structured.scale_window,
        recalibrated_lsvi_beta=recalibrated.uncertainty_beta,
        recalibrated_lsvi_refit_sweeps=recalibrated.refit_sweeps,
        recalibrated_lsvi_shrinkage_samples=(
            recalibrated.residual_shrinkage_samples
        ),
        progress_every=spec.reporting.progress_every,
        methods=spec.methods,
        include_default_setup_baseline=False,
        reuse_warmup=False,
        smoke=spec.stream.smoke,
        setup_configuration_spaces=dict(
            spec.setup.configuration_spaces
        ),
        solve_controller_specs=dict(solve.controller_specs),
        joint_experiment_spec=spec,
        shared_online_prefix=spec.setup.shared_online_prefix,
        failure_penalty_sec=spec.failure_penalty_sec,
    )


def legacy_namespace_from_spec(
    spec: JointExperimentSpec,
) -> argparse.Namespace:
    """Convert the typed runtime config for legacy Namespace callers."""

    runtime = runtime_config_from_spec(spec)
    values = {
        config_field.name: getattr(runtime, config_field.name)
        for config_field in fields(runtime)
    }
    # ``method_specs`` is the historical CLI representation.  Keep it only on
    # the compatibility Namespace; the canonical runtime carries typed methods.
    values["method_specs"] = [
        method.to_runner_token() for method in runtime.methods
    ]
    return argparse.Namespace(**values)


__all__ = [
    "CandidateScheduleSpec",
    "JointExperimentSpec",
    "JointExperimentRuntimeConfig",
    "JointSolveSpec",
    "LinTsSpec",
    "ProblemSpec",
    "ReportingSpec",
    "SCHEMA_VERSION",
    "SETUP_KEYS",
    "SeedSpec",
    "SetupExperimentSpec",
    "StreamSpec",
    "TOP_LEVEL_KEYS",
    "csv",
    "expand_grid",
    "legacy_namespace_from_spec",
    "mapping",
    "parse_joint_experiment_config",
    "runtime_config_from_spec",
    "sequence",
]
