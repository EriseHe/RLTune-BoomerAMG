from __future__ import annotations

import platform
from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence


def _parse_float_values(raw: str) -> tuple[float, ...]:
    return tuple(
        float(part.strip())
        for part in str(raw).split(",")
        if part.strip()
    )


def _shared_controller_protocol(
    args: Any,
    *,
    action_profile: str,
) -> Dict[str, Any]:
    """Metadata shared by the online linear solve-controller families."""
    return {
        "state": "setup_full",
        "action_profile": action_profile,
        "actions": list(_parse_float_values(args.weights)),
        "action_basis": {
            "mode": "compact_rbf",
            "centers": list(_parse_float_values(args.action_rbf_centers)),
            "sigma": float(args.action_rbf_sigma),
        },
        "epsilon": {
            "start": float(args.epsilon_start),
            "final": float(args.epsilon_final),
            "decay_steps": float(args.epsilon_decay_steps),
            "kind": "uniform epsilon-greedy floor",
        },
        "prepared_solver_default_forced_once": True,
        "frozen": False,
    }


def _recursive_lstdq_v1_protocol(
    args: Any,
    shared_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    return dict(shared_protocol) | {
        "version": "v1",
        "target": "on-policy LSTDQ(lambda) Bellman equation",
        "ridge": float(args.recursive_lstdq_ridge),
        "trace_lambda": float(args.recursive_lstdq_lambda),
        "uncertainty_beta": float(args.recursive_lstdq_beta),
        "uncertainty": "unclipped post-fit sandwich covariance",
        "lcb_lower_bound_sec": float(
            args.recursive_lstdq_lcb_lower_bound_sec
        ),
    }


def _recursive_lstdq_v2_protocol(
    args: Any,
    shared_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    return dict(shared_protocol) | {
        "version": "v2",
        "target": "same recursive LSTDQ mean as v1",
        "ridge": float(args.recursive_lstdq_ridge),
        "trace_lambda": float(args.recursive_lstdq_lambda),
        "uncertainty_beta": float(args.recursive_lstdq_v2_beta),
        "uncertainty": "rolling-MAD-scaled feature coverage",
        "coverage_ridge": float(
            args.recursive_lstdq_v2_coverage_ridge
        ),
        "residual_window": int(
            args.recursive_lstdq_v2_residual_window
        ),
        "minimum_scale_samples": int(
            args.recursive_lstdq_v2_min_samples
        ),
        "residual_floor_sec": float(
            args.recursive_lstdq_residual_floor_sec
        ),
        "lcb_lower_bound_sec": float(
            args.recursive_lstdq_lcb_lower_bound_sec
        ),
    }


def _structured_model_protocol(
    args: Any,
    shared_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    return dict(shared_protocol) | {
        "target": [
            "native_cycle_cost",
            "signed_log_residual_progress",
        ],
        "estimator": "shared recursive least squares",
        "planning": "receding time per log-residual reduction",
        "recovery_cost_in_physical_model": False,
        "ridge": float(args.structured_model_ridge),
        "minimum_samples": int(args.structured_model_min_samples),
        "scale_window": int(args.structured_model_scale_window),
    }


def _recalibrated_lsvi_protocol(
    args: Any,
    shared_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    return dict(shared_protocol) | {
        "target": "stagewise optimistic Bellman backup",
        "horizon": int(args.max_cycles),
        "ridge": float(args.lsvi_ridge),
        "uncertainty_beta": float(args.recalibrated_lsvi_beta),
        "residual_floor_sec": float(args.lsvi_residual_floor_sec),
        "batch_size_episodes": int(
            args.lsvi_refit_interval_episodes
        ),
        "backward_shared_sweeps": int(
            args.recalibrated_lsvi_refit_sweeps
        ),
        "residual_shrinkage_samples": float(
            args.recalibrated_lsvi_shrinkage_samples
        ),
        "value_cap": "maximum observed realized return",
    }


def _base_protocol(
    args: Any,
    *,
    git_revision: str,
    stream_manifest: Mapping[str, Any],
    methods: Sequence[str],
    family_by_method: Mapping[str, str],
    composable_specs: Sequence[Any],
    candidates: Sequence[Any],
    setup_configuration_spaces: Mapping[str, Any],
    bandit_methods: Sequence[str],
    behavior_modes: Sequence[str],
    setup_bandit_kinds: Sequence[str],
) -> Dict[str, Any]:
    resolved_setup_spaces = {
        name: configuration_space.as_dict()
        for name, configuration_space in setup_configuration_spaces.items()
    }
    solve_activation_cases = {
        spec.name: int(spec.solve_activation_case)
        for spec in composable_specs
        if int(spec.solve_activation_case) > 0
    }
    setup_from_scratch = int(args.warmup_cases) == 0
    dynamic_activation_rules = {
        spec.name: asdict(spec.solve_activation)
        for spec in composable_specs if spec.solve_activation is not None
    }
    joint_from_scratch = setup_from_scratch and not (
        solve_activation_cases or dynamic_activation_rules
    )
    if solve_activation_cases or dynamic_activation_rules:
        purpose = (
            f"{int(args.online_cases)} persistent joint-online instances; "
            "setup-bandit learning starts at instance 1 and solve controllers "
            "activate only after their default-solve prefixes"
        )
    elif setup_from_scratch:
        purpose = (
            f"{int(args.online_cases)}-instance joint-online setup-bandit and "
            "solve-controller learning from scratch"
        )
    else:
        purpose = (
            f"up to {int(args.warmup_cases)} setup-bandit warmup instances "
            f"+ {int(args.online_cases)} persistent joint-online instances"
        )
    return {
        "git_revision": git_revision,
        "platform": platform.platform(),
        "purpose": purpose,
        "stream": stream_manifest,
        "stream_partition": {
            "warmup": [0, int(args.warmup_cases)],
            "online": [int(args.warmup_cases), int(args.train_cases)],
        },
        "solve": {
            "tolerance": float(args.tol),
            "max_cycles": int(args.max_cycles),
            "smoother_profile": str(
                getattr(args, "smoother_profile", "legacy_l1_jacobi")
            ),
        },
        "methods": list(methods),
        "timing": {
            "schema_version": 2,
            "controller_components": ["feature", "decision", "update", "lifecycle"],
            "lifecycle": "rollback snapshot, episode initialization and rollback",
            "setup_feedback_includes_controller_lifecycle": True,
            "solve_td_target": "native cycle cost plus applicable terminal recovery",
            "method_wall_scope": (
                "one complete method call, including matrix construction and wrapper work; "
                "excluding outer trajectory I/O, checkpointing and plotting"
            ),
            "method_wall_in_learning_feedback": False,
        },
        "families": family_by_method,
        "method_labels": {
            spec.name: spec.label for spec in composable_specs
        },
        "method_specs": [asdict(spec) for spec in composable_specs],
        "candidates": [
            asdict(candidate) | {"name": candidate.name}
            for candidate in candidates
        ],
        "setup_bandit": {
            "seed": int(args.bandit_seed),
            "matrix_grid_n": int(args.grid_n),
            "problem": str(
                getattr(args, "problem", "scalar_anisotropic_diffusion")
            ),
            "grid_shape": list(
                stream_manifest.get("grid", [int(args.grid_n)] * 3)
            ),
            "setup_param_resolution": int(args.setup_param_resolution),
            "warmup_instances": int(args.warmup_cases),
            "method_warmup_instances": {
                spec.name: int(spec.setup_warmup_cases)
                for spec in composable_specs
                if spec.setup_kind in setup_bandit_kinds
            },
            "solve_activation_case": solve_activation_cases,
            **(
                {"dynamic_solve_activation": dynamic_activation_rules}
                if dynamic_activation_rules else {}
            ),
            "setup_from_scratch": setup_from_scratch,
            "joint_from_scratch": joint_from_scratch,
            "common_snapshot_mutable_state_cloned": not bool(
                setup_configuration_spaces
            ),
            "immutable_action_catalog_and_g_actions_shared": not bool(
                setup_configuration_spaces
            ),
            "per_branch_parameter_space": bool(setup_configuration_spaces),
            "candidate_mode": str(args.setup_candidate_mode),
            "aot_schedule": (
                {
                    "max_selections_per_case": int(
                        args.aot_max_selections_per_case
                    ),
                    "scheduled_rounds_per_branch": (
                        int(args.online_cases) + int(args.warmup_cases)
                    ) * int(args.aot_max_selections_per_case),
                    "pairing": "fixed-width block per online instance",
                    "payload": "candidate_ids_only",
                    "action_features": "shared_factorized_ram_cache",
                }
                if str(args.setup_candidate_mode) == "aot"
                else None
            ),
            "configuration_spaces": resolved_setup_spaces,
            "method_configuration_spaces": {
                spec.name: spec.setup_space
                for spec in composable_specs
                if spec.setup_kind in setup_bandit_kinds
            },
            "frozen_after_warmup": False,
            "independent_updates_per_method": True,
            "fallback_runtime_included_in_feedback": True,
            "linear_ts_v2": {
                "posterior": "Gaussian linear posterior",
                "precision_factor": "rank-one Cholesky updates",
                "relative_sampling_scale": float(
                    args.lin_ts_relative_sampling_scale
                ),
                "loss_scale_prior_sec": float(
                    args.lin_ts_loss_scale_prior
                ),
                "runtime_scale_reference": "first successful observation",
                "posterior_rng_separate_from_candidate_rng": True,
            },
            "failure_head": {
                "representation": "shared_context_action_features",
                "ridge": 1.0,
                "beta": 2.0,
            },
            "setup_construction_reselection": {
                "max_learned_attempts_including_first": 3,
                "temporary_exact_arm_exclusion": True,
                "ranking": ["failure_ucb", "runtime_lcb"],
                "runtime_target": "realized_suffix_cost",
            },
            "solve_failure_reselection": False,
            "default_fallback_attempts": 1,
            "online_branches": list(bandit_methods),
        },
        "sarsa": {
            "algorithm": "true-online SARSA(lambda)",
            "update_frequency": (
                "one update per AMG cycle plus terminal update"
            ),
            "state": "setup_full",
            "actions": list(_parse_float_values(args.weights)),
            "behavior_modes": list(behavior_modes),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "uncertainty": {
                "beta": float(args.uncertainty_beta),
                "ridge": float(args.uncertainty_ridge),
                "td_floor_sec": float(
                    args.uncertainty_td_floor_sec
                ),
            },
            "prepared_solver_default_forced_once": True,
            "frozen": False,
        },
        "ppo": {
            "model": str(args.ppo_model),
            "frozen": True,
            "action_mode": str(args.ppo_action_mode),
            "w_center": float(args.ppo_w_center),
            "w_scale": float(args.ppo_w_scale),
            "initial_observation_weight": float(
                args.ppo_initial_observation_weight
            ),
            "prepared_solver_default_forced_once": bool(
                args.ppo_force_default_first_action
            ),
        },
        "randomized_method_order_per_instance": True,
        "method_order_seed": int(args.method_order_seed),
    }


def build_joint_protocol(
    args: Any,
    *,
    git_revision: str,
    stream_manifest: Mapping[str, Any],
    methods: Sequence[str],
    family_by_method: Mapping[str, str],
    composable_specs: Sequence[Any],
    candidates: Sequence[Any],
    setup_configuration_spaces: Mapping[str, Any],
    bandit_methods: Sequence[str],
    behavior_modes: Sequence[str],
    setup_bandit_kinds: Sequence[str],
    recursive_mc_method: str,
    recursive_lstdq_method: str,
    batched_lsvi_method: str,
    solve_controller_protocols: Mapping[
        str,
        Mapping[str, Any],
    ] | None = None,
) -> Dict[str, Any]:
    """Build the persisted experiment protocol without running an experiment."""
    protocol = _base_protocol(
        args,
        git_revision=git_revision,
        stream_manifest=stream_manifest,
        methods=methods,
        family_by_method=family_by_method,
        composable_specs=composable_specs,
        candidates=candidates,
        setup_configuration_spaces=setup_configuration_spaces,
        bandit_methods=bandit_methods,
        behavior_modes=behavior_modes,
        setup_bandit_kinds=setup_bandit_kinds,
    )
    if args.study_mode == "composable":
        protocol.pop("sarsa")
        if not any(spec.solve_kind == "ppo" for spec in composable_specs):
            protocol.pop("ppo")
        if solve_controller_protocols is None:
            raise ValueError(
                "Composable protocol metadata must come from controller bundles"
            )
        resolved_controller_protocols = {
            str(method): dict(metadata)
            for method, metadata in solve_controller_protocols.items()
        }
        protocol["solve_controllers"] = resolved_controller_protocols
        section_by_kind = {
            "recursive_mc": "recursive_mc_lcb",
            "recursive_lstdq_v1": "recursive_lstdq_lcb",
            "recursive_lstdq_v2": "recursive_lstdq_v2_lcb",
            "recursive_lstdq_v3": "recursive_lstdq_v3_lcb",
            "rblspi": "recursive_blstdq_rblspi",
            "stagewise_lsvi": "stagewise_lsvi_lcb",
            "structured_model_based": "structured_model_based",
            "recalibrated_lsvi": "recalibrated_lsvi_lcb",
        }
        for spec in composable_specs:
            metadata = resolved_controller_protocols.get(spec.name)
            section = section_by_kind.get(spec.solve_kind)
            if metadata is not None and section is not None:
                protocol.setdefault(section, metadata)
    elif args.study_mode == "lsvi_lcb":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        protocol["stagewise_lsvi_lcb"] = {
            "state": "setup_full",
            "action_profile": str(args.shared_action_profile),
            "actions": list(_parse_float_values(args.weights)),
            "action_basis": {
                "mode": "compact_rbf",
                "centers": list(
                    _parse_float_values(args.action_rbf_centers)
                ),
                "sigma": float(args.action_rbf_sigma),
            },
            "horizon": int(args.max_cycles),
            "ridge": float(args.lsvi_ridge),
            "uncertainty_beta": float(args.lsvi_beta),
            "residual_floor_sec": float(args.lsvi_residual_floor_sec),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "update": "backward refit after every solve",
            "frozen": False,
        }
    elif args.study_mode == "solve_controller_screen":
        protocol.pop("sarsa")
        protocol.pop("ppo")
        shared_protocol = _shared_controller_protocol(
            args,
            action_profile=str(args.shared_action_profile),
        )
        protocol["recursive_lstdq_lcb"] = (
            _recursive_lstdq_v1_protocol(args, shared_protocol)
        )
        protocol["recursive_lstdq_v2_lcb"] = (
            _recursive_lstdq_v2_protocol(args, shared_protocol)
        )
        protocol["structured_model_based"] = (
            _structured_model_protocol(args, shared_protocol)
        )
        protocol["recalibrated_lsvi_lcb"] = (
            _recalibrated_lsvi_protocol(args, shared_protocol)
        )
    elif args.study_mode in {
        "recursive_lcb_suite",
        "recursive_lcb_ppo",
        "recursive_lstdq_lcb",
    }:
        protocol.pop("sarsa")
        if args.study_mode != "recursive_lcb_ppo":
            protocol.pop("ppo")
        shared_protocol = _shared_controller_protocol(
            args,
            action_profile=str(args.shared_action_profile),
        )
        if recursive_mc_method in methods:
            protocol["recursive_mc_lcb"] = dict(shared_protocol) | {
                "target": "undiscounted episodic cost-to-go",
                "estimator": (
                    "episode-forgetting recursive least squares"
                ),
                "ridge": float(args.recursive_mc_ridge),
                "uncertainty_beta": float(args.recursive_mc_beta),
                "residual_floor_sec": float(
                    args.recursive_mc_residual_floor_sec
                ),
                "episode_half_life": float(
                    args.recursive_mc_episode_half_life
                ),
                "uncertainty": (
                    "post-fit episode-cluster sandwich covariance"
                ),
                "history_storage": "none after episode update",
            }
        if recursive_lstdq_method in methods:
            protocol["recursive_lstdq_lcb"] = dict(shared_protocol) | {
                "target": "on-policy LSTDQ(lambda) Bellman equation",
                "estimator": (
                    "recursive LSTD with Sherman-Morrison inverse"
                ),
                "ridge": float(args.recursive_lstdq_ridge),
                "trace_lambda": float(args.recursive_lstdq_lambda),
                "uncertainty_beta": float(args.recursive_lstdq_beta),
                "lcb_lower_bound_sec": float(
                    args.recursive_lstdq_lcb_lower_bound_sec
                ),
                "uncertainty": (
                    "unclipped post-fit sandwich covariance"
                ),
                "history_storage": "none",
            }
        if batched_lsvi_method in methods:
            protocol["batched_lsvi_lcb"] = dict(shared_protocol) | {
                "target": "stagewise optimistic Bellman backup",
                "horizon": int(args.max_cycles),
                "ridge": float(args.lsvi_ridge),
                "uncertainty_beta": float(args.lsvi_beta),
                "residual_floor_sec": float(
                    args.lsvi_residual_floor_sec
                ),
                "batch_size_episodes": int(
                    args.lsvi_refit_interval_episodes
                ),
                "policy_updates": (
                    int(args.online_cases)
                    // int(args.lsvi_refit_interval_episodes)
                ),
                "update": (
                    "exact all-history backward refit at fixed "
                    "batch boundaries"
                ),
            }
    return protocol
