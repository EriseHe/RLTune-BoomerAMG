from __future__ import annotations
import platform
from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence


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
    setup_bandit_kinds: Sequence[str],
) -> Dict[str, Any]:
    resolved_setup_spaces = {
        name: configuration_space.as_dict()
        for (name, configuration_space) in setup_configuration_spaces.items()
    }
    solve_activation_cases = {
        spec.name: int(spec.solve_activation_case)
        for spec in composable_specs
        if int(spec.solve_activation_case) > 0
    }
    setup_from_scratch = int(args.warmup_cases) == 0
    dynamic_activation_rules = {
        spec.name: asdict(spec.solve_activation)
        for spec in composable_specs
        if spec.solve_activation is not None
    }
    joint_from_scratch = setup_from_scratch and (
        not (solve_activation_cases or dynamic_activation_rules)
    )
    if solve_activation_cases or dynamic_activation_rules:
        purpose = f"{int(args.online_cases)} persistent joint-online instances; setup-bandit learning starts at instance 1 and solve controllers activate only after their default-solve prefixes"
    elif setup_from_scratch:
        purpose = f"{int(args.online_cases)}-instance joint-online setup-bandit and solve-controller learning from scratch"
    else:
        purpose = f"up to {int(args.warmup_cases)} setup-bandit warmup instances + {int(args.online_cases)} persistent joint-online instances"
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
        "failure_feedback": {
            "mode": "rollback_unrecovered"
            if getattr(args, "failure_penalty_sec", None) is None
            else "budgeted_penalty",
            "penalty_sec": getattr(args, "failure_penalty_sec", None),
            "setup_target": "cost of successfully completed primary/recovery procedure plus selection and previous-update estimate; unrecovered observations rolled back"
            if getattr(args, "failure_penalty_sec", None) is None
            else "measured protocol cost plus selection and previous-update estimate, plus final-failure penalty",
            "solve_target": "native cycle costs plus successful terminal recovery; unrecovered episodes rolled back"
            if getattr(args, "failure_penalty_sec", None) is None
            else "native cycle costs, terminal recovery cost and final-failure penalty",
            "penalty_is_measured_runtime": False,
            "failure_head_label": "primary attempt failure; used only for setup-construction reselection",
        },
        "timing": {
            "schema_version": 2,
            "controller_components": ["feature", "decision", "update", "lifecycle"],
            "lifecycle": "rollback snapshot, episode initialization and rollback",
            "setup_feedback_includes_controller_lifecycle": True,
            "solve_td_target": "native cycle cost plus applicable terminal recovery",
            "method_wall_scope": "one complete method call, including matrix construction and wrapper work; excluding outer trajectory I/O, checkpointing and plotting",
            "method_wall_in_learning_feedback": False,
        },
        "families": family_by_method,
        "method_labels": {spec.name: spec.label for spec in composable_specs},
        "method_specs": [asdict(spec) for spec in composable_specs],
        "candidates": [
            asdict(candidate) | {"name": candidate.name} for candidate in candidates
        ],
        "setup_bandit": {
            "seed": int(args.bandit_seed),
            "matrix_grid_n": int(args.grid_n),
            "problem": str(getattr(args, "problem", "scalar_anisotropic_diffusion")),
            "grid_shape": list(stream_manifest.get("grid", [int(args.grid_n)] * 3)),
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
                if dynamic_activation_rules
                else {}
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
            "aot_schedule": {
                "max_selections_per_case": int(args.aot_max_selections_per_case),
                "scheduled_rounds_per_branch": (
                    int(args.online_cases) + int(args.warmup_cases)
                )
                * int(args.aot_max_selections_per_case),
                "pairing": "fixed-width block per online instance",
                "payload": "candidate_ids_only",
                "action_features": "shared_factorized_ram_cache",
            }
            if str(args.setup_candidate_mode) == "aot"
            else None,
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
                "relative_sampling_scale": float(args.lin_ts_relative_sampling_scale),
                "loss_scale_prior_sec": float(args.lin_ts_loss_scale_prior),
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
    setup_bandit_kinds: Sequence[str],
    solve_controller_protocols: Mapping[str, Mapping[str, Any]] | None = None,
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
        setup_bandit_kinds=setup_bandit_kinds,
    )
    if solve_controller_protocols is None:
        raise ValueError(
            "Composable protocol metadata must come from controller bundles"
        )
    resolved_controller_protocols = {
        str(method): dict(metadata)
        for (method, metadata) in solve_controller_protocols.items()
    }
    protocol["solve_controllers"] = resolved_controller_protocols
    section_by_kind = {"recursive_lstdq_v3": "recursive_lstdq_v3_lcb"}
    for spec in composable_specs:
        metadata = resolved_controller_protocols.get(spec.name)
        section = section_by_kind.get(spec.solve_kind)
        if metadata is not None and section is not None:
            protocol.setdefault(section, metadata)
    return protocol
