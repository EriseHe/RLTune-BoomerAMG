"""Four independent Module 1 branches using the frozen recovery decision.

The setup branches reuse the audited Failure Model / adjacent-cost executor.
Default+LSTDQ executes one ordinary RL attempt, followed by exact Default only
if necessary. Neither fixed-setup branch owns a setup learner.
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from experiments.paper_final.common.artifacts import digest
from experiments.paper_final.online.case_loop import report_online_outcome
from experiments.paper_final.online.configuration import DEFAULT_STRUCTURED_QUOTAS
from experiments.paper_final.recovery.execution import EpisodeRecorder, run_fixed_case
from experiments.paper_final.recovery.joint_validation import execute_learned
from experiments.paper_final.recovery import metrics as sweep
from setup.space import DEFAULT_SETUP_PARAMS

METHODS = ("default_setup_default_solve", "default_setup_lstdq", "bandit_default", "bandit_lstdq")
LABELS = dict(zip(METHODS, ("Default", "Default + LSTDQ", "LinUCB", "LinUCB + LSTDQ")))
ORDERS = tuple(itertools.permutations(METHODS))


def balanced_orders(cases, seed, methods=METHODS):
    """Randomize complete permutation blocks; every position is balanced."""
    permutations = ORDERS if tuple(methods) == METHODS else tuple(itertools.permutations(methods))
    rng = np.random.default_rng(seed)
    orders = []
    while len(orders) < cases:
        orders.extend(permutations[int(i)] for i in rng.permutation(len(permutations)))
    return orders[:cases]


def audit_execution(output):
    """Verify four physical measurements per input and checkpoint identities."""
    from experiments.paper_final.common.artifacts import read, file_hash, dump, read_records

    output = Path(output)
    protocol = read(output / "protocol.json")
    if protocol.get("shared_prefix") or protocol.get("copied_measurements"):
        raise ValueError("Shared-prefix runs require their archived source snapshot")
    seeds = {}
    for label in protocol["seeds"]:
        stage = output / f"seed_{label}"
        result = read(stage / "result.json")
        config = read(stage / "experiment_config.json")
        assert result["complete"] and result["passed"]
        assert file_hash(stage / "experiment_config.json") == protocol["stages"][label]["config_sha256"]
        for relative, expected in result["checkpoint_sha256"].items():
            assert file_hash(stage / relative) == expected
        activation = next(m.get("solve_activation_case", 0) for m in config["methods"] if m["id"] == METHODS[3])
        orders = balanced_orders(result["cases"], config["seeds"]["method_order"])
        events = list(read_records(stage / "execution.jsonl"))
        assert len(events) == sum(map(len, orders))
        offset, previous_end, pids = 0, 0, set()
        for index, order in enumerate(orders):
            block = events[offset:offset + len(order)]
            offset += len(order)
            assert tuple(row["method"] for row in block) == order
            for position, row in enumerate(block):
                assert row["input_index"] == index and row["position"] == position
                assert previous_end <= row["start_ns"] <= row["end_ns"]
                previous_end = row["end_ns"]
                pids.add(row["pid"])
        assert len(pids) == 1
        initial = read(stage / "initial_audit.json")
        assert initial["passed"] and initial["independence"]["passed"]
        boundary = read(stage / "activation_audit.json") if activation < result["cases"] else None
        if boundary is not None:
            assert boundary["passed"] and boundary["problem_index"] == activation + 1
            assert boundary["joint_controller_steps"] == 0
            if activation == 0:
                assert boundary["fixed_controller_steps"] == 0
                assert set(boundary["setup_observations"].values()) == {0}
        seeds[label] = dict(passed=True, measured_method_cases=len(events),
                            logical_method_cases=4 * result["cases"], copied_prefix_rows=0,
                            worker_pid=pids.pop(), serial_execution=True, complete_seeded_orders=True,
                            first_start_ns=events[0]["start_ns"], last_end_ns=events[-1]["end_ns"],
                            config_and_checkpoints_unchanged=True, activation=boundary)
    # A later batch may legally receive a recycled OS PID. Such executions must
    # not overlap; concurrently active seeds must still have distinct workers.
    for first, second in itertools.combinations(seeds.values(), 2):
        if first["worker_pid"] == second["worker_pid"]:
            assert (first["last_end_ns"] <= second["first_start_ns"] or
                    second["last_end_ns"] <= first["first_start_ns"])
    audit = dict(passed=True, seed_audits=seeds,
                 total_method_cases=sum(s["logical_method_cases"] for s in seeds.values()),
                 measured_method_cases=sum(s["measured_method_cases"] for s in seeds.values()))
    dump(output / "completion_audit.json", audit)
    return audit


def validate_methods(spec):
    by_name = {method.name: method for method in spec.methods}
    if set(by_name) != set(METHODS):
        raise ValueError(f"This runner requires exactly {METHODS}")
    expected = (("default", "default"), ("default", "recursive_lstdq"),
                ("linucb", "default"), ("linucb", "recursive_lstdq"))
    for name, kinds in zip(METHODS, expected):
        method = by_name[name]
        if (method.setup_kind, method.solve_kind) != kinds:
            raise ValueError(f"Wrong setup/solve combination for {name}")
        if method.solve_tolerance not in (None, spec.solve.state.tol):
            raise ValueError("All four branches must use the same tolerance")
        if method.seed_offset != 0:
            raise ValueError("The paired protocol requires zero per-method seed offsets")
    learned = [by_name[name] for name in METHODS[2:]]
    if any(method.candidate_sampling != "structured512" for method in learned):
        raise ValueError("The frozen protocol requires the structured512 sampling strategy")
    if len({(m.setup_space, m.setup_context) for m in learned}) != 1:
        raise ValueError("The two setup learners must have identical spaces and contexts")
    controllers = [by_name[name] for name in (METHODS[1], METHODS[3])]
    if len({(m.setup_space, m.solve_context) for m in controllers}) != 1:
        raise ValueError("Both solve controllers must have identical observation encodings")
    if spec.solve.state.encoding_version != "space_aware_v2":
        raise ValueError("The expanded search space requires the versioned solve encoder")
    if any(m.solve_activation_case != 0 for m in controllers):
        raise ValueError("Both LSTDQ controllers must start on the first input; prefixes are unsupported")
    return by_name


def execute_fixed_rl(runtime, index):
    from hypre.bindings import augment_setup_params
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case
    from solve.core.episode import run_td_episode

    started = time.perf_counter()
    mkw, context = runtime.stream[index]
    tick = time.perf_counter()
    recorder = EpisodeRecorder(runtime.bundle.controller)
    snapshot_runtime = time.perf_counter() - tick
    traces = []

    def solve():
        raw = run_td_episode(mkw=dict(mkw), params=dict(DEFAULT_SETUP_PARAMS),
            controller=recorder, encoder=runtime.bundle.encoder, problem_context=context,
            solve_tol=runtime.args.tol, solve_max_cycles=runtime.args.max_cycles,
            learn=True, explore=True, record_action_metadata=True,
            fallback_attempt=None, failure_penalty_sec=0.0)
        raw["recovery_protocol_applied"] = False
        traces.append(raw)
        return raw

    def fallback():
        return solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(mkw),
            solver_tol=runtime.args.tol, solver_max_iter=runtime.args.max_cycles,
            augment_params=augment_setup_params)

    raw = run_fixed_case(solve, recorder, snapshot_runtime, False,
                         fallback_solver=fallback, max_attempts=1)
    for attempt, trace in zip(raw["primary_attempts"], traces):
        attempt.update(params=dict(DEFAULT_SETUP_PARAMS),
                       **{key: trace.get(key, []) for key in sweep.TRACE_FIELDS})
    out = report_online_outcome(raw, bandit_timing={})
    out["method_wall_runtime"] = time.perf_counter() - started
    row = dict(problem_index=index+1, stream_index=index, method=runtime.name,
               mkw=mkw, context=context.tolist(), params=dict(DEFAULT_SETUP_PARAMS),
               outcome=out, rl_active=True, shared_prefix=False)
    audit_fixed_rl(row, runtime.bundle.controller.spec.uncertainty_beta)
    return row


def audit_rl_traces(row, beta):
    out = row["outcome"]
    if out.get("replayed_transitions", 0) or out.get("target_replay_runtime", 0):
        raise ValueError("RL recovery-target replay is forbidden")
    for attempt in out["primary_attempts"]:
        if attempt.get("rl_recovery_cost") not in (None, 0.0):
            raise ValueError("Recovery cost entered an RL target")
        keys = ("cycle_actions", "cycle_selected_q_values", "cycle_selected_uncertainties",
                "cycle_selection_scores", "cycle_explored", "cycle_forced_default_actions")
        arrays = [attempt[key] for key in keys]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("Incomplete cycle diagnostics")
        for q, width, score in zip(*arrays[1:4]):
            if not all(v is not None and math.isfinite(v) for v in (q, width, score)) or width < 0:
                raise ValueError("Invalid RL selection diagnostics")
            if not math.isclose(max(q-beta*width, 0.0), score, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("RL score differs from the configured beta")


def audit_fixed_rl(row, beta):
    out = row["outcome"]
    if not row["rl_active"] or row["params"] != DEFAULT_SETUP_PARAMS:
        raise ValueError("Default+LSTDQ changed its fixed setup or activation")
    if out["bandit_update_committed"] or out["bandit_observation_count"] or out["learned_reselection_count"]:
        raise ValueError("Default+LSTDQ used a setup learner")
    if out["primary_attempt_count"] != 1 or len(out["primary_attempts"]) != 1 or out["fixed_setup_retries"]:
        raise ValueError("Default+LSTDQ must execute exactly one primary attempt")
    first = out["primary_attempts"][0]
    if out["fallback_used"] != (first["status"] != "success"):
        raise ValueError("Default+LSTDQ fallback is not conditional on failure")
    native = first["setup_runtime"] + first["solve_runtime"]
    native += out["fallback_setup_runtime"] + out["fallback_solve_runtime"]
    inclusive = native + out["infer_runtime"]
    if not math.isclose(out["runtime"], native, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Fixed-RL native accounting does not close")
    if not math.isclose(out["end_to_end_runtime"], inclusive, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Fixed-RL inclusive accounting does not close")
    if out["failed"] and out["controller_update_committed"]:
        raise ValueError("An unrecovered case retained RL learning")
    audit_rl_traces(row, beta)


def execute(runtime, index, previous):
    if runtime.method.setup_kind == "linucb":
        row, previous = execute_learned(runtime, index, previous,
            feedback_mode="adjacent_delayed", activation_case=runtime.method.solve_activation_case)
    elif runtime.bundle is not None:
        row = execute_fixed_rl(runtime, index)
    else:
        row = sweep.execute_default_case(runtime, index)
    row.update(method=runtime.name, shared_prefix=False, online_index=index)
    return row, previous


def audit_features(runtime):
    """Compare optimized features/predictions against direct generic formulas."""
    model = runtime.branch.policy.model
    rng = np.random.default_rng(20261009)
    arms = np.unique(np.r_[model.actions.default_arm_index,
                           rng.integers(0, len(model.actions), size=512)])
    fast = model._action_feature_cache.features(arms)
    direct = model._encoder.encode_actions([model.actions[int(arm)] for arm in arms])
    np.testing.assert_allclose(fast, direct, rtol=1e-14, atol=1e-14)
    x = np.linspace(0.1, 0.9, model.d_x)
    theta = rng.normal(size=model.d_phi)
    phi = np.asarray([np.concatenate([x, g, *(x[i]*g for i in model.context_interaction_indices)])
                      for g in direct])
    predicted = model._linear_prediction_subset(x, arms=arms, theta=theta)
    np.testing.assert_allclose(predicted, phi @ theta, rtol=1e-12, atol=1e-12)
    params = dict(DEFAULT_SETUP_PARAMS)
    for value in (4, 6, 7):
        np.testing.assert_array_equal(model._encoder.encode_action(dict(params, agg_interp_type=value)),
                                      model._encoder.encode_action(params))
    if dict(model.actions[int(model.actions.default_arm_index)]) != DEFAULT_SETUP_PARAMS:
        raise ValueError("Catalog reference is not exact Default")
    spec = getattr(runtime, "spec", None)
    candidate_spec = spec.setup.candidate_schedule if spec is not None else None
    expected_pool = (candidate_spec.pool_size if candidate_spec is not None else None) or 512
    expected_quotas = (candidate_spec.structured_quotas if candidate_spec is not None else None) or DEFAULT_STRUCTURED_QUOTAS
    if (model.alpha != 1.0 or model.l2_reg != 1.0 or model._candidate_schedule.pool_size != expected_pool
            or model._cand.candidate_pool_size != expected_pool
            or model.initial_guess_rounds != 1 or model._cand.elite_cache_size != 64):
        raise ValueError("Inherited environment changed the frozen LinUCB settings")
    schedule = model._candidate_schedule
    actual_quotas = dict(anchor=schedule.structured_anchor_size,
        balanced_global=schedule.structured_global_size,
        coordinate_local=schedule.structured_local_size, sobol=schedule.structured_sobol_size)
    if actual_quotas != expected_quotas or sum(actual_quotas.values()) != expected_pool:
        raise ValueError("Structured candidate quotas differ from the frozen configuration")
    for row in schedule._ids[:min(100, len(schedule._ids))]:
        if row.size != expected_pool or np.unique(row).size != expected_pool:
            raise ValueError("Candidate schedule does not contain the configured number of distinct arms")
    sampled = np.asarray(schedule._ids[:min(100, len(schedule._ids))]).reshape(-1)
    values, _ = model.actions.decode_parameter_arrays(sampled)
    families, counts = np.unique(np.column_stack([values[k] for k in
        ("agg_num_levels", "agg_interp_type", "coarsen_type", "interp_type")]), axis=0, return_counts=True)
    return dict(passed=True, actions=len(model.actions), action_features=model.g_dim,
                regression_features=model.d_phi, context_features=model.d_x,
                checked_actions=len(arms), feature_max_error=float(np.max(np.abs(fast-direct))),
                conditional_inactive_mask=True, exact_default=True,
                categorical_candidate_families=len(families), candidate_rows_checked=min(100, len(schedule._ids)),
                candidate_family_counts=[dict(depth=int(f[0]), aggressive_interpolation=int(f[1]),
                    coarsening=int(f[2]), interpolation=int(f[3]), count=int(count)) for f, count in zip(families, counts)],
                learner=dict(alpha=model.alpha, alpha_decay=model.alpha_decay,
                    l2=model.l2_reg, initial_guess_rounds=model.initial_guess_rounds,
                    elite_cache_size=model._cand.elite_cache_size, candidate_pool_size=schedule.pool_size,
                    structured_quotas=actual_quotas),
                action_catalog=model.actions.description())


def audit_independence(runtimes):
    models = [runtimes[name].branch.policy.model for name in METHODS[2:]]
    for key in ("A_inv", "b", "failure_b"):
        if np.shares_memory(getattr(models[0], key), getattr(models[1], key)):
            raise ValueError(f"Setup models share {key}")
    if models[0].rng is models[1].rng or models[0]._candidate_schedule is models[1]._candidate_schedule:
        raise ValueError("Setup RNG or schedule cursors are shared")
    bundles = [runtimes[name].bundle for name in (METHODS[1], METHODS[3])]
    controllers = [bundle.controller for bundle in bundles]
    for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance"):
        if np.shares_memory(getattr(controllers[0], key), getattr(controllers[1], key)):
            raise ValueError(f"Solve controllers share {key}")
    if controllers[0].rng is controllers[1].rng:
        raise ValueError("Solve controllers share RNG")
    if any(c.steps or c.episodes for c in controllers) or any(m.t for m in models):
        raise ValueError("Learners trained before measurement")
    encodings = [bundle.protocol_metadata()["state_encoder"] for bundle in bundles]
    if digest(encodings[0]) != digest(encodings[1]):
        raise ValueError("Solve-state encodings differ across branches")
    return dict(passed=True, independent_setup_models=2, independent_controllers=2,
                identical_solve_encoding=True, solve_state=encodings[0])
