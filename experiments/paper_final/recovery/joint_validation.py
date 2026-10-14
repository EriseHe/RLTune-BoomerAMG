"""Shared-state continuation and unchanged no-recovery solve-RL targets."""
from __future__ import annotations

import copy
from dataclasses import asdict
import time

import numpy as np

from experiments.paper_final.common.artifacts import digest
from experiments.paper_final.online.case_loop import report_online_outcome
from experiments.paper_final.recovery.execution import EpisodeRecorder
from experiments.paper_final.recovery.retry_comparison import run_deferred_case
from experiments.paper_final.recovery import metrics as sweep
from problems.registry import context_for_setup_method


def copy_complete_prefix(source, target, checkpoint, *, previous_update):
    """Restore learned state plus the previous-arm anchor omitted by checkpoints.

    Both continuation models are freshly initialized before this function is
    called. Checkpoints restore regression, binary failure targets, candidate
    statistics, schedule position, and RNG. History also influences structured
    candidate anchors, so preserve it explicitly instead of resetting it.
    """
    if source._last_phi is not None or source._last_arm is not None:
        raise ValueError("The prefix has an unresolved action")
    if target.t or target.history:
        raise ValueError("Continuation target must be fresh")
    target.load_mutable_state(checkpoint)
    target.history = copy.deepcopy(source.history)
    target.candidate_stats_history = copy.deepcopy(source.candidate_stats_history)
    # This cache is derived immutable topology, not learned statistics. Start
    # both fresh continuations with the same empty cache, leaving the prefix
    # object untouched and unused for future selections.
    target._local_neighbor_cache = {}
    for key in ("A_inv", "b", "failure_b"):
        np.testing.assert_array_equal(getattr(source,key),getattr(target,key))
        if np.shares_memory(getattr(source,key),getattr(target,key)):
            raise ValueError("Continuation shares mutable setup arrays")
    if source.rng is target.rng or digest(source.rng.bit_generator.state) != digest(target.rng.bit_generator.state):
        raise ValueError("Setup RNG was not independently restored")
    if digest([asdict(s) for s in source.history]) != digest([asdict(s) for s in target.history]):
        raise ValueError("Structured candidate history was not restored")
    source_cursor = 0 if source._candidate_schedule is None else source._candidate_schedule.cursor
    target_cursor = 0 if target._candidate_schedule is None else target._candidate_schedule.cursor
    if target_cursor != source_cursor:
        raise ValueError("Candidate schedule boundary differs")
    return float(previous_update)


def execute_learned(runtime, index, previous_update, *, feedback_mode, activation_case=1000):
    from hypre.bindings import augment_setup_params
    from setup.space import DEFAULT_SETUP_PARAMS
    from solve.core.episode import run_td_episode
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case

    started=time.perf_counter()
    args=runtime.args
    mkw,context=runtime.stream[index]
    active=runtime.bundle is not None and index>=activation_case
    recorder,snapshot_runtime=None,0.0
    if active:
        tick=time.perf_counter()
        recorder=EpisodeRecorder(runtime.bundle.controller)
        snapshot_runtime=time.perf_counter()-tick
    traces=[]
    def native(params):
        return solve_no_rl_case(params=dict(params),mkw=dict(mkw),solver_tol=args.tol,
            solver_max_iter=args.max_cycles,augment_params=augment_setup_params)
    def solve(params):
        if recorder is None:
            raw=native(params)
        else:
            raw=run_td_episode(mkw=dict(mkw),params=dict(params),controller=recorder,
                encoder=runtime.bundle.encoder,problem_context=context,
                solve_tol=args.tol,solve_max_cycles=args.max_cycles,learn=True,explore=True,
                record_action_metadata=True,fallback_attempt=None,failure_penalty_sec=0.0)
            raw["recovery_protocol_applied"]=False
        traces.append(raw)
        return raw
    learner_context=context_for_setup_method(problem_kind=args.problem,
        setup_kind=runtime.method.setup_kind,setup_context=runtime.method.setup_context,
        matrix_kwargs=mkw,stream_context=context)
    model=runtime.branch.policy.model
    diagnostic_start=len(model.selection_diagnostics)
    result=run_deferred_case(policy=runtime.branch.policy,
        parameter_space=runtime.branch.parameter_space,context=learner_context,
        solver=solve,fallback_solver=lambda:native(DEFAULT_SETUP_PARAMS),variant=sweep.VARIANT,
        previous_update=previous_update,before_attempt=lambda attempt:runtime.rows.before_attempt(index,attempt),
        feedback_mode=feedback_mode,recorder=recorder,snapshot_runtime=snapshot_runtime)
    runtime.rows.finish_case(index)
    for attempt,raw in zip(result.outcome["primary_attempts"],traces):
        attempt.update({key:raw.get(key,[]) for key in sweep.TRACE_FIELDS})
    out=report_online_outcome(result.outcome,bandit_timing=result.timing)
    out["method_wall_runtime"]=time.perf_counter()-started
    row=dict(problem_index=index+1,stream_index=index,method=runtime.name,mkw=mkw,
        context=context.tolist(),params=result.params,outcome=out,rl_active=active,
        shared_prefix=index<activation_case,selection_diagnostics=model.selection_diagnostics[diagnostic_start:])
    beta=None if runtime.bundle is None else runtime.bundle.controller.spec.uncertainty_beta
    sweep.audit_case(row,beta,variant=sweep.VARIANT,feedback_mode=feedback_mode)
    if len(row["selection_diagnostics"])!=out["primary_attempt_count"]:
        raise ValueError("Missing setup selection diagnostic")
    return row,result.update_runtime_sec
