"""Benchmark saved Module 4 setup states without solves or learner updates.

Run ``baseline`` before editing the scorer, then ``compare`` afterwards. The
output retains the original function and the raw timing samples. Every timed
selection resets its RNG, candidate cursor and last-arm anchor outside timing.
"""

from __future__ import annotations

from experiments.runtime import configure_single_thread
configure_single_thread()

import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import textwrap
import time
from types import MethodType
from typing import Optional, Tuple

import numpy as np

from experiments.paper_final.online.setup_branches import (
    GenericBanditPolicy, build_online_linucb_branch,
)
from experiments.paper_final.recovery.execution import CandidateRows
from experiments.paper_final.recovery.retry_comparison import use_retry_rule
from experiments.paper_final.run_06_retry_repeatability import observed_model
from experiments.paper_final.run_06_retry_screen import request_performance_qos
from problems.registry import context_for_setup_method, learning_context_for_setup
from setup.learners.linucb.learner import SharedLinUCB, SharedLinUCBStep
from setup.space import SetupConfigurationSpace


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def compile_scorer(source):
    namespace = dict(np=np, Optional=Optional, Tuple=Tuple)
    exec(compile(source, "<preserved-selection-scorer>", "exec"), namespace)
    return namespace["_mean_uncertainty_subset"]


def block_scorer(self, x, *, arms, action_features=None, contraction="staged"):
    if action_features is not None:
        G = np.asarray(action_features, dtype=float)
    elif arms is None:
        if self._g_actions is None:
            raise RuntimeError("full-space scoring is unavailable for factorized catalogs")
        G = self._g_actions
    else:
        G = self._features_for_arms(np.asarray(arms, dtype=int))
    mean = self._linear_prediction_subset(
        x, theta=self.A_inv @ self.b, arms=arms, action_features=G)
    dx, g = self.d_x, self.g_dim
    weights = np.asarray([1.0, *(float(x[i]) for i in self.context_interaction_indices)])
    count = weights.size
    Ac = self.A_inv[:, :dx] @ x
    q0 = float(x @ Ac[:dx])
    u = weights @ Ac[dx:].reshape(count, g)
    blocks = self.A_inv[dx:, dx:].reshape(count, g, count, g)
    if contraction == "staged":
        right = np.einsum("iajb,j->iab", blocks, weights, optimize=False)
        M = np.einsum("i,iab->ab", weights, right, optimize=False)
    else:
        M = np.einsum("i,iajb,j->ab", weights, blocks, weights,
                      optimize=contraction == "optimized")
    quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False)
    quad += np.sum((G @ M) * G, axis=1)
    return mean, np.sqrt(np.maximum(0.0, quad))


def prepare_fixtures(stage, output):
    raw = json.loads((stage / "experiment_config.json").read_text())
    mode = next(m["setup_context"] for m in raw["methods"] if m["setup"] == "linucb")
    space_name = next(m["setup_space"] for m in raw["methods"] if m["setup"] == "linucb")
    contract = learning_context_for_setup(raw["problem"]["kind"], "linucb", mode)
    source_schedule = stage / "aot_candidate_schedules" / space_name / "structured512"
    schedule = output / "candidate_schedule_copy"
    if not schedule.exists():
        shutil.copytree(source_schedule, schedule)
    for name in ("metadata.json", "candidate_ids.npy"):
        if file_hash(source_schedule/name) != file_hash(schedule/name):
            raise ValueError("Candidate schedule copy differs from source")
    metadata = json.loads((schedule / "metadata.json").read_text())
    branch, _ = build_online_linucb_branch(
        seed=raw["seeds"]["bandit"], action_space_mode="full_cartesian",
        parameter_resolution=raw["setup"]["parameter_resolution"],
        configuration_space=SetupConfigurationSpace.from_mapping(
            space_name, raw["setup"]["configuration_spaces"][space_name]),
        candidate_schedule_dir=schedule, candidate_schedule_rounds=metadata["rounds"],
        candidate_pool_size=metadata["pool_size"],
        structured_quotas=metadata.get("structured_quotas"),
        candidate_schedule_seed=metadata["seed"], candidate_sampling="structured512",
        context_dim=contract.dimension, context_interaction_indices=contract.interaction_indices)
    model = observed_model(use_retry_rule(branch.policy.model, "failure_model_adjacent"))
    # Constructor must reuse the exact copied schedule, never regenerate it.
    for name in ("metadata.json", "candidate_ids.npy"):
        assert file_hash(source_schedule/name) == file_hash(schedule/name)
    input_rows = [json.loads(s) for s in (stage/"inputs.jsonl").read_text().splitlines()]
    checkpoints = []
    provenance = {str(stage/"experiment_config.json"): file_hash(stage/"experiment_config.json"),
                  str(stage/"inputs.jsonl"): file_hash(stage/"inputs.jsonl")}
    for method in ("bandit_default", "bandit_lstdq"):
        wanted = {1, 1000, 1001, 2000, 2001, 3000, 3001, 4000, 4001, 5000}
        records = {}
        trajectory = stage / "trajectories" / (method + ".jsonl")
        provenance[str(trajectory)] = file_hash(trajectory)
        with trajectory.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["problem_index"] in wanted:
                    records[record["problem_index"]] = record
        for k in (1000, 2000, 3000, 4000, 5000):
            checkpoint = stage / "bandit_checkpoints" / f"{method}_{k}.npz"
            provenance[str(checkpoint)] = file_hash(checkpoint)
            case = k if k < len(input_rows) else 0
            row = input_rows[case]
            context = context_for_setup_method(
                problem_kind=raw["problem"]["kind"], setup_kind="linucb",
                matrix_kwargs=row["mkw"], stream_context=row["context"], setup_context=mode)
            previous = records[k]["selection_diagnostics"][-1]
            anchor = previous["selected"]["arm"] if "selected" in previous else previous["arm"]
            expected = records[case+1]["selection_diagnostics"][0] if case else None
            checkpoints.append(dict(name=f"{method}_{k}", checkpoint=checkpoint,
                context=context, anchor=anchor, case=case, expected=expected))
    return model, checkpoints, provenance


def reset(model, state, scorer, *, cold=False):
    model._mean_uncertainty_subset = MethodType(scorer, model)
    model.rng.bit_generator.state = copy.deepcopy(state["rng"])
    model.history = [state["anchor"]]
    model.candidate_stats_history = []
    model.selection_diagnostics = []
    model._capture = None
    model._last_phi = model._last_arm = None
    model.set_candidate_schedule_cursor(state["cursor"])
    if cold:
        model._local_neighbor_cache.clear()


def summarize(samples):
    values = np.asarray(samples, dtype=float)
    return dict(median_ms=float(np.median(values)), p25_ms=float(np.quantile(values, .25)),
                p75_ms=float(np.quantile(values, .75)), samples_ms=list(samples))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_file = output / "reference_scorer.py"
    live_source = textwrap.dedent(inspect.getsource(SharedLinUCB._mean_uncertainty_subset))
    if args.phase == "baseline":
        if baseline_file.exists():
            raise ValueError("Baseline already exists; choose a fresh output")
        baseline_file.write_text(live_source)
    reference = baseline_file.read_text()
    old_quad = 'np.einsum("ij,jk,ik->i", G, M, G, optimize=False)'
    if old_quad not in reference:
        raise ValueError("Baseline does not contain the original quadratic contraction")
    baseline = compile_scorer(reference)
    variants = {"original": baseline}
    if args.phase != "baseline":
        variants.update(matmul=compile_scorer(reference.replace(old_quad, "np.sum((G @ M) * G, axis=1)")),
                        block_staged=block_scorer)
        if args.phase == "prototypes":
            variants.update(
                block_direct=lambda *a, **k: block_scorer(*a, **k, contraction="direct"),
                block_optimized=lambda *a, **k: block_scorer(*a, **k, contraction="optimized"))
        else:
            variants["production"] = SharedLinUCB._mean_uncertainty_subset
    qos = request_performance_qos()
    model, fixtures, provenance = prepare_fixtures(args.stage.resolve(), output)
    policy = GenericBanditPolicy(model)
    rows = CandidateRows(model, cases=model._candidate_schedule.rounds//3, attempts=3,
                         directory=output/"unused")
    order_rng = np.random.default_rng(90301)
    results = []
    for fixture in fixtures:
        model.load_mutable_state(fixture["checkpoint"])
        model.A_inv.setflags(write=False)
        model.b.setflags(write=False)
        model.failure_b.setflags(write=False)
        x = fixture["context"]
        state = dict(rng=copy.deepcopy(model.rng.bit_generator.state), cursor=fixture["case"]*3,
                     anchor=SharedLinUCBStep(model.t, fixture["anchor"], 0., 0., 0.))
        reset(model, state, baseline)
        arms, G, _ = model._structured_candidate_subset(512)
        expected = fixture["expected"]
        if expected is not None and arms.tolist() != expected["candidate_arms"]:
            raise AssertionError("Reconstructed candidates differ from recorded next input")
        reference_values = baseline(model, x, arms=arms, action_features=G)
        ref_score = reference_values[0] - model._effective_alpha()*reference_values[1]
        ref_ties = np.flatnonzero(ref_score <= np.min(ref_score) + 1e-12)
        failure_mean = model._linear_prediction_subset(
            x, theta=model.A_inv@model.failure_b, arms=arms, action_features=G)
        def retry_ties(score, width):
            upper = failure_mean + model.failure_beta*width
            best = np.lexsort((score, upper))[0]
            return np.flatnonzero(np.isclose(upper, upper[best], atol=1e-12, rtol=0)
                & np.isclose(score, score[best], atol=1e-12, rtol=0))
        ref_retry = retry_ties(ref_score, reference_values[1])
        entry = dict(name=fixture["name"], candidate_count=len(arms),
                     next_input_reconstructed=expected is not None,
                     input_index=fixture["case"]+1, observations=model.t,
                     dimensions=dict(context=model.d_x, action=model.g_dim, setup=model.d_phi),
                     agreement={}, timings={})
        reference_choices = {}
        for name, scorer in variants.items():
            mean, width = scorer(model, x, arms=arms, action_features=G)
            score = mean - model._effective_alpha()*width
            np.testing.assert_allclose(mean, reference_values[0], atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(width, reference_values[1], atol=1e-11, rtol=1e-11)
            np.testing.assert_array_equal(np.flatnonzero(score <= np.min(score)+1e-12), ref_ties)
            np.testing.assert_array_equal(retry_ties(score, width), ref_retry)
            entry["agreement"][name] = dict(mean_max_abs=float(np.max(abs(mean-reference_values[0]))),
                width_max_abs=float(np.max(abs(width-reference_values[1]))),
                score_max_abs=float(np.max(abs(score-ref_score))), normal_ties_match=True, retry_ties_match=True)
            for retry in (False, True):
                reset(model, state, scorer)
                rows.before_attempt(fixture["case"], int(retry))
                if retry:
                    model.select_recovery(x, excluded_arms=[int(arms[ref_ties[0]])])
                else:
                    policy.select(x)
                choice = (model._last_arm, model.selection_diagnostics[-1]["candidate_arms"],
                          copy.deepcopy(model.rng.bit_generator.state))
                if name == "original":
                    reference_choices[retry] = choice
                assert choice == reference_choices[retry], (name, fixture["name"], retry)
        calls = {
            "kernel": lambda: model._mean_uncertainty_subset(x, arms=arms, action_features=G),
            "selection_warm": lambda: (rows.before_attempt(fixture["case"], 0), policy.select(x)),
            "selection_cold_neighbors": lambda: (rows.before_attempt(fixture["case"], 0), policy.select(x)),
            "retry_warm": lambda: (rows.before_attempt(fixture["case"], 1),
                model.select_recovery(x, excluded_arms=[int(arms[ref_ties[0]])])),
        }
        for scope, call in calls.items():
            samples = {name: [] for name in variants}
            for repetition in range(args.repetitions + args.warmup):
                for name in order_rng.permutation(list(variants)):
                    reset(model, state, variants[name], cold=scope=="selection_cold_neighbors")
                    if scope in ("selection_warm", "retry_warm"):
                        # Populate exactly the candidate-row anchors outside timing.
                        call()
                        reset(model, state, variants[name])
                    start = time.perf_counter_ns()
                    call()
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    if repetition >= args.warmup:
                        samples[name].append(elapsed)
            entry["timings"][scope] = {name: summarize(v) for name, v in samples.items()}
        results.append(entry)
        print(f"{fixture['name']}: " + ", ".join(
            f"{name} {entry['timings']['selection_warm'][name]['median_ms']:.3f} ms"
            for name in variants), flush=True)
        # load_mutable_state assigns into existing storage.
        model.A_inv.setflags(write=True)
        model.b.setflags(write=True)
        model.failure_b.setflags(write=True)
    for path, digest in provenance.items():
        assert file_hash(path) == digest, "Historical input changed"
    summary = {scope: {name: dict(
        median_of_fixture_medians_ms=float(np.median([
            f["timings"][scope][name]["median_ms"] for f in results])),
        mean_of_fixture_medians_ms=float(np.mean([
            f["timings"][scope][name]["median_ms"] for f in results])))
        for name in variants} for scope in results[0]["timings"]}
    payload = dict(phase=args.phase, stage=str(args.stage.resolve()),
        environment=dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                         numpy=np.__version__, qos=qos, thread_limits={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}),
        repetitions=args.repetitions, warmup=args.warmup, original_source_sha256=file_hash(baseline_file),
        live_scorer_matches_baseline=ast.dump(ast.parse(live_source)) == ast.dump(ast.parse(reference)),
        live_source_sha256=hashlib.sha256(live_source.encode()).hexdigest(),
        checkpoint_and_input_sha256=provenance, summary=summary, fixtures=results,
        native_solves=0, learner_updates=0,
        notes=["Final 5000 checkpoints replay input 1 and candidate row 0; they are offline fixtures.",
               "Eight earlier states reconstruct the actual next recorded primary candidate list.",
               "Retries are offline next-row probes, not replays of a historical failed solve.",
               "Warm selection includes candidate generation, feature gathering, tie-breaking and observed diagnostics.",
               "Cold-neighbor timing clears only the deterministic topology cache before each call.",
               "Kernel receives precomputed candidate features; checkpoint loading and RNG resets are excluded."])
    write(output / f"{args.phase}.json", payload)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "prototypes", "compare"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=6)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("Require positive repetitions and nonnegative warmup")
    run(args)


if __name__ == "__main__":
    main()
