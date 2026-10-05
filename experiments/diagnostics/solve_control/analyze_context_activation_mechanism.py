"""Audit recorded context/activation runs without executing any PDE solves.

All comparisons are descriptive properties of the recorded adaptive paths.
Checkpoint perturbations keep recorded estimating equations fixed; they are
not counterfactual retraining or off-policy estimates of solver performance.
"""
from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from experiments.joint.solve_control.composable_joint_4k import build_composable_solve_runtime, resolve_composable_study
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json
from experiments.diagnostics.solve_control.run_lstdq_v3_shadow_replay import _features_for_row
from experiments.joint.solve_control.setup_branches import build_online_linucb_branch
from solve.controllers.common.linear_lcb import _greedy_cost_index


METHODS = [f"context{d}d_{s}" for d in (3, 4) for s in ("fixed", "dynamic")]
FIELDS = ("end_to_end_runtime", "setup_runtime", "solve_runtime", "infer_runtime",
          "bandit_overhead_runtime", "primary_solve_runtime", "fallback_solve_runtime",
          "fallback_setup_runtime", "activation_runtime")


def counts(values, limit=8):
    return [{"value": key, "count": n} for key, n in Counter(values).most_common(limit)]


def summarize(rows):
    outcomes = [r["outcome"] for r in rows]
    controlled = [o for o in outcomes if o.get("cycle_actions")]
    flat = lambda key: np.array([x for o in controlled for x in o.get(key, [])], dtype=float)
    actions, ratios, times = [flat(k) for k in ("cycle_actions", "cycle_residual_ratios", "cycle_times")]
    explored = flat("cycle_explored").astype(bool)
    greedy = 1.0 + .05 * flat("cycle_greedy_indices")
    # A native error may record the attempted cycle without its post-step
    # decision metadata. Match within episodes rather than shifting later rows.
    matched_actions = [(a, 1.0 + .05*g) for o in controlled
                       for a, g in zip(o["cycle_actions"], o.get("cycle_greedy_indices", []))]
    success = [o for o in controlled if o["first_primary_status"] == "success"]
    params = [r["params"] for r in rows]
    result = {
        "cases": len(rows),
        "totals_sec": {k: sum(float(o.get(k, 0)) for o in outcomes) for k in FIELDS},
        "primary_failures": sum(o["first_primary_status"] != "success" for o in outcomes),
        "primary_cycles_mean": float(np.mean([o["primary_cycles"] for o in outcomes])),
        "controlled_episodes": len(controlled), "controlled_cycles": len(actions),
        "successful_controlled_cycles_mean": float(np.mean([len(o["cycle_actions"]) for o in success])) if success else None,
        "cycle_ms_mean": float(times.mean()*1000) if len(times) else None,
        "cycle_log_contraction_mean": float(np.mean(-np.log(np.maximum(ratios, 1e-300)))) if len(ratios) else None,
        "expanding_cycle_fraction": float(np.mean(ratios > 1)) if len(ratios) else None,
        "epsilon_mean": float(flat("cycle_epsilons").mean()) if len(actions) else None,
        "exploration_fraction": float(explored.mean()) if len(explored) else None,
        "weight_mean": float(actions.mean()) if len(actions) else None,
        "greedy_weight_mean": float(greedy.mean()) if len(greedy) else None,
        "greedy_low_weight_fraction": float(np.mean(greedy <= 1.5)) if len(greedy) else None,
        "nongreedy_action_fraction": float(np.mean([abs(a-g) > 1e-10 for a, g in matched_actions])) if matched_actions else None,
        "action_counts": counts(np.round(actions, 3).tolist(), 41),
        "greedy_counts": counts(np.round(greedy, 3).tolist(), 41),
        "mean_selected_q_ms": float(flat("cycle_selected_q_values").mean()*1000) if len(actions) else None,
        "mean_selected_uncertainty_ms": float(flat("cycle_selected_uncertainties").mean()*1000) if len(actions) else None,
        "score_floor_fraction": float(np.mean(flat("cycle_selection_scores") <= 1e-12)) if len(actions) else None,
        "setup_counts": {k: counts([p[k] for p in params]) for k in params[0]},
        "top_full_setups": counts([json.dumps(p, sort_keys=True) for p in params]),
        "setup_changes": sum(a != b for a, b in zip(params[:-1], params[1:])),
        "actions_by_cycle": [],
    }
    for cycle in range(16):
        selected = [o for o in controlled if len(o["cycle_actions"]) > cycle]
        if selected:
            result["actions_by_cycle"].append({
                "cycle": cycle+1, "count": len(selected),
                "weight_mean": float(np.mean([o["cycle_actions"][cycle] for o in selected])),
                "greedy_mean": float(np.mean([1+.05*o["cycle_greedy_indices"][cycle] for o in selected
                                               if len(o.get("cycle_greedy_indices", [])) > cycle])),
                "ratio_geomean": float(np.exp(np.mean(np.log([o["cycle_residual_ratios"][cycle] for o in selected
                                                               if len(o.get("cycle_residual_ratios", [])) > cycle])))),
                "time_ms": float(np.mean([o["cycle_times"][cycle] for o in selected])*1000),
            })
    return result


def trajectory_audit(rows):
    windows = {"all": (0, 5000), "before_1001": (0, 1000), "after_1000": (1000, 5000),
               **{f"{lo+1}_{lo+1000}": (lo, lo+1000) for lo in range(0, 5000, 1000)}}
    report = {"windows": {}, "pairs": {}}
    for method, data in rows.items():
        report["windows"][method] = {key: summarize(data[lo:hi]) for key, (lo, hi) in windows.items()}
    for dim in (3, 4):
        f, d = [rows[f"context{dim}d_{s}"] for s in ("fixed", "dynamic")]
        tau = next(r["online_index"]+1 for r in d if r["rl_activation"]["crossing_case"] is not None)
        delta = np.array([b["outcome"]["end_to_end_runtime"] - a["outcome"]["end_to_end_runtime"] for a, b in zip(f, d)])
        different = [i for i, (a, b) in enumerate(zip(f, d)) if a["params"] != b["params"]]
        first = different[0]
        report["pairs"][str(dim)] = {
            "first_rl_case": tau+1,
            "first_setup_divergence_case": first+1,
            "first_setup_divergence": {"fixed": f[first], "dynamic": d[first]},
            "previous_same_setup": {"fixed": f[first-1], "dynamic": d[first-1]} if first else None,
            "dynamic_minus_fixed_cumulative_sec": np.cumsum(delta).tolist(),
            "phases": [],
        }
        for lo, hi in [(0, tau), (tau, 1000), (1000, 5000), (4000, 5000)]:
            report["pairs"][str(dim)]["phases"].append({
                "first": lo+1, "last": hi,
                "dynamic_minus_fixed_sec": float(delta[lo:hi].sum()),
                "fixed": summarize(f[lo:hi]), "dynamic": summarize(d[lo:hi]),
            })
    return report


def checkpoint_audit(run):
    audits = {}
    for method in METHODS:
        audits[method] = {}
        for n in (1000, 2000, 3000, 4000, 5000):
            path = run / "checkpoints" / f"{method}_episode_{n}.npz"
            with np.load(path, allow_pickle=False) as z:
                a, inv, b, theta, s = [z[k] for k in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance")]
                singular = np.linalg.svd(a, compute_uv=False)
                batch = np.linalg.solve(a, b)
                audits[method][n] = {
                    "episodes": int(z["episodes"]), "steps": int(z["steps"]),
                    "inverse_rebuilds": int(z["inverse_rebuild_count"]),
                    "condition_A": float(singular[0]/singular[-1]),
                    "min_singular_A": float(singular[-1]),
                    "relative_inverse_residual": float(np.linalg.norm(a@inv-np.eye(len(b)), 'fro')/np.sqrt(len(b))),
                    "relative_theta_batch_error": float(np.linalg.norm(theta-batch)/max(np.linalg.norm(batch), 1e-30)),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "min_covariance_eigenvalue": float(np.linalg.eigvalsh((s+s.T)/2)[0]),
                    "moment_trace": float(np.trace(s)),
                }
    return audits


def score_probe(controller, features):
    means, uncertainty, scores = controller._values(features, cycle=None)
    common = dict(allowed_indices=controller._all_action_indices, anchor_index=controller.anchor_index)
    lcb = _greedy_cost_index(scores, secondary_values=means, **common)
    mean = _greedy_cost_index(means, **common)
    return {"lcb_weight": float(controller.weights[lcb]), "mean_weight": float(controller.weights[mean]),
            "q_min": float(means[mean]), "selected_q": float(means[lcb]),
            "selected_uncertainty": float(uncertainty[lcb]), "lcb_floor": bool(scores[lcb] <= 1e-12)}


def frozen_probes(run, rows):
    raw = json.loads((run / "experiment_config.json").read_text())
    runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
    bundles = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles
    methods = ["context4d_fixed", "context4d_dynamic"]
    for method in methods:
        bundles[method].load(run / "checkpoints" / f"{method}_final.npz")
    targets = {method: bundles[method].controller for method in methods}
    # Remove the first 219 dynamic episodes from the fixed estimating equation.
    # Reuse a new controller built by the standard factory; never edit a checkpoint.
    trimmed = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles["context4d_dynamic"]
    trimmed.load(run / "checkpoints" / "context4d_dynamic_final.npz")
    c = trimmed.controller
    with np.load(run / "checkpoints" / "context4d_dynamic_episode_1000.npz", allow_pickle=False) as early:
        c.a_matrix -= early["a_matrix"]
        c.a_matrix += c.spec.ridge*np.eye(c.joint_dim)
        c.b -= early["b"]
        c.a_inverse[:] = np.linalg.inv(c.a_matrix)
        c.theta[:] = c.a_inverse@c.b
        c.episode_moment_covariance -= early["episode_moment_covariance"]
        c.episode_moment_covariance += c.spec.residual_floor_sec**2*c.spec.ridge*np.eye(c.joint_dim)
    targets["dynamic_without_first_219_equations"] = c
    probe_results = []
    for source in methods:
        for row in rows[source][4000:5000:20]:
            if row["outcome"]["first_primary_status"] != "success":
                continue
            features = _features_for_row(bundles[source], row)
            for cycle, state in enumerate(features):
                scores = {name: score_probe(ctrl, state) for name, ctrl in targets.items()}
                probe_results.append({"source": source, "case": row["online_index"]+1,
                                      "cycle": cycle+1, "scores": scores})
    summary = {}
    for source in methods:
        points = [r for r in probe_results if r["source"] == source]
        summary[source] = {"states": len(points), "policies": {}}
        for name in targets:
            values = [r["scores"][name] for r in points]
            summary[source]["policies"][name] = {
                k: float(np.mean([v[k] for v in values]))
                for k in values[0]
            }
            summary[source]["policies"][name]["lcb_vs_mean_disagreement"] = float(np.mean([v["lcb_weight"] != v["mean_weight"] for v in values]))
        for name in ("context4d_fixed", "dynamic_without_first_219_equations"):
            summary[source][f"dynamic_vs_{name}_disagreement"] = float(np.mean([
                p["scores"]["context4d_dynamic"]["lcb_weight"] != p["scores"][name]["lcb_weight"] for p in points]))
    return {"summary": summary, "states": probe_results,
            "interpretation": "Frozen decisions on recorded states only; no counterfactual returns or retraining."}


def setup_score_probes(run, rows):
    raw = json.loads((run / "experiment_config.json").read_text())
    runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
    study = resolve_composable_study(runtime)
    report = {}
    # The builder can prepare candidate files; isolate it from source evidence.
    with tempfile.TemporaryDirectory(prefix="activation-bandit-probe-") as tmp:
        schedule = Path(tmp) / "schedule"
        shutil.copytree(run / "aot_candidate_schedules/recommended/structured512/seed_offset_2000029", schedule)
        for method in ("context4d_fixed", "context4d_dynamic"):
            branch, _ = build_online_linucb_branch(
                seed=runtime.bandit_seed+2000029, learner_kind="linucb", tune_dim=7,
                tune7_variant="categorical", action_space_mode="full_cartesian",
                solver_tol=1e-6, solver_max_iter=50, parameter_resolution=20,
                configuration_space=study.setup_configuration_spaces["recommended"],
                candidate_schedule_dir=schedule, candidate_schedule_rounds=15000,
                candidate_sampling="structured512", context_dim=5,
                context_interaction_indices=(1,2,3,4))
            model = branch.policy.model
            model.load_mutable_state(run / "final_bandit_states" / f"{method}.npz")
            stats = model._cand.export_statistics()
            experience = {int(a): {"n": int(n), "historical_mean_sec": float(v/n)}
                          for a,n,v in zip(stats["arms"],stats["obs_count"],stats["loss_sum"]) if n}
            values = []
            for i in range(4000,5000,40):
                arms = np.array([rows[m][i]["arm_index"] for m in ("context4d_fixed", "context4d_dynamic")])
                scores, means, uncertainty = model._score_subset(np.array(rows[method][i]["context"]),
                                                                 arms=arms, alpha=model._effective_alpha())
                values.append({"case": i+1, "arms_fixed_dynamic": arms.tolist(),
                               "scores_sec": scores.tolist(), "means_sec": means.tolist(),
                               "bonuses_sec": (model._effective_alpha()*uncertainty).tolist(),
                               "history": [experience.get(int(a), {"n": 0}) for a in arms]})
            report[method] = {"alpha": float(model._effective_alpha()), "observations": model.t,
                              "mean_predicted_cost_fixed_dynamic": np.mean([v["means_sec"] for v in values],axis=0).tolist(),
                              "prefers_fixed_setup_fraction": float(np.mean([v["scores_sec"][0]<v["scores_sec"][1] for v in values])),
                              "cases": values}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for method in METHODS:
        with (args.run / "trajectories" / f"{method}.jsonl").open() as f:
            rows[method] = [json.loads(line) for line in f]
        assert len(rows[method]) == 5000
        assert all(r["online_index"] == i for i, r in enumerate(rows[method]))
    reference = rows[METHODS[0]]
    assert all(all(a["mkw"] == b["mkw"] for a, b in zip(reference, rs)) for rs in rows.values())
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.run.resolve()), "trajectory": trajectory_audit(rows)}
    print("Trajectory audit complete", flush=True)
    result["checkpoints"] = checkpoint_audit(args.run)
    print("Checkpoint audit complete", flush=True)
    result["frozen_probes"] = frozen_probes(args.run, rows)
    result["setup_score_probes"] = setup_score_probes(args.run, rows)
    _write_json(args.output / "mechanism_audit.json", _json_ready(result))
    print(args.output / "mechanism_audit.json", flush=True)


if __name__ == "__main__":
    main()
