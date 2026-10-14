"""Figure 5: approved consecutive diffusion inputs and verified Default replay."""
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

EXAMPLE_CASE = "diffusion_60"
EXAMPLE_SEED = "1"
EXAMPLE_INPUTS = (3882, 3883, 3884)
COLORS = ("#3A9AB2", "#D5A407", "#F11B00")
STYLES = ("-", "--", "-.")
MARKERS = ("o", "s", "^")


def prepare(chosen):
    stage = chosen["stage"]
    assert stage.name == "seed_1" and stage.parent.name == EXAMPLE_CASE
    replay_path = (stage.parents[3] / "figure_evidence"
                   / "default_residual_replay_3882_3884.json")
    replay = json.loads(replay_path.read_text())
    assert replay["passed"] and replay["relaxation_weight"] == 1.0
    baseline = {row["input"]: row for row in replay["cases"]}
    sources = {str(replay_path): hashlib.sha256(replay_path.read_bytes()).hexdigest()}
    records = {}
    for method in ("bandit_lstdq", "default_setup_default_solve"):
        path = stage / "trajectories" / f"{method}.jsonl"
        digest, selected = hashlib.sha256(), {}
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                row = json.loads(line)
                if row["problem_index"] in EXAMPLE_INPUTS:
                    selected[row["problem_index"]] = row
        assert set(selected) == set(EXAMPLE_INPUTS)
        assert digest.hexdigest() == chosen["result"]["audits"][method]["trajectory_sha256"]
        if method == "default_setup_default_solve":
            assert digest.hexdigest() == replay["source_sha256"]
        records[method] = selected
        sources[str(path)] = digest.hexdigest()
    examples = []
    for j, index in enumerate(EXAMPLE_INPUTS):
        joint = records["bandit_lstdq"][index]
        default = records["default_setup_default_solve"][index]
        original, reference = joint["outcome"], default["outcome"]
        for outcome in (original, reference):
            assert len(outcome["primary_attempts"]) == 1
            assert outcome["primary_attempts"][0]["status"] == "success"
            assert not outcome["failed"] and not outcome["fallback_used"]
        first = original["primary_attempts"][0]
        assert not any(first["cycle_forced_default_actions"])
        assert not first["cycle_explored"][0]
        weights, residuals = first["cycle_actions"], [1.0] + first["cycle_residuals"]
        assert len(weights) == first["cycles"] == len(residuals) - 1
        assert residuals[-1] <= 1e-6 and np.isfinite(residuals).all()
        np.testing.assert_array_equal(chosen["w"][1, index-1, :len(weights)], weights)
        np.testing.assert_allclose(chosen["r"][1, index-1, :len(residuals)], residuals, atol=1e-14)
        replayed = baseline[index]
        assert joint["mkw"] == default["mkw"] == replayed["pde_parameters"]
        assert default["params"] == replayed["setup"]
        assert replayed["cycle_count_matches"] and replayed["final_residual_matches"]
        assert replayed["cycles"] == reference["iterations"]
        assert replayed["replay_final_residual"] == replayed["original_final_residual"]
        assert len(replayed["relative_residuals"]) == replayed["cycles"] + 1
        np.testing.assert_allclose(chosen["a"][[3, 0], index-1, 2],
                                   [original["native_runtime"], reference["native_runtime"]],
                                   rtol=0, atol=1e-10)
        examples.append(dict(label="ABC"[j], problem_index=index, cycles=len(weights),
            weights=weights, relative_residuals=residuals,
            exploration_cycles=[k for k, value in enumerate(first["cycle_explored"]) if value],
            cycle_epsilons=first["cycle_epsilons"], setup=first["params"],
            matrix_kwargs=joint["mkw"], context=joint["context"], attempt_count=1,
            color=COLORS[j], linestyle=STYLES[j], marker=MARKERS[j],
            native_sec=original["native_runtime"], default_native_sec=reference["native_runtime"],
            native_reduction_pct=100*(1-original["native_runtime"]/reference["native_runtime"]),
            default_cycles=replayed["cycles"], default_relative_residuals=replayed["relative_residuals"]))
    return dict(case=EXAMPLE_CASE, grid=60, seed=1, method="bandit_lstdq", window=[3001, 5000],
        examples=examples, source_sha256=sources, default_residual_replay=replay,
        initial_weights=[row["weights"][0] for row in examples],
        combined_native_reduction_pct=100*(1-sum(row["native_sec"] for row in examples)
                                           /sum(row["default_native_sec"] for row in examples)),
        selection_rule="User-approved consecutive inputs 3882–3884 with different non-exploratory initial weights.",
        interpretation="Individual executed online schedules, including marked random actions; Default residual histories are verified replays. Original measured timings are retained. This is a complete-method comparison, not a controller-only ablation or an optimality claim.")


def draw(data):
    fig = plt.figure(figsize=(9.15, 3.0))
    # Two main blocks; the right block contains a tightly paired comparison.
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.4], wspace=.30)
    schedule = fig.add_subplot(outer[0, 0])
    residual_grid = outer[0, 1].subgridspec(1, 2, wspace=.06)
    convergence = fig.add_subplot(residual_grid[0, 0])
    default = fig.add_subplot(residual_grid[0, 1], sharey=convergence)
    axes = (schedule, convergence, default)
    for ex in data["examples"]:
        default.semilogy(np.arange(ex["default_cycles"]+1), ex["default_relative_residuals"],
            color="#111111", linestyle=ex["linestyle"], marker=ex["marker"],
            lw=.85, ms=2.5, markerfacecolor="white", markeredgewidth=.6, zorder=1)
    schedule.axhline(1.0, color="#111111", ls="--", lw=.9, zorder=0)
    for ex in reversed(data["examples"]):
        style = dict(color=ex["color"], linestyle=ex["linestyle"], marker=ex["marker"],
                     lw=1.05, ms=3.0, markerfacecolor="white", markeredgewidth=.7, zorder=3)
        schedule.plot(np.arange(ex["cycles"]), ex["weights"],
            label=f'{ex["label"]}: input {ex["problem_index"]} ({ex["cycles"]} cycles)', **style)
        convergence.semilogy(np.arange(ex["cycles"]+1), ex["relative_residuals"], **style)
        if ex["exploration_cycles"]:
            k = np.array(ex["exploration_cycles"])
            schedule.scatter(k, np.array(ex["weights"])[k], marker="x", s=30,
                             color="#202020", linewidths=1.0, zorder=6)
    end = max(max(ex["cycles"], ex["default_cycles"]) for ex in data["examples"])
    ticks = np.arange(0, end+1, 2)
    schedule.set(xlabel=r"Cycle index $k$", ylabel=r"Relaxation weight $w_k$",
                 xlim=(-.35, end-.3), ylim=(.88, 3.13), xticks=ticks, yticks=[1, 1.5, 2, 2.5, 3])
    for ax in (convergence, default):
        ax.set(xlabel="Completed cycles", xlim=(-.35, end+.5), ylim=(1e-7, 2.0),
               xticks=np.arange(0, end+1, 4), yticks=[1, 1e-2, 1e-4, 1e-6])
        ax.axhline(1e-6, color="#666666", ls=":", lw=.7, zorder=0)
    convergence.set_ylabel("Relative residual")
    default.tick_params(axis="y", labelleft=False)
    schedule.set_title("(a) Learned relaxation schedules", loc="left", pad=8)
    convergence.set_title("(b) LinUCB + LSTDQ", loc="left", pad=8)
    default.set_title(r"(c) Default ($w=1$)", loc="left", pad=8)
    handles, labels = schedule.get_legend_handles_labels()
    handles, labels = list(reversed(handles)), list(reversed(labels))
    handles.extend([Line2D([], [], color="#111111", ls="--", lw=.9),
                    Line2D([], [], color="#202020", marker="x", ms=5, ls="None", markeredgewidth=1.0)])
    labels.extend([r"Default ($w=1$)", "Exploration"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .915), ncol=5,
               frameon=False, handlelength=2.3, columnspacing=1.2, fontsize=7)
    fig.suptitle(r"Learned relaxation schedules (Diffusion $60^3$, seed 1, LinUCB + LSTDQ)",
                 fontsize=9.5, y=.995)
    fig.subplots_adjust(left=.065, right=.992, bottom=.175, top=.755)
    for ax in axes:
        ax.grid(False)
    return fig
