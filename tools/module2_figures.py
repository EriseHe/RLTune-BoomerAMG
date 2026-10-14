"""Tables and figures for Module 2."""
import json
import numpy as np
import matplotlib.pyplot as plt
from module1_figures import COLORS, WEIGHT_CMAP, style
from paper_figure_io import save

METHODS = ("reference", "fixed", "oracle", "periodic", "rl")

def publish(data, output):
    paper = output
    (paper / "sections").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text())
    protocol = json.loads((data / "protocol.json").read_text())
    summary["protocol"] = protocol
    with np.load(data / "arrays.npz", allow_pickle=False) as a:
        seeds = a["seeds"].tolist()
        cases = a["case_ids"]
        costs = a["costs_sec"]
        weights = a["first_repetition_primary_weights"]
        success = a["all_repetitions_success"]
        totals = costs.sum(axis=2)
        native = 100 * (1 - totals[:, :, 0] / totals[:, :1, 0])
        inclusive = 100 * (1 - totals[:, :, 1] / totals[:, :1, 1])
        np.testing.assert_allclose(native, a["native_reductions_pct"])
        np.testing.assert_allclose(inclusive, a["inclusive_reductions_pct"])
    assert success.all() and seeds == [1, 2, 3, 4, 5, 6]
    manifest = {"illustrative_seed": seeds[int(np.argmax(native[:, 4]))]}
    comparisons, rows = [], []
    for j, (method, label) in enumerate(zip(METHODS[:-1], (r"Fixed $w=1$", "Global best fixed",
                                                        "Per-instance best fixed", r"Periodic $(2.85,1.10)$"))):
        values = []
        for field in (0, 1):
            totals = costs[..., field].sum(axis=2)
            gain = 100 * (1 - totals[:, 4] / totals[:, j])
            mean, sd = float(gain.mean()), float(gain.std(ddof=1))
            expected = next(r for r in summary["rl_comparisons"] if r["comparator"] == method and
                            r["metric"] == ("native_continuation_sec", "inclusive_continuation_sec")[field])
            assert expected["seeds"] == 6
            np.testing.assert_allclose([mean, sd], [expected["mean_pct"], expected["sample_sd_pct"]], atol=1e-10)
            comparisons.append(dict(comparator=method, cost=("native", "inclusive")[field],
                                    values=gain.tolist(), mean=mean, sample_sd=sd))
            values.append(rf"${mean:.1f}\,\pm\,{sd:.1f}$\%")
        rows.append(label + " & " + " & ".join(values) + r" \\")
    (paper / "sections/module2_rows.tex").write_text("\n".join(rows) + "\n")

    style()
    methods = ("fixed", "oracle", "periodic", "rl")
    labels = ("Global best fixed", "Per-instance best fixed", "Periodic (2.85, 1.10)", "Frozen RL")
    colors = (COLORS[2], COLORS[1], WEIGHT_CMAP(.63), COLORS[3])
    fig, ax = plt.subplots(figsize=(6.85, 2.35))
    extent, statistics = [], {}
    for i, (method, color) in enumerate(zip(methods, colors)):
        gains = native[:, METHODS.index(method)]
        mean, sd = float(gains.mean()), float(gains.std(ddof=1))
        ax.scatter(gains, i + np.linspace(-.13, .13, 6), s=16, facecolors="white",
                   edgecolors=color, linewidths=.85, zorder=3)
        ax.errorbar(mean, i, xerr=sd, fmt="D", color=color, capsize=2.5,
                    markersize=4, elinewidth=.9, zorder=4)
        extent.extend([gains.min(), gains.max(), mean - sd, mean + sd])
        statistics[method] = dict(values=gains.tolist(), mean=mean, sample_sd=sd)
    padding = .055 * (max(extent) - min(extent))
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.4, -.4),
           xlim=(min(extent) - padding, max(extent) + padding),
           xlabel=r"Native continuation-time reduction relative to $w=1$ (%)")
    fig.suptitle(r"Matched-policy savings (Diffusion $60^3$, six checkpoints)", y=.99, fontsize=9)
    ax.text(.98, 1.075, "Open circles: checkpoints; diamonds: mean ± SD", transform=ax.transAxes,
            ha="right", fontsize=6.5)
    fig.subplots_adjust(left=.235, right=.985, top=.78, bottom=.22)
    save(fig, paper, "matched_policy_savings")

    selected = int(np.argmax(native[:, 4]))
    seed = seeds[selected]
    assert seed == manifest["illustrative_seed"]
    order = np.lexsort((cases, -np.isfinite(weights[selected, 0]).sum(axis=1)))
    heatmap_methods = ("reference", "oracle", "periodic", "rl")
    indices = [METHODS.index(m) for m in heatmap_methods]
    # Display every recorded primary cycle in the selected illustration.
    cap = int(np.max(np.isfinite(weights[selected, indices]).sum(axis=2)))
    assert 1 <= cap <= summary["protocol"]["max_cycles"] == 50
    shown = {m: weights[selected, METHODS.index(m), order, :cap] for m in heatmap_methods}
    ticks = np.unique(np.r_[np.arange(0, cap, 10), cap - 1])
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.6), sharex=True, sharey=True)
    panel_labels = (r"Fixed $w=1$", "Per-instance best fixed", "Periodic (2.85, 1.10)", "Frozen RL")
    for ax, method, title, letter in zip(axes.flat, heatmap_methods, panel_labels, "abcd"):
        im = ax.imshow(shown[method].T, origin="upper", interpolation="nearest", aspect="auto",
                       extent=(.5, 100.5, cap - .5, -.5), cmap=WEIGHT_CMAP, vmin=1, vmax=3, rasterized=True)
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.set(xticks=[1, 25, 50, 75, 100], yticks=ticks)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Cycle index $k$")
    for ax in axes[1]:
        ax.set_xlabel("Test problem rank")
    fig.subplots_adjust(left=.10, right=.855, top=.83, bottom=.14, wspace=.12, hspace=.4)
    color_ax = fig.add_axes([.89, .14, .017, .69])
    fig.colorbar(im, cax=color_ax, label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3])
    fig.suptitle(rf"Matched-policy relaxation weights (Diffusion $60^3$, seed {seed})", y=.97, fontsize=9)
    save(fig, paper, "matched_policy_weights")
    np.savez_compressed(output / "displayed_weights.npz", **shown, case_order_0_based=order,
                        case_ids_sorted=np.asarray(cases)[order])

    return comparisons
