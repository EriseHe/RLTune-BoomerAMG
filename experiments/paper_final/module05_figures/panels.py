"""Figure designs. All performance panels use paired saved costs."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from .data import SOURCES, SOURCE_NAMES, METHODS, LABELS, reduction
from .style import COLORS, SEED_COLORS, BLANK, panel, clean


CORE = ("fixed", "oracle", "periodic", "prefix", "rl")
BASE_CAPTION = (
    "Diffusion 60³; six frozen Module 04 checkpoints, two frozen hierarchy sources, and the same 100 fresh test inputs. "
    "Native solve cost excludes controller overhead and common primary setup, but includes recovery setup and solve work. "
    "Ten prescribed test inputs have three timing repetitions; timings are averaged within each case/policy before aggregation. "
    "Best fixed and per-problem fixed are test-hindsight diagnostics over the 41-point grid 1.00:0.05:3.00. "
    "Schedules were chosen on separate development inputs."
)


def method_legend(fig, methods, *, y=-.02, ncol=3):
    handles = [Line2D([], [], color=COLORS[m], label=LABELS[m], linewidth=1.5) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=ncol, frameon=False)


def overview(data, out, *, full_cost=False):
    methods = ("fixed", "periodic", "prefix", "rl", "native", "chebyshev") if full_cost else CORE
    field = "native_total" if full_cost else "native"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, layout="constrained")
    offsets = np.linspace(-.14, .14, len(data.seeds))
    for j, (ax, source) in enumerate(zip(axes, SOURCES)):
        for row, method in enumerate(methods):
            vals = data.gains(source, method, field=field)
            ax.scatter(vals, row+offsets, s=14, facecolors="white", edgecolors=COLORS[method], linewidth=.8, zorder=3)
            ax.errorbar(vals.mean(), row, xerr=vals.std(ddof=1), fmt="D", markersize=4,
                        color=COLORS[method], capsize=3, linewidth=1.3, zorder=4)
        ax.axvline(0, color=".35", linewidth=.6)
        ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
        ax.set_ylim(len(methods)-.6, -.6)
        ax.set_xlim((-7, 33) if full_cost else (27, 46))
        ax.set_xlabel("Setup + solve time reduction (%)" if full_cost else "Solve time reduction (%)")
        panel(ax, "ab"[j], SOURCE_NAMES[source]); clean(ax)
    fig.suptitle("Conventional references: setup costs included" if full_cost else "No-overhead savings on matched hierarchies")
    handles = [Line2D([], [], marker="o", markerfacecolor="white", color=".4", linestyle="none", label="Individual seed"),
               Line2D([], [], marker="D", color=".4", label="Mean ± one SD")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.56, -.08), ncol=2, frameon=False)
    caption = BASE_CAPTION + " Each seed contributes one cumulative reduction against matched weight 1; open points are seeds, diamonds and whiskers are mean ± sample SD, not confidence intervals."
    if full_cost:
        caption += " This figure uses native setup + solve costs instead of solve-only costs. Native references include their own smoother preparation; these are the tested configurations, not exhaustive conventional tuning. Controller overhead remains excluded."
    out.save(fig, "11_native_references" if full_cost else "01_policy_savings", "Native setup + solve comparison" if full_cost else "Aggregate savings and seed variation", caption,
             takeaway="Preparation cost changes the conventional-smoother comparison." if full_cost else "RL beats constants on Joint hierarchies, while prefix–tail is similarly effective.")


def paired_advantage(data, out):
    comparators = ("fixed", "oracle", "periodic", "prefix")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, layout="constrained")
    for j, (ax, source) in enumerate(zip(axes, SOURCES)):
        ax.axvspan(-22, 0, color="#a84949", alpha=.035)
        ax.axvspan(0, 18, color="#2166ac", alpha=.035)
        for row, comparator in enumerate(comparators):
            vals = data.gains(source, "rl", reference=comparator)
            for i, seed in enumerate(data.seeds):
                ax.plot(vals[i], row+(i-2.5)*.055, "o", color=SEED_COLORS[i], markersize=3.4)
            ax.errorbar(vals.mean(), row, xerr=vals.std(ddof=1), fmt="D", color="black", markersize=3.5,
                        capsize=3, linewidth=.8, zorder=5)
        ax.axvline(0, color=".25", linewidth=.75)
        ax.set(xlim=(-22, 18), ylim=(3.6, -.6), yticks=range(4), yticklabels=[LABELS[m] for m in comparators],
               xlabel="RL time reduction vs comparator (%)")
        panel(ax, "ab"[j], SOURCE_NAMES[source]); clean(ax)
    fig.suptitle("Paired advantage of RL: positive values favor RL")
    handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=f"Seed {s}", markersize=3.5) for s,c in zip(data.seeds, SEED_COLORS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.56,-.08), ncol=6, frameon=False, columnspacing=1.0)
    out.save(fig, "02_paired_rl_advantage", "Does RL beat each comparator?", BASE_CAPTION +
        " Each value is 100(1 − total RL cost / total comparator cost), paired within checkpoint and hierarchy source. Black diamonds show means, whiskers show one sample SD across six seeds. Positive values favor RL; zero means equal total cost.",
        takeaway="The strongest contrast is RL vs constants; the Joint prefix–tail comparison straddles zero.")


def trajectory_image(ax, values, *, cmap, norm):
    """Display recorded [problem, cycle] arrays using the Module 04 axis convention."""
    problems, cycles = values.shape
    im = ax.imshow(np.ma.masked_invalid(values.T), aspect="auto", origin="lower",
                   cmap=cmap, norm=norm, interpolation="nearest",
                   extent=(.5, problems+.5, .5, cycles+.5))
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_yticks([1, *range(5, cycles+1, 5)])
    return im


def matched_heatmaps(data, out, *, residual=False, all_seeds=True):
    width = data.maximum_cycles()
    cmap = plt.get_cmap("magma_r" if residual else "viridis").copy(); cmap.set_bad(BLANK)
    norm = LogNorm(1e-6, 10, clip=False) if residual else Normalize(1, 3)
    seeds = data.seeds if all_seeds else [min(data.seeds)]
    for source in SOURCES:
        for seed in seeds:
            fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.8), sharex=True, sharey=True, layout="constrained")
            for j, (ax, method) in enumerate(zip(axes.flat, METHODS)):
                values = data.heatmap(seed, source, method, "residuals" if residual else "actions", width)
                im = trajectory_image(ax, values, cmap=cmap, norm=norm)
                panel(ax, "abcdef"[j], LABELS[method])
                if j >= 4: ax.set_xlabel("Test problem rank")
                if j % 2 == 0: ax.set_ylabel("AMG cycle")
                if method == "prefix":
                    spec = data.protocol["policies"][data.policy(seed, source, method, data.cases[0])]
                    ax.axhline(spec["prefix_cycles"]+.5, color=".3", linestyle="--", linewidth=.65)
                    ax.text(.035, .93, f"After cycle {spec['prefix_cycles']}: " +
                            rf"$w={spec['tail_weight']:g}$", transform=ax.transAxes, va="top", fontsize=7)
            cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.025, shrink=.91, extend="min" if residual else "neither")
            cb.set_label(r"Relative residual $\|r_k\|/\|r_0\|$" if residual else r"Relaxation weight $w_k$")
            if residual:
                cb.set_ticks([1e-6, 1e-4, 1e-2, 1, 10])
            else:
                cb.set_ticks([1, 1.5, 2, 2.5, 3])
            fig.suptitle(f"{'Residual contraction' if residual else 'Executed relaxation weights'} · seed {seed}\n{SOURCE_NAMES[source]}")
            fig.text(.48, -.025, "Identical problem order; gray = solve completed; dashed line = prefix–tail boundary.", ha="center", fontsize=7)
            main = seed == min(data.seeds) and source == "bandit_lstdq"
            prefix = "04_matched_residual_heatmaps" if residual else "03_matched_action_heatmaps"
            name = prefix if main else f"all_seeds/{'residuals' if residual else 'actions'}_{source}_seed{seed}"
            caption = BASE_CAPTION + f" Shown checkpoint: seed {seed}; hierarchy source: {SOURCE_NAMES[source]}. " + (
                "Colors show recorded relative residual after each cycle; logarithmic scale is identical in all residual panels, and the under-range color includes residuals below 10⁻⁶. " if residual else
                "Colors show the actual executed relaxation weight, with the identical 1–3 color scale in every action panel. ") + (
                "Only the first prescribed test execution is drawn; action sequences are not averaged across repetitions. "
                "The horizontal axis is test problem rank in a single shared order determined by mean weight-1 cycle count across both sources and all checkpoints, ties by original case ID. "
                "The vertical axis is the one-based AMG cycle, increasing upward, matching Module 04's axis orientation. "
                "In the prefix–tail panel, the dashed horizontal line separates the alternating prefix from the constant-weight tail; the tail continues until convergence. "
                "Gray cells are unexecuted after termination, never zero-filled weights or extrapolated residuals. Every displayed primary solve succeeded without recovery. "
                "The main example uses the lowest numbered checkpoint, not the best-performing checkpoint; all checkpoints and both sources are exported.")
            out.save(fig, name, f"{'Residual' if residual else 'Action'} heatmaps — {SOURCE_NAMES[source]}, seed {seed}", caption,
                     category="main" if main else "all-seeds", takeaway="Compare the pattern of actions with the length of the executed solve.",
                     inputs={"seed": seed, "source": source, "trace_repeat": 0, "maximum_cycle_shown": width,
                             "x_axis": "test_problem_rank", "y_axis": "amg_cycle_increasing_upward"})


def rl_seed_atlas(data, out):
    width = data.maximum_cycles(); cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(BLANK)
    for source in SOURCES:
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.65), sharex=True, sharey=True, layout="constrained")
        for j, (ax, seed) in enumerate(zip(axes.flat, data.seeds)):
            im = trajectory_image(ax, data.heatmap(seed, source, "rl", "actions", width),
                                  cmap=cmap, norm=Normalize(1,3))
            gain = data.gains(source, "rl")[j]
            panel(ax, "abcdef"[j], f"Seed {seed} · saving {gain:.1f}%")
            if j >= 4: ax.set_xlabel("Test problem rank")
            if j % 2 == 0: ax.set_ylabel("AMG cycle")
        cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.025, shrink=.9); cb.set_label(r"RL weight $w_k$"); cb.set_ticks([1,1.5,2,2.5,3])
        fig.suptitle(f"Frozen RL across all six checkpoints\n{SOURCE_NAMES[source]}")
        fig.text(.48,-.02,"Gray = terminated; each panel uses the same 100 problem columns.",ha="center",fontsize=7)
        out.save(fig, f"05_rl_seed_atlas_{source}", f"RL action variation — {SOURCE_NAMES[source]}", BASE_CAPTION +
                 " Each panel is one trained checkpoint. Actual first-test action sequences are shown, using the common reference-difficulty problem order horizontally and one-based AMG cycles increasing upward. Color/cycle axes are identical across panels. Annotated savings use all prescribed timing repetitions, unlike the single trajectories shown. "
                 "Hierarchy selections can differ between checkpoints; the panels are not identical-hierarchy comparisons across seeds.",
                 takeaway="Different learned action patterns can achieve similar savings; seed variation remains visible.")


def gain_heatmaps(data, out):
    comps = ("fixed", "oracle", "periodic", "prefix")
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 6.5), sharex=True, sharey=True, layout="constrained")
    clipped = {}
    for i, comparator in enumerate(comps):
        for j, source in enumerate(SOURCES):
            ax = axes[i,j]; values = data.paired_gains(source, comparator)
            clipped[f"{source}/{comparator}"] = int(np.sum(np.abs(values)>40))
            im = ax.imshow(values, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),
                           interpolation="nearest", extent=(.5,100.5,6.5,.5))
            panel(ax, "abcdefgh"[i*2+j], f"RL vs {LABELS[comparator]}")
            ax.set_yticks(range(1,7)); ax.set_xticks([1,25,50,75,100])
            if j == 0: ax.set_ylabel("Training seed")
            if i == 3: ax.set_xlabel("Test problem rank")
    for j,source in enumerate(SOURCES): axes[0,j].text(.5,1.35,SOURCE_NAMES[source],transform=axes[0,j].transAxes,ha="center",fontsize=9)
    cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,extend="both",shrink=.86)
    cb.set_label("Per-problem RL time reduction (%)");cb.set_ticks([-40,-20,0,20,40])
    fig.suptitle("Where RL helps or hurts: blue favors RL",y=1.045)
    out.save(fig,"06_casewise_gain_heatmaps","Per-problem gain maps",BASE_CAPTION+
        " Each cell is 100(1 − RL cost / comparator cost) for one checkpoint and one test problem, after averaging its prescribed repetitions. "
        "The same 100 input ranks appear in every panel. Blue favors RL; red favors the comparator. Colors saturate outside ±40%, indicated by colorbar extensions; no cases are omitted. "
        "These cellwise percentages differ from cumulative cost reductions and should not be averaged to reproduce the main table. The 600 cells per panel reuse 100 inputs.",
        takeaway="The advantage over constants is widespread; the prefix–tail comparison is mixed.",inputs={"color_saturation_counts":clipped})


def paired_scatter(data,out):
    comparators=("fixed","prefix")
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.0),layout="constrained")
    allvalues=np.concatenate([data.values(h,m).ravel() for h in SOURCES for m in (*comparators,"rl")])*1000
    low=max(0,float(allvalues.min())*.85); high=float(allvalues.max())*1.07
    for i,source in enumerate(SOURCES):
        for j,comp in enumerate(comparators):
            ax=axes[i,j]; x=data.values(source,comp)*1000;y=data.values(source,"rl")*1000
            for k,seed in enumerate(data.seeds):ax.scatter(x[k],y[k],s=9,color=SEED_COLORS[k],alpha=.5,linewidths=0,label=f"Seed {seed}",rasterized=True)
            ax.plot([low,high],[low,high],color=".3",linestyle="--",linewidth=.8)
            ax.set(xlim=(low,high),ylim=(low,high),xlabel=f"{LABELS[comp]} cost (ms)",ylabel="Frozen RL cost (ms)")
            ax.set_aspect("equal",adjustable="box");clean(ax,"both")
            panel(ax,"abcd"[i*2+j],f"{'Setup-only' if i==0 else 'Joint'} · vs {LABELS[comp]}")
            ax.text(.05,.94,"Below diagonal: RL faster",transform=ax.transAxes,va="top",fontsize=7)
    fig.suptitle("Matched per-problem native solve costs")
    handles=[Line2D([],[],marker="o",color=c,linestyle="none",markersize=4,label=f"Seed {s}") for s,c in zip(data.seeds,SEED_COLORS)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,-.035),ncol=6,frameon=False,columnspacing=1)
    out.save(fig,"07_paired_cost_scatter","Matched-cost scatter plots",BASE_CAPTION+
        " Each point is one checkpoint/problem pair; 600 points per panel reuse the same 100 inputs. Costs are in milliseconds to display the scale of the individual solves. "
        "Both axes have the same scale, and all panels share limits. Points below the identity line favor RL; every observation is shown. Colors identify training seeds, not different matrix samples.",
        takeaway="RL is mostly below best fixed; RL and prefix–tail cluster around equal cost.")


def weight_scan(data,out):
    weights=np.array([1+i*.05 for i in range(41)])
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.15),layout="constrained")
    recovery={}
    for source,color,label in zip(SOURCES,["#8b6096","#2166ac"],["Setup-only hierarchies","Joint hierarchies"]):
        ratios=[]; recovered=[]
        for w in weights:
            policy=f"fixed_{w:.2f}"
            total=np.array([sum(data.cells[s,source,c,policy]["native"] for c in data.cases) for s in data.seeds])
            ratios.append(total/data.totals(source,"reference"))
            recovered.append(sum(data.cells[s,source,c,policy]["recovery"] for s in data.seeds for c in data.cases))
        ratios=np.array(ratios); mean=ratios.mean(axis=1)
        axes[0].plot(weights,mean,color=color,label=label)
        axes[0].fill_between(weights,ratios.min(axis=1),ratios.max(axis=1),color=color,alpha=.14)
        savings=100*(1-ratios);keep=(weights>=1.3-1e-8)&(weights<=1.75+1e-8)
        axes[1].plot(weights[keep],savings[keep].mean(axis=1),"o-",color=color,markersize=3)
        axes[1].fill_between(weights[keep],savings[keep].min(axis=1),savings[keep].max(axis=1),color=color,alpha=.14)
        recovery[source]=recovered
    # Derive the all-recovery boundary from saved data; never hardcode a conclusion.
    universal=np.all(np.array(list(recovery.values()))==len(data.seeds)*len(data.cases),axis=0)
    suffix=[i for i in range(len(weights)) if np.all(universal[i:])]
    if suffix:
        boundary=weights[suffix[0]]
        axes[0].axvspan(boundary,3,color="#a84b4b",alpha=.075)
        axes[0].text((boundary+3)/2,.68,"Every tested case\nused recovery",ha="center",fontsize=7)
    axes[0].axhline(1,color=".4",linestyle="--",linewidth=.7)
    axes[0].set(yscale="log",xlabel=r"Constant relaxation weight $w$",ylabel=r"Cost ratio $C(w)/C(1)$",xlim=(1,3),ylim=(.5,7))
    axes[0].set_yticks([.5,1,2,5]);axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].set(xlabel=r"Constant relaxation weight $w$",ylabel="Solve time reduction (%)",xlim=(1.28,1.77))
    panel(axes[0],"a","Complete 41-weight search");panel(axes[1],"b","Zoom near the fixed-weight minimum")
    for ax in axes:clean(ax,"both")
    fig.suptitle("Fixed-weight response: the optimum and the recovery region")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.55,-.08),ncol=2,frameon=False)
    out.save(fig,"08_fixed_weight_scan","The exhaustive fixed-weight scan",BASE_CAPTION+
        " Panel (a) shows the full grid, including all failed primary attempts and recovery costs, on a logarithmic cost-ratio axis. "
        "Panel (b) explicitly zooms to 1.30–1.75 and shows time reductions. Curves are seed means and shaded bands are seed minima/maxima, not confidence bands. The shaded recovery region is derived from raw outcomes.",
        takeaway="A narrow useful fixed-weight region sits near 1.55–1.60; large constants require recovery.",inputs={"weights":weights.tolist(),"recovered_cases":recovery})


def convergence_examples(data,out,*,all_seeds=True):
    # Select only from reference difficulty; do not select on an RL win or loss.
    ranks=[int(round(q*(len(data.cases)-1))) for q in (.1,.5,.9)]
    case_ids=[int(data.order[r]) for r in ranks]
    methods=("reference","fixed","periodic","prefix","rl")
    for source in SOURCES:
        for seed in data.seeds if all_seeds else [min(data.seeds)]:
            fig,axes=plt.subplots(2,3,figsize=(7.2,4.8),sharex="col",sharey="row",layout="constrained",height_ratios=[1.35,1])
            for j,case in enumerate(case_ids):
                maximum=0
                for method in methods:
                    trace=data.trace(seed,source,method,case);n=len(trace["actions"]);maximum=max(maximum,n)
                    axes[0,j].semilogy(np.arange(n+1),np.r_[1.,trace["residuals"]],color=COLORS[method],label=LABELS[method],
                                      linewidth=1.3 if method=="rl" else .95,alpha=1 if method=="rl" else .9)
                    axes[1,j].step(np.arange(1,n+1),trace["actions"],where="mid",marker=".",markersize=2,
                                   color=COLORS[method],linewidth=1.1 if method=="rl" else .8,alpha=.9)
                axes[0,j].axhline(1e-6,color=".3",linestyle=":",linewidth=.7)
                axes[0,j].set_ylim(2e-8,3)
                axes[1,j].set(xlim=(0,maximum+1.25),ylim=(.9,3.1),xlabel="AMG cycle",yticks=[1,1.5,2,2.5,3])
                panel(axes[0,j],"abc"[j],f"Rank {ranks[j]+1} · problem {case+1}")
                for ax in axes[:,j]:clean(ax,"both");ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
            axes[0,0].set_ylabel(r"$\|r_k\|/\|r_0\|$");axes[1,0].set_ylabel(r"Executed weight $w_k$")
            fig.suptitle(f"Convergence and action schedules · seed {seed}\n{SOURCE_NAMES[source]}")
            method_legend(fig,methods,y=-.065,ncol=5)
            main=source=="bandit_lstdq" and seed==min(data.seeds)
            name="09_convergence_examples" if main else f"all_seeds/convergence_{source}_seed{seed}"
            out.save(fig,name,f"Convergence examples — {SOURCE_NAMES[source]}, seed {seed}",BASE_CAPTION+
                " Cases are chosen reproducibly at ranks nearest the 10th, 50th and 90th percentiles of the shared reference-cycle difficulty order. "
                "Selection does not depend on RL's advantage. Top panels show actual first-execution residuals, including the initial relative residual of one; the dotted line is the 10⁻⁶ tolerance. "
                "Bottom panels mark the actual weight at each integer cycle and join adjacent samples with a step line. Curves stop after execution; no future actions or residual continuation is inferred. "
                "The opening prefix can temporarily increase residual without preventing eventual convergence.",category="main" if main else "all-seeds",
                takeaway="Nonconstant actions can improve the whole trajectory even when an individual cycle is less contractive.",
                inputs={"seed":seed,"source":source,"case_ids_0_based":case_ids,"difficulty_ranks_1_based":[r+1 for r in ranks]})


def overhead(data,out):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True,layout="constrained")
    for j,(ax,source) in enumerate(zip(axes,SOURCES)):
        for row,method in enumerate(CORE):
            native=data.gains(source,method).mean();inclusive=data.gains(source,method,field="inclusive").mean()
            ax.plot([inclusive,native],[row,row],color=COLORS[method],linewidth=2.1)
            ax.plot(native,row,"o",color=COLORS[method],markersize=5)
            ax.plot(inclusive,row,"s",markerfacecolor="white",markeredgecolor=COLORS[method],markersize=5)
            if method=="rl":ax.text((inclusive+native)/2,row+.35,f"{inclusive-native:.2f} pp",ha="center",fontsize=7)
        ax.set(yticks=range(len(CORE)),yticklabels=[LABELS[m] for m in CORE],ylim=(4.8,-.5),xlim=(30,44),xlabel="Solve time reduction vs weight 1 (%)")
        panel(ax,"ab"[j],SOURCE_NAMES[source]);clean(ax)
    fig.suptitle("Controller overhead: change in the observed savings")
    handles=[Line2D([],[],marker="o",color=".3",linestyle="none",label="Native solve only"),
             Line2D([],[],marker="s",markerfacecolor="white",color=".3",linestyle="none",label="Solve + controller")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.56,-.08),ncol=2,frameon=False)
    out.save(fig,"10_overhead_tradeoff","No overhead versus controller-inclusive performance",BASE_CAPTION+
        " Filled circles show the mean native reduction and open squares include controller work, each relative to the corresponding cost definition for matched weight 1. "
        "Lines join two accounting views of the same trials, not separate experiments; pp denotes percentage points. Common primary setup is excluded in both views.",
        takeaway="RL's overhead reduces its savings by about 2.4–2.6 percentage points; schedule overhead is small.")


def ecdf(data,out):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.05),sharey=True,layout="constrained")
    comparators=("fixed","oracle","periodic","prefix")
    lo=min(float(data.paired_gains(s,m).min()) for s in SOURCES for m in comparators)
    hi=max(float(data.paired_gains(s,m).max()) for s in SOURCES for m in comparators)
    for j,(ax,source) in enumerate(zip(axes,SOURCES)):
        for comp in comparators:
            values=np.sort(data.paired_gains(source,comp).ravel())
            ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=COLORS[comp],label=f"vs {LABELS[comp]}")
        ax.axvline(0,color=".25",linestyle="--",linewidth=.75)
        ax.set(xlim=(lo-2,hi+2),ylim=(0,1),xlabel="Per-problem RL time reduction (%)")
        panel(ax,"ab"[j],SOURCE_NAMES[source]);clean(ax,"both")
    axes[0].set_ylabel("Empirical cumulative fraction")
    fig.suptitle("Distribution of paired gains: further right favors RL")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.56,-.105),ncol=2,frameon=False)
    out.save(fig,"12_paired_gain_ecdf","Distribution of per-problem advantages",BASE_CAPTION+
        " Each empirical CDF pools the 600 checkpoint/problem pairs descriptively; these reuse 100 inputs and are not 600 independent matrices. "
        "The horizontal coordinate is a per-problem relative gain, not a cumulative time reduction. All observations are included, and neither a confidence band nor an independence assumption is implied.",
        takeaway="The full distribution distinguishes consistent savings from near-ties and unfavorable tails.")
