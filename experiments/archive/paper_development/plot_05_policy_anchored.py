"""Run 04 figures for the anchored (2.6,1) baseline and all six checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from experiments.archive.paper_development.module05_figures.style import Exporter, apply_style, COLORS, SEED_COLORS, BLANK, panel, clean
from experiments.archive.paper_development.module05_figures.panels import trajectory_image
import matplotlib.pyplot as plt

METHODS = ("reference", "fixed", "oracle", "periodic", "periodic13", "rl")
MAIN = ("fixed", "oracle", "periodic", "rl")
PALETTE = {**COLORS,"periodic13":"#00856c"}
NOTE = ("Module 05 Run 04: 60³ diffusion, the same 100 previously observed test problems and six Run 03 Joint checkpoints, including refreshed seed 4. "
        "Three new timing repetitions are averaged within case before summing. Native continuation excludes common initial setup and controller overhead, and includes all failed attempts and recovery. "
        "Fixed-weight asterisks identify retained successful 41-grid choices, independently retimed here. "
        "The (2.6,1) coefficient is prescribed by the anchored smoothing minimax rule; it replaces (2.5,1). Periodic (1,3) is retained. "
        "The theory concerns normalized SPD smoothing and an exact two-grid bound, not full multilevel runtime optimality. This is not a fresh holdout or new independent seeds. ")



def read(path): return json.loads(Path(path).read_text())


def select_seed(data):
    """Select by cumulative native reduction, never a fastest repeat or case mean."""
    seeds=data["seeds"]
    if len(seeds)!=6 or len(set(seeds))!=6:
        raise ValueError("Selection requires all six distinct checkpoint seeds")
    base=np.asarray(data["costs"]["reference"]["native"],dtype=float)
    rl=np.asarray(data["costs"]["rl"]["native"],dtype=float)
    expected=(len(seeds),data["protocol"]["test_cases"])
    if base.shape!=expected or rl.shape!=expected:
        raise ValueError("Missing or misaligned matched problems")
    if not np.isfinite(base).all() or not np.isfinite(rl).all() or (base<=0).any() or (rl<0).any():
        raise ValueError("Invalid timing observations")
    gains=100*(1-rl.sum(axis=1)/base.sum(axis=1))
    ranking=sorted(range(len(seeds)),key=lambda i:(-gains[i],seeds[i]))
    selected=ranking[0]
    return {"seed":seeds[selected],"reduction_pct":float(gains[selected]),
        "criterion":"Maximum cumulative no-overhead RL time reduction versus matched weight 1",
        "formula":"100 * (1 - sum_cases(mean_repeats(RL native cost)) / sum_cases(mean_repeats(w1 native cost)))",
        "costs":"Native solve plus recovery setup/solve; common primary setup and controller overhead excluded",
        "tie_break":"Lowest checkpoint seed on an exact tie",
        "selection_timing":"After completed Run 04 evaluation; illustrative best observed seed",
        "trajectory_repetition":1,
        "all_seed_statistics_retained":True,
        "ranking":[{"seed":seeds[i],"reduction_pct":float(gains[i])} for i in ranking]}


def generate(output):
    output=Path(output)
    if read(output/"numerical_complete.json")["status"] != "complete":
        raise ValueError("Complete the audited timing experiment before selecting figures")
    data=read(output/"analysis/summary.json")
    selection=select_seed(data)
    best_seed=selection["seed"]
    selection_caption=(f"Main trajectory examples use seed {best_seed}, selected after evaluation for the largest "
        f"cumulative native RL time reduction versus matched weight 1 ({selection['reduction_pct']:.2f}%). "
        "This is the best-performing observed seed, not an estimate of typical performance. "
        "All six checkpoints remain in aggregate plots and the seed atlas. ")
    arrays=np.load(output/"analysis/trace_arrays.npz")
    destination=output/"analysis/paper_figures_best_seed";destination.mkdir(parents=True,exist_ok=True)
    apply_style(); export=Exporter(destination,600)
    seeds=data["seeds"]; labels=data["labels"]; order=np.asarray(data["case_order_0_based"])
    if tuple(data["main_methods"])!=MAIN: raise ValueError("Main heatmap methods differ from the requested four")
    def values(method,field="native",old=False):
        return np.asarray(data["comparison_costs" if old else "costs"][method][field])
    def gains(method,reference="reference",field="native",old=False):
        return 100*(1-values(method,field,old).sum(axis=1)/values(reference,field,old).sum(axis=1))
    def trace(seed,method,field): return arrays[f"seed{seed}__{method}__{field}"]
    def save(fig,name,title,caption,**kw): export.save(fig,name,title,NOTE+caption,**kw)
    try:
        fig,ax=plt.subplots(figsize=(7.2,3.7),layout="constrained")
        for i,m in enumerate(METHODS):
            g=gains(m)
            ax.scatter(g,i+np.linspace(-.14,.14,6),s=16,facecolors="white",edgecolors=PALETTE[m],zorder=3)
            ax.errorbar(g.mean(),i,xerr=g.std(ddof=1),fmt="D",color=PALETTE[m],capsize=3,markersize=4,zorder=4)
        ax.set(yticks=range(len(METHODS)),yticklabels=[labels[m] for m in METHODS],ylim=(len(METHODS)-.4,-.6),xlabel="Native solve time reduction vs weight 1 (%)")
        ax.axvline(0,color=".4",lw=.7);clean(ax)
        fig.suptitle("Run 04 · All six policies · No controller overhead")
        ax.text(.99,.02,"Open points: seeds; diamonds: mean ± SD",transform=ax.transAxes,ha="right",fontsize=7)
        save(fig,"01_policy_savings","Six-policy timing comparison","Each point is one checkpoint's cumulative reduction; whiskers are sample SD across six checkpoints, not confidence intervals.")

        width=max(30,max(int(np.isfinite(trace(s,m,"actions")).sum(axis=1).max()) for s in seeds for m in MAIN))
        for residual in (False,True):
            field="residuals" if residual else "actions"
            cmap=plt.get_cmap("magma_r" if residual else "viridis").copy();cmap.set_bad(BLANK)
            norm=LogNorm(1e-6,10) if residual else Normalize(1,3)
            for seed in seeds:
                seed_width=max(30,max(int(np.isfinite(trace(seed,m,"actions")).sum(axis=1).max()) for m in MAIN))
                fig,axes=plt.subplots(2,2,figsize=(7.2,4.65),sharex=True,sharey=True,layout="constrained")
                for j,(ax,m) in enumerate(zip(axes.flat,MAIN)):
                    matrix=trace(seed,m,field)[order,:seed_width]
                    im=trajectory_image(ax,matrix,cmap=cmap,norm=norm)
                    panel(ax,"abcd"[j],labels[m])
                    if j>=2:ax.set_xlabel("Test problem rank")
                    if j%2==0:ax.set_ylabel("AMG cycle")
                    bad=arrays[f"seed{seed}__{m}__primary_incomplete"][order]
                    if bad.any():
                        lengths=np.isfinite(matrix).sum(axis=1)
                        ax.scatter(np.flatnonzero(bad)+1,lengths[bad]+.35,marker="v",s=10,color="#d95f02",zorder=3)
                cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,shrink=.9,extend="min" if residual else "neither")
                cb.set_label(r"Relative residual $\|r_k\|/\|r_0\|$" if residual else r"Relaxation weight $w_k$")
                cb.set_ticks([1e-6,1e-4,1e-2,1,10] if residual else [1,1.5,2,2.5,3])
                fig.suptitle(f"Run 04 · {'Residual contraction' if residual else 'Executed relaxation weights'} · seed {seed}\nJoint hierarchies"+(" · Best-performing seed" if seed==best_seed else ""))
                fig.text(.48,-.02,"Identical sorted problem columns; gray = no further primary cycle recorded.",ha="center",fontsize=7)
                name=("03_residual_heatmaps" if residual else "02_action_heatmaps") if seed==best_seed else f"all_seeds/{field}_seed{seed}"
                save(fig,name,f"{'Residual' if residual else 'Action'} heatmaps — seed {seed}",
                     "Four requested policies; default and periodic (1,3) are omitted here. Actual first repetition is shown, with no averaged/extrapolated actions. "
                     "The same baseline-difficulty order as Run 01 is used. Gray marks the end of the primary trace; an orange marker indicates recovery/final failure if present. "
                     "Each four-panel figure shares a cycle limit that includes every recorded primary cycle for its checkpoint. "
                     +selection_caption,category="main" if seed==best_seed else "all-seeds",
                     inputs={"seed":seed,"methods":list(MAIN),"x_axis":"test_problem_rank","y_axis":"amg_cycle_increasing_upward", "best_seed_selection":selection})

        fig,axes=plt.subplots(3,2,figsize=(7.2,5.55),sharex=True,sharey=True,layout="constrained")
        cmap=plt.get_cmap("viridis").copy();cmap.set_bad(BLANK)
        for i,(ax,s) in enumerate(zip(axes.flat,seeds)):
            im=trajectory_image(ax,trace(s,"rl","actions")[order,:width],cmap=cmap,norm=Normalize(1,3))
            panel(ax,"abcdef"[i],f"Seed {s} · saving {gains('rl')[i]:.1f}%")
            if i>=4:ax.set_xlabel("Test problem rank")
            if i%2==0:ax.set_ylabel("AMG cycle")
        cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,shrink=.9);cb.set_label("RL relaxation weight")
        fig.suptitle("Run 04 · Frozen RL across all six checkpoints")
        save(fig,"04_rl_seed_atlas","Frozen RL across all seeds","Patterns show first repetitions; annotated cumulative savings average all three repetitions. Shared inputs do not imply identical hierarchies between checkpoints.")

        fig,axes=plt.subplots(2,2,figsize=(7.2,4.65),sharex=True,sharey=True,layout="constrained")
        compared=("reference","periodic","periodic13","rl");seed=best_seed
        w=max(30,max(int(np.isfinite(trace(seed,m,"actions")).sum(axis=1).max()) for m in compared))
        for j,(ax,m) in enumerate(zip(axes.flat,compared)):
            matrix=trace(seed,m,"actions")[order,:w]
            im=trajectory_image(ax,matrix,cmap=cmap,norm=Normalize(1,3))
            panel(ax,"abcd"[j],labels[m])
            if j>=2:ax.set_xlabel("Test problem rank")
            if j%2==0:ax.set_ylabel("AMG cycle")
            bad=arrays[f"seed{seed}__{m}__primary_incomplete"][order]
            if bad.any():ax.scatter(np.flatnonzero(bad)+1,np.isfinite(matrix).sum(axis=1)[bad]+.3,marker="v",s=10,color="#d95f02")
        cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,shrink=.9);cb.set_label("Relaxation weight")
        fig.suptitle(f"Run 04 · Reference, periodic schedules and RL\nBest-performing seed {seed} · Joint hierarchies")
        save(fig,"05_periodic_comparison","Reference and the two periodic schedules","The prescribed anchored (2.6,1) and existing (1,3) schedules are evaluated on matched hierarchies. Neither is assumed runtime-optimal. Gray ends the primary trace; orange markers indicate recovery/final failure if present. "+selection_caption, inputs={"seed":seed,"best_seed_selection":selection})

        fig,ax=plt.subplots(figsize=(7.2,3.4),layout="constrained")
        comparators=[m for m in METHODS if m!="rl"]
        for i,m in enumerate(comparators):
            g=gains("rl",m)
            for j,s in enumerate(seeds):ax.plot(g[j],i+(j-2.5)*.06,"o",color=SEED_COLORS[j],ms=3.8)
            ax.errorbar(g.mean(),i,xerr=g.std(ddof=1),fmt="D",color="black",capsize=3,ms=3.8)
        ax.axvline(0,color=".3",lw=.8);clean(ax)
        ax.set(yticks=range(len(comparators)),yticklabels=[labels[m] for m in comparators],ylim=(len(comparators)-.4,-.6),xlabel="RL time reduction vs comparator (%)")
        fig.suptitle("Run 04 · Direct paired advantage of frozen RL")
        save(fig,"06_rl_advantage","Direct RL advantages","Positive values favor RL. Colored dots are checkpoint seeds; black diamonds/whiskers are mean ± sample SD. Costs include all prescribed attempts and recovery.")

        fig,ax=plt.subplots(figsize=(7.2,3.4),layout="constrained")
        from experiments.archive.paper_development.verify_anchored_schedule import eta
        xx=np.linspace(1,3,401);ax.plot(xx,[eta(x) for x in xx],color="#455a64")
        for x,color,label in ((2.5,"#777777","Prior 2.5"),(2.6,"#d58b19","Run 04: 2.6"),((3+np.sqrt(5))/2,"#2166ac",r"Continuous $a_*$")):
            ax.scatter([x],[eta(x)],color=color,s=22,label=label,zorder=3)
        ax.set(xlabel="High coefficient a, paired with 1",ylabel=r"$\eta(a)=\max_{0\leq t\leq1}t(1-t)^2(1-at)^2$")
        ax.legend(frameon=False);clean(ax,"both")
        fig.suptitle("Anchored polynomial-smoothing objective")
        save(fig,"07_anchored_objective","Why the prescribed coefficient is 2.6","Every ordinate evaluates the objective at its analytic stationary points. The unique continuous minimizer is (3+sqrt(5))/2; 2.6 minimizes it over the 0.05 action grid. This is a smoothing-bound surrogate, not a prediction of measured runtime.")

        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,layout="constrained")
        for j,(ax,comp) in enumerate(zip(axes,("oracle","periodic"))):
            for i,s in enumerate(seeds):
                vals=[]
                for rep in (1,2,3):
                    r=next(r for r in data["per_repeat"] if r["seed"]==s and r["repeat"]==rep and r["method"]=="rl")
                    b=next(r for r in data["per_repeat"] if r["seed"]==s and r["repeat"]==rep and r["method"]==comp)
                    vals.append(100*(1-r["native_continuation_sec"]/b["native_continuation_sec"]))
                ax.plot([1,2,3],vals,"o-",color=SEED_COLORS[i],ms=3,label=f"Seed {s}")
            ax.axhline(0,color=".5",lw=.7);ax.set(xticks=[1,2,3],xlabel="Timing repetition",ylabel="RL time reduction (%)");clean(ax,"both")
            panel(ax,"ab"[j],f"RL vs {labels[comp]}")
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,-.06),ncol=6,frameon=False)
        fig.suptitle("Run 04 · Repeatability on all 100 problems")
        save(fig,"08_timing_repetitions","Three complete timing repetitions","Each point sums one full 100-problem repetition. This shows session repeatability, not three independently trained controllers.")
    finally:
        export.close()
    export.entries.sort(key=lambda entry:(entry["category"]!="main",entry["name"]))
    (destination/"figure_catalog.json").write_text(json.dumps(export.entries,indent=2)+"\n")
    cards=[];captions=["# Module 05 — Run 04 best-seed figure captions", "", selection_caption, ""]
    for entry in export.entries:
        files=entry["files"]
        links=" · ".join(f'<a href="{html.escape(files[e])}">{e.upper()}</a>' for e in ("pdf","svg","png"))
        cards.append(f'<article><img loading="lazy" src="{files["preview"]}" alt="{html.escape(entry["title"])}"><h2>{html.escape(entry["title"])}</h2><p>{links}</p><details><summary>Caption</summary><p>{html.escape(entry["caption"])}</p></details></article>')
        captions.extend([f'## {entry["name"]}',"",entry["caption"],""])
    (destination/"CAPTIONS.md").write_text("\n".join(captions))
    (destination/"index.html").write_text('''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Module 05 · Run 04</title>
<style>body{font:15px/1.5 system-ui;background:#f7f8fa;color:#24313d;max-width:1250px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:white;border:1px solid #dbe1e6;padding:15px}img{width:100%}a{color:#155d9a}details{font-size:13px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Module 05 · Run 04 · Best-seed examples</h1><p>Same six Run 03 checkpoints, including refreshed seed 4 · 100 identical test problems · three new repetitions per problem.</p>
<p>Main heatmaps show global fixed, per-instance fixed, Anchored (2.6,1), and RL. Default and Periodic (1,3) remain in other comparisons. All fixed choices are retained from Run 03 and retimed here.</p>
<p><a href="main_figure_atlas.pdf">Main figures PDF</a> · <a href="figure_pack.zip">Complete figure pack</a> · <a href="CAPTIONS.md">Captions</a></p>'''+"<p>"+html.escape(selection_caption)+'</p><div class="grid">'+"\n".join(cards)+"</div></html>")
    for src in (output/"analysis/summary.json",output/"analysis/trace_arrays.npz",output/"protocol.json",output/"selections.json"):
        shutil.copy2(src,destination/src.name)
    provenance={"run_number":4,"figure_count":len(export.entries),"dpi":600,
                "main_heatmap_methods":list(MAIN),"source_hashes":read(output/"source_manifest.json"),
                "raw_audit":data["audit"],"best_seed_selection":selection,
                "presentation_revision":{"reason":"User-requested best observed RL seed for example trajectories",
                    "script":"experiments/archive/paper_development/plot_05_policy_anchored.py",
                    "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "summary_sha256":hashlib.sha256((output/"analysis/summary.json").read_bytes()).hexdigest(),
                    "trace_arrays_sha256":hashlib.sha256((output/"analysis/trace_arrays.npz").read_bytes()).hexdigest(),
                    "solver_implementation_unchanged":True,"prescribed_schedule":[2.6,1.]}}
    (destination/"provenance.json").write_text(json.dumps(provenance,indent=2))
    shutil.copytree(output/"provenance/source",destination/"reproduction",dirs_exist_ok=True)
    (destination/"best_seed_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    shutil.copy2(__file__,destination/"reproduction/experiments/paper_final"/Path(__file__).name)
    with zipfile.ZipFile(destination/"figure_pack.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(destination.rglob("*")):
            if p.is_file() and p.name!="figure_pack.zip":z.write(p,p.relative_to(destination))
    for e in export.entries:
        for name in e["files"].values():
            if not (destination/name).stat().st_size:raise ValueError("Empty figure export")
    return provenance


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="Completed Run 04 directory")
    args=parser.parse_args()
    result=generate(args.output)
    print(json.dumps({"best_seed_selection":result["best_seed_selection"],"figure_count":result["figure_count"]},indent=2))
