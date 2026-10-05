"""Current-data adapter for all eight Run-04 figure types and timing variants.

Reuses the original Module 05/06 style, exporter, trajectory renderer and
gallery system. No solver runs here; W1 residual traces have a separate audit.
"""
from __future__ import annotations

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()


import argparse
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from experiments.paper_final.module05_figures.style import Exporter, apply_style, COLORS, BLANK, panel, clean
from experiments.paper_final.module05_figures.panels import trajectory_image
from experiments.paper_final.module05_figures.data import case_order, padded_trace, sha256
from experiments.paper_final.plot_05_policy import write_gallery
from experiments.paper_final.verify_period_two_minimax import extrema
import matplotlib.pyplot as plt
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/paper_final/06_policy/20260928_frozen_five_methods"
METHODS = ("bandit_default", "bandit_fixed_prior", "bandit_periodic", "bandit_lstdq")
LABELS = dict(zip(METHODS, ("W1", "Fixed 1.60", "Periodic (2.85, 1.10)", "Frozen RL")))
PALETTE = dict(zip(METHODS, (COLORS["reference"], COLORS["fixed"], COLORS["periodic"], COLORS["rl"])))
CROSSED = (("Periodic-trained", "bandit_periodic", "periodic_setup_rl"),
           ("RL-trained", "rl_setup_periodic", "bandit_lstdq"))
NOTE = ("Current Module 06: 100 fresh diffusion 60³ inputs, one Module 05 trained checkpoint set, three prescribed timing repetitions. "
        "The four main methods are W1, fixed 1.60, periodic (2.85,1.10), and frozen RL, each using its own adapted frozen setup selector. "
        "All learners remain frozen; all formal solves completed without recovery. Inputs are shared, but own-pairing hierarchies may differ. "
        "Formal costs always come from the original completed run, with all three timings averaged within each input. ")
TRACE_NOTE = ("Trajectory panels show the stated actual timing repetition, never averaged actions or a best-repeat selection. "
              "W1 batch logs give cycle counts rather than step traces: its known constant actions are reconstructed, and its residuals "
              "come from 100 separate same-input/same-setup W1 diagnostic replays whose cycle counts and final residuals match all 300 original W1 trials. "
              "These deterministic W1 residual traces are shared across repetition figures; diagnostic replay times never replace formal timings. "
              "Every other trajectory is read directly from that repetition's raw record. "
              "Columns share ascending mean W1-cycle order, with original input index breaking ties. Gray ends the recorded primary solve. ")


def read(path):
    return json.loads(Path(path).read_text())


class CurrentData:
    def __init__(self, root):
        self.root = root
        self.protocol = read(root / "protocol.json")
        self.summary = read(root / "summary.json")
        completion = read(root / "complete.json")
        replay_audit = read(root / "analysis/w1_trace_audit.json")
        assert completion["passed"] and replay_audit["passed"]
        assert sha256(root / "raw.jsonl") == completion["raw_sha256"]
        assert sha256(root / "summary.json") == completion["summary_sha256"]
        assert sha256(root / "analysis/w1_trace_replay.jsonl") == replay_audit["trace_sha256"]
        assert (self.summary["cases"], self.summary["repetitions"], self.summary["training_replicates"]) == (100, 3, 1)
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
        self.rows = {(r["method"], r["repeat"], r["case_id"]): r for r in raw}
        assert len(raw) == len(self.rows) == 2100
        assert set(self.rows) == {(m, rep, i) for m in self.protocol["pairs"] for rep in range(3) for i in range(100)}
        choices = read(root / "choices.json")["test"]
        for (m, rep, i), r in self.rows.items():
            o = r["outcome"]
            expected = choices[self.protocol["pairs"][m]["source"]][i]
            assert all(r[k] == expected[k] for k in ("input_id", "hierarchy_id", "params", "arm_index"))
            assert r["success"] and not o["failed"] and not o["fallback_used"]
            assert o["primary_attempt_count"] == 1 and o["primary_cycles"] == r["primary_cycles"]
            if m != "bandit_default":
                assert len(o["cycle_actions"]) == len(o["cycle_residuals"]) == r["primary_cycles"]
        self.replayed = {r["case_id"]: r["outcome"] for r in
                         (json.loads(line) for line in (root / "analysis/w1_trace_replay.jsonl").read_text().splitlines())}
        assert set(self.replayed) == set(range(100))
        for i, replay in self.replayed.items():
            for rep in range(3):
                original = self.rows["bandit_default", rep, i]["outcome"]
                assert replay["iterations"] == original["primary_cycles"]
                np.testing.assert_allclose(replay["residual_norm"], original["primary_residual_norm"], rtol=1e-10, atol=1e-14)
        self.order = case_order(self.values("bandit_default", "primary_cycles"))
        self.width = max(30, max(r["primary_cycles"] for r in raw))
        self.bootstrap = np.random.default_rng(self.protocol["seeds"]["bootstrap"]).integers(0, 100, (10000, 100))
        for m, means in self.summary["overall_mean_seconds"].items():
            for key, value in means.items():
                np.testing.assert_allclose(self.values(m, key).mean(), value, rtol=1e-12, atol=1e-14)

    def values(self, method, field):
        return np.array([[self.rows[method, rep, i][field] for i in range(100)] for rep in range(3)])

    def compare(self, target, reference, field):
        a, b = self.values(reference, field).mean(axis=0), self.values(target, field).mean(axis=0)
        bs = 100*(1-b[self.bootstrap].mean(axis=1)/a[self.bootstrap].mean(axis=1))
        return float(100*(1-b.sum()/a.sum())), np.quantile(bs, [.025, .975])

    def trace(self, method, rep, field):
        result = []
        for i in range(100):
            o = self.rows[method, rep, i]["outcome"]
            n = o["primary_cycles"]
            if method == "bandit_default":
                seq = np.ones(n) if field == "actions" else self.replayed[i]["cycle_residuals"]
            else:
                seq = o["cycle_"+field]
            assert len(seq) == n and np.isfinite(seq).all()
            if field == "actions":
                assert np.all((np.asarray(seq) >= 1) & (np.asarray(seq) <= 3))
                if method == "bandit_fixed_prior": np.testing.assert_array_equal(seq, np.full(n, 1.6))
                if method in ("bandit_periodic", "rl_setup_periodic"):
                    np.testing.assert_array_equal(seq, [(2.85, 1.1)[j % 2] for j in range(n)])
            else:
                assert np.all(np.asarray(seq) > 0)
            result.append(padded_trace(seq, self.width))
        return np.asarray(result)


def generate(root):
    inputs = [root/n for n in ("raw.jsonl", "summary.json", "complete.json", "protocol.json", "choices.json",
                               "analysis/w1_trace_replay.jsonl", "analysis/w1_trace_audit.json")]
    inputs.append(ROOT / "docs/theory/period_two_minimax_verification_20260928.json")
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    d = CurrentData(root)
    destination = root / "analysis/paper_figures"
    destination.mkdir(parents=True, exist_ok=True)
    apply_style()
    exporter = Exporter(destination, 600)
    exporter.atlas.infodict().update(Title="Module 06: current frozen-policy results",
                                   Subject="100 fresh diffusion 60 cubed inputs; one trained checkpoint set; three timings")
    all_pages = PdfPages(destination / "all_figures.pdf", metadata={"Title": "Module 06: all current-data figures"})
    comparisons = []

    def save(fig, name, title, caption, *, category="main", **kw):
        all_pages.savefig(fig, dpi=600, bbox_inches="tight", pad_inches=.06)
        exporter.save(fig, name, title, NOTE+caption, category=category, **kw)
        print(json.dumps({"figure": name}), flush=True)

    def point(ax, method, reference, field, y, color, marker="D"):
        gain, bounds = d.compare(method, reference, field)
        ax.errorbar(gain, y, xerr=[[gain-bounds[0]], [bounds[1]-gain]], fmt=marker,
                    color=color, ms=4, capsize=3, zorder=4)
        comparisons.append({"target": method, "reference": reference, "field": field,
                            "reduction_pct": gain, "paired_input_bootstrap_95_pct": bounds.tolist()})

    def heatmap(rep, residual=False):
        field = "residuals" if residual else "actions"
        cmap = plt.get_cmap("magma_r" if residual else "viridis").copy(); cmap.set_bad(BLANK)
        norm = LogNorm(1e-6, 10) if residual else Normalize(1, 3)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True, sharey=True, layout="constrained")
        for j, (ax, method) in enumerate(zip(axes.flat, METHODS)):
            im = trajectory_image(ax, d.trace(method, rep, field)[d.order], cmap=cmap, norm=norm)
            panel(ax, "abcd"[j], LABELS[method])
            if j >= 2: ax.set_xlabel("Test problem rank")
            if j % 2 == 0: ax.set_ylabel("AMG cycle")
        cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.025, shrink=.9, extend="min" if residual else "neither")
        cb.set_label(r"Relative residual $\|r_k\|/\|r_0\|$" if residual else r"Relaxation weight $w_k$")
        cb.set_ticks([1e-6, 1e-4, 1e-2, 1, 10] if residual else [1, 1.5, 2, 2.5, 3])
        fig.suptitle(f"Module 06 · {'Residual contraction' if residual else 'Executed relaxation weights'} · repetition {rep+1}\nFresh 60³ diffusion · Own adapted hierarchies")
        fig.text(.48, -.02, "Identical W1-difficulty order; gray = no further primary cycle recorded.", ha="center", fontsize=7)
        name = ("03_residual_heatmaps" if residual else "02_action_heatmaps") if rep == 0 else f"all_repetitions/{field}_repeat{rep+1}"
        save(fig, name, f"{'Residual' if residual else 'Action'} heatmaps — repetition {rep+1}", TRACE_NOTE,
             category="main" if rep == 0 else "timing-repeats", inputs={"methods": list(METHODS), "repeat_zero_based": rep})

    try:
        # 01: Same aggregate-reduction design; current statistical unit replaces
        # the unavailable six-training-seed sample standard deviation.
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True, layout="constrained")
        for j, (ax, field, title) in enumerate(zip(axes, ("native_total_sec", "inclusive_total_sec"),
                ("Native setup + solve", "Complete method, all overhead"))):
            for i, m in enumerate(METHODS):
                repeated = 100*(1-d.values(m, field).sum(axis=1)/d.values("bandit_default", field).sum(axis=1))
                ax.scatter(repeated, i+np.linspace(-.12,.12,3), s=16, facecolor="white", edgecolor=PALETTE[m], zorder=3)
                point(ax, m, "bandit_default", field, i, PALETTE[m])
            ax.axvline(0, color=".45", lw=.7); clean(ax)
            ax.set(yticks=range(4), yticklabels=[LABELS[m] for m in METHODS], ylim=(3.5,-.6), xlabel="Time reduction vs W1 (%)")
            panel(ax, "ab"[j], title)
        fig.suptitle("Module 06 · Four frozen methods · 100 fresh problems")
        fig.text(.5,-.025,"Open circles: timings; diamonds: pooled estimate; whiskers: paired input-bootstrap 95% CI.",
                 ha="center",fontsize=7)
        save(fig, "01_policy_savings", "Four-policy timing comparison",
             "Open points are the three timing repetitions, not independently trained seeds. Diamonds pool all three repetitions. "
             "Whiskers are paired input-bootstrap 95% intervals over 100 inputs, conditional on this checkpoint set. "
             "The left panel includes native setup+solve; the right also charges setup selection and controller overhead.")

        heatmap(0)
        heatmap(0, residual=True)

        # 04: Retain the six-panel RL atlas using actual available dimensions.
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), sharex=True, sharey=True, layout="constrained")
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(BLANK)
        for source_index, (source, _periodic, rl) in enumerate(CROSSED):
            for rep in range(3):
                ax = axes[source_index, rep]
                im = trajectory_image(ax, d.trace(rl, rep, "actions")[d.order], cmap=cmap, norm=Normalize(1,3))
                panel(ax, "abcdef"[source_index*3+rep], f"{source}\nRepetition {rep+1}")
                if source_index == 1: ax.set_xlabel("Test problem rank")
                if rep == 0: ax.set_ylabel("AMG cycle")
        cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.025, shrink=.9); cb.set_label("RL relaxation weight")
        fig.suptitle("Module 06 · Frozen RL trajectory atlas\nTwo hierarchy sources × three timing repetitions")
        save(fig, "04_rl_trajectory_atlas", "RL across hierarchy sources and repetitions",
             TRACE_NOTE+"All panels use the same frozen controller. Rows change the frozen hierarchy source; columns change timing repetition. "
             "This is not a six-training-seed atlas. Runtime observations can change actions while model parameters remain frozen.")

        # 05: Same four-panel periodic comparison, now the available 2x2 cross.
        fig, axes = plt.subplots(2,2,figsize=(7.2,4.65),sharex=True,sharey=True,layout="constrained")
        for i,(source,pm,rm) in enumerate(CROSSED):
            for j,(m,label) in enumerate(((pm,"Periodic (2.85, 1.10)"),(rm,"Frozen RL"))):
                ax=axes[i,j]
                im=trajectory_image(ax,d.trace(m,0,"actions")[d.order],cmap=cmap,norm=Normalize(1,3))
                panel(ax,"abcd"[2*i+j],f"{label}\n{source} setup")
                if i==1: ax.set_xlabel("Test problem rank")
                if j==0: ax.set_ylabel("AMG cycle")
        cb=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,shrink=.9);cb.set_label("Relaxation weight")
        fig.suptitle("Module 06 · Periodic/RL crossed comparison · repetition 1")
        fig.text(.48,-.02,"Same hierarchy within each row; identical sorted input columns.",ha="center",fontsize=7)
        save(fig,"05_periodic_comparison","Periodic and RL on both hierarchy sources",
             TRACE_NOTE+"Within each row the hierarchy, matrix, RHS and initial iterate are matched. "
             "Both own pairings and both crossed pairings use current Module 06 recorded data.")

        fig,ax=plt.subplots(figsize=(7.2,3.4),layout="constrained")
        refs=METHODS[:-1]
        for i,m in enumerate(refs):
            point(ax,"bandit_lstdq",m,"native_continuation_sec",i-.10,COLORS["rl"],marker="o")
            point(ax,"bandit_lstdq",m,"inclusive_total_sec",i+.10,"#444444",marker="D")
        ax.axvline(0,color=".45",lw=.8);clean(ax)
        ax.set(yticks=range(3),yticklabels=[LABELS[m] for m in refs],ylim=(2.6,-.6),xlabel="RL time reduction vs comparator (%)")
        ax.legend(handles=[Line2D([],[],color=COLORS["rl"],marker="o",label="Native solve + recovery"),
                           Line2D([],[],color="#444444",marker="D",label="Complete method")],loc="lower right")
        fig.suptitle("Module 06 · Direct paired advantage of frozen RL")
        save(fig,"06_rl_advantage","Direct RL advantages with and without full method costs",
             "Positive values favor RL. Blue circles compare native continuation, excluding initial setup and controller overhead; "
             "gray diamonds compare complete-method cost, including setup selection/construction and controller computation. "
             "Different own-pairing hierarchies mean this is a complete-policy comparison, not an isolated controller effect. "
             "Whiskers are paired input-bootstrap 95% intervals; there is one trained checkpoint set.")

        theory=read(ROOT/"docs/theory/period_two_minimax_verification_20260928.json")
        assert theory["passed"] and theory["grid_minimizers_ordered"][0]==["2.85","1.10"]
        fig,ax=plt.subplots(figsize=(7.2,3.4),layout="constrained")
        xx=np.linspace(2.6,3,401)
        eta=lambda x:float(extrema(mp.mpf(str(x)),mp.mpf("1.10"))[0])
        ax.plot(xx,[eta(x) for x in xx],color="#455a64",label=r"Slice $b=1.10$")
        for x,color,label in ((2.85,COLORS["periodic"],"Grid pair (2.85, 1.10)"),(2.90,COLORS["rl"],"Rounded pair (2.90, 1.10)")):
            ax.scatter([x],[eta(x)],color=color,s=25,label=label,zorder=3)
        ax.axhline(.04,color=".55",ls="--",lw=.8,label="Continuous two-coefficient lower bound")
        ax.set(xlabel="High coefficient a (low coefficient fixed at 1.10)",ylabel=r"$\eta(a,1.10)=\max_{0\leq t\leq1}t(1-at)^2(1-1.10t)^2$")
        ax.legend(frameon=False,loc="upper left");clean(ax,"both")
        fig.suptitle("Module 06 · Period-two weighted smoothing objective")
        save(fig,"07_period_two_objective","Why the prescribed grid pair is (2.85,1.10)",
             "This updates the old anchored-(2.6,1) objective figure to the current free period-two criterion. "
             "The displayed curve is the b=1.10 slice, evaluated at analytic stationary points and endpoints. "
             "The saved 80-digit verification checks all 861 unordered grid pairs and selects (2.85,1.10). "
             "The dashed 1/25 is the global continuous two-coefficient lower bound, not the optimum on the b=1.10 slice. "
             "This theory objective does not predict an equal percentage change in solver runtime.")

        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,layout="constrained")
        for j,(ax,m) in enumerate(zip(axes,("bandit_fixed_prior","bandit_periodic"))):
            for field,color,label in (("native_continuation_sec",COLORS["rl"],"Native solve + recovery"),("inclusive_total_sec","#444444","Complete method")):
                g=100*(1-d.values("bandit_lstdq",field).sum(axis=1)/d.values(m,field).sum(axis=1))
                ax.plot([1,2,3],g,"o-",color=color,ms=3,label=label)
            ax.axhline(0,color=".5",lw=.7);clean(ax,"both")
            ax.set(xticks=[1,2,3],xlabel="Timing repetition",ylabel="RL time reduction (%)")
            panel(ax,"ab"[j],f"RL vs {LABELS[m]}")
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,-.08),ncol=2,frameon=False)
        fig.suptitle("Module 06 · Timing repeatability on all 100 problems")
        save(fig,"08_timing_repetitions","Three complete timing repetitions",
             "Each point sums one full 100-input timing repetition, with all setup choices and learned parameters frozen. "
             "The lines compare native continuation and complete-method costs. These are repeated timings, not independently trained seeds.")

        for rep in (1,2):
            heatmap(rep)
            heatmap(rep,residual=True)
    finally:
        exporter.close()
        all_pages.close()

    exporter.entries.sort(key=lambda e:(e["category"]!="main",e["name"]))
    (destination/"figure_catalog.json").write_text(json.dumps(exporter.entries,indent=2)+"\n")
    arrays={"case_order_0_based":d.order}
    for m in (*METHODS,"periodic_setup_rl","rl_setup_periodic"):
        for rep in range(3):
            for field in ("actions","residuals"):
                arrays[f"{m}__repeat{rep+1}__{field}"]=d.trace(m,rep,field)
    np.savez_compressed(destination/"figure_arrays.npz",**arrays)
    data={"methods":list(METHODS),"labels":LABELS,"training_replicates":1,"timing_repetitions":3,
          "case_order_0_based":d.order.tolist(),"reported_mean_seconds":d.summary["overall_mean_seconds"],
          "comparisons":comparisons,"costs":{m:{field:d.values(m,field).tolist() for field in
          ("native_total_sec","inclusive_total_sec","native_continuation_sec","inclusive_continuation_sec","primary_cycles")}
          for m in d.protocol["pairs"]}}
    (destination/"figure_data.json").write_text(json.dumps(data,indent=2)+"\n")
    scripts=[Path(__file__), ROOT/"experiments/paper_final/plot_05_policy.py",
             ROOT/"experiments/paper_final/plot_05_policy_anchored.py",
             ROOT/"experiments/paper_final/collect_06_w1_traces.py",
             *sorted((ROOT/"experiments/paper_final/module05_figures").glob("*.py"))]
    for p in scripts:
        target=destination/"reproduction"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    provenance={"result_directory":str(root),"source_hashes":source_hashes,
        "plot_code_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in scripts},
        "figure_count":len(exporter.entries),"main_figure_count":8,"timing_variant_count":4,
        "dpi":600,"all_eight_run04_figure_roles_retained":True,
        "adaptations":{"01":"Current paired-input intervals; three timing repeats, one checkpoint set",
                       "02_03":"W1, fixed1.60, periodic(2.85,1.10), RL; own adapted setups",
                       "04":"Six panels are two actual hierarchy sources by three timings, not six training seeds",
                       "05":"Current recorded periodic/RL 2x2 cross",
                       "06":"Current native-continuation and complete-method paired comparisons",
                       "07":"Current free period-two objective replaces anchored objective",
                       "08":"Actual three timing repetitions; no invented training seeds",
                       "supplement":"Repetitions 2 and 3, actions and residuals"},
        "w1_trace_replay_audit":read(root/"analysis/w1_trace_audit.json"),
        "reused_system":"module05_figures style, Exporter, trajectory_image, case_order, padded_trace; original gallery writer",
        "formal_timing_replaced":False,"plotting_solver_executions":0}
    write_gallery(destination,exporter.entries,provenance)
    page=(destination/"index.html").read_text().replace("Module 05","Module 06").replace("all-seeds","timing-repeats")
    before,body=page.split("<header>",1);_old,after=body.split("</header>",1)
    header='''<header><h1>Module 06 · Current-data figure gallery</h1>
<p>100 fresh diffusion 60³ inputs · one trained checkpoint set · three timing repetitions.</p>
<p>The eight Run 04 figure roles use current Module 06 data and the original plotting system. Main heatmaps contain W1, Fixed 1.60, Periodic (2.85, 1.10), and frozen RL, each on its own adapted setup.</p>
<p class="note">The six-panel RL atlas shows two hierarchy sources × three timing repetitions. These are not six training seeds. W1 residuals were supplemented by verified diagnostic replays; formal timing results are unchanged.</p>
<p><a href="all_figures.pdf">All 12 figures (PDF)</a> · <a href="main_figure_atlas.pdf">Eight main figures (PDF)</a> · <a href="figure_pack.zip">Complete figure pack</a> · <a href="CAPTIONS.md">Captions</a> · <a href="provenance.json">Provenance</a></p>
<nav><button class="active" data-filter="main">Main figures</button><button data-filter="timing-repeats">Timing variants</button><button data-filter="all">Everything</button></nav></header>'''
    (destination/"index.html").write_text(before+header+after)
    captions=(destination/"CAPTIONS.md").read_text().replace("# Module 05 figure captions","# Module 06 current-data figure captions",1)
    (destination/"CAPTIONS.md").write_text(captions)
    (destination/"README.md").write_text("# Module 06 current figure set\n\n"+NOTE+"\n\n"+
        "All eight Run 04 main figure roles are updated. Four supplementary panels retain the actual second and third timing repetitions; this run has one trained checkpoint set.\n\n"+
        "[Open gallery](index.html) · [All 12 figures PDF](all_figures.pdf) · [Eight main figures PDF](main_figure_atlas.pdf) · [Captions](CAPTIONS.md) · [Complete figure pack](figure_pack.zip)\n")
    for path,digest in source_hashes.items():
        assert sha256(ROOT/path)==digest,path
    with zipfile.ZipFile(destination/"figure_pack.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(destination.rglob("*")):
            if p.is_file() and p.name!="figure_pack.zip": archive.write(p,p.relative_to(destination))
    print(json.dumps({"output":str(destination),"figures":len(exporter.entries),"main":8,"timing_variants":4,
                      "raw_data_unchanged":True}),flush=True)
    return provenance


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    generate(parser.parse_args().output.resolve())
