"""Package Run 04 measurements, retained fixed-grid evidence, theory and figures."""
from __future__ import annotations

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT
import argparse
from contextlib import ExitStack
import io
import json
from pathlib import Path
import shutil
import statistics
import zipfile
from experiments.archive.paper_development.verify_05_policy_anchored import verify,read,digest

ROOT=REPOSITORY_ROOT
RUN=ROOT/"results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases"
PRIOR=ROOT/"results/paper_final/05_policy/releases/module05_final_diffusion60_six_seeds_20260927"
DEST=ROOT/"results/paper_final/05_policy/releases/module05_run04_anchored26_diffusion60_six_seeds_20260927"


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+"\n")


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def pdf_summary(dest,data):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    from pypdf import PdfReader,PdfWriter
    width=A4[0]-80;story=[];buf=io.BytesIO()
    body=ParagraphStyle("body",fontName="Helvetica",fontSize=9,leading=13,spaceAfter=8)
    head=ParagraphStyle("head",parent=body,fontName="Helvetica-Bold",fontSize=12,leading=17,spaceBefore=9,textColor=colors.HexColor("#173c52"))
    title=ParagraphStyle("title",parent=head,fontSize=22,leading=28,spaceAfter=14)
    cell=ParagraphStyle("cell",parent=body,fontSize=7.2,leading=10,alignment=1)
    def p(text,style=body):story.append(Paragraph(text,style))
    labels={"reference":"Matched<br/>w=1","fixed":"Global<br/>fixed*","oracle":"Per-instance<br/>fixed*","periodic":"Anchored<br/>(2.6,1)","periodic13":"Periodic<br/>(1,3)","rl":"Frozen<br/>RL"}
    def table(direct=False,inclusive=False):
        methods=data["methods"][:-1] if direct else data["methods"]
        field="inclusive" if inclusive else "native"
        gains={m:[100*(1-sum(data["costs"]["rl" if direct else m][field][i])/sum(data["costs"][m if direct else "reference"][field][i])) for i in range(6)] for m in methods}
        rows=[[Paragraph("Seed",cell)]+[Paragraph(labels[m],cell) for m in methods]]
        for i in range(6):rows.append([str(i+1)]+[f"{gains[m][i]:.2f}" for m in methods])
        rows.append([Paragraph("Mean<br/>+/- SD",cell)]+[Paragraph(f"{statistics.mean(gains[m]):.2f}<br/>+/- {statistics.stdev(gains[m]):.2f}",cell) for m in methods])
        t=Table(rows,colWidths=[44]+[(width-44)/len(methods)]*len(methods),rowHeights=[32]+[21]*6+[34])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e1edf3")),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,1),(-1,-1),8.5),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white,colors.HexColor("#f5f7f9")]),("LINEABOVE",(0,-1),(-1,-1),.4,colors.lightgrey)]));story.append(t);story.append(Spacer(1,10))
    comparisons={(r["comparator"],r["cost"]):r for r in data["rl_comparisons"]}
    r=comparisons["reference","native"];a=comparisons["periodic","native"];ai=comparisons["periodic","inclusive"]
    p("Module 05 - Run 04",title)
    p("Prescribed anchored relaxation (2.6,1) | Diffusion 60<super>3</super>",head)
    p("Six unchanged Run 03 setup/controller checkpoints, including refreshed seed 4; the same 100 test problems and three fresh timing repetitions. Fixed weights and hierarchy choices are retained. The only policy replacement is (2.5,1) to (2.6,1).")
    p(f"<b>Measured result:</b> frozen RL reduces native continuation time by <b>{r['mean']:.2f}% +/- {r['sd']:.2f} percentage points</b> versus matched w=1. Its direct reduction versus the anchored schedule is <b>{a['mean']:.2f}% +/- {a['sd']:.2f}</b>, with {a['winning_seeds']}/6 winning checkpoints.")
    p("Table 1. Native time reduction versus matched w=1 (%)",head);table()
    p("Costs are averaged over three repetitions within case, summed across 100 cases, then compared as 100 x (1 - method/reference). Mean and sample SD summarize six checkpoint reductions. Native continuation includes primary solves and all recovery setup/solve; common initial setup and controller overhead are excluded.")
    p("Theoretical basis",head)
    p("For normalized SPD l1-Jacobi, the anchored objective max t(1-t)<super>2</super>(1-at)<super>2</super> has the unique minimizer (3+sqrt(5))/2. Analytic evaluation on the 0.05 action grid selects 2.6. The accompanying reviewed note proves this and an exact two-grid bound, and checks the implemented smoother. The complete multilevel runtime is assessed empirically.")
    story.append(PageBreak())
    p("Direct comparisons and audit",title)
    p("Table 2. RL native time reduction versus each comparator (%)",head);table(direct=True)
    p(f"<b>Controller-inclusive comparison:</b> RL's mean direct reduction versus Anchored (2.6,1) is {ai['mean']:.2f}% +/- {ai['sd']:.2f} percentage points, with {ai['winning_seeds']}/6 winning checkpoints. Positive values favor RL. The complete inclusive tables remain in the archive.")
    au=data["audit"]
    p(f"<b>Reliability:</b> {au['trials']:,} distinct final executions cover 10,800 policy-role observations; identical fixed choices share one execution. There are {au['failures']} unrecovered failures and {au['recovered_trials']} recovered executions. All recovery costs remain charged. The 29,520 retained fixed-grid records reconstruct all accepted fixed choices.")
    p("Interpretation",head)
    p("The schedule was prescribed before Run 04 timing, with the high weight first and phase reset per primary solve. The coefficient theorem does not select phase or imply multilevel runtime optimality. The retained (1,3) comparator remains in aggregate figures. Global and per-instance fixed baselines are successful best-observed 41-grid diagnostics followed by independent retiming.")
    p("The accepted checkpoints and cases were already examined in earlier work, and seed 4 was retrained after inspecting its results. This is a prescribed-policy follow-up, not a fresh holdout or new independent seed replication. The 100 cases are shared across checkpoints. Cross-session changes versus Run 03 do not isolate a causal speedup from changing 2.5 to 2.6.")
    selection=read(dest/"figures/best_seed_selection.json")
    p(f"Main trajectory examples use seed {selection['seed']}, explicitly selected as the best observed seed ({selection['reduction_pct']:.2f}% native reduction versus w=1). All six seeds remain in aggregate and supplementary figures. The following pages contain eight current main figures, including the anchored smoothing objective.")
    def footer(c,doc):
        c.setFont("Helvetica",7);c.setFillColor(colors.HexColor("#637582"));c.drawString(40,24,"Module 05 Run 04 | Native and controller-inclusive costs retained");c.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(buf,pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title="Module 05 Run 04 - anchored (2.6,1)").build(story,onFirstPage=footer,onLaterPages=footer)
    buf.seek(0);reader=PdfReader(buf)
    if len(reader.pages)!=2:raise AssertionError(f"Expected two summary pages, got {len(reader.pages)}")
    writer=PdfWriter();writer.append(reader);writer.append(dest/"figures/main_figure_atlas.pdf")
    writer.add_metadata({"/Title":"Module 05 Run 04 - prescribed (2.6,1)"})
    with (dest/"RUN04_RESULTS.pdf").open("wb") as f:writer.write(f)


def build(dest=DEST):
    if dest.exists():raise FileExistsError(f"Use a new release directory: {dest}")
    assert read(RUN/"complete.json")["status"]=="complete"
    from experiments.archive.paper_development.verify_05_policy_final import verify as verify_prior
    verify_prior(PRIOR)
    dest.mkdir(parents=True)
    summary=read(RUN/"analysis/summary.json");protocol=read(RUN/"protocol.json")
    protocol["scan_extra_repeat_case_ids"]=read(PRIOR/"data/protocol.json")["scan_extra_repeat_case_ids"]
    protocol["retained_fixed_scan_origin"]=str(PRIOR)
    for k in ("comparison_costs","run03_comparison","comparison_run"):summary.pop(k,None)
    summary["protocol"]=protocol
    dump(dest/"data/protocol.json",protocol);dump(dest/"data/summary.json",summary)
    dump(dest/"data/inputs.json",{"test":read(RUN/"inputs.json")["test"]})
    copy(RUN/"selections.json",dest/"data/selections.json")
    copy(RUN/"analysis/trace_arrays.npz",dest/"data/trace_arrays.npz")
    jobs=read(RUN/"jobs_test.json")
    with ExitStack() as stack:
        handles={}
        for seed in range(1,7):
            d=dest/f"data/seeds/seed_{seed}";d.mkdir(parents=True)
            dump(d/"jobs.json",[j for j in jobs if j["seed"]==seed])
            copy(PRIOR/f"data/seeds/seed_{seed}/fixed_weight_scan.jsonl",d/"fixed_weight_scan.jsonl")
            handles[seed]=stack.enter_context((d/"evaluation.jsonl").open("w"))
        for name,sha in read(RUN/"raw/test/complete.json")["raw_sha256"].items():
            assert digest(RUN/name)==sha
            with (RUN/name).open() as f:
                for line in f:handles[json.loads(line)["seed"]].write(line)
    summary["audit"]["raw_sha256"]={str(p.relative_to(dest)):digest(p)
        for p in sorted((dest/"data/seeds").glob("*/evaluation.jsonl"))}
    summary["scan_audit"]={"trials":29520,"per_seed_trials":4920,"retained_from_prior_selection":True,
        "raw_sha256":{str(p.relative_to(dest)):digest(p)
            for p in sorted((dest/"data/seeds").glob("*/fixed_weight_scan.jsonl"))}}
    dump(dest/"data/summary.json",summary)
    hashes={}
    for name,sha in read(RUN/"prepared.json")["checkpoint_sha256"].items():
        assert digest(RUN/name)==sha;copy(RUN/name,dest/name);hashes[name]=sha
    dump(dest/"provenance/checkpoint_hashes.json",hashes)
    for name in ("environment.json","preflight.json","prepared.json","complete.json","parent_audit.json","source_manifest.json"):
        copy(RUN/name,dest/"provenance"/name)
    copy(RUN/"raw/test/complete.json",dest/"provenance/raw_completion.json")
    dump(dest/"provenance/retained_scans.json",{str(p.relative_to(PRIOR)):digest(p) for p in sorted((PRIOR/"data/seeds").glob("*/fixed_weight_scan.jsonl"))})
    shutil.copytree(RUN/"provenance/source",dest/"reproduction/source")
    shutil.copytree(RUN/"theory",dest/"theory")
    figures=RUN/"analysis/paper_figures_best_seed";entries=read(figures/"figure_catalog.json")
    for e in entries:
        for f in e["files"].values():copy(figures/f,dest/"figures"/f)
    for name in ("figure_catalog.json","main_figure_atlas.pdf","CAPTIONS.md","best_seed_selection.json","provenance.json"):
        copy(figures/name,dest/"figures"/name)
    for name in ("per_seed.csv","per_repeat.csv","repetition_diagnostics.csv"):
        copy(RUN/"analysis"/name,dest/"tables"/name)
    from experiments.archive.paper_development.package_05_policy_final import rows_for_table,write_csv
    for name,direct,inclusive in (("policy_reductions_no_overhead",False,False),("rl_advantage_no_overhead",True,False),("policy_reductions_with_overhead",False,True),("rl_advantage_with_overhead",True,True)):
        write_csv(dest/"tables"/(name+".csv"),rows_for_table(summary,direct=direct,inclusive=inclusive))
    copy(RUN/"REPORT.md",dest/"REPORT.md")
    (dest/"README.md").write_text("# Module 05 Run 04\n\nStart with RUN04_RESULTS.pdf and theory/anchored_schedule_review.pdf.\n\nThis archive contains only Run 04 final evaluations plus the retained fixed-grid evidence supporting its unchanged baselines. Run 03 final evaluations are not included.\n\nRun `python3 reproduction/verify_archive.py` from the extracted archive to verify hashes, coverage, schedule execution, every cost array and all fixed-grid choices. The verifier uses only the standard library.\n")
    for name in ("package_05_policy_anchored.py","build_anchored_note_pdf.py"):
        copy(ROOT/"experiments/paper_final"/name,dest/"reproduction/source/experiments/paper_final"/name)
    copy(ROOT/"experiments/archive/paper_development/verify_05_policy_anchored.py",dest/"reproduction/verify_archive.py")
    pdf_summary(dest,summary)
    audit=verify(dest,check_manifest=False)
    audit.pop("manifest_verified",None)
    audit["integrity_validation"]="Every file hash is verified after manifest creation and before ZIP publication."
    dump(dest/"COMPLETION.json",audit)
    files=sorted(p for p in dest.rglob("*") if p.is_file())
    (dest/"MANIFEST.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(dest)}\n" for p in files))
    verify(dest)
    zip_path=dest.with_suffix(".zip")
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file():z.write(p,Path(dest.name)/p.relative_to(dest))
    print(json.dumps({"archive":str(zip_path),"bytes":zip_path.stat().st_size,"audit":audit},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEST)
    build(p.parse_args().output)
