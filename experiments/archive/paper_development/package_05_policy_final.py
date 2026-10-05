"""Create the accepted, current-only Module 05 completion archive.

Requires reportlab and pypdf for the summary PDF. Packaging and verification
do not execute solvers, train controllers, or alter the frozen experiment.
"""
from __future__ import annotations

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import zipfile

from experiments.archive.paper_development.verify_05_policy_final import verify, digest, read

ROOT = REPOSITORY_ROOT
RUN = ROOT / "results/paper_final/05_policy/20260927_run03_new_seed4_joint_6seeds_100cases"
SCAN_ORIGIN = ROOT / "results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases"
PACKAGE_ID = "module05_final_diffusion60_six_seeds_20260927"
DEFAULT = ROOT / "results/paper_final/05_policy/releases" / PACKAGE_ID
METHODS = ("reference", "fixed", "oracle", "periodic", "periodic13", "rl")
LABELS = {"reference": "Default w=1", "fixed": "Global best fixed*",
          "oracle": "Per-instance best fixed*", "periodic": "Tuned periodic (2.5,1)",
          "periodic13": "Periodic (1,3)", "rl": "Frozen RL"}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_for_table(summary, *, direct=False, inclusive=False):
    field = "inclusive" if inclusive else "native"
    methods = METHODS[:-1] if direct else METHODS
    rows = []
    for i, seed in enumerate(summary["seeds"]):
        row = {"seed": seed}
        for method in methods:
            target, reference = ("rl", method) if direct else (method, "reference")
            row[method] = 100 * (1 - sum(summary["costs"][target][field][i]) /
                                sum(summary["costs"][reference][field][i]))
        rows.append(row)
    for label, func in (("Mean", statistics.mean), ("SD", statistics.stdev)):
        rows.append({"seed": label, **{m: func(r[m] for r in rows[:6]) for m in methods}})
    return rows


def markdown_table(rows):
    methods = [m for m in rows[0] if m != "seed"]
    lines = ["| Seed | " + " | ".join(LABELS[m] for m in methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for row in rows[:6]:
        lines.append(f"| {row['seed']} | " + " | ".join(f"{row[m]:.2f}%" for m in methods) + " |")
    lines.append("| Mean +/- SD | " + " | ".join(f"{rows[6][m]:.2f} +/- {rows[7][m]:.2f}" for m in methods) + " |")
    return "\n".join(lines)


def latex_table(rows, caption, label):
    methods = [m for m in rows[0] if m != "seed"]
    lines = [r"\begin{table}", r"\centering", r"\small", "\\caption{" + caption + "}",
             "\\label{" + label + "}", "\\begin{tabular}{l" + "r" * len(methods) + "}",
             r"\hline", "Seed & " + " & ".join(LABELS[m] for m in methods) + r" \\", r"\hline"]
    for row in rows[:6]:
        lines.append(str(row["seed"]) + " & " + " & ".join(f"{row[m]:.2f}" for m in methods) + r" \\")
    lines += [r"\hline", r"Mean $\pm$ SD & " + " & ".join(
        f"${rows[6][m]:.2f} \\pm {rows[7][m]:.2f}$" for m in methods) + r" \\",
        r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def summary_pdf(destination, summary, tables, figure_entries):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from pypdf import PdfReader, PdfWriter

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=39, leftMargin=39,
                            topMargin=40, bottomMargin=40, title="Module 05 - final completion record")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyFinal", fontName="Helvetica", fontSize=9.1, leading=13, spaceAfter=8))
    styles.add(ParagraphStyle("SmallFinal", fontName="Helvetica", fontSize=7.8, leading=10.8, spaceAfter=7))
    styles.add(ParagraphStyle("TitleFinal", fontName="Helvetica-Bold", fontSize=23, leading=28, spaceAfter=8,
                              textColor=colors.HexColor("#143951")))
    styles.add(ParagraphStyle("SubFinal", fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=14,
                              textColor=colors.HexColor("#455766")))
    styles.add(ParagraphStyle("SectionFinal", fontName="Helvetica-Bold", fontSize=11.5, leading=15, spaceBefore=9,
                              spaceAfter=7, textColor=colors.HexColor("#143951")))
    styles.add(ParagraphStyle("CellFinal", fontName="Helvetica", fontSize=7.3, leading=9, alignment=TA_CENTER))
    styles.add(ParagraphStyle("HeadFinal", fontName="Helvetica-Bold", fontSize=7.4, leading=9.5,
                              alignment=TA_CENTER, textColor=colors.white))
    story = []
    def p(text, style="BodyFinal"):
        story.append(Paragraph(text, styles[style]))
    def table(rows):
        methods = [m for m in rows[0] if m != "seed"]
        short = {"reference": "Default<br/>w=1", "fixed": "Global<br/>fixed*", "oracle": "Per-instance<br/>fixed*",
                 "periodic": "Tuned periodic<br/>(2.5,1)", "periodic13": "Periodic<br/>(1,3)", "rl": "Frozen RL"}
        cells = [[Paragraph("Seed", styles["HeadFinal"])] +
                 [Paragraph(short[m], styles["HeadFinal"]) for m in methods]]
        for row in rows[:6]:
            cells.append([str(row["seed"])] + [f"{row[m]:.2f}" for m in methods])
        cells.append([Paragraph("Mean<br/>+/- SD", styles["CellFinal"])] +
                     [Paragraph(f"{rows[6][m]:.2f}<br/>+/- {rows[7][m]:.2f}", styles["CellFinal"]) for m in methods])
        width = A4[0] - 78
        t = Table(cells, colWidths=[48] + [(width - 48) / len(methods)] * len(methods), repeatRows=1,
                  rowHeights=[32] + [23] * 6 + [34])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#143951")),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 1), (-1, -1), 8.2),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f0f4f7")]),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e7eff4")),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#143951")),
            ("LINEABOVE", (0, -1), (-1, -1), .5, colors.HexColor("#94a8b7"))]))
        story.extend([t, Spacer(1, 8)])

    p("Module 05 - Complete", "TitleFinal")
    p("Final frozen-policy evaluation | Diffusion 60<super>3</super> | 27 September 2026", "SubFinal")
    p("<b>Accepted data:</b> six checkpoints, 100 shared test problems, Joint hierarchies, and three fresh timing repetitions per case. "
      "Seed 4 uses its retrained Module 04 setup selector and RL controller; seeds 1, 2, 3, 5 and 6 retain their original checkpoints.")
    p("<b>Main finding:</b> frozen RL reduces native continuation time by <b>41.50% +/- 1.52 percentage points</b> versus matched w=1. "
      "The six-seed range is <b>40.24%-43.83%</b>. RL has lower cumulative native cost than every tested comparator on all six checkpoints.")
    p("Table 1. Time reduction versus matched w=1 (%)", "SectionFinal")
    table(tables["policy_reductions_no_overhead"])
    p("Primary metric: native solve plus recovery setup/solve, with controller overhead and common initial setup excluded. "
      "For each seed, average all three repetitions within each case, sum the 100 case costs, then compute 100 x (1 - method/reference). "
      "The final row is the mean and sample SD across six checkpoint reductions; it is not a confidence interval.", "SmallFinal")
    p("Baseline definitions", "SectionFinal")
    p("<b>Global fixed*</b> uses one weight for all 100 problems within each checkpoint. <b>Per-instance fixed*</b> uses one constant weight per problem. "
      "Both were selected from the successful 41-weight scan (1.00 to 3.00 in steps of 0.05), then independently retimed here. "
      "They are best-observed finite-grid diagnostics, not continuous optima. Global weights are 1.55 for seed 1 and 1.60 for seeds 2-6.")
    p("The development-selected periodic schedule repeats (2.5,1). The prescribed periodic schedule repeats (1,3). "
      "Neither is claimed globally optimal among all schedules. RL and setup selectors remain frozen throughout evaluation.")

    story.append(PageBreak())
    p("Findings, reliability and scope", "TitleFinal")
    p("Table 2. RL time reduction versus each comparator (%)", "SectionFinal")
    table(tables["rl_advantage_no_overhead"])
    p("Each column uses its named comparator as the denominator. These are direct relative time reductions, not differences between the percentages in Table 1. "
      "Positive values favor RL. The mean native gains are 11.36% over global fixed, 9.31% over per-instance fixed, "
      "3.58% over tuned (2.5,1), and 11.10% over periodic (1,3).", "SmallFinal")
    p("Completion and reliability", "SectionFinal")
    p("The final evaluation contains <b>10,260 distinct executions</b> representing 6 x 100 x 3 x 6 = 10,800 policy-role observations; "
      "identical fixed choices share an execution. All completed successfully after the prescribed recovery, with <b>zero unrecovered failures</b>. "
      "The 15 recovered executions belong to periodic (1,3), across five seed/problem pairs. Recovery costs remain charged. "
      "The archive retains the 29,520 fixed-grid search executions supporting the accepted six checkpoints.")
    p("Interpretation for the paper", "SectionFinal")
    p("The improvement is consistent across the retained checkpoints on these shared inputs. This is a final accepted checkpoint set after inspecting "
      "earlier outcomes, including a retrained seed 4; it is not a new prespecified six-seed replication. Seed 1 also participated in earlier development. "
      "The 100 test inputs were fresh relative to training and are shared across checkpoints, so this is not 600 independent test matrices.")
    p("These comparisons establish frozen-policy utility in the stated native-cost metric. They do not isolate the necessity of within-solve feedback "
      "or establish a globally optimal dynamic policy. Controller-inclusive results are retained separately: RL reduces time versus w=1 by "
      "39.14% on average, but its mean direct gain over tuned (2.5,1) is -0.29%, so an overhead-inclusive advantage over that schedule is not established.")
    p("Figures and archive contents", "SectionFinal")
    p("The following seven pages show current policy savings, action heatmaps, residual heatmaps, all six RL checkpoints, periodic comparisons, "
      "direct RL advantages, and timing repeatability. Main trajectory examples use seed 2, explicitly selected as best observed (43.83% native reduction). "
      "All six seeds remain in aggregate figures and supplementary heatmaps. The archive includes editable figures, CSV/LaTeX tables, raw records, "
      "checkpoint copies, environment/source provenance, and an independent verification script.", "SmallFinal")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e1e8"))
        canvas.line(39, 31, A4[0] - 39, 31)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#637582"))
        canvas.drawString(39, 21, "Module 05 | Final accepted six-checkpoint set | No-overhead results first")
        canvas.drawRightString(A4[0] - 39, 21, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    summary_reader = PdfReader(buf)
    if len(summary_reader.pages) != 2:
        raise AssertionError(f"Brief summary should occupy two pages, got {len(summary_reader.pages)}")
    writer = PdfWriter()
    writer.append(summary_reader)
    writer.add_outline_item("Completion summary and tables", 0)
    for entry in figure_entries:
        if entry["category"] == "main":
            page = len(writer.pages)
            writer.append(destination / "figures" / entry["files"]["pdf"])
            writer.add_outline_item(entry["title"], page)
    writer.add_metadata({"/Title": "Module 05 - final six-seed completion record",
                         "/Subject": "Diffusion 60 cubed; final accepted checkpoint set; no-overhead results first"})
    with (destination / "MODULE05_FINAL.pdf").open("wb") as handle:
        writer.write(handle)


def build(destination):
    destination = Path(destination)
    if destination.exists() and not (destination / ".package_id").exists():
        raise FileExistsError(f"Refusing an unrelated existing directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".package_id").write_text(PACKAGE_ID + "\n")
    assert read(RUN / "complete.json")["status"] == "complete"
    summary = read(RUN / "analysis/summary.json")
    original_protocol = read(RUN / "protocol.json")
    jobs = read(RUN / "jobs_test.json")
    source_records = []
    with ExitStack() as stack:
        handles = {}
        for seed in range(1, 7):
            directory = destination / f"data/seeds/seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for kind in ("evaluation", "fixed_weight_scan"):
                handles[seed, kind] = stack.enter_context((directory / f"{kind}.jsonl").open("w"))
            dump(directory / "jobs.json", [j for j in jobs if j["seed"] == seed])
        phases = [(RUN, "test", "evaluation"), (RUN, "scan", "fixed_weight_scan"),
                  (SCAN_ORIGIN, "test", "fixed_weight_scan"), (SCAN_ORIGIN, "repetitions", "fixed_weight_scan")]
        for source, phase, kind in phases:
            marker = read(source / "raw" / phase / "complete.json")
            for name, sha in sorted(marker["raw_sha256"].items()):
                path = source / name
                assert digest(path) == sha, f"Source raw hash changed: {path}"
                kept = Counter()
                with path.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        if source == SCAN_ORIGIN and not (row["seed"] != 4 and row["source"] == "bandit_lstdq" and row["kind"] == "fixed"):
                            continue
                        handles[row["seed"], kind].write(line)
                        kept[row["seed"]] += 1
                source_records.append({"source_file": str(path.relative_to(ROOT)), "sha256": sha,
                    "destination_kind": kind, "kept_rows_by_seed": dict(kept),
                    "filter": "Retained Joint fixed-weight searches for seeds 1,2,3,5,6" if source == SCAN_ORIGIN else "All current rows"})
    print("Exported final evaluations and retained fixed-grid searches", flush=True)

    keep_protocol = ("module", "problem", "training_seeds", "sources", "test_cases", "repetitions_per_case", "methods",
        "main_heatmap_methods", "profile", "tolerance", "max_cycles", "max_workers", "input_distribution", "input_overlap_audit",
        "hierarchy_matching", "recovery", "candidate_selection", "seeds", "scan_extra_repeat_case_ids", "timing", "source_git_commit")
    protocol = {k: original_protocol[k] for k in keep_protocol}
    protocol.update(status="complete", accepted_measurement_run=3,
        primary_metric="Native solve plus recovery setup/solve; common initial setup and controller overhead excluded",
        statistic="Mean of the six seed-level ratio-of-sums reductions; sample SD across six checkpoints",
        checkpoint_set={"retrained_module04_setup_and_rl": [4], "original_module04_setup_and_rl": [1, 2, 3, 5, 6]},
        selection_disclosure="Final accepted set following inspection of earlier outcomes; seed 4 was retrained. Seed 1 was used in earlier development.",
        fixed_selection="Successful best-observed 41-weight scan on the same 100 test cases; each choice frozen before three new timings. Search data included per seed.",
        periodic_selection="Previously development-selected (2.5,1); prescribed (1,3). No global periodic optimum claim.",
        weights=[round(1 + .05 * i, 2) for i in range(41)],
        policies=original_protocol["policies"],
        scan_execution_count=29520,
        stopping_claim="Frozen policy comparison only; feedback-necessity and conventional-smoother claims require separate evidence")
    dump(destination / "data/protocol.json", protocol)
    dump(destination / "data/inputs.json", {"test": read(RUN / "inputs.json")["test"]})
    selections = {"case_order_0_based": summary["case_order_0_based"], "chosen_policies": {},
                  "selection_frozen_before_final_retiming": True}
    selected_rows = []
    for seed in range(1, 7):
        sj = sorted((j for j in jobs if j["seed"] == seed and j["repeat"] == 0), key=lambda j: j["case_id"])
        for method in METHODS:
            choices = {str(j["case_id"]): j["method_policies"][method] for j in sj}
            selections["chosen_policies"][f"{seed}/bandit_lstdq/{method}"] = choices if method == "oracle" else next(iter(set(choices.values())))
            if method != "oracle":
                assert len(set(choices.values())) == 1
        for job in sj:
            selected_rows.append({"seed": seed, "case_id": job["case_id"],
                "global_fixed_weight": float(job["method_policies"]["fixed"].split("_")[1]),
                "per_instance_fixed_weight": float(job["method_policies"]["oracle"].split("_")[1])})
    dump(destination / "data/selections.json", selections)
    summary = {k: v for k, v in summary.items() if k not in
               {"comparison_run", "comparison_costs", "run02_comparison", "protocol", "scan_audit"}}
    summary["protocol"] = protocol
    summary["audit"]["raw_sha256"] = {str(p.relative_to(destination)): digest(p)
        for p in sorted((destination / "data/seeds").glob("seed_*/evaluation.jsonl"))}
    summary["scan_audit"] = {"trials": 29520, "per_seed_trials": 4920, "weight_count": 41,
        "raw_sha256": {str(p.relative_to(destination)): digest(p)
                       for p in sorted((destination / "data/seeds").glob("seed_*/fixed_weight_scan.jsonl"))}}
    dump(destination / "data/summary.json", summary)
    copy(RUN / "analysis/trace_arrays.npz", destination / "data/trace_arrays.npz")

    checkpoint_hashes = {}
    expected_cp = read(RUN / "prepared.json")["checkpoint_sha256"]
    for name, sha in expected_cp.items():
        if name.endswith("setup_after_selection.npz"):
            continue
        assert digest(RUN / name) == sha
        copy(RUN / name, destination / name)
        checkpoint_hashes[name] = sha
    dump(destination / "provenance/checkpoint_hashes.json", checkpoint_hashes)
    for name in ("environment.json", "preflight.json", "setup_freeze_audit.json", "complete.json", "final_evaluation_plan.json"):
        copy(RUN / name, destination / "provenance" / name)
    frozen_manifest = read(RUN / "source_manifest.json")
    retained_manifest = {}
    for name, sha in frozen_manifest.items():
        if Path(name).parts[0] not in {"solve", "setup", "problems", "hypre", "experiments"}:
            continue
        src = RUN / "provenance/source" / name
        assert digest(src) == sha
        copy(src, destination / "reproduction/source" / name)
        retained_manifest[name] = sha
    dump(destination / "provenance/frozen_source_hashes.json", retained_manifest)
    dump(destination / "provenance/record_sources.json", source_records)
    dump(destination / "provenance/derivation.json", {
        "accepted_run": str(RUN.relative_to(ROOT)), "source_summary_sha256": digest(RUN / "analysis/summary.json"),
        "source_protocol_sha256": digest(RUN / "protocol.json"), "source_inputs_sha256": digest(RUN / "inputs.json"),
        "source_selections_sha256": digest(RUN / "selections.json"),
        "transformations": ["Split current evaluation rows by seed without changing record contents",
            "Retain only the fixed-grid searches supporting the accepted checkpoint of each seed",
            "Remove cross-run timing comparisons and unused inherited hierarchy/policy selections",
            "Keep only the seven current main figures and ten supplementary heatmaps"],
        "original_experiment_files_unchanged": True})

    tables = {"policy_reductions_no_overhead": rows_for_table(summary),
              "rl_advantage_no_overhead": rows_for_table(summary, direct=True),
              "policy_reductions_with_overhead": rows_for_table(summary, inclusive=True),
              "rl_advantage_with_overhead": rows_for_table(summary, direct=True, inclusive=True)}
    for name, rows in tables.items():
        write_csv(destination / "tables" / f"{name}.csv", rows)
    write_csv(destination / "tables/selected_weights.csv", selected_rows)
    for name in ("per_seed", "per_repeat", "repetition_diagnostics"):
        write_csv(destination / "tables" / f"{name}.csv", summary[name])
    reliability = []
    for method in METHODS:
        rr = [r for r in summary["per_repeat"] if r["method"] == method]
        agg = next(r for r in summary["aggregate"] if r["method"] == method)
        reliability.append({"method": method, "policy_role_observations": 1800,
            "unrecovered_trials": sum(r["failures"] for r in rr),
            "recovered_trials": sum(r["recoveries"] for r in rr),
            "recovered_seed_problem_pairs": agg["recovered_case_seed_pairs"]})
    write_csv(destination / "tables/reliability.csv", reliability)
    (destination / "tables/paper_tables.tex").write_text(
        "% Percent reductions; recovery retained; no controller overhead. SD is across six checkpoints.\n" +
        latex_table(tables["policy_reductions_no_overhead"], "Native time reduction versus matched weight one (percent).", "tab:module05-default") +
        latex_table(tables["rl_advantage_no_overhead"], "Direct native time reduction of frozen RL versus each comparator (percent).", "tab:module05-rl"))

    figures = RUN / "analysis/paper_figures_best_seed"
    entries = []
    for entry in read(figures / "figure_catalog.json"):
        if entry["name"] == "07_run02_vs_run03":
            continue
        entry = json.loads(json.dumps(entry))
        for ext, name in entry["files"].items():
            new_name = name.replace("08_timing_repetitions", "07_timing_repetitions")
            copy(figures / name, destination / "figures" / new_name)
            entry["files"][ext] = new_name
        entry["name"] = entry["name"].replace("08_timing_repetitions", "07_timing_repetitions")
        entries.append(entry)
    assert len(entries) == 17 and sum(e["category"] == "main" for e in entries) == 7
    dump(destination / "figures/figure_catalog.json", entries)
    copy(figures / "best_seed_selection.json", destination / "figures/best_seed_selection.json")
    captions = ["# Final Module 05 figure captions", "",
                "Seven main figures and ten supplementary heatmaps. All plots use the accepted six-checkpoint set.", ""]
    for entry in entries:
        captions += ["## " + entry["name"], "", entry["caption"], ""]
    (destination / "figures/CAPTIONS.md").write_text("\n".join(captions))
    dump(destination / "provenance/figure_generation.json", {
        "included_figure_count": 17, "included_main_figure_count": 7,
        "included_figures": [e["name"] for e in entries],
        "filter": "Cross-run comparison excluded from this current-only release",
        "generation_record": read(figures / "provenance.json")})
    for filename in ("plot_05_policy_refresh.py", "polish_05_policy_refresh_figures.py"):
        copy(ROOT / "experiments/paper_final" / filename,
             destination / "reproduction/source/experiments/paper_final" / filename)
    copy(__file__, destination / "reproduction/package_05_policy_final.py")
    copy(ROOT / "experiments/archive/paper_development/verify_05_policy_final.py", destination / "reproduction/verify_archive.py")
    (destination / "reproduction/README.md").write_text(
        "# Reproduction and verification\n\n"
        "From the extracted archive run `python3 reproduction/verify_archive.py`. This requires only Python's standard library. "
        "It checks every file hash, exact final coverage, all four cost arrays, the reported native reductions, "
        "and every global/per-instance fixed choice from the retained scans.\n\n"
        "The captured source under `source/` and native library/environment hashes are audit material. "
        "Compiled native libraries and the full conda environment are not bundled. Solver execution still requires the pinned research environment. "
        "The frozen experiment runners reference their original preparation directories; verification of this archive does not require those directories.\n\n"
        "`package_05_policy_final.py` is the packaging source, invoked as a module inside the original repository. "
        "The plotting sources record how the supplied figures were generated. No native experiment was repeated while packaging.\n")

    finding = (
        "Frozen RL reduces native continuation time by **41.50% +/- 1.52 percentage points** versus matched w=1, "
        "with a six-checkpoint range of **40.24%-43.83%**. Its mean direct reductions are **11.36%** versus global best fixed, "
        "**9.31%** versus per-instance best fixed, **3.58%** versus the development-selected (2.5,1) schedule, "
        "and **11.10%** versus periodic (1,3). Every cumulative native comparison favors RL on all six checkpoints.")
    readme = "\n\n".join([
        "# Module 05 - completed final archive", "Registered 2026-09-27. Final accepted dataset: diffusion 60^3, Joint hierarchies, six checkpoints, "
        "100 shared test problems and three fresh timing repetitions per problem/policy.",
        "Seed 4 uses the retrained Module 04 setup selector and RL controller evaluated in Module 05 Run 03. "
        "Seeds 1, 2, 3, 5 and 6 retain their original Module 04 checkpoints. 'Run 03' is a Module 05 execution number, not Module 03 training data.",
        "## Main findings", finding,
        "No-overhead cost includes native solve and recovery setup/solve. Common initial setup and controller overhead are excluded. "
        "Costs average all three timing repetitions within each problem before summing; reductions are ratios of cumulative costs. "
        "Mean +/- SD describes six seed-level reductions, not 600 independent matrices or a confidence interval.",
        "## Reduction versus matched w=1", markdown_table(tables["policy_reductions_no_overhead"]),
        "## Direct RL reduction versus each comparator", markdown_table(tables["rl_advantage_no_overhead"]),
        "Each direct comparison uses the comparator's own cumulative cost as denominator; it is not subtraction of Table 1 percentages.",
        "## Baselines and reliability",
        "*Fixed denotes successful best-observed grid selection over 41 weights from 1.00 to 3.00, step 0.05, followed by three independent retimings. "
        "Global fixed is one weight across 100 problems per checkpoint; per-instance fixed is one constant per problem. "
        "The global weight is 1.55 for seed 1 and 1.60 for seeds 2-6. Neither is a continuous optimum. "
        "The (2.5,1) schedule was selected on separate development inputs; (1,3) is prescribed, and neither is claimed the global best periodic schedule.",
        "All 10,260 unique final executions completed successfully after recovery, covering 10,800 policy roles because duplicate fixed choices share "
        "one execution. There are zero unrecovered failures and 15 recovered periodic-(1,3) executions across five seed/problem pairs. "
        "All recovery costs are retained. The 29,520 fixed-grid records are the search evidence still used by this accepted checkpoint set; "
        "they are separate from the fresh final evaluation and contain no superseded seed-4 scan.",
        "## Interpretation and selection disclosure",
        "This accepted set follows inspection of earlier outcomes and includes a retrained seed 4; it is not a new prespecified six-seed replication. "
        "Seed 1 participated in earlier development. The 100 test inputs are fresh relative to training and shared across the six checkpoints. "
        "The main heatmaps use seed 2 because it has the best observed native reduction (43.83%); all seeds remain in aggregate results and supplementary heatmaps.",
        "These comparisons establish frozen-policy utility in the stated metric, not the necessity of RL or within-solve feedback. "
        "Secondary controller-inclusive tables are included: RL's mean reduction versus w=1 is 39.14%, while its mean direct reduction versus tuned (2.5,1) "
        "is -0.29%. An overhead-inclusive advantage over that schedule is not established.",
        "## Contents",
        "- `MODULE05_FINAL.pdf`: two-page summary and tables followed by seven current main figures.\n"
        "- `tables/`: both primary no-overhead tables, secondary inclusive tables, timing records, selected weights, reliability, and LaTeX tables.\n"
        "- `figures/`: one current set of 17 figures (PDF, SVG, 600-dpi PNG, previews), with captions and best-seed selection.\n"
        "- `data/seeds/seed_1` through `seed_6`: final raw trials, retained fixed scans, and exact execution jobs.\n"
        "- `data/`: current-only summary, input definitions, trace arrays, protocol and fixed choices.\n"
        "- `checkpoints/`: the six accepted controllers, encoder definitions, setup states and training configurations. "
        "Seed 4's additional setup-evaluation state differs only by its candidate cursor; it is not an older model.\n"
        "- `provenance/`, `reproduction/`, `COMPLETION.json`, and `MANIFEST.sha256`: audit and verification material.",
        "## Verify",
        "After extracting, run `python3 reproduction/verify_archive.py`. The verifier needs only the Python standard library and no earlier results folders. "
        "It checks archive hashes, final coverage, numerical summaries, and all retained fixed-weight selections. "
        "Superseded evaluation runs, old seed-4 checkpoints, old comparison figures, nested ZIPs, and live progress logs are excluded.",
        "Module 05 is complete for the user-approved diffusion-60 frozen-policy comparison. Wider PDE/size tests, conventional baselines and retrained "
        "feedback ablations are separate research scope, not unfinished work in this registered module.", ""])
    (destination / "README.md").write_text(readme)
    summary_pdf(destination, summary, tables, entries)
    print("Created brief summary, tables and current figure PDF", flush=True)

    audit = verify(destination, check_manifest=False)
    completion = {"module": "05", "status": "complete", "package_id": PACKAGE_ID,
        "registered_at": datetime.now(timezone.utc).isoformat(), "completed_at": read(RUN / "complete.json")["at"],
        "scope": "Diffusion 60^3; six accepted Joint checkpoints; 100 shared test problems; six policy roles; three timings",
        "checkpoint_set": protocol["checkpoint_set"], "selection_disclosure": protocol["selection_disclosure"],
        "audit": {k: v for k, v in audit.items() if k != "manifest_verified"},
        "integrity_validation": "Every manifest hash is checked before ZIP publication and by the standalone verifier",
        "figure_count": 17, "main_figure_count": 7, "pdf_pages": 9,
        "approved_scope_complete": True}
    dump(destination / "COMPLETION.json", completion)
    manifest = "".join(f"{digest(p)}  {p.relative_to(destination)}\n" for p in sorted(destination.rglob("*"))
                       if p.is_file() and p.name != "MANIFEST.sha256")
    (destination / "MANIFEST.sha256").write_text(manifest)
    verified = verify(destination)
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, Path(destination.name) / path.relative_to(destination))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == sum(p.is_file() for p in destination.rglob("*"))
    archive_hash = digest(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{archive_hash}  {archive_path.name}\n")
    return {"archive": str(archive_path), "size_mb": archive_path.stat().st_size / 1e6,
            "sha256": archive_hash, "audit": verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT)
    print(json.dumps(build(parser.parse_args().destination), indent=2))
