"""Package the completed new Module 05/06 data and figures without rerunning.

Uses only the standard library. Source experiments and checksums are retained
at their repository-relative paths inside the archive.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN05 = ROOT / "results/paper_final/05_online_policies/20260928_shared_prefix"
RUN06 = ROOT / "results/paper_final/06_policy/20260928_frozen_five_methods"
NAME = "module05_module06_20260928"
RELEASE = ROOT / "results/paper_final/releases" / NAME
ZIP = RELEASE.with_suffix(".zip")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return sha_stream(handle)


def sha_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024*1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_current_data():
    training = RUN05 / "training"
    figure_dir = training / "analysis/paper_figures"
    figures = read(figure_dir / "figure_data.json")
    source_map = read(training / "complete.json")["trajectory_hashes"]
    checked = 0
    for method, expected_sha in source_map.items():
        path = training / "trajectories" / f"{method}.jsonl"
        assert sha(path) == expected_sha
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == 5000 and len({r["stream_index"] for r in rows}) == 5000
        # Independent row selection by recorded problem index, rather than
        # sharing the plotting adapter's slicing or aggregation helpers.
        for window in figures["windows"].values():
            selected = [r for r in rows if window["problem_start"] <= r["stream_index"]+1 <= window["problem_stop"]]
            values = window["methods"][method]
            assert len(selected) == values["cases"]
            mapping = {"setup_runtime": "setup_runtime", "native_solve_runtime": "solve_runtime",
                       "controller_runtime": "infer_runtime", "setup_bandit_overhead": "bandit_overhead_runtime",
                       "native_total_runtime": "runtime", "end_to_end_runtime": "end_to_end_runtime"}
            for display, field in mapping.items():
                actual = sum(row["outcome"][field] for row in selected)
                assert math.isclose(actual, values["totals_sec"][display], rel_tol=1e-12, abs_tol=1e-9)
                assert math.isclose(actual/len(selected), values["means_sec"][display], rel_tol=1e-12, abs_tol=1e-12)
                checked += 1
    completion = read(RUN06 / "complete.json")
    assert completion["passed"]
    assert sha(RUN06 / "raw.jsonl") == completion["raw_sha256"]
    assert sha(RUN06 / "summary.json") == completion["summary_sha256"]
    module06 = read(RUN06 / "summary.json")
    rows = [json.loads(line) for line in (RUN06 / "raw.jsonl").open()]
    assert len(rows) == len({(r["method"], r["repeat"], r["case_id"]) for r in rows}) == 2100
    for method, metrics in module06["overall_mean_seconds"].items():
        selected = [r for r in rows if r["method"] == method]
        assert len(selected) == 300
        for field, reference in metrics.items():
            actual = sum(r[field] for r in selected)/300
            assert math.isclose(actual, reference, rel_tol=1e-11, abs_tol=1e-11)
            checked += 1
    result = {"passed": True, "training_logical_rows": 25000, "frozen_trials": 2100,
              "independently_checked_window_metric_cells": checked,
              "module05_figure_count": 2, "module06_figure_count": 12,
              "source_trajectory_hashes_verified": True,
              "formal_module06_raw_and_summary_hashes_verified": True,
              "module05_layout_review": "Both exported previews inspected at final layout; labels, component bars, legend and shared-settings panel are visible and do not overlap.",
              "module06_existing_figure_verification": read(RUN06 / "analysis/paper_figures/VERIFICATION.json"),
              "scope": "Rechecks current numerical summaries and archive contents; no retraining, native solves, or new statistical replication."}
    dump(figure_dir / "VERIFICATION.json", {k: v for k, v in result.items() if k != "module06_existing_figure_verification"})
    return result


def main():
    RELEASE.mkdir(parents=True, exist_ok=True)
    validation = validate_current_data()
    entries = {}
    roles = {}
    excluded = []

    def add(path, role):
        path = Path(path)
        assert path.is_file(), path
        relative = path.relative_to(ROOT).as_posix()
        entries[relative] = path
        roles[relative] = role

    for directory, role in ((RUN05, "module05_results"), (RUN06, "module06_results")):
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.name in (".DS_Store", ".run.lock", ".launch.lock", ".lock") or "__pycache__" in path.parts or path.suffix == ".pyc":
                excluded.append({"path": str(path.relative_to(ROOT)), "reason": "Local lock, cache or OS metadata; no experiment observations."})
                continue
            add(path, role)

    locked = {}
    for path in (RUN05 / "training/source_manifest.json", RUN06 / "source_manifest.json"):
        manifest = read(path)
        manifest = manifest.get("files", manifest)
        for name, expected_sha in manifest.items():
            assert name not in locked or locked[name] == expected_sha
            assert sha(ROOT / name) == expected_sha, f"Run-time source differs: {name}"
            locked[name] = expected_sha
            add(ROOT / name, "verified_run_time_source")

    extras = [ROOT / "experiments/paper_final" / name for name in (
        "plot_05_checkpoint_training.py", "plot_06_policy.py", "collect_06_w1_traces.py",
        "verify_period_two_minimax.py", "plot_05_policy.py", "plot_05_policy_anchored.py", "package_05_06.py")]
    extras += list((ROOT / "experiments/paper_final/module05_figures").glob("*.py"))
    extras += [ROOT / name for name in (
        "environment.yml", "setup/requirements.txt", "README.md", "hypre/Makefile",
        "hypre/bindings/Makefile", "hypre/interfaces/Makefile", "hypre/source/LICENSE-MIT", "hypre/source/LICENSE-APACHE",
        "experiments/paper_final/README.md", "experiments/paper_final/05_online_policies/README.md",
        "experiments/paper_final/06_policy/README.md", "docs/theory/period_two_weighted_minimax_20260928.md",
        "docs/theory/period_two_minimax_verification_20260928.json")]
    for directory in (ROOT / "experiments/paper_final/05_online_policies", ROOT / "experiments/paper_final/06_policy"):
        extras += [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".command", ".py")]
    for path in extras:
        add(path, "current_analysis_or_documentation" if str(path.relative_to(ROOT)) not in locked else "verified_run_time_source")

    figure05 = "results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures"
    figure06 = "results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures"
    note = f"""# Current Module 05 and Module 06 — September 28, 2026

This archive contains the **new shared-prefix Module 05 training** and its
**fresh frozen Module 06 evaluation**. It is distinct from the historical
Module 05 Run 04 release. There is one independently trained checkpoint set.

## Start with the figures

- [Module 05: both runtime figures, two-page PDF]({figure05}/module05_runtime_comparisons.pdf)
- [Module 05: full 5000 image]({figure05}/all_5000_runtime_breakdown.png)
- [Module 05: final 1000 image]({figure05}/last_1000_runtime_breakdown.png)
- [Module 05: precise values and ranking explanation]({figure05}/REPORT.md)
- [Module 06: all 12 figures, PDF]({figure06}/all_figures.pdf)
- [Module 06: browser gallery]({figure06}/index.html)
- [Module 06: formal numerical report](results/paper_final/06_policy/20260928_frozen_five_methods/REPORT.md)
- [Module 06: controller-transfer diagnostic](results/paper_final/06_policy/20260928_frozen_five_methods/analysis/controller_transfer_diagnostic/REPORT.md)

## The ranking is window-dependent

| End-to-end comparison | Periodic | LSTDQ |
|---|---:|---:|
| Module 05 full 5000, total seconds | 768.017604 | 763.506032 |
| Module 05 final 1000, total seconds | 129.165991 | 135.394106 |
| Module 06 frozen, mean milliseconds per input | 119.988299 | 124.671113 |

RL leads the cumulative training total by 0.59%, while Periodic already leads
in the final training thousand. The frozen test agrees with the late-training
ordering. RL's native solve remains a little faster in both late training and
the frozen test; its setup construction and controller costs exceed that
saving. Full component values and the historical Run 04 distinction are in
the linked Module 05 report. These are descriptive training results, not
independent-retraining significance claims.

## Included

- All five Module 05 training trajectories, 5000 aligned rows each; matrices/RHS
  generation inputs; actual method-order records; all five final setup
  checkpoints; the frozen LSTDQ controller; shared-prefix/fork audits.
- Default-hierarchy fixed-weight calibration, the historical 1.60 selection
  records, functional checks and preparation logs, separately labelled.
- Module 06's 2100 formal trials: five own pairings plus two crossed pairings,
  100 fresh inputs and three timing repetitions; selections, candidate
  schedules, model copies and freeze audits.
- Two Module 05 figures and all 12 current Module 06 figures in PNG, PDF and
  SVG, captions, chart data, provenance and reproduction code.
- The separately verified W1 residual-trace replay. Its diagnostic timings
  do not replace formal measurements.
- The controller-transfer diagnosis and its reproducible read-only probes.
- All {len(locked)} files from the verified run-time source manifests, including
  the original plotting helper and captured native libraries, plus current
  analysis scripts, environment information and the current period-two theory.

Files retain repository-relative paths under this archive's root. Start here
rather than the repository's broader README. References inside original
logs/manifests may retain their original absolute workstation paths; those
files were preserved byte for byte. Historical experiment data and Module 04
results are not bundled. Native dependencies/environment are documented,
not automatically installed or guaranteed portable to another operating system.

Module 05 executed its common 1000-problem W1 prefix **once**, then five
4000-problem continuations: **21,000 physical executions**, or 25,000 logical
branch exposures. Each full-stream bar includes that prefix once for the
branch; summing those five bars would overcount actual training work.

Module 06's three repetitions are repeated measurements of the same frozen
models and the same 100 inputs, not three independent training seeds. Models
remain frozen even when timing-sensitive state features change RL actions.
No extra training or formal tests were launched to produce this archive.

## Reproduce analysis

Use the recorded research Python environment and run from the extracted root:

```sh
python -m experiments.paper_final.plot_05_checkpoint_training
python -m experiments.paper_final.plot_06_policy
PYTHONPATH=. python results/paper_final/06_policy/20260928_frozen_five_methods/analysis/controller_transfer_diagnostic/reproduce.py
```

These commands read existing data/checkpoints. Full native experiment reruns
need the recorded solver dependencies and a new output location; original
completed data must not be overwritten. Original absolute source-path checks
may need deliberate relocation when moving to another machine.

`MANIFEST.json` records every payload's size, SHA-256, role and source path.
`MANIFEST.sha256` can be checked from the extracted root with standard SHA-256
tools. `verify_archive.py` checks every entry using only Python's standard
library. `PACKAGE_AUDIT.json` records data validation and source verification.
"""
    (RELEASE / "START_HERE.md").write_text(note)
    verifier = '''"""Verify this extracted archive using only Python's standard library."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, spec in manifest["files"].items():
    path = root / name
    assert path.is_file(), name
    assert path.stat().st_size == spec["bytes"], name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == spec["sha256"], name
print(f"Verified {len(manifest['files'])} archived files.")
'''
    (RELEASE / "verify_archive.py").write_text(verifier)
    audit = {"data_validation": validation, "verified_run_time_source_files": len(locked),
        "excluded_local_metadata": excluded, "module04_files_modified": False,
        "new_formal_experiments": 0, "new_training": 0,
        "native_portability": "Captured macOS libraries are evidence of the executed build; dependency compatibility on another machine is not verified."}
    dump(RELEASE / "PACKAGE_AUDIT.json", audit)
    for name in ("START_HERE.md", "verify_archive.py", "PACKAGE_AUDIT.json"):
        entries[name] = RELEASE / name
        roles[name] = "package_documentation"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_root": NAME,
        "files": {name: {"sha256": sha(path), "bytes": path.stat().st_size,
                          "role": roles[name], "source": str(path.relative_to(ROOT))}
                  for name, path in sorted(entries.items())}}
    dump(RELEASE / "MANIFEST.json", manifest)
    (RELEASE / "MANIFEST.sha256").write_text("".join(f"{spec['sha256']}  {name}\n" for name, spec in manifest["files"].items()))
    total_bytes = sum(s["bytes"] for s in manifest["files"].values())
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, f"{NAME}/{name}")
        for name in ("MANIFEST.json", "MANIFEST.sha256"):
            archive.write(RELEASE / name, f"{NAME}/{name}")
    print(f"Built {ZIP.name}; checking every archived file...", flush=True)
    with zipfile.ZipFile(ZIP) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(entries)+2
        assert archive.testzip() is None
        for name, spec in manifest["files"].items():
            with archive.open(f"{NAME}/{name}") as handle:
                assert sha_stream(handle) == spec["sha256"], name
            assert sha(entries[name]) == spec["sha256"], f"Source changed during packaging: {name}"
    report = {"passed": True, "archive": str(ZIP), "archive_sha256": sha(ZIP),
        "archive_bytes": ZIP.stat().st_size, "uncompressed_payload_bytes": total_bytes,
        "payload_files": len(entries), "archive_members": len(entries)+2,
        "counts_by_role": dict(Counter(roles.values())),
        "every_archived_payload_hash_verified": True, "all_original_payloads_unchanged": True}
    dump(RELEASE / "archive_verification.json", report)
    ZIP.with_suffix(".zip.sha256").write_text(f"{report['archive_sha256']}  {ZIP.name}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
