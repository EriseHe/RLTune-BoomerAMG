"""Supplement the batch-W1 log with verified residual traces; never retime results."""
from experiments.paper_final import run_06_frozen_methods as study

import json
from pathlib import Path
import numpy as np


def collect(output):
    study.verify_prepared(output)
    study.base.environment_check(output / "analysis/w1_trace_environment")
    study.configure_smoother_profile(study.base.PROFILE)
    inputs = study.base.read(output / "inputs.json")["test"]
    choices = study.base.read(output / "choices.json")["test"]["bandit_default"]
    raw = list(study.base.read_records(output / "raw.jsonl"))
    original = {(r["repeat"], r["case_id"]): r for r in raw if r["method"] == "bandit_default"}
    assert len(original) == 300
    destination = output / "analysis/w1_trace_replay.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    originals = {name: study.base.file_hash(output / name) for name in ("raw.jsonl", "summary.json", "complete.json", "protocol.json", "choices.json", "inputs.json")}
    saved = list(study.base.read_records(destination)) if destination.exists() else []
    if len({r["case_id"] for r in saved}) != len(saved):
        raise AssertionError("Repeated W1 trace replay")
    rows = {r["case_id"]: r for r in saved}
    def verify(row):
        i = row["case_id"]
        assert row["input_id"] == inputs[i]["input_id"]
        assert row["hierarchy_id"] == choices[i]["hierarchy_id"]
        o = row["outcome"]
        assert not o["failed"] and o["residual_norm"] <= 1e-6
        assert len(o["cycle_actions"]) == len(o["cycle_residuals"]) == o["iterations"]
        np.testing.assert_array_equal(o["cycle_actions"], np.ones(o["iterations"]))
        for rep in range(3):
            previous = original[rep, i]["outcome"]
            assert o["iterations"] == previous["primary_cycles"]
            np.testing.assert_allclose(o["residual_norm"], previous["primary_residual_norm"], rtol=1e-10, atol=1e-14)
    for row in saved:
        verify(row)
    with destination.open("a") as handle:
        for inp, choice in zip(inputs, choices):
            if inp["case_id"] in rows:
                continue
            outcome = study.solve_fixed_w_case(params=choice["params"], mkw=inp["mkw"], w=1.,
                                               sweeps_down=1, sweeps_up=1, solve_tol=1e-6, solve_max_cycles=50)
            row = {"case_id": inp["case_id"], "input_id": inp["input_id"], "hierarchy_id": choice["hierarchy_id"],
                   "params": choice["params"], "outcome": outcome, "at": study.base.now(),
                   "purpose": "Residual trajectory only; replay timing excluded from all evaluation costs"}
            verify(row)
            handle.write(json.dumps(study.base.ready(row), allow_nan=False)+"\n")
            handle.flush()
            rows[inp["case_id"]] = row
            if len(rows) % 20 == 0:
                print(json.dumps({"verified_w1_traces": len(rows), "total": 100}), flush=True)
    for name, digest in originals.items():
        assert study.base.file_hash(output / name) == digest
    study.verify_prepared(output)
    audit = {"passed": True, "at": study.base.now(), "traces": 100, "comparisons_to_original_trials": 300,
             "cycle_counts_match": True, "final_residuals_match": True, "residual_rtol": 1e-10, "residual_atol": 1e-14,
             "recorded_timing_replaced": False, "learners_loaded_or_updated": False,
             "original_hashes": originals, "trace_sha256": study.base.file_hash(destination),
             "collector_sha256": study.base.file_hash(Path(__file__)),
             "note": "One deterministic W1 residual trace per input is reused in repetition figures; other policies use their actual recorded repetition traces."}
    study.base.dump(output / "analysis/w1_trace_audit.json", audit)
    print(json.dumps({"passed": True, "verified_traces": 100, "timing_results_unchanged": True}), flush=True)


if __name__ == "__main__":
    collect(study.OUTPUT)
