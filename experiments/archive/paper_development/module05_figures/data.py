"""Validate and align recorded cases without importing the native solver."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCES = ("bandit_default", "bandit_lstdq")
SOURCE_NAMES = {"bandit_default": "Setup-only hierarchies", "bandit_lstdq": "Joint hierarchies"}
METHODS = ("reference", "fixed", "oracle", "periodic", "prefix", "rl")
SUMMARY_NAMES = {
    "reference": "Weight 1", "fixed": "Best fixed - test hindsight",
    "oracle": "Per-problem best fixed - test hindsight", "periodic": "Development periodic",
    "prefix": "Development prefix-tail", "rl": "Frozen LSTDQ",
    "native": "Native HYPRE smoother", "chebyshev": "Native Chebyshev",
}
LABELS = {"reference": r"Reference $w=1$", "fixed": "Best fixed*", "oracle": "Per-problem fixed*",
          "periodic": "Tuned periodic", "prefix": "Tuned prefix–tail", "rl": "Frozen RL",
          "native": "Native HYPRE smoother", "chebyshev": "Native Chebyshev"}
FIELDS = ("native", "inclusive", "native_total", "inclusive_total", "cycles")


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reduction(reference, candidate):
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if np.any(reference <= 0):
        raise ValueError("Reduction denominator must be positive")
    return 100 * (1 - candidate / reference)


def case_order(cycle_cube):
    """One shared order; ties use original input index, never an RL outcome."""
    difficulty = np.mean(cycle_cube, axis=tuple(range(cycle_cube.ndim - 1)))
    return np.lexsort((np.arange(len(difficulty)), difficulty))


def padded_trace(values, width):
    """NaN means unexecuted. Never extend the last action past termination."""
    if len(values) > width:
        raise ValueError("Plot width would truncate observed cycles")
    out = np.full(width, np.nan)
    out[:len(values)] = values
    return out


def select_oracles(cells, seed, source, case_ids, weights):
    """Select from mean costs, with every test case equally weighted."""
    valid = [name for name in weights if all(cells[seed, source, case, name]["success"] for case in case_ids)]
    if not valid:
        raise ValueError("No all-success global fixed weight; cannot draw the current figure protocol")
    best = min(valid, key=lambda name: (sum(cells[seed, source, case, name]["native"] for case in case_ids), name))
    individual = {}
    for case in case_ids:
        choices = [name for name in weights if cells[seed, source, case, name]["success"]]
        if not choices:
            raise ValueError("Uncovered per-case oracle; do not silently omit this case")
        individual[case] = min(choices, key=lambda name: (cells[seed, source, case, name]["native"], name))
    return best, individual


@dataclass
class Dataset:
    root: Path
    protocol: dict
    summary: dict
    cells: dict
    traces: dict
    chosen: dict
    order: np.ndarray
    provenance: dict

    @property
    def seeds(self):
        return self.protocol["training_seeds"]

    @property
    def cases(self):
        return list(range(self.protocol["test_cases"]))

    def policy(self, seed, source, method, case):
        value = self.chosen[seed, source, method]
        return value[case] if isinstance(value, dict) else value

    def values(self, source, method, field="native", *, ordered=False):
        ids = self.order if ordered else self.cases
        return np.array([[self.cells[seed, source, int(case), self.policy(seed, source, method, int(case))][field]
                          for case in ids] for seed in self.seeds])

    def totals(self, source, method, field="native"):
        return self.values(source, method, field).sum(axis=1)

    def gains(self, source, method, reference="reference", field="native"):
        return reduction(self.totals(source, reference, field), self.totals(source, method, field))

    def paired_gains(self, source, comparator):
        return reduction(self.values(source, comparator, ordered=True), self.values(source, "rl", ordered=True))

    def trace(self, seed, source, method, case):
        key = (seed, source, case, self.policy(seed, source, method, case))
        t = self.traces[key]
        cell = self.cells[key]
        if not cell["success"] or cell["recovery"]:
            raise ValueError(f"A displayed policy used recovery; plot primary/fallback explicitly: {key}")
        return t

    def heatmap(self, seed, source, method, field, width):
        return np.vstack([padded_trace(self.trace(seed, source, method, int(case))[field], width) for case in self.order])

    def maximum_cycles(self):
        return max(len(self.trace(seed, source, method, case)["actions"])
                   for seed in self.seeds for source in SOURCES for method in METHODS for case in self.cases)

    def data_export(self):
        return {"case_order_0_based": self.order.tolist(),
                "order_rule": "Mean reference-weight cycle count across both hierarchy sources and all six checkpoints; ties by case ID",
                "representative_seed": min(self.seeds), "trajectory_repeat": 0,
                "timing_repeats": "All prescribed repetitions averaged within each case/policy; cases equally weighted",
                "methods": LABELS, "sources": SOURCE_NAMES,
                "case_costs": {source: {method: {field: self.values(source, method, field).tolist() for field in FIELDS}
                    for method in SUMMARY_NAMES} for source in SOURCES},
                "chosen_policies": {f"{s}/{h}/{m}": v for (s, h, m), v in self.chosen.items()}}


def load_dataset(root):
    root = Path(root).resolve()
    if not (root / "complete.json").exists() or read(root / "status.json")["status"] != "complete":
        raise ValueError("The experiment must be complete")
    protocol, summary = read(root / "protocol.json"), read(root / "analysis/summary.json")
    selections = read(root / "selected_baselines.json")["selections"]
    seeds, cases = protocol["training_seeds"], list(range(protocol["test_cases"]))
    prepared = read(root / "prepared.json")
    hashes = {"protocol.json": prepared["protocol_sha256"], "inputs.json": prepared["inputs_sha256"],
              "jobs_test.json": prepared["jobs_sha256"]["test"]}
    for phase in ["development", "test", "repetitions"]:
        hashes.update(read(root / "raw" / phase / "complete.json")["raw_sha256"])
    for name, expected in hashes.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Saved experiment input/data changed: {name}")

    cells, traces = {}, {}
    phase_counts = {}
    for phase in ["test", "repetitions"]:
        seen = set()
        for path in sorted((root / "raw" / phase).glob("worker_*.jsonl")):
            with path.open() as handle:
                for line in handle:
                    r = json.loads(line)
                    key = (r["seed"], r["source"], r["case_id"], r["policy"])
                    trial = (*key, r["repeat"])
                    if trial in seen:
                        raise ValueError(f"Duplicate trial: {trial}")
                    seen.add(trial)
                    o = r["outcome"]
                    values = np.array([r["native_continuation_sec"], r["inclusive_continuation_sec"],
                                       r["native_total_sec"], r["inclusive_total_sec"],
                                       o.get("primary_cycles", o.get("iterations", 0))])
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError("Invalid cost or cycle count")
                    cell = cells.setdefault(key, {"sum": np.zeros(len(FIELDS)), "count": 0,
                                                  "success": True, "recovery": False, "repeats": set()})
                    if r["repeat"] in cell["repeats"]:
                        raise ValueError("Duplicate across phases")
                    cell["repeats"].add(r["repeat"])
                    cell["sum"] += values
                    cell["count"] += 1
                    cell["success"] &= r["success"]
                    cell["recovery"] |= o.get("fallback_used", False)
                    if phase == "test" and r["kind"] != "native":
                        actions, residuals, times = [np.asarray(o.get(k, []), dtype=float)
                            for k in ("cycle_actions", "cycle_residuals", "cycle_times")]
                        if len(actions) != len(residuals) or len(actions) != len(times):
                            raise ValueError("Unaligned cycle trace")
                        traces[key] = {"actions": actions, "residuals": residuals, "times": times}
        expected_count = read(root / "raw" / phase / "complete.json")["trials"]
        if len(seen) != expected_count:
            raise ValueError("Incomplete phase")
        phase_counts[phase] = len(seen)
    for key, cell in cells.items():
        repeats = {0, 1, 2} if key[2] in protocol["test_repeat_case_ids"] else {0}
        if cell["repeats"] != repeats:
            raise ValueError(f"Incorrect repeat coverage: {key}")
        cell.update(dict(zip(FIELDS, cell.pop("sum") / cell["count"])))
        del cell["repeats"]

    chosen = {}
    weights = sorted(k for k, spec in protocol["policies"].items() if spec["kind"] == "fixed")
    saved = {(r["seed"], r["source"], r["method"]): r for r in summary["per_seed"]}
    for seed in seeds:
        for source in SOURCES:
            best, per_case = select_oracles(cells, seed, source, cases, weights)
            chosen[seed, source, "fixed"], chosen[seed, source, "oracle"] = best, per_case
            chosen[seed, source, "reference"] = "fixed_1.00"
            chosen[seed, source, "rl"] = "rl_frozen_lcb"
            chosen[seed, source, "periodic"] = selections[f"{seed}/{source}"]["periodic"]["policy"]
            chosen[seed, source, "prefix"] = selections[f"{seed}/{source}"]["prefix"]["policy"]
            chosen[seed, source, "native"] = "native_hypre_default"
            chosen[seed, source, "chebyshev"] = "native_chebyshev16"
    data = Dataset(root, protocol, summary, cells, traces, chosen, np.arange(len(cases)),
                   {"source_hashes": hashes, "raw_counts": phase_counts, "hashes_verified": True})
    # Recompute every displayed aggregate from raw data, including oracle choices.
    for source in SOURCES:
        for method, name in SUMMARY_NAMES.items():
            for field, saved_field in [("native", "native_continuation_sec"), ("inclusive", "inclusive_continuation_sec"),
                                       ("native_total", "native_total_sec"), ("inclusive_total", "inclusive_total_sec")]:
                expected = [saved[seed, source, name][saved_field] for seed in seeds]
                np.testing.assert_allclose(data.totals(source, method, field), expected, rtol=1e-11, atol=1e-10)
    data.order = case_order(np.stack([data.values(source, "reference", "cycles") for source in SOURCES]))
    data.provenance["summary_matches_recomputed_raw_costs"] = True
    return data
