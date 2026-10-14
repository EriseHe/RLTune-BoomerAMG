"""Paired runtime comparisons for completed experiment streams."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np


def paired_metric(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    seed: int,
) -> Dict[str, Any]:
    baseline = np.asarray(baseline_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("Paired metric arrays must be non-empty and aligned")
    difference = baseline - candidate
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    improvement = (
        float(100.0 * (baseline_mean - candidate_mean) / baseline_mean)
        if baseline_mean != 0.0
        else float("nan")
    )
    rng = np.random.default_rng(int(seed))
    boot = np.empty(2000, dtype=float)
    for index in range(boot.size):
        sample = rng.integers(0, baseline.size, size=baseline.size)
        sampled_baseline = float(np.mean(baseline[sample]))
        sampled_candidate = float(np.mean(candidate[sample]))
        boot[index] = (
            100.0 * (sampled_baseline - sampled_candidate) / sampled_baseline
            if sampled_baseline != 0.0
            else float("nan")
        )
    return {
        "cases": int(baseline.size),
        "baseline_mean_sec": baseline_mean,
        "candidate_mean_sec": candidate_mean,
        "baseline_total_sec": float(np.sum(baseline)),
        "candidate_total_sec": float(np.sum(candidate)),
        "candidate_savings_sec": float(np.sum(difference)),
        "candidate_improvement_pct": improvement,
        "candidate_improvement_95pct": [
            float(np.nanpercentile(boot, 2.5))
            if np.any(np.isfinite(boot))
            else float("nan"),
            float(np.nanpercentile(boot, 97.5))
            if np.any(np.isfinite(boot))
            else float("nan"),
        ],
        "candidate_win_rate": float(np.mean(difference > 0.0)),
    }


def method_comparison(
    candidate_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    if len(candidate_records) != len(baseline_records):
        raise ValueError("Method streams must contain the same number of instances")
    accessors: Dict[str, Callable[[Dict[str, Any]], float]] = {
        "setup_runtime": lambda row: float(row["outcome"]["setup_runtime"]),
        "native_solve_runtime": lambda row: float(row["outcome"]["solve_runtime"]),
        "native_total_runtime": lambda row: float(row["outcome"]["runtime"]),
        "controller_runtime": lambda row: float(
            row["outcome"].get("infer_runtime", 0.0)
        ),
        "setup_bandit_overhead": lambda row: float(
            row["outcome"].get("bandit_overhead_runtime", 0.0)
        ),
        "solve_plus_controller": lambda row: float(
            row["outcome"]["solve_runtime"] + row["outcome"].get("infer_runtime", 0.0)
        ),
        "end_to_end_runtime": lambda row: float(row["outcome"]["end_to_end_runtime"]),
    }
    comparison = {
        name: paired_metric(
            np.asarray([accessor(row) for row in baseline_records], dtype=float),
            np.asarray([accessor(row) for row in candidate_records], dtype=float),
            seed=int(seed + offset),
        )
        for offset, (name, accessor) in enumerate(accessors.items())
    }
    comparison["same_setup_rate"] = float(
        np.mean(
            [
                candidate["params"] == baseline["params"]
                for candidate, baseline in zip(candidate_records, baseline_records)
            ]
        )
    )
    return comparison


__all__ = ["method_comparison", "paired_metric"]
