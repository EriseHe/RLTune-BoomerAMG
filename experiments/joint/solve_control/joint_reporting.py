from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from experiments.joint.solve_control.joint_online_common import _method_stream_summary
from experiments.joint.solve_control.comparison import method_comparison


_STRUCTURED_MODEL_BASED_METHOD = "bandit_structured_model_based"


def _empty_stream_summary() -> Dict[str, Any]:
    runtime_fields = (
        "setup_runtime",
        "native_solve_runtime",
        "native_total_runtime",
        "controller_runtime",
        "setup_bandit_overhead",
        "setup_bandit_select",
        "setup_bandit_loss_eval",
        "setup_bandit_update",
        "end_to_end_runtime",
    )
    return {
        "cases": 0,
        "failed_count": 0,
        "mean_runtime_sec": 0.0,
        "mean_runtime_with_overhead_sec": 0.0,
        "mean_setup_runtime_sec": 0.0,
        "mean_solve_runtime_sec": 0.0,
        "mean_solve_with_overhead_sec": 0.0,
        "mean_policy_overhead_sec": 0.0,
        "mean_feature_runtime_sec": 0.0,
        "mean_decision_runtime_sec": 0.0,
        "mean_update_runtime_sec": 0.0,
        "mean_iterations": 0.0,
        "totals_sec": {field: 0.0 for field in runtime_fields},
        "means_sec": {field: 0.0 for field in runtime_fields},
        "unique_setup_count": 0,
        "primary_failure_count": 0,
        "setup_fallback_count": 0,
        "recovered_failure_count": 0,
        "unrecovered_failure_count": 0,
        "bandit_update_count": 0,
        "controller_update_count": 0,
    }


def _comparison_windows(case_count: int) -> Dict[str, tuple[int, int]]:
    count = int(case_count)
    if count <= 0:
        raise ValueError("case_count must be positive")

    windows = {f"all_{count}": (0, count)}
    if count >= 4_000:
        windows.update(
            {
                "first_2000": (0, 2_000),
                "last_2000": (count - 2_000, count),
            }
        )
    if count >= 1_000:
        windows.update(
            {
                "first_1000": (0, 1_000),
                "last_1000": (count - 1_000, count),
                "last_500": (count - 500, count),
                "last_300": (count - 300, count),
            }
        )
    return windows


def _window_result(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    seed: int,
) -> Dict[str, Any]:
    available_references = {
        "vs_default_setup_default_solve": "default_setup_default_solve",
        "vs_bandit_default": "bandit_default",
        "vs_fixed_w1.6": "bandit_fixed_w1.6",
        "vs_ppo": "bandit_ppo",
        "vs_context8d_fixed": "context8d_fixed",
    }
    references = {
        label: method
        for label, method in available_references.items()
        if method in records
    }
    return {
        "methods": {
            method: _method_stream_summary(rows)
            for method, rows in records.items()
        },
        "comparisons": {
            label: {
                method: method_comparison(
                    rows,
                    records[reference],
                    seed=int(seed + reference_index * 100_000 + method_index * 101),
                )
                for method_index, (method, rows) in enumerate(records.items())
                if method != reference
            }
            for reference_index, (label, reference) in enumerate(references.items())
        },
    }


def _action_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actions: list[float] = []
    explored: list[bool] = []
    uncertainties: list[float] = []
    selected_q_values: list[float] = []
    selected_scores: list[float] = []
    calibration_ratios: list[float] = []
    for row in rows:
        outcome = row["outcome"]
        actions.extend(float(value) for value in outcome.get("cycle_actions", []))
        explored.extend(bool(value) for value in outcome.get("cycle_explored", []))
        uncertainties.extend(
            float(value)
            for value in outcome.get("cycle_selected_uncertainties", [])
        )
        selected_q_values.extend(
            float(value)
            for value in outcome.get("cycle_selected_q_values", [])
            if np.isfinite(float(value))
        )
        selected_scores.extend(
            float(value)
            for value in outcome.get("cycle_selection_scores", [])
            if np.isfinite(float(value))
        )
        errors = outcome.get("postfit_td_errors") or outcome.get("td_errors", [])
        widths = outcome.get("cycle_selected_uncertainties", [])
        for error, width in zip(errors, widths):
            if np.isfinite(float(error)) and float(width) > 0.0:
                calibration_ratios.append(abs(float(error)) / float(width))
    ratios = np.asarray(calibration_ratios, dtype=float)
    residual_width_diagnostic = {
        "paired_updates": int(ratios.size),
        "median_abs_residual_over_parameter_width": (
            float(np.median(ratios)) if ratios.size else 0.0
        ),
        "p90_abs_residual_over_parameter_width": (
            float(np.quantile(ratios, 0.9)) if ratios.size else 0.0
        ),
        "fraction_within_1x_parameter_width": (
            float(np.mean(ratios <= 1.0)) if ratios.size else 0.0
        ),
        "fraction_within_2x_parameter_width": (
            float(np.mean(ratios <= 2.0)) if ratios.size else 0.0
        ),
        "fraction_within_4x_parameter_width": (
            float(np.mean(ratios <= 4.0)) if ratios.size else 0.0
        ),
        "semantics": (
            "TD residual divided by selected parameter-uncertainty width; "
            "not nominal predictive-interval coverage"
        ),
    }
    # Preserve the historical result shape for existing report readers.  The
    # canonical diagnostic above carries the corrected statistical semantics.
    legacy_calibration = {
        "paired_updates": residual_width_diagnostic["paired_updates"],
        "median_abs_error_over_uncertainty": (
            residual_width_diagnostic[
                "median_abs_residual_over_parameter_width"
            ]
        ),
        "p90_abs_error_over_uncertainty": (
            residual_width_diagnostic[
                "p90_abs_residual_over_parameter_width"
            ]
        ),
        "coverage_at_1x": residual_width_diagnostic[
            "fraction_within_1x_parameter_width"
        ],
        "coverage_at_2x": residual_width_diagnostic[
            "fraction_within_2x_parameter_width"
        ],
        "coverage_at_4x": residual_width_diagnostic[
            "fraction_within_4x_parameter_width"
        ],
        "deprecated_semantics": residual_width_diagnostic["semantics"],
    }
    return {
        "decisions": int(len(actions)),
        "mean_weight": float(np.mean(actions)) if actions else float("nan"),
        "explored_rate": float(np.mean(explored)) if explored else 0.0,
        "mean_selected_uncertainty_sec": (
            float(np.mean(uncertainties)) if uncertainties else 0.0
        ),
        "mean_selected_q_sec": (
            float(np.mean(selected_q_values)) if selected_q_values else 0.0
        ),
        "lower_bound_saturation_count": int(
            sum(np.isclose(value, 0.0, atol=1.0e-12, rtol=0.0) for value in selected_scores)
        ),
        "lower_bound_saturation_rate": (
            float(
                np.mean(
                    np.isclose(
                        np.asarray(selected_scores, dtype=float),
                        0.0,
                        atol=1.0e-12,
                        rtol=0.0,
                    )
                )
            )
            if selected_scores
            else 0.0
        ),
        "residual_to_parameter_width": residual_width_diagnostic,
        "confidence_calibration": legacy_calibration,
    }


def _write_summary_csv(
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
) -> None:
    fields = (
        "method",
        "family",
        "cases",
        "mean_setup_runtime_sec",
        "mean_native_solve_runtime_sec",
        "mean_native_total_runtime_sec",
        "mean_controller_runtime_sec",
        "mean_setup_bandit_overhead_sec",
        "mean_end_to_end_runtime_sec",
        "failures",
        "mean_iterations",
        "setup_fallbacks",
        "primary_failures",
        "recovered_failures",
        "unrecovered_failures",
        "bandit_updates",
        "controller_updates",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for method, rows in records.items():
            summary = _method_stream_summary(rows)
            means = summary["means_sec"]
            writer.writerow(
                {
                    "method": method,
                    "family": family_by_method[method],
                    "cases": len(rows),
                    "mean_setup_runtime_sec": means["setup_runtime"],
                    "mean_native_solve_runtime_sec": means["native_solve_runtime"],
                    "mean_native_total_runtime_sec": means["native_total_runtime"],
                    "mean_controller_runtime_sec": means["controller_runtime"],
                    "mean_setup_bandit_overhead_sec": means["setup_bandit_overhead"],
                    "mean_end_to_end_runtime_sec": means["end_to_end_runtime"],
                    "failures": summary["failed_count"],
                    "mean_iterations": summary["mean_iterations"],
                    "setup_fallbacks": summary["setup_fallback_count"],
                    "primary_failures": summary["primary_failure_count"],
                    "recovered_failures": summary["recovered_failure_count"],
                    "unrecovered_failures": summary["unrecovered_failure_count"],
                    "bandit_updates": summary["bandit_update_count"],
                    "controller_updates": summary["controller_update_count"],
                }
            )


def _write_solve_screen_report(
    path: Path,
    *,
    window_results: Dict[str, Any],
    window_actions: Dict[str, Dict[str, Any]],
) -> None:
    """Render the locked screening metrics without requiring plotting tools."""

    all_window = next(
        (
            name
            for name in window_results
            if str(name).startswith("all_")
        ),
        None,
    )
    preferred_windows = tuple(
        name
        for name in (
            all_window,
            "first_1000",
            "last_1000",
            "last_500",
        )
        if name is not None
    )
    reference_options = (
        (
            "vs_default_setup_default_solve",
            "default_setup_default_solve",
            "default setup + default solve",
        ),
        (
            "vs_fixed_w1.6",
            "bandit_fixed_w1.6",
            "Online LinUCB + fixed `w=1.6`",
        ),
    )
    selected_reference = next(
        (
            option
            for option in reference_options
            if any(
                option[0] in window.get("comparisons", {})
                for window in window_results.values()
            )
        ),
        None,
    )
    if selected_reference is None:
        comparison_key = None
        baseline_method = None
        comparison_note = (
            "No supported reference branch was included, so paired "
            "improvement columns are reported as n/a."
        )
    else:
        comparison_key, baseline_method, baseline_label = selected_reference
        comparison_note = (
            "Improvement and paired 95% intervals are relative to "
            f"{baseline_label}."
        )
    lines = [
        "# Solve-Controller Screening Report",
        "",
        "All runtimes are per-instance means in milliseconds.",
        comparison_note,
        "",
    ]
    report_windows = tuple(
        name for name in preferred_windows if name in window_results
    )
    if not report_windows:
        report_windows = tuple(window_results)
    for window_name in report_windows:
        window = window_results[window_name]
        comparisons = (
            window.get("comparisons", {}).get(comparison_key, {})
            if comparison_key is not None
            else {}
        )
        lines.extend(
            [
                f"## {window_name}",
                "",
                "| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, summary in window["methods"].items():
            means = summary["means_sec"]
            if method == baseline_method:
                improvement_text = "baseline"
                same_setup_text = "1.000"
            elif method in comparisons:
                comparison = comparisons[method]
                metric = comparison["end_to_end_runtime"]
                interval = metric["candidate_improvement_95pct"]
                improvement_text = (
                    f"{metric['candidate_improvement_pct']:.2f}% "
                    f"[{interval[0]:.2f}, {interval[1]:.2f}]"
                )
                same_setup_text = (
                    f"{float(comparison['same_setup_rate']):.3f}"
                )
            else:
                improvement_text = "n/a"
                same_setup_text = "n/a"
            lines.append(
                "| {method} | {setup:.3f} | {solve:.3f} | {native:.3f} | "
                "{controller:.3f} | {bandit:.3f} | {e2e:.3f} | {cycles:.2f} | "
                "{improvement} | {primary}/{recovered}/{unrecovered} | {same} |".format(
                    method=method,
                    setup=1000.0 * float(means["setup_runtime"]),
                    solve=1000.0 * float(means["native_solve_runtime"]),
                    native=1000.0 * float(means["native_total_runtime"]),
                    controller=1000.0 * float(means["controller_runtime"]),
                    bandit=1000.0 * float(means["setup_bandit_overhead"]),
                    e2e=1000.0 * float(means["end_to_end_runtime"]),
                    cycles=float(summary["mean_iterations"]),
                    improvement=improvement_text,
                    primary=int(summary["primary_failure_count"]),
                    recovered=int(summary["recovered_failure_count"]),
                    unrecovered=int(summary["unrecovered_failure_count"]),
                    same=same_setup_text,
                )
            )
        action_rows = window_actions.get(window_name, {})
        if action_rows:
            lines.extend(
                [
                    "",
                    "Controller diagnostics:",
                    "",
                    "| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for method, action in action_rows.items():
                diagnostic = action["residual_to_parameter_width"]
                is_model_based = method == _STRUCTURED_MODEL_BASED_METHOD
                uncertainty_text = (
                    "n/a"
                    if is_model_based
                    else f"{1000.0 * float(action['mean_selected_uncertainty_sec']):.3f}"
                )
                saturation_text = (
                    "n/a"
                    if is_model_based
                    else f"{float(action['lower_bound_saturation_rate']):.3f}"
                )
                calibration_text = (
                    "n/a"
                    if is_model_based
                    else (
                        f"{float(diagnostic['fraction_within_1x_parameter_width']):.3f}/"
                        f"{float(diagnostic['fraction_within_2x_parameter_width']):.3f}/"
                        f"{float(diagnostic['fraction_within_4x_parameter_width']):.3f}"
                    )
                )
                lines.append(
                    "| {method} | {decisions} | {weight:.3f} | {explored:.3f} | "
                    "{uncertainty} | {saturation} | {calibration} |".format(
                        method=method,
                        decisions=int(action["decisions"]),
                        weight=float(action["mean_weight"]),
                        explored=float(action["explored_rate"]),
                        uncertainty=uncertainty_text,
                        saturation=saturation_text,
                        calibration=calibration_text,
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
