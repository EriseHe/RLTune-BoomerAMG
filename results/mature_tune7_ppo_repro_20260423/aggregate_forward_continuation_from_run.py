from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

def _repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()
OUT_DIR = REPO / "results/mature_tune7_ppo_repro_20260423"


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def env_seeds(name: str, default: str) -> list[int]:
    raw = env_str(name, default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def load_window_result(run_dir: Path, seed: int, window_name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    result_path = run_dir / f"forward_dual_seed{seed}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing result file for seed {seed}: {result_path}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    return row["windows"][window_name], row["protocol"], str(result_path)


def aggregate_window(run_dir: Path, window_name: str, seeds: list[int]) -> dict[str, Any]:
    out_rows = []
    combined_acc = {
        "default_setup_default_solve": {"total": 0.0, "cases": 0, "failed": 0},
        "bandit_only": {"total": 0.0, "cases": 0, "failed": 0},
        "fixed_w_1.60": {"total": 0.0, "cases": 0, "failed": 0},
        "ppo_best": {"total": 0.0, "cases": 0, "failed": 0},
    }

    for seed in seeds:
        window, protocol, source_path = load_window_result(run_dir, seed, window_name)
        methods = window["methods"]
        rel = window["relative_pct"]
        fallback_cases = int(protocol.get("eval_cases", protocol.get("cases_per_seed", 0)))
        out_row = {"seed": seed, "path": source_path}
        for method_name in ("default_setup_default_solve", "bandit_only", "fixed_w_1.60", "ppo_best"):
            if method_name not in methods:
                continue
            method = methods[method_name]
            cases_i = int(method.get("cases", fallback_cases))
            total_i = float(method.get("total_runtime", float(method["mean_runtime"]) * cases_i))
            out_row[f"{method_name}_mean"] = float(method["mean_runtime"])
            out_row[f"{method_name}_total"] = total_i
            out_row[f"{method_name}_failed"] = int(method["failed_count"])
            combined_acc[method_name]["total"] += total_i
            combined_acc[method_name]["cases"] += cases_i
            combined_acc[method_name]["failed"] += int(method["failed_count"])
        for rel_name, rel_value in rel.items():
            out_row[rel_name] = float(rel_value)
        out_rows.append(out_row)

    combined = {}
    for method_name, acc in combined_acc.items():
        if acc["cases"] <= 0:
            continue
        combined[method_name] = {
            "cases": int(acc["cases"]),
            "mean_runtime": float(acc["total"] / acc["cases"]),
            "total_runtime": float(acc["total"]),
            "failed_count": int(acc["failed"]),
        }

    relative_pct = {}
    if "default_setup_default_solve" in combined and "bandit_only" in combined:
        relative_pct["bandit_vs_default_setup"] = 100.0 * (
            combined["default_setup_default_solve"]["mean_runtime"] - combined["bandit_only"]["mean_runtime"]
        ) / combined["default_setup_default_solve"]["mean_runtime"]
    if "default_setup_default_solve" in combined and "fixed_w_1.60" in combined:
        relative_pct["fixed_w_1.60_vs_default_setup"] = 100.0 * (
            combined["default_setup_default_solve"]["mean_runtime"] - combined["fixed_w_1.60"]["mean_runtime"]
        ) / combined["default_setup_default_solve"]["mean_runtime"]
    if "default_setup_default_solve" in combined and "ppo_best" in combined:
        relative_pct["ppo_best_vs_default_setup"] = 100.0 * (
            combined["default_setup_default_solve"]["mean_runtime"] - combined["ppo_best"]["mean_runtime"]
        ) / combined["default_setup_default_solve"]["mean_runtime"]
    if "bandit_only" in combined and "fixed_w_1.60" in combined:
        relative_pct["fixed_w_1.60_vs_bandit"] = 100.0 * (
            combined["bandit_only"]["mean_runtime"] - combined["fixed_w_1.60"]["mean_runtime"]
        ) / combined["bandit_only"]["mean_runtime"]
    if "bandit_only" in combined and "ppo_best" in combined:
        relative_pct["ppo_best_vs_bandit"] = 100.0 * (
            combined["bandit_only"]["mean_runtime"] - combined["ppo_best"]["mean_runtime"]
        ) / combined["bandit_only"]["mean_runtime"]
    if "fixed_w_1.60" in combined and "ppo_best" in combined:
        relative_pct["ppo_best_vs_fixed_w_1.60"] = 100.0 * (
            combined["fixed_w_1.60"]["mean_runtime"] - combined["ppo_best"]["mean_runtime"]
        ) / combined["fixed_w_1.60"]["mean_runtime"]
    return {
        "cases_per_seed": next(
            int(round(row["bandit_only_total"] / row["bandit_only_mean"]))
            for row in out_rows
            if "bandit_only_total" in row and "bandit_only_mean" in row
        ),
        "rows": out_rows,
        "combined": combined,
        "relative_pct": relative_pct,
    }


def main() -> None:
    run_dir = env_path("RUN_DIR", OUT_DIR)
    seeds = env_seeds("FORWARD_SEEDS", "39393939,39394939,39400939,39406939,39412939")
    eval_a_name = env_str("EVAL_A_NAME", "eval_500")
    eval_b_name = env_str("EVAL_B_NAME", "eval_1000")
    result_path = env_path("SUMMARY_PATH", run_dir / "forward_continuation_summary.json")
    result = {
        "stage": "summary",
        "protocol": "forward continuation from archived model snapshot",
        "run_dir": str(run_dir),
        "seeds": seeds,
        "windows": {
            eval_a_name: aggregate_window(run_dir, eval_a_name, seeds),
            eval_b_name: aggregate_window(run_dir, eval_b_name, seeds),
        },
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "result_write", "path": str(result_path)}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
